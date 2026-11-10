"""
ner_predict.py -- inference, dev evaluation, test prediction, submission, DONE.json.

    python ner_predict.py --run-dir OUT/E2_lorma/M2_llamastyle            # after training
    python ner_predict.py --ensemble RUN1,RUN2,... --out OUT/X5_ensemble  # extra X5

Inference: every document is cut into overlapping windows ([EOS] + up to chunk_len-1 tokens,
stride `infer_stride`), windows are batched, and per-token scores are merged back with weights
that favour the centre of each window (ner_data.merge_window_scores). Linear head: averaged
softmax probabilities, decoded on their logs. CRF head: averaged emissions + learned transitions.
Decoding mode and O-shift are tuned on dev (ner_eval.tune_decoding).
"""
import argparse
import contextlib
import os
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

from ner_common import find_phase1_models, read_json, write_json
from ner_data import IGNORE, SnapRules, load_cache, make_windows, merge_window_scores
from ner_eval import (decode_scores, error_analysis, headline, strict_scores, tune_decoding,
                      write_errors_csv)
from ner_tok import TokenizedDoc


def amp_setup(device):
    """The Phase-1 notebooks' AMP pattern (cell 21): bf16 on Ampere+, fp16 on T4/P100."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    def amp_ctx():
        if amp_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return amp_dtype, amp_ctx


# =============================================================================================
# window inference
# =============================================================================================
@torch.no_grad()
def infer_scores(model, docs, doc_ids, chunk_len, stride, device, amp_ctx, eos_id=0, pad_id=1,
                 batch_size=32):
    """{doc_id: (T, C) float32}. Linear head -> merged probabilities; CRF -> merged emissions."""
    was_training = model.training
    model.eval()
    items = []                                           # (doc_id, a, b)
    for d in doc_ids:
        for (a, b) in make_windows(len(docs[d]["ids"]), chunk_len, stride):
            items.append((d, a, b))
    order = sorted(range(len(items)), key=lambda i: items[i][2] - items[i][1])
    per_window = [None] * len(items)
    use_probs = getattr(model, "crf", None) is None
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(items[i][2] - items[i][1] for i in idx) + 1
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            d, a, b = items[i]
            n = b - a
            x[r, 0] = eos_id
            x[r, 1:n + 1] = torch.from_numpy(docs[d]["ids"][a:b].astype(np.int64))
            m[r, :n + 1] = 1
        x, m = x.to(device), m.to(device)
        with amp_ctx():
            logits, _ = model(x, m)
        logits = logits.float()
        out = torch.softmax(logits, -1) if use_probs else logits
        out = out.cpu().numpy()
        for r, i in enumerate(idx):
            d, a, b = items[i]
            per_window[i] = out[r, 1:b - a + 1]
    model.train(was_training)
    by_doc = {}
    for i, (d, a, b) in enumerate(items):
        by_doc.setdefault(d, ([], []))
        by_doc[d][0].append((a, b))
        by_doc[d][1].append(per_window[i])
    return {d: merge_window_scores(len(docs[d]["ids"]), w, s) for d, (w, s) in by_doc.items()}


def to_decode_scores(scores, is_probs):
    if is_probs:
        return {d: np.log(np.clip(v, 1e-9, 1.0)) for d, v in scores.items()}
    return scores


def decode_all(dec_scores, docs, labels, mode, o_shift, crf=None, snap=None):
    return {d: decode_scores(v, TokenizedDoc.from_dict(docs[d]), labels, mode, o_shift,
                             crf=crf, snap=snap, text=docs[d]["text"]) for d, v in dec_scores.items()}


def quick_dev_f1(model, meta, docs, cfg, device, amp_ctx, snap=None):
    """Used during training: constrained Viterbi, no O-shift. Returns (macro_f1, metrics)."""
    dev_ids = meta["split"]["dev"]
    sc = infer_scores(model, docs, dev_ids, cfg["chunk_len"], cfg["infer_stride"], device, amp_ctx,
                      meta["eos_id"], meta["pad_id"], cfg.get("eval_batch", 32))
    crf = model.crf.export() if getattr(model, "crf", None) is not None else None
    pred = decode_all(to_decode_scores(sc, crf is None), docs, meta["labels"], "viterbi", 0.0,
                      crf, snap)
    gold = {d: docs[d]["gold"] for d in dev_ids}
    m = strict_scores(pred, gold, meta["labels"])
    return headline(m), m


# =============================================================================================
# submission
# =============================================================================================
def write_submission(sub_dir, preds_by_cache_id, docs, test_ids):
    os.makedirs(sub_dir, exist_ok=True)
    pred = {}
    debug = {}
    for d in test_ids:
        oid = docs[d]["orig_id"]
        spans = sorted(preds_by_cache_id.get(d, []))
        pred[oid] = [[int(s), int(e), l] for (s, e, l) in spans]
        debug[oid] = [[int(s), int(e), l, docs[d]["text"][s:e]] for (s, e, l) in spans]
    pj = os.path.join(sub_dir, "prediction.json")
    write_json(pj, pred, indent=None)
    write_json(os.path.join(sub_dir, "entities_debug.json"), debug)
    zp = os.path.join(sub_dir, "submission.zip")
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")               # at the ROOT of the archive
    os.replace(zp + ".tmp", zp)
    return zp


def _model_stats(model, device):
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": n_all, "trainable_params": n_tr,
            "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9
                                if device.type == "cuda" else 0.0)}


def evaluate_and_predict(models, meta, docs, cfg, out_dir, device, log=print):
    """Shared by single runs and ensembles. `models` is a list (len 1 for a normal run)."""
    _, amp_ctx = amp_setup(device)
    labels = meta["labels"]
    snap = SnapRules.from_json(meta.get("snap_rules") or {})
    dev_ids, test_ids = meta["split"]["dev"], meta["split"]["test"]
    is_probs = all(getattr(m, "crf", None) is None for m in models)
    crf = None if is_probs else models[0].crf.export()
    if not is_probs and len(models) > 1:
        raise ValueError("ensembles are supported for linear heads only")

    def scores_for(ids):
        t0 = time.time()
        acc = None
        for m in models:
            sc = infer_scores(m, docs, ids, cfg["chunk_len"], cfg["infer_stride"], device,
                              amp_ctx, meta["eos_id"], meta["pad_id"], cfg.get("eval_batch", 32))
            acc = sc if acc is None else {d: acc[d] + sc[d] for d in sc}
        acc = {d: v / len(models) for d, v in acc.items()}
        n_tok = sum(len(docs[d]["ids"]) for d in ids)
        return to_decode_scores(acc, is_probs), n_tok / max(1e-9, time.time() - t0)

    # ---- dev: tune decoding, full report ----------------------------------------------------
    dev_sc, dev_tps = scores_for(dev_ids)
    gold = {d: docs[d]["gold"] for d in dev_ids}
    best, table = tune_decoding(dev_sc, {d: docs[d] for d in dev_ids}, gold, labels,
                                crf=crf, snap=snap)
    pred_dev = decode_all(dev_sc, docs, labels, best["mode"], best["o_shift"], crf, snap)
    metrics = strict_scores(pred_dev, gold, labels)
    texts = {d: docs[d]["text"] for d in dev_ids}
    analysis, rows = error_analysis(pred_dev, gold, texts, labels)
    # effect of decoding choices (report section)
    argmax_row = next(r for r in table if r["mode"] == "argmax")
    vit0 = next(r for r in table if r["mode"] == "viterbi" and r["o_shift"] == 0.0)
    dev_dir = os.path.join(out_dir, "dev")
    write_errors_csv(os.path.join(dev_dir, "errors.csv"), rows)
    report = {"decoding": best, "decoding_table": table,
              "argmax_vs_viterbi": {"argmax": argmax_row["macro_f1"], "viterbi_shift0": vit0["macro_f1"],
                                    "viterbi_tuned": best["macro_f1"]},
              "metrics": metrics, "analysis": analysis,
              "inference_tokens_per_s": dev_tps, **_model_stats(models[0], device)}
    write_json(os.path.join(dev_dir, "dev_report.json"), report)
    log(f"   dev strict macro-F1 {headline(metrics):.4f} (micro {metrics['micro']['f1']:.4f}) | "
        f"decode {best['mode']} shift {best['o_shift']} | argmax {argmax_row['macro_f1']:.4f}")

    # ---- test: fold3 ----------------------------------------------------------------------------
    test_sc, _ = scores_for(test_ids)
    pred_test = decode_all(test_sc, docs, labels, best["mode"], best["o_shift"], crf, snap)
    zp = write_submission(os.path.join(out_dir, "submission"), pred_test, docs, test_ids)
    from validate_submission import validate
    vr = validate(zp, {docs[d]["orig_id"]: docs[d]["text"] for d in test_ids}, labels)
    write_json(os.path.join(out_dir, "submission", "validation_report.json"), vr)
    if not vr["ok"]:
        raise RuntimeError(f"submission failed validation: {vr['errors'][:5]}")
    log(f"   submission written and validated: {zp} ({vr['n_spans']} spans)")
    return metrics, report


def run_single(run_dir, data_dir=None, device=None, log=print):
    from ner_models import rebuild_trained_model
    cfg = read_json(os.path.join(run_dir, "logs", "run_config.json"))
    data_dir = data_dir or cfg["data_dir"]
    meta, docs = load_cache(data_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_pt = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"{best_pt} missing -- training did not produce a best model")
    p1 = find_phase1_models()
    if cfg.get("phase1_dir"):
        p1.setdefault(cfg["model"], cfg["phase1_dir"])
    model = rebuild_trained_model(cfg, best_pt, p1, device)
    metrics, report = evaluate_and_predict([model], meta, docs, cfg, run_dir, device, log)
    train_sum = read_json(os.path.join(run_dir, "logs", "train_summary.json"), {})
    done = {"exp": cfg["exp"], "model": cfg["model"], "technique": cfg["technique"],
            "arch": cfg["arch"], "dev_macro_f1": headline(metrics),
            "dev_macro_precision": metrics["macro_all"]["precision"],
            "dev_macro_recall": metrics["macro_all"]["recall"],
            "dev_macro_present_f1": metrics["macro_present"]["f1"],
            "dev_micro_f1": metrics["micro"]["f1"], "decoding": report["decoding"],
            "trainable_params": report["trainable_params"], "total_params": report["total_params"],
            "inference_tokens_per_s": report["inference_tokens_per_s"],
            "train": train_sum, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(os.path.join(run_dir, "DONE.json"), done)
    return done


def run_ensemble(run_dirs, out_dir, device=None, log=print):
    from ner_models import rebuild_trained_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # each member is a run folder (logs/run_config.json, checkpoints/best.pt) or a BEST/ folder
    # (run_config.json, best.pt)
    cfgs = [read_json(os.path.join(r, "logs", "run_config.json")) or
            read_json(os.path.join(r, "run_config.json")) for r in run_dirs]
    missing = [r for r, c in zip(run_dirs, cfgs) if c is None]
    if missing:
        raise FileNotFoundError(f"no run_config.json in {missing}")
    data_dirs = {c["data_dir"] for c in cfgs}
    meta, docs = load_cache(sorted(data_dirs)[0])
    p1 = find_phase1_models()
    models = []
    for r, c in zip(run_dirs, cfgs):
        bp = os.path.join(r, "checkpoints", "best.pt")
        if not os.path.exists(bp):
            bp = os.path.join(r, "best.pt")                    # a BEST/ folder
        models.append(rebuild_trained_model(c, bp, p1, device))
    cfg = dict(cfgs[0])
    metrics, report = evaluate_and_predict(models, meta, docs, cfg, out_dir, device, log)
    write_json(os.path.join(out_dir, "DONE.json"),
               {"exp": "X5", "model": "+".join(c["model"] for c in cfgs),
                "technique": "ensemble:" + "+".join(c["technique"] for c in cfgs),
                "members": run_dirs, "dev_macro_f1": headline(metrics),
                "dev_micro_f1": metrics["micro"]["f1"], "decoding": report["decoding"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ensemble", default=None, help="comma-separated run (or BEST) dirs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.ensemble:
        run_ensemble([p for p in a.ensemble.split(",") if p], a.out)
    else:
        done = run_single(a.run_dir, a.data_dir)
        print(f"DONE {done['exp']} {done['model']}: dev macro-F1 {done['dev_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
