"""
rr_predict.py -- window inference, dev report, bias decision, test prediction, submission,
validator, raw logits, DONE.json.

    python rr_predict.py --run-dir OUT/E1_lorma/M2_llamastyle        # after training
    python rr_predict.py --validate PATH/submission.zip              # check any zip

Inference: each document is packed into overlapping windows exactly as in training
(rr_data.build_windows, inference stride); per-sentence logits are taken from the window where
the sentence has the most context ("center") or averaged over windows ("mean"). Exactly one
label per sentence, in order, mapped back through label_vocab with the official spelling.

Submission: ONE prediction.json {doc_id: [label, ...]} covering every CL and IT test id, zipped
as submission.zip with prediction.json at the ROOT. `validate_submission` re-opens the zip and
checks: the only entry is exactly "prediction.json"; valid JSON without NaN/Infinity; a dict of
lists of strings; id set == union of the official test ids; per-document length == number of
test sentences; every label in the vocabulary. It runs after every write; failure aborts.

The validator part is torch-free (used by the data gates); torch is imported lazily.

Inputs : run dir (logs/run_config.json, checkpoints/best.pt), data cache
Outputs: dev/dev_report.json, dev/confusion.csv, logits/{dev,test}_logits.npy (+ index),
         submission/{prediction.json, submission.zip, validation_report.json}, DONE.json
"""
import argparse
import contextlib
import json
import os
import sys
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rr_common import (EXIT, SPEC, default_input_roots, find_phase1_models, read_json,
                       write_json)
from rr_data import build_windows, combine_slot_scores, load_cache
from rr_eval import (bias_decision, full_report, headline, labels_from_logits,
                     write_confusion_csv)


# =============================================================================================
# submission writing + validation (torch-free)
# =============================================================================================
def write_submission(sub_dir: str, pred: Dict[str, List[str]]) -> str:
    os.makedirs(sub_dir, exist_ok=True)
    pj = os.path.join(sub_dir, SPEC["submission_entry"])
    with open(pj + ".tmp", "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, allow_nan=False)
    os.replace(pj + ".tmp", pj)
    zp = os.path.join(sub_dir, SPEC["submission_zip"])
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname=SPEC["submission_entry"])           # at the ROOT of the archive
    os.replace(zp + ".tmp", zp)
    return zp


def _no_nan(x):
    raise ValueError(f"non-finite number {x} in JSON")


def validate_submission(zip_path: str, test_n_sent: Dict[str, int],
                        labels: Sequence[str]) -> Dict:
    """test_n_sent = {official test id: number of sentences} for CL and IT together."""
    errors: List[str] = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if names != [SPEC["submission_entry"]]:
                errors.append(f"zip must contain exactly ['{SPEC['submission_entry']}'] at the "
                              f"root, got {names}")
            if SPEC["submission_entry"] not in names:
                return {"ok": False, "errors": errors}
            raw = z.read(SPEC["submission_entry"]).decode("utf-8")
    except (zipfile.BadZipFile, OSError) as e:
        return {"ok": False, "errors": [f"cannot open zip: {e}"]}
    try:
        pred = json.loads(raw, parse_constant=_no_nan)
    except ValueError as e:
        return {"ok": False, "errors": errors + [f"invalid JSON: {e}"]}
    if not isinstance(pred, dict):
        return {"ok": False, "errors": errors + ["prediction.json must be an object {id: [..]}"]}
    want, got = set(test_n_sent), set(pred)
    if want - got:
        errors.append(f"{len(want - got)} test ids missing, e.g. {sorted(want - got)[:3]}")
    if got - want:
        errors.append(f"{len(got - want)} unknown ids, e.g. {sorted(got - want)[:3]}")
    labset = set(labels)
    n_sent = 0
    for did, labs in pred.items():
        if not isinstance(labs, list):
            errors.append(f"{did}: value must be a list of labels")
            continue
        if did in test_n_sent and len(labs) != test_n_sent[did]:
            errors.append(f"{did}: {len(labs)} labels for {test_n_sent[did]} sentences")
        bad = [l for l in labs if not isinstance(l, str) or l not in labset]
        if bad:
            errors.append(f"{did}: labels outside the vocabulary: {bad[:3]}")
        n_sent += len(labs)
    return {"ok": not errors, "errors": errors[:20], "n_docs": len(pred), "n_labels": n_sent}


def g2_validator_roundtrip(test_n_sent: Dict[str, int], labels: Sequence[str],
                           work_dir: str) -> Dict:
    """Gate G2b: accept a dummy all-'None' prediction, reject 5 corrupted variants."""
    d = os.path.join(work_dir, "g2_validator")
    good = {k: ["None"] * n for k, n in test_n_sent.items()}
    zp = write_submission(os.path.join(d, "good"), good)
    r = validate_submission(zp, test_n_sent, labels)
    if not r["ok"]:
        from rr_common import GateFailure
        raise GateFailure(f"G2: validator rejects the dummy prediction: {r['errors']}")
    k0 = sorted(test_n_sent)[0]
    variants = {
        "missing_id": {k: v for k, v in good.items() if k != k0},
        "extra_id": {**good, "__not_a_test_id__": ["None"]},
        "short_list": {**good, k0: good[k0][:-1]},
        "long_list": {**good, k0: good[k0] + ["None"]},
        "bad_label": {**good, k0: ["NotALabel"] + good[k0][1:]},
    }
    rejected = {}
    for name, p in variants.items():
        zp = write_submission(os.path.join(d, name), p)
        rv = validate_submission(zp, test_n_sent, labels)
        if rv["ok"]:
            from rr_common import GateFailure
            raise GateFailure(f"G2: validator ACCEPTS the corrupted variant '{name}'")
        rejected[name] = rv["errors"][:2]
    # a zip with a folder layer must be rejected too
    nested = os.path.join(d, "nested.zip")
    with zipfile.ZipFile(nested, "w") as z:
        z.writestr("folder/prediction.json", json.dumps(good))
    if validate_submission(nested, test_n_sent, labels)["ok"]:
        from rr_common import GateFailure
        raise GateFailure("G2: validator accepts prediction.json inside a folder")
    rejected["folder_layer"] = ["rejected"]
    return {"accepted_dummy": True, "rejected": rejected}


def official_test_counts(meta: Dict, docs: Dict[str, Dict]) -> Dict[str, int]:
    return {docs[k]["id"]: len(docs[k]["text"]) for k in meta["split"]["test"]}


# =============================================================================================
# inference (torch)
# =============================================================================================
def amp_setup(device):
    """The Phase-1 notebooks' AMP pattern: bf16 on Ampere+, fp16 on T4/P100, none on CPU."""
    import torch
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    def amp_ctx():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return amp_dtype, amp_ctx


def token_batches(windows, max_tokens: int) -> List[List[int]]:
    """Windows sorted by length, grouped so that B * max_len <= max_tokens."""
    order = sorted(range(len(windows)), key=lambda i: len(windows[i]["tokens"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(windows[i]["tokens"])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    return batches


def window_outputs(model, windows, meta: Dict, device, amp_ctx, eval_batch: int = 16,
                   return_vectors: bool = False):
    """[(n_slots, C) float32 CPU tensor per window], [(n_slots, d)] or None."""
    import torch
    from rr_models import collate
    was = model.training
    model.eval()
    out: List = [None] * len(windows)
    vecs: Optional[List] = [None] * len(windows) if return_vectors else None
    with torch.no_grad():
        for idx in token_batches(windows, int(eval_batch) * SPEC["context"]):
            b = collate(windows, idx, meta["pad_id"], device)
            with amp_ctx():
                logits, v = model(b["input_ids"], b["attention_mask"], b["markers"], b["starts"],
                                  b["marker_mask"])
            logits = logits.float().cpu()
            if return_vectors:
                v = v.float().cpu()
            for r, i in enumerate(idx):
                m = len(windows[i]["markers"])
                out[i] = logits[r, :m]
                if return_vectors:
                    vecs[i] = v[r, :m]
    model.train(was)
    return out, vecs


def eval_keys(meta: Dict, docs: Dict, split: str, cfg: Dict) -> List[str]:
    doms = set(cfg.get("domains") or SPEC["domains"])
    return [k for k in meta["split"][split] if docs[k]["domain"] in doms]


def doc_logits(models, docs: Dict, keys: Sequence[str], cfg: Dict, meta: Dict, device,
               amp_ctx) -> Tuple[Dict[str, np.ndarray], float]:
    """{key: (n_sent, C)} averaged over `models` (log-softmax space when > 1), tokens/s."""
    import torch
    wins = build_windows(docs, keys, cfg, meta, "infer")
    n_sent = {k: len(docs[k]["ids"]) for k in keys}
    t0 = time.time()
    acc = None
    for m in models:
        slot, _ = window_outputs(m, wins, meta, device, amp_ctx, cfg.get("eval_batch", 16))
        if len(models) > 1:
            slot = [torch.log_softmax(s, -1) for s in slot]
        per = combine_slot_scores(wins, [s.numpy() for s in slot], n_sent,
                                  cfg.get("window_combine", "center"))
        acc = per if acc is None else {k: acc[k] + per[k] for k in per}
    acc = {k: v / len(models) for k, v in acc.items()}
    n_tok = sum(len(w["tokens"]) for w in wins) * len(models)
    return acc, n_tok / max(1e-9, time.time() - t0)


def quick_dev_f1(model, meta: Dict, docs: Dict, cfg: Dict, device, amp_ctx) -> Tuple[float, Dict]:
    """Pooled dev Macro-F1 (argmax, no bias) -- used for early stopping and the LR probe."""
    keys = eval_keys(meta, docs, "dev", cfg)
    lg, _ = doc_logits([model], docs, keys, cfg, meta, device, amp_ctx)
    rep = full_report({k: docs[k]["labels"] for k in keys}, labels_from_logits(lg),
                      {k: docs[k]["domain"] for k in keys}, meta["labels"])
    return headline(rep), rep


# =============================================================================================
# finalisation from logits (shared by single runs, domain combination and ensembles)
# =============================================================================================
def save_logits(path_dir: str, split: str, logits: Dict[str, np.ndarray]) -> None:
    os.makedirs(path_dir, exist_ok=True)
    keys = sorted(logits)
    arr = np.concatenate([logits[k] for k in keys]).astype(np.float16) if keys else \
        np.zeros((0, len(SPEC["labels"])), np.float16)
    np.save(os.path.join(path_dir, f"{split}_logits.npy"), arr)
    off, idx = 0, {}
    for k in keys:
        idx[k] = [off, off + len(logits[k])]
        off += len(logits[k])
    write_json(os.path.join(path_dir, f"{split}_index.json"), idx)


def load_logits(path_dir: str, split: str) -> Optional[Dict[str, np.ndarray]]:
    p = os.path.join(path_dir, f"{split}_logits.npy")
    idx = read_json(os.path.join(path_dir, f"{split}_index.json"))
    if not os.path.exists(p) or idx is None:
        return None
    arr = np.load(p).astype(np.float32)
    return {k: arr[a:b] for k, (a, b) in idx.items()}


def finalize_from_logits(dev_lg: Dict[str, np.ndarray], test_lg: Dict[str, np.ndarray],
                         meta: Dict, docs: Dict, out_dir: str, log=print,
                         tune_bias: bool = True) -> Dict:
    """Dev report + bias decision + (if all domains are covered) a validated submission."""
    labels = meta["labels"]
    dkeys = sorted(dev_lg)
    gold = {k: docs[k]["labels"] for k in dkeys}
    dom = {k: docs[k]["domain"] for k in dkeys}
    rep = full_report(gold, labels_from_logits(dev_lg), dom, labels)
    dec = bias_decision(dev_lg, gold, dom, len(labels)) if tune_bias else \
        {"apply": False, "bias": [0.0] * len(labels), "rule": "disabled"}
    rep_b = full_report(gold, labels_from_logits(dev_lg, dec["bias"]), dom, labels)
    dev_dir = os.path.join(out_dir, "dev")
    write_confusion_csv(os.path.join(dev_dir, "confusion.csv"), rep["confusion"], labels)
    report = {"metrics": rep, "metrics_with_bias_in_sample": rep_b, "bias": dec}
    write_json(os.path.join(dev_dir, "dev_report.json"), report)
    save_logits(os.path.join(out_dir, "logits"), "dev", dev_lg)
    save_logits(os.path.join(out_dir, "logits"), "test", test_lg)
    log(f"   dev Macro-F1 pooled {headline(rep):.4f} | CL "
        f"{rep['per_domain'].get('CL', {}).get('macro_f1', float('nan')):.4f} | IT "
        f"{rep['per_domain'].get('IT', {}).get('macro_f1', float('nan')):.4f} | bias "
        f"{'APPLIED' if dec['apply'] else 'not applied'} (held-out gain "
        f"{dec.get('heldout_gain_mean', 0.0):+.4f})")
    out = {"report": report, "submission": None, "validation": None}
    all_test = set(meta["split"]["test"])
    if set(test_lg) == all_test:
        pred_idx = labels_from_logits(test_lg, dec["bias"])
        pred = {docs[k]["id"]: [labels[i] for i in pred_idx[k].tolist()] for k in sorted(test_lg)}
        zp = write_submission(os.path.join(out_dir, "submission"), pred)
        vr = validate_submission(zp, official_test_counts(meta, docs), labels)
        write_json(os.path.join(out_dir, "submission", "validation_report.json"), vr)
        if not vr["ok"]:
            raise SubmissionInvalid(f"submission failed validation: {vr['errors'][:5]}")
        hist = {}
        for v in pred.values():
            for l in v:
                hist[l] = hist.get(l, 0) + 1
        write_json(os.path.join(out_dir, "submission", "test_label_histogram.json"), hist)
        log(f"   submission written and validated: {zp} ({vr['n_docs']} docs, "
            f"{vr['n_labels']} labels)")
        out.update(submission=zp, validation=vr)
    else:
        log("   partial-domain run: no submission written (combined later)")
    return out


class SubmissionInvalid(RuntimeError):
    pass


def done_record(cfg: Dict, fin: Dict, extra: Dict) -> Dict:
    rep = fin["report"]["metrics"]
    pooled = rep["pooled"]
    return {"exp": cfg.get("exp"), "job": cfg.get("job"), "model": cfg.get("model"),
            "arch": cfg.get("arch"), "technique": cfg.get("technique"),
            "input_mode": cfg.get("input_mode"), "head": cfg.get("head"),
            "attn_mode": cfg.get("attn_mode"), "loss": cfg.get("loss"),
            "domains": cfg.get("domains"), "variant": cfg.get("variant"),
            "dev_macro_f1": pooled["macro_f1"], "dev_macro_precision": pooled["macro_precision"],
            "dev_macro_recall": pooled["macro_recall"],
            "dev_domain_avg_macro_f1": rep["domain_avg_macro_f1"],
            "dev_per_domain_macro_f1": {d: v["macro_f1"] for d, v in rep["per_domain"].items()},
            "dev_macro_f1_bias_in_sample": fin["report"]["metrics_with_bias_in_sample"]
            ["pooled"]["macro_f1"],
            "bias_applied": fin["report"]["bias"]["apply"],
            "bias_heldout_gain": fin["report"]["bias"].get("heldout_gain_mean"),
            "submission": fin["submission"], "submission_valid": bool(fin["validation"] and
                                                                      fin["validation"]["ok"]),
            **extra, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_single(run_dir: str, data_dir: Optional[str] = None, log=print) -> Dict:
    import torch
    from rr_models import model_size, rebuild_trained_model
    cfg = read_json(os.path.join(run_dir, "logs", "run_config.json"))
    meta, docs = load_cache(data_dir or cfg["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, amp_ctx = amp_setup(device)
    best_pt = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"{best_pt} missing -- training did not produce a best model")
    p1 = find_phase1_models(default_input_roots())
    if cfg.get("phase1_dir"):
        p1.setdefault(cfg["model"], cfg["phase1_dir"])
    model = rebuild_trained_model(cfg, best_pt, p1, device)
    size = model_size(model)                      # as trained (before any merge)
    if cfg["technique"] == "prefix":              # drop the reparameterisation MLP
        from rr_peft import get_technique
        get_technique("prefix").merge(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dkeys, tkeys = eval_keys(meta, docs, "dev", cfg), eval_keys(meta, docs, "test", cfg)
    dev_lg, dev_tps = doc_logits([model], docs, dkeys, cfg, meta, device, amp_ctx)
    test_lg, _ = doc_logits([model], docs, tkeys, cfg, meta, device, amp_ctx)
    infer_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    fin = finalize_from_logits(dev_lg, test_lg, meta, docs, run_dir, log,
                               tune_bias=cfg.get("bias_tuning", True))
    train = read_json(os.path.join(run_dir, "logs", "train_summary.json"), {})
    eff = {"inference_params": model_size(model)["total_params"],"total_params": size["total_params"], "trainable_params": size["trainable_params"],
           "trainable_pct": size["trainable_pct"], "train_peak_mem_gb": train.get("peak_mem_gb"),
           "train_tokens_per_s": train.get("tokens_per_s"),
           "train_minutes": (train.get("train_seconds") or 0) / 60,
           "infer_tokens_per_s": dev_tps, "infer_peak_mem_gb": infer_mem}
    peft_report = dict(train.get("peft_report") or {})
    if cfg["technique"] == "adalora":             # ranks of the submitted (best) checkpoint
        from rr_peft import adalora_ranks
        peft_report.update(adalora_ranks(model))
    extra = {"efficiency": eff, "train": train, "peft_report": peft_report}
    if cfg["technique"] == "kd":
        extra["kd_report"] = kd_comparison(cfg, model, size, dev_tps, infer_mem, meta, docs,
                                           device, amp_ctx, p1, headline(fin["report"]["metrics"]))
    done = done_record(cfg, fin, extra)
    write_json(os.path.join(run_dir, "DONE.json"), done)
    return done


def kd_comparison(cfg, student, s_size, s_tps, s_mem, meta, docs, device, amp_ctx, p1,
                  s_f1) -> Dict:
    """Teacher vs student: dev Macro-F1, size, inference speed and memory (same windows)."""
    import torch
    from rr_kd import load_teacher, resolve_teacher
    from rr_models import model_size
    teacher, tcfg = load_teacher(cfg, device, p1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    keys = eval_keys(meta, docs, "dev", cfg)
    t_lg, t_tps = doc_logits([teacher], docs, keys, tcfg, meta, device, amp_ctx)
    t_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    t_rep = full_report({k: docs[k]["labels"] for k in keys}, labels_from_logits(t_lg),
                        {k: docs[k]["domain"] for k in keys}, meta["labels"])
    t_size = model_size(teacher)
    rc, _ = resolve_teacher(cfg["teacher_dir"], cfg["model"])
    return {"teacher": rc, "teacher_model": tcfg["model"], "teacher_dev_macro_f1": headline(t_rep),
            "student_dev_macro_f1": s_f1, "f1_retained": s_f1 / max(1e-9, headline(t_rep)),
            "teacher_params": t_size["total_params"], "student_params": s_size["total_params"],
            "size_ratio": s_size["total_params"] / t_size["total_params"],
            "teacher_infer_tokens_per_s": t_tps, "student_infer_tokens_per_s": s_tps,
            "speedup": s_tps / max(1e-9, t_tps), "teacher_infer_peak_mem_gb": t_mem,
            "student_infer_peak_mem_gb": s_mem,
            "memory_ratio": (s_mem / t_mem) if t_mem else None,
            "student_layers": cfg.get("layer_ids")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--validate", default=None, help="validate a submission.zip and exit")
    a = ap.parse_args()
    if a.validate:
        from rr_common import DATA_DIR
        meta, docs = load_cache(a.data_dir or DATA_DIR)
        r = validate_submission(a.validate, official_test_counts(meta, docs), meta["labels"])
        print(json.dumps(r, indent=1))
        sys.exit(EXIT["ok"] if r["ok"] else EXIT["validation_failure"])
    try:
        done = run_single(a.run_dir, a.data_dir)
    except SubmissionInvalid as e:
        print("VALIDATION FAILED:", e, file=sys.stderr)
        sys.exit(EXIT["validation_failure"])
    except FileNotFoundError as e:
        print("MISSING:", e, file=sys.stderr)
        sys.exit(EXIT["missing_dependency"])
    print(f"DONE {done['exp']} {done['job']}: dev Macro-F1 {done['dev_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
