"""
summ_train.py -- ONE Phase-6 training job (one Phase-1 model x one technique), run as a
subprocess on ONE GPU by run_experiment.py. Reads RUN/logs/run_config.json.

    python summ_train.py --run-dir RUN [--mode train|overfit]

train: tokenizer + token cache -> layout (head / headtail / dancer) -> model + technique ->
       critical self-checks on the real device -> pre-flight (train tok/s, generation speed) ->
       time plan (epochs / train fraction / dev_gen size) -> training on summary tokens only,
       dev loss 2x per epoch, best.pt by dev loss, early stopping, 30-min resumable last.pt,
       time guard that keeps the generation reserve -> decode dev_gen + test -> dev-tuned length
       cap (ROUGE / BERTScore at 5 caps) -> validated submission.zip -> DONE.json
KD students (technique "kd") load the finished teacher run and distil ONLINE:
  loss = alpha * CE(student, gold) + (1 - alpha) * T^2 * KL(teacher_T || student_T)
  over the summary tokens (teacher logits at exactly the same positions; all Phase-1 models
  share the BPE vocabulary).
Exit codes (as Phases 4/5): 0 ok, 1 error, 2 does not fit, 4 too many NaN steps, 5 watchdog,
6 critical self-check failed, 7 OOM at one sequence per micro-batch.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback
import zipfile
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cjpe_common import DEFAULT_EPOCHS, read_json, write_json, now_str, write_csv
from cjpe_models import assert_one_device, load_trainable_state, model_size
from cjpe_train import (Ctx, JobExit, amp_setup, is_oom, make_optimizer, make_scaler,
                        make_scheduler, peak_gb, save_best, save_last, set_seed)
from gen_ml import (LENGTH_RATIOS, bertscore, cap_words, diagnostics, free_bertscore, mean_of,
                    primary, rouge_all)
from ner_models import torch_load
from summ_data import Layout, load_cache, summary_stats
from summ_model import (EOS_ID, apply_summ_technique, build_summ_model, collate_train,
                        generate_many, rebuild_summ)

PEFT_ZERO_INIT = ("lorma", "lora", "adalora", "adapters", "pfeiffer")


# =============================================================================================
# data
# =============================================================================================
def build_data(ctx, tok, layout: Layout, cache: Dict) -> Dict:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg["seed"]))
    d = {}
    for split in ("train", "dev"):
        ex, owner = [], []
        for i, r in enumerate(cache[split]):
            for e in layout.examples(r):
                ex.append(e)
                owner.append(i)
        if layout.repr == "dancer" and split == "train":
            empty = [k for k, (_, t) in enumerate(ex) if len(t) <= 1]
            full = [k for k, (_, t) in enumerate(ex) if len(t) > 1]
            cap = int(float(cfg.get("empty_frac", 0.25)) * len(full) / 0.75)
            if len(empty) > cap:
                keep = set(full) | set(rng.choice(empty, cap, replace=False).tolist())
                ex = [e for k, e in enumerate(ex) if k in keep]
                owner = [o for k, o in enumerate(owner) if k in keep]
        cap = int(cfg.get("dev_loss_seqs", 600))
        if split == "dev" and len(ex) > cap:               # dev loss on a fixed random subset
            keep = sorted(rng.choice(len(ex), cap, replace=False).tolist())
            ex, owner = [ex[k] for k in keep], [owner[k] for k in keep]
        d[split] = {"ex": ex, "owner": owner}
    n_gen = min(len(cache["dev"]), int(cfg.get("dev_gen", 64)))
    d["dev_gen"] = list(range(n_gen))
    return d


def assert_integrity(cache: Dict, d: Dict, layout: Layout, vocab: int) -> Dict:
    tr = {r["id"] for r in cache["train"]}
    dv = {r["id"] for r in cache["dev"]}
    te = {r["id"] for r in cache["test"]}
    assert not (tr & te), f"{len(tr & te)} train ids are test ids"
    assert not (tr & dv), f"{len(tr & dv)} train ids are dev ids"
    assert d["train"]["ex"], "no training examples"
    for s in ("train", "dev"):
        for p, t in d[s]["ex"][:5000]:
            assert len(p) + len(t) <= layout.ctx, f"{s}: sequence {len(p) + len(t)} > {layout.ctx}"
            assert int(p.max()) < vocab and (len(t) == 0 or int(t.max()) < vocab), "token id"
    for r in cache["test"][:200]:
        for p in layout.prompts(r):
            assert len(p) < layout.ctx, "test prompt too long"
    return {"train_x_dev": 0, "train_x_test": 0, "n_train_seqs": len(d["train"]["ex"]),
            "n_dev_seqs": len(d["dev"]["ex"])}


def micro_groups(ex, idx: List[int], micro_tokens: int) -> List[List[int]]:
    """Split one effective batch into micro-batches of <= micro_tokens padded tokens."""
    idx = sorted(idx, key=lambda i: len(ex[i][0]) + len(ex[i][1]))
    out, cur, mx = [], [], 0
    for i in idx:
        L = len(ex[i][0]) + len(ex[i][1])
        if cur and max(mx, L) * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


# =============================================================================================
# loss / step
# =============================================================================================
class Stepper:
    def __init__(self, ctx, model, opt, sched, scaler, tech, amp_ctx, cfg, teacher=None):
        self.ctx, self.model, self.opt, self.sched, self.scaler = ctx, model, opt, sched, scaler
        self.tech, self.amp_ctx, self.cfg, self.teacher = tech, amp_ctx, cfg, teacher
        self.micro = int(cfg.get("micro_tokens", 8192))
        self.skipped = self.overflow = 0
        self.params = [p for p in model.parameters() if p.requires_grad]

    def batch_loss(self, b):
        """Sum of per-token losses (fp32) and the number of target tokens."""
        logits, tgt = self.model.lm_logits_at_targets(b["input_ids"], b["attention_mask"],
                                                      b["labels"])
        ce = F.cross_entropy(logits, tgt, reduction="sum")
        if self.teacher is None:
            return ce
        with torch.no_grad():
            tl, _ = self.teacher.lm_logits_at_targets(b["input_ids"], b["attention_mask"],
                                                      b["labels"])
        T = float(self.cfg.get("kd_T", 2.0))
        a = float(self.cfg.get("kd_alpha", 0.5))
        kl = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(tl.float() / T, -1),
                      reduction="sum", log_target=True)
        return a * ce + (1 - a) * T * T * kl

    def step(self, ex, idx, gstep, total):
        while True:
            try:
                return self._step(ex, idx, gstep, total)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                one = max(len(ex[i][0]) + len(ex[i][1]) for i in idx)
                if self.micro <= one:
                    raise JobExit(7, "CUDA OOM with one sequence per micro-batch")
                self.micro = max(one, self.micro // 2)
                self.ctx.log(f"   [oom] micro-batch -> {self.micro} tokens")

    def _step(self, ex, idx, gstep, total):
        self.model.train()
        n_t = sum(len(ex[i][1]) for i in idx)
        tot, toks = 0.0, 0
        for g in micro_groups(ex, idx, self.micro):
            b = collate_train([ex[i] for i in g], self.ctx.device)
            with self.amp_ctx():
                loss = self.batch_loss(b) / max(1, n_t)
            extra = self.tech.extra_loss(self.model)
            if extra is not None:
                loss = loss + extra * (b["n_targets"] / max(1, n_t))
            self.scaler.scale(loss).backward()
            tot += float(loss.detach())
            toks += b["n_tokens"]
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.get("grad_clip", 1.0)))
        if not math.isfinite(tot):
            self.skipped += 1
            self.opt.zero_grad(set_to_none=True)
            self.scaler.update()
            return None, toks
        if not torch.isfinite(gn):
            self.overflow += 1
        self.tech.after_backward(self.model, gstep, total)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        if self.sched is not None:
            self.sched.step()
        return tot, toks


@torch.no_grad()
def dev_loss(ctx, model, ex, amp_ctx, micro: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    for g in micro_groups(ex, list(range(len(ex))), micro):
        b = collate_train([ex[i] for i in g], ctx.device)
        with amp_ctx():
            logits, tgt = model.lm_logits_at_targets(b["input_ids"], b["attention_mask"],
                                                     b["labels"])
        tot += float(F.cross_entropy(logits.float(), tgt, reduction="sum"))
        n += int(tgt.numel())
    return tot / max(1, n)


# =============================================================================================
# self-checks
# =============================================================================================
def job_checks(base, model, cfg, d, layout, device, amp_ctx, tech) -> Dict:
    res = {"checks": [], "critical_failed": [], "advisory_failed": []}

    def add(name, kind, fn):
        rec = {"name": name, "kind": kind}
        try:
            v = fn()
            if isinstance(v, dict):
                rec.update(v)
            rec.setdefault("verdict", "pass")
        except Exception as e:
            rec["verdict"] = "fail"
            rec["error"] = f"{type(e).__name__}: {e}"[:1500]
            rec["trace"] = traceback.format_exc()[-1500:]
        if rec["verdict"] != "pass":
            (res["critical_failed"] if kind == "critical" else res["advisory_failed"]).append(name)
        res["checks"].append(rec)

    ex = d["train"]["ex"]
    sel = sorted(range(len(ex)), key=lambda i: len(ex[i][0]) + len(ex[i][1]))[:2]
    b = collate_train([ex[i] for i in sel], device)
    add("all_tensors_on_one_device", "critical",
        lambda: (assert_one_device(model, device), {"device": str(device)})[1])

    def fwd():
        model.train()
        with amp_ctx():
            logits, tgt = model.lm_logits_at_targets(b["input_ids"], b["attention_mask"],
                                                     b["labels"])
            loss = F.cross_entropy(logits.float(), tgt)
        assert logits.shape[0] == tgt.numel() == b["n_targets"], "target positions"
        ex_ = tech.extra_loss(model)
        if ex_ is not None:
            loss = loss + ex_
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        gr = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gr), "no gradient"
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gr), "non-finite grad"
        model.zero_grad(set_to_none=True)
        return {"loss": float(loss)}
    add("forward_backward_finite", "critical", fwd)

    def masked_loss():
        """labels mask: the loss must not depend on document tokens' labels (only targets)."""
        p, t = ex[sel[0]]
        assert len(t) >= 1 and (b["labels"][0] != -100).sum() == len(t)
        return {}
    add("loss_only_on_summary_tokens", "critical", masked_loss)

    def gen():
        p = layout.prompts({"doc": d["_cache_dev0"]})[0]
        out = generate_many(model, [p], device, amp_ctx, 4, 0, int(cfg.get("num_beams", 1)), 4, 1)
        assert len(out) == 1 and len(out[0]) <= 4
        return {"tokens": out[0]}
    add("generation_runs", "critical", gen)

    if base is not None:
        def zero_init():
            model.eval()
            base.eval()
            with torch.no_grad():
                a, _ = model.lm_logits_at_targets(b["input_ids"], b["attention_mask"], b["labels"])
                c, _ = base.lm_logits_at_targets(b["input_ids"], b["attention_mask"], b["labels"])
            diff = float((a - c).abs().max())
            thr = (5e-2 if cfg["technique"] in ("adapters", "pfeiffer") else 1e-3) * max(
                1.0, float(c.abs().max()))
            return {"value": diff, "threshold": thr, "verdict": "pass" if diff <= thr else "warn"}
        add("zero_init_equals_base", "advisory", zero_init)
    model.zero_grad(set_to_none=True)
    res["ok"] = not res["critical_failed"]
    res["selfcheck"] = "fail" if res["critical_failed"] else (
        "warn" if res["advisory_failed"] else "pass")
    return res


# =============================================================================================
# throughput + plan
# =============================================================================================
def measure(ctx, stepper, d, layout, cache, gen_batch: int, n_steps: int = 4):
    model, ex = stepper.model, d["train"]["ex"]
    eff = int(ctx.cfg["effective_batch"])
    sel = np.random.default_rng(1).permutation(len(ex))[:max(1, min(len(ex), eff))].tolist()

    def one():
        toks = 0
        for g in micro_groups(ex, sel, stepper.micro):
            b = collate_train([ex[i] for i in g], ctx.device)
            with stepper.amp_ctx():
                loss = stepper.batch_loss(b) / max(1, b["n_targets"])
            stepper.scaler.scale(loss).backward()
            toks += b["n_tokens"]
        for p in stepper.params:
            p.grad = None
        return toks
    model.train()
    while True:
        try:
            one()
            break
        except RuntimeError as e:
            if not is_oom(e):
                raise
            for p in stepper.params:
                p.grad = None
            torch.cuda.empty_cache()
            big = max(len(ex[i][0]) + len(ex[i][1]) for i in sel)
            if stepper.micro <= big:
                raise JobExit(7, "CUDA OOM with one sequence per micro-batch (pre-flight)")
            stepper.micro = max(big, stepper.micro // 2)
            ctx.log(f"   [oom] pre-flight: micro-batch -> {stepper.micro} tokens")
    sync = (lambda: torch.cuda.synchronize()) if ctx.device.type == "cuda" else (lambda: None)
    sync()
    t = time.time()
    toks = sum(one() for _ in range(n_steps))
    sync()
    r_train = toks / max(1e-6, time.time() - t)
    # generation: seconds for one batch of real prompts x 24 new tokens (min_new forces length)
    prompts = []
    for r in cache["dev"][:max(1, gen_batch)]:
        prompts += layout.prompts(r)
    prompts = sorted(prompts, key=len, reverse=True)[:gen_batch]
    generate_many(model, prompts[:2], ctx.device, stepper.amp_ctx, 4, 4, 1, 4, gen_batch)
    sync()
    t = time.time()
    generate_many(model, prompts, ctx.device, stepper.amp_ctx, 24, 24,
                  int(ctx.cfg.get("num_beams", 1)), 4, gen_batch)
    sync()
    sec_per_step = (time.time() - t) / 24.0
    return r_train, sec_per_step, len(prompts)


def gen_seconds(n_prompts: int, max_new: int, sec_per_step: float, batch: int) -> float:
    return math.ceil(n_prompts / max(1, batch)) * max_new * sec_per_step


def make_plan(ctx, d, layout, cache, r_train, sps, gen_batch, max_new) -> Dict:
    cfg = ctx.cfg
    budget = float(cfg["budget_s"])
    used = time.time() - ctx.t0
    avail = budget - used - max(120.0, 0.04 * budget)
    n_test = sum(len(layout.prompts(r)) for r in cache["test"])
    per_dev = max(1.0, sum(len(layout.prompts(r)) for r in cache["dev"][:50]) /
                  max(1, min(50, len(cache["dev"]))))
    bert_s = 150.0 if cfg.get("bertscore", True) else 0.0
    n_gen = len(d["dev_gen"])
    while True:
        reserve = 1.3 * gen_seconds(n_test + int(n_gen * per_dev), max_new, sps, gen_batch) \
            + bert_s + 60.0
        if reserve <= 0.45 * avail or n_gen <= 16:
            break
        n_gen = max(16, n_gen // 2)
    d["dev_gen"] = d["dev_gen"][:n_gen]
    tok_tr = sum(len(p) + len(t) for p, t in d["train"]["ex"])
    tok_dv = sum(len(p) + len(t) for p, t in d["dev"]["ex"])
    evals = int(cfg.get("evals_per_epoch", 2))
    per_epoch = tok_tr / r_train + evals * tok_dv / (2.5 * r_train)
    e_max = int(cfg.get("epochs_max") or DEFAULT_EPOCHS.get(cfg["technique"], 3))
    e_min = int(cfg.get("min_epochs", 1))
    left = avail - reserve
    e_fit = int(max(0.0, left) // max(1e-6, per_epoch))
    fixed = cfg.get("fixed_plan")
    if fixed:
        ch = {"epochs": int(fixed["epochs"]), "train_fraction": float(fixed.get("frac", 1.0)),
              "reason": "fixed plan"}
    elif e_fit >= e_min:
        ch = {"epochs": min(e_max, e_fit), "train_fraction": 1.0, "reason": "fits"}
    else:
        frac = left / max(1e-6, e_min * per_epoch)
        mf = float(cfg.get("min_train_fraction", 0.1))
        if frac < mf and left < 300:
            raise JobExit(2, f"not even {mf:.0%} of one epoch fits: {left / 60:.1f} min left "
                             f"after the generation reserve of {reserve / 60:.1f} min")
        ch = {"epochs": e_min, "train_fraction": round(max(mf, min(1.0, frac)), 3),
              "reason": "subsample to fit"}
    n_used = int(round(ch["train_fraction"] * len(d["train"]["ex"])))
    proj = used + ch["epochs"] * per_epoch * ch["train_fraction"] + reserve
    return {**ch, "budget_min": budget / 60, "available_min": avail / 60,
            "tokens_per_s_train": r_train, "gen_sec_per_step": sps, "gen_batch": gen_batch,
            "max_new": max_new, "reserve_min": reserve / 60, "dev_gen": n_gen,
            "train_seqs": len(d["train"]["ex"]), "train_seqs_used": n_used,
            "train_docs": len(cache["train"]), "epoch_min": per_epoch / 60, "epochs_fit": e_fit,
            "projected_minutes": proj / 60, "epochs_max": e_max, "test_prompts": n_test,
            "micro_tokens": None, **layout.describe()}


# =============================================================================================
# generation + evaluation + submission
# =============================================================================================
def summarise(ctx, model, tok, layout, recs, amp_ctx, plan) -> List[str]:
    cfg = ctx.cfg
    prompts, owner = [], []
    for i, r in enumerate(recs):
        for p in layout.prompts(r):
            prompts.append(p)
            owner.append(i)
    outs = generate_many(model, prompts, ctx.device, amp_ctx, int(plan["max_new"]),
                         int(plan.get("min_new", 0)),
                         int(cfg.get("num_beams", 1)), int(cfg.get("no_repeat_ngram", 4)),
                         int(plan["gen_batch"]), ctx.log)
    pieces = [[] for _ in recs]
    for o, i in zip(outs, owner):
        pieces[i].append(tok.decode(o, skip_special_tokens=True))
    texts = [layout.join(p) for p in pieces]
    empty = [i for i, t in enumerate(texts) if len(t.split()) < 3]
    if empty:                                            # never submit an empty summary
        first = [layout.prompts(recs[i])[0] for i in empty]
        mn = min(32, int(plan["max_new"]))
        re_ = generate_many(model, first, ctx.device, amp_ctx, max(mn, int(plan["max_new"])), mn,
                            1, int(cfg.get("no_repeat_ngram", 4)), int(plan["gen_batch"]))
        for i, o in zip(empty, re_):
            texts[i] = tok.decode(o, skip_special_tokens=True).strip() or "."
        ctx.log(f"   [gen] {len(empty)} empty outputs regenerated with min_new={mn}")
    return texts, len(empty)


def evaluate_dev(ctx, preds, recs, median_words) -> Dict:
    refs = [r["summary"] for r in recs]
    curve, all_c, all_r = [], [], []
    for ratio in LENGTH_RATIOS:
        cap = None if ratio is None else max(10, int(round(ratio * median_words)))
        c = [cap_words(p, cap) for p in preds]
        curve.append({"ratio": "none" if ratio is None else ratio, "cap_words": cap, "cands": c})
        all_c += c
        all_r += refs
    bs, info = (None, "disabled")
    if ctx.cfg.get("bertscore", True):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        bs, info = bertscore(all_c, all_r, dev, log=ctx.log)
        free_bertscore()
    n = len(preds)
    rows = []
    for k, c in enumerate(curve):
        rg = rouge_all(c["cands"], refs)
        rec = {"ratio": c["ratio"], "cap_words": c["cap_words"],
               "mean_words": float(np.mean([len(x.split()) for x in c["cands"]])) if n else 0,
               "rouge1": rg["rouge1"], "rouge2": rg["rouge2"], "rougeL": rg["rougeL"],
               "bertscore_f1": float(np.mean(bs[k * n:(k + 1) * n])) if bs else None,
               "_r1": rg.get("rouge1_each"), "_rl": rg.get("rougeL_each"),
               "_bs": bs[k * n:(k + 1) * n] if bs else None}
        rec["score"], rec["metric"] = primary(rec)
        rows.append(rec)
    best = max(rows, key=lambda r: -1 if r["score"] is None else r["score"])
    return {"curve": rows, "best": best, "bertscore_info": info,
            "cands": curve[rows.index(best)]["cands"]}


def write_outputs(ctx, cache, dev_idx, ev, dev_preds_raw, test_texts, n_empty) -> Dict:
    run = ctx.run_dir
    os.makedirs(os.path.join(run, "dev"), exist_ok=True)
    os.makedirs(os.path.join(run, "submission"), exist_ok=True)
    recs = [cache["dev"][i] for i in dev_idx]
    best = ev["best"]
    diag = []
    with open(os.path.join(run, "dev", "dev_predictions.jsonl"), "w") as f:
        for k, (r, c) in enumerate(zip(recs, ev["cands"])):
            dg = diagnostics(c, r.get("text"), r["summary"])
            row = {"id": r["id"], "n_doc_tokens": r["n_tokens"], "rouge1": best["_r1"][k],
                   "rougeL": best["_rl"][k],
                   "bertscore_f1": best["_bs"][k] if best["_bs"] else None, **dg}
            diag.append(row)
            f.write(json.dumps({**row, "prediction": c, "prediction_uncapped":
                                dev_preds_raw[k], "reference": r["summary"]}) + "\n")
    write_csv(os.path.join(run, "dev", "length_curve.csv"),
              [{k: v for k, v in c.items() if not k.startswith("_")} for c in ev["curve"]],
              ["ratio", "cap_words", "mean_words", "rouge1", "rouge2", "rougeL", "bertscore_f1",
               "score", "metric"])
    buckets = []
    for name, lo, hi in (("<2k", 0, 2000), ("2k-5k", 2000, 5000), ("5k-10k", 5000, 10000),
                         (">10k", 10000, 10 ** 12)):
        g = [x for x in diag if lo <= x["n_doc_tokens"] < hi]
        if g:
            buckets.append({"doc_tokens": name, "n": len(g), "rouge1": mean_of(g, "rouge1"),
                            "rougeL": mean_of(g, "rougeL"), "bertscore_f1": mean_of(g, "bertscore_f1"),
                            "ref_content_recall": mean_of(g, "ref_content_recall"),
                            "words": mean_of(g, "words")})
    write_csv(os.path.join(run, "dev", "length_buckets.csv"), buckets,
              ["doc_tokens", "n", "rouge1", "rougeL", "bertscore_f1", "ref_content_recall", "words"])
    keys = ("words", "ref_words", "compression", "repeated_trigram_rate", "novel_bigram_rate",
            "content_support", "number_precision", "citation_precision", "ref_content_recall")
    rep = {"scores": {k: best[k] for k in ("rouge1", "rouge2", "rougeL", "bertscore_f1", "score",
                                           "metric", "ratio", "cap_words")},
           "diagnostics": {k: mean_of(diag, k) for k in keys}, "length_buckets": buckets,
           "bertscore_info": ev["bertscore_info"], "n_dev_gen": len(recs),
           "empty_regenerated_dev": n_empty}
    write_json(os.path.join(run, "dev", "dev_report.json"), rep)
    # ---- submission (list format, PDF 12.3.5), length cap chosen on dev -----------------------
    cap = best["cap_words"]
    test_final = [cap_words(t, cap) for t in test_texts]
    items = [{"id": r["id"], "summary": s} for r, s in zip(cache["test"], test_final)]
    from validate_submission import validate
    with open(os.path.join(run, "submission", "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    zp = os.path.join(run, "submission", "submission.zip")
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(run, "submission", "prediction.json"), "prediction.json")
    os.replace(zp + ".tmp", zp)
    v = validate(zp, [r["id"] for r in cache["test"]])
    rep["validation"] = v
    rep["test_mean_words"] = v.get("mean_words")
    return rep


# =============================================================================================
# the job
# =============================================================================================
def run_train(ctx: Ctx, mode: str) -> int:
    cfg = ctx.cfg
    set_seed(int(cfg["seed"]))
    dev = ctx.device
    amp_dtype, amp_ctx = amp_setup(dev)
    ctx.status("preflight")
    ctx.log(f"start {cfg['experiment']}/{cfg['technique']}/{cfg['model']} on {dev}"
            f"{' (' + torch.cuda.get_device_name(0) + ')' if dev.type == 'cuda' else ''}, "
            f"budget {cfg['budget_s'] / 60:.0f} min, attempt {cfg.get('attempt', 1)}")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inj = os.environ.get("PHASE6_INJECT_FAIL", "")
    if inj and inj.split(":")[0] == cfg["job"] and int(cfg.get("attempt", 1)) == 1:
        raise JobExit(int(inj.split(":")[1]), "injected failure (test)")

    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.from_pretrained(cfg["bpe_dir"])
    cache = load_cache(cfg["cache_path"])
    stats = summary_stats(cache)

    # ---- model + technique ----------------------------------------------------------------
    model, p1cfg = build_summ_model(cfg["phase1_dir"], cfg.get("layer_ids"))
    vocab = int(p1cfg["config"]["vocab_size"])
    base = copy.deepcopy(model) if (cfg["technique"] in PEFT_ZERO_INIT and
                                    not cfg.get("skip_job_checks")) else None
    sample = cache["train"][0]["doc"][:64]
    groups, tech = apply_summ_technique(model, cfg["technique"], cfg, ctx.log, sample)
    model.to(dev)
    assert_one_device(model, dev)
    size = model_size(model)
    write_json(os.path.join(ctx.run_dir, "logs", "model_size.json"), size)

    layout = Layout(tok, cfg, stats, model.n_virtual)
    d = build_data(ctx, tok, layout, cache)
    d["_cache_dev0"] = cache["dev"][0]["doc"]
    integ = assert_integrity(cache, d, layout, vocab)
    write_json(os.path.join(ctx.run_dir, "logs", "integrity.json"), integ)
    max_new = layout.max_new(d["train"]["ex"])
    min_new = min(max_new, layout.min_new(d["train"]["ex"], float(cfg.get("min_new_frac", 0.3))))
    ctx.log(f"   data: train {len(cache['train'])} docs -> {len(d['train']['ex'])} sequences, dev "
            f"{len(cache['dev'])} ({len(d['dev']['ex'])} seqs), test {len(cache['test'])}; "
            f"layout {layout.describe()}; max_new {max_new}; summary tokens median "
            f"{stats['summary_tokens'].get('median')} p90 {stats['summary_tokens'].get('p90')}")

    checks = {"selfcheck": "skipped"}
    if not cfg.get("skip_job_checks"):
        if base is not None:
            base.to(dev)
        checks = job_checks(base, model, cfg, d, layout, dev, amp_ctx, tech)
    del base
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    write_json(os.path.join(ctx.run_dir, "logs", "job_checks.json"), checks)
    for c in checks.get("checks", []):
        v = c.get("value")
        ctx.log(f"   [check] {c['name']:<28} {c['kind']:<9} {c['verdict']}"
                + (f" value={v:.3g} thr={c.get('threshold'):.3g}" if isinstance(v, float) else "")
                + (f"  {c.get('error', '')[:200]}" if c["verdict"] == "fail" else ""))
    if checks.get("critical_failed"):
        raise JobExit(6, f"critical self-check failed: {checks['critical_failed']}")

    teacher = None
    if cfg["technique"] == "kd":
        tdir = cfg["teacher_dir"]
        trc = read_json(os.path.join(tdir, "logs", "run_config.json"))
        teacher = rebuild_summ(trc, os.path.join(tdir, "checkpoints", "best.pt"), None, dev)
        for p in teacher.parameters():
            p.requires_grad = False
        ctx.log(f"   [kd] teacher {trc['model']} ({trc['technique']}) loaded from {tdir}; "
                f"alpha {cfg.get('kd_alpha')} T {cfg.get('kd_T')}")

    opt = make_optimizer(groups)
    scaler = make_scaler(amp_dtype == torch.float16)
    stepper = Stepper(ctx, model, opt, None, scaler, tech, amp_ctx, cfg, teacher)
    if mode == "overfit":
        return run_overfit(ctx, stepper, d)

    # ---- pre-flight + plan ----------------------------------------------------------------
    plan_path = os.path.join(ctx.run_dir, "logs", "plan.json")
    last_pt = os.path.join(ctx.run_dir, "checkpoints", "last.pt")
    resume = None
    if os.path.exists(last_pt) and os.path.exists(plan_path) and cfg.get("resume", True):
        try:
            resume = torch_load(last_pt)
            plan = read_json(plan_path)
            d["dev_gen"] = d["dev_gen"][:int(plan["dev_gen"])]
            ctx.log(f"   [resume] from step {resume['st']['step']} (plan reused)")
        except Exception as e:
            ctx.log(f"   [resume] unreadable last.pt ({e}); starting fresh")
            resume = None
    if resume is None:
        gb = int(cfg.get("gen_batch", 16))
        r_train, sps, _n = measure(ctx, stepper, d, layout, cache, gb)
        ctx.log(f"   [preflight] train {r_train:,.0f} tok/s, generation {sps * 1000:.0f} ms/step "
                f"(batch {gb}), micro-batch {stepper.micro} tokens, peak {peak_gb():.2f} GB")
        plan = make_plan(ctx, d, layout, cache, r_train, sps, gb, max_new)
        plan["micro_tokens"] = stepper.micro
        plan["min_new"] = min_new
        plan["selfcheck"] = checks.get("selfcheck")
        write_json(plan_path, plan)
    stepper.micro = int(plan.get("micro_tokens") or stepper.micro)
    ctx.log(f"   [plan] CHOSEN: epochs={plan['epochs']} train_fraction={plan['train_fraction']} "
            f"({plan['train_seqs_used']}/{plan['train_seqs']} seqs), dev_gen {plan['dev_gen']}, "
            f"max_new {plan['max_new']}, generation reserve {plan['reserve_min']:.0f} min, "
            f"projected {plan['projected_minutes']:.0f} of {plan['budget_min']:.0f} min "
            f"({plan['reason']})")

    ex = d["train"]["ex"]
    rng0 = np.random.default_rng(int(cfg["seed"]))
    keep = sorted(rng0.permutation(len(ex))[:max(1, plan["train_seqs_used"])].tolist())
    eff = int(cfg["effective_batch"])
    spe = max(1, math.ceil(len(keep) / eff))
    total = spe * int(plan["epochs"])
    sched = make_scheduler(opt, total, float(cfg.get("warmup_frac", 0.06)))
    stepper.sched = sched
    evals = int(cfg.get("evals_per_epoch", 2))
    eval_at = sorted({max(1, round(spe * (k + 1) / evals)) for k in range(evals)})
    patience = int(cfg.get("patience", 2)) * evals
    st = {"step": 0, "epoch": 0, "pos": 0, "best": float("inf"), "best_epoch": None, "bad": 0,
          "evals": [], "skipped": 0, "train_seconds": 0.0}
    if resume is not None:
        load_trainable_state(model, resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
        st.update(resume["st"])
        try:
            random.setstate(resume["rng"]["py"])
            np.random.set_state(resume["rng"]["np"])
            torch.set_rng_state(resume["rng"]["torch"])
        except Exception:
            pass
        write_json(os.path.join(ctx.run_dir, "logs", "resume.json"), {"resumed_at_step": st["step"]})
        del resume
    deadline = ctx.t0 + float(cfg["budget_s"]) - 60 * float(plan["reserve_min"])
    ctx.status("training", plan=plan["reason"])
    ctx.log(f"   [train] {len(keep)} seqs/epoch, {spe} steps/epoch x {plan['epochs']} = {total} "
            f"steps; training deadline in {(deadline - time.time()) / 60:.0f} min")
    last_ck, t_train, stop = time.time(), time.time(), "completed"
    kill_at = int(cfg.get("kill_at_step", -1))
    ema = None
    while st["epoch"] < int(plan["epochs"]):
        order = [keep[i] for i in np.random.default_rng(int(cfg["seed"]) + 1000 * st["epoch"])
                 .permutation(len(keep))]
        while st["pos"] < spe:
            if time.time() > deadline:
                stop = "time guard"
                break
            batch = order[st["pos"] * eff:(st["pos"] + 1) * eff]
            if not batch:
                st["pos"] = spe
                break
            loss, _t = stepper.step(ex, batch, st["step"], total)
            st["step"] += 1
            st["pos"] += 1
            if loss is None:
                st["skipped"] += 1
                if st["step"] >= 20 and st["skipped"] > 0.05 * st["step"]:
                    raise JobExit(4, f"{st['skipped']} of {st['step']} steps non-finite")
            else:
                ema = loss if ema is None else 0.98 * ema + 0.02 * loss
            if st["step"] % 50 == 0:
                ctx.log(f"   step {st['step']}/{total} ep {st['epoch'] + 1} loss {ema:.4f} lr "
                        f"{sched.get_last_lr()[0]:.2e} peak {peak_gb():.1f} GB skipped "
                        f"{st['skipped']} fp16-overflow {stepper.overflow}")
            if st["step"] == kill_at:
                save_last(ctx, model, opt, sched, scaler, st)
                os._exit(137)
            if st["pos"] in eval_at:
                dl = dev_loss(ctx, model, d["dev"]["ex"], amp_ctx, stepper.micro)
                fe = st["epoch"] + st["pos"] / spe
                rec = {"type": "eval", "step": st["step"], "epoch": round(fe, 3),
                       "train_loss": ema, "dev_loss": dl, "lr": sched.get_last_lr()[0],
                       "minutes": (time.time() - ctx.t0) / 60, "skipped_steps": st["skipped"]}
                ctx.jlog(rec)
                st["evals"].append(rec)
                sel = tech.selectable(st["step"], total)
                ctx.log(f"   [eval] epoch {fe:.2f}: dev loss {dl:.4f} (ppl {math.exp(min(dl, 30)):.1f})"
                        f"{'' if sel else ' (not selectable yet)'}")
                if not math.isfinite(dl):
                    raise JobExit(5, "watchdog: non-finite dev loss")
                if sel and dl < st["best"] - 1e-6:
                    st["best"], st["best_epoch"], st["bad"] = dl, round(fe, 3), 0
                    save_best(ctx, model)
                elif sel:
                    st["bad"] += 1
                write_json(os.path.join(ctx.run_dir, "logs", "progress.json"),
                           {"best_dev_loss": st["best"], "best_epoch": st["best_epoch"],
                            "step": st["step"], "total_steps": total, "epochs_run": round(fe, 3)})
                if sel and st["bad"] >= patience:
                    stop = "early stopping"
                    break
            if time.time() - last_ck > 60 * float(cfg.get("ckpt_every_min", 30)):
                save_last(ctx, model, opt, sched, scaler, st)
                last_ck = time.time()
        if stop != "completed":
            break
        st["epoch"] += 1
        st["pos"] = 0
    st["train_seconds"] = time.time() - t_train
    ctx.log(f"   [train] stopped: {stop} at step {st['step']}/{total}")
    del teacher
    stepper.teacher = None
    best_pt = os.path.join(ctx.run_dir, "checkpoints", "best.pt")
    tech.force_final(model)
    if not os.path.exists(best_pt) or (cfg["technique"] == "adalora" and
                                       not math.isfinite(st["best"])):
        st["best"] = dev_loss(ctx, model, d["dev"]["ex"], amp_ctx, stepper.micro)
        st["best_epoch"] = st["epoch"] + st["pos"] / spe
        save_best(ctx, model)
    load_trainable_state(model, torch_load(best_pt))
    if os.path.exists(last_pt):
        os.remove(last_pt)
    return finish(ctx, model, tok, layout, cache, d, amp_ctx, st, plan, size, checks, stop, tech)


def finish(ctx, model, tok, layout, cache, d, amp_ctx, st, plan, size, checks, stop, tech) -> int:
    cfg = ctx.cfg
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    ctx.status("evaluating")
    t = time.time()
    recs = [cache["dev"][i] for i in d["dev_gen"]]
    dev_raw, n_e1 = summarise(ctx, model, tok, layout, recs, amp_ctx, plan)
    ctx.status("predicting")
    test_texts, n_e2 = summarise(ctx, model, tok, layout, cache["test"], amp_ctx, plan)
    dt = time.time() - t
    model.cpu()
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    ev = evaluate_dev(ctx, dev_raw, recs, layout.median_words)
    rep = write_outputs(ctx, cache, d["dev_gen"], ev, dev_raw, test_texts, n_e1)
    b = ev["best"]
    v = rep["validation"]
    metrics = {
        "dev": rep["scores"], "dev_score": b["score"], "score_metric": b["metric"],
        "dev_bertscore_f1": b["bertscore_f1"], "dev_rouge1": b["rouge1"],
        "dev_rouge2": b["rouge2"], "dev_rougeL": b["rougeL"], "length_ratio": b["ratio"],
        "cap_words": b["cap_words"], "dev_loss": st["best"],
        "dev_ppl": math.exp(min(30.0, st["best"])) if math.isfinite(st["best"]) else None,
        "bertscore_info": ev["bertscore_info"], **rep["diagnostics"],
        "empty_outputs": n_e1 + n_e2, "test_mean_words": rep.get("test_mean_words"),
        "doc_repr": layout.repr, "chunk_tokens": getattr(layout, "C", None),
        "summary_budget": layout.S, "max_new": plan["max_new"], "min_new": plan.get("min_new"),
        "num_beams": int(cfg.get("num_beams", 1)), "dev_gen_docs": len(recs),
        "best_epoch": st["best_epoch"],
        "epochs_run": st["evals"][-1]["epoch"] if st["evals"] else 0, "steps": st["step"],
        "skipped_steps": st["skipped"], "stop_reason": stop,
        "train_minutes": st["train_seconds"] / 60, "total_minutes": (time.time() - ctx.t0) / 60,
        "ms_per_doc_infer": 1000 * dt / max(1, len(recs) + len(cache["test"])),
        "peak_mem_gb": peak_gb(), "train_fraction": plan["train_fraction"],
        "epochs_planned": plan["epochs"], "train_seqs": plan["train_seqs"],
        "train_seqs_used": plan["train_seqs_used"], "tokens_per_s": plan["tokens_per_s_train"],
        "gen_tokens_per_s": plan["gen_batch"] / max(1e-9, plan["gen_sec_per_step"]),
        "projected_minutes": plan["projected_minutes"], "selfcheck": checks.get("selfcheck"),
        "validation_ok": v["ok"], "degenerate": v.get("degenerate"),
        "technique_report": tech.final_report(model), **size,
    }
    write_json(os.path.join(ctx.run_dir, "metrics.json"), metrics)
    if not v["ok"]:
        ctx.status("failed", f"submission validation failed: {v['errors']}")
        return 4
    if v.get("degenerate"):
        ctx.log("   !!! WARNING: every test document got the same summary (degenerate)")
    write_json(os.path.join(ctx.run_dir, "DONE.json"),
               {"finished": now_str(), "dev_score": b["score"], "score_metric": b["metric"],
                "stop_reason": stop, "submission": os.path.join(ctx.run_dir, "submission",
                                                                 "submission.zip")})
    ctx.status("done", f"dev {b['metric']} {b['score']}")
    f = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    ctx.log(f"   DONE dev BERTScore-F1 {f(b['bertscore_f1'])}, ROUGE-1 {f(b['rouge1'])}, ROUGE-2 "
            f"{f(b['rouge2'])}, ROUGE-L {f(b['rougeL'])} (length cap {b['ratio']} x median = "
            f"{b['cap_words']} words; {ev['bertscore_info']}); dev loss {f(st['best'])}; test mean "
            f"{rep.get('test_mean_words') or 0:.0f} words; submission validated")
    return 0


def run_overfit(ctx, stepper, d) -> int:
    """8 training sequences, up to 80 steps: the summary-token loss must fall below 25% of its
    start (or below 0.5)."""
    ex = d["train"]["ex"]
    idx = list(range(min(8, len(ex))))
    for g in stepper.opt.param_groups:
        g["lr"] = g["initial_lr"] = max(g["lr"], float(ctx.cfg.get("overfit_lr", 3e-3)))
    stepper.sched = None
    first = last = None
    n = 0
    for n in range(1, 81):
        l, _ = stepper.step(ex, idx, n, 80)
        if l is not None:
            first = l if first is None else first
            last = l
        if last is not None and (last < 0.5 or last < 0.25 * first):
            break
    ok = last is not None and first is not None and (last < 0.5 or last < 0.25 * first)
    res = {"passed": bool(ok), "first_loss": first, "final_loss": last, "steps": n}
    write_json(os.path.join(ctx.run_dir, "logs", "overfit.json"), res)
    ctx.status("done", f"overfit {res}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", default="train", choices=["train", "overfit"])
    a = ap.parse_args()
    cfg = read_json(os.path.join(a.run_dir, "logs", "run_config.json"))
    ctx = Ctx(a.run_dir, cfg)
    code, reason = 1, "unknown"
    try:
        code = run_train(ctx, a.mode)
        reason = "ok" if code == 0 else f"exit {code}"
    except JobExit as e:
        code, reason = e.code, e.reason
        ctx.log(f"   JOB EXIT {code}: {reason}")
    except Exception as e:
        code = 7 if is_oom(e) else 1
        reason = f"{type(e).__name__}: {e}"
        ctx.log(f"   JOB ERROR: {reason}\n{traceback.format_exc()[-3000:]}")
    finally:
        st = read_json(os.path.join(a.run_dir, "status.json"), {}) or {}
        if code != 0 and st.get("state") not in ("failed",):
            ctx.status("failed", reason, exit_code=code)
        elif code == 0 and st.get("state") not in ("done", "degenerate"):
            ctx.status("done", reason)
    sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
