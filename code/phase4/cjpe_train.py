"""
cjpe_train.py -- ONE training job (one model x one technique), run as a subprocess on ONE GPU
(the orchestrator sets CUDA_VISIBLE_DEVICES). Reads RUN/logs/run_config.json.

    python cjpe_train.py --run-dir RUN [--mode train|smoke|overfit|teacher_reuse]

train  : self-checks on the real device -> throughput pre-flight -> time plan (section 8.4)
         -> training with dev evaluation (2x per epoch), threshold tuning, early stopping,
         30-minute resumable checkpoints, time guard -> best.pt -> final dev report, test
         prediction, validated submission.zip -> DONE.json (or status DEGENERATE)
         (+ teacher logits for train/dev/test when run_config["is_teacher"])
smoke  : self-checks + 20-step throughput only (00 notebook)
overfit: 32 training documents must reach loss < 0.1 within 100 steps (selftest)
teacher_reuse : load a finished E1 run, write its teacher logits + dev report

Exit codes: 0 ok, 1 error, 2 does not fit the time budget, 4 too many NaN steps,
5 early-fail watchdog, 6 critical self-check failed, 7 OOM at one document per micro-batch.
Every state change is written to RUN/status.json (also from `finally`), so a crash always
leaves a diagnosable file (rule R3).
"""
import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cjpe_common import (DEFAULT_EPOCHS, MAX_TRAINABLE_PCT, SPEC, disk_guard, DiskFull,
                         log_to, read_json, write_json, now_str)
from cjpe_eval import binary_scores, tune_tau
from cjpe_kd import kd_loss, save_teacher_logits, teacher_map
from cjpe_models import (assert_one_device, build_doc_model, collate_docs, forward_batch,
                         load_trainable_state, model_size, rebuild_trained, trainable_state_dict)
from cjpe_peft import apply_technique, merge_model
from cjpe_predict import (batches_by_chunks, predict_logits, softmax1, write_dev_outputs,
                          write_submission)
from cjpe_tok import doc_chunks, load_cache
from ner_models import torch_load


class JobExit(Exception):
    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


# =============================================================================================
# small helpers
# =============================================================================================
class Ctx:
    def __init__(self, run_dir: str, cfg: Dict):
        self.run_dir = run_dir
        self.cfg = cfg
        self.log_path = os.path.join(run_dir, "logs", "console.log")
        self.t0 = time.time()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def log(self, msg: str) -> None:
        log_to(self.log_path, msg, prefix=f"[{self.cfg.get('job', '?')}] ")

    def status(self, state: str, reason: str = "", **extra) -> None:
        p = os.path.join(self.run_dir, "status.json")
        st = read_json(p, {}) or {}
        hist = st.get("history", [])
        hist.append({"state": state, "time": now_str(), "reason": reason[:500]})
        st.update({"state": state, "reason": reason[:2000], "time": now_str(),
                   "history": hist[-50:], **extra})
        write_json(p, st)

    def jlog(self, rec: Dict) -> None:
        with open(os.path.join(self.run_dir, "logs", "train_log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


def amp_setup(device):
    """The Phase-1 AMP pattern: fp16 autocast + GradScaler on T4 (no bf16), bf16 on Ampere+."""
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = None

    def amp_ctx():
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)
    return dtype, amp_ctx


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# =============================================================================================
# data
# =============================================================================================
def load_split_docs(cfg: Dict, K: Optional[int] = None):
    """{split: {"ids", "labels", "chunks", "n_tokens"}} for the configured representation."""
    cache = load_cache(cfg["cache_path"])
    K = int(K or cfg["K"])
    out = {}
    for split in ("train", "dev", "test"):
        rows = cache[split]
        out[split] = {"ids": [r["id"] for r in rows],
                      "labels": [r["label"] for r in rows],
                      "chunks": [doc_chunks(r, cfg["doc_repr"], K) for r in rows],
                      "n_tokens": [int(r["n_tokens"]) for r in rows]}
    return out


def assert_data_integrity(d: Dict, cfg: Dict) -> Dict:
    """Section 4.3: training ids are train ids only, disjoint from dev and test."""
    tr, dv, te = set(d["train"]["ids"]), set(d["dev"]["ids"]), set(d["test"]["ids"])
    assert not (tr & te), f"{len(tr & te)} training ids are test ids"
    assert not (tr & dv), f"{len(tr & dv)} training ids are dev ids"
    assert all(l in (0, 1) for l in d["train"]["labels"]), "train labels must be 0/1"
    assert all(l in (0, 1) for l in d["dev"]["labels"]), "dev labels must be 0/1"
    for s in ("train", "dev", "test"):
        for ch in d[s]["chunks"][:2000]:
            assert ch, f"{s}: a document has no chunk"
            assert all(len(c) <= 1024 for c in ch), f"{s}: sequence > 1024 tokens"
            assert all(int(c.max()) < 32000 for c in ch), f"{s}: token id >= 32000"
    return {"train_ids_subset_of_train_json": True, "train_x_dev": 0, "train_x_test": 0}


def stratified_subsample(labels: List[int], frac: float, seed: int) -> List[int]:
    if frac >= 0.999:
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        idx = np.array([i for i, l in enumerate(labels) if l == c])
        if len(idx):
            k = max(1, int(round(frac * len(idx))))
            keep += rng.choice(idx, k, replace=False).tolist()
    return sorted(keep)


def class_weights(labels: List[int], mode: str, device) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    c = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float64)
    c = np.maximum(c, 1)
    w = (c.sum() / (2 * c)) if mode == "inv" else np.sqrt(c.sum() / (2 * c))
    w = w / (w * c / c.sum()).sum()                    # mean weight over the data = 1
    return torch.tensor(w, dtype=torch.float32, device=device)


# =============================================================================================
# self-checks on the REAL device (lesson L1 / L2)
# =============================================================================================
def job_checks(base: Optional[torch.nn.Module], model, tech, cfg: Dict, d: Dict, device,
               amp_ctx) -> Dict:
    """critical: device placement, output shape, finite loss and gradients, data sanity.
    advisory: zero-init equals the base, merged equals unmerged, trainable-% cap."""
    res = {"checks": [], "critical_failed": [], "advisory_failed": []}

    def add(name, kind, fn):
        rec = {"name": name, "kind": kind}
        try:
            val = fn()
            if isinstance(val, dict):
                rec.update(val)
            rec.setdefault("verdict", "pass")
        except Exception as e:
            rec["verdict"] = "fail"
            rec["error"] = f"{type(e).__name__}: {e}"[:1500]
            rec["trace"] = traceback.format_exc()[-1500:]
        if rec["verdict"] != "pass":
            (res["critical_failed"] if kind == "critical" else res["advisory_failed"]).append(name)
        res["checks"].append(rec)

    tr = d["train"]
    idx = [i for i in range(min(len(tr["ids"]), 50))][:2]
    b = collate_docs([tr["chunks"][i] for i in idx], device)
    y = torch.tensor([tr["labels"][i] for i in idx], device=device)

    add("all_tensors_on_one_device", "critical", lambda: (assert_one_device(model, device),
                                                         {"device": str(device)})[1])

    def fwd():
        model.train()
        with amp_ctx():
            logits, w = forward_batch(model, b)
        assert logits.shape == (len(idx), 2), f"logits {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), "non-finite logits"
        loss = F.cross_entropy(logits.float(), y)
        ex = tech.extra_loss(model)
        if ex is not None:
            loss = loss + ex
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        gr = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gr), "no gradient"
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gr), "non-finite grad"
        model.zero_grad(set_to_none=True)
        wz = w[~(torch.arange(w.shape[1], device=w.device)[None] <
                 torch.bincount(b["doc_index"], minlength=w.shape[0])[:, None])]
        assert float(wz.detach().abs().sum()) == 0.0, "masked chunk got attention weight"
        return {"loss": float(loss), "logits_shape": list(logits.shape)}
    add("forward_backward_finite", "critical", fwd)

    def tiny_batch():
        ch = [[c[:64] for c in tr["chunks"][i][:1]] for i in idx]
        return collate_docs(ch, device)

    if base is not None and cfg["technique"] in ("lorma", "lora", "adalora", "adapters"):
        def zero_init():
            tb = tiny_batch()
            model.eval()
            base.eval()
            with torch.no_grad():
                a = model.chunk_vectors(tb["input_ids"], tb["attention_mask"])
                c = base.chunk_vectors(tb["input_ids"], tb["attention_mask"])
            diff = float((a - c).abs().max())
            scale = max(1.0, float(c.abs().max()))
            thr = (5e-2 if cfg["technique"] == "adapters" else 1e-4) * scale
            return {"value": diff, "threshold": thr, "verdict": "pass" if diff <= thr else "warn"}
        add("zero_init_equals_base", "advisory", zero_init)
    if cfg["technique"] in ("lorma", "lora", "adalora"):
        def merge_eq():
            m = copy.deepcopy(model)
            g = torch.Generator().manual_seed(7)
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if any(t in n for t in ("lora_B", "lorma_B", "ada_E")):
                        p.add_((torch.randn(p.shape, generator=g) * 0.02).to(p.device, p.dtype))
            tb = tiny_batch()
            m.eval()
            with torch.no_grad():
                u = m.chunk_vectors(tb["input_ids"], tb["attention_mask"])
                n_merged = merge_model(m)
                v = m.chunk_vectors(tb["input_ids"], tb["attention_mask"])
            diff = float((u - v).abs().max())
            thr = 1e-3 * max(1.0, float(u.abs().max()))
            del m
            return {"value": diff, "threshold": thr, "merged_layers": n_merged,
                    "verdict": "pass" if diff <= thr else "warn"}
        add("merged_equals_unmerged", "advisory", merge_eq)
    s = model_size(model)
    real = s["total_params"] >= SPEC["rules"]["foundation_params"][0] * 0.4
    if cfg["technique"] in MAX_TRAINABLE_PCT and real:
        cap = MAX_TRAINABLE_PCT[cfg["technique"]]
        add("trainable_pct_cap", "advisory",
            lambda: {"value": s["trainable_pct"], "threshold": cap,
                     "verdict": "pass" if s["trainable_pct"] <= cap else "warn"})
    model.zero_grad(set_to_none=True)
    res["ok"] = not res["critical_failed"]
    res["selfcheck"] = "fail" if res["critical_failed"] else (
        "warn" if res["advisory_failed"] else "pass")
    return res


# =============================================================================================
# one optimizer step (with OOM back-off)
# =============================================================================================
class Stepper:
    def __init__(self, ctx, model, opt, sched, scaler, tech, amp_ctx, cw, cfg, tmap=None):
        self.ctx, self.model, self.opt, self.sched, self.scaler = ctx, model, opt, sched, scaler
        self.tech, self.amp_ctx, self.cw, self.cfg, self.tmap = tech, amp_ctx, cw, cfg, tmap
        self.micro = int(cfg["micro_chunks"])
        self.skipped = 0
        self.overflow = 0
        self.params = [p for p in model.parameters() if p.requires_grad]

    def loss_of(self, logits, y, ids):
        if self.tmap is not None:
            t = torch.tensor(np.stack([self.tmap[i] for i in ids]), device=logits.device)
            return kd_loss(logits, t, y, float(self.cfg.get("kd_alpha", 0.5)),
                           float(self.cfg.get("kd_T", 2.0)), self.cw) * len(ids)
        per = F.cross_entropy(logits.float(), y, reduction="none")
        if self.cw is not None:
            per = per * self.cw[y]
        return per.sum()

    def step(self, data, doc_idx: List[int], global_step: int, total_steps: int):
        """Forward/backward over `doc_idx` (one effective batch) + optimizer step.
        Returns (mean loss, tokens) or (None, tokens) for a skipped non-finite step."""
        while True:
            try:
                return self._step(data, doc_idx, global_step, total_steps)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                self.opt.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                one_doc = max(len(data["chunks"][i]) for i in doc_idx)
                if self.micro <= one_doc:
                    raise JobExit(7, "CUDA OOM with one document per micro-batch")
                self.micro = max(one_doc, self.micro // 2)
                self.ctx.log(f"   [oom] micro-batch -> {self.micro} chunks (accumulation grows)")

    def _step(self, data, doc_idx, global_step, total_steps):
        model = self.model
        model.train()
        n = len(doc_idx)
        tot, toks = 0.0, 0
        for g in batches_by_chunks(data["chunks"], self.micro, doc_idx):
            b = collate_docs([data["chunks"][i] for i in g], self.ctx.device)
            y = torch.tensor([data["labels"][i] for i in g], device=self.ctx.device)
            with self.amp_ctx():
                logits, _ = forward_batch(model, b)
            loss = self.loss_of(logits, y, [data["ids"][i] for i in g]) / n
            ex = self.tech.extra_loss(model)
            if ex is not None:
                loss = loss + ex * (len(g) / n)
            self.scaler.scale(loss).backward()
            tot += float(loss.detach())
            toks += b["n_tokens"]
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.get("grad_clip", 1.0)))
        if not math.isfinite(tot):
            # a non-finite LOSS is a real problem (counted; > 5% of steps aborts the job)
            self.skipped += 1
            self.opt.zero_grad(set_to_none=True)
            self.scaler.update()
            return None, toks
        if not torch.isfinite(gn):
            # finite loss but inf/nan gradients = normal fp16 loss-scale overflow: GradScaler
            # skips the optimizer step itself and lowers the scale (not counted as a failure)
            self.overflow += 1
        self.tech.after_backward(model, global_step, total_steps)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        if self.sched is not None:
            self.sched.step()
        return tot, toks


def make_optimizer(groups):
    for g in groups:
        g["initial_lr"] = g["lr"]
    return torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups],
                             betas=(0.9, 0.98), eps=1e-8)


def make_scheduler(opt, total_steps: int, warmup_frac: float = 0.06):
    warm = max(1, int(round(warmup_frac * total_steps)))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, 1.0 - (step - warm) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# =============================================================================================
# throughput + time plan (section 8.4)
# =============================================================================================
def measure_throughput(ctx, stepper, d, n_steps: int = 6):
    """Training tokens/s with real batches (no optimizer change: grads are dropped) and
    inference tokens/s."""
    model = stepper.model
    tr = d["train"]
    eff = int(ctx.cfg["effective_batch"])
    rng = np.random.default_rng(1)
    order = rng.permutation(len(tr["ids"])).tolist()
    sel = order[:max(1, min(len(order), eff))]
    params = stepper.params

    def one():
        toks = 0
        for g in batches_by_chunks(tr["chunks"], stepper.micro, sel):
            b = collate_docs([tr["chunks"][i] for i in g], ctx.device)
            y = torch.tensor([tr["labels"][i] for i in g], device=ctx.device)
            with stepper.amp_ctx():
                logits, _ = forward_batch(model, b)
            stepper.scaler.scale(F.cross_entropy(logits.float(), y)).backward()
            toks += b["n_tokens"]
        for p in params:
            p.grad = None
        return toks
    model.train()
    while True:
        try:
            one()                                  # warm-up (allocations, kernels)
            break
        except RuntimeError as e:
            if not is_oom(e):
                raise
            for p in params:
                p.grad = None
            torch.cuda.empty_cache()
            one_doc = max(len(tr["chunks"][i]) for i in sel)
            if stepper.micro <= one_doc:
                raise JobExit(7, "CUDA OOM with one document per micro-batch (pre-flight)")
            stepper.micro = max(one_doc, stepper.micro // 2)
            ctx.log(f"   [oom] pre-flight: micro-batch -> {stepper.micro} chunks")
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    toks = 0
    for _ in range(n_steps):
        toks += one()
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    r_train = toks / max(1e-6, time.time() - t)
    # inference speed
    dv = d["dev"]
    sub = list(range(min(len(dv["ids"]), 64)))
    predict_logits(model, [dv["chunks"][i] for i in sub[:8]], ctx.device, stepper.amp_ctx,
                   int(ctx.cfg["infer_chunks"]))                     # warm-up
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    ntok = sum(sum(len(c) for c in dv["chunks"][i]) for i in sub)
    predict_logits(model, [dv["chunks"][i] for i in sub], ctx.device, stepper.amp_ctx,
                   int(ctx.cfg["infer_chunks"]))
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    r_inf = ntok / max(1e-6, time.time() - t)
    return r_train, max(r_inf, r_train)


def tokens_of(chunks, idx=None) -> int:
    it = chunks if idx is None else (chunks[i] for i in idx)
    return sum(sum(len(c) for c in ch) for ch in it)


def make_plan(ctx, d_by_K: Dict[int, Dict], r_train: float, r_inf: float) -> Dict:
    cfg = ctx.cfg
    budget = float(cfg["budget_s"])
    used = time.time() - ctx.t0
    margin = max(150.0, 0.05 * budget)
    avail = budget - used - margin
    e_max = int(cfg.get("epochs_max") or DEFAULT_EPOCHS.get(cfg["technique"], 4))
    e_min = int(cfg.get("min_epochs", 2))
    evals = int(cfg.get("evals_per_epoch", 2))
    rows = []
    choice = None
    fixed = cfg.get("fixed_plan")
    for K in sorted(d_by_K, reverse=True):
        d = d_by_K[K]
        ep = tokens_of(d["train"]["chunks"]) / r_train
        dev_inf = tokens_of(d["dev"]["chunks"]) / r_inf
        once = (tokens_of(d["test"]["chunks"]) + tokens_of(d["dev"]["chunks"])) / r_inf
        if cfg.get("is_teacher"):
            once += tokens_of(d["train"]["chunks"]) / r_inf
        per_epoch = ep + evals * dev_inf
        e_fit = int(max(0.0, avail - once) // max(1e-6, per_epoch))
        rows.append({"K": K, "epoch_min": per_epoch / 60, "inference_min": once / 60,
                     "epochs_fit": e_fit})
        if fixed:
            if K == max(d_by_K):
                choice = {"K": K, "epochs": int(fixed["epochs"]),
                          "train_fraction": float(fixed.get("frac", 1.0)), "reason": "fixed plan"}
            continue
        if choice is None and e_fit >= e_min:
            choice = {"K": K, "epochs": min(e_max, e_fit), "train_fraction": 1.0,
                      "reason": "fits" if K == max(d_by_K) else "K reduced to fit"}
    if choice is None:
        K = min(d_by_K)
        r = rows[-1]
        per_epoch = r["epoch_min"] * 60
        frac = (avail - r["inference_min"] * 60) / max(1e-6, e_min * per_epoch)
        min_frac = float(cfg.get("min_train_fraction", 0.3))
        if frac >= min_frac:
            choice = {"K": K, "epochs": e_min, "train_fraction": round(min(1.0, frac), 3),
                      "reason": "stratified subsample to fit"}
        else:
            e = 1 if (avail - r["inference_min"] * 60) >= min_frac * per_epoch else 0
            if e == 0:
                raise JobExit(2, f"not even {min_frac:.0%} of one epoch fits: available "
                                 f"{avail / 60:.1f} min, one epoch {per_epoch / 60:.1f} min")
            choice = {"K": K, "epochs": 1, "train_fraction": min_frac,
                      "reason": "minimum subsample, 1 epoch (time guard will cut if needed)"}
    d = d_by_K[choice["K"]]
    n_tr = len(d["train"]["ids"])
    n_used = len(stratified_subsample(d["train"]["labels"], choice["train_fraction"],
                                      int(cfg["seed"])))
    proj = (choice["epochs"] * (tokens_of(d["train"]["chunks"]) * n_used / max(1, n_tr) / r_train
                                + evals * tokens_of(d["dev"]["chunks"]) / r_inf)
            + (tokens_of(d["test"]["chunks"]) + tokens_of(d["dev"]["chunks"])) / r_inf + used)
    plan = {**choice, "K_requested": int(cfg["K"]), "budget_min": budget / 60,
            "available_min": avail / 60, "tokens_per_s_train": r_train,
            "tokens_per_s_infer": r_inf, "train_docs": n_tr, "train_docs_used": n_used,
            "projected_minutes": proj / 60, "options": rows, "epochs_max": e_max,
            "epochs_min": e_min, "micro_chunks": None}
    return plan


def print_plan(ctx, plan: Dict) -> None:
    ctx.log(f"   [plan] budget {plan['budget_min']:.0f} min, available {plan['available_min']:.0f} "
            f"min, train {plan['tokens_per_s_train']:,.0f} tok/s, infer "
            f"{plan['tokens_per_s_infer']:,.0f} tok/s")
    ctx.log("   [plan]   K  epoch(min)  inference(min)  epochs that fit")
    for r in plan["options"]:
        ctx.log(f"   [plan]  {r['K']:>2}  {r['epoch_min']:>10.1f}  {r['inference_min']:>14.1f}  "
                f"{r['epochs_fit']:>15}")
    flag = "" if plan["reason"] == "fits" else "   <<< " + plan["reason"].upper()
    ctx.log(f"   [plan] CHOSEN: K={plan['K']} epochs={plan['epochs']} train_fraction="
            f"{plan['train_fraction']} ({plan['train_docs_used']}/{plan['train_docs']} docs), "
            f"projected {plan['projected_minutes']:.0f} min{flag}")


# =============================================================================================
# evaluation during training
# =============================================================================================
def dev_eval(ctx, model, d, amp_ctx) -> Dict:
    dv = d["dev"]
    logits = predict_logits(model, dv["chunks"], ctx.device, amp_ctx,
                            int(ctx.cfg["infer_chunks"]))
    p1 = softmax1(logits)
    y = np.asarray(dv["labels"])
    tt = tune_tau(p1, y) if ctx.cfg.get("tau_scan", True) else {"tau": 0.5,
                                                                "f1_tuned": None}
    tau = tt["tau"]
    s = binary_scores(y, (p1 >= tau).astype(int))
    loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item())
    return {"p1": p1, "logits": logits, "tau": tau, "scores": s, "f1": s["macro_f1"],
            "f1_at_0p5": binary_scores(y, (p1 >= 0.5).astype(int))["macro_f1"],
            "dev_loss": loss, "p1_std": float(p1.std())}


# =============================================================================================
# checkpoints
# =============================================================================================
def save_best(ctx, model) -> None:
    p = os.path.join(ctx.run_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    full = ctx.cfg["technique"] in ("full_ft", "kd", "student_hard", "partial_ft")
    torch.save(trainable_state_dict(model, half=full), p + ".tmp")
    os.replace(p + ".tmp", p)


def save_last(ctx, model, opt, sched, scaler, st: Dict) -> None:
    p = os.path.join(ctx.run_dir, "checkpoints", "last.pt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        disk_guard(os.path.dirname(ctx.run_dir), 3.0, log=ctx.log,
                   keep=os.path.join(ctx.run_dir, "checkpoints"))
    except DiskFull as e:
        ctx.log(f"   [disk] {e}: skipping last.pt")
        return
    obj = {"model": trainable_state_dict(model), "opt": opt.state_dict(),
           "sched": sched.state_dict(), "scaler": scaler.state_dict(), "st": st,
           "rng": {"py": random.getstate(), "np": np.random.get_state(),
                   "torch": torch.get_rng_state()}}
    torch.save(obj, os.path.join(ctx.run_dir, "checkpoints", "last.tmp"))
    os.replace(os.path.join(ctx.run_dir, "checkpoints", "last.tmp"), p)


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
    inj = os.environ.get("PHASE4_INJECT_FAIL", "")          # tests only: "<job>:<exit code>"
    if inj and inj.split(":")[0] == cfg["job"] and int(cfg.get("attempt", 1)) == 1:
        raise JobExit(int(inj.split(":")[1]), "injected failure (test)")

    # ---- data (all candidate K values, for the planner) ---------------------------------
    K0 = int(cfg["K"])
    reducible = cfg["doc_repr"] in ("hier", "first_k", "last_k")
    Ks = [K0] + ([k for k in (2, 1) if k < K0] if reducible and mode == "train" else [])
    d_by_K = {K: load_split_docs(cfg, K) for K in Ks}
    d = d_by_K[K0]
    integ = assert_data_integrity(d, cfg)
    write_json(os.path.join(ctx.run_dir, "logs", "integrity.json"), integ)
    ctx.log(f"   data: train {len(d['train']['ids'])}, dev {len(d['dev']['ids'])}, test "
            f"{len(d['test']['ids'])} docs; repr {cfg['doc_repr']} K={K0}; integrity ok")

    # ---- teacher logits (KD students) ---------------------------------------------------
    tmap = None
    if cfg["technique"] == "kd":
        tl = cfg.get("teacher_logits")
        if not tl or not os.path.exists(tl):
            raise JobExit(1, f"teacher logits not found ({tl})")
        tmap = teacher_map(tl, "train")
        miss = [i for i in d["train"]["ids"] if i not in tmap]
        if miss:
            raise JobExit(1, f"{len(miss)} train docs have no teacher logits")

    # ---- model ---------------------------------------------------------------------------
    model, _ = build_doc_model(cfg["phase1_dir"], cfg["attn_mode"], cfg.get("layer_ids"),
                               float(cfg.get("dropout", 0.1)))
    need_base = (not cfg.get("skip_job_checks")) and cfg["technique"] in (
        "lorma", "lora", "adalora", "adapters")
    base = copy.deepcopy(model) if need_base else None
    groups, tech = apply_technique(model, cfg["technique"], cfg, log=ctx.log)
    model.to(dev)
    assert_one_device(model, dev)
    size = model_size(model)
    write_json(os.path.join(ctx.run_dir, "logs", "model_size.json"), size)

    checks = {"skipped": True, "selfcheck": "skipped"}
    if not cfg.get("skip_job_checks"):
        if base is not None:
            base.to(dev)
        checks = job_checks(base, model, tech, cfg, d, dev, amp_ctx)
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
    if checks.get("advisory_failed"):
        ctx.log(f"   WARNING advisory self-checks: {checks['advisory_failed']} (continuing)")

    tr_all = d["train"]
    cw = class_weights(tr_all["labels"], cfg.get("class_weight", "sqrt_inv"), dev)
    opt = make_optimizer(groups)
    scaler = make_scaler(amp_dtype == torch.float16)
    stepper = Stepper(ctx, model, opt, None, scaler, tech, amp_ctx, cw, cfg, tmap)

    if mode == "overfit":
        return run_overfit(ctx, stepper, d)

    # ---- pre-flight throughput + plan ----------------------------------------------------
    plan_path = os.path.join(ctx.run_dir, "logs", "plan.json")
    last_pt = os.path.join(ctx.run_dir, "checkpoints", "last.pt")
    resume = None
    if os.path.exists(last_pt) and os.path.exists(plan_path) and cfg.get("resume", True):
        try:
            resume = torch_load(last_pt)
            plan = read_json(plan_path)
            ctx.log(f"   [resume] from step {resume['st']['step']} (plan reused)")
        except Exception as e:
            ctx.log(f"   [resume] unreadable last.pt ({e}); starting fresh")
            resume = None
    if resume is None:
        r_train, r_inf = measure_throughput(ctx, stepper, d)
        ctx.log(f"   [preflight] train {r_train:,.0f} tok/s, inference {r_inf:,.0f} tok/s, "
                f"micro-batch {stepper.micro} chunks, peak {peak_gb():.2f} GB")
        if mode == "smoke":
            write_json(os.path.join(ctx.run_dir, "logs", "smoke.json"),
                       {"tokens_per_s_train": r_train, "tokens_per_s_infer": r_inf,
                        "micro_chunks": stepper.micro, "peak_mem_gb": peak_gb(),
                        "selfcheck": checks.get("selfcheck"), **size})
            ctx.status("smoke_ok", "smoke ok")
            return 0
        plan = make_plan(ctx, d_by_K, r_train, r_inf)
        plan["micro_chunks"] = stepper.micro
        plan["selfcheck"] = checks.get("selfcheck")
        write_json(plan_path, plan)
    print_plan(ctx, plan)
    stepper.micro = int(plan.get("micro_chunks") or stepper.micro)
    if plan["K"] != K0:
        d = d_by_K[plan["K"]]
    del d_by_K

    keep = stratified_subsample(d["train"]["labels"], float(plan["train_fraction"]),
                                int(cfg["seed"]))
    eff = int(cfg["effective_batch"])
    steps_per_epoch = max(1, math.ceil(len(keep) / eff))
    total_steps = steps_per_epoch * int(plan["epochs"])
    sched = make_scheduler(opt, total_steps, float(cfg.get("warmup_frac", 0.06)))
    stepper.sched = sched
    evals = int(cfg.get("evals_per_epoch", 2))
    eval_at = sorted({max(1, round(steps_per_epoch * (k + 1) / evals)) for k in range(evals)})
    patience = int(cfg.get("patience", 2)) * evals

    st = {"step": 0, "epoch": 0, "pos": 0, "best_f1": -1.0, "best_epoch": None, "bad": 0,
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
        with open(os.path.join(ctx.run_dir, "logs", "resume.json"), "w") as f:
            json.dump({"resumed_at_step": st["step"]}, f)
        del resume

    # inference reserve after the training deadline: final dev + test (+ teacher train logits)
    r_inf = float(plan["tokens_per_s_infer"])
    reserve = (tokens_of(d["dev"]["chunks"]) + tokens_of(d["test"]["chunks"])) / r_inf
    if cfg.get("is_teacher"):
        reserve += tokens_of(d["train"]["chunks"]) / r_inf
    reserve = 1.25 * reserve + 90.0
    deadline = ctx.t0 + float(cfg["budget_s"]) - reserve
    ctx.status("training", plan=plan["reason"])
    ctx.log(f"   [train] {len(keep)} docs/epoch, {steps_per_epoch} steps/epoch x "
            f"{plan['epochs']} epochs = {total_steps} steps; eval at steps {eval_at} of each "
            f"epoch; training deadline in {(deadline - time.time()) / 60:.0f} min")

    last_ckpt = time.time()
    t_train = time.time()
    stop_reason = "completed"
    kill_at = int(cfg.get("kill_at_step", -1))
    ema = None
    while st["epoch"] < int(plan["epochs"]):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 * st["epoch"])
        order = [keep[i] for i in rng.permutation(len(keep))]
        while st["pos"] < steps_per_epoch:
            if time.time() > deadline:
                stop_reason = "time guard"
                break
            s_in = st["pos"]
            batch = order[s_in * eff:(s_in + 1) * eff]
            if not batch:
                st["pos"] = steps_per_epoch
                break
            loss, toks = stepper.step(d["train"], batch, st["step"], total_steps)
            st["step"] += 1
            st["pos"] += 1
            if loss is None:
                st["skipped"] += 1
                if st["step"] >= 20 and st["skipped"] > 0.05 * st["step"]:
                    raise JobExit(4, f"{st['skipped']} of {st['step']} steps non-finite")
            else:
                ema = loss if ema is None else 0.98 * ema + 0.02 * loss
            if st["step"] % 50 == 0:
                ctx.log(f"   step {st['step']}/{total_steps} ep {st['epoch'] + 1} loss "
                        f"{ema:.4f} lr {sched.get_last_lr()[0]:.2e} peak {peak_gb():.1f} GB "
                        f"skipped {st['skipped']} fp16-overflow {stepper.overflow}")
            if st["step"] == kill_at:
                save_last(ctx, model, opt, sched, scaler, st)
                os._exit(137)
            if st["pos"] in eval_at:
                ev = dev_eval(ctx, model, d, amp_ctx)
                frac_ep = st["epoch"] + st["pos"] / steps_per_epoch
                rec = {"type": "eval", "step": st["step"], "epoch": round(frac_ep, 3),
                       "train_loss": ema, "dev_loss": ev["dev_loss"], "dev_macro_f1": ev["f1"],
                       "dev_acc": ev["scores"]["accuracy"], "tau": ev["tau"],
                       "f1_at_0p5": ev["f1_at_0p5"], "lr": sched.get_last_lr()[0],
                       "minutes": (time.time() - ctx.t0) / 60, "skipped_steps": st["skipped"]}
                ctx.jlog(rec)
                st["evals"].append(rec)
                sel = tech.selectable(st["step"], total_steps)
                ctx.log(f"   [eval] epoch {frac_ep:.2f}: dev macro-F1 {ev['f1']:.4f} (tau "
                        f"{ev['tau']:.2f}; @0.5 {ev['f1_at_0p5']:.4f}) loss {ev['dev_loss']:.4f}"
                        f"{'' if sel else ' (not selectable yet)'}")
                if sel and ev["f1"] > st["best_f1"] + 1e-9:
                    st["best_f1"], st["best_epoch"], st["bad"] = ev["f1"], round(frac_ep, 3), 0
                    save_best(ctx, model)
                elif sel:
                    st["bad"] += 1
                write_json(os.path.join(ctx.run_dir, "logs", "progress.json"),
                           {"best_dev_macro_f1": st["best_f1"], "best_epoch": st["best_epoch"],
                            "step": st["step"], "total_steps": total_steps,
                            "epochs_run": round(frac_ep, 3)})
                if st["epoch"] == 0 and st["pos"] == eval_at[-1] and ev["f1"] < 0.40 \
                        and ev["p1_std"] < 1e-3:
                    raise JobExit(5, f"watchdog: dev macro-F1 {ev['f1']:.3f} with a constant "
                                     f"output after epoch 1")
                if sel and st["bad"] >= patience:
                    stop_reason = "early stopping"
                    break
            if time.time() - last_ckpt > 60 * float(cfg.get("ckpt_every_min", 30)):
                save_last(ctx, model, opt, sched, scaler, st)
                last_ckpt = time.time()
        if stop_reason != "completed":
            break
        st["epoch"] += 1
        st["pos"] = 0
    st["train_seconds"] = time.time() - t_train
    ctx.log(f"   [train] stopped: {stop_reason} at step {st['step']}/{total_steps}")

    # ---- final model ---------------------------------------------------------------------
    best_pt = os.path.join(ctx.run_dir, "checkpoints", "best.pt")
    tech.force_final(model)
    if not os.path.exists(best_pt) or (cfg["technique"] == "adalora" and st["best_f1"] < 0):
        ev = dev_eval(ctx, model, d, amp_ctx)
        st["best_f1"], st["best_epoch"] = ev["f1"], st["epoch"] + st["pos"] / steps_per_epoch
        save_best(ctx, model)
        ctx.log(f"   [final] no selectable checkpoint during training -> current weights "
                f"(dev macro-F1 {ev['f1']:.4f})")
    load_trainable_state(model, torch_load(best_pt))
    lp = os.path.join(ctx.run_dir, "checkpoints", "last.pt")
    if os.path.exists(lp):
        os.remove(lp)
    return finish(ctx, model, tech, d, amp_ctx, st, plan, size, checks, stop_reason)


def finish(ctx, model, tech, d, amp_ctx, st, plan, size, checks, stop_reason) -> int:
    cfg = ctx.cfg
    ctx.status("evaluating")
    t = time.time()
    ev = dev_eval(ctx, model, d, amp_ctx)
    snips = load_snippets(cfg)
    rep = write_dev_outputs(ctx.run_dir, d["dev"]["ids"], d["dev"]["labels"], ev["p1"],
                            ev["tau"], d["dev"]["n_tokens"], snips)
    ctx.status("predicting")
    tl = predict_logits(model, d["test"]["chunks"], ctx.device, amp_ctx, int(cfg["infer_chunks"]))
    dt = time.time() - t
    p_test = softmax1(tl)
    np.savez_compressed(os.path.join(ctx.run_dir, "dev", "probs.npz"),
                        dev_ids=np.asarray(d["dev"]["ids"]), dev_p1=ev["p1"],
                        dev_logits=ev["logits"], test_ids=np.asarray(d["test"]["ids"]),
                        test_p1=p_test, test_logits=tl)
    vrep = write_submission(ctx.run_dir, d["test"]["ids"], p_test, ev["tau"])
    if cfg.get("is_teacher"):
        trl = predict_logits(model, d["train"]["chunks"], ctx.device, amp_ctx,
                             int(cfg["infer_chunks"]))
        save_teacher_logits(os.path.join(ctx.run_dir, "teacher_logits.npz"),
                            {"train": {"ids": d["train"]["ids"], "logits": trl},
                             "dev": {"ids": d["dev"]["ids"], "logits": ev["logits"]},
                             "test": {"ids": d["test"]["ids"], "logits": tl}})
        ctx.log("   [teacher] logits cached for train / dev / test")
    n_docs = len(d["dev"]["ids"]) + len(d["test"]["ids"])
    s = rep["scores"]
    metrics = {
        "dev": s, "dev_macro_f1": s["macro_f1"], "dev_macro_f1_at_0p5":
            rep["scores_at_0p5"]["macro_f1"], "tau": ev["tau"], "ece": rep["ece"],
        "best_epoch": st["best_epoch"],
        "epochs_run": st["evals"][-1]["epoch"] if st["evals"] else 0,
        "steps": st["step"], "skipped_steps": st["skipped"], "stop_reason": stop_reason,
        "train_minutes": st["train_seconds"] / 60, "total_minutes": (time.time() - ctx.t0) / 60,
        "docs_per_s_infer": n_docs / max(1e-6, dt), "peak_mem_gb": peak_gb(),
        "test_pred_pos_rate": vrep.get("pos_rate"), "K_effective": plan["K"],
        "train_fraction": plan["train_fraction"], "epochs_planned": plan["epochs"],
        "train_docs_used": plan["train_docs_used"],
        "tokens_per_s": plan["tokens_per_s_train"], "projected_minutes":
            plan["projected_minutes"], "selfcheck": checks.get("selfcheck"),
        "validation_ok": vrep["ok"], "degenerate": vrep.get("degenerate"),
        "technique_report": tech.final_report(model), **size,
    }
    write_json(os.path.join(ctx.run_dir, "metrics.json"), metrics)
    if not vrep["ok"]:
        ctx.status("failed", f"submission validation failed: {vrep['errors']}")
        return 4
    if vrep.get("degenerate"):
        ctx.status("degenerate", "every test prediction is the same class")
        ctx.log("   !!! DEGENERATE: every test prediction is the same class")
        return 0
    write_json(os.path.join(ctx.run_dir, "DONE.json"),
               {"finished": now_str(), "dev_macro_f1": s["macro_f1"], "tau": ev["tau"],
                "stop_reason": stop_reason, "submission": os.path.join(
                    ctx.run_dir, "submission", "submission.zip")})
    ctx.status("done", f"dev macro-F1 {s['macro_f1']:.4f}")
    ctx.log(f"   DONE dev macro-F1 {s['macro_f1']:.4f} (P {s['macro_p']:.4f} R "
            f"{s['macro_r']:.4f}, tau {ev['tau']:.2f}); test positive rate "
            f"{vrep.get('pos_rate'):.3f}; submission validated")
    return 0


def load_snippets(cfg: Dict) -> Dict[str, str]:
    p = cfg.get("snippets_path")
    if p and os.path.exists(p):
        return read_json(p, {}) or {}
    return {}


def run_overfit(ctx, stepper, d) -> int:
    """32 train docs, effective batch 8, up to 100 steps: loss must go below 0.1."""
    tr = d["train"]
    idx = list(range(min(32, len(tr["ids"]))))
    for g in stepper.opt.param_groups:
        g["lr"] = g["initial_lr"] = max(g["lr"], float(ctx.cfg.get("overfit_lr", 3e-3)))
    stepper.sched = None
    last = None
    n = 0
    for n in range(1, 101):
        rng = np.random.default_rng(n)
        order = [idx[i] for i in rng.permutation(len(idx))]
        losses = []
        for k in range(0, len(order), 8):
            l, _ = stepper.step(tr, order[k:k + 8], n, 100)
            if l is not None:
                losses.append(l)
        last = float(np.mean(losses)) if losses else float("nan")
        if last < 0.1:
            break
    res = {"passed": bool(last is not None and last < 0.1), "final_loss": last, "epochs": n,
           "n_docs": len(idx)}
    write_json(os.path.join(ctx.run_dir, "logs", "overfit.json"), res)
    ctx.status("done", f"overfit {res}")
    return 0 if res["passed"] else 1


def run_teacher_reuse(ctx: Ctx) -> int:
    cfg = ctx.cfg
    dev = ctx.device
    _dt, amp_ctx = amp_setup(dev)
    src = cfg["reuse_from"]
    rc = read_json(os.path.join(src, "logs", "run_config.json"))
    model = rebuild_trained(rc, os.path.join(src, "checkpoints", "best.pt"), cfg["phase1_dir"],
                            dev)
    d = load_split_docs(cfg)
    logits = {s: predict_logits(model, d[s]["chunks"], dev, amp_ctx, int(cfg["infer_chunks"]))
              for s in ("train", "dev", "test")}
    save_teacher_logits(os.path.join(ctx.run_dir, "teacher_logits.npz"),
                        {s: {"ids": d[s]["ids"], "logits": logits[s]} for s in logits})
    st = {"best_epoch": None, "epoch": 0, "pos": 0, "step": 0, "skipped": 0, "evals": [],
          "train_seconds": 0.0}
    rm = read_json(os.path.join(src, "metrics.json"), {}) or {}
    plan = {"K": cfg["K"], "train_fraction": rm.get("train_fraction", 1.0),
            "epochs": rm.get("epochs_planned"), "train_docs_used": rm.get("train_docs_used"),
            "tokens_per_s_train": rm.get("tokens_per_s"), "projected_minutes": 0.0}
    ctx.log(f"   [teacher] reused finished E1 run {src}")
    return finish(ctx, model, __import__("cjpe_peft").Technique(), d, amp_ctx, st, plan,
                  model_size(model), {"selfcheck": "reused"}, f"reused {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", default="train", choices=["train", "smoke", "overfit",
                                                        "teacher_reuse"])
    a = ap.parse_args()
    cfg = read_json(os.path.join(a.run_dir, "logs", "run_config.json"))
    ctx = Ctx(a.run_dir, cfg)
    code, reason = 1, "unknown"
    try:
        code = run_teacher_reuse(ctx) if a.mode == "teacher_reuse" else run_train(ctx, a.mode)
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
        elif code == 0 and st.get("state") not in ("done", "degenerate", "smoke_ok"):
            ctx.status("done", reason)
    sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
