"""
pcr_train.py -- ONE Phase-7 training job (one Phase-1 model x one technique), run as a
subprocess on ONE GPU by run_experiment.py. Reads RUN/logs/run_config.json.

    python pcr_train.py --run-dir RUN [--mode train|overfit]

train: chunk cache (+ BM25 channel) -> bi-encoder + technique -> critical self-checks on the
       real device (zero-init == Phase-1, GradCache gradient == exact gradient, gradients reach
       the trainable tensors, padding invariance) -> pre-flight (train s/step, encode tok/s) ->
       time plan (epochs / train fraction / inference chunks) -> contrastive training ->
       full dev + test encoding -> dev-tuned document aggregation (mean / max-chunk) and
       dense x BM25 fusion weight -> validated submission.zip (top-7 per test query) -> DONE.json

Loss (per step: B (query, positive) pairs, H BM25 hard negatives per query, every other
document of the batch as an in-batch negative; other relevant cases of a query are masked):
    L = 0.5 * [CE(q -> docs) + CE(positive -> queries)]     scores = s * cos, s learned
KD students (technique "kd"; teacher = the full-FT run of this notebook, online):
    L = alpha * L_student + (1 - alpha) * T^2 * KL(softmax(S_teacher / T) || softmax(S_student / T))
over exactly the same query x document score matrix.
GradCache (Gao et al. 2021): document vectors are computed without graph, the loss gradient
w.r.t. them is cached, then every micro-batch is re-encoded with graph and back-propagated
-> large contrastive batches at micro-batch memory (training runs without dropout so both
passes are identical; checked).
Exit codes as Phases 4-6: 0 ok, 1 error, 2 does not fit, 4 too many NaN steps, 5 watchdog,
6 critical self-check failed, 7 OOM at one chunk per micro-batch.
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

from cjpe_common import now_str, read_json, write_csv, write_json
from cjpe_models import assert_one_device, load_trainable_state, model_size
from cjpe_train import (Ctx, JobExit, amp_setup, is_oom, make_optimizer, make_scaler,
                        make_scheduler, peak_gb, save_best, save_last, set_seed)
from ner_models import torch_load
from pcr_data import load_cache
from pcr_ml import (K_EVAL, average_precision, eval_scores, fuse, mask_self, micro_prf_at_k,
                    topk, tune_fusion)
from pcr_model import (apply_pcr_technique, build_pcr_model, collate, doc_vectors,
                       encode_chunks, micro_batches, rebuild_pcr)

ZERO_INIT = ("lorma", "lora", "ia3", "full_ft", "partial_ft", "kd", "student_hard")
EPOCHS = {"full_ft": 4, "partial_ft": 4, "kd": 4, "student_hard": 4, "lorma": 5, "lora": 5,
          "qlora": 5, "ia3": 6, "prefix": 6, "prompt": 8}


# =============================================================================================
# data
# =============================================================================================
def pick_chunks(chunks: List[np.ndarray], k: int, rng=None) -> List[np.ndarray]:
    """k chunks: random subset (training) or evenly spaced (inference); order kept."""
    n = len(chunks)
    if n <= k:
        return list(chunks)
    if rng is not None:
        idx = sorted(rng.choice(n, k, replace=False).tolist())
    else:
        idx = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    return [chunks[i] for i in idx]


def build_data(cfg: Dict, cache: Dict) -> Dict:
    tr = cache["splits"]["train"]
    top = cache["bm25"]["train_top"]
    pairs, hard = [], []
    for i, q in enumerate(tr["q"]):
        rel = set(q.get("rel") or [])
        hard.append([int(j) for j in top[i] if int(j) not in rel][:int(cfg.get("hard_pool", 30))])
        for j in sorted(rel):
            pairs.append((i, j))
    dv = cache["splits"]["dev"]
    rng = np.random.default_rng(int(cfg["seed"]))
    qs = [i for i, q in enumerate(dv["q"]) if q.get("rel")]
    qs = sorted(rng.permutation(qs)[:int(cfg.get("proxy_queries", 80))].tolist())
    R = int(cfg.get("proxy_depth", 25))
    bm = cache["bm25"]["dev"]
    cand = {i: sorted(set(topk(bm[i:i + 1], R)[0].tolist()) | set(dv["q"][i]["rel"]))
            for i in qs}
    docs = sorted(set(j for c in cand.values() for j in c))
    return {"pairs": pairs, "hard": hard, "proxy": {"queries": qs, "cand": cand, "docs": docs}}


def assert_integrity(cache: Dict, d: Dict, vocab: int, C: int) -> Dict:
    sp = cache["splits"]
    assert d["pairs"], "no (query, relevant candidate) training pairs"
    for s in ("train", "dev", "test"):
        assert sp[s]["pool"], f"{s}: empty candidate pool"
        for x in (sp[s]["q"][:300] + sp[s]["pool"][:300]):
            for c in x["chunks"]:
                assert 1 <= len(c) <= C and int(c.max()) < vocab, f"{s}: bad chunk"
        n = len(sp[s]["pool"])
        for q in sp[s]["q"]:
            assert all(0 <= j < n for j in (q.get("rel") or [])), f"{s}: relevant index"
    assert cache["bm25"]["dev"].shape == (len(sp["dev"]["q"]), len(sp["dev"]["pool"]))
    assert cache["bm25"]["test"].shape == (len(sp["test"]["q"]), len(sp["test"]["pool"]))
    return {"train_pairs": len(d["pairs"]), "pools_separate": True,
            "queries_without_hard_negatives": int(sum(1 for h in d["hard"] if not h))}


# =============================================================================================
# the step
# =============================================================================================
class Stepper:
    def __init__(self, ctx, model, opt, sched, scaler, tech, amp_ctx, cfg, cache, d,
                 teacher=None):
        self.ctx, self.model, self.opt, self.sched, self.scaler = ctx, model, opt, sched, scaler
        self.tech, self.amp_ctx, self.cfg, self.teacher = tech, amp_ctx, cfg, teacher
        self.tr = cache["splits"]["train"]
        self.d = d
        self.micro = int(cfg.get("micro_tokens", 8192))
        self.k_train = 1 if cfg.get("doc_repr") == "head" else int(cfg.get("k_train", 2))
        self.H = int(cfg.get("hard_negs", 3))
        self.skipped = self.overflow = 0
        self.params = [p for p in model.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- batch construction
    def make_batch(self, pair_idx: List[int], rng) -> Dict:
        pairs = [self.d["pairs"][k] for k in pair_idx]
        qs = list(dict.fromkeys(q for q, _ in pairs))
        cand = list(dict.fromkeys(p for _, p in pairs))
        for q in qs:
            h = self.d["hard"][q]
            if h:
                for j in rng.choice(h, min(self.H, len(h)), replace=False).tolist():
                    if j not in cand:
                        cand.append(int(j))
        chunks, owner = [], []
        pick = (lambda x: x[:1]) if self.k_train == 1 else \
            (lambda x: pick_chunks(x, self.k_train, rng))
        docs = [self.tr["q"][q]["chunks"] for q in qs] + [self.tr["pool"][j]["chunks"] for j in cand]
        for n, ch in enumerate(docs):
            for c in pick(ch):
                chunks.append(c)
                owner.append(n)
        rel = [set(self.tr["q"][q].get("rel") or []) for q in qs]
        qpos = {q: i for i, q in enumerate(qs)}
        cpos = {c: j for j, c in enumerate(cand)}
        pos_q = [qpos[q] for q, _ in pairs]
        pos_c = [cpos[p] for _, p in pairs]
        relm = torch.zeros(len(qs), len(cand), dtype=torch.bool)
        for i in range(len(qs)):
            for j, c in enumerate(cand):
                relm[i, j] = c in rel[i]
        return {"chunks": chunks, "owner": owner, "nq": len(qs), "nc": len(cand),
                "pos_q": pos_q, "pos_c": pos_c, "relm": relm, "n_tokens": sum(map(len, chunks))}

    # ---------------------------------------------------------------- loss
    def loss_from_vectors(self, V: torch.Tensor, b: Dict, VT: Optional[torch.Tensor] = None):
        dev = V.device
        D = doc_vectors(V, torch.as_tensor(b["owner"], device=dev), b["nq"] + b["nc"])
        q, c = D[:b["nq"]], D[b["nq"]:]
        s = self.model.classifier.scale()
        S = s * q @ c.T                                                   # (nq, nc)
        relm = b["relm"].to(dev)
        pq = torch.as_tensor(b["pos_q"], device=dev)
        pc = torch.as_tensor(b["pos_c"], device=dev)
        n = len(b["pos_q"])
        keep = torch.ones(n, S.shape[1], dtype=torch.bool, device=dev)
        keep &= ~relm[pq]                                                 # other positives
        keep[torch.arange(n, device=dev), pc] = True
        L1 = S[pq].masked_fill(~keep, -1e4)
        loss = F.cross_entropy(L1, pc)
        S2 = (s * c[pc] @ q.T)                                            # (pairs, nq)
        keep2 = ~relm[:, pc].T                                            # queries citing it
        keep2[torch.arange(n, device=dev), pq] = True
        L2 = S2.masked_fill(~keep2, -1e4)
        loss = 0.5 * (loss + F.cross_entropy(L2, pq))
        if VT is not None:
            DT = doc_vectors(VT, torch.as_tensor(b["owner"], device=dev), b["nq"] + b["nc"])
            ST = self.teacher.classifier.scale().detach() * DT[:b["nq"]] @ DT[b["nq"]:].T
            T = float(self.cfg.get("kd_T", 2.0))
            a = float(self.cfg.get("kd_alpha", 0.5))
            lt = ST[pq].masked_fill(~keep, -1e4) / T
            kl = F.kl_div(F.log_softmax(L1 / T, -1), F.log_softmax(lt, -1), reduction="batchmean",
                          log_target=True)
            loss = a * loss + (1 - a) * T * T * kl
        return loss

    # ---------------------------------------------------------------- gradcache step
    def step(self, pair_idx, rng, gstep, total):
        while True:
            try:
                return self._step(pair_idx, rng, gstep, total)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                one = int(self.cfg.get("chunk_tokens", 256)) + self.model.n_virtual
                if self.micro <= one:
                    raise JobExit(7, "CUDA OOM with one chunk per micro-batch")
                self.micro = max(one, self.micro // 2)
                self.ctx.log(f"   [oom] micro-batch -> {self.micro} tokens")

    def grads(self, b: Dict) -> float:
        """GradCache forward/backward of one batch; returns the loss."""
        model, dev = self.model, self.ctx.device
        model.eval()                                            # no dropout: passes identical
        V = encode_chunks(model, b["chunks"], dev, self.amp_ctx, self.micro).requires_grad_()
        VT = None
        if self.teacher is not None:
            VT = encode_chunks(self.teacher, b["chunks"], dev, self.amp_ctx, self.micro)
        loss = self.loss_from_vectors(V, b, VT)
        self.scaler.scale(loss).backward()
        G = V.grad
        for g in micro_batches(b["chunks"], self.micro, model.n_virtual):
            ids, mask = collate([b["chunks"][i] for i in g], dev)
            with self.amp_ctx():
                e = model(ids, mask)
            (e.float() * G[torch.as_tensor(g, device=dev)]).sum().backward()
        return float(loss.detach())

    def _step(self, pair_idx, rng, gstep, total):
        b = self.make_batch(pair_idx, rng)
        loss = self.grads(b)
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.get("grad_clip", 1.0)))
        if not math.isfinite(loss):
            self.skipped += 1
            self.opt.zero_grad(set_to_none=True)
            self.scaler.update()
            return None, b["n_tokens"]
        if not torch.isfinite(gn):
            self.overflow += 1
        self.tech.after_backward(self.model, gstep, total)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        if self.sched is not None:
            self.sched.step()
        return loss, b["n_tokens"]


# =============================================================================================
# encoding whole splits + scores
# =============================================================================================
def encode_docs(model, docs: List[Dict], k: int, device, amp_ctx, micro: int):
    """-> (doc vectors (n, d), chunk vectors (N, d), owner (N,)) on CPU float32."""
    chunks, owner = [], []
    for n, x in enumerate(docs):
        for c in (x["chunks"][:1] if k == 1 else pick_chunks(x["chunks"], k)):
            chunks.append(c)
            owner.append(n)
    model.eval()
    V = encode_chunks(model, chunks, device, amp_ctx, micro)
    o = torch.as_tensor(owner, device=V.device)
    D = doc_vectors(V, o, len(docs))
    return D.cpu(), V.cpu(), o.cpu()


def split_scores(model, cache, split, k, device, amp_ctx, micro) -> Dict[str, np.ndarray]:
    sp = cache["splits"][split]
    Q, _, _ = encode_docs(model, sp["q"], k, device, amp_ctx, micro)
    D, V, o = encode_docs(model, sp["pool"], k, device, amp_ctx, micro)
    mean = (Q @ D.T).numpy()
    cs = Q @ V.T                                                          # (nq, N chunks)
    mx = torch.full((Q.shape[0], D.shape[0]), -2.0)
    mx = mx.scatter_reduce(1, o[None].expand(Q.shape[0], -1), cs, reduce="amax")
    qi, pi = [x["id"] for x in sp["q"]], [x["id"] for x in sp["pool"]]
    return {"mean": mask_self(mean, qi, pi), "max": mask_self(mx.numpy(), qi, pi)}


AGGS = {"mean": lambda s: s["mean"], "max": lambda s: s["max"],
        "mean+max": lambda s: 0.5 * (s["mean"] + s["max"])}


@torch.no_grad()
def proxy_eval(ctx, model, cache, d, amp_ctx, micro, k) -> Dict:
    """MAP / F1@7 of re-ranking each sampled dev query's BM25 top-25 + gold (cheap, smooth)."""
    sp = cache["splits"]["dev"]
    pr = d["proxy"]
    Q, _, _ = encode_docs(model, [sp["q"][i] for i in pr["queries"]], k, ctx.device, amp_ctx, micro)
    D, _, _ = encode_docs(model, [sp["pool"][j] for j in pr["docs"]], k, ctx.device, amp_ctx, micro)
    col = {j: n for n, j in enumerate(pr["docs"])}
    aps, ranked, gold = [], [], []
    for r, i in enumerate(pr["queries"]):
        c = pr["cand"][i]
        s = (Q[r:r + 1] @ D[[col[j] for j in c]].T)[0].numpy()
        g = {n for n, j in enumerate(c) if j in set(sp["q"][i]["rel"])}
        aps.append(average_precision(s, g))
        ranked.append(np.argsort(-s, kind="stable")[:K_EVAL].tolist())
        gold.append(g)
    m = micro_prf_at_k(ranked, gold)
    return {"proxy_map": float(np.mean(aps)) if aps else 0.0, "proxy_f1": m["micro_f1"]}


# =============================================================================================
# self-checks
# =============================================================================================
def job_checks(base, model, stepper, cfg, d, device, amp_ctx) -> Dict:
    res = {"checks": [], "critical_failed": [], "advisory_failed": []}
    tr = stepper.tr
    sample = [c for x in tr["q"][:3] + tr["pool"][:3] for c in x["chunks"][:2]]
    nullctx = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731

    def add(name, kind, fn):
        rec = {"name": name, "kind": kind}
        try:
            v = fn()
            rec.update(v if isinstance(v, dict) else {"verdict": "pass" if v else "fail"})
        except Exception as e:
            rec.update({"verdict": "fail", "error": f"{type(e).__name__}: {e}"[:400]})
        res["checks"].append(rec)
        if rec["verdict"] == "fail":
            res["critical_failed" if kind == "critical" else "advisory_failed"].append(name)

    def zero_init():
        a = encode_chunks(base, sample, device, nullctx, 10 ** 6)
        b = encode_chunks(model, sample, device, nullctx, 10 ** 6)
        v = float((a - b).abs().max())
        return {"verdict": "pass" if v < 2e-3 else "fail", "value": v, "threshold": 2e-3}
    if base is not None:
        add("zero_init_equals_phase1", "critical", zero_init)

    def gradcache_exact():
        rng = np.random.default_rng(0)
        b = stepper.make_batch(list(range(min(4, len(d["pairs"])))), rng)
        b["chunks"] = [c[:48] for c in b["chunks"]]
        old_amp, old_scaler = stepper.amp_ctx, stepper.scaler
        stepper.amp_ctx, stepper.scaler = nullctx, make_scaler(False)
        try:
            stepper.opt.zero_grad(set_to_none=True)
            stepper.micro = max(64, 2 * max(len(c) for c in b["chunks"]) + model.n_virtual)
            stepper.grads(b)
            g1 = [None if p.grad is None else p.grad.detach().clone() for p in stepper.params]
            stepper.opt.zero_grad(set_to_none=True)
            V = encode_chunks(model, b["chunks"], device, nullctx, 10 ** 6, grad=True)
            stepper.loss_from_vectors(V, b).backward()
            g2 = [None if p.grad is None else p.grad.detach().clone() for p in stepper.params]
            stepper.opt.zero_grad(set_to_none=True)
        finally:
            stepper.amp_ctx, stepper.scaler = old_amp, old_scaler
            stepper.micro = int(cfg.get("micro_tokens", 8192))
        num = sum(float((a - b_).abs().max()) for a, b_ in zip(g1, g2) if a is not None
                  and b_ is not None)
        den = sum(float(b_.abs().max()) for b_ in g2 if b_ is not None) + 1e-12
        nz = sum(1 for a in g1 if a is not None and float(a.abs().max()) > 0)
        v = num / den
        ok = v < 1e-3 and nz >= 0.5 * len(stepper.params)
        return {"verdict": "pass" if ok else "fail", "value": v, "threshold": 1e-3,
                "nonzero_grad_tensors": nz, "trainable_tensors": len(stepper.params)}
    add("gradcache_equals_exact_gradient_and_grads_flow", "critical", gradcache_exact)

    def padding():
        c = sample[:3]
        one = torch.cat([encode_chunks(model, [x], device, nullctx, 10 ** 6) for x in c])
        many = encode_chunks(model, c, device, nullctx, 10 ** 6)
        v = float((one - many).abs().max())
        return {"verdict": "pass" if v < 2e-3 else "fail", "value": v, "threshold": 2e-3}
    add("padding_invariance", "advisory", padding)
    add("pools_never_crossed", "critical", lambda: d["proxy"]["docs"] and max(d["proxy"]["docs"])
        < len(stepper.ctx_cache["splits"]["dev"]["pool"]))
    res["selfcheck"] = "fail" if res["critical_failed"] else (
        "pass" if not res["advisory_failed"] else "pass (advisory: "
        + ",".join(res["advisory_failed"]) + ")")
    return res


# =============================================================================================
# pre-flight + plan
# =============================================================================================
def measure(ctx, stepper, cache, amp_ctx, k_inf: int, n_steps: int = 3):
    rng = np.random.default_rng(1)
    snap = {n: p.detach().clone() for n, p in stepper.model.named_parameters() if p.requires_grad}
    B = int(stepper.cfg.get("batch_pairs", 16))
    pairs = list(range(len(stepper.d["pairs"])))
    t, n = 0.0, 0
    for k in range(n_steps + 1):
        idx = rng.choice(pairs, min(B, len(pairs)), replace=False).tolist()
        sync()
        t0 = time.time()
        b = stepper.make_batch(idx, rng)
        stepper.grads(b)
        stepper.opt.zero_grad(set_to_none=True)
        sync()
        if k:
            t += time.time() - t0
            n += 1
    with torch.no_grad():
        for nm, p in stepper.model.named_parameters():
            if nm in snap:
                p.copy_(snap[nm])
    sps = t / max(1, n)
    sp = cache["splits"]["dev"]
    docs = (sp["pool"] * 3)[:96]
    chunks = [c for x in docs for c in pick_chunks(x["chunks"], k_inf)][:384]
    sync()
    t0 = time.time()
    encode_chunks(stepper.model, chunks, ctx.device, amp_ctx, stepper.micro)
    sync()
    r_inf = sum(map(len, chunks)) / max(1e-6, time.time() - t0)
    return sps, r_inf


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_tokens(cache, k: int) -> int:
    n = 0
    for s in ("dev", "test"):
        for part in ("q", "pool"):
            for x in cache["splits"][s][part]:
                n += sum(len(c) for c in pick_chunks(x["chunks"], k))
    return n


def make_plan(ctx, cache, d, sps, r_inf, proxy_s, k_inf) -> Dict:
    cfg = ctx.cfg
    B = int(cfg.get("batch_pairs", 16))
    spe = max(1, math.ceil(len(d["pairs"]) / B))
    fixed = cfg.get("fixed_plan") or {}
    want = int(fixed.get("epochs", cfg.get("epochs_max") or EPOCHS.get(cfg["technique"], 4)))
    budget = float(cfg["budget_s"]) - (time.time() - ctx.t0)
    k = k_inf
    while True:
        inf_s = 1.25 * infer_tokens(cache, k) / max(1.0, r_inf) + 120
        n_evals = int(cfg.get("evals_per_epoch", 2)) * max(1, want) + 1
        avail = budget - inf_s - n_evals * proxy_s - 60
        steps = int(avail / max(sps, 1e-6))
        frac = min(1.0, float(fixed.get("frac", 1.0)))
        epochs = want if want == 0 else max(0, min(want, steps // max(1, int(spe * frac))))
        reason = "full plan"
        if want and epochs == 0:
            frac = steps / spe
            epochs = 1
            reason = "less than one epoch fits"
        if frac >= float(cfg.get("min_train_fraction", 0.1)) or want == 0 or k <= 2:
            break
        k = max(2, k // 2)
    if want and frac < float(cfg.get("min_train_fraction", 0.1)):
        raise JobExit(2, f"only {max(0, steps)} steps fit ({sps:.1f}s/step, inference "
                         f"{inf_s / 60:.0f} min) -- below the minimum train fraction")
    used = max(1, int(round(frac * len(d["pairs"])))) if want else 0
    return {"epochs": int(epochs), "train_fraction": round(frac, 4), "train_pairs": len(d["pairs"]),
            "train_pairs_used": used, "steps_per_epoch": max(1, math.ceil(used / B)),
            "sec_per_step": sps, "infer_tokens_per_s": r_inf, "k_infer": k,
            "reserve_min": inf_s / 60, "proxy_s": proxy_s, "budget_min": budget / 60,
            "projected_minutes": (epochs * math.ceil(used / B) * sps + inf_s + n_evals * proxy_s)
            / 60, "reason": reason if k == k_inf else f"{reason}; inference chunks {k_inf}->{k}"}


# =============================================================================================
# outputs
# =============================================================================================
def finish(ctx, model, cache, d, amp_ctx, st, plan, size, checks, stop, tech, micro) -> int:
    cfg = ctx.cfg
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    ctx.status("evaluating")
    k = int(plan["k_infer"]) if cfg.get("doc_repr") != "head" else 1
    t = time.time()
    sd = split_scores(model, cache, "dev", k, ctx.device, amp_ctx, micro)
    ctx.status("predicting")
    stt = split_scores(model, cache, "test", k, ctx.device, amp_ctx, micro)
    dt = time.time() - t
    sp = cache["splits"]
    n_docs = sum(len(sp[s][p]) for s in ("dev", "test") for p in ("q", "pool"))
    dev_q = sp["dev"]["q"]
    gold = [q.get("rel") or [] for q in dev_q]
    bm_dev, bm_test = cache["bm25"]["dev"], cache["bm25"]["test"]
    grid = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
    wmin = float(cfg.get("min_dense_weight", 0.2))
    rows, best = [], None
    for agg, f in AGGS.items():
        _w, _b, curve = tune_fusion(f(sd), bm_dev, gold, grid)
        for r in curve:
            r["doc_agg"] = agg
            rows.append(r)
            if r["dense_weight"] >= wmin - 1e-9 and (best is None or r["micro_f1"] >=
                                                     best["micro_f1"] - 1e-12):   # ties: more dense
                best = r
    dense = {a: next(r for r in rows if r["doc_agg"] == a and r["dense_weight"] == 1.0)
             for a in AGGS}
    bm_only = next(r for r in rows if r["dense_weight"] == 0.0)
    agg, w = best["doc_agg"], best["dense_weight"]
    S_dev = fuse(AGGS[agg](sd), bm_dev, w)
    S_test = fuse(AGGS[agg](stt), bm_test, w)
    out_dev = os.path.join(ctx.run_dir, "dev")
    os.makedirs(out_dev, exist_ok=True)
    write_csv(os.path.join(out_dev, "fusion_curve.csv"), rows,
              ["doc_agg", "dense_weight", "micro_f1", "micro_p", "micro_r", "tp", "n_pred",
               "n_gold"])
    rk = topk(S_dev, K_EVAL)
    rk_dense = topk(zrow_safe(AGGS[agg](sd)), 100)
    pool_ids = [x["id"] for x in sp["dev"]["pool"]]
    with open(os.path.join(out_dev, "dev_predictions.jsonl"), "w") as f:
        for i, q in enumerate(dev_q):
            g = set(gold[i])
            hit = [int(j) in g for j in rk[i]]
            rpos = [n + 1 for n, j in enumerate(rk_dense[i]) if int(j) in g]
            f.write(json.dumps({"id": q["id"], "predicted": [pool_ids[j] for j in rk[i]],
                                "gold": [pool_ids[j] for j in sorted(g)], "hits": int(sum(hit)),
                                "n_gold": len(g), "n_chunks": len(q["chunks"]),
                                "n_chars": q.get("n_chars"),
                                "dense_gold_ranks": rpos[:20],
                                "dense_recall_at_100": len(rpos) / max(1, len(g))}) + "\n")
    np.savez_compressed(os.path.join(out_dev, "scores.npz"),
                        dev=AGGS[agg](sd).astype(np.float16), test=AGGS[agg](stt).astype(np.float16))
    # ---- submission --------------------------------------------------------------------
    sub = os.path.join(ctx.run_dir, "submission")
    os.makedirs(sub, exist_ok=True)
    tq, tp = sp["test"]["q"], [x["id"] for x in sp["test"]["pool"]]
    rt = topk(S_test, K_EVAL)
    pred = {q["id"]: [tp[j] for j in rt[i]] for i, q in enumerate(tq)}
    with open(os.path.join(sub, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(pred, f)
    zp = os.path.join(sub, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(sub, "prediction.json"), "prediction.json")
    from validate_submission import validate
    v = validate(zp, [q["id"] for q in tq], tp)
    rec100 = float(np.mean([json.loads(l)["dense_recall_at_100"] for l in
                            open(os.path.join(out_dev, "dev_predictions.jsonl"))] or [0]))
    rep = {"chosen": {"doc_agg": agg, "dense_weight": w, **best}, "dense_only": dense,
           "bm25_only": bm_only, "bm25_params": cache["bm25"]["params"], "k_infer": k}
    write_json(os.path.join(out_dev, "dev_report.json"), rep)
    model.cpu()
    metrics = {
        "dev_score": best["micro_f1"], "score_metric": "micro_f1@7 (dense x BM25, dev-tuned)",
        "dev_micro_f1": best["micro_f1"], "dev_micro_p": best["micro_p"],
        "dev_micro_r": best["micro_r"], "dense_weight": w, "doc_agg": agg,
        "dev_dense_f1": dense[agg]["micro_f1"], "dev_dense_mean_f1": dense["mean"]["micro_f1"],
        "dev_dense_max_f1": dense["max"]["micro_f1"], "dev_bm25_f1": bm_only["micro_f1"],
        "dev_dense_recall_at_100": rec100, "dev_proxy_map": st.get("best"),
        "dev_proxy_map_init": st.get("init_proxy"), "k_infer": k,
        "k_train": 1 if cfg.get("doc_repr") == "head" else int(cfg.get("k_train", 2)),
        "doc_repr": cfg.get("doc_repr"), "pooling": cfg.get("pooling"),
        "attn_mode": cfg.get("attn_mode"), "chunk_tokens": cache["C"],
        "best_epoch": st["best_epoch"],
        "epochs_run": st["evals"][-1]["epoch"] if st["evals"] else 0, "steps": st["step"],
        "skipped_steps": st["skipped"], "stop_reason": stop,
        "train_minutes": st["train_seconds"] / 60, "total_minutes": (time.time() - ctx.t0) / 60,
        "ms_per_doc_infer": 1000 * dt / max(1, n_docs), "peak_mem_gb": peak_gb(),
        "train_fraction": plan["train_fraction"], "epochs_planned": plan["epochs"],
        "train_pairs": plan["train_pairs"], "train_queries": len(sp["train"]["q"]),
        "tokens_per_s": plan.get("train_tokens_per_s"),
        "infer_tokens_per_s": plan["infer_tokens_per_s"],
        "projected_minutes": plan["projected_minutes"], "selfcheck": checks.get("selfcheck"),
        "validation_ok": v["ok"], "degenerate": v.get("degenerate"),
        "technique_report": tech.final_report(model), **size,
    }
    write_json(os.path.join(ctx.run_dir, "metrics.json"), metrics)
    if not v["ok"]:
        ctx.status("failed", f"submission validation failed: {v['errors']}")
        return 4
    write_json(os.path.join(ctx.run_dir, "DONE.json"),
               {"finished": now_str(), "dev_score": best["micro_f1"], "stop_reason": stop,
                "submission": zp})
    ctx.status("degenerate" if v.get("degenerate") else "done",
               f"dev micro-F1@7 {best['micro_f1']:.4f}")
    ctx.log(f"   DONE dev Micro-F1@7 {best['micro_f1']:.4f} (P {best['micro_p']:.4f} R "
            f"{best['micro_r']:.4f}) with {agg} aggregation, dense weight {w}; dense only "
            f"{dense[agg]['micro_f1']:.4f}, BM25 only {bm_only['micro_f1']:.4f}; dense recall@100 "
            f"{rec100:.3f}; submission validated")
    return 0


def zrow_safe(S):
    from pcr_ml import zrow
    return zrow(S)


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
    inj = os.environ.get("PHASE7_INJECT_FAIL", "")
    if inj and inj.split(":")[0] == cfg["job"] and int(cfg.get("attempt", 1)) == 1:
        raise JobExit(int(inj.split(":")[1]), "injected failure (test)")
    cache = load_cache(cfg["cache_path"])
    d = build_data(cfg, cache)

    model, p1cfg = build_pcr_model(cfg["phase1_dir"], cfg.get("layer_ids"),
                                   cfg.get("attn_mode", "causal"), cfg.get("pooling", "mean"))
    vocab = int(p1cfg["config"]["vocab_size"])
    base = copy.deepcopy(model) if (cfg["technique"] in ZERO_INIT and
                                    not cfg.get("skip_job_checks")) else None
    sample = cache["splits"]["train"]["q"][0]["chunks"][0][:64].tolist()
    groups, tech = apply_pcr_technique(model, cfg["technique"], cfg, ctx.log, sample)
    model.to(dev)
    assert_one_device(model, dev)
    size = model_size(model)
    write_json(os.path.join(ctx.run_dir, "logs", "model_size.json"), size)
    integ = assert_integrity(cache, d, vocab, int(cache["C"]))
    write_json(os.path.join(ctx.run_dir, "logs", "integrity.json"), integ)
    sp = cache["splits"]
    ctx.log(f"   data: train {len(sp['train']['q'])} queries / {len(d['pairs'])} pairs / pool "
            f"{len(sp['train']['pool'])}; dev {len(sp['dev']['q'])} / {len(sp['dev']['pool'])}; "
            f"test {len(sp['test']['q'])} / {len(sp['test']['pool'])}; chunks of {cache['C']} "
            f"tokens (<= {cache['K']} per case); proxy {len(d['proxy']['queries'])} dev queries")

    teacher = None
    if cfg["technique"] == "kd":
        tdir = cfg["teacher_dir"]
        trc = read_json(os.path.join(tdir, "logs", "run_config.json"))
        teacher = rebuild_pcr(trc, os.path.join(tdir, "checkpoints", "best.pt"), dev, sample)
        for p in teacher.parameters():
            p.requires_grad = False
        ctx.log(f"   [kd] teacher {trc['model']} ({trc['technique']}) loaded from {tdir}; "
                f"alpha {cfg.get('kd_alpha')} T {cfg.get('kd_T')}")

    opt = make_optimizer(groups)
    scaler = make_scaler(amp_dtype == torch.float16)
    stepper = Stepper(ctx, model, opt, None, scaler, tech, amp_ctx, cfg, cache, d, teacher)
    stepper.ctx_cache = cache

    checks = {"selfcheck": "skipped"}
    if not cfg.get("skip_job_checks"):
        if base is not None:
            base.to(dev)
        tch, stepper.teacher = stepper.teacher, None
        checks = job_checks(base, model, stepper, cfg, d, dev, amp_ctx)
        stepper.teacher = tch
    del base
    write_json(os.path.join(ctx.run_dir, "logs", "job_checks.json"), checks)
    for c in checks.get("checks", []):
        v = c.get("value")
        ctx.log(f"   [check] {c['name']:<46} {c['kind']:<9} {c['verdict']}"
                + (f" value={v:.3g}" if isinstance(v, float) else "")
                + (f"  {c.get('error', '')[:200]}" if c["verdict"] == "fail" else ""))
    if checks.get("critical_failed"):
        raise JobExit(6, f"critical self-check failed: {checks['critical_failed']}")
    if mode == "overfit":
        return run_overfit(ctx, stepper)

    # ---- pre-flight + plan ------------------------------------------------------------------
    plan_path = os.path.join(ctx.run_dir, "logs", "plan.json")
    last_pt = os.path.join(ctx.run_dir, "checkpoints", "last.pt")
    k_inf = 1 if cfg.get("doc_repr") == "head" else int(cfg.get("k_infer", 8))
    k_proxy = 1 if cfg.get("doc_repr") == "head" else min(k_inf, 4)
    resume = None
    if os.path.exists(last_pt) and os.path.exists(plan_path):
        try:
            resume = torch_load(last_pt)
            plan = read_json(plan_path)
            ctx.log(f"   [resume] from step {resume['st']['step']} (plan reused)")
        except Exception as e:
            ctx.log(f"   [resume] unreadable last.pt ({e}); starting fresh")
            resume = None
    init = None
    if resume is None:
        sps, r_inf = measure(ctx, stepper, cache, amp_ctx, k_inf)
        t0 = time.time()
        init = proxy_eval(ctx, model, cache, d, amp_ctx, stepper.micro, k_proxy)
        proxy_s = time.time() - t0
        ctx.log(f"   [preflight] train {sps:.2f} s/step ({cfg.get('batch_pairs')} pairs + "
                f"{cfg.get('hard_negs')} hard negatives each), encode {r_inf:,.0f} tok/s, proxy "
                f"eval {proxy_s:.0f}s, micro-batch {stepper.micro} tokens, peak {peak_gb():.2f} GB")
        ctx.log(f"   [zero-shot] Phase-1 representation (before training): dev proxy MAP "
                f"{init['proxy_map']:.4f}, F1@7 {init['proxy_f1']:.4f}")
        plan = make_plan(ctx, cache, d, sps, r_inf, proxy_s, k_inf)
        plan["micro_tokens"] = stepper.micro
        plan["selfcheck"] = checks.get("selfcheck")
        plan["init_proxy"] = init["proxy_map"]
        write_json(plan_path, plan)
    stepper.micro = int(plan.get("micro_tokens") or stepper.micro)
    ctx.log(f"   [plan] CHOSEN: epochs={plan['epochs']} train_fraction={plan['train_fraction']} "
            f"({plan['train_pairs_used']}/{plan['train_pairs']} pairs), inference chunks "
            f"{plan['k_infer']}, inference reserve {plan['reserve_min']:.0f} min, projected "
            f"{plan['projected_minutes']:.0f} of {plan['budget_min']:.0f} min ({plan['reason']})")

    B = int(cfg.get("batch_pairs", 16))
    used = int(plan["train_pairs_used"])
    rng0 = np.random.default_rng(int(cfg["seed"]))
    keep = sorted(rng0.permutation(len(d["pairs"]))[:max(1, used)].tolist())
    spe = max(1, math.ceil(len(keep) / B))
    total = spe * int(plan["epochs"])
    sched = make_scheduler(opt, max(1, total), float(cfg.get("warmup_frac", 0.06)))
    stepper.sched = sched
    evals = int(cfg.get("evals_per_epoch", 2))
    eval_at = sorted({max(1, round(spe * (k + 1) / evals)) for k in range(evals)})
    patience = int(cfg.get("patience", 2)) * evals
    st = {"step": 0, "epoch": 0, "pos": 0, "best": -1.0, "best_epoch": None, "bad": 0,
          "evals": [], "skipped": 0, "train_seconds": 0.0, "init_proxy": plan.get("init_proxy")}
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
        del resume
    elif init is not None:
        st["best"], st["best_epoch"] = init["proxy_map"], 0.0
        save_best(ctx, model)
    deadline = ctx.t0 + float(cfg["budget_s"]) - 60 * float(plan["reserve_min"])
    ctx.status("training", plan=plan["reason"])
    ctx.log(f"   [train] {len(keep)} pairs/epoch, {spe} steps/epoch x {plan['epochs']} = {total} "
            f"steps; training deadline in {(deadline - time.time()) / 60:.0f} min")
    last_ck, t_train, stop = time.time(), time.time(), "completed"
    kill_at = int(cfg.get("kill_at_step", -1))
    ema, tok_n, tok_t = None, 0, 0.0
    while st["epoch"] < int(plan["epochs"]):
        order = [keep[i] for i in np.random.default_rng(int(cfg["seed"]) + 1000 * st["epoch"])
                 .permutation(len(keep))]
        rng = np.random.default_rng(int(cfg["seed"]) + 7 + 1000 * st["epoch"] + st["pos"])
        while st["pos"] < spe:
            if time.time() > deadline:
                stop = "time guard"
                break
            batch = order[st["pos"] * B:(st["pos"] + 1) * B]
            if not batch:
                st["pos"] = spe
                break
            t1 = time.time()
            loss, ntok = stepper.step(batch, rng, st["step"], total)
            tok_n, tok_t = tok_n + ntok, tok_t + time.time() - t1
            st["step"] += 1
            st["pos"] += 1
            if loss is None:
                st["skipped"] += 1
                if st["step"] >= 20 and st["skipped"] > 0.05 * st["step"]:
                    raise JobExit(4, f"{st['skipped']} of {st['step']} steps non-finite")
            else:
                ema = loss if ema is None else 0.95 * ema + 0.05 * loss
            if st["step"] % 25 == 0:
                ctx.log(f"   step {st['step']}/{total} ep {st['epoch'] + 1} loss {ema:.4f} lr "
                        f"{sched.get_last_lr()[0]:.2e} scale {float(model.classifier.scale().detach()):.1f}"
                        f" peak {peak_gb():.1f} GB skipped {st['skipped']}")
            if st["step"] == kill_at:
                save_last(ctx, model, opt, sched, scaler, st)
                os._exit(137)
            if st["pos"] in eval_at:
                pe = proxy_eval(ctx, model, cache, d, amp_ctx, stepper.micro, k_proxy)
                fe = st["epoch"] + st["pos"] / spe
                rec = {"type": "eval", "step": st["step"], "epoch": round(fe, 3),
                       "train_loss": ema, "dev_proxy_map": pe["proxy_map"],
                       "dev_proxy_f1": pe["proxy_f1"], "lr": sched.get_last_lr()[0],
                       "minutes": (time.time() - ctx.t0) / 60, "skipped_steps": st["skipped"]}
                ctx.jlog(rec)
                st["evals"].append(rec)
                ctx.log(f"   [eval] epoch {fe:.2f}: dev proxy MAP {pe['proxy_map']:.4f} F1@7 "
                        f"{pe['proxy_f1']:.4f} (zero-shot {st['init_proxy'] or 0:.4f})")
                if ema is not None and not math.isfinite(ema):
                    raise JobExit(5, "watchdog: non-finite training loss")
                if pe["proxy_map"] > st["best"] + 1e-6:
                    st["best"], st["best_epoch"], st["bad"] = pe["proxy_map"], round(fe, 3), 0
                    save_best(ctx, model)
                else:
                    st["bad"] += 1
                write_json(os.path.join(ctx.run_dir, "logs", "progress.json"),
                           {"best_proxy_map": st["best"], "best_epoch": st["best_epoch"],
                            "step": st["step"], "total_steps": total, "epochs_run": round(fe, 3)})
                if st["bad"] >= patience:
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
    plan["train_tokens_per_s"] = tok_n / max(1e-6, tok_t)
    ctx.log(f"   [train] stopped: {stop} at step {st['step']}/{total}")
    stepper.teacher = teacher = None
    best_pt = os.path.join(ctx.run_dir, "checkpoints", "best.pt")
    tech.force_final(model)
    if not os.path.exists(best_pt):
        save_best(ctx, model)
    load_trainable_state(model, torch_load(best_pt))
    if os.path.exists(last_pt):
        os.remove(last_pt)
    return finish(ctx, model, cache, d, amp_ctx, st, plan, size, checks, stop, tech, stepper.micro)


def run_overfit(ctx, stepper) -> int:
    """4 fixed pairs, up to 60 steps: the contrastive loss must fall below 30% of its start."""
    idx = list(range(min(4, len(stepper.d["pairs"]))))
    for g in stepper.opt.param_groups:
        g["lr"] = g["initial_lr"] = max(g["lr"], float(ctx.cfg.get("overfit_lr", 3e-3)))
    stepper.sched = None
    first = last = None
    n = 0
    for n in range(1, 61):
        l, _ = stepper.step(idx, np.random.default_rng(0), n, 60)
        if l is not None:
            first = l if first is None else first
            last = l
        if last is not None and last < 0.3 * first:
            break
    ok = last is not None and first is not None and last < 0.3 * first
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
