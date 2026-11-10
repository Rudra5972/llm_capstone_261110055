"""
ner_train.py -- one training run for one (experiment, model). Launched by run_experiment.py:

    python ner_train.py --run-dir RUN --mode train                       # 1 GPU / CPU
    python -m torch.distributed.run --nproc_per_node=2 ner_train.py --run-dir RUN --mode train

Reads RUN/logs/run_config.json. Modes:
  smoke    ~30 optimizer steps; writes logs/smoke.json (tokens/s, memory, projected minutes)
  overfit  64 entity chunks, 150 steps; must drive the loss down (catches label misalignment)
  probe    lr x {1/3, 1, 3} for a short budget each; writes logs/lr_probe.json
  train    the real run (checkpoints, per-epoch dev F1, early stopping, time guard, resume)

Exit codes: 0 ok | 2 time budget reached (checkpoint saved) | 3 out of memory in DDP |
            4 NaN abort | 5 early-fail watchdog | 1 any other error
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ner_common import read_json, write_json
from ner_data import IGNORE, SnapRules, build_train_chunks, load_cache
from ner_models import (build_ner_model, count_params, load_trainable_state,
                        rebuild_trained_model, torch_load, trainable_state_dict)
from ner_peft import apply_technique
from ner_predict import amp_setup, quick_dev_f1

EXIT_TIME, EXIT_OOM, EXIT_NAN, EXIT_EARLYFAIL = 2, 3, 4, 5
FULL_PARAM = ("full_ft", "kd", "student_hard", "partial_ft")


def is_oom(e):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()


class Ctx:
    """Distributed / device context."""

    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        cuda = torch.cuda.is_available()
        if self.world > 1:
            if cuda:
                torch.cuda.set_device(self.local)
            dist.init_process_group("nccl" if cuda else "gloo",
                                    timeout=datetime.timedelta(minutes=10))
        self.device = torch.device("cuda", self.local) if cuda else torch.device("cpu")
        self.main = self.rank == 0

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def bcast(self, values):
        """Broadcast a list of floats from rank 0."""
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.broadcast(t, 0)
        return t.tolist()

    def all_min(self, v):
        t = torch.tensor([float(v)], device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return t.item()

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def log(ctx, run_dir, msg, record=None):
    if not ctx.main:
        return
    print(msg, flush=True)
    if record is not None:
        with open(os.path.join(run_dir, "logs", "train_log.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================================
# data
# =============================================================================================
class EpochOrder:
    """Deterministic global permutation per epoch; rank r takes every world-th item after
    skipping the first `skip` (already consumed, used on resume)."""

    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def indices(self, epoch, rank, world, skip=0):
        perm = np.random.default_rng(self.seed + 1000 * epoch).permutation(self.n)
        rest = perm[skip:]
        if len(rest) == 0:
            return []
        # every rank must run the SAME number of micro-batches (or DDP deadlocks): pad by
        # wrapping around, exactly like torch's DistributedSampler
        per_rank = math.ceil(len(rest) / world)
        if per_rank * world > len(rest):
            rest = np.concatenate([rest, np.resize(perm, per_rank * world - len(rest))])
        return rest[rank::world].tolist()


def collate(xs, ys, idx, pad_id):
    L = max(len(xs[i]) for i in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    y = torch.full((len(idx), L), IGNORE, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        n = len(xs[i])
        x[r, :n] = torch.from_numpy(xs[i].astype(np.int64))
        y[r, :n] = torch.from_numpy(ys[i].astype(np.int64))
        m[r, :n] = 1
    return x, y, m


# =============================================================================================
# model / optimizer
# =============================================================================================
def make_model(cfg, ctx, lr_mult=1.0):
    mid = cfg["model"]
    model, _ = build_ner_model(cfg["phase1_dir"], cfg["n_labels"], cfg["attn_mode"], cfg["head"],
                               cfg.get("layer_ids"), cfg.get("dropout", 0.1))
    c = dict(cfg)
    c["lr"] = cfg["lr"] * lr_mult
    c["head_lr"] = cfg["head_lr"] * (lr_mult if cfg["technique"] in FULL_PARAM else 1.0)
    groups = apply_technique(model, cfg["technique"], c,
                             log=(lambda m: print(m, flush=True)) if ctx.main else (lambda m: None))
    model.to(ctx.device)
    for g in groups:
        g["initial_lr"] = g["lr"]
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups],
                            betas=(0.9, 0.98), eps=1e-8)
    return model, opt


def make_scheduler(opt, total_steps, warmup_frac):
    warm = max(1, int(round(warmup_frac * total_steps)))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def wrap_ddp(model, ctx):
    if ctx.world == 1:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    kw = {"device_ids": [ctx.local]} if ctx.device.type == "cuda" else {}
    return DDP(model, find_unused_parameters=False, **kw)


def load_teacher(cfg, ctx):
    tdir = cfg["teacher_dir"]
    tcfg = read_json(os.path.join(tdir, "run_config.json"))
    from ner_common import find_phase1_models
    p1 = find_phase1_models()
    if tcfg.get("phase1_dir"):
        p1.setdefault(tcfg["model"], tcfg["phase1_dir"])
    teacher = rebuild_trained_model(tcfg, os.path.join(tdir, "best.pt"), p1, ctx.device)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval(), tcfg


def kd_loss(student_logits, teacher_logits, labels, alpha, T, class_weights=None):
    mask = labels != IGNORE
    s = student_logits[mask].float()
    t = teacher_logits[mask].float()
    hard = F.cross_entropy(s, labels[mask], weight=class_weights)
    soft = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True,
                    reduction="batchmean") * (T * T)
    return alpha * hard + (1 - alpha) * soft


def class_weight_tensor(cfg, ys, num_tags, device):
    if cfg.get("class_weight", "none") != "sqrt_inv":
        return None
    cnt = np.bincount(np.concatenate([y[y >= 0] for y in ys]).astype(np.int64), minlength=num_tags)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


# =============================================================================================
# the training loop (shared by every mode)
# =============================================================================================
def train_loop(cfg, ctx, run_dir, meta, docs, xs, ys, mode, lr_mult=1.0, max_steps=None,
               epochs=None, resume=True, eval_each_epoch=True):
    """Returns a summary dict. Raises SystemExit with the exit codes in the module doc."""
    torch.manual_seed(cfg["seed"] + ctx.rank)
    random.seed(cfg["seed"] + ctx.rank)
    np.random.seed(cfg["seed"] + ctx.rank)
    model, opt = make_model(cfg, ctx, lr_mult)
    raw_model = model
    amp_dtype, amp_ctx = amp_setup(ctx.device)
    scaler = make_scaler(amp_dtype == torch.float16)
    teacher = None
    if cfg["technique"] == "kd":
        teacher, _ = load_teacher(cfg, ctx)
    if cfg["technique"] in ("kd", "student_hard") and cfg.get("enforce_size", True):
        n_s = count_params(raw_model)
        assert 50e6 <= n_s <= 300e6, f"student has {n_s / 1e6:.1f}M params (must be 50M-300M)"
        if teacher is not None:
            assert n_s < count_params(teacher), "student must be smaller than the teacher"
    cw = class_weight_tensor(cfg, ys, raw_model.num_tags, ctx.device)
    snap = SnapRules.from_json(meta.get("snap_rules") or {})
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    half_best = cfg["technique"] in FULL_PARAM

    # ---- micro-batch: memory probe on the longest possible batch -----------------------------
    mb = int(cfg.get("micro_batch", 16))
    if ctx.device.type == "cuda":
        L = cfg["chunk_len"]
        while True:
            try:
                x = torch.randint(3, meta["vocab_size"], (mb, L), device=ctx.device)
                with amp_ctx():
                    logits, loss = raw_model(x, torch.ones_like(x), labels=torch.zeros_like(x))
                scaler.scale(loss).backward()
                raw_model.zero_grad(set_to_none=True)
                del x, logits, loss
                torch.cuda.empty_cache()
                break
            except RuntimeError as e:
                if not is_oom(e) or mb == 1:
                    raise
                raw_model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                mb //= 2
        mb = int(ctx.all_min(mb))
        torch.cuda.reset_peak_memory_stats(ctx.device)
    eff = int(cfg.get("effective_batch", 32))
    accum = max(1, math.ceil(eff / (mb * ctx.world)))
    per_step = mb * ctx.world * accum
    n = len(xs)
    steps_per_epoch = max(1, math.ceil(n / per_step))
    epochs = epochs or cfg["epochs"]
    total_steps = max_steps or steps_per_epoch * epochs
    sched = make_scheduler(opt, total_steps, cfg.get("warmup_frac", 0.06))
    model = wrap_ddp(raw_model, ctx)
    order = EpochOrder(n, cfg["seed"])

    # ---- resume -------------------------------------------------------------------------------
    st = {"epoch": 0, "consumed": 0, "global_step": 0, "best_f1": -1.0, "best_epoch": -1,
          "bad_epochs": 0, "history": [], "skipped": 0, "train_seconds": 0.0, "tokens": 0}
    last_pt = os.path.join(ckpt_dir, "last.pt")
    if mode == "train" and resume and os.path.exists(last_pt):
        ck = torch_load(last_pt)
        load_trainable_state(raw_model, ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        st.update(ck["state"])
        if ck.get("total_steps") != total_steps:
            log(ctx, run_dir, f"   [resume] total steps changed {ck.get('total_steps')} -> "
                              f"{total_steps} (micro-batch/world change); schedule rescaled")
        log(ctx, run_dir, f"   [resume] epoch {st['epoch']} consumed {st['consumed']} "
                          f"step {st['global_step']} best F1 {st['best_f1']:.4f}")

    def save_last():
        if not ctx.main:
            return
        tmp = last_pt + ".tmp"
        torch.save({"trainable": trainable_state_dict(raw_model), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "state": st, "total_steps": total_steps,
                    "micro_batch": mb, "world": ctx.world}, tmp)
        os.replace(tmp, last_pt)

    def save_best():
        if not ctx.main:
            return
        p = os.path.join(ckpt_dir, "best.pt")
        torch.save(trainable_state_dict(raw_model, half=half_best), p + ".tmp")
        os.replace(p + ".tmp", p)

    deadline = float(cfg.get("deadline_unix", 0) or 0)
    margin = float(cfg.get("deadline_margin_s", 300))
    ckpt_every = float(cfg.get("ckpt_every_min", 30)) * 60
    last_ckpt_t = time.time()
    t_start = time.time()
    clip = float(cfg.get("grad_clip", 1.0))
    log_every = int(cfg.get("log_every", 50))
    losses = []
    first_loss = None
    stop_reason = None
    tok_count, tok_t0 = 0, time.time()
    min_steps_for_watchdog = min(100, max(1, total_steps // 3))
    watchdog_done = st["global_step"] >= min_steps_for_watchdog

    model.train()
    while st["epoch"] < epochs and stop_reason is None:
        idx = order.indices(st["epoch"], ctx.rank, ctx.world, st["consumed"])
        pos = 0
        while pos < len(idx):
            if max_steps and st["global_step"] >= max_steps:
                stop_reason = "max_steps"
                break
            # ---------------- one optimizer step (accum micro-batches) ----------------
            step_loss = 0.0
            micro = 0
            for a in range(accum):
                chunk = idx[pos:pos + mb]
                if not chunk:
                    break
                pos += len(chunk)
                micro += 1
                x, y, m = collate(xs, ys, chunk, meta["pad_id"])
                x, y, m = x.to(ctx.device), y.to(ctx.device), m.to(ctx.device)
                sync = (a == accum - 1) or pos >= len(idx)
                cm = model.no_sync() if (ctx.world > 1 and not sync) else _null()
                try:
                    with cm:
                        with amp_ctx():
                            logits, loss = model(x, m, labels=y, class_weights=cw)
                            if teacher is not None:
                                with torch.no_grad():
                                    t_logits, _ = teacher(x, m)
                                loss = kd_loss(logits, t_logits, y, cfg.get("kd_alpha", 0.5),
                                               cfg.get("kd_T", 2.0), cw)
                        scaler.scale(loss / accum).backward()
                except RuntimeError as e:
                    if not is_oom(e):
                        raise
                    if ctx.world > 1:
                        log(ctx, run_dir, "   [oom] out of memory under DDP -> exit for retry")
                        save_last()
                        raise SystemExit(EXIT_OOM)
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if mb == 1:
                        raise
                    mb //= 2
                    accum *= 2
                    log(ctx, run_dir, f"   [oom] micro-batch -> {mb}, accumulation -> {accum}")
                    pos -= len(chunk)
                    step_loss, micro = 0.0, 0
                    break
                lv = float(loss.detach())
                step_loss += lv / accum
                tok_count += int(m.sum())
            if micro == 0:
                continue
            # ---------------- step (NaN-safe; identical decision on all ranks) ----------------
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in raw_model.parameters()
                                                    if p.requires_grad], clip)
            # gnorm is identical on every rank (DDP all-reduces the gradients), so the skip
            # decision is too. fp16: GradScaler itself skips overflow steps; the first ~30 steps
            # are its normal scale calibration and are not counted as failures.
            finite = bool(torch.isfinite(gnorm))
            if finite or amp_dtype == torch.float16:
                scaler.step(opt)
            if not finite and st["global_step"] >= 30:
                st["skipped"] += 1
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            st["global_step"] += 1
            st["consumed"] = min(n, st["consumed"] + per_step)
            if first_loss is None and math.isfinite(step_loss):
                first_loss = step_loss
            losses.append(step_loss)
            collapsed = amp_dtype == torch.float16 and scaler.get_scale() < 1e-4
            if collapsed or (st["skipped"] >= 5 and
                             st["skipped"] > 0.05 * max(1, st["global_step"] - 30)):
                save_last()
                log(ctx, run_dir, f"   [nan] {st['skipped']} steps with non-finite gradients "
                                  f"(loss scale {scaler.get_scale():.3g}) -> abort")
                raise SystemExit(EXIT_NAN)
            if st["global_step"] % log_every == 0 or st["global_step"] == 1:
                el = time.time() - tok_t0
                rec = {"step": st["global_step"], "epoch": st["epoch"],
                       "loss": float(np.mean(losses[-log_every:])),
                       "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
                       "tokens_per_s": tok_count * ctx.world / max(el, 1e-9),
                       "peak_mem_gb": (torch.cuda.max_memory_allocated(ctx.device) / 1e9
                                       if ctx.device.type == "cuda" else 0.0),
                       "skipped": st["skipped"]}
                log(ctx, run_dir, f"   step {rec['step']:>5}/{total_steps} | ep {st['epoch']} | "
                                  f"loss {rec['loss']:.4f} | lr {rec['lr']:.2e} | gnorm "
                                  f"{rec['grad_norm']:.2f} | {rec['tokens_per_s']:,.0f} tok/s | "
                                  f"mem {rec['peak_mem_gb']:.1f} GB", rec if mode == "train" else None)
            # ---------------- time guard / periodic checkpoint ----------------
            if mode == "train":
                over = [1.0 if (deadline and time.time() > deadline - margin) else 0.0]
                over = ctx.bcast(over)[0]
                if over:
                    st["train_seconds"] += time.time() - t_start
                    save_last()
                    log(ctx, run_dir, "   [time] notebook time budget reached -> checkpoint saved")
                    raise SystemExit(EXIT_TIME)
                if time.time() - last_ckpt_t > ckpt_every:
                    save_last()
                    last_ckpt_t = time.time()
                    log(ctx, run_dir, f"   [ckpt] last.pt written at step {st['global_step']}")
        if stop_reason is not None:
            break
        # ---------------- end of epoch ----------------
        st["epoch"] += 1
        st["consumed"] = 0
        if not eval_each_epoch:
            continue
        f1 = -1.0
        if ctx.main:
            f1, _m = quick_dev_f1(raw_model, meta, docs, cfg, ctx.device, amp_ctx, snap)
            raw_model.train()
        f1 = ctx.bcast([f1])[0]
        improved = f1 > st["best_f1"]
        if improved:
            st["best_f1"], st["best_epoch"], st["bad_epochs"] = f1, st["epoch"], 0
            save_best()
        else:
            st["bad_epochs"] += 1
        st["history"].append({"epoch": st["epoch"], "dev_macro_f1": f1,
                              "train_loss": float(np.mean(losses[-steps_per_epoch:]))})
        log(ctx, run_dir, f"   == epoch {st['epoch']}/{epochs} | dev strict macro-F1 {f1:.4f} "
                          f"{'(best)' if improved else ''} | train loss "
                          f"{st['history'][-1]['train_loss']:.4f}",
            {"epoch_end": st["epoch"], "dev_macro_f1": f1} if mode == "train" else None)
        if mode == "train":
            if not watchdog_done and st["global_step"] >= min_steps_for_watchdog:
                watchdog_done = True
                if st["best_f1"] < float(cfg.get("watchdog_min_f1", 0.05)):
                    save_last()
                    log(ctx, run_dir, f"   [watchdog] dev F1 {st['best_f1']:.4f} < "
                                      f"{cfg.get('watchdog_min_f1', 0.05)} after "
                                      f"{st['global_step']} steps -> stop. Check data_report.json "
                                      f"(G1), the learning rate and logs/overfit.json.")
                    raise SystemExit(EXIT_EARLYFAIL)
            if st["bad_epochs"] >= int(cfg.get("patience", 3)):
                stop_reason = "early_stop"
            if time.time() - last_ckpt_t > 15 * 60:
                save_last()
                last_ckpt_t = time.time()
    st["train_seconds"] += time.time() - t_start
    el = time.time() - tok_t0
    summary = {"status": stop_reason or "done", "epochs_run": st["epoch"],
               "global_step": st["global_step"], "steps_per_epoch": steps_per_epoch,
               "total_steps": total_steps, "micro_batch": mb, "accum": accum, "world": ctx.world,
               "best_dev_macro_f1": st["best_f1"], "best_epoch": st["best_epoch"],
               "history": st["history"], "skipped_steps": st["skipped"],
               "first_loss": first_loss,
               "last_loss": float(np.mean(losses[-10:])) if losses else None,
               "tokens_per_s": tok_count * ctx.world / max(el, 1e-9),
               "seconds_per_step": el / max(1, len(losses)),
               "peak_mem_gb": (torch.cuda.max_memory_allocated(ctx.device) / 1e9
                               if ctx.device.type == "cuda" else 0.0),
               "trainable_params": count_params(raw_model, True),
               "total_params": count_params(raw_model), "train_seconds": st["train_seconds"],
               "lr_mult": lr_mult}
    return summary, raw_model


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# =============================================================================================
# modes
# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["train", "smoke", "overfit", "probe"], default="train")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run_dir = a.run_dir
    cfg = read_json(os.path.join(run_dir, "logs", "run_config.json"))
    ctx = Ctx()
    code = 0
    try:
        meta, docs = load_cache(cfg["data_dir"])
        train_ids = meta["split"]["train"] + (meta["split"]["dev"] if cfg.get("refit") else [])
        tr_docs = [docs[d] for d in train_ids]
        xs, ys, cstats = build_train_chunks(tr_docs, cfg["chunk_len"], cfg["train_stride"],
                                            cfg.get("neg_keep", 0.3), cfg["seed"], meta["eos_id"])
        if ctx.main:
            print(f"   train chunks: {cstats}", flush=True)
        # first-batch assertions (section 9.4)
        x, y, m = collate(xs, ys, list(range(min(4, len(xs)))), meta["pad_id"])
        assert x.max().item() < meta["vocab_size"] and x.min().item() >= 0, "token id out of range"
        assert ((y == IGNORE) | ((y >= 0) & (y < 1 + 2 * len(meta["labels"])))).all(), "bad label"
        assert (y[m == 0] == IGNORE).all(), "labels on padding"
        assert (y[:, 0] == IGNORE).all() and (x[:, 0] == meta["eos_id"]).all(), "EOS slot wrong"
        assert x.shape[1] <= cfg["chunk_len"] <= 1024

        if a.mode == "smoke":
            steps = a.max_steps or 30
            s, _ = train_loop(cfg, ctx, run_dir, meta, docs, xs, ys, "smoke", max_steps=steps,
                              resume=False, eval_each_epoch=False)
            if ctx.main:
                epochs = cfg["epochs"]
                train_min = s["seconds_per_step"] * s["steps_per_epoch"] * epochs / 60
                n_dev_tok = sum(len(docs[d]["ids"]) for d in meta["split"]["dev"])
                eval_min = epochs * n_dev_tok / max(1.0, 3 * s["tokens_per_s"]) / 60
                s["projected_train_minutes"] = train_min + eval_min
                s["loss_decreased"] = (s["last_loss"] is not None and s["first_loss"] is not None
                                       and s["last_loss"] < s["first_loss"])
                write_json(os.path.join(run_dir, "logs", f"smoke{a.tag}.json"), s)
                print(f"   smoke ok: {s['tokens_per_s']:,.0f} tok/s | mem {s['peak_mem_gb']:.1f} GB"
                      f" | micro-batch {s['micro_batch']} x accum {s['accum']} x world "
                      f"{s['world']} | projected {s['projected_train_minutes']:.1f} min", flush=True)
        elif a.mode == "overfit":
            pos = [i for i, yy in enumerate(ys) if (yy > 0).any()][:64]
            c2 = dict(cfg)
            c2["dropout"] = 0.0
            c2["warmup_frac"] = 0.05
            # memorising 64 chunks is a pipeline check, not a tuning run: use a high LR
            mult = 10.0 if cfg["technique"] in FULL_PARAM else 3.0
            s, _ = train_loop(c2, ctx, run_dir, meta, docs, [xs[i] for i in pos],
                              [ys[i] for i in pos], "overfit", lr_mult=mult,
                              max_steps=a.max_steps or 150, epochs=10 ** 6, resume=False,
                              eval_each_epoch=False)
            if ctx.main:
                passed = s["last_loss"] < 0.05 or s["last_loss"] < 0.2 * s["first_loss"]
                s["passed"] = bool(passed)
                # skip the model only if it learned (almost) nothing -- a likely label bug
                s["fatal"] = bool(not passed and s["last_loss"] > 0.6 * s["first_loss"])
                write_json(os.path.join(run_dir, "logs", "overfit.json"), s)
                print(f"   overfit-64: loss {s['first_loss']:.3f} -> {s['last_loss']:.4f} "
                      f"({'PASS' if passed else 'FAIL'})", flush=True)
        elif a.mode == "probe":
            results = []
            spe = None
            for mult in cfg.get("probe_mults", [1 / 3, 1.0, 3.0]):
                budget = a.max_steps or cfg.get("probe_steps")
                if budget is None:
                    if spe is None:
                        eff = cfg.get("effective_batch", 32)
                        spe = max(1, math.ceil(len(xs) / eff))
                    budget = min(200, max(30, spe))
                s, model = train_loop(cfg, ctx, run_dir, meta, docs, xs, ys, "probe",
                                      lr_mult=mult, max_steps=budget, epochs=10 ** 6,
                                      resume=False, eval_each_epoch=False)
                f1 = -1.0
                if ctx.main:
                    _amp, amp_ctx = amp_setup(ctx.device)
                    f1, _ = quick_dev_f1(model, meta, docs, cfg, ctx.device, amp_ctx,
                                         SnapRules.from_json(meta.get("snap_rules") or {}))
                f1 = ctx.bcast([f1])[0]
                results.append({"lr_mult": mult, "lr": cfg["lr"] * mult, "steps": budget,
                                "dev_macro_f1": f1, "last_loss": s["last_loss"]})
                if ctx.main:
                    print(f"   probe lr x{mult:.3g} ({cfg['lr'] * mult:.2e}): dev F1 {f1:.4f} "
                          f"loss {s['last_loss']:.4f}", flush=True)
                del model
                if ctx.device.type == "cuda":
                    torch.cuda.empty_cache()
            best = max(results, key=lambda r: (r["dev_macro_f1"], -(r["last_loss"] or 1e9)))
            if ctx.main:
                write_json(os.path.join(run_dir, "logs", "lr_probe.json"),
                           {"best_lr_mult": best["lr_mult"], "best_lr": best["lr"],
                            "results": results})
        else:
            s, _ = train_loop(cfg, ctx, run_dir, meta, docs, xs, ys, "train")
            if ctx.main:
                s["chunk_stats"] = cstats
                write_json(os.path.join(run_dir, "logs", "train_summary.json"), s)
                if s["best_epoch"] < 0:
                    raise RuntimeError("training produced no best checkpoint")
                print(f"   training {s['status']}: best dev macro-F1 {s['best_dev_macro_f1']:.4f} "
                      f"at epoch {s['best_epoch']} ({s['train_seconds'] / 60:.1f} min)", flush=True)
    except SystemExit as e:
        code = int(e.code or 0)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
