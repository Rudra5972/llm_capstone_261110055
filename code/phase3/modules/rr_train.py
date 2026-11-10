"""
rr_train.py -- one training run for one job (experiment x model [x variant]).

    python rr_train.py --run-dir RUN --mode train                       # 1 GPU / CPU
    python -m torch.distributed.run --nproc_per_node=2 rr_train.py --run-dir RUN --mode train

Same structure as the Phase-2 ner_train.py (DDP via torchrun, fp16 autocast + GradScaler,
30-minute checkpoints, resume, NaN / OOM handling, exit codes), adapted to sentence windows.
Reads RUN/logs/run_config.json. Modes:
  smoke    selftest.job_checks on the REAL model (zero-init == base logits, merge, trainable %,
           prefix visibility, output lengths) + ~30 optimizer steps; writes logs/smoke.json
           (tokens/s, memory, projected minutes)
  overfit  32 labelled sentences, <= 300 steps: train accuracy must exceed 95%
  probe    lr x {1/3, 1, 3} for ~1 epoch each; writes logs/lr_probe.json
  train    the real run: cosine schedule with warm-up, AdamW, clip 1.0, dev pooled Macro-F1 each
           epoch, early stopping with patience, best checkpoint, time guard, resume

Exit codes: 0 ok | 2 time budget reached (checkpoint saved) | 3 out of memory in DDP |
            4 NaN abort | 5 early-fail watchdog | 6 smoke self-check failed | 1 other error
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
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from rr_common import default_input_roots, find_phase1_models, read_json, write_json
from rr_data import IGNORE, build_windows, class_counts, class_weights, load_cache
from rr_models import (build_rr_model, collate, load_trainable_state, model_size, torch_load,
                       trainable_state_dict)
from rr_peft import get_technique
from rr_predict import amp_setup, eval_keys, quick_dev_f1

EXIT_TIME, EXIT_OOM, EXIT_NAN, EXIT_EARLYFAIL, EXIT_SELFCHECK = 2, 3, 4, 5, 6
FULL_PARAM = ("full_ft", "kd")


def is_oom(e: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()


class Ctx:
    """Distributed / device context (Phase-2 code)."""

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


class EpochOrder:
    """Deterministic permutation per epoch; rank r takes every world-th item after skipping the
    first `skip` (already consumed; resume). Pads by wrap-around so all ranks step equally."""

    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def indices(self, epoch, rank, world, skip=0):
        perm = np.random.default_rng(self.seed + 1000 * epoch).permutation(self.n)
        rest = perm[skip:]
        if len(rest) == 0:
            return []
        per_rank = math.ceil(len(rest) / world)
        if per_rank * world > len(rest):
            rest = np.concatenate([rest, np.resize(perm, per_rank * world - len(rest))])
        return rest[rank::world].tolist()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# =============================================================================================
# model / optimizer / schedule / loss
# =============================================================================================
def make_model(cfg: Dict, ctx: Ctx, lr_mult: float = 1.0):
    model, _ = build_rr_model(cfg["phase1_dir"], cfg["n_labels"], cfg["attn_mode"], cfg["head"],
                              cfg.get("layer_ids"), cfg.get("dropout", 0.1),
                              cfg.get("pool", "marker"))
    c = dict(cfg)
    c["lr"] = cfg["lr"] * lr_mult
    c["head_lr"] = cfg["head_lr"] * (lr_mult if cfg["technique"] in FULL_PARAM else 1.0)
    tech = get_technique(cfg["technique"])
    if cfg["technique"] == "kd" and cfg.get("kd_hidden_weight", 0):
        from rr_kd import attach_projection
        attach_projection(model, cfg)
    groups = tech.apply(model, c, log=(lambda m: print(m, flush=True)) if ctx.main
                        else (lambda m: None))
    model.to(ctx.device)
    for g in groups:
        g["initial_lr"] = g["lr"]
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups],
                            betas=(0.9, 0.98), eps=1e-8)
    return model, opt, tech


def make_scheduler(opt, total_steps: int, warmup_frac: float, kind: str = "cosine"):
    warm = max(1, int(round(warmup_frac * total_steps)))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        if kind == "linear":
            return max(0.0, 1.0 - prog)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def make_scaler(enabled: bool):
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


def make_loss(cfg: Dict, counts: np.ndarray, device):
    """ce | weighted_ce (inverse-sqrt frequency) | focal (gamma 2) | logit_adjust (Menon et al.
    2021, tau * log prior added to the training logits only)."""
    kind = cfg.get("loss", "weighted_ce")
    w = torch.tensor(class_weights(counts), device=device) if kind == "weighted_ce" else None
    prior = torch.tensor((counts + 1e-3) / (counts.sum() + 1e-3 * len(counts)),
                         dtype=torch.float32, device=device)
    gamma = float(cfg.get("focal_gamma", 2.0))
    tau = float(cfg.get("logit_adjust_tau", 1.0))

    def fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if kind in ("ce", "weighted_ce"):
            return F.cross_entropy(logits, y, weight=w)
        if kind == "focal":
            logp = F.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
            return (-(1 - logp.exp()) ** gamma * logp).mean()
        if kind == "logit_adjust":
            return F.cross_entropy(logits + tau * torch.log(prior), y)
        raise ValueError(f"unknown loss {kind!r}")
    return fn


def train_windows(cfg: Dict, meta: Dict, docs: Dict) -> List[Dict]:
    keys = [k for k in meta["split"]["train"] + (meta["split"]["dev"] if cfg.get("refit") else [])
            if docs[k]["domain"] in set(cfg.get("domains") or [])]
    return build_windows(docs, keys, cfg, meta, "train")


# =============================================================================================
# the training loop (shared by every mode)
# =============================================================================================
def train_loop(cfg, ctx, run_dir, meta, docs, windows, mode, lr_mult=1.0, max_steps=None,
               epochs=None, resume=True, eval_each_epoch=True, labels_override=None,
               kill_at_step=None):
    """Returns (summary, raw_model, technique). Raises SystemExit with the module's codes."""
    torch.manual_seed(cfg["seed"] + ctx.rank)
    random.seed(cfg["seed"] + ctx.rank)
    np.random.seed(cfg["seed"] + ctx.rank)
    model, opt, tech = make_model(cfg, ctx, lr_mult)
    raw_model = model
    amp_dtype, amp_ctx = amp_setup(ctx.device)
    scaler = make_scaler(amp_dtype == torch.float16)
    tr_keys = [k for k in meta["split"]["train"] if docs[k]["domain"] in set(cfg["domains"])]
    counts = class_counts(docs, tr_keys, cfg["n_labels"])
    loss_fn = make_loss(cfg, counts, ctx.device)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    half_best = cfg["technique"] in FULL_PARAM

    # ---- KD: teacher outputs, cached once per training window ---------------------------------
    t_cache = None
    if cfg["technique"] == "kd" and mode != "overfit":
        from rr_kd import build_teacher_cache, load_teacher
        teacher, _tcfg = load_teacher(cfg, ctx.device, find_phase1_models(default_input_roots()))
        n_s, n_t = model_size(raw_model)["total_params"], model_size(teacher)["total_params"]
        log(ctx, run_dir, f"   [kd] student {n_s / 1e6:.1f}M params < teacher {n_t / 1e6:.1f}M "
                          f"({n_s / n_t:.2f}x)")
        assert n_s < n_t, "the KD student must be smaller than the teacher"
        if ctx.main:
            t_cache = build_teacher_cache(teacher, windows, cfg, meta, ctx.device, amp_ctx,
                                          os.path.join(run_dir, "teacher_cache.pt"),
                                          float(cfg.get("kd_hidden_weight", 0)) > 0,
                                          log=lambda m: log(ctx, run_dir, m))
        ctx.barrier()
        if not ctx.main:
            t_cache = torch.load(os.path.join(run_dir, "teacher_cache.pt"), map_location="cpu")
        del teacher
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- micro-batch: memory probe on the longest window --------------------------------------
    mb = int(cfg.get("micro_batch", 4))
    if ctx.device.type == "cuda":
        longest = max(range(len(windows)), key=lambda i: len(windows[i]["tokens"]))
        while True:
            try:
                b = collate(windows, [longest] * mb, meta["pad_id"], ctx.device)
                with amp_ctx():
                    logits, _v = raw_model(b["input_ids"], b["attention_mask"], b["markers"],
                                           b["starts"], b["marker_mask"])
                    loss = logits.float().mean()
                scaler.scale(loss).backward()
                raw_model.zero_grad(set_to_none=True)
                del b, logits, loss, _v
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
    eff = int(cfg.get("effective_batch", 8))
    accum = max(1, math.ceil(eff / (mb * ctx.world)))
    per_step = mb * ctx.world * accum
    n = len(windows)
    steps_per_epoch = max(1, math.ceil(n / per_step))
    epochs = epochs or cfg["epochs"]
    total_steps = max_steps or steps_per_epoch * epochs
    sched = make_scheduler(opt, total_steps, cfg.get("warmup_frac", 0.06),
                           cfg.get("schedule", "cosine"))
    model = wrap_ddp(raw_model, ctx)
    order = EpochOrder(n, cfg["seed"])

    # ---- resume ---------------------------------------------------------------------------------
    st = {"epoch": 0, "consumed": 0, "global_step": 0, "best_f1": -1.0, "best_epoch": -1,
          "bad_epochs": 0, "history": [], "skipped": 0, "train_seconds": 0.0, "resumed_from": None,
          "max_f1_seen": -1.0}
    last_pt = os.path.join(ckpt_dir, "last.pt")
    if mode == "train" and resume and os.path.exists(last_pt):
        ck = torch_load(last_pt)
        load_trainable_state(raw_model, ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        st.update(ck["state"])
        st["resumed_from"] = st["global_step"]
        if ck.get("total_steps") != total_steps:
            log(ctx, run_dir, f"   [resume] total steps changed {ck.get('total_steps')} -> "
                              f"{total_steps} (micro-batch/world change)")
        log(ctx, run_dir, f"   [resume] epoch {st['epoch']} consumed {st['consumed']} "
                          f"step {st['global_step']} best F1 {st['best_f1']:.4f}")
        if ctx.main:
            write_json(os.path.join(run_dir, "logs", "resume.json"),
                       {"resumed_at_step": st["global_step"], "epoch": st["epoch"],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    def save_last():
        if not ctx.main:
            return
        tmp = last_pt + ".tmp"
        torch.save({"trainable": trainable_state_dict(raw_model), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "state": st, "total_steps": total_steps, "micro_batch": mb,
                    "world": ctx.world}, tmp)
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
    log_every = int(cfg.get("log_every", 25))
    alpha, T = float(cfg.get("kd_alpha", 0.5)), float(cfg.get("kd_T", 2.0))
    hidden_w = float(cfg.get("kd_hidden_weight", 0.0))
    losses: List[float] = []
    first_loss = None
    stop_reason = None
    tok_count, tok_t0 = 0, time.time()
    min_steps_for_watchdog = min(150, max(1, total_steps // 3))
    watchdog_done = st["global_step"] >= min_steps_for_watchdog

    model.train()
    while st["epoch"] < epochs and stop_reason is None:
        idx = order.indices(st["epoch"], ctx.rank, ctx.world, st["consumed"])
        pos = 0
        while pos < len(idx):
            if max_steps and st["global_step"] >= max_steps:
                stop_reason = "max_steps"
                break
            step_loss, micro = 0.0, 0
            for a in range(accum):
                chunk = idx[pos:pos + mb]
                if not chunk:
                    break
                pos += len(chunk)
                micro += 1
                b = collate(windows, chunk, meta["pad_id"], ctx.device, labels_override)
                sync = (a == accum - 1) or pos >= len(idx)
                cm = model.no_sync() if (ctx.world > 1 and not sync) else _null()
                try:
                    with cm:
                        with amp_ctx():
                            logits, vec = model(b["input_ids"], b["attention_mask"], b["markers"],
                                                b["starts"], b["marker_mask"])
                        valid = b["labels"] != IGNORE
                        y = b["labels"][valid]
                        s_logits = logits[valid].float()
                        if t_cache is not None:
                            t_l = torch.cat([t_cache["logits"][i] for i in chunk]).to(ctx.device)
                            t_v = (torch.cat([t_cache["vectors"][i] for i in chunk]).to(ctx.device)
                                   if hidden_w > 0 else None)
                            from rr_kd import kd_loss
                            loss = kd_loss(s_logits, t_l, y, loss_fn, alpha, T,
                                           vec[valid] if hidden_w > 0 else None, t_v, hidden_w,
                                           raw_model.kd_proj)
                        else:
                            loss = loss_fn(s_logits, y) if y.numel() else logits.sum() * 0.0
                        extra = tech.extra_loss(raw_model)
                        if extra is not None:
                            loss = loss + extra
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
                step_loss += float(loss.detach()) / accum
                tok_count += int(b["attention_mask"].sum())
            if micro == 0:
                continue
            # ---------------- step (NaN-safe; identical decision on all ranks) ----------------
            scaler.unscale_(opt)
            params = [p for p in raw_model.parameters() if p.requires_grad]
            gnorm = torch.nn.utils.clip_grad_norm_(params, clip)
            finite = bool(torch.isfinite(gnorm))
            if finite:
                tech.after_backward(raw_model, st["global_step"], total_steps)
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
            if mode == "train":
                if kill_at_step is not None and st["global_step"] == kill_at_step:
                    st["train_seconds"] += time.time() - t_start
                    save_last()
                    log(ctx, run_dir, f"   [test] simulated kill after step {kill_at_step}")
                    sys.stdout.flush()
                    os._exit(137)
                over = [1.0 if (deadline and time.time() > deadline - margin) else 0.0]
                if ctx.bcast(over)[0]:
                    st["train_seconds"] += time.time() - t_start
                    save_last()
                    log(ctx, run_dir, "   [time] job time budget reached -> checkpoint saved")
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
            f1, _ = quick_dev_f1(raw_model, meta, docs, cfg, ctx.device, amp_ctx)
            raw_model.train()
        f1 = ctx.bcast([f1])[0]
        st["max_f1_seen"] = max(st.get("max_f1_seen", -1.0), f1)
        selectable = tech.selectable(st["global_step"], total_steps)
        improved = selectable and f1 > st["best_f1"]
        if improved:
            st["best_f1"], st["best_epoch"], st["bad_epochs"] = f1, st["epoch"], 0
            save_best()
        elif selectable:
            st["bad_epochs"] += 1
        st["history"].append({"epoch": st["epoch"], "dev_macro_f1": f1, "selectable": selectable,
                              "train_loss": float(np.mean(losses[-steps_per_epoch:]))
                              if losses else None})
        log(ctx, run_dir, f"   == epoch {st['epoch']}/{epochs} | dev pooled Macro-F1 {f1:.4f} "
                          f"{'(best)' if improved else ''}"
                          f"{'' if selectable else ' (not selectable yet)'} | train loss "
                          f"{st['history'][-1]['train_loss'] or float('nan'):.4f}",
            {"epoch_end": st["epoch"], "dev_macro_f1": f1} if mode == "train" else None)
        if mode == "train":
            if not watchdog_done and st["global_step"] >= min_steps_for_watchdog:
                watchdog_done = True
                if st["max_f1_seen"] < float(cfg.get("watchdog_min_f1", 0.01)):
                    save_last()
                    log(ctx, run_dir, f"   [watchdog] dev F1 {st['max_f1_seen']:.4f} after "
                                      f"{st['global_step']} steps -> stop (check GATES.json, "
                                      f"the learning rate and logs/overfit.json)")
                    raise SystemExit(EXIT_EARLYFAIL)
            if st["bad_epochs"] >= int(cfg.get("patience", 4)):
                stop_reason = "early_stop"
            if time.time() - last_ckpt_t > 15 * 60:
                save_last()
                last_ckpt_t = time.time()
    st["train_seconds"] += time.time() - t_start
    el = time.time() - tok_t0
    summary = {"status": stop_reason or "done", "epochs_run": st["epoch"],
               "global_step": st["global_step"], "steps_per_epoch": steps_per_epoch,
               "total_steps": total_steps, "micro_batch": mb, "accum": accum, "world": ctx.world,
               "windows": n, "best_dev_macro_f1": st["best_f1"], "best_epoch": st["best_epoch"],
               "history": st["history"], "skipped_steps": st["skipped"],
               "first_loss": first_loss,
               "last_loss": float(np.mean(losses[-10:])) if losses else None,
               "tokens_per_s": tok_count * ctx.world / max(el, 1e-9),
               "seconds_per_step": el / max(1, len(losses)),
               "peak_mem_gb": (torch.cuda.max_memory_allocated(ctx.device) / 1e9
                               if ctx.device.type == "cuda" else 0.0),
               **model_size(raw_model), "train_seconds": st["train_seconds"],
               "resumed_from_step": st["resumed_from"], "lr_mult": lr_mult,
               "peft_report": tech.final_report(raw_model)}
    return summary, raw_model, tech


@torch.no_grad()
def subset_accuracy(model, windows, labels_override, meta, device, amp_ctx) -> float:
    model.eval()
    idx = sorted(labels_override)
    b = collate(windows, idx, meta["pad_id"], device, labels_override)
    with amp_ctx():
        logits, _ = model(b["input_ids"], b["attention_mask"], b["markers"], b["starts"],
                          b["marker_mask"])
    valid = b["labels"] != IGNORE
    return float((logits[valid].argmax(-1) == b["labels"][valid]).float().mean())


def overfit_subset(windows, n_sent: int = 32):
    """The first windows holding n_sent labelled sentences; other slots ignored."""
    over, left = {}, n_sent
    for i, w in enumerate(windows):
        if left <= 0:
            break
        lab = w["labels"].copy()
        keep = np.nonzero(lab != IGNORE)[0][:left]
        mask = np.full_like(lab, IGNORE)
        mask[keep] = lab[keep]
        if len(keep):
            over[i] = mask
            left -= len(keep)
    return over


# =============================================================================================
# modes
# =============================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["train", "smoke", "overfit", "probe"], default="train")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kill-at-step", type=int, default=None, help="test hook (selftest #7)")
    a = ap.parse_args()
    run_dir = a.run_dir
    cfg = read_json(os.path.join(run_dir, "logs", "run_config.json"))
    ctx = Ctx()
    code = 0
    try:
        meta, docs = load_cache(cfg["data_dir"])
        windows = train_windows(cfg, meta, docs)
        if ctx.main:
            n_lab = sum(int((w["labels"] != IGNORE).sum()) for w in windows)
            print(f"   train windows: {len(windows)} ({cfg['input_mode']}) | labelled sentence "
                  f"slots {n_lab} | max tokens {max(len(w['tokens']) for w in windows)}",
                  flush=True)
        # first-batch assertions
        b = collate(windows, list(range(min(4, len(windows)))), meta["pad_id"])
        assert int(b["input_ids"].max()) < meta["vocab_size"] and int(b["input_ids"].min()) >= 0
        y = b["labels"]
        assert ((y == IGNORE) | ((y >= 0) & (y < cfg["n_labels"]))).all(), "bad label"
        assert (y[~b["marker_mask"]] == IGNORE).all(), "labels on padded marker slots"
        mk = b["markers"][b["marker_mask"]]
        assert (b["input_ids"].gather(1, b["markers"])[b["marker_mask"]] == meta["eos_id"]).all(), \
            "a marker does not point at EOS"
        assert b["input_ids"].shape[1] + int(cfg.get("n_virtual", 0)) <= 1024 and mk.numel() > 0

        if a.mode == "smoke":
            if ctx.main and not cfg.get("skip_job_checks"):
                import selftest
                res = selftest.job_checks(cfg, meta, docs, ctx.device)
                write_json(os.path.join(run_dir, "logs", "job_checks.json"), res)
                if not res["ok"]:
                    print(f"   job self-checks FAILED: {res['failed']}", flush=True)
                    raise SystemExit(EXIT_SELFCHECK)
                print(f"   job self-checks passed: {sorted(res['checks'])}", flush=True)
            steps = a.max_steps or 30
            s, _m, _t = train_loop(cfg, ctx, run_dir, meta, docs, windows, "smoke",
                                   max_steps=steps, resume=False, eval_each_epoch=False)
            if ctx.main:
                epochs = cfg["epochs"]
                train_min = s["seconds_per_step"] * s["steps_per_epoch"] * epochs / 60
                n_dev_tok = sum(len(x) + 1 for k in eval_keys(meta, docs, "dev", cfg)
                                for x in docs[k]["ids"]) * 2
                eval_min = epochs * n_dev_tok / max(1.0, 3 * s["tokens_per_s"]) / 60
                s["projected_train_minutes"] = train_min + eval_min
                s["loss_decreased"] = (s["last_loss"] is not None and s["first_loss"] is not None
                                       and s["last_loss"] < s["first_loss"])
                write_json(os.path.join(run_dir, "logs", f"smoke{a.tag}.json"), s)
                print(f"   smoke ok: {s['tokens_per_s']:,.0f} tok/s | mem {s['peak_mem_gb']:.1f} GB"
                      f" | micro-batch {s['micro_batch']} x accum {s['accum']} x world "
                      f"{s['world']} | projected {s['projected_train_minutes']:.1f} min", flush=True)
        elif a.mode == "overfit":
            over = overfit_subset(windows, 32)
            sub = sorted(over)
            sub_w = [windows[i] for i in sub]
            sub_over = {j: over[i] for j, i in enumerate(sub)}
            c2 = dict(cfg)
            c2["dropout"] = 0.0
            c2["warmup_frac"] = 0.05
            c2["effective_batch"] = len(sub_w)
            c2["loss"] = "ce"
            c2["head_lr"] = max(float(cfg["head_lr"]), 1e-2)
            mult = 10.0 if cfg["technique"] in FULL_PARAM else 3.0
            s, model, _t = train_loop(c2, ctx, run_dir, meta, docs, sub_w, "overfit",
                                      lr_mult=mult, max_steps=a.max_steps or 300, epochs=10 ** 6,
                                      resume=False, eval_each_epoch=False,
                                      labels_override=sub_over)
            if ctx.main:
                _amp, amp_ctx = amp_setup(ctx.device)
                acc = subset_accuracy(model, sub_w, sub_over, meta, ctx.device, amp_ctx)
                s["train_accuracy"] = acc
                s["n_sentences"] = int(sum((v != IGNORE).sum() for v in sub_over.values()))
                s["passed"] = bool(acc > 0.95)
                s["fatal"] = bool(acc < 0.5)
                write_json(os.path.join(run_dir, "logs", "overfit.json"), s)
                print(f"   overfit-32: train accuracy {acc:.3f} after {s['global_step']} steps "
                      f"(loss {s['first_loss']:.3f} -> {s['last_loss']:.4f}) "
                      f"{'PASS' if s['passed'] else 'FAIL'}", flush=True)
        elif a.mode == "probe":
            results = []
            for mult in cfg.get("probe_mults", [1 / 3, 1.0, 3.0]):
                budget = a.max_steps or cfg.get("probe_steps")
                if budget is None:
                    eff = cfg.get("effective_batch", 8)
                    budget = min(200, max(30, math.ceil(len(windows) / eff)))
                s, model, _t = train_loop(cfg, ctx, run_dir, meta, docs, windows, "probe",
                                          lr_mult=mult, max_steps=budget, epochs=10 ** 6,
                                          resume=False, eval_each_epoch=False)
                f1 = -1.0
                if ctx.main:
                    _amp, amp_ctx = amp_setup(ctx.device)
                    f1, _ = quick_dev_f1(model, meta, docs, cfg, ctx.device, amp_ctx)
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
            s, _m, _t = train_loop(cfg, ctx, run_dir, meta, docs, windows, "train",
                                   kill_at_step=a.kill_at_step)
            if ctx.main:
                write_json(os.path.join(run_dir, "logs", "train_summary.json"), s)
                if s["best_epoch"] < 0:
                    raise RuntimeError("training produced no best checkpoint")
                print(f"   training {s['status']}: best dev Macro-F1 {s['best_dev_macro_f1']:.4f} "
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
