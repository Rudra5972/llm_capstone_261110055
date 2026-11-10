"""
run_experiment.py -- one notebook = one technique, trained on all four Phase-1 models.

    python run_experiment.py --exp E2                      # LoRMA on M4, M2, M3, M1
    python run_experiment.py --exp E1 --models M2,M3       # split a notebook across accounts
    python run_experiment.py --exp X3_crf --technique full_ft --head crf --models M4  # an extra

Per model: [restore earlier outputs] -> pre-flight (2-GPU smoke test with fallback to 1 GPU,
projected minutes) -> overfit-64 (first model) -> LR probe -> train -> dev report + fold3
prediction + validated submission.zip -> cleanup (last.pt deleted) -> DONE.json.
Then best-of-4 selection into BEST/ (select_best.py) and INDEX.md.

This process never touches the GPU itself: every GPU job is a subprocess, so a crash or a
CUDA error in one model cannot take down the whole notebook.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from ner_common import (DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_MODEL_ORDER, EXPERIMENTS, MODELS,
                        STUDENT_LAYERS, DiskFull, default_input_roots, dir_size_gb, disk_guard,
                        find_files, find_phase1_models, free_gb, read_json, write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "ner_train.py")
PREDICT = os.path.join(HERE, "ner_predict.py")


# =============================================================================================
# subprocess helpers
# =============================================================================================
def run_cmd(cmd, log_path, env=None, timeout=None):
    """Stream a subprocess's output to the notebook and a log file, with a hard wall-clock
    timeout (a hung DDP job prints nothing, so the timeout must not depend on output).
    Returns (exit_code, timed_out); exit_code is -9 on timeout."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = HERE + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    full_env.setdefault("NCCL_IB_DISABLE", "1")
    full_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    full_env.setdefault("OMP_NUM_THREADS", "2")
    if env:
        full_env.update(env)
    lf = open(log_path, "a", encoding="utf-8")
    lf.write(f"\n$ {' '.join(cmd)}\n")
    lf.flush()
    p = subprocess.Popen(cmd, cwd=HERE, env=full_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

    def pump():
        for line in p.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    timed_out = False
    try:
        code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)            # kills torchrun AND its workers
        except OSError:
            pass
        p.wait()
        code = -9
    th.join(timeout=10)
    lf.close()
    return code, timed_out


def gpu_count():
    try:
        out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.device_count())"],
                             capture_output=True, text=True, timeout=120)
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def train_cmd(run_dir, mode, nproc, extra=()):
    base = ["--run-dir", run_dir, "--mode", mode, *extra]
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(nproc), TRAIN, *base]
    return [sys.executable, TRAIN, *base]


# =============================================================================================
# configs
# =============================================================================================
def student_layers(mid, phase1_dir):
    """Every second layer of the Phase-1 model (= STUDENT_LAYERS for the real 12/14-layer
    models; derived from config.json so tiny test models work too)."""
    cfg = read_json(os.path.join(phase1_dir, "config.json"))["config"]
    n = cfg.get("n_layer", cfg.get("num_hidden_layers"))
    ids = list(range(0, n, 2))
    if n in (12, 14):
        assert ids == STUDENT_LAYERS[mid], (ids, STUDENT_LAYERS[mid])
    return ids


def make_run_config(a, exp, folder, mid, technique, meta, data_dir, phase1_dir, deadline,
                    teacher_dir):
    lr = a.lr or DEFAULT_LR[technique]
    full_param = technique in ("full_ft", "kd", "student_hard")
    return {
        "exp": exp, "exp_folder": folder, "model": mid, "arch": MODELS[mid]["arch"],
        "kind": MODELS[mid]["kind"], "technique": technique, "phase1_dir": phase1_dir,
        "data_dir": data_dir, "labels": meta["labels"], "n_labels": len(meta["labels"]),
        "attn_mode": a.attn_mode, "head": a.head, "chunk_len": a.chunk_len,
        "train_stride": a.train_stride, "infer_stride": a.infer_stride, "neg_keep": a.neg_keep,
        "class_weight": a.class_weight, "lr": lr, "head_lr": 10 * lr if full_param else 1e-3,
        "weight_decay": 0.01, "warmup_frac": 0.06, "grad_clip": 1.0, "dropout": 0.1,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lorma_r": 16, "lorma_alpha": 32,
        "adapter_bottleneck": 64, "partial_top_k": 6, "layer_decay": 0.85,
        "kd_alpha": 0.5, "kd_T": 2.0, "teacher_dir": teacher_dir,
        "layer_ids": student_layers(mid, phase1_dir) if technique in ("kd", "student_hard") else None,
        "epochs": a.epochs or DEFAULT_EPOCHS[technique], "patience": a.patience,
        "effective_batch": a.effective_batch, "micro_batch": a.micro_batch, "eval_batch": 32,
        "seed": 42, "ckpt_every_min": a.ckpt_every_min, "deadline_unix": deadline,
        "deadline_margin_s": 300, "refit": a.refit,
        # PEFT methods can start slower; a false early stop would waste the whole run
        "watchdog_min_f1": (a.watchdog_min_f1 if technique in ("full_ft", "kd", "student_hard",
                                                               "partial_ft")
                            else min(a.watchdog_min_f1, 0.02)),
        "log_every": a.log_every, "probe_steps": a.probe_steps, "enforce_size": not a.cpu_test,
    }


# =============================================================================================
# restore outputs of earlier sessions (attached as input datasets)
# =============================================================================================
def restore_from_inputs(exp_folder, out_root, roots, log):
    """Copy finished (DONE.json) or resumable (last.pt) model folders of this experiment from
    attached datasets into the working output dir, unless already present."""
    for done in find_files("DONE.json", roots, max_hits=1000) + \
            find_files("last.pt", roots, max_hits=1000):
        parts = done.split(os.sep)
        if "phase2_out" not in parts or exp_folder not in parts:
            continue
        i = parts.index(exp_folder)
        if i + 1 >= len(parts) - 1:
            continue
        model_folder = parts[i + 1]
        src = os.sep.join(parts[:i + 2])
        dst = os.path.join(out_root, exp_folder, model_folder)
        if model_folder == "BEST" or os.path.exists(os.path.join(dst, "DONE.json")):
            continue
        if os.path.exists(dst) and not os.path.exists(os.path.join(src, "DONE.json")):
            continue                                    # never overwrite local progress
        log(f"   restoring {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def find_teacher(out_root, roots):
    cands = [os.path.join(out_root, "E1_full_ft", "BEST")]
    for s in find_files("SELECTION.json", roots, max_hits=1000):
        if os.sep + "E1_full_ft" + os.sep + "BEST" in s:
            cands.append(os.path.dirname(s))
    for c in cands:
        if all(os.path.exists(os.path.join(c, f)) for f in ("best.pt", "run_config.json")):
            return c
    return None


# =============================================================================================
# INDEX.md
# =============================================================================================
def write_index(out_root, exp_folder, records, next_cmd, a):
    st_path = os.path.join(out_root, "index_state.json")
    state = read_json(st_path, {})
    state[exp_folder] = {"records": records, "next": next_cmd,
                         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(st_path, state)
    lines = ["# Phase-2 outputs (generated by run_experiment.py)", "",
             f"Output size: {dir_size_gb(out_root):.2f} GB | free disk: {free_gb(out_root):.1f} GB",
             ""]
    for ef in sorted(state):
        s = state[ef]
        lines += [f"## {ef}  (updated {s['updated']})", "",
                  "| model | status | dev macro-F1 | minutes | note |", "|---|---|---|---|---|"]
        for r in s["records"]:
            f1 = f"{r['dev_macro_f1']:.4f}" if isinstance(r.get("dev_macro_f1"), float) else ""
            mins = f"{r['minutes']:.1f}" if isinstance(r.get("minutes"), (int, float)) else ""
            lines.append(f"| {r['model']} | {r['status']} | {f1} | {mins} | {r.get('note', '')} |")
        sel = read_json(os.path.join(out_root, ef, "BEST", "SELECTION.json"))
        if sel:
            lines += ["", f"**Upload:** `{ef}/BEST/submission.zip` (winner {sel['winner']}, "
                          f"dev macro-F1 {sel['dev_macro_f1']:.4f})"]
        if s.get("next"):
            lines += ["", f"**Next:** `{s['next']}`"]
        lines.append("")
    lines += ["Save this session's output (Save Version, or Output -> New Dataset) before it "
              "ends; attach it to the next session so finished models are skipped and "
              "unfinished ones resume.", ""]
    with open(os.path.join(out_root, "INDEX.md"), "w") as f:
        f.write("\n".join(lines))


# =============================================================================================
# main loop
# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="E1..E6, X1, X2, or a custom name with --technique")
    ap.add_argument("--technique", default=None)
    ap.add_argument("--models", default=None, help="comma list, default M4,M2,M3,M1")
    ap.add_argument("--nproc", default="auto", choices=["auto", "1", "2"])
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--time-budget-h", type=float, default=11.0)
    ap.add_argument("--out", default="/kaggle/working/phase2_out")
    ap.add_argument("--data-dir", default="/kaggle/working/phase2_data")
    ap.add_argument("--attn-mode", default="bidir", choices=["bidir", "causal"])
    ap.add_argument("--head", default="linear", choices=["linear", "crf"])
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--train-stride", type=int, default=384)
    ap.add_argument("--infer-stride", type=int, default=256)
    ap.add_argument("--neg-keep", type=float, default=0.3)
    ap.add_argument("--class-weight", default="none", choices=["none", "sqrt_inv"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--ckpt-every-min", type=float, default=30)
    ap.add_argument("--watchdog-min-f1", type=float, default=0.05)
    ap.add_argument("--probe-steps", type=int, default=None)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--refit", action="store_true", help="train on train+dev (extra X6)")
    ap.add_argument("--teacher-dir", default=None)
    ap.add_argument("--smoke-timeout-s", type=int, default=420)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--cpu-test", action="store_true", help="tiny-model CI mode")
    a = ap.parse_args()

    t_start = time.time()
    deadline = t_start + a.time_budget_h * 3600
    exp = a.exp
    spec = EXPERIMENTS.get(exp, {})
    technique = a.technique or spec.get("technique")
    if technique is None:
        sys.exit(f"unknown experiment {exp!r}; give --technique for a custom extra")
    folder = spec.get("folder") if not a.technique else exp
    folder = folder or exp
    models = [m.strip() for m in (a.models or ",".join(spec.get("models", DEFAULT_MODEL_ORDER))).split(",")]
    out_root = os.path.abspath(a.out)
    exp_dir = os.path.join(out_root, folder)
    os.makedirs(exp_dir, exist_ok=True)
    console = os.path.join(exp_dir, "console.log")
    roots = default_input_roots()

    def say(msg):
        print(msg, flush=True)
        with open(console, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    say(f"== {folder}: technique {technique}, models {models}, budget {a.time_budget_h} h")

    # ---- data (CPU) ---------------------------------------------------------------------------
    from ner_data import GateFailure, find_or_build_cache, load_cache
    try:
        data_dir = find_or_build_cache(os.path.abspath(a.data_dir), roots, log=say)
    except GateFailure as e:
        say(f"STOP AND ASK (data gate): {e}")
        sys.exit(2)
    meta, _docs = load_cache(data_dir)
    del _docs

    p1 = find_phase1_models(roots)
    say(f"   Phase-1 weights found for: {sorted(p1)}")
    teacher_dir = None
    if technique == "kd":
        teacher_dir = a.teacher_dir or find_teacher(out_root, roots)
        if not teacher_dir:
            say("E6 needs E1_full_ft/BEST (best.pt + run_config.json). Run E1 first, or attach "
                "its output dataset to this notebook.")
            write_index(out_root, folder, [], f"run E1 first, then: python run_experiment.py --exp {exp}", a)
            sys.exit(2)
        say(f"   KD teacher: {teacher_dir}")
    if a.resume == "auto":
        restore_from_inputs(folder, out_root, roots, say)

    n_gpu = 0 if a.cpu_test else gpu_count()
    say(f"   GPUs visible: {n_gpu}")
    records, next_cmd = [], None
    first_model = True
    stop_all = False
    for mid in models:
        rec = {"model": mid, "status": "pending"}
        records.append(rec)
        run_dir = os.path.join(exp_dir, MODELS[mid]["folder"])
        logs = os.path.join(run_dir, "logs")
        os.makedirs(logs, exist_ok=True)
        cons = os.path.join(logs, "console.log")
        t_model = time.time()
        done = read_json(os.path.join(run_dir, "DONE.json"))
        if done:
            rec.update(status="done (skipped, already finished)", dev_macro_f1=done["dev_macro_f1"])
            say(f"-- {mid}: already finished (dev macro-F1 {done['dev_macro_f1']:.4f}), skipping")
            continue
        if stop_all:
            rec.update(status="not started")
            continue
        if mid not in p1:
            rec.update(status="FAILED", note="Phase-1 weights not attached")
            say(f"-- {mid}: Phase-1 weights for {MODELS[mid]['arch']} not found, skipping")
            continue
        try:
            disk_guard(out_root, a.min_free_gb, say, keep=os.path.join(run_dir, "checkpoints"))
        except DiskFull as e:
            rec.update(status="STOPPED", note=str(e))
            say(f"STOP: {e}")
            stop_all = True
            continue
        if deadline - time.time() < 20 * 60:
            rec.update(status="not started", note="session time budget exhausted")
            stop_all = True
            continue
        say(f"\n-- {folder} / {mid} ({MODELS[mid]['arch']}) --")

        cfg_path = os.path.join(logs, "run_config.json")
        old = read_json(cfg_path)
        resuming = os.path.exists(os.path.join(run_dir, "checkpoints", "last.pt")) and old
        cfg = make_run_config(a, exp, folder, mid, technique, meta, data_dir, p1[mid], deadline,
                              teacher_dir)
        if resuming:                                     # keep probed lr etc., refresh paths
            for k in ("lr", "head_lr", "lr_probe"):
                if k in old:
                    cfg[k] = old[k]
        write_json(cfg_path, cfg)

        # ---- pre-flight: nproc decision + projection ------------------------------------------
        dec = read_json(os.path.join(logs, "nproc_decision.json"))
        nproc, env = 1, {}
        if a.nproc != "auto":
            nproc = int(a.nproc)
        if dec and a.nproc == "auto":
            nproc, env = dec["nproc"], dec.get("env", {})
        smoke = None
        if dec is None or not os.path.exists(os.path.join(logs, "smoke.json")):
            tries = []
            if a.nproc == "auto" and n_gpu >= 2:
                tries = [(2, {}), (2, {"NCCL_P2P_DISABLE": "1"}), (1, {})]
            else:
                tries = [(nproc, {})]
            for npr, ev in tries:
                code, to = run_cmd(train_cmd(run_dir, "smoke", npr), cons, ev, a.smoke_timeout_s)
                if code == 0 and os.path.exists(os.path.join(logs, "smoke.json")):
                    nproc, env = npr, ev
                    break
                say(f"   smoke test with {npr} process(es) {ev or ''} failed "
                    f"({'timeout' if to else f'exit {code}'})")
            else:
                rec.update(status="FAILED", note="pre-flight smoke test failed (see console.log)")
                first_model = False
                continue
            write_json(os.path.join(logs, "nproc_decision.json"),
                       {"nproc": nproc, "env": env, "gpus": n_gpu,
                        "reason": "first configuration whose smoke test passed"})
        smoke = read_json(os.path.join(logs, "smoke.json"), {})
        proj = float(smoke.get("projected_train_minutes", 0) or 0)
        probe_min = 0 if (a.skip_probe or os.path.exists(os.path.join(logs, "lr_probe.json"))) \
            else proj * 0.5
        need = (proj + probe_min) * 1.25 + 5
        left = (deadline - time.time()) / 60
        say(f"   pre-flight: {nproc} process(es) {env or ''} | projected {proj:.1f} min train "
            f"+ {probe_min:.1f} min probe | {left:.0f} min left in budget")
        if need > left and not resuming:
            rec.update(status="not started", note=f"needs ~{need:.0f} min, {left:.0f} left")
            next_cmd = (f"python run_experiment.py --exp {exp} --models "
                        f"{','.join(models[models.index(mid):])}")
            stop_all = True
            continue

        # ---- overfit-64 (first model of the notebook) ----------------------------------------
        if first_model and not a.skip_overfit:
            ov = read_json(os.path.join(logs, "overfit.json"))
            if ov is None:
                run_cmd(train_cmd(run_dir, "overfit", 1), cons)
                ov = read_json(os.path.join(logs, "overfit.json"))
            if ov is None or not ov.get("passed"):
                fatal = ov is None or ov.get("fatal")
                say(f"   overfit-64 {'FAILED' if ov else 'crashed'}"
                    f"{' -> skipping this model' if fatal else ' (partial progress: warning only)'}")
                if fatal:
                    rec.update(status="FAILED", note="overfit-64 failed: check G1 / labels")
                    first_model = False
                    continue
        first_model = False

        # ---- LR probe -----------------------------------------------------------------------------
        if not a.skip_probe and not resuming:
            pr = read_json(os.path.join(logs, "lr_probe.json"))
            if pr is None:
                code, _ = run_cmd(train_cmd(run_dir, "probe", nproc), cons, env)
                pr = read_json(os.path.join(logs, "lr_probe.json"))
                if pr is None and nproc > 1:
                    code, _ = run_cmd(train_cmd(run_dir, "probe", 1), cons)
                    pr = read_json(os.path.join(logs, "lr_probe.json"))
            if pr:
                cfg["lr"] = pr["best_lr"]
                if technique in ("full_ft", "kd", "student_hard"):
                    cfg["head_lr"] = 10 * cfg["lr"]
                cfg["lr_probe"] = pr
                write_json(cfg_path, cfg)
                say(f"   LR probe -> lr {cfg['lr']:.2e}")
            else:
                say("   LR probe failed; keeping the default learning rate")

        # ---- train --------------------------------------------------------------------------------
        code, _ = run_cmd(train_cmd(run_dir, "train", nproc), cons, env)
        if code not in (0, 2, 4, 5) and nproc > 1:
            say(f"   training exited with {code} under DDP -> retrying on 1 GPU (resumes)")
            write_json(os.path.join(logs, "nproc_decision.json"),
                       {"nproc": 1, "env": {}, "gpus": n_gpu,
                        "reason": f"DDP training failed with exit {code}"})
            code, _ = run_cmd(train_cmd(run_dir, "train", 1), cons)
        if code == 2:
            rec.update(status="PAUSED", note="time budget; last.pt saved")
            next_cmd = (f"python run_experiment.py --exp {exp} --models "
                        f"{','.join(models[models.index(mid):])}   (attach this output first)")
            stop_all = True
            continue
        if code != 0:
            why = {4: "non-finite gradients (try a lower --lr)",
                   5: "early-fail watchdog (dev F1 ~ 0)",
                   3: "out of memory"}.get(code, f"exit {code}")
            rec.update(status="FAILED", note=f"training: {why}")
            continue

        # ---- predict + submission ----------------------------------------------------------------
        code, _ = run_cmd([sys.executable, PREDICT, "--run-dir", run_dir], cons, env)
        done = read_json(os.path.join(run_dir, "DONE.json"))
        if code != 0 or not done:
            rec.update(status="FAILED", note="prediction/submission step failed (best.pt kept)")
            continue
        for f in ("last.pt", "last.pt.tmp"):
            p = os.path.join(run_dir, "checkpoints", f)
            if os.path.exists(p):
                os.remove(p)
        mins = (time.time() - t_model) / 60
        rec.update(status="done", dev_macro_f1=done["dev_macro_f1"], minutes=mins)
        say(f"   {mid} finished in {mins:.1f} min | dev strict macro-F1 {done['dev_macro_f1']:.4f}"
            f" | output so far {dir_size_gb(out_root):.2f} GB")
        write_index(out_root, folder, records, next_cmd, a)

    # ---- best-of-4 ------------------------------------------------------------------------------
    from select_best import select
    sel = select(exp_dir, log=say)
    write_index(out_root, folder, records, next_cmd, a)
    say(f"\n== {folder} finished this session in {(time.time() - t_start) / 60:.1f} min")
    if sel:
        say(f"   UPLOAD: {os.path.join(exp_dir, 'BEST', 'submission.zip')} "
            f"(winner {sel['winner']}, dev macro-F1 {sel['dev_macro_f1']:.4f})")
    if next_cmd:
        say(f"   NEXT: {next_cmd}")
    say("   Remember to save this session's output (Save Version / Output -> New Dataset).")


if __name__ == "__main__":
    main()
