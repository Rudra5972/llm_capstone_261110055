"""
run_experiment.py -- one notebook = one technique x all four Phase-1 models, as a per-GPU job
queue (two models train in parallel on a T4 x2, one subprocess per job and GPU).

    python run_experiment.py --exp E1                     # LoRMA+ on M4, M2, M3, M1
    python run_experiment.py --exp E6                     # teacher_M* (full FT) -> students
    python run_experiment.py --exp E6 --teacher-dir /kaggle/input/.../phase3_out/E0_full_ft
    python run_experiment.py --exp E2 --models M4,M2      # split a notebook across accounts
    python run_experiment.py --exp P0                     # causal-vs-bidir pilot -> PILOT.json
    python run_experiment.py --exp E10 --models M4 --ablation-techniques lorma,lora
    python run_experiment.py --exp E11                    # ensemble + final report (CPU)

Flow: data cache + gates G0-G2 (exit 3 on failure) -> CPU selftest (once per notebook) ->
attention mode (PILOT.json if attached, else bidir) -> GPU sanity check on every GPU (a GPU
that fails is dropped; fallback to one GPU) -> job queue. Per job: [restore earlier outputs]
-> smoke (real-model self-checks + throughput -> per-job time budget) -> overfit-32 (first job
of the notebook) -> LR probe x{1/3,1,3} -> train (30-min checkpoints, resume) -> predict (dev
report, bias decision, validated submission.zip, raw logits) -> DONE.json. After every job and
at the end: best-of-4 selection into BEST/ (valid even if the 11 h budget is hit), RESULTS.json,
EFFICIENCY.json, INDEX.md.

Exit codes: 0 ok | 2 missing dependency (Phase-1 weights, data, KD teacher) | 3 gate or
selftest failure | 4 submission validation failure | 1 other error / nothing finished.
This process never touches the GPU itself: every GPU job is a subprocess, so a crash or a CUDA
error in one job cannot take down the notebook.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from rr_common import (DATA_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_MODEL_ORDER, EXIT,
                       EXPERIMENTS, MODELS, OUT_MARKER, OUT_ROOT, PILOT_FOLDER, SPEC,
                       DiskFull, GateFailure, default_input_roots, dir_size_gb, disk_guard,
                       find_files, find_phase1_models, free_gb, module_md5s, read_json,
                       student_layers, write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "rr_train.py")
PREDICT = os.path.join(HERE, "rr_predict.py")
SELFTEST = os.path.join(HERE, "selftest.py")

# E10: variants applied to each ablation technique (doc + linear + weighted_ce = baseline)
ABLATIONS: Dict[str, Dict] = {
    "base": {},
    "sent": {"input_mode": "sent"},
    "ctx1": {"input_mode": "ctx_1"},
    "seqctx": {"head": "seqctx"},
    "ce": {"loss": "ce"},
    "focal": {"loss": "focal"},
    "logitadj": {"loss": "logit_adjust"},
    "onlyCL": {"domains": ["CL"]},
    "onlyIT": {"domains": ["IT"]},
}
EFF_BATCH = {"doc": 8, "sent": 64, "ctx": 32}          # windows per optimizer step
MICRO_BATCH = {"doc": 4, "sent": 32, "ctx": 16}         # start value of the memory probe


# =============================================================================================
# subprocess helpers
# =============================================================================================
_print_lock = threading.Lock()


def run_cmd(cmd: List[str], log_path: str, env: Optional[Dict] = None,
            timeout: Optional[float] = None, prefix: str = "") -> Tuple[int, bool]:
    """Stream a subprocess's output (prefixed) to the notebook and a log file, with a hard
    wall-clock timeout. Returns (exit_code, timed_out); exit_code is -9 on timeout."""
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
            with _print_lock:
                print(prefix + line, end="", flush=True)
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
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.wait()
        code = -9
    th.join(timeout=10)
    lf.close()
    return code, timed_out


def env_info() -> Dict:
    code = ("import json, torch, sys; print(json.dumps({'python': sys.version.split()[0], "
            "'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpus': "
            "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=180)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def gpu_sanity(gpu: str, timeout: int = 180) -> bool:
    """A small CUDA matmul + backward on one GPU (the 'smoke test on both GPUs')."""
    code = ("import torch; x = torch.randn(2048, 2048, device='cuda', requires_grad=True); "
            "y = (x.half() @ x.half()).float().sum(); y.backward(); "
            "torch.cuda.synchronize(); print('ok', torch.cuda.get_device_name(0))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=timeout, env=env)
        return out.returncode == 0 and "ok" in out.stdout
    except Exception:
        return False


def train_cmd(run_dir: str, mode: str, nproc: int = 1, extra=()) -> List[str]:
    base = ["--run-dir", run_dir, "--mode", mode, *extra]
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(nproc), TRAIN, *base]
    return [sys.executable, TRAIN, *base]


# =============================================================================================
# configs and jobs
# =============================================================================================
def make_run_config(a, exp: str, folder: str, job: Dict, meta: Dict, data_dir: str,
                    phase1_dir: str, attn: str, env: Dict) -> Dict:
    technique, mid = job["technique"], job["model"]
    full = technique in ("full_ft", "kd")
    lr = (a.lr if (a.lr and not job.get("is_teacher")) else None) or DEFAULT_LR[technique]
    cfg = {
        "exp": exp, "exp_folder": folder, "job": job["folder"], "model": mid,
        "arch": MODELS[mid]["arch"], "kind": MODELS[mid]["kind"], "technique": technique,
        "phase1_dir": phase1_dir, "data_dir": data_dir, "labels": meta["labels"],
        "n_labels": len(meta["labels"]), "attn_mode": attn, "head": a.head, "pool": a.pool,
        "input_mode": a.input_mode, "max_len": SPEC["context"],
        "max_sent_tokens": a.max_sent_tokens, "train_stride_frac": 0.75,
        "infer_stride_frac": 0.5, "window_combine": a.window_combine, "domain_prefix": True,
        "domains": list(SPEC["domains"]), "loss": a.loss, "focal_gamma": 2.0,
        "logit_adjust_tau": 1.0, "lr": lr, "head_lr": 10 * lr if full else max(1e-3, lr),
        "weight_decay": 0.01, "warmup_frac": 0.06, "schedule": "cosine", "grad_clip": 1.0,
        "dropout": 0.1,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lorma_r": 16, "lorma_alpha": 32,
        "rslora_r": 32, "rslora_alpha": 16,
        "adalora_init_r": 12, "adalora_target_r": 8, "adalora_alpha": 32,
        "adalora_tinit_frac": 0.1, "adalora_tfinal_frac": 0.2, "adalora_delta_t": 10,
        "adalora_beta1": 0.85, "adalora_beta2": 0.85, "adalora_orth": 0.1,
        "qlora_r": 16, "qlora_alpha": 32, "qlora_block": 64, "qlora_double_quant": True,
        "prefix_len": 20, "prefix_reparam": True, "prefix_hidden": 256, "prompt_len": 20,
        "kd_alpha": 0.5, "kd_T": 2.0, "kd_hidden_weight": a.kd_hidden_weight,
        "teacher_dir": job.get("teacher_dir"),
        "layer_ids": student_layers(mid, phase1_dir) if technique == "kd" else None,
        "epochs": (a.epochs if (a.epochs and not job.get("is_teacher")) else None)
        or DEFAULT_EPOCHS[technique],
        "patience": a.patience, "micro_batch": a.micro_batch, "eval_batch": 16, "seed": 42,
        "ckpt_every_min": a.ckpt_every_min, "deadline_unix": 0, "deadline_margin_s": 300,
        "watchdog_min_f1": a.watchdog_min_f1, "log_every": 25, "probe_steps": a.probe_steps,
        "bias_tuning": True, "variant": job.get("variant"), "refit": False,
        "skip_job_checks": a.skip_job_checks,
        "module_md5": module_md5s(HERE), "environment": env,
    }
    over = job.get("overrides") or {}
    cfg.update(over)
    mode = cfg["input_mode"].split("_")[0]
    if "effective_batch" not in over:
        cfg["effective_batch"] = a.effective_batch or EFF_BATCH[mode]
    if "micro_batch" not in over:
        cfg["micro_batch"] = a.micro_batch or MICRO_BATCH[mode]
    cfg["n_virtual"] = int(cfg["prompt_len"]) if technique == "prompt" else 0
    return cfg


def make_jobs(a, exp: str, technique: str, models: List[str], exp_dir: str) -> List[Dict]:
    def job(mid, folder, tech, **kw):
        return {"model": mid, "folder": folder, "technique": tech, "kind": "train",
                "after": [], "overrides": {}, **kw}
    jobs: List[Dict] = []
    if exp == "P0":
        for mode in ("causal", "bidir"):
            jobs.append(job("M2", f"lora_{mode}_M2", "lora", overrides={"attn_mode": mode,
                                                                        "epochs": 1},
                            skip_probe=True, skip_overfit=True, variant=f"pilot_{mode}"))
    elif technique == "kd":
        teachers, students = [], []
        for mid in models:
            sfolder = MODELS[mid]["folder"]
            if a.teacher_dir:
                students.append(job(mid, sfolder, "kd", teacher_dir=a.teacher_dir))
            else:
                tfolder = "teacher_" + sfolder
                teachers.append(job(mid, tfolder, "full_ft", is_teacher=True,
                                    variant="kd_teacher"))
                students.append(job(mid, sfolder, "kd", after=[tfolder],
                                    teacher_dir=os.path.join(exp_dir, tfolder)))
        if teachers:                             # interleave: 2 teachers, their students, ...
            for i in range(0, len(models), 2):
                jobs += teachers[i:i + 2] + students[i:i + 2]
        else:
            jobs = students
    elif technique == "ablation":
        for tech in [t.strip() for t in a.ablation_techniques.split(",") if t.strip()]:
            for mid in models:
                for vname, ov in ABLATIONS.items():
                    jobs.append(job(mid, f"{tech}_{vname}_{mid}", tech, overrides=dict(ov),
                                    variant=vname, skip_probe=True, skip_overfit=True))
                jobs.append({"model": mid, "folder": f"{tech}_perdomain_{mid}", "technique": tech,
                             "kind": "combine", "variant": "per_domain",
                             "after": [f"{tech}_onlyCL_{mid}", f"{tech}_onlyIT_{mid}"]})
    else:
        for mid in models:
            jobs.append(job(mid, MODELS[mid]["folder"], technique))
    return jobs


# =============================================================================================
# restore / pilot / index
# =============================================================================================
def restore_from_inputs(exp_folder: str, out_root: str, roots: List[str], log) -> None:
    """Copy finished (DONE.json) or resumable (last.pt) job folders of this experiment from
    attached datasets into the working output dir, unless already present."""
    for hit in find_files("DONE.json", roots, max_hits=5000) + \
            find_files("last.pt", roots, max_hits=5000):
        parts = hit.split(os.sep)
        if OUT_MARKER not in parts or exp_folder not in parts:
            continue
        i = parts.index(exp_folder)
        if i + 1 >= len(parts) - 1:
            continue
        job_folder = parts[i + 1]
        src = os.sep.join(parts[:i + 2])
        dst = os.path.join(out_root, exp_folder, job_folder)
        if job_folder == "BEST" or os.path.exists(os.path.join(dst, "DONE.json")):
            continue
        if os.path.exists(dst) and not os.path.exists(os.path.join(src, "DONE.json")):
            continue                                    # never overwrite local progress
        log(f"   restoring {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def resolve_attn_mode(a, out_root: str, roots: List[str], log) -> Tuple[str, str]:
    if a.attn_mode in ("causal", "bidir"):
        return a.attn_mode, "command line"
    cands = [os.path.join(out_root, "PILOT.json")] + find_files("PILOT.json", roots)
    for c in cands:
        p = read_json(c)
        if p and p.get("winner") in ("causal", "bidir"):
            return p["winner"], f"pilot {c}"
    log("   !!! no PILOT.json attached: using attention mode 'bidir' (Phase-2 default). Run "
        "the 00 notebook (or --exp P0) and attach its output to use the pilot's winner.")
    return "bidir", "default (no pilot)"


def write_pilot(out_root: str, exp_dir: str, log) -> Optional[Dict]:
    res = {}
    for mode in ("causal", "bidir"):
        d = read_json(os.path.join(exp_dir, f"lora_{mode}_M2", "DONE.json"))
        if d:
            res[mode] = {"dev_macro_f1": d["dev_macro_f1"],
                         "dev_per_domain": d.get("dev_per_domain_macro_f1"),
                         "train_minutes": (d.get("efficiency") or {}).get("train_minutes")}
    if len(res) < 2:
        log("   pilot incomplete -- PILOT.json not written")
        return None
    winner = max(res, key=lambda m: (res[m]["dev_macro_f1"], m == "bidir"))
    pilot = {"winner": winner, "results": res, "setup": "M2 + LoRA, 1 epoch each",
             "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(os.path.join(out_root, "PILOT.json"), pilot)
    write_json(os.path.join(exp_dir, "PILOT.json"), pilot)
    log(f"   PILOT: causal {res['causal']['dev_macro_f1']:.4f} vs bidir "
        f"{res['bidir']['dev_macro_f1']:.4f} -> attention mode '{winner}' (PILOT.json)")
    return pilot


def write_index(out_root: str, exp_folder: str, records: List[Dict], note: str) -> None:
    st_path = os.path.join(out_root, "index_state.json")
    state = read_json(st_path, {})
    state[exp_folder] = {"records": records, "note": note,
                         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(st_path, state)
    lines = ["# Phase-3 outputs (generated by run_experiment.py)", "",
             f"Output size: {dir_size_gb(out_root):.2f} GB | free disk: {free_gb(out_root):.1f} GB",
             ""]
    for ef in sorted(state):
        s = state[ef]
        lines += [f"## {ef}  (updated {s['updated']})", "",
                  "| job | gpu | status | dev Macro-F1 | CL | IT | minutes | note |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in s["records"]:
            f = lambda k: f"{r[k]:.4f}" if isinstance(r.get(k), float) else ""  # noqa: E731
            mins = f"{r['minutes']:.1f}" if isinstance(r.get("minutes"), (int, float)) else ""
            lines.append(f"| {r['job']} | {r.get('gpu', '')} | {r['status']} | "
                         f"{f('dev_macro_f1')} | {f('cl')} | {f('it')} | {mins} | "
                         f"{r.get('note', '')} |")
        sel = read_json(os.path.join(out_root, ef, "BEST", "SELECTION.json"))
        if sel:
            lines += ["", f"**Upload:** `{ef}/BEST/submission.zip` (winner {sel['winner']}, "
                          f"dev Macro-F1 {sel['dev_macro_f1']:.4f})"]
        if s.get("note"):
            lines += ["", f"**Note:** {s['note']}"]
        lines.append("")
    lines += ["Save this session's output (Save Version) before it ends; attach it to the next "
              "session so finished jobs are skipped and unfinished ones resume.", ""]
    with open(os.path.join(out_root, "INDEX.md"), "w") as f:
        f.write("\n".join(lines))


def results_tables(exp_dir: str) -> None:
    """RESULTS.json (one row per finished job) and EFFICIENCY.json."""
    rows, eff = [], []
    for job in sorted(os.listdir(exp_dir)):
        d = read_json(os.path.join(exp_dir, job, "DONE.json"))
        if job == "BEST" or not d:
            continue
        e = d.get("efficiency") or {}
        rows.append({"job": job, "model": d.get("model"), "technique": d.get("technique"),
                     "variant": d.get("variant"), "dev_macro_f1": d["dev_macro_f1"],
                     "dev_macro_precision": d.get("dev_macro_precision"),
                     "dev_macro_recall": d.get("dev_macro_recall"),
                     "dev_per_domain_macro_f1": d.get("dev_per_domain_macro_f1"),
                     "dev_domain_avg_macro_f1": d.get("dev_domain_avg_macro_f1"),
                     "bias_applied": d.get("bias_applied"),
                     "submission_valid": d.get("submission_valid"),
                     "kd_report": d.get("kd_report"), "peft_report": d.get("peft_report"),
                     **{k: e.get(k) for k in ("total_params", "trainable_params",
                                              "trainable_pct", "train_minutes")}})
        eff.append({"job": job, "model": d.get("model"), "technique": d.get("technique"), **e})
    write_json(os.path.join(exp_dir, "RESULTS.json"), rows)
    write_json(os.path.join(exp_dir, "EFFICIENCY.json"), eff)


# =============================================================================================
# the job runner
# =============================================================================================
class Runner:
    def __init__(self, a, exp, folder, exp_dir, out_root, jobs, meta, data_dir, p1, attn, env,
                 deadline, say):
        self.a, self.exp, self.folder, self.exp_dir, self.out_root = a, exp, folder, exp_dir, out_root
        self.jobs, self.meta, self.data_dir, self.p1, self.attn = jobs, meta, data_dir, p1, attn
        self.env, self.deadline, self.say = env, deadline, say
        self.lock = threading.Lock()
        self.status: Dict[str, str] = {}          # folder -> pending|running|done|failed|skipped
        self.records: Dict[str, Dict] = {}
        self.overfit_claimed = a.skip_overfit
        self.validation_failed = False
        self.missing_dependency = False
        for j in jobs:
            self.status[j["folder"]] = "pending"
            self.records[j["folder"]] = {"job": j["folder"], "status": "pending"}

    # ---------------------------------------------------------------- queue
    def _next(self) -> Optional[Dict]:
        """Next runnable job (deps done), None if nothing left, 'wait' if blocked."""
        with self.lock:
            waiting = False
            for j in self.jobs:
                if self.status[j["folder"]] != "pending":
                    continue
                deps = [self.status.get(d, "done" if self._done(d) else "missing")
                        for d in j["after"]]
                if any(s in ("failed", "skipped", "missing") for s in deps):
                    self.status[j["folder"]] = "failed"
                    self.records[j["folder"]].update(status="FAILED", note="dependency failed")
                    continue
                if all(s == "done" for s in deps):
                    self.status[j["folder"]] = "running"
                    return j
                waiting = True
            return "wait" if waiting else None

    def _done(self, folder: str) -> bool:
        return os.path.exists(os.path.join(self.exp_dir, folder, "DONE.json"))

    def worker(self, gpu: str) -> None:
        while True:
            j = self._next()
            if j is None:
                return
            if j == "wait":
                time.sleep(5)
                continue
            ok = False
            try:
                ok = self.run_job(j, gpu)
            except DiskFull as e:
                self.records[j["folder"]].update(status="STOPPED", note=str(e))
            except Exception as e:                     # never kill the other worker
                self.records[j["folder"]].update(status="FAILED", note=f"orchestrator: {e}")
            with self.lock:
                self.status[j["folder"]] = "done" if ok else (
                    "skipped" if self.records[j["folder"]]["status"] in ("not started",
                                                                         "PAUSED") else "failed")
                self.select_and_index()

    def select_and_index(self) -> None:
        from rr_select import select
        try:
            if self.exp != "P0":                       # the pilot is not an upload
                select(self.exp_dir, log=lambda m: self.say(m))
            results_tables(self.exp_dir)
        except Exception as e:
            self.say(f"   selection failed: {e}")
        write_index(self.out_root, self.folder, list(self.records.values()), "")

    # ---------------------------------------------------------------- one job
    def run_job(self, j: Dict, gpu: str) -> bool:
        a, folder = self.a, j["folder"]
        rec = self.records[folder]
        rec["gpu"] = gpu
        run_dir = os.path.join(self.exp_dir, folder)
        logs = os.path.join(run_dir, "logs")
        os.makedirs(logs, exist_ok=True)
        cons = os.path.join(logs, "console.log")
        pre = f"[{folder}@{gpu if gpu.startswith('cpu') else 'gpu' + gpu}] "
        t0 = time.time()
        done = read_json(os.path.join(run_dir, "DONE.json"))
        if done:
            rec.update(status="done (already finished)", dev_macro_f1=done["dev_macro_f1"],
                       cl=(done.get("dev_per_domain_macro_f1") or {}).get("CL"),
                       it=(done.get("dev_per_domain_macro_f1") or {}).get("IT"))
            self.say(f"{pre}already finished (dev Macro-F1 {done['dev_macro_f1']:.4f}), skipping")
            return True
        if j["kind"] == "combine":
            from rr_ensemble import combine_domains
            srcs = [os.path.join(self.exp_dir, d) for d in j["after"]]
            done = combine_domains(srcs, run_dir, self.data_dir, j, log=lambda m: self.say(pre + m))
            rec.update(status="done", dev_macro_f1=done["dev_macro_f1"],
                       minutes=(time.time() - t0) / 60)
            return True
        if j["model"] not in self.p1:
            rec.update(status="FAILED", note="Phase-1 weights not attached")
            self.missing_dependency = True
            return False
        if j["technique"] == "kd":
            from rr_kd import resolve_teacher
            if not resolve_teacher(j["teacher_dir"], j["model"]):
                rec.update(status="FAILED", note=f"KD teacher missing in {j['teacher_dir']}")
                self.say(f"{pre}KD teacher for {j['model']} not found in {j['teacher_dir']}")
                self.missing_dependency = True
                return False
        disk_guard(self.out_root, a.min_free_gb, self.say, keep=os.path.join(run_dir, "checkpoints"))
        if self.deadline - time.time() < 15 * 60:
            rec.update(status="not started", note="notebook time budget exhausted")
            return False
        self.say(f"\n{pre}-- {self.folder} / {folder} ({j['technique']}, "
                 f"{MODELS[j['model']]['arch']}) --")
        env = {"CUDA_VISIBLE_DEVICES": "" if gpu.startswith("cpu") else gpu}
        cfg_path = os.path.join(logs, "run_config.json")
        old = read_json(cfg_path)
        resuming = bool(old) and os.path.exists(os.path.join(run_dir, "checkpoints", "last.pt"))
        cfg = make_run_config(a, self.exp, self.folder, j, self.meta, self.data_dir,
                              self.p1[j["model"]], self.attn, self.env)
        if old:                                           # keep the probed lr
            for k in ("lr", "head_lr", "lr_probe"):
                if k in old:
                    cfg[k] = old[k]
        write_json(cfg_path, cfg)

        # ---- smoke: real-model self-checks + throughput ----------------------------------------
        smoke = read_json(os.path.join(logs, "smoke.json"))
        if smoke is None:
            code, to = run_cmd(train_cmd(run_dir, "smoke"), cons, env, a.smoke_timeout_s, pre)
            smoke = read_json(os.path.join(logs, "smoke.json"))
            if code != 0 or smoke is None:
                why = "self-checks failed" if code == 6 else ("timeout" if to else f"exit {code}")
                rec.update(status="FAILED", note=f"smoke: {why} (see logs/console.log)")
                if code == 6:
                    self.validation_failed = True
                return False
        proj = float(smoke.get("projected_train_minutes", 0) or 0)
        skip_probe = a.skip_probe or j.get("skip_probe") or resuming
        probe_min = 0.0 if (skip_probe or os.path.exists(os.path.join(logs, "lr_probe.json"))) \
            else proj * 0.45
        need = (proj + probe_min) * 1.3 + 5
        left = (self.deadline - time.time()) / 60
        self.say(f"{pre}projected {proj:.1f} min train + {probe_min:.1f} min probe | {left:.0f} "
                 f"min left in the notebook budget")
        if need > left and not resuming:
            rec.update(status="not started", note=f"needs ~{need:.0f} min, {left:.0f} left")
            return False

        # ---- overfit-32 (first job of the notebook) ----------------------------------------------
        with self.lock:
            do_overfit = not self.overfit_claimed and not j.get("skip_overfit")
            if do_overfit:
                self.overfit_claimed = True
        if do_overfit and read_json(os.path.join(logs, "overfit.json")) is None:
            run_cmd(train_cmd(run_dir, "overfit"), cons, env, 1800, pre)
            ov = read_json(os.path.join(logs, "overfit.json"))
            if ov is None or not ov.get("passed"):
                fatal = ov is None or ov.get("fatal")
                self.say(f"{pre}overfit-32 {'FAILED' if ov else 'crashed'}"
                         f"{' -> skipping this job' if fatal else ' (warning only)'}")
                if fatal:
                    rec.update(status="FAILED", note="overfit-32 failed: check GATES.json")
                    return False

        # ---- LR probe ----------------------------------------------------------------------------
        if not skip_probe:
            pr = read_json(os.path.join(logs, "lr_probe.json"))
            if pr is None:
                run_cmd(train_cmd(run_dir, "probe"), cons, env, max(600, proj * 60 * 1.5), pre)
                pr = read_json(os.path.join(logs, "lr_probe.json"))
            if pr:
                cfg["lr"] = pr["best_lr"]
                if j["technique"] in ("full_ft", "kd"):
                    cfg["head_lr"] = 10 * cfg["lr"]
                cfg["lr_probe"] = pr
                self.say(f"{pre}LR probe -> lr {cfg['lr']:.2e}")
            else:
                self.say(f"{pre}LR probe failed; keeping the default learning rate")

        # ---- train (per-job time budget from the measured throughput) -------------------------
        job_deadline = min(self.deadline, time.time() + 60 * (2.0 * proj + 15))
        if resuming:
            job_deadline = self.deadline
        cfg["deadline_unix"] = job_deadline
        write_json(cfg_path, cfg)
        code, to = run_cmd(train_cmd(run_dir, "train"), cons, env,
                           max(60.0, job_deadline - time.time() + 600), pre)
        if code not in (0, 2, 4, 5) and not to:
            self.say(f"{pre}training exited with {code} -> one retry (resumes from last.pt)")
            code, to = run_cmd(train_cmd(run_dir, "train"), cons, env,
                               max(60.0, self.deadline - time.time()), pre)
        if code == 2 or to:
            best_exists = os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))
            if not best_exists:
                rec.update(status="PAUSED", note="time budget; last.pt saved, attach output "
                                                 "and rerun to resume")
                return False
            self.say(f"{pre}time budget reached: predicting with the best checkpoint so far")
        elif code != 0:
            why = {4: "non-finite gradients (lower --lr)", 5: "early-fail watchdog (dev F1 ~ 0)",
                   3: "out of memory"}.get(code, f"exit {code}")
            rec.update(status="FAILED", note=f"training: {why}")
            return False

        # ---- predict + submission -------------------------------------------------------------------
        code, _ = run_cmd([sys.executable, PREDICT, "--run-dir", run_dir], cons, env, 3600, pre)
        done = read_json(os.path.join(run_dir, "DONE.json"))
        if code != 0 or not done:
            if code == EXIT["validation_failure"]:
                self.validation_failed = True
            rec.update(status="FAILED", note=f"prediction/submission step failed (exit {code})")
            return False
        for f in ("last.pt", "last.pt.tmp"):
            p = os.path.join(run_dir, "checkpoints", f)
            if os.path.exists(p):
                os.remove(p)
        tc = os.path.join(run_dir, "teacher_cache.pt")
        if os.path.exists(tc):
            os.remove(tc)
        mins = (time.time() - t0) / 60
        rec.update(status="done", dev_macro_f1=done["dev_macro_f1"], minutes=mins,
                   cl=(done.get("dev_per_domain_macro_f1") or {}).get("CL"),
                   it=(done.get("dev_per_domain_macro_f1") or {}).get("IT"))
        self.say(f"{pre}finished in {mins:.1f} min | dev pooled Macro-F1 "
                 f"{done['dev_macro_f1']:.4f} | output so far {dir_size_gb(self.out_root):.2f} GB")
        return True


# =============================================================================================
# main
# =============================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="E0..E11 or P0 (pilot)")
    ap.add_argument("--models", default=None, help="comma list, default M4,M2,M3,M1")
    ap.add_argument("--gpus", default="auto", help="auto | comma list of GPU ids | cpu")
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--time-budget-h", type=float, default=11.0)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--attn-mode", default="auto", choices=["auto", "bidir", "causal"])
    ap.add_argument("--input-mode", default="doc")
    ap.add_argument("--head", default="linear", choices=["linear", "seqctx"])
    ap.add_argument("--pool", default="marker", choices=["marker", "mean"])
    ap.add_argument("--loss", default="weighted_ce",
                    choices=["ce", "weighted_ce", "focal", "logit_adjust"])
    ap.add_argument("--window-combine", default="center", choices=["center", "mean"])
    ap.add_argument("--max-sent-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--effective-batch", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--ckpt-every-min", type=float, default=30)
    ap.add_argument("--watchdog-min-f1", type=float, default=0.01)
    ap.add_argument("--probe-steps", type=int, default=None)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--skip-job-checks", action="store_true")
    ap.add_argument("--teacher-dir", default=None)
    ap.add_argument("--kd-hidden-weight", type=float, default=0.0)
    ap.add_argument("--ablation-techniques", default="lorma,lora")
    ap.add_argument("--smoke-timeout-s", type=int, default=1200)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    return ap


def main() -> None:
    a = build_parser().parse_args()

    t_start = time.time()
    deadline = t_start + a.time_budget_h * 3600
    exp = a.exp
    if exp == "P0":
        spec = {"folder": PILOT_FOLDER, "technique": "lora", "title": "attention-mode pilot"}
    elif exp in EXPERIMENTS:
        spec = EXPERIMENTS[exp]
    else:
        sys.exit(f"unknown experiment {exp!r}")
    technique, folder = spec["technique"], spec["folder"]
    out_root = os.path.abspath(a.out)
    exp_dir = os.path.join(out_root, folder)
    os.makedirs(exp_dir, exist_ok=True)
    console = os.path.join(exp_dir, "console.log")
    roots = default_input_roots()

    def say(msg: str) -> None:
        with _print_lock:
            print(msg, flush=True)
        with open(console, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    say(f"== {folder}: {spec['title']} | budget {a.time_budget_h} h | deadline of the brief "
        f"{SPEC['deadline']}")

    # ---- data + gates ---------------------------------------------------------------------------
    from rr_data import find_or_build_cache, load_cache
    try:
        data_dir = find_or_build_cache(os.path.abspath(a.data_dir), roots, log=say)
    except GateFailure as e:
        say(f"STOP AND ASK (data gate): {e}")
        sys.exit(EXIT["gate_failure"])
    except FileNotFoundError as e:
        say(f"MISSING INPUT: {e}")
        sys.exit(EXIT["missing_dependency"])
    meta, _docs = load_cache(data_dir)
    del _docs
    shutil.copy2(os.path.join(data_dir, "GATES.json"), os.path.join(out_root, "GATES.json"))
    if technique == "ensemble":                      # E11: CPU only, works on saved logits
        from rr_aggregate import build_report
        from rr_ensemble import run_ensemble
        code = run_ensemble(out_root, roots, exp_dir, data_dir, log=say)
        build_report(out_root, roots, log=say)
        sys.exit(code)
    p1 = find_phase1_models(roots)
    say(f"   Phase-1 weights found for: {sorted(p1)}")
    if not p1:
        say("MISSING: no Phase-1 model (final/config.json + model.pt) under the inputs")
        sys.exit(EXIT["missing_dependency"])

    # ---- CPU selftest (tiny models), once per output folder -------------------------------------
    st_path = os.path.join(out_root, "SELFTEST.json")
    if not a.skip_selftest and not (read_json(st_path) or {}).get("ok"):
        code, _ = run_cmd([sys.executable, SELFTEST, "--out", st_path, "--data-dir", data_dir],
                          os.path.join(out_root, "selftest.log"), {"CUDA_VISIBLE_DEVICES": ""},
                          1800, "[selftest] ")
        if code != 0 or not (read_json(st_path) or {}).get("ok"):
            say("STOP: selftest failed (see selftest.log / SELFTEST.json)")
            sys.exit(EXIT["gate_failure"])

    attn, why = ("bidir", "pilot jobs set their own mode") if exp == "P0" else \
        resolve_attn_mode(a, out_root, roots, say)
    say(f"   attention mode: {attn} ({why})")
    models = [m.strip() for m in (a.models or ",".join(
        ["M4"] if technique == "ablation" else DEFAULT_MODEL_ORDER)).split(",") if m.strip()]
    jobs = make_jobs(a, exp, technique, models, exp_dir)
    if a.resume == "auto":
        restore_from_inputs(folder, out_root, roots, say)

    # ---- workers: GPU sanity on every GPU, fallback to the ones that pass ---------------------
    env = env_info()
    say(f"   environment: {env}")
    if a.gpus.startswith("cpu"):
        gpus = [g.strip() for g in a.gpus.split(",")]       # e.g. "cpu" or "cpu0,cpu1" (tests)
    else:
        cand = [g.strip() for g in a.gpus.split(",")] if a.gpus != "auto" else \
            [str(i) for i in range(len(env.get("gpus") or []))]
        res = {}
        ths = [threading.Thread(target=lambda g=g: res.__setitem__(g, gpu_sanity(g)))
               for g in cand]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        gpus = [g for g in cand if res.get(g)]
        bad = [g for g in cand if not res.get(g)]
        say(f"   GPU sanity: ok {gpus} | failed {bad}" + (" -> falling back" if bad else ""))
        if not gpus:
            say("   no usable GPU -> running on CPU (slow)")
            gpus = ["cpu"]
    runner = Runner(a, exp, folder, exp_dir, out_root, jobs, meta, data_dir, p1, attn, env,
                    deadline, say)
    say(f"   {len(jobs)} jobs on {len(gpus)} worker(s) {gpus}: {[j['folder'] for j in jobs]}")
    threads = [threading.Thread(target=runner.worker, args=(g,), daemon=True) for g in gpus]
    for i, t in enumerate(threads):
        t.start()
        if i == 0 and len(threads) > 1:
            time.sleep(20)                         # stagger CPU-heavy start-up
    for t in threads:
        t.join()

    # ---- best-of-N + tables ------------------------------------------------------------------------
    runner.select_and_index()
    pilot = write_pilot(out_root, exp_dir, say) if exp == "P0" else None
    sel = read_json(os.path.join(exp_dir, "BEST", "SELECTION.json"))
    code = EXIT["ok"]
    if sel:
        from rr_data import load_cache as _lc
        from rr_predict import official_test_counts, validate_submission
        m2, d2 = _lc(data_dir)
        vr = validate_submission(os.path.join(exp_dir, "BEST", "submission.zip"),
                                 official_test_counts(m2, d2), m2["labels"])
        write_json(os.path.join(exp_dir, "BEST", "validation_report.json"), vr)
        if not vr["ok"]:
            say(f"   !!! BEST submission FAILED validation: {vr['errors'][:3]}")
            code = EXIT["validation_failure"]
        else:
            say(f"\n   UPLOAD: {os.path.join(exp_dir, 'BEST', 'submission.zip')} (winner "
                f"{sel['winner']}, dev pooled Macro-F1 {sel['dev_macro_f1']:.4f}) -- validated")
    if runner.validation_failed:
        code = EXIT["validation_failure"]
    elif runner.missing_dependency:
        code = EXIT["missing_dependency"]
    elif not sel and not pilot:
        code = EXIT["error"]
    say(f"\n== {folder} finished this session in {(time.time() - t_start) / 60:.1f} min "
        f"(exit {code})")
    for r in runner.records.values():
        say(f"   {r['job']:<32} {r['status']:<28} "
            f"{r.get('dev_macro_f1', float('nan')):.4f}  {r.get('note', '')}")
    say("   Save this session's output (Save Version) -- it is the input of later sessions.")
    sys.exit(code)


if __name__ == "__main__":
    main()
