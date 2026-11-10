"""
run_experiment.py -- the Phase-7 orchestrator (one notebook = one call).

    python run_experiment.py --exp E2 --models M4,M2,M3,M1 --gpu-mode jobs --resume auto \
                             --time-budget-h 2.0

Order of work (sections 3, 4, 8, 9 of the brief):
  1. Phase-1 discovery (by the `arch` field), the job list, results_<exp>.csv with one PENDING
     row per job + INDEX.md                                   <- outputs exist from minute one
  2. data discovery (attached dataset, else the HF download with the Kaggle secret `HF`),
     schema-tolerant loading of the 3 query files + 3 candidate pools, data audit, PROVENANCE
  3. chunk token cache + BM25 channel tuned on Dev (reused when an attached cache has the same
     key); eval_data.npz (dev gold, BM25 dev/test scores, ids) for E8
  4. job queue: one slot per visible GPU (no DDP); each job is a `pcr_train.py` subprocess
     with CUDA_VISIBLE_DEVICES=<slot>; per-job time budget from the notebook budget;
     retry ladder on failure; the CSV and INDEX.md are refreshed every minute
  5. BEST/ per experiment, ALL_RESULTS.csv (+ pivots), UPLOAD_PLAN.csv, final INDEX.md
Exit code: 0 if at least one job finished (DONE), else 1 (2 = data/weights missing).
The notebook prints the code and never raises (rule R1).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional

from cjpe_common import (CACHE_DIR, CODE_DIR, COUNTED, CSV_COLUMNS, DEFAULT_LR,
                         DEFAULT_MODEL_ORDER, EXPERIMENTS, JOB_EXIT, MODELS, OUT_ROOT, SPEC,
                         default_input_roots, dir_size_gb, find_phase1_models, free_gb, log_to,
                         module_md5s, now_str, read_json, student_layers, write_csv, write_json)

LOG = None


def log(msg: str) -> None:
    log_to(LOG, msg)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODEL_ORDER))
    ap.add_argument("--gpu-mode", default="jobs", choices=["jobs"])
    ap.add_argument("--resume", default="auto", choices=["auto", "off"])
    ap.add_argument("--time-budget-h", type=float, default=2.0)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--doc-repr", default="hier", choices=["hier", "head"],
                    help="hier: mean of up to --k-infer chunks spread over the case; head: chunk 1")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=16, help="chunks cached per case")
    ap.add_argument("--k-train", type=int, default=2, help="chunks per case per training step")
    ap.add_argument("--k-infer", type=int, default=8, help="chunks per case at inference")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--attn-mode", default="causal", choices=["causal", "bidir"])
    ap.add_argument("--batch-pairs", type=int, default=16)
    ap.add_argument("--hard-negs", type=int, default=3)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--min-dense-weight", type=float, default=0.2,
                    help="fusion weight of the trained encoder is tuned on dev in [this, 1]")
    ap.add_argument("--n-virtual", type=int, default=20, help="prefix / prompt length")
    ap.add_argument("--epochs-max", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=1)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-model", default="M4")
    ap.add_argument("--teacher-frac", type=float, default=0.4)
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-T", type=float, default=2.0)
    ap.add_argument("--ablation-model", default="M4")
    ap.add_argument("--ablation-technique", default="lorma")
    ap.add_argument("--e7-jobs", default="ia3,prompt,zeroshot,pool_last,attn_bidir,repr_head,"
                                         "hn_0")
    ap.add_argument("--e7-fixed-frac", type=float, default=0.5)
    ap.add_argument("--e7-fixed-epochs", type=int, default=1)
    ap.add_argument("--skip-job-checks", action="store_true")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--n-proc", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--reserve-min", type=float, default=6.0)
    ap.add_argument("--lr-mult", type=float, default=1.0)
    return ap


# =============================================================================================
# jobs
# =============================================================================================
def make_jobs(a, p1: Dict[str, str]) -> List[Dict]:
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    e = a.exp
    jobs = []
    if e in COUNTED and e != "E6":
        tech = EXPERIMENTS[e]["technique"]
        for m in models:
            jobs.append({"job": MODELS[m]["folder"], "model": m, "technique": tech})
    elif e == "E6":
        t = a.teacher_model
        jobs.append({"job": f"teacher_{MODELS[t]['folder']}", "model": t,
                     "technique": "full_ft", "is_teacher": True})
        for m in models:
            jobs.append({"job": f"student_{MODELS[m]['folder']}", "model": m, "technique": "kd",
                         "student": True, "depends": [f"teacher_{MODELS[t]['folder']}"]})
    elif e == "E7":
        am = a.ablation_model
        fixed = {"epochs": a.e7_fixed_epochs, "frac": a.e7_fixed_frac}
        for spec in [s.strip() for s in a.e7_jobs.split(",") if s.strip()]:
            f = MODELS[am]["folder"]
            j = {"job": f"{spec}_{f}", "model": am, "technique": a.ablation_technique,
                 "fixed_plan": fixed}
            if spec in ("ia3", "prompt", "prefix", "qlora", "lora", "lorma", "full_ft",
                        "partial_ft"):
                j = {"job": f"{spec}_{f}", "model": am, "technique": spec}
            elif spec == "zeroshot":
                j["fixed_plan"] = {"epochs": 0}
            elif spec.startswith("pool_"):
                j["pooling"] = spec[5:]
            elif spec.startswith("attn_"):
                j["attn_mode"] = spec[5:]
            elif spec.startswith("repr_"):
                j["doc_repr"] = spec[5:]
            elif spec.startswith("hn_"):
                j["hard_negs"] = int(spec[3:])
            else:
                continue
            jobs.append(j)
    for j in jobs:
        j.setdefault("depends", [])
        j["state"] = "pending"
        j["attempt"] = 0
        j["overrides"] = {}
    return jobs


def planned_row(a, j: Dict) -> Dict:
    return {"experiment": a.exp, "technique": j["technique"] + (" (teacher)" if j.get(
        "is_teacher") else ""), "model_id": j["model"], "arch": MODELS[j["model"]]["arch"],
            "job": j["job"], "status": "PENDING", "doc_repr": j.get("doc_repr", a.doc_repr),
            "pooling": j.get("pooling", a.pooling), "attn_mode": j.get("attn_mode", a.attn_mode),
            "seed": a.seed}


# =============================================================================================
# INDEX.md
# =============================================================================================
def write_index(a, out_root, exp_dir, jobs, extra: Optional[List[str]] = None) -> None:
    from aggregate import write_exp_csvs
    rows = write_exp_csvs(exp_dir, [planned_row(a, j) for j in jobs]) if jobs else []
    lines = [f"# Phase 7 (prior case retrieval) -- {a.exp}: {EXPERIMENTS[a.exp]['title']}", "",
             f"Updated {now_str()}. Deadline {SPEC['deadline']}.", "",
             "| job | model | technique | status | Micro-F1@7 | dense only | BM25 only | dense w "
             "| epochs | train frac | minutes | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, n=4: (f"{float(r[k]):.{n}f}" if r.get(k) not in (None, "") else "")  # noqa
        lines.append(f"| {r.get('job')} | {r.get('model_id')} | {r.get('technique')} | "
                     f"{r.get('status')} | {f('dev_micro_f1')} | {f('dev_dense_f1')} | "
                     f"{f('dev_bm25_f1')} | {f('dense_weight', 1)} | {f('epochs_run', 2)} | "
                     f"{f('train_fraction', 2)} | {f('train_minutes', 1)} | "
                     f"{str(r.get('fail_reason') or '').replace('|', '/')[:120]} |")
    todo = [r for r in rows if r.get("status") in ("FAILED", "SKIPPED_TIME", "PENDING")]
    if todo:
        ms = sorted({r["model_id"] for r in todo if r.get("model_id")})
        lines += ["", "## Next command for the unfinished models (here or on another account)",
                  "", "```", f"!cd /kaggle/working/phase7 && python run_experiment.py --exp "
                             f"{a.exp} --models {','.join(ms)} --resume auto --time-budget-h "
                             f"{a.time_budget_h}", "```",
                  "Attach this notebook's output as an input so finished models are skipped."]
    if extra:
        lines += [""] + extra
    p = os.path.join(out_root, "INDEX.md")
    os.makedirs(out_root, exist_ok=True)
    with open(p + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(p + ".tmp", p)


# =============================================================================================
# data
# =============================================================================================
def prepare_data(a, out_root: str, bpe_dir: str) -> Dict:
    import numpy as np
    from ner_common import file_md5
    from pcr_data import build_cache, cache_key, chunk_stats, find_cache, load_all, load_cache
    roots = default_input_roots()
    data, schemas, info = load_all(roots, log)
    write_json(os.path.join(out_root, "PROVENANCE.json"),
               {"source": info["source"], "files": {k: {"path": v["path"], "md5": v["md5"],
                                                        "n": v["n"]} for k, v in schemas.items()},
                "rule": "each split's queries are ranked against that split's pool only; dev "
                        "labels are used for model selection, never for training"})
    key = cache_key({s: schemas[s]["md5"] for s in schemas},
                    file_md5(os.path.join(bpe_dir, "bpe_model.json")), a.chunk_tokens,
                    a.max_chunks)
    path = find_cache(key, [a.cache_dir] + roots)
    if path:
        log(f"   [cache] reusing {path}")
    else:
        t0 = time.time()
        log(f"   [cache] tokenising every case into <= {a.max_chunks} chunks of "
            f"{a.chunk_tokens} tokens ({a.n_proc} processes) + BM25 ...")
        path = build_cache(data, bpe_dir, a.cache_dir, key, a.chunk_tokens, a.max_chunks,
                           a.n_proc, log)
        log(f"   [cache] built in {time.time() - t0:.0f}s")
    cache = load_cache(path)
    sp, bm = cache["splits"], cache["bm25"]
    write_json(os.path.join(out_root, "test_ids.json"), [x["id"] for x in sp["test"]["q"]])
    write_json(os.path.join(out_root, "test_pool_ids.json"), [x["id"] for x in sp["test"]["pool"]])
    np.savez_compressed(os.path.join(out_root, "eval_data.npz"),
                        bm25_dev=bm["dev"].astype(np.float16), bm25_test=bm["test"].astype(np.float16),
                        dev_q=np.array([x["id"] for x in sp["dev"]["q"]]),
                        dev_pool=np.array([x["id"] for x in sp["dev"]["pool"]]),
                        test_q=np.array([x["id"] for x in sp["test"]["q"]]),
                        test_pool=np.array([x["id"] for x in sp["test"]["pool"]]),
                        dev_gold=np.array([json.dumps(x.get("rel") or []) for x in sp["dev"]["q"]]),
                        dev_chars=np.array([x.get("n_chars") or 0 for x in sp["dev"]["q"]]))
    audit = {**info["audit"], **chunk_stats(cache), "bm25": {k: v for k, v in bm.items()
                                                             if k in ("params", "dev_micro_f1",
                                                                      "dev_micro_p", "dev_micro_r",
                                                                      "dev_curve")}}
    write_json(os.path.join(out_root, "data_audit.json"), audit)
    log(f"   [audit] chunks per case (mean, cap {a.max_chunks}): train pool "
        f"{audit['train_pool_chunks_mean']:.1f}, dev pool {audit['dev_pool_chunks_mean']:.1f}, "
        f"test pool {audit['test_pool_chunks_mean']:.1f}; {100 * audit['test_pool_chunks_full_frac']:.0f}% "
        f"of test candidates longer than {a.max_chunks} chunks (sampled evenly)")
    log(f"   [bm25] baseline (lexical channel, same for every job): dev Micro-F1@7 "
        f"{bm['dev_micro_f1']:.4f} with {bm['params']}")
    return {"cache_path": path, "cache_key": key,
            "n": {s: len(sp[s]["q"]) for s in sp}, "pool": {s: len(sp[s]["pool"]) for s in sp},
            "data_md5": {s: schemas[s]["md5"] for s in schemas}}


# =============================================================================================
# scheduler
# =============================================================================================
def n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def job_config(a, j: Dict, p1: Dict[str, str], data: Dict, exp_dir: str, slot: int,
               budget_s: float, attn: str, bpe_dir: str, jobs_by_name: Dict) -> Dict:
    tech = j["technique"]
    lr = DEFAULT_LR.get(tech, 5e-4) * a.lr_mult * j["overrides"].get("lr_mult", 1.0)
    cfg = {
        "experiment": a.exp, "exp_folder": EXPERIMENTS[a.exp]["folder"], "job": j["job"],
        "model": j["model"], "model_folder": MODELS[j["model"]]["folder"],
        "arch": MODELS[j["model"]]["arch"], "phase1_dir": p1[j["model"]], "bpe_dir": bpe_dir,
        "technique": tech, "doc_repr": j.get("doc_repr", a.doc_repr),
        "pooling": j.get("pooling", a.pooling), "attn_mode": j.get("attn_mode", a.attn_mode),
        "chunk_tokens": a.chunk_tokens, "k_train": a.k_train, "k_infer": a.k_infer,
        "batch_pairs": a.batch_pairs, "hard_negs": int(j.get("hard_negs", a.hard_negs)),
        "min_dense_weight": a.min_dense_weight, "lr": lr, "head_lr": max(lr, 1e-4),
        "weight_decay": 0.01, "grad_clip": 1.0, "warmup_frac": 0.06, "epochs_max": a.epochs_max,
        "min_epochs": a.min_epochs, "evals_per_epoch": a.evals_per_epoch,
        "patience": a.patience, "effective_batch": a.batch_pairs,
        "micro_tokens": int(j["overrides"].get("micro_tokens", a.micro_tokens)),
        "n_virtual": a.n_virtual, "seed": a.seed,
        "budget_s": budget_s, "cache_path": data["cache_path"], "cache_key": data["cache_key"],
        "data_md5": data["data_md5"], "n_train": data["n"]["train"], "n_dev": data["n"]["dev"],
        "n_test": data["n"]["test"], "skip_job_checks": bool(
            a.skip_job_checks or j["overrides"].get("skip_job_checks")),
        "is_teacher": bool(j.get("is_teacher")), "fixed_plan": j.get("fixed_plan"),
        "gpu_slot": slot, "attempt": j["attempt"], "started_at": now_str(),
        "leak_guard": "passed", "ckpt_every_min": 30,
        "min_train_fraction": 0.1, "kd_alpha": a.kd_alpha, "kd_T": a.kd_T,
        "lorma_r": 16, "lorma_alpha": 32, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "partial_top_k": 6, "layer_decay": 0.85, "module_md5": module_md5s(),
    }
    if j.get("student"):
        cfg["layer_ids"] = student_layers(j["model"], p1[j["model"]])
        tdir = os.path.join(exp_dir, j["depends"][0])
        cfg["teacher_dir"] = tdir
        if not (os.path.exists(os.path.join(tdir, "DONE.json")) and
                os.path.exists(os.path.join(tdir, "checkpoints", "best.pt"))):
            # teacher failed: train the pruned student on the gold labels only (CSV says so)
            cfg["technique"] = "student_hard"
            cfg["teacher_failed"] = True
    return cfg


def copy_from_inputs(a, exp_dir: str, job: str) -> Optional[str]:
    """--resume auto: a finished (or resumable) copy of this job in an attached input. The E6
    teacher (full FT of M4) is also taken from a finished E2_full_ft output when attached."""
    folder = EXPERIMENTS[a.exp]["folder"]
    from cjpe_common import iter_out_dirs
    cands = [(folder, job)]
    if a.exp == "E6" and job.startswith("teacher_"):
        cands.append((EXPERIMENTS["E2"]["folder"], job[len("teacher_"):]))
    for od in iter_out_dirs(None, default_input_roots()):
      for fo, jb in cands:
        src = os.path.join(od, fo, jb)
        if fo != folder and not os.path.exists(os.path.join(src, "DONE.json")):
            continue
        if os.path.exists(os.path.join(src, "DONE.json")) or os.path.exists(
                os.path.join(src, "checkpoints", "last.pt")):
            dst = os.path.join(exp_dir, job)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            if fo != folder:                                  # E2 run reused as the E6 teacher
                rp = os.path.join(dst, "logs", "run_config.json")
                rc = read_json(rp, {}) or {}
                rc.update({"experiment": a.exp, "exp_folder": folder, "job": job,
                           "is_teacher": True, "reused_from": src})
                write_json(rp, rc)
            return src
    return None


def archive_attempt(run_dir: str, attempt: int) -> None:
    lg = os.path.join(run_dir, "logs")
    for f in ("console.log", "job_checks.json", "plan.json", "train_log.jsonl",
              "progress.json"):
        p = os.path.join(lg, f)
        if os.path.exists(p):
            os.replace(p, os.path.join(lg, f"attempt{attempt}_{f}"))
    for f in ("checkpoints/last.pt", "checkpoints/best.pt", "metrics.json", "DONE.json"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)


def next_attempt(a, j: Dict, code: int, reducible: bool) -> Optional[str]:
    """The retry ladder (rule R5). Returns a description, or None when giving up."""
    if code == 2:
        return None
    ladder = []
    if code in (4, 5):
        ladder.append(("lr_mult", 0.3, "learning rate x0.3"))
    if code == 7 or code == 1:
        ladder.append(("micro_tokens", max(1024, a.micro_tokens // 2), "micro-batch halved"))
        ladder.append(("micro_tokens", max(512, a.micro_tokens // 4), "micro-batch / 4"))
    if code == 6:
        ladder.append(("skip_job_checks", True, "self-checks skipped"))
    used = j.setdefault("ladder_used", [])
    for k, v, desc in ladder:
        if desc not in used:
            used.append(desc)
            j["overrides"][k] = v
            return desc
    return None


def schedule(a, jobs: List[Dict], p1, data, exp_dir, attn, bpe_dir, t_end, out_root) -> None:
    slots = max(1, n_gpus())
    log(f"   [queue] {len(jobs)} jobs on {slots} slot(s) "
        f"({'GPU' if n_gpus() else 'CPU'}); notebook budget ends at "
        f"{time.strftime('%H:%M', time.localtime(t_end))}")
    by = {j["job"]: j for j in jobs}
    running: Dict[int, Dict] = {}
    last_refresh = 0.0
    reserve = 60 * a.reserve_min
    while True:
        # ---- launch ---------------------------------------------------------------------
        for slot in range(slots):
            while slot not in running:
                ready = [j for j in jobs if j["state"] == "pending" and all(
                    by[d]["state"] in ("done", "failed", "skipped") for d in j["depends"]
                    if d in by)]
                if not ready:
                    break
                j = ready[0]
                remaining = t_end - time.time() - reserve
                n_pending = sum(1 for x in jobs if x["state"] == "pending")
                if j.get("is_teacher") and any(x.get("student") for x in jobs):
                    budget = a.teacher_frac * remaining
                else:
                    budget = remaining * slots / max(slots, n_pending + len(running))
                    budget = min(budget, remaining)
                run_dir = os.path.join(exp_dir, j["job"])
                if budget < 8 * 60:
                    j["state"] = "skipped"
                    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
                    cfg = job_config(a, j, p1, data, exp_dir, slot, budget, attn, bpe_dir, by)
                    write_json(os.path.join(run_dir, "logs", "run_config.json"), cfg)
                    write_json(os.path.join(run_dir, "status.json"),
                               {"state": "skipped_time", "time": now_str(), "exit_code": 2,
                                "reason": f"only {budget / 60:.0f} min left in the notebook budget"})
                    log(f"   [queue] SKIP {j['job']}: only {budget / 60:.0f} min left")
                    continue
                j["attempt"] += 1
                cfg = job_config(a, j, p1, data, exp_dir, slot, budget, attn, bpe_dir, by)
                os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
                write_json(os.path.join(run_dir, "logs", "run_config.json"), cfg)
                st = read_json(os.path.join(run_dir, "status.json"), {}) or {}
                write_json(os.path.join(run_dir, "status.json"),
                           {**st, "state": "preflight", "time": now_str(), "reason": "",
                            "attempt": j["attempt"]})
                mode = "train"
                env = dict(os.environ, PYTHONPATH=CODE_DIR, PYTHONUNBUFFERED="1")
                if n_gpus():
                    env["CUDA_VISIBLE_DEVICES"] = str(slot)
                cmd = [sys.executable, os.path.join(CODE_DIR, "pcr_train.py"), "--run-dir", run_dir,
                       "--mode", mode]
                log(f"   [queue] START {j['job']} ({cfg['technique']}, attempt {j['attempt']}) on "
                    f"slot {slot}, budget {budget / 60:.0f} min")
                j["proc"] = subprocess.Popen(cmd, env=env, cwd=CODE_DIR)
                j["state"] = "running"
                j["slot"] = slot
                running[slot] = j
        if not running:
            break
        # ---- poll -----------------------------------------------------------------------
        time.sleep(5)
        for slot, j in list(running.items()):
            code = j["proc"].poll()
            if code is None:
                continue
            del running[slot]
            run_dir = os.path.join(exp_dir, j["job"])
            st = read_json(os.path.join(run_dir, "status.json"), {}) or {}
            if code == 0 and st.get("state") in ("done", "degenerate", "smoke_ok"):
                j["state"] = "done"
                log(f"   [queue] END {j['job']}: {st.get('state')} ({st.get('reason', '')})")
                continue
            reason = f"exit {code} ({JOB_EXIT.get(code, '?')}): {st.get('reason', '')}"
            log(f"   [queue] FAIL {j['job']}: {reason[:300]}")
            reducible = False
            remaining = t_end - time.time() - reserve
            desc = None
            if j["attempt"] <= a.max_retries and remaining > 15 * 60:
                desc = next_attempt(a, j, code, reducible)
            if desc:
                archive_attempt(run_dir, j["attempt"])
                write_json(os.path.join(run_dir, "status.json"),
                           {"state": "pending", "time": now_str(),
                            "reason": f"retry after {reason[:300]} -> {desc}"})
                j["state"] = "pending"
                log(f"   [queue] RETRY {j['job']} with {desc}")
            else:
                j["state"] = "failed"
                write_json(os.path.join(run_dir, "status.json"),
                           {**st, "state": "skipped_time" if code == 2 else "failed",
                            "exit_code": code, "time": now_str(), "reason": reason[:2000]})
        if time.time() - last_refresh > 60:
            try:
                write_index(a, out_root, exp_dir, jobs)
            except Exception as e:
                log(f"   [index] refresh failed: {e}")
            last_refresh = time.time()


# =============================================================================================
# main
# =============================================================================================
def finalize(a, out_root, exp_dir, jobs, extra=None) -> int:
    from aggregate import aggregate, error_analysis, ensemble
    from select_best import plan_uploads, select_exp
    rows = []
    try:
        from aggregate import write_exp_csvs
        rows = write_exp_csvs(exp_dir, [planned_row(a, j) for j in jobs])
        if a.exp in COUNTED or a.exp == "E7":
            select_exp(exp_dir, rows, log)
    except Exception as e:
        log(f"   [final] BEST selection failed: {e}\n{traceback.format_exc()[-1500:]}")
    try:
        allrows = aggregate(out_root, default_input_roots(), log)
        log(f"   [final] ALL_RESULTS.csv: {len(allrows)} rows")
        plan = plan_uploads(allrows, os.path.join(out_root, "UPLOAD_PLAN.csv"), log)
        if a.exp == "E8":
            ensemble(allrows, out_root, log)
            error_analysis(allrows, out_root, None, log)
    except Exception as e:
        log(f"   [final] aggregation failed: {e}\n{traceback.format_exc()[-1500:]}")
    write_index(a, out_root, exp_dir, jobs, extra)
    done = [r for r in rows if r.get("status") == "DONE"]
    write_json(os.path.join(exp_dir, "summary.json"),
               {"experiment": a.exp, "finished": now_str(), "rows": rows,
                "output_gb": dir_size_gb(out_root)})
    log(f"   [final] {len(done)}/{len(rows)} jobs DONE; output {dir_size_gb(out_root):.2f} GB")
    return 0 if (done or a.exp == "E8") else 1


def main() -> int:
    global LOG
    a = build_parser().parse_args()
    t_start = time.time()
    t_end = t_start + 3600 * a.time_budget_h
    out_root = a.out_root
    exp_dir = os.path.join(out_root, EXPERIMENTS[a.exp]["folder"])
    os.makedirs(exp_dir, exist_ok=True)
    LOG = os.path.join(exp_dir, "orchestrator.log")
    log(f"=== Phase 7 {a.exp}: {EXPERIMENTS[a.exp]['title']} | {now_str()} | budget "
        f"{a.time_budget_h} h | {n_gpus()} GPU(s) | free disk {free_gb(out_root):.0f} GB ===")
    if a.exp == "E8":
        jobs: List[Dict] = []
        write_index(a, out_root, exp_dir, jobs)
        return finalize(a, out_root, exp_dir, jobs)

    # ---- 1. Phase-1 models + job list + PENDING rows ----------------------------------------
    p1 = find_phase1_models(default_input_roots())
    for m in MODELS:
        log(f"   [phase1] {m} {MODELS[m]['arch']:<16} -> {p1.get(m, 'NOT FOUND')}")
    jobs = make_jobs(a, p1)
    need = sorted({j["model"] for j in jobs})
    missing = [m for m in need if m not in p1]
    write_index(a, out_root, exp_dir, jobs)
    if missing:
        msg = (f"Phase-1 weights missing for {missing} ({[MODELS[m]['arch'] for m in missing]}). "
               "Attach the four Phase-1 datasets (final/model.pt + config.json + bpe_model.json).")
        log("!" * 90 + f"\n   {msg}\n" + "!" * 90)
        jobs = [j for j in jobs if j["model"] not in missing]
        write_json(os.path.join(out_root, "BLOCKED.json"), {"reason": msg, "time": now_str()})
        if not jobs:
            write_index(a, out_root, exp_dir, [], [f"**BLOCKED:** {msg}"])
            return 2
    # bpe_model.json: next to a Phase-1 model.pt, or anywhere in any attached dataset
    # (e.g. a separate "bpe_model" dataset).
    from ner_common import find_files
    bpe_dir = None
    for d in list(p1.values()):
        if os.path.exists(os.path.join(d, "bpe_model.json")):
            bpe_dir = d
            break
    if bpe_dir is None:
        hits = find_files("bpe_model.json", default_input_roots(), 50)
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            bpe_dir = os.path.dirname(hits[0])
            log(f"   [bpe] using {hits[0]} ({len(hits)} found)")
    if bpe_dir is None:
        msg = ("bpe_model.json not found in any attached dataset (searched "
               f"{default_input_roots()})")
        write_json(os.path.join(out_root, "BLOCKED.json"), {"reason": msg})
        write_index(a, out_root, exp_dir, jobs, [f"**BLOCKED:** {msg}"])
        return 2

    attn = "causal"
    try:
        data = prepare_data(a, out_root, bpe_dir)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log("!" * 90 + f"\n   DATA STEP FAILED: {msg}\n" + "!" * 90)
        log(traceback.format_exc()[-2000:])
        write_json(os.path.join(out_root, "BLOCKED.json"), {"reason": msg, "time": now_str()})
        for j in jobs:
            rd = os.path.join(exp_dir, j["job"])
            os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
            write_json(os.path.join(rd, "status.json"), {"state": "failed", "reason": msg,
                                                         "time": now_str()})
        write_index(a, out_root, exp_dir, jobs, [f"**BLOCKED (data):** {msg}"])
        return 2

    # ---- 4. resume / queue ---------------------------------------------------------------
    for j in jobs:
        rd = os.path.join(exp_dir, j["job"])
        if a.resume == "auto":
            if os.path.exists(os.path.join(rd, "DONE.json")):
                j["state"] = "done"
                log(f"   [resume] {j['job']} already DONE -> skipped")
                continue
            src = None if os.path.exists(os.path.join(rd, "checkpoints", "last.pt")) else \
                copy_from_inputs(a, exp_dir, j["job"])
            if src:
                if os.path.exists(os.path.join(rd, "DONE.json")):
                    j["state"] = "done"
                    log(f"   [resume] {j['job']} finished in an earlier session ({src}) -> copied")
                else:
                    log(f"   [resume] {j['job']} resumes from {src}")
        os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
        if j["state"] == "pending" and not os.path.exists(os.path.join(rd, "status.json")):
            write_json(os.path.join(rd, "status.json"), {"state": "pending", "time": now_str()})
    write_index(a, out_root, exp_dir, jobs)
    try:
        schedule(a, jobs, p1, data, exp_dir, attn, bpe_dir, t_end, out_root)
    except Exception as e:
        log(f"   [queue] scheduler error: {e}\n{traceback.format_exc()[-2000:]}")
        for j in jobs:
            if j.get("proc") is not None and j["proc"].poll() is None:
                j["proc"].terminate()
    return finalize(a, out_root, exp_dir, jobs)


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"run_experiment.py crashed: {e}\n{traceback.format_exc()}", flush=True)
        code = 1
    sys.exit(code)
