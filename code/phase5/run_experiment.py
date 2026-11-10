"""
run_experiment.py -- the Phase-4 orchestrator (one notebook = one call).

    python run_experiment.py --exp E2 --models M4,M2,M3,M1 --gpu-mode jobs --resume auto \
                             --time-budget-h 3.5

Order of work (sections 3, 4, 8, 9 of the brief):
  1. Phase-1 discovery (by the `arch` field), the job list, results_<exp>.csv with one PENDING
     row per job + INDEX.md                                   <- outputs exist from minute one
  2. data discovery (attached dataset, else the HF download with the Kaggle secret `HF`),
     schema-tolerant loading, data audit, LEAK GUARD, PROVENANCE.json
  3. token cache (reused when an attached cache has the same key)
  4. job queue: one slot per visible GPU (no DDP); each job is a `cjpe_train.py` subprocess
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
    ap.add_argument("--time-budget-h", type=float, default=3.5)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--doc-repr", default="hier",
                    choices=["hier", "headtail", "first_k", "last_k", "sections", "full"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--attn-mode", default=None, choices=[None, "bidir", "causal"])
    ap.add_argument("--class-weight", default="sqrt_inv", choices=["none", "sqrt_inv", "inv"])
    ap.add_argument("--no-tau-scan", action="store_true")
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--micro-chunks", type=int, default=16)
    ap.add_argument("--infer-chunks", type=int, default=48)
    ap.add_argument("--epochs-max", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=None,
                    help="default: 1 for full / partial FT and KD students, 2 for PEFT")
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-model", default="M4")
    ap.add_argument("--teacher-frac", type=float, default=0.38)
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-T", type=float, default=2.0)
    ap.add_argument("--ablation-model", default="M4")
    ap.add_argument("--ablation-technique", default="lorma")
    ap.add_argument("--e7-jobs", default="bitfit,repr_spread,repr_cite,repr_last_k,loss_asl,"
                                         "loss_focal,cw_none")
    ap.add_argument("--loss", default="bce", choices=["bce", "asl", "focal"])
    ap.add_argument("--e7-fixed-frac", type=float, default=0.5)
    ap.add_argument("--e7-fixed-epochs", type=int, default=2)
    ap.add_argument("--skip-job-checks", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--n-proc", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--audit-sample", type=int, default=2000)
    ap.add_argument("--reserve-min", type=float, default=8.0)
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
            if spec == "bitfit":
                for m in ("M4", "M1"):
                    jobs.append({"job": f"bitfit_{MODELS[m]['folder']}", "model": m,
                                 "technique": "bitfit"})
            elif spec == "lora":
                jobs.append({"job": f"lora_{MODELS[am]['folder']}", "model": am,
                             "technique": "lora"})
            elif spec.startswith("repr_"):
                r = spec[5:]
                jobs.append({"job": f"repr_{r}_{MODELS[am]['folder']}", "model": am,
                             "technique": a.ablation_technique, "doc_repr": r,
                             "fixed_plan": fixed, "K": 2 if r == "sections" else
                             (16 if r == "full" else a.k)})
                if r in ("spread", "cite"):
                    jobs[-1]["K"] = 4
            elif spec.startswith("loss_"):
                jobs.append({"job": f"{spec}_{MODELS[am]['folder']}", "model": am,
                             "technique": a.ablation_technique, "loss": spec[5:],
                             "fixed_plan": fixed})
            elif spec.startswith("cw_"):
                jobs.append({"job": f"{spec}_{MODELS[am]['folder']}", "model": am,
                             "technique": a.ablation_technique, "class_weight": spec[3:],
                             "fixed_plan": fixed})
    elif e == "P0":
        for tech in ("full_ft", "lorma", "adalora", "adapters", "partial_ft"):
            jobs.append({"job": f"smoke_{tech}_{MODELS['M4']['folder']}", "model": "M4",
                         "technique": tech, "mode": "smoke"})
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
            "class_weight": j.get("class_weight", a.class_weight), "train_files": "train.json",
            "seed": a.seed}


# =============================================================================================
# INDEX.md
# =============================================================================================
def write_index(a, out_root, exp_dir, jobs, extra: Optional[List[str]] = None) -> None:
    from aggregate import write_exp_csvs
    rows = write_exp_csvs(exp_dir, [planned_row(a, j) for j in jobs]) if jobs else []
    lines = [f"# Phase 5 (LSI) -- {a.exp}: {EXPERIMENTS[a.exp]['title']}", "",
             f"Updated {now_str()}. Deadline {SPEC['deadline']}.", "",
             "| job | model | technique | status | dev macro-F1 | tau | epochs | K | train frac "
             "| minutes | reason |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, n=4: (f"{float(r[k]):.{n}f}" if r.get(k) not in (None, "") else "")  # noqa
        lines.append(f"| {r.get('job')} | {r.get('model_id')} | {r.get('technique')} | "
                     f"{r.get('status')} | {f('dev_macro_f1')} | {f('tau', 2)} "
                     f"| {f('epochs_run', 2)} | {r.get('K_effective') or ''} | "
                     f"{f('train_fraction', 2)} | {f('train_minutes', 1)} | "
                     f"{str(r.get('fail_reason') or '').replace('|', '/')[:120]} |")
    todo = [r for r in rows if r.get("status") in ("FAILED", "SKIPPED_TIME", "PENDING")]
    if todo:
        ms = sorted({r["model_id"] for r in todo if r.get("model_id")})
        lines += ["", "## Next command for the unfinished models (here or on another account)",
                  "", "```", f"!cd /kaggle/working/phase5 && python run_experiment.py --exp "
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
def prepare_data(a, out_root: str, bpe_dir: str, need_head: int, need_tail: int,
                 sections: bool) -> Dict:
    from bpe_tokenizer import BPETokenizer
    from cjpe_data import (DataError, basic_audit, discover_files, leak_guard, load_split,
                           print_audit, print_examples, provenance, token_audit)
    from cjpe_tok import build_cache, cache_key, find_cache
    roots = default_input_roots()
    found = discover_files(roots, log=log)
    if found["source"] is None:
        raise FileNotFoundError(
            "Phase-5 data not found. Attach a Kaggle dataset with Phase_5_LSI/Train/train.json, "
            "Train/dev.json, Test/test.json -- OR add the Kaggle secret `HF` (read-only "
            "Hugging Face token of an account that accepted the dataset terms) and turn "
            f"Internet ON. Found so far: {found['paths']}")
    log(f"   [data] source: {found['source']}: {found['paths']}")
    data, schemas = {}, {}
    for s in ("train", "dev", "test"):
        data[s], schemas[s] = load_split(found["paths"][s], s, log)
        log(f"   [data] {s}: {schemas[s]['n']} records, {schemas[s]['container']}, keys "
            f"id={schemas[s]['id_key']!r} text={schemas[s]['text_key']!r} "
            f"label={schemas[s]['label_key']!r}")
    print_examples(data, log)
    audit = basic_audit(data, schemas)
    audit["leak_guard"] = leak_guard(data, log)
    if audit["leak_guard"]["train_test_overlap_pct"] > 2.0:
        log("   !!! [leak-guard] more than 2% of train overlaps test. Those cases were DROPPED; "
            "report this to the TAs (section 13).")
    tok = BPETokenizer.from_pretrained(bpe_dir)
    tok._bpe_dir = bpe_dir
    t0 = time.time()
    audit["tokens"] = token_audit(data, tok, a.audit_sample, a.seed, a.n_proc)
    log(f"   [audit] token statistics in {time.time() - t0:.0f}s")
    print_audit(audit, log)
    write_json(os.path.join(out_root, "data_audit.json"), audit)
    prov = provenance(schemas, audit["leak_guard"], len(data["train"]), found["source"])
    write_json(os.path.join(out_root, "PROVENANCE.json"), prov)
    # cache
    from ner_common import file_md5
    key = cache_key({s: schemas[s]["md5"] for s in schemas},
                    file_md5(os.path.join(bpe_dir, "bpe_model.json")))
    # the leak guard is part of the cache content: include the dropped counts in the key
    key = f"{key}{audit['leak_guard']['dropped_from_train']}x{audit['leak_guard']['dropped_from_dev']}"
    path = find_cache(key, need_head, need_tail, sections, [a.cache_dir] + roots)
    if path:
        log(f"   [cache] reusing {path}")
    else:
        log(f"   [cache] building (key {key}, head {need_head} / tail {need_tail} tokens per "
            f"document, {a.n_proc} processes) ...")
        path = build_cache(data, bpe_dir, a.cache_dir, key, need_head, need_tail, sections,
                           a.n_proc, log)
    snips = {d["id"]: d["text"][:200].replace("\n", " ") for d in data["dev"]}
    sp = os.path.join(a.cache_dir, "dev_snippets.json")
    write_json(sp, snips)
    test_ids = [d["id"] for d in data["test"]]
    write_json(os.path.join(out_root, "test_ids.json"), test_ids)
    # Phase 5 label map: statute ids seen in train.json (the only ids we may predict)
    label_ids = sorted({int(x) for d in data["train"] for x in (d["label"] or [])})
    dev_only = sorted({int(x) for d in data["dev"] for x in (d["label"] or [])} - set(label_ids))
    per_doc = sorted(len(d["label"] or []) for d in data["train"])
    kmax = max(1, per_doc[min(len(per_doc) - 1, int(0.99 * len(per_doc)))]) if per_doc else 5
    write_json(os.path.join(out_root, "label_map.json"),
               {"label_ids": label_ids, "dev_only_ids": dev_only, "kmax": kmax})
    log(f"   [labels] {len(label_ids)} statute ids in train; {len(dev_only)} ids only in dev "
        f"(never predicted); kmax (p99 labels/doc) = {kmax}")
    if not label_ids:
        raise ValueError("train.json has no statute labels")
    return {"cache_path": path, "label_ids": label_ids, "kmax": kmax, "snippets_path": sp, "n": {s: len(v) for s, v in data.items()},
            "cache_key": key, "data_md5": {s: schemas[s]["md5"] for s in schemas},
            "dev_texts": {d["id"]: d["text"] for d in data["dev"]} if a.exp == "E8" else None}


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
    lr = DEFAULT_LR.get(tech, 1e-3) * a.lr_mult * j["overrides"].get("lr_mult", 1.0)
    head_lr = 10 * lr if tech in ("full_ft", "kd", "student_hard") else 1e-3
    cfg = {
        "experiment": a.exp, "exp_folder": EXPERIMENTS[a.exp]["folder"], "job": j["job"],
        "model": j["model"], "model_folder": MODELS[j["model"]]["folder"],
        "arch": MODELS[j["model"]]["arch"], "phase1_dir": p1[j["model"]], "bpe_dir": bpe_dir,
        "technique": tech, "attn_mode": attn, "doc_repr": j.get("doc_repr", a.doc_repr),
        "K": int(j["overrides"].get("K", j.get("K", a.k))),
        "class_weight": j.get("class_weight", a.class_weight), "lr": lr, "head_lr": head_lr,
        "weight_decay": 0.01, "grad_clip": 1.0, "warmup_frac": 0.06,
        "epochs_max": a.epochs_max,
        "min_epochs": a.min_epochs if a.min_epochs is not None else (
            1 if tech in ("full_ft", "kd", "student_hard", "partial_ft") else 2),
        "evals_per_epoch": a.evals_per_epoch, "patience": a.patience,
        "effective_batch": a.effective_batch,
        "micro_chunks": int(j["overrides"].get("micro_chunks", a.micro_chunks)),
        "infer_chunks": a.infer_chunks, "seed": a.seed, "budget_s": budget_s,
        "cache_path": data["cache_path"], "snippets_path": data["snippets_path"],
        "cache_key": data["cache_key"], "data_md5": data["data_md5"],
        "n_train": data["n"]["train"], "n_dev": data["n"]["dev"], "n_test": data["n"]["test"],
        "tau_scan": not a.no_tau_scan, "skip_job_checks": bool(
            a.skip_job_checks or j["overrides"].get("skip_job_checks")),
        "is_teacher": bool(j.get("is_teacher")), "fixed_plan": j.get("fixed_plan"),
        "gpu_slot": slot, "attempt": j["attempt"], "started_at": now_str(),
        "train_files": "train.json", "leak_guard": "passed", "ckpt_every_min": 30,
        "min_train_fraction": 0.3, "kd_alpha": a.kd_alpha, "kd_T": a.kd_T,
        "label_ids": data["label_ids"], "kmax": data["kmax"],
        "loss": j.get("loss", a.loss),
        "lorma_r": 16, "lorma_alpha": 32, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "adalora_init_r": 16, "adalora_target_r": 8, "adalora_alpha": 32,
        "adapter_bottleneck": 64, "partial_top_k": 6, "layer_decay": 0.85,
        "module_md5": module_md5s(),
    }
    if j.get("student"):
        cfg["layer_ids"] = student_layers(j["model"], p1[j["model"]])
        tdir = os.path.join(exp_dir, j["depends"][0])
        cfg["teacher_logits"] = os.path.join(tdir, "teacher_logits.npz")
        if not os.path.exists(cfg["teacher_logits"]):
            # teacher failed: train the pruned student on hard labels (marked in the CSV)
            cfg["technique"] = "student_hard"
            cfg["teacher_failed"] = True
    if j.get("reuse_from"):
        cfg["reuse_from"] = j["reuse_from"]
    return cfg


def copy_from_inputs(a, exp_dir: str, job: str) -> Optional[str]:
    """--resume auto: a finished (or resumable) copy of this job in an attached input."""
    folder = EXPERIMENTS[a.exp]["folder"]
    from cjpe_common import iter_out_dirs
    for od in iter_out_dirs(None, default_input_roots()):
        src = os.path.join(od, folder, job)
        if os.path.exists(os.path.join(src, "DONE.json")) or os.path.exists(
                os.path.join(src, "checkpoints", "last.pt")):
            dst = os.path.join(exp_dir, job)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
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
        ladder.append(("micro_chunks", max(1, a.micro_chunks // 2), "micro-batch halved"))
    if code == 6:
        ladder.append(("skip_job_checks", True, "self-checks skipped"))
    if reducible:
        ladder.append(("K", 2, "K=2"))
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
                mode = j.get("mode", "teacher_reuse" if j.get("reuse_from") else "train")
                env = dict(os.environ, PYTHONPATH=CODE_DIR, PYTHONUNBUFFERED="1")
                if n_gpus():
                    env["CUDA_VISIBLE_DEVICES"] = str(slot)
                cmd = [sys.executable, os.path.join(CODE_DIR, "cjpe_train.py"), "--run-dir", run_dir,
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
            reducible = (j.get("doc_repr", a.doc_repr) in ("hier", "first_k", "last_k")
                         and int(j["overrides"].get("K", j.get("K", a.k))) > 2)
            remaining = t_end - time.time() - reserve
            desc = None
            if j["attempt"] <= a.max_retries and remaining > 15 * 60 and j.get("mode") != \
                    "smoke":
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
            error_analysis(allrows, out_root, getattr(a, "_dev_texts", None), log)
    except Exception as e:
        log(f"   [final] aggregation failed: {e}\n{traceback.format_exc()[-1500:]}")
    write_index(a, out_root, exp_dir, jobs, extra)
    done = [r for r in rows if r.get("status") == "DONE"]
    write_json(os.path.join(exp_dir, "summary.json"),
               {"experiment": a.exp, "finished": now_str(), "rows": rows,
                "output_gb": dir_size_gb(out_root)})
    log(f"   [final] {len(done)}/{len(rows)} jobs DONE; output {dir_size_gb(out_root):.2f} GB")
    smoke = [r for r in rows if r.get("status") == "SMOKE_OK"]
    return 0 if (done or smoke or a.exp == "E8") else 1


def main() -> int:
    global LOG
    a = build_parser().parse_args()
    t_start = time.time()
    t_end = t_start + 3600 * a.time_budget_h
    out_root = a.out_root
    exp_dir = os.path.join(out_root, EXPERIMENTS[a.exp]["folder"])
    os.makedirs(exp_dir, exist_ok=True)
    LOG = os.path.join(exp_dir, "orchestrator.log")
    log(f"=== Phase 4 {a.exp}: {EXPERIMENTS[a.exp]['title']} | {now_str()} | budget "
        f"{a.time_budget_h} h | {n_gpus()} GPU(s) | free disk {free_gb(out_root):.0f} GB ===")
    if a.exp == "E8":
        jobs: List[Dict] = []
        write_index(a, out_root, exp_dir, jobs)
        try:
            from cjpe_data import discover_files, load_split
            f = discover_files(default_input_roots(), log=log)
            if f["source"]:
                a._dev_texts = {d["id"]: d["text"] for d in load_split(f["paths"]["dev"], "dev",
                                                                      log)[0]}
        except Exception as e:
            log(f"   [E8] dev texts unavailable ({e}); n-gram analysis skipped")
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

    # attention mode: PILOT.json if attached, else bidir (Phase-2/3 default)
    attn = a.attn_mode
    if attn is None:
        from cjpe_common import find_files
        pil = find_files("PILOT.json", default_input_roots(), 5)
        pj = read_json(pil[0], {}) if pil else {}
        attn = (pj or {}).get("attn_mode") if (pj or {}).get("attn_mode") in ("bidir",
                                                                             "causal") else "bidir"
        log(f"   [attn] {attn}" + (f" (from {pil[0]})" if pil and pj.get("attn_mode") else
                                   " (default; no PILOT.json attached)"))

    # ---- 2+3. data, audit, leak guard, cache ---------------------------------------------
    from cjpe_tok import needs_for
    nh, nt = needs_for(a.doc_repr, a.k)
    if a.exp in ("E7", "P0"):                              # ablations need up to 4 + 4 chunks
        nh, nt = max(nh, 4 * 511), max(nt, 4 * 511)
    sections = a.exp in ("E7", "P0")
    if a.exp == "E7" and ("repr_full" in a.e7_jobs or "repr_spread" in a.e7_jobs):
        nh = max(nh, 16 * 511)
    try:
        data = prepare_data(a, out_root, bpe_dir, nh, nt, sections)
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

    # E6: reuse a finished E1 teacher with the same fingerprint, if attached
    if a.exp == "E6":
        from cjpe_kd import find_reusable_teacher
        t = jobs[0]
        fp = {"cache_key": data["cache_key"], "doc_repr": a.doc_repr, "K": a.k,
              "attn_mode": attn, "technique": "full_ft"}
        src = find_reusable_teacher(default_input_roots(), MODELS[t["model"]]["folder"], fp)
        if src:
            t["reuse_from"] = src
            log(f"   [kd] reusing the E1 teacher {src}")
        else:
            log(f"   [kd] no matching E1 run attached -> training the teacher "
                f"({t['model']}, full fine-tuning) inside E6")

    # selftest (CPU, tiny models) unless skipped
    if not a.skip_selftest and a.exp in ("P0",):
        try:
            r = subprocess.run([sys.executable, os.path.join(CODE_DIR, "selftest.py"), "--out",
                                os.path.join(out_root, "SELFTEST.json")], cwd=CODE_DIR,
                               env=dict(os.environ, PYTHONPATH=CODE_DIR), timeout=1200)
            log(f"   [selftest] exit {r.returncode}")
        except Exception as e:
            log(f"   [selftest] could not run: {e}")

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
