"""
select_best.py -- per-experiment BEST selection and the six-upload plan (sections 11.2, 11.3).

    python select_best.py --exp-dir /kaggle/working/phase5_out/E2_lorma      # BEST/ for one exp
    python select_best.py --plan [--out-root ...]                             # UPLOAD_PLAN.csv

BEST: highest dev macro-F1 among DONE jobs (not DEGENERATE, not the E6 teacher); ties -> fewer
trainable parameters.

Plan: exactly one row per counted experiment E1..E6, every Phase-1 model at least once (relaxed
only when impossible, and then reported), FAILED / DEGENERATE / missing-zip rows excluded,
maximise the mean dev macro-F1, and the six prediction.json files must be pairwise different
(identical submissions do not count on Codabench). Exhaustive search (<= 4^6 combinations).
"""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import zipfile
from typing import Dict, List, Optional

from cjpe_common import (COUNTED, EXPERIMENTS, MODELS, OUT_ROOT, default_input_roots,
                         read_csv, read_json, write_csv, write_json)


def _f(x, default=-1.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def prediction_hash(zip_path: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(zip_path) as z:
            items = json.loads(z.read("prediction.json"))
        m = sorted((str(r["id"]), sorted(int(x) for x in r["pred"])) for r in items)
        return hashlib.md5(json.dumps(m).encode()).hexdigest()
    except Exception:
        return None


def is_candidate(r: Dict, require_technique: Optional[str] = None) -> bool:
    if r.get("status") != "DONE":
        return False
    if r.get("technique", "").endswith("(teacher)"):
        return False
    if require_technique and r.get("technique") != require_technique:
        return False
    z = r.get("submission_zip")
    return bool(z) and os.path.exists(z)


def select_exp(exp_dir: str, rows: List[Dict], log=print) -> Optional[Dict]:
    exp = os.path.basename(exp_dir.rstrip("/"))
    cands = [r for r in rows if is_candidate(r)]
    best_dir = os.path.join(exp_dir, "BEST")
    if not cands:
        log(f"   [select] {exp}: no finished, non-degenerate model -> no BEST/")
        degen = [r for r in rows if r.get("status") == "DEGENERATE" and r.get("submission_zip")
                 and os.path.exists(r["submission_zip"])]
        if not degen:
            return None
        log(f"   [select] {exp}: only DEGENERATE models exist; keeping the best of them in BEST/ "
            f"(flagged)")
        cands = degen
    cands.sort(key=lambda r: (-_f(r.get("dev_macro_f1")), _f(r.get("trainable_params"), 1e18)))
    w = cands[0]
    os.makedirs(best_dir, exist_ok=True)
    shutil.copy2(w["submission_zip"], os.path.join(best_dir, "submission.zip"))
    run_dir = os.path.dirname(os.path.dirname(w["submission_zip"]))
    rc = os.path.join(run_dir, "logs", "run_config.json")
    if os.path.exists(rc):
        shutil.copy2(rc, os.path.join(best_dir, "run_config.json"))
    sel = {"experiment": w.get("experiment"), "job": w.get("job"), "model_id": w.get("model_id"),
           "technique": w.get("technique"), "dev_macro_f1": _f(w.get("dev_macro_f1")),
           "status": w.get("status"), "run_dir": run_dir,
           "ranking": [{"job": r.get("job"), "model_id": r.get("model_id"),
                        "dev_macro_f1": _f(r.get("dev_macro_f1")),
                        "trainable_params": r.get("trainable_params")} for r in cands]}
    write_json(os.path.join(best_dir, "SELECTION.json"), sel)
    log(f"   [select] {exp}: BEST = {w.get('job')} (dev macro-F1 {_f(w.get('dev_macro_f1')):.4f})")
    return sel


def plan_uploads(rows: List[Dict], out_path: str, log=print) -> Dict:
    by_exp: Dict[str, List[Dict]] = {}
    for e in COUNTED:
        tech = EXPERIMENTS[e]["technique"]
        c = [r for r in rows if r.get("experiment") == e and is_candidate(r, tech)]
        for r in c:
            r["_hash"] = prediction_hash(r["submission_zip"])
        by_exp[e] = sorted([r for r in c if r["_hash"]],
                           key=lambda r: -_f(r.get("dev_macro_f1")))
    missing = [e for e in COUNTED if not by_exp[e]]
    exps = [e for e in COUNTED if by_exp[e]]
    best, best_key, relaxed = None, None, False
    # last resort (Phase 5): if no combination has pairwise different predictions (e.g. weak
    # models all predicting the most frequent statute), still plan one upload per experiment
    for require_all, distinct in ((True, True), (False, True), (True, False), (False, False)):
        for combo in itertools.product(*[by_exp[e] for e in exps]):
            hashes = [r["_hash"] for r in combo]
            if distinct and len(set(hashes)) != len(hashes):
                continue
            models = {r.get("model_id") for r in combo}
            if require_all and len(models) < min(4, len(exps)):
                continue
            key = (len(models), sum(_f(r.get("dev_macro_f1")) for r in combo))
            if best is None or key > best_key:
                best, best_key = combo, key
        if best is not None:
            relaxed = not (require_all and distinct)
            break
    plan_rows = []
    if best:
        first = [r for r in best if r.get("experiment") in ("E1", "E2")]
        ordered = first[:1] + [r for r in best if r not in first[:1]]
        for k, r in enumerate(ordered, 1):
            plan_rows.append({"order": k, "experiment": r.get("experiment"),
                              "technique": r.get("technique"), "model_id": r.get("model_id"),
                              "job": r.get("job"), "dev_macro_f1": r.get("dev_macro_f1"),
                              "path": r.get("submission_zip"), "role": "required"})
        used = {r.get("submission_zip") for r in best}
        used_h = {r["_hash"] for r in best}
        extra = 0
        for m in MODELS:
            pool = [r for e in exps for r in by_exp[e] if r.get("model_id") == m
                    and r.get("submission_zip") not in used and r["_hash"] not in used_h]
            if pool and extra < 4:
                r = max(pool, key=lambda r: _f(r.get("dev_macro_f1")))
                extra += 1
                used_h.add(r["_hash"])
                plan_rows.append({"order": len(plan_rows) + 1, "experiment": r.get("experiment"),
                                  "technique": r.get("technique"), "model_id": m,
                                  "job": r.get("job"), "dev_macro_f1": r.get("dev_macro_f1"),
                                  "path": r.get("submission_zip"), "role": "optional extra"})
    write_csv(out_path, plan_rows, ["order", "experiment", "technique", "model_id", "job",
                                    "dev_macro_f1", "role", "path"])
    models = sorted({r["model_id"] for r in plan_rows if r["role"] == "required"})
    rep = {"n_required": sum(1 for r in plan_rows if r["role"] == "required"),
           "missing_experiments": missing, "models_covered": models,
           "all_models_covered": len(models) == 4, "relaxed_model_constraint": relaxed}
    log("   [plan] upload order (Codabench, <= 10/day):")
    for r in plan_rows:
        log(f"   [plan] {r['order']:>2}. {r['experiment']} {r['technique']:<11} {r['model_id']} "
            f"dev F1 {_f(r['dev_macro_f1']):.4f} [{r['role']}]  {r['path']}")
    if missing:
        log(f"   [plan] MISSING experiments (no valid submission yet): {missing}")
    if plan_rows and not rep["all_models_covered"]:
        log(f"   [plan] WARNING: only models {models} appear among the six uploads")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT)
    a = ap.parse_args()
    if a.exp_dir:
        f = [p for p in os.listdir(a.exp_dir) if p.startswith("results_") and p.endswith(".csv")]
        rows = read_csv(os.path.join(a.exp_dir, f[0])) if f else []
        select_exp(a.exp_dir, rows)
    if a.plan:
        from aggregate import aggregate
        rows = aggregate(a.out_root, default_input_roots())
        plan_uploads(rows, os.path.join(a.out_root, "UPLOAD_PLAN.csv"))


if __name__ == "__main__":
    main()
