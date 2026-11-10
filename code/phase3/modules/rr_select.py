"""
rr_select.py -- best-of-4 per experiment, and the rule / model-coverage check.

    python rr_select.py --exp E1                 # (re)build OUT/E1_lorma/BEST + summary
    python rr_select.py --coverage               # winners, rule compliance, extra uploads

Selection rule: highest dev POOLED Macro-F1 (argmax, before post-hoc bias: the same unbiased
number for every run); ties -> fewer trainable parameters. Only runs with a validated
submission covering CL + IT are eligible (KD teachers and partial-domain ablations are not).
BEST/ receives the winner's submission.zip, best.pt, run_config.json, DONE.json, dev report,
confusion.csv and raw logits, plus SELECTION.json.

Coverage: the brief requires all four Phase-1 models to be used and >= 6 experiments with
>= 5 PEFT techniques (LoRMA compulsory) and >= 1 KD. If the winners do not include every
model, the best submission of each missing model is listed as an extra upload.

Inputs : OUT/<exp>/<job>/DONE.json (+ attached phase3_out folders for --coverage)
Outputs: BEST/, SELECTION.json, summary.csv/json, coverage.json
"""
import argparse
import csv
import os
import shutil
from typing import Dict, List, Optional

from rr_common import (EXPERIMENTS, MODELS, OUT_ROOT, SPEC, TECHNIQUES, default_input_roots,
                       iter_finished_runs, read_json, write_json)


def _rows(exp_dir: str) -> List[Dict]:
    rows = []
    if not os.path.isdir(exp_dir):
        return rows
    for job in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, job)
        done = read_json(os.path.join(d, "DONE.json"))
        if job == "BEST" or not done:
            continue
        eff = done.get("efficiency") or {}
        rows.append({"job": job, "model": done.get("model"), "technique": done.get("technique"),
                     "variant": done.get("variant"), "dir": d,
                     "dev_macro_f1": done["dev_macro_f1"],
                     "dev_cl_f1": (done.get("dev_per_domain_macro_f1") or {}).get("CL"),
                     "dev_it_f1": (done.get("dev_per_domain_macro_f1") or {}).get("IT"),
                     "dev_macro_precision": done.get("dev_macro_precision"),
                     "dev_macro_recall": done.get("dev_macro_recall"),
                     "trainable_params": eff.get("trainable_params"),
                     "total_params": eff.get("total_params"),
                     "train_minutes": eff.get("train_minutes"),
                     "submission_valid": bool(done.get("submission_valid")),
                     "eligible": bool(done.get("submission_valid")) and not job.startswith(
                         "teacher_")})
    return rows


def _key(r: Dict):
    return (r["dev_macro_f1"], -(r["trainable_params"] or 0))


def select(exp_dir: str, log=print) -> Optional[Dict]:
    rows = _rows(exp_dir)
    if rows:
        with open(os.path.join(exp_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        write_json(os.path.join(exp_dir, "summary.json"), rows)
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        log("   no finished run with a valid submission yet -- nothing to select")
        return None
    win = max(elig, key=_key)
    best = os.path.join(exp_dir, "BEST")
    if os.path.isdir(best):
        shutil.rmtree(best)
    os.makedirs(best)
    src = win["dir"]
    for rel, dst in (("submission/submission.zip", "submission.zip"),
                     ("submission/prediction.json", "prediction.json"),
                     ("checkpoints/best.pt", "best.pt"), ("logs/run_config.json", "run_config.json"),
                     ("logs/train_summary.json", "train_summary.json"), ("DONE.json", "DONE.json"),
                     ("dev/dev_report.json", "dev_report.json"), ("dev/confusion.csv",
                                                                  "confusion.csv")):
        p = os.path.join(src, rel)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(best, dst))
    if os.path.isdir(os.path.join(src, "logits")):
        shutil.copytree(os.path.join(src, "logits"), os.path.join(best, "logits"))
    sel = {"winner": win["job"], "model": win["model"], "technique": win["technique"],
           "dev_macro_f1": win["dev_macro_f1"], "from": src, "n_eligible": len(elig),
           "rule": "max dev pooled Macro-F1 (no bias), ties -> fewer trainable params",
           "table": sorted(({k: r[k] for k in ("job", "model", "dev_macro_f1", "dev_cl_f1",
                                               "dev_it_f1", "trainable_params")} for r in elig),
                           key=lambda r: -r["dev_macro_f1"])}
    write_json(os.path.join(best, "SELECTION.json"), sel)
    log(f"   BEST of {len(elig)} -> {win['job']} ({win['technique']}), dev Macro-F1 "
        f"{win['dev_macro_f1']:.4f}")
    return sel


def coverage(out_root: str, input_roots, log=print) -> Dict:
    folders = {v["folder"]: k for k, v in EXPERIMENTS.items()}
    found: Dict[str, set] = {}
    for exp_f, _job, run_dir in iter_finished_runs(out_root, input_roots):
        found.setdefault(exp_f, set()).add(os.path.dirname(run_dir))
    winners, per_model = {}, {}
    for folder, exp in folders.items():
        best = None
        for d in sorted(found.get(folder, [])):
            for r in _rows(d):
                if not r["eligible"]:
                    continue
                r["exp"] = exp
                r["zip"] = os.path.join(r["dir"], "submission", "submission.zip")
                if best is None or _key(r) > _key(best):
                    best = r
                pm = per_model.get(r["model"])
                if pm is None or _key(r) > _key(pm):
                    per_model[r["model"]] = r
        if best:
            winners[exp] = best
    log("Winners (one upload per notebook):")
    for exp, spec in EXPERIMENTS.items():
        w = winners.get(exp)
        log(f"   {exp:>4} {spec['title'][:40]:<40} " +
            (f"{w['model']} dev F1 {w['dev_macro_f1']:.4f} -> {w['zip']}" if w else "-"))
    techs = {winners[e]["technique"] for e in winners if EXPERIMENTS[e]["counts"]}
    peft = sorted(t for t in techs if TECHNIQUES.get(t, {}).get("peft"))
    rules = SPEC["rules"]
    n_exp = sum(1 for e in winners if EXPERIMENTS[e]["counts"])
    compliance = {
        "experiments_counted": n_exp, "min_experiments": rules["min_experiments"],
        "peft_techniques": peft, "min_peft_techniques": rules["min_peft_techniques"],
        "lorma_done": "lorma" in techs, "kd_done": "kd" in techs,
        "ok": (n_exp >= rules["min_experiments"] and len(peft) >= rules["min_peft_techniques"]
               and "lorma" in techs and "kd" in techs)}
    log(f"\nRule check: {compliance}")
    used = {w["model"] for w in winners.values()}
    extra = []
    missing = [m for m in MODELS if m not in used]
    if missing:
        log(f"Models not represented among the winners: {missing}. Also upload:")
        for m in missing:
            r = per_model.get(m)
            if r:
                log(f"   {m}: best run {r['exp']}/{r['job']} (dev F1 {r['dev_macro_f1']:.4f}) -> "
                    f"{r['zip']}")
                extra.append({"model": m, "exp": r["exp"], "zip": r["zip"]})
            else:
                log(f"   {m}: no finished run anywhere yet")
    n = len(winners) + len(extra)
    log(f"\nPlanned uploads: {n} (max {SPEC['submissions']['max_per_day']} per day, "
        f"{SPEC['submissions']['max']} per phase; scoring takes 15-30 min each)")
    return {"winners": winners, "extra_uploads": extra, "compliance": compliance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    if a.coverage:
        write_json(os.path.join(a.out, "coverage.json"), coverage(a.out, default_input_roots()))
        return
    folder = EXPERIMENTS.get(a.exp, {}).get("folder", a.exp)
    select(os.path.join(a.out, folder))


if __name__ == "__main__":
    main()
