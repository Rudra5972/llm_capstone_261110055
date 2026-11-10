"""
select_best.py -- best-of-4 per experiment, and the model-coverage guard.

    python select_best.py --exp E1                 # (re)build OUT/E1_full_ft/BEST + summary
    python select_best.py --coverage               # which model won each notebook + extra uploads

Selection rule (section 6.1): highest dev strict macro-F1 (all 12 labels); ties -> fewer
trainable parameters. BEST/ gets a copy of the winner's submission.zip, best.pt, run_config.json
and DONE.json, plus SELECTION.json.

Coverage (section 10): the six submissions must involve different Phase-1 models. If the six
winners do not include all four models, list for each missing model its best submission across
techniques, to upload in addition.
"""
import argparse
import csv
import os
import shutil

from ner_common import (EXPERIMENTS, MODELS, REQUIRED_EXPS, default_input_roots,
                        iter_finished_runs, read_json, write_json)


def _model_rows(exp_dir):
    rows = []
    for mid, m in MODELS.items():
        d = os.path.join(exp_dir, m["folder"])
        done = read_json(os.path.join(d, "DONE.json"))
        if not done:
            continue
        tr = done.get("train") or {}
        rows.append({"model": mid, "arch": m["arch"], "dir": d,
                     "dev_macro_f1": done["dev_macro_f1"],
                     "dev_macro_precision": done.get("dev_macro_precision"),
                     "dev_macro_recall": done.get("dev_macro_recall"),
                     "dev_micro_f1": done.get("dev_micro_f1"),
                     "trainable_params": done.get("trainable_params"),
                     "total_params": done.get("total_params"),
                     "tokens_per_s": tr.get("tokens_per_s"),
                     "train_minutes": (tr.get("train_seconds") or 0) / 60,
                     "peak_mem_gb": tr.get("peak_mem_gb"),
                     "epochs_run": tr.get("epochs_run"), "best_epoch": tr.get("best_epoch")})
    return rows


def _key(r):
    """The selection rule: dev strict macro-F1, ties -> fewer trainable parameters."""
    return (r["dev_macro_f1"], -(r["trainable_params"] or 0))


def select(exp_dir, log=print):
    rows = _model_rows(exp_dir)
    if rows:
        cols = list(rows[0].keys())
        with open(os.path.join(exp_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        write_json(os.path.join(exp_dir, "summary.json"), rows)
    if not rows:
        log("   no finished models yet -- nothing to select")
        return None
    win = max(rows, key=_key)
    best = os.path.join(exp_dir, "BEST")
    os.makedirs(best, exist_ok=True)
    src = win["dir"]
    for rel, dst in (("submission/submission.zip", "submission.zip"),
                     ("checkpoints/best.pt", "best.pt"),
                     ("logs/run_config.json", "run_config.json"),
                     ("DONE.json", "DONE.json")):
        p = os.path.join(src, rel)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(best, dst))
    sel = {"winner": win["model"], "arch": win["arch"], "dev_macro_f1": win["dev_macro_f1"],
           "from": src, "n_models_finished": len(rows),
           "table": sorted(({k: r[k] for k in ("model", "dev_macro_f1", "trainable_params")}
                            for r in rows), key=lambda r: -r["dev_macro_f1"])}
    write_json(os.path.join(best, "SELECTION.json"), sel)
    log(f"   BEST of {len(rows)} -> {win['model']} ({win['arch']}), dev macro-F1 "
        f"{win['dev_macro_f1']:.4f}")
    return sel


def _exp_dirs(out_root, input_roots):
    """{exp_folder: {experiment dirs}} for every finished run found."""
    found = {}
    for exp_f, _mod_f, run_dir in iter_finished_runs(out_root, input_roots):
        found.setdefault(exp_f, set()).add(os.path.dirname(run_dir))
    return found


def coverage(out_root, input_roots, log=print):
    folders = {EXPERIMENTS[e]["folder"]: e for e in REQUIRED_EXPS}
    found = _exp_dirs(out_root, input_roots)
    winners, per_model_best = {}, {}
    for folder, exp in folders.items():
        best_row = None
        for d in sorted(found.get(folder, [])):
            for r in _model_rows(d):
                r["exp"] = exp
                r["zip"] = os.path.join(r["dir"], "submission", "submission.zip")
                if best_row is None or _key(r) > _key(best_row):
                    best_row = r
                pm = per_model_best.get(r["model"])
                if pm is None or _key(r) > _key(pm):
                    per_model_best[r["model"]] = r
        winners[exp] = best_row
    log("Winners (one upload per notebook):")
    for exp in REQUIRED_EXPS:
        w = winners[exp]
        log(f"   {exp}: " + (f"{w['model']} dev macro-F1 {w['dev_macro_f1']:.4f}  ->  {w['zip']}"
                             if w else "not finished yet"))
    used = {w["model"] for w in winners.values() if w}
    missing = [m for m in MODELS if m not in used]
    extra = []
    if missing:
        log(f"\nModels not represented among the winners: {missing}. Also upload:")
        for m in missing:
            r = per_model_best.get(m)
            if r:
                log(f"   {m}: its best run is {r['exp']} (dev macro-F1 {r['dev_macro_f1']:.4f}) "
                    f"->  {r['zip']}")
                extra.append({"model": m, "exp": r["exp"], "zip": r["zip"]})
            else:
                log(f"   {m}: no finished run anywhere yet")
    else:
        log("\nAll four Phase-1 models are represented among the winners.")
    n = sum(1 for w in winners.values() if w) + len(extra)
    log(f"\nPlanned uploads: {n} (max 10 per day, 40 per phase; each takes 15-30 min to score)")
    return {"winners": winners, "extra_uploads": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default="/kaggle/working/phase2_out")
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    if a.coverage:
        res = coverage(a.out, default_input_roots())
        write_json(os.path.join(a.out, "coverage.json"), res)
        return
    folder = EXPERIMENTS.get(a.exp, {}).get("folder", a.exp)
    select(os.path.join(a.out, folder))


if __name__ == "__main__":
    main()
