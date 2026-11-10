"""
aggregate.py -- the (technique x model) results table for the report.

    python aggregate.py [--out /kaggle/working/phase2_out] [--codabench scores.json]

Reads every phase2_out/*/*/DONE.json under --out and the attached input datasets, and writes
results.csv / results.md (+ per-label F1 table). `scores.json` is optional and filled in by
hand: {"E1/M4": 0.61, ...} with the Codabench strict macro-F1 of the uploads.
"""
import argparse
import csv
import os

from ner_common import EXPERIMENTS, MODELS, default_input_roots, iter_finished_runs, read_json

FOLDER_TO_EXP = {v["folder"]: k for k, v in EXPERIMENTS.items()}
FOLDER_TO_MODEL = {v["folder"]: k for k, v in MODELS.items()}


def collect(out_root, input_roots):
    rows = []
    for exp_folder, model_folder, run_dir in iter_finished_runs(out_root, input_roots):
        d = read_json(os.path.join(run_dir, "DONE.json"))
        rep = read_json(os.path.join(run_dir, "dev", "dev_report.json"), {})
        tr = d.get("train") or {}
        row = {"exp": FOLDER_TO_EXP.get(exp_folder, exp_folder),
               "technique": d.get("technique"), "model": FOLDER_TO_MODEL.get(model_folder, d.get("model")),
               "arch": d.get("arch"), "dev_macro_f1": d.get("dev_macro_f1"),
               "dev_macro_p": d.get("dev_macro_precision"), "dev_macro_r": d.get("dev_macro_recall"),
               "dev_micro_f1": d.get("dev_micro_f1"), "trainable_params": d.get("trainable_params"),
               "total_params": d.get("total_params"),
               "trainable_pct": (100 * d["trainable_params"] / d["total_params"]
                                 if d.get("trainable_params") and d.get("total_params") else None),
               "train_min": (tr.get("train_seconds") or 0) / 60, "epochs": tr.get("epochs_run"),
               "best_epoch": tr.get("best_epoch"), "train_tok_s": tr.get("tokens_per_s"),
               "infer_tok_s": d.get("inference_tokens_per_s"), "peak_mem_gb": tr.get("peak_mem_gb"),
               "decoding": f"{(d.get('decoding') or {}).get('mode')} "
                           f"{(d.get('decoding') or {}).get('o_shift')}"}
        per = (rep.get("metrics") or {}).get("per_label", {})
        for lab, v in per.items():
            row[f"f1_{lab}"] = v["f1"]
        rows.append(row)
    rows.sort(key=lambda r: (r["exp"], r["model"] or ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/phase2_out")
    ap.add_argument("--codabench", default=None)
    a = ap.parse_args()
    rows = collect(a.out, default_input_roots())
    cb = read_json(a.codabench, {}) if a.codabench else {}
    for r in rows:
        r["codabench_macro_f1"] = cb.get(f"{r['exp']}/{r['model']}")
    if not rows:
        print("no finished runs found")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    main_cols = ["exp", "technique", "model", "dev_macro_f1", "dev_macro_p", "dev_macro_r",
                 "trainable_params", "trainable_pct", "train_min", "peak_mem_gb", "infer_tok_s",
                 "codabench_macro_f1"]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if v < 10 else f"{v:,.1f}"
        if isinstance(v, int):
            return f"{v:,}"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(main_cols) + " |", "|" + "---|" * len(main_cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in main_cols) + " |")
    labs = [c for c in cols if c.startswith("f1_")]
    if labs:
        lines += ["", "Per-label dev F1", "", "| exp | model | " + " | ".join(l[3:] for l in labs) + " |",
                  "|" + "---|" * (len(labs) + 2)]
        for r in rows:
            lines.append(f"| {r['exp']} | {r['model']} | " + " | ".join(fmt(r.get(l)) for l in labs) + " |")
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(rows) + 2]))
    print(f"\nwritten: {os.path.join(a.out, 'results.csv')} and results.md")


if __name__ == "__main__":
    main()
