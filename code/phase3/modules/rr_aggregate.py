"""
rr_aggregate.py -- the report tables and README_PHASE3.md, generated from the run outputs.

    python rr_aggregate.py [--out /kaggle/working/phase3_out] [--codabench scores.json]

Reads every phase3_out/<exp>/<job>/DONE.json (here and in attached inputs) and writes:
  RESULTS_ALL.json / results_all.csv   one row per finished job
  REPORT_TABLES.md   technique x model Macro-F1 (pooled, CL, IT); per-class F1 + confusion of
                     the best model; context / head / loss / domain ablations (E10); PEFT vs
                     full-FT gap; KD teacher vs student; efficiency (params, memory, speed)
  README_PHASE3.md   architecture, methodology, how to run, results, rule compliance, citations
`scores.json` (optional, filled by hand): {"E1/M4_bloom_alibi": 0.61, ...} Codabench scores.
"""
import argparse
import csv
import os
from typing import Dict, List, Optional

from rr_common import (EXPERIMENTS, MODELS, OUT_ROOT, SPEC, TECHNIQUES, default_input_roots,
                       iter_finished_runs, read_json, write_json)

FOLDER_TO_EXP = {v["folder"]: k for k, v in EXPERIMENTS.items()}
MODEL_FOLDERS = {v["folder"]: k for k, v in MODELS.items()}

CITATIONS = [
    "Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.",
    "Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. "
    "arXiv:2305.14314.",
    "Zhang et al. (2023). Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning "
    "(AdaLoRA). ICLR. arXiv:2303.10512.",
    "Kalajdzievski (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA "
    "(rsLoRA). arXiv:2312.03732.",
    "Bihany, Patel & Modi (2025). LoRMA: Low-Rank Multiplicative Adaptation for LLMs. "
    "arXiv:2506.07621.",
    "Li & Liang (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL. "
    "arXiv:2101.00190.",
    "Lester, Al-Rfou & Constant (2021). The Power of Scale for Parameter-Efficient Prompt "
    "Tuning. EMNLP. arXiv:2104.08691.",
    "Liu et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than "
    "In-Context Learning (IA3). NeurIPS. arXiv:2205.05638.",
    "Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. "
    "arXiv:1503.02531.",
    "Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV (focal loss ablation).",
    "Menon et al. (2021). Long-tail learning via logit adjustment. ICLR (logit-adjust ablation).",
]


def abbrev(label: str) -> str:
    """Unique short column names: capitals of CamelCase labels (PrecedentNotReliedUpon -> PNRU)."""
    caps = "".join(c for c in label if c.isupper())
    return caps if len(caps) > 1 else label[:5]


def fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return "" if v is None else str(v)


def table(cols: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(x) for x in r) + " |")
    return out


def collect(out_root: str, roots: List[str]) -> List[Dict]:
    rows = []
    for exp_f, job_f, run_dir in iter_finished_runs(out_root, roots):
        d = read_json(os.path.join(run_dir, "DONE.json"))
        e = d.get("efficiency") or {}
        pd = d.get("dev_per_domain_macro_f1") or {}
        rows.append({"exp": FOLDER_TO_EXP.get(exp_f, exp_f), "job": job_f, "dir": run_dir,
                     "technique": d.get("technique"), "model": d.get("model"),
                     "variant": d.get("variant"), "dev_macro_f1": d.get("dev_macro_f1"),
                     "dev_cl": pd.get("CL"), "dev_it": pd.get("IT"),
                     "dev_domain_avg": d.get("dev_domain_avg_macro_f1"),
                     "dev_p": d.get("dev_macro_precision"), "dev_r": d.get("dev_macro_recall"),
                     "bias_applied": d.get("bias_applied"),
                     "submission_valid": d.get("submission_valid"),
                     "total_params": e.get("total_params"),
                     "trainable_params": e.get("trainable_params"),
                     "trainable_pct": e.get("trainable_pct"),
                     "train_minutes": e.get("train_minutes"),
                     "train_peak_mem_gb": e.get("train_peak_mem_gb"),
                     "train_tok_s": e.get("train_tokens_per_s"),
                     "infer_tok_s": e.get("infer_tokens_per_s"),
                     "infer_peak_mem_gb": e.get("infer_peak_mem_gb"),
                     "kd_report": d.get("kd_report"), "peft_report": d.get("peft_report")})
    rows.sort(key=lambda r: (r["exp"], r["job"]))
    return rows


def build_report(out_root: str, roots: List[str], codabench: Optional[str] = None,
                 log=print) -> Dict:
    rows = collect(out_root, roots)
    if not rows:
        log("   no finished runs found -- nothing to aggregate")
        return {}
    cb = read_json(codabench, {}) if codabench else {}
    for r in rows:
        r["codabench"] = cb.get(f"{r['exp']}/{r['job']}")
    write_json(os.path.join(out_root, "RESULTS_ALL.json"), rows)
    flat = [{k: v for k, v in r.items() if k not in ("kd_report", "peft_report")} for r in rows]
    with open(os.path.join(out_root, "results_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    main = [r for r in rows if r["job"] in MODEL_FOLDERS and r["exp"] in EXPERIMENTS
            and EXPERIMENTS[r["exp"]]["technique"] not in ("ablation", "ensemble")]
    L: List[str] = ["# Phase-3 report tables (generated by rr_aggregate.py)", ""]
    # 1. technique x model
    L += ["## Dev Macro-F1 per technique and model (pooled / CL / IT)", ""]
    techs = [e for e in EXPERIMENTS if EXPERIMENTS[e]["technique"] not in ("ablation",
                                                                           "ensemble")]
    trs = []
    for e in techs:
        cells = [e, EXPERIMENTS[e]["title"]]
        for mid in MODELS:
            r = next((x for x in main if x["exp"] == e and x["model"] == mid), None)
            cells.append(f"{r['dev_macro_f1']:.4f} / {r['dev_cl'] or 0:.3f} / "
                         f"{r['dev_it'] or 0:.3f}" if r else "")
        trs.append(cells)
    L += table(["exp", "technique"] + [f"{m} ({MODELS[m]['arch']})" for m in MODELS], trs) + [""]
    # 2. best model: per-class F1 + confusion
    elig = [r for r in main if r["submission_valid"] and EXPERIMENTS[r["exp"]]["counts"]]
    best = max(elig, key=lambda r: r["dev_macro_f1"]) if elig else None
    if best:
        rep = read_json(os.path.join(best["dir"], "dev", "dev_report.json"), {})
        m = (rep.get("metrics") or {})
        per = (m.get("pooled") or {}).get("per_class", {})
        L += [f"## Best counted model: {best['exp']} / {best['job']} (dev pooled Macro-F1 "
              f"{best['dev_macro_f1']:.4f})", "", "Per-class dev scores:", ""]
        L += table(["label", "P", "R", "F1", "support", "predicted"],
                   [[k, v["precision"], v["recall"], v["f1"], v["support"], v["predicted"]]
                    for k, v in per.items()]) + [""]
        if m.get("confusion"):
            labs = m["labels"]
            L += ["Confusion matrix (rows gold, columns predicted):", ""]
            L += table(["gold \\ pred"] + [abbrev(l) for l in labs],
                       [[labs[i]] + list(row) for i, row in enumerate(m["confusion"])]) + [""]
    # 3. ablations
    abl = [r for r in rows if r["exp"] == "E10"]
    if abl:
        L += ["## Ablations (E10)", ""]
        L += table(["job", "technique", "model", "variant", "pooled F1", "CL", "IT"],
                   [[r["job"], r["technique"], r["model"], r["variant"], r["dev_macro_f1"],
                     r["dev_cl"], r["dev_it"]] for r in abl]) + [""]
    # 4. PEFT vs full FT
    ft = {r["model"]: r["dev_macro_f1"] for r in main if r["technique"] == "full_ft"}
    if ft:
        L += ["## PEFT vs full fine-tuning gap (dev pooled Macro-F1, technique - full FT)", ""]
        gap_rows = []
        for e in techs:
            if EXPERIMENTS[e]["technique"] in ("full_ft",):
                continue
            cells = [e, EXPERIMENTS[e]["technique"]]
            for mid in MODELS:
                r = next((x for x in main if x["exp"] == e and x["model"] == mid), None)
                cells.append(f"{r['dev_macro_f1'] - ft[mid]:+.4f}" if r and mid in ft else "")
            gap_rows.append(cells)
        L += table(["exp", "technique"] + list(MODELS), gap_rows) + [""]
    # 5. KD
    kd = [r for r in main if r["kd_report"]]
    if kd:
        L += ["## Knowledge distillation: teacher (full FT) vs pruned student", ""]
        L += table(["model", "teacher F1", "student F1", "F1 retained", "teacher params",
                    "student params", "size ratio", "speed-up", "memory ratio"],
                   [[r["model"], r["kd_report"]["teacher_dev_macro_f1"],
                     r["kd_report"]["student_dev_macro_f1"], r["kd_report"]["f1_retained"],
                     r["kd_report"]["teacher_params"], r["kd_report"]["student_params"],
                     r["kd_report"]["size_ratio"], r["kd_report"]["speedup"],
                     r["kd_report"].get("memory_ratio")] for r in kd]) + [""]
    # 6. efficiency
    L += ["## Efficiency", ""]
    L += table(["exp", "job", "technique", "total params", "trainable", "trainable %",
                "train min", "train peak GB", "train tok/s", "infer tok/s"],
               [[r["exp"], r["job"], r["technique"], r["total_params"], r["trainable_params"],
                 r["trainable_pct"], r["train_minutes"], r["train_peak_mem_gb"], r["train_tok_s"],
                 r["infer_tok_s"]] for r in main]) + [""]
    # 7. AdaLoRA ranks
    ada = [r for r in main if (r["peft_report"] or {}).get("adalora_rank_per_layer")]
    for r in ada:
        L += [f"AdaLoRA final rank per layer ({r['model']}): "
              f"{r['peft_report']['adalora_rank_per_layer']}", ""]
    # 8. compliance
    from rr_select import coverage
    cov = coverage(out_root, roots, log=lambda *x: None)
    L += ["## Rule compliance", "", f"`{cov['compliance']}`", ""]
    with open(os.path.join(out_root, "REPORT_TABLES.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    write_readme(out_root, L, cov)
    log(f"   report written: {os.path.join(out_root, 'REPORT_TABLES.md')} and README_PHASE3.md "
        f"({len(rows)} finished jobs)")
    return {"rows": len(rows), "best": best["job"] if best else None}


def write_readme(out_root: str, tables: List[str], cov: Dict) -> None:
    R = ["# CS-781 Capstone -- Phase 3: Rhetorical Role Prediction", "",
         "Sentence-level rhetorical-role classification of Indian legal judgments (Corporate Law "
         "and IT domains, 13 labels) with the four Phase-1 foundation models (M1 GPT-2 style, M2 "
         "Llama style, M3 Mistral style with sliding-window attention, M4 BLOOM style with ALiBi) "
         "and parameter-efficient fine-tuning implemented from scratch in plain PyTorch.", "",
         "## Architecture", "",
         "* `RRModel` = Phase-1 trunk (embeddings -> blocks -> final norm, LM head dropped) + "
         "sentence head. Documents are packed into <= 1024-token windows; every sentence ends "
         "with the EOS token, whose hidden state is the sentence vector; a domain prefix "
         "(`[CL]`/`[IT]`) starts each window. Long documents use overlapping windows; each "
         "sentence is predicted from the window where it has the most context.",
         "* Heads: LayerNorm -> Dropout -> Linear(768, 13) (default); a BiGRU over the sentence "
         "sequence (`seqctx`) in the ablation.",
         "* Loss: inverse-sqrt-frequency weighted cross-entropy (Macro-F1 punishes ignoring rare "
         "labels); focal / logit-adjusted / plain CE in the ablation.",
         "* Post-hoc per-class logit biases, applied only when a 2-fold held-out check on dev "
         "shows a gain.", "",
         "## Techniques", ""]
    for t, v in TECHNIQUES.items():
        R.append(f"* **{t}** -- {v['title']}" + ("" if v["peft"] or t == "kd"
                                                 else " (reference, not counted)"))
    R += ["", "rsLoRA shares the LoRA module (only the scale alpha/sqrt(r) and the rank differ) "
          "but is listed by the brief as its own technique, so it runs as its own experiment.",
          "QLoRA is simulated without bitsandbytes: block-wise NF4 storage (block 64, double-"
          "quantised scales) with on-the-fly de-quantisation.", "",
          "## How to run", "",
          "1. `00_data_tests_pilot`: data gates (G0-G2), CPU selftest, causal-vs-bidir pilot -> "
          "`PILOT.json`.",
          "2. E1, E2, E6 (priority 1) then E3, E4, E5 (priority 2), then E7, E8, E9, E0, E10 "
          "(priority 3) -- each is an independent Kaggle notebook (T4 x2): "
          "`python run_experiment.py --exp E<k>`.",
          "3. E11: `python run_experiment.py --exp E11` (ensemble + this report) with the other "
          "outputs attached.",
          "4. Upload `phase3_out/<exp>/BEST/submission.zip` for each experiment.", "",
          "## Results", ""]
    R += tables[2:]
    R += ["", "## Rule compliance", "", f"* experiments counted: "
          f"{cov['compliance']['experiments_counted']} (min {SPEC['rules']['min_experiments']})",
          f"* PEFT techniques: {cov['compliance']['peft_techniques']} (min "
          f"{SPEC['rules']['min_peft_techniques']}; LoRMA done: {cov['compliance']['lorma_done']})",
          f"* KD experiment done: {cov['compliance']['kd_done']}",
          "* every experiment starts from a Phase-1 model; no external weights, corpora or APIs;"
          " no transformers / peft / bitsandbytes / sklearn.", "", "## References", ""]
    R += [f"* {c}" for c in CITATIONS]
    with open(os.path.join(out_root, "README_PHASE3.md"), "w") as f:
        f.write("\n".join(R) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--codabench", default=None)
    a = ap.parse_args()
    build_report(a.out, default_input_roots(), a.codabench)


if __name__ == "__main__":
    main()
