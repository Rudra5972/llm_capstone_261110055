"""
aggregate.py -- one CSV row per (experiment, job) from the files a job leaves behind, the
per-experiment CSVs, ALL_RESULTS.csv + pivots, and the E8 report (ensemble, error analysis).

    python aggregate.py [--out-root /kaggle/working/phase5_out] [--report]

`build_row(run_dir)` reads run_config.json, status.json, plan.json, progress.json, metrics.json,
model_size.json and job_checks.json; it never fails (missing files -> empty cells), so the CSV
can be rewritten at any moment (rule R2: a row exists for every planned job from minute one).
"""
import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from cjpe_common import (COUNTED, CSV_COLUMNS, EPOCH_COLUMNS, EXPERIMENTS, MODELS, OUT_ROOT,
                         default_input_roots, iter_out_dirs, read_csv, read_json, write_csv,
                         write_json, log_to)

STATE_TO_STATUS = {"done": "DONE", "degenerate": "DEGENERATE", "failed": "FAILED",
                   "skipped_time": "SKIPPED_TIME", "preflight": "RUNNING",
                   "training": "RUNNING", "evaluating": "RUNNING", "predicting": "RUNNING",
                   "pending": "PENDING", "queued": "PENDING", "smoke_ok": "SMOKE_OK"}


def build_row(run_dir: str, teacher_total: Optional[float] = None) -> Dict:
    rc = read_json(os.path.join(run_dir, "logs", "run_config.json"), {}) or {}
    st = read_json(os.path.join(run_dir, "status.json"), {}) or {}
    plan = read_json(os.path.join(run_dir, "logs", "plan.json"), {}) or {}
    prog = read_json(os.path.join(run_dir, "logs", "progress.json"), {}) or {}
    met = read_json(os.path.join(run_dir, "metrics.json"), {}) or {}
    size = read_json(os.path.join(run_dir, "logs", "model_size.json"), {}) or {}
    chk = read_json(os.path.join(run_dir, "logs", "job_checks.json"), {}) or {}
    dev = met.get("dev", {})
    status = STATE_TO_STATUS.get(st.get("state", "pending"), "PENDING")
    if status == "FAILED" and st.get("exit_code") == 2:
        status = "SKIPPED_TIME"
    if status == "DONE" and not os.path.exists(os.path.join(run_dir, "DONE.json")):
        status = "FAILED" if met else status
    zp = os.path.join(run_dir, "submission", "submission.zip")
    tech = rc.get("technique", "")
    if rc.get("is_teacher"):
        tech = f"{tech} (teacher)"
    tot = size.get("total_params") or met.get("total_params")
    row = {
        "experiment": rc.get("experiment"), "technique": tech, "model_id": rc.get("model"),
        "arch": rc.get("arch"), "attn_mode": rc.get("attn_mode"), "doc_repr": rc.get("doc_repr"),
        "K_effective": met.get("K_effective", plan.get("K")), "max_len": 512 if
        rc.get("doc_repr") != "headtail" else 1024, "head_type": "linear(768->L) sigmoid",
        "pooling": "chunk-mean + attention", "class_weight": rc.get("class_weight"),
        "status": status, "fail_reason": "" if status in ("DONE", "PENDING", "RUNNING",
                                                           "SMOKE_OK") else
        st.get("reason", ""), "selfcheck": (met.get("selfcheck", chk.get("selfcheck", "")) or "") + (
            " DEGENERATE" if met.get("degenerate") else ""),
        "train_docs": plan.get("train_docs"), "train_fraction": met.get(
            "train_fraction", plan.get("train_fraction")),
        "dev_docs": rc.get("n_dev"), "test_docs": rc.get("n_test"),
        "epochs_planned": met.get("epochs_planned", plan.get("epochs")),
        "epochs_run": met.get("epochs_run", prog.get("epochs_run")),
        "best_epoch": met.get("best_epoch", prog.get("best_epoch")),
        "lr_backbone": rc.get("lr"), "lr_head": rc.get("head_lr"),
        "batch_effective": rc.get("effective_batch"), "rule": met.get("rule"),
        "tau": met.get("tau"), "kmax": met.get("kmax", rc.get("kmax")),
        "n_labels": met.get("n_labels", len(rc.get("label_ids") or []) or None),
        "loss": rc.get("loss", "bce"),
        "dev_macro_f1": met.get("dev_macro_f1", prog.get("best_dev_macro_f1")
                                if status == "RUNNING" else None),
        "dev_macro_p": dev.get("macro_p"), "dev_macro_r": dev.get("macro_r"),
        "dev_micro_f1": dev.get("micro_f1"), "dev_subset_acc": dev.get("subset_acc"),
        "dev_macro_f1_all": dev.get("macro_f1_all"),
        "dev_macro_f1_global": met.get("dev_macro_f1_global"),
        "dev_macro_f1_at_0p5": met.get("dev_macro_f1_at_0p5"),
        "cv_macro_global": met.get("cv_macro_global"),
        "cv_macro_perlabel": met.get("cv_macro_perlabel"),
        "dev_macro_f1_rare": dev.get("macro_f1_rare"),
        "dev_macro_f1_medium": dev.get("macro_f1_medium"),
        "dev_macro_f1_frequent": dev.get("macro_f1_frequent"), "n_rare": dev.get("n_rare"),
        "n_medium": dev.get("n_medium"), "n_frequent": dev.get("n_frequent"),
        "dev_mean_pred_labels": dev.get("mean_pred_labels"),
        "dev_mean_gold_labels": dev.get("mean_gold_labels"),
        "dev_only_labels": met.get("dev_only_labels"), "ece": met.get("ece"),
        "test_mean_labels": met.get("test_mean_labels"),
        "ms_per_doc_infer": met.get("ms_per_doc_infer"),
        "trainable_params": size.get("trainable_params"), "total_params": tot,
        "trainable_pct": size.get("trainable_pct"),
        "params_ratio_vs_teacher": (tot / teacher_total) if (tot and teacher_total and
                                                            rc.get("technique") == "kd") else None,
        "projected_minutes": plan.get("projected_minutes"),
        "train_minutes": met.get("train_minutes"), "tokens_per_s": plan.get("tokens_per_s_train"),
        "docs_per_s_infer": met.get("docs_per_s_infer"), "peak_mem_gb": met.get("peak_mem_gb"),
        "gpu_slot": rc.get("gpu_slot"), "attempts": rc.get("attempt"),
        "train_files": "train.json", "leak_guard": rc.get("leak_guard", "passed"),
        "seed": rc.get("seed"), "job": rc.get("job", os.path.basename(run_dir)),
        "submission_zip": zp if os.path.exists(zp) else "",
        "started_at": rc.get("started_at"), "finished_at": st.get("time") if status in (
            "DONE", "FAILED", "DEGENERATE", "SKIPPED_TIME") else "",
    }
    return row


def epoch_rows(run_dir: str) -> List[Dict]:
    rc = read_json(os.path.join(run_dir, "logs", "run_config.json"), {}) or {}
    p = os.path.join(run_dir, "logs", "train_log.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "eval":
                    out.append({"experiment": rc.get("experiment"), "model_id": rc.get("model"),
                                "job": rc.get("job"), **r})
    return out


def exp_rows(exp_dir: str, planned: Optional[List[Dict]] = None) -> List[Dict]:
    """Rows for every job folder of an experiment (+ PENDING rows for planned jobs)."""
    jobs = sorted(d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))
                  and d != "BEST") if os.path.isdir(exp_dir) else []
    teacher_total = None
    for j in jobs:
        rc = read_json(os.path.join(exp_dir, j, "logs", "run_config.json"), {}) or {}
        if rc.get("is_teacher"):
            teacher_total = (read_json(os.path.join(exp_dir, j, "logs", "model_size.json"), {})
                             or {}).get("total_params")
    rows = {}
    for p in planned or []:
        rows[p["job"]] = dict(p)
    for j in jobs:
        if not os.path.exists(os.path.join(exp_dir, j, "logs", "run_config.json")):
            st = read_json(os.path.join(exp_dir, j, "status.json"), {}) or {}
            if j in rows and st.get("state") in STATE_TO_STATUS:
                rows[j]["status"] = STATE_TO_STATUS[st["state"]]
                if rows[j]["status"] not in ("PENDING", "RUNNING"):
                    rows[j]["fail_reason"] = st.get("reason", "")
            continue
        r = build_row(os.path.join(exp_dir, j), teacher_total)
        if j in rows:
            r = {**rows[j], **{k: v for k, v in r.items() if v not in (None, "")}}
            if r.get("status") == "PENDING" and rows[j].get("status") not in (None, "PENDING"):
                r["status"] = rows[j]["status"]
                r["fail_reason"] = rows[j].get("fail_reason", "")
        rows[j] = r
    return list(rows.values())


def write_exp_csvs(exp_dir: str, planned: Optional[List[Dict]] = None) -> List[Dict]:
    folder = os.path.basename(exp_dir.rstrip("/"))
    rows = exp_rows(exp_dir, planned)
    write_csv(os.path.join(exp_dir, f"results_{folder}.csv"), rows, CSV_COLUMNS)
    ep = []
    for j in sorted(os.listdir(exp_dir)) if os.path.isdir(exp_dir) else []:
        if os.path.isdir(os.path.join(exp_dir, j)) and j != "BEST":
            ep += epoch_rows(os.path.join(exp_dir, j))
    write_csv(os.path.join(exp_dir, f"epoch_log_{folder}.csv"), ep, EPOCH_COLUMNS)
    return rows


def _status_rank(s: str) -> int:
    return {"DONE": 5, "DEGENERATE": 4, "RUNNING": 3, "FAILED": 2, "SKIPPED_TIME": 2,
            "PENDING": 1}.get(s, 0)


def aggregate(out_root: str = OUT_ROOT, input_roots=None, log=print) -> List[Dict]:
    """Merge every results_*.csv (working dir first, then attached inputs) -> ALL_RESULTS.csv.
    submission_zip paths are re-resolved where the CSV was found."""
    input_roots = default_input_roots() if input_roots is None else input_roots
    best: Dict = {}
    for od in iter_out_dirs(out_root, input_roots):
        for csvp in sorted(glob.glob(os.path.join(od, "*", "results_*.csv"))):
            exp_dir = os.path.dirname(csvp)
            for r in read_csv(csvp):
                job = r.get("job") or ""
                z = os.path.join(exp_dir, job, "submission", "submission.zip")
                r["submission_zip"] = z if (job and os.path.exists(z)) else ""
                r["_run_dir"] = os.path.join(exp_dir, job)
                key = (r.get("experiment"), job)
                old = best.get(key)
                if old is None or _status_rank(r.get("status", "")) > _status_rank(
                        old.get("status", "")):
                    best[key] = r
    rows = sorted(best.values(), key=lambda r: (str(r.get("experiment")), str(r.get("job"))))
    if out_root:
        os.makedirs(out_root, exist_ok=True)
        write_csv(os.path.join(out_root, "ALL_RESULTS.csv"), rows, CSV_COLUMNS)
        write_pivots(out_root, rows)
    return rows


def write_pivots(out_root: str, rows: List[Dict]) -> None:
    for metric, name in (("dev_macro_f1", "ALL_RESULTS_pivot.csv"),
                         ("dev_micro_f1", "ALL_RESULTS_pivot_micro_f1.csv"),
                         ("dev_macro_f1_rare", "ALL_RESULTS_pivot_rare_macro_f1.csv"),
                         ("trainable_pct", "ALL_RESULTS_pivot_trainable_pct.csv"),
                         ("train_minutes", "ALL_RESULTS_pivot_train_minutes.csv")):
        table: Dict[str, Dict] = {}
        for r in rows:
            if r.get("status") not in ("DONE", "DEGENERATE"):
                continue
            k = f"{r.get('experiment')}:{r.get('technique')}"
            if r.get("experiment") == "E7":
                k = f"E7:{r.get('job', '').rsplit('_M', 1)[0]}"
            table.setdefault(k, {"experiment_technique": k})
            table[k][r.get("model_id")] = r.get(metric)
        write_csv(os.path.join(out_root, name), list(table.values()),
                  ["experiment_technique", "M1", "M2", "M3", "M4"])


# =============================================================================================
# E8 report: ensemble of BEST models + error analysis + figures
# =============================================================================================
def _best_per_exp(rows: List[Dict]) -> List[Dict]:
    out = []
    for e in COUNTED:
        c = [r for r in rows if r.get("experiment") == e and r.get("status") == "DONE"
             and not r.get("technique", "").endswith("(teacher)")]
        if c:
            out.append(max(c, key=lambda r: float(r.get("dev_macro_f1") or -1)))
    return out


def _npz(path: str):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def ensemble(rows: List[Dict], out_root: str, log=print) -> Optional[Dict]:
    """Mean sigmoid probability of the best DONE run of each counted experiment; the decision
    rule is re-chosen on dev exactly as for single runs (dev-only). Written to
    E8_report/ensemble/ and marked optimistic (dev chose the members)."""
    from lsi_ml import apply_rule, choose_rule, ml_scores, rule_taus
    from cjpe_predict import write_submission
    mem = []
    for m in _best_per_exp(rows):
        dvp = _npz(os.path.join(m["_run_dir"], "dev", "dev_probs.npz"))
        tsp = _npz(os.path.join(m["_run_dir"], "submission", "test_probs.npz"))
        if dvp is not None and tsp is not None:
            mem.append((m, dvp, tsp))
    if len(mem) < 2:
        log("   [ensemble] fewer than 2 members with saved probabilities -> skipped")
        return None
    lab = mem[0][1]["label_ids"]
    mem = [x for x in mem if np.array_equal(x[1]["label_ids"], lab)]
    dev_ids = [str(i) for i in mem[0][1]["ids"]]
    test_ids = [str(i) for i in mem[0][2]["ids"]]
    G = mem[0][1]["gold"]
    P_dev = np.zeros((len(dev_ids), len(lab)), np.float32)
    P_test = np.zeros((len(test_ids), len(lab)), np.float32)
    for m, dvp, tsp in mem:
        di = {str(i): k for k, i in enumerate(dvp["ids"])}
        ti = {str(i): k for k, i in enumerate(tsp["ids"])}
        P_dev += dvp["probs"].astype(np.float32)[[di[i] for i in dev_ids]]
        P_test += tsp["probs"].astype(np.float32)[[ti[i] for i in test_ids]]
    P_dev /= len(mem)
    P_test /= len(mem)
    kmax = int(read_json(os.path.join(out_root, "label_map.json"), {}).get("kmax", 10)
               if os.path.exists(os.path.join(out_root, "label_map.json")) else 10)
    rule = choose_rule(P_dev, G, kmax)
    s = ml_scores(G, apply_rule(P_dev, rule_taus(rule, len(lab)), kmax))
    ens_dir = os.path.join(out_root, "E8_report", "ensemble")
    os.makedirs(os.path.join(ens_dir, "dev"), exist_ok=True)
    np.savez_compressed(os.path.join(ens_dir, "dev", "dev_probs.npz"), ids=np.asarray(dev_ids),
                        gold=G, probs=P_dev.astype(np.float16), label_ids=lab)
    for f in ("per_label_dev.csv", "errors.csv"):
        with open(os.path.join(ens_dir, "dev", f), "w") as fh:
            fh.write("see member runs\n")
    v = write_submission(ens_dir, test_ids, P_test, rule, [int(x) for x in lab])
    best_single = max(float(m.get("dev_macro_f1") or -1) for m, _, _ in mem)
    rep = {"members": [(m.get("experiment"), m.get("job"), m.get("dev_macro_f1"))
                       for m, _, _ in mem], "rule": {k: v_ for k, v_ in rule.items()
                                                    if k != "taus"},
           "dev": s, "validation_ok": v["ok"], "best_single_dev_macro_f1": best_single,
           "use_it": bool(v["ok"] and s["macro_f1"] > best_single),
           "note": "dev score is optimistic: dev chose the members and the thresholds"}
    write_json(os.path.join(ens_dir, "ensemble_report.json"), rep)
    log(f"   [ensemble] {len(mem)} members, dev macro-F1 {s['macro_f1']:.4f} (optimistic; best "
        f"single {best_single:.4f}) -> {'consider uploading' if rep['use_it'] else 'not better'}")
    return rep


def error_analysis(rows: List[Dict], out_root: str, dev_texts: Optional[Dict[str, str]] = None,
                   log=print) -> None:
    """PER_LABEL_ALL.csv, frequency-bucket table, confused statute pairs, threshold-rule
    comparison, length buckets (best run per experiment) + plots."""
    from collections import Counter
    rep_dir = os.path.join(out_root, "E8_report")
    os.makedirs(rep_dir, exist_ok=True)
    per_label, buckets, rules, lengths, pairs = [], [], [], [], Counter()
    for r in rows:
        if r.get("status") != "DONE":
            continue
        rd = r["_run_dir"]
        for x in read_csv(os.path.join(rd, "dev", "per_label_dev.csv")):
            per_label.append({"experiment": r.get("experiment"), "model_id": r.get("model_id"),
                              "job": r.get("job"), **x})
        rules.append({k: r.get(k) for k in ("experiment", "model_id", "job", "rule", "tau",
                                            "dev_macro_f1", "dev_macro_f1_global",
                                            "dev_macro_f1_at_0p5", "cv_macro_global",
                                            "cv_macro_perlabel")})
        buckets.append({k: r.get(k) for k in ("experiment", "model_id", "job",
                                              "dev_macro_f1_rare", "n_rare",
                                              "dev_macro_f1_medium", "n_medium",
                                              "dev_macro_f1_frequent", "n_frequent")})
    for b in _best_per_exp(rows):
        rep = read_json(os.path.join(b["_run_dir"], "dev", "dev_report.json"), {}) or {}
        for x in rep.get("by_length") or []:
            lengths.append({"experiment": b.get("experiment"), "job": b.get("job"), **x})
        for x in rep.get("confused_pairs") or []:
            pairs[(x["missed"], x["predicted_instead"])] += int(x["docs"])
    if per_label:
        write_csv(os.path.join(out_root, "PER_LABEL_ALL.csv"), per_label,
                  list(per_label[0].keys()))
        # statute-level summary over the best run of each experiment
        best_jobs = {(b.get("experiment"), b.get("job")) for b in _best_per_exp(rows)}
        agg: Dict[str, Dict] = {}
        for x in per_label:
            if (x["experiment"], x["job"]) not in best_jobs:
                continue
            a = agg.setdefault(x["label"], {"label": x["label"], "train_support":
                                            x["train_support"], "dev_support": x["dev_support"],
                                            "f1_sum": 0.0, "n": 0})
            a["f1_sum"] += float(x["f1"] or 0)
            a["n"] += 1
        lab_rows = sorted(({"label": a["label"], "train_support": a["train_support"],
                            "dev_support": a["dev_support"],
                            "mean_f1_best_runs": a["f1_sum"] / max(1, a["n"])}
                           for a in agg.values()), key=lambda r: r["mean_f1_best_runs"])
        if lab_rows:
            write_csv(os.path.join(rep_dir, "statutes_hardest_first.csv"), lab_rows,
                      list(lab_rows[0].keys()))
    for name, data in (("frequency_buckets.csv", buckets), ("threshold_rules.csv", rules),
                       ("length_buckets.csv", lengths)):
        if data:
            write_csv(os.path.join(rep_dir, name), data, list(data[0].keys()))
    cp = [{"missed": a, "predicted_instead": b, "docs": n} for (a, b), n in pairs.most_common(40)]
    if cp:
        write_csv(os.path.join(rep_dir, "confused_statute_pairs.csv"), cp,
                  ["missed", "predicted_instead", "docs"])
    write_json(os.path.join(rep_dir, "error_analysis.json"),
               {"n_runs": len(rules), "top_confused_pairs": cp[:15]})
    log(f"   [report] per-label ({len(per_label)} rows), buckets, rules, confused pairs "
        f"-> {rep_dir}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = [r for r in rows if r.get("status") == "DONE" and r.get("experiment") in COUNTED
              and not r.get("technique", "").endswith("(teacher)")]
        if ok:
            fig, ax = plt.subplots(figsize=(9, 4))
            labels = [f"{r['experiment']}\n{r['model_id']}" for r in ok]
            ax.bar(range(len(ok)), [float(r["dev_macro_f1"]) for r in ok], color="#4C72B0")
            ax.set_xticks(range(len(ok)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("dev macro-F1")
            ax.set_ylim(0, 1)
            ax.set_title("Phase 5 LSI: dev macro-F1 per experiment and model")
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "dev_macro_f1.png"), dpi=130)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in ok:
                try:
                    ax.scatter(float(r["trainable_pct"]), float(r["dev_macro_f1"]), s=25)
                    ax.annotate(f"{r['experiment']}/{r['model_id']}",
                                (float(r["trainable_pct"]), float(r["dev_macro_f1"])),
                                fontsize=6)
                except (TypeError, ValueError):
                    pass
            ax.set_xscale("log")
            ax.set_xlabel("trainable parameters (%)")
            ax.set_ylabel("dev macro-F1")
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "f1_vs_trainable_pct.png"), dpi=130)
            plt.close(fig)
    except Exception as e:                                   # plots are optional
        log(f"   [report] plots skipped ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rows = aggregate(a.out_root)
    print(f"ALL_RESULTS.csv: {len(rows)} rows")
    if a.report:
        ensemble(rows, a.out_root)
        error_analysis(rows, a.out_root)


if __name__ == "__main__":
    main()
