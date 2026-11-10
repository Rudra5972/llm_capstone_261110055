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
        "arch": rc.get("arch"), "doc_repr": met.get("doc_repr", plan.get("doc_repr",
                                                                         rc.get("doc_repr"))),
        "chunk_tokens": met.get("chunk_tokens", plan.get("chunk_tokens")),
        "summary_budget": met.get("summary_budget"), "max_new": met.get("max_new",
                                                                         plan.get("max_new")),
        "num_beams": rc.get("num_beams"), "status": status,
        "fail_reason": "" if status in ("DONE", "PENDING", "RUNNING") else st.get("reason", ""),
        "selfcheck": (met.get("selfcheck", chk.get("selfcheck", "")) or "") + (
            " DEGENERATE" if met.get("degenerate") else ""),
        "train_docs": plan.get("train_docs"), "train_seqs": plan.get("train_seqs"),
        "train_fraction": met.get("train_fraction", plan.get("train_fraction")),
        "dev_docs": rc.get("n_dev"), "dev_gen_docs": met.get("dev_gen_docs", plan.get("dev_gen")),
        "test_docs": rc.get("n_test"),
        "epochs_planned": met.get("epochs_planned", plan.get("epochs")),
        "epochs_run": met.get("epochs_run", prog.get("epochs_run")),
        "best_epoch": met.get("best_epoch", prog.get("best_epoch")), "lr": rc.get("lr"),
        "batch_effective": rc.get("effective_batch"),
        "dev_loss": met.get("dev_loss", prog.get("best_dev_loss")),
        "trainable_params": size.get("trainable_params"), "total_params": tot,
        "trainable_pct": size.get("trainable_pct"),
        "params_ratio_vs_teacher": (tot / teacher_total) if (tot and teacher_total and rc.get(
            "technique") in ("kd", "student_hard")) else None,
        "projected_minutes": plan.get("projected_minutes"),
        "tokens_per_s": plan.get("tokens_per_s_train"), "gpu_slot": rc.get("gpu_slot"),
        "attempts": rc.get("attempt"), "train_files": "train.json",
        "leak_guard": rc.get("leak_guard", "passed"), "seed": rc.get("seed"),
        "job": rc.get("job", os.path.basename(run_dir)),
        "submission_zip": zp if os.path.exists(zp) else "",
        "started_at": rc.get("started_at"), "finished_at": st.get("time") if status in (
            "DONE", "FAILED", "DEGENERATE", "SKIPPED_TIME") else "",
    }
    for k in ("score_metric", "dev_score", "dev_bertscore_f1", "dev_rouge1", "dev_rouge2",
              "dev_rougeL", "dev_ppl", "length_ratio", "cap_words", "words", "ref_words",
              "compression", "repeated_trigram_rate", "novel_bigram_rate", "content_support",
              "number_precision", "citation_precision", "ref_content_recall", "empty_outputs",
              "test_mean_words", "train_minutes", "gen_tokens_per_s", "ms_per_doc_infer",
              "peak_mem_gb"):
        row[k] = met.get(k)
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
    for metric, name in (("dev_bertscore_f1", "ALL_RESULTS_pivot.csv"),
                         ("dev_rouge1", "ALL_RESULTS_pivot_rouge1.csv"),
                         ("dev_rougeL", "ALL_RESULTS_pivot_rougeL.csv"),
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
# E8 report: consensus ensemble of the BEST runs + summary-quality analysis + figures
# =============================================================================================
def _score(r: Dict, metric: str) -> float:
    try:
        return float(r.get(metric))
    except (TypeError, ValueError):
        return -1.0


def common_metric(rows: List[Dict]) -> str:
    """BERTScore when EVERY finished run has it, else the ROUGE mean (never mix the two)."""
    done = [r for r in rows if r.get("status") == "DONE"]
    if done and all(r.get("dev_bertscore_f1") not in (None, "") for r in done):
        return "dev_bertscore_f1"
    for r in rows:
        try:
            r["dev_rouge_mean"] = 0.5 * (float(r["dev_rouge1"]) + float(r["dev_rougeL"]))
        except (TypeError, ValueError, KeyError):
            r["dev_rouge_mean"] = None
    return "dev_rouge_mean"


def _best_per_exp(rows: List[Dict]) -> List[Dict]:
    m = common_metric(rows)
    out = []
    for e in COUNTED:
        c = [r for r in rows if r.get("experiment") == e and r.get("status") == "DONE"
             and not r.get("technique", "").endswith("(teacher)")]
        if c:
            out.append(max(c, key=lambda r: _score(r, m)))
    return out


def _jsonl(path: str) -> List[Dict]:
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def ensemble(rows: List[Dict], out_root: str, log=print) -> Optional[Dict]:
    """Consensus (MBR) ensemble: per document, the member summary with the highest mean ROUGE-L
    to the other members' summaries. Scored on the dev documents every member decoded (ROUGE;
    optimistic because dev chose the members); written as a ready submission."""
    import zipfile
    from gen_ml import consensus_pick, rouge_all
    from validate_submission import validate
    mem = []
    for m in _best_per_exp(rows):
        dv = {x["id"]: x for x in _jsonl(os.path.join(m["_run_dir"], "dev", "dev_predictions.jsonl"))}
        tp = os.path.join(m["_run_dir"], "submission", "prediction.json")
        if dv and os.path.exists(tp):
            mem.append((m, dv, {x["id"]: x["summary"] for x in json.load(open(tp, encoding="utf-8"))}))
    if len(mem) < 3:
        log("   [ensemble] fewer than 3 members with saved predictions -> skipped")
        return None
    ids = sorted(set.intersection(*[set(dv) for _, dv, _ in mem]))
    refs = [mem[0][1][i]["reference"] for i in ids]
    ens = [mem[0][1][i]["prediction"] for i in ids]
    for k, i in enumerate(ids):
        c = [dv[i]["prediction"] for _, dv, _ in mem]
        ens[k] = c[consensus_pick(c)]
    r_ens = rouge_all(ens, refs)
    singles = {f"{m.get('experiment')}:{m.get('model_id')}":
               rouge_all([dv[i]["prediction"] for i in ids], refs) for m, dv, _ in mem}
    mean = lambda r: 0.5 * (r["rouge1"] + r["rougeL"])  # noqa: E731
    best_single = max(mean(v) for v in singles.values())
    test_ids = read_json(os.path.join(out_root, "test_ids.json"), None) or sorted(mem[0][2])
    items = []
    for i in test_ids:
        c = [tp.get(i, "") for _, _, tp in mem]
        items.append({"id": i, "summary": c[consensus_pick(c)] or "."})
    d = os.path.join(out_root, "E8_report", "ensemble")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(d, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(d, "prediction.json"), "prediction.json")
    v = validate(os.path.join(d, "submission.zip"), test_ids)
    rep = {"members": list(singles), "dev_docs": len(ids),
           "ensemble_rouge1": r_ens["rouge1"], "ensemble_rougeL": r_ens["rougeL"],
           "best_single_rouge_mean": best_single, "validation_ok": v["ok"],
           "use_it": bool(v["ok"] and mean(r_ens) > best_single + 0.002),
           "note": "ROUGE only (BERTScore needs the GPU); dev chose the members -> optimistic"}
    write_json(os.path.join(d, "ensemble_report.json"), rep)
    log(f"   [ensemble] {len(mem)} members on {len(ids)} dev docs: ROUGE mean "
        f"{mean(r_ens):.4f} vs best single {best_single:.4f} -> use_it={rep['use_it']}")
    return rep


def error_analysis(rows: List[Dict], out_root: str, dev_texts=None, log=print) -> None:
    """Per-experiment summary-quality tables: length curves, document-length buckets,
    redundancy / abstractiveness / factuality proxies, the worst dev documents, and plots."""
    rep_dir = os.path.join(out_root, "E8_report")
    os.makedirs(rep_dir, exist_ok=True)
    curves, buckets, worst = [], [], []
    done = [r for r in rows if r.get("status") == "DONE" and r.get("_run_dir")]
    for r in done:
        base = {"experiment": r.get("experiment"), "technique": r.get("technique"),
                "model_id": r.get("model_id"), "job": r.get("job"), "doc_repr": r.get("doc_repr")}
        for x in read_csv(os.path.join(r["_run_dir"], "dev", "length_curve.csv")):
            curves.append({**base, **x})
        for x in read_csv(os.path.join(r["_run_dir"], "dev", "length_buckets.csv")):
            buckets.append({**base, **x})
        preds = _jsonl(os.path.join(r["_run_dir"], "dev", "dev_predictions.jsonl"))
        for x in sorted(preds, key=lambda x: x.get("rougeL") or 0)[:5]:
            worst.append({**base, "id": x["id"], "rougeL": x.get("rougeL"),
                          "bertscore_f1": x.get("bertscore_f1"), "words": x.get("words"),
                          "ref_words": x.get("ref_words"), "n_doc_tokens": x.get("n_doc_tokens"),
                          "prediction_start": (x.get("prediction") or "")[:300]})
    if curves:
        write_csv(os.path.join(rep_dir, "length_curves.csv"), curves, list(curves[0].keys()))
    if buckets:
        write_csv(os.path.join(rep_dir, "doc_length_buckets.csv"), buckets, list(buckets[0].keys()))
    if worst:
        write_csv(os.path.join(rep_dir, "worst_dev_docs.csv"), worst, list(worst[0].keys()))
    qual_cols = ["experiment", "technique", "model_id", "doc_repr", "dev_bertscore_f1",
                 "dev_rouge1", "dev_rouge2", "dev_rougeL", "words", "ref_words", "compression",
                 "repeated_trigram_rate", "novel_bigram_rate", "content_support",
                 "number_precision", "citation_precision", "ref_content_recall", "trainable_pct",
                 "total_params", "params_ratio_vs_teacher", "train_minutes", "ms_per_doc_infer",
                 "peak_mem_gb"]
    write_csv(os.path.join(rep_dir, "quality_table.csv"), done, qual_cols)
    log(f"   [report] {len(done)} finished runs: quality_table, length_curves ({len(curves)}), "
        f"doc_length_buckets ({len(buckets)}), worst_dev_docs ({len(worst)})")
    m = common_metric(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = [r for r in done if _score(r, m) >= 0]
        if ok:
            fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(ok)), 4))
            ax.bar(range(len(ok)), [_score(r, m) for r in ok], color="#4C72B0")
            ax.set_xticks(range(len(ok)))
            ax.set_xticklabels([f"{r['experiment']}\n{r['model_id']}" for r in ok], fontsize=7)
            ax.set_ylabel(m)
            ax.set_title("Phase 6 summarisation: dev score per experiment and model")
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "dev_score.png"), dpi=130)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in ok:
                try:
                    ax.scatter(float(r["words"]), _score(r, m), s=25)
                    ax.annotate(f"{r['experiment']}/{r['model_id']}", (float(r["words"]),
                                                                       _score(r, m)), fontsize=6)
                except (TypeError, ValueError, KeyError):
                    pass
            ax.set_xlabel("mean generated words (dev)")
            ax.set_ylabel(m)
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "score_vs_length.png"), dpi=130)
            plt.close(fig)
    except Exception as e:
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
