"""
aggregate.py -- one CSV row per (experiment, job) from the files a job leaves behind, the
per-experiment CSVs, ALL_RESULTS.csv + pivots, and the E8 report (ensemble, error analysis).

    python aggregate.py [--out-root /kaggle/working/phase7_out] [--report]

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
        "arch": rc.get("arch"), "attn_mode": rc.get("attn_mode"), "pooling": rc.get("pooling"),
        "doc_repr": rc.get("doc_repr"), "chunk_tokens": rc.get("chunk_tokens"),
        "k_train": met.get("k_train", rc.get("k_train")),
        "k_infer": met.get("k_infer", plan.get("k_infer")), "hard_negs": rc.get("hard_negs"),
        "status": status,
        "fail_reason": "" if status in ("DONE", "PENDING", "RUNNING") else st.get("reason", ""),
        "selfcheck": (met.get("selfcheck", chk.get("selfcheck", "")) or "") + (
            " DEGENERATE" if met.get("degenerate") else ""),
        "train_queries": met.get("train_queries", rc.get("n_train")),
        "train_pairs": plan.get("train_pairs"),
        "train_fraction": met.get("train_fraction", plan.get("train_fraction")),
        "dev_queries": rc.get("n_dev"), "test_queries": rc.get("n_test"),
        "epochs_planned": met.get("epochs_planned", plan.get("epochs")),
        "epochs_run": met.get("epochs_run", prog.get("epochs_run")),
        "best_epoch": met.get("best_epoch", prog.get("best_epoch")), "lr": rc.get("lr"),
        "batch_pairs": rc.get("batch_pairs"),
        "dev_proxy_map": met.get("dev_proxy_map", prog.get("best_proxy_map")),
        "dev_proxy_map_init": met.get("dev_proxy_map_init", plan.get("init_proxy")),
        "trainable_params": size.get("trainable_params"), "total_params": tot,
        "trainable_pct": size.get("trainable_pct"),
        "params_ratio_vs_teacher": (tot / teacher_total) if (tot and teacher_total and rc.get(
            "technique") in ("kd", "student_hard")) else None,
        "projected_minutes": plan.get("projected_minutes"), "gpu_slot": rc.get("gpu_slot"),
        "attempts": rc.get("attempt"), "leak_guard": rc.get("leak_guard", "passed"),
        "seed": rc.get("seed"), "job": rc.get("job", os.path.basename(run_dir)),
        "submission_zip": zp if os.path.exists(zp) else "",
        "started_at": rc.get("started_at"), "finished_at": st.get("time") if status in (
            "DONE", "FAILED", "DEGENERATE", "SKIPPED_TIME") else "",
    }
    for k in ("score_metric", "dev_score", "dev_micro_f1", "dev_micro_p", "dev_micro_r",
              "dev_dense_f1", "dev_dense_mean_f1", "dev_dense_max_f1", "dev_bm25_f1",
              "dense_weight", "doc_agg", "dev_dense_recall_at_100", "train_minutes",
              "tokens_per_s", "infer_tokens_per_s", "ms_per_doc_infer", "peak_mem_gb"):
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
    for metric, name in (("dev_micro_f1", "ALL_RESULTS_pivot.csv"),
                         ("dev_dense_f1", "ALL_RESULTS_pivot_dense_only.csv"),
                         ("dev_micro_r", "ALL_RESULTS_pivot_recall.csv"),
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
# E8 report: score-fusion ensemble of the BEST runs + retrieval error analysis + figures
# =============================================================================================
def _score(r: Dict, metric: str) -> float:
    try:
        return float(r.get(metric))
    except (TypeError, ValueError):
        return -1.0


def common_metric(rows: List[Dict]) -> str:
    """Phase 7 has one dev metric for every run (no network-dependent scorer)."""
    return "dev_micro_f1"


def _best_per_exp(rows: List[Dict]) -> List[Dict]:
    out = []
    for e in COUNTED:
        c = [r for r in rows if r.get("experiment") == e and r.get("status") == "DONE"
             and not r.get("technique", "").endswith("(teacher)")]
        if c:
            out.append(max(c, key=lambda r: _score(r, "dev_micro_f1")))
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


def _eval_data(out_root: str):
    for od in iter_out_dirs(out_root, default_input_roots()):
        p = os.path.join(od, "eval_data.npz")
        if os.path.exists(p):
            return np.load(p, allow_pickle=False)
    return None


def ensemble(rows: List[Dict], out_root: str, log=print) -> Optional[Dict]:
    """Mean of the members' per-query z-scored dense scores, fused with BM25 (weight tuned on
    dev, as every job does); also reciprocal-rank fusion. Written as a ready submission."""
    import zipfile
    from pcr_ml import K_EVAL, eval_scores, full_rank, fuse, rrf, topk, tune_fusion, zrow
    from validate_submission import validate
    ed = _eval_data(out_root)
    mem = []
    for m in _best_per_exp(rows):
        p = os.path.join(m["_run_dir"], "dev", "scores.npz")
        if os.path.exists(p):
            z = np.load(p)
            mem.append((m, z["dev"].astype(np.float32), z["test"].astype(np.float32)))
    if ed is None or len(mem) < 2:
        log("   [ensemble] eval_data.npz or fewer than 2 members with saved scores -> skipped")
        return None
    gold = [json.loads(g) for g in ed["dev_gold"]]
    bd, bt = ed["bm25_dev"].astype(np.float32), ed["bm25_test"].astype(np.float32)
    shape_ok = [x for x in mem if x[1].shape == bd.shape and x[2].shape == bt.shape]
    if len(shape_ok) < 2:
        log("   [ensemble] members were scored on different data -> skipped")
        return None
    dev_mean = np.mean([zrow(x[1]) for x in shape_ok], 0)
    test_mean = np.mean([zrow(x[2]) for x in shape_ok], 0)
    grid = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
    w, best, curve = tune_fusion(dev_mean, bd, gold, [g for g in grid if g >= 0.2])
    sys_dev = [full_rank(fuse(x[1], bd, float(x[0].get("dense_weight") or 1))) for x in shape_ok]
    rrf_f1 = eval_scores(rrf(sys_dev, bd.shape[1]), gold)["micro_f1"]
    singles = {f"{m.get('experiment')}:{m.get('model_id')}": _score(m, "dev_micro_f1")
               for m, _, _ in shape_ok}
    best_single = max(singles.values())
    S = fuse(test_mean, bt, w)
    tq, tp = [str(x) for x in ed["test_q"]], [str(x) for x in ed["test_pool"]]
    rt = topk(S, K_EVAL)
    pred = {q: [tp[j] for j in rt[i]] for i, q in enumerate(tq)}
    d = os.path.join(out_root, "E8_report", "ensemble")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(pred, f)
    with zipfile.ZipFile(os.path.join(d, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(d, "prediction.json"), "prediction.json")
    v = validate(os.path.join(d, "submission.zip"), tq, tp)
    rep = {"members": list(singles), "dense_weight": w, "ensemble_dev_micro_f1": best["micro_f1"],
           "ensemble_dev_micro_p": best["micro_p"], "ensemble_dev_micro_r": best["micro_r"],
           "rrf_dev_micro_f1": rrf_f1, "best_single_dev_micro_f1": best_single,
           "validation_ok": v["ok"], "use_it": bool(v["ok"] and best["micro_f1"] >
                                                     best_single + 0.002),
           "note": "dev chose the members and the fusion weight -> optimistic"}
    write_json(os.path.join(d, "ensemble_report.json"), rep)
    log(f"   [ensemble] {len(shape_ok)} members: dev Micro-F1@7 {best['micro_f1']:.4f} (RRF "
        f"{rrf_f1:.4f}) vs best single {best_single:.4f} -> use_it={rep['use_it']}")
    return rep


def error_analysis(rows: List[Dict], out_root: str, dev_texts=None, log=print) -> None:
    """Retrieval tables: per-run dense / BM25 / fused scores, hits by query-length bucket and
    by gold-set size, the queries every run misses, and plots."""
    rep_dir = os.path.join(out_root, "E8_report")
    os.makedirs(rep_dir, exist_ok=True)
    done = [r for r in rows if r.get("status") in ("DONE", "DEGENERATE") and r.get("_run_dir")]
    buckets, per_q = [], {}
    for r in done:
        base = {"experiment": r.get("experiment"), "technique": r.get("technique"),
                "model_id": r.get("model_id"), "job": r.get("job")}
        preds = _jsonl(os.path.join(r["_run_dir"], "dev", "dev_predictions.jsonl"))
        if not preds:
            continue
        chars = np.array([p.get("n_chars") or 0 for p in preds], float)
        qs = np.quantile(chars, [0.33, 0.66]) if len(chars) > 3 else [0, 0]
        for name, sel in (("short", chars <= qs[0]), ("medium", (chars > qs[0]) & (chars <= qs[1])),
                          ("long", chars > qs[1])):
            P = [p for p, s_ in zip(preds, sel) if s_]
            tp = sum(p["hits"] for p in P)
            ng = sum(p["n_gold"] for p in P)
            prec, rec = tp / max(1, 7 * len(P)), tp / max(1, ng)
            buckets.append({**base, "query_length": name, "queries": len(P),
                            "micro_p": prec, "micro_r": rec,
                            "micro_f1": 2 * prec * rec / max(1e-12, prec + rec)})
        for p in preds:
            per_q.setdefault(p["id"], []).append(p["hits"])
    if buckets:
        write_csv(os.path.join(rep_dir, "query_length_buckets.csv"), buckets, list(buckets[0]))
    miss = [{"id": q, "runs": len(h), "runs_with_zero_hits": sum(1 for x in h if x == 0),
             "mean_hits": float(np.mean(h))} for q, h in per_q.items()]
    miss.sort(key=lambda x: (-x["runs_with_zero_hits"], x["mean_hits"]))
    if miss:
        write_csv(os.path.join(rep_dir, "hardest_dev_queries.csv"), miss[:50], list(miss[0]))
    cols = ["experiment", "technique", "model_id", "attn_mode", "pooling", "doc_repr",
            "hard_negs", "dev_micro_f1", "dev_micro_p", "dev_micro_r", "dev_dense_f1",
            "dev_dense_mean_f1", "dev_dense_max_f1", "dev_bm25_f1", "dense_weight", "doc_agg",
            "dev_dense_recall_at_100", "dev_proxy_map_init", "dev_proxy_map", "trainable_pct",
            "total_params", "params_ratio_vs_teacher", "train_minutes", "ms_per_doc_infer",
            "peak_mem_gb"]
    write_csv(os.path.join(rep_dir, "retrieval_table.csv"), done, cols)
    log(f"   [report] {len(done)} finished runs: retrieval_table, query_length_buckets "
        f"({len(buckets)}), hardest_dev_queries ({len(miss)})")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = [r for r in done if _score(r, "dev_micro_f1") >= 0]
        if ok:
            x = np.arange(len(ok))
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(ok)), 4))
            ax.bar(x - 0.2, [_score(r, "dev_micro_f1") for r in ok], 0.4, label="fused (submitted)")
            ax.bar(x + 0.2, [_score(r, "dev_dense_f1") for r in ok], 0.4, label="dense only")
            bm = _score(ok[0], "dev_bm25_f1")
            if bm >= 0:
                ax.axhline(bm, color="k", ls="--", lw=1, label="BM25 only")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r['experiment']}\n{r['model_id']}" for r in ok], fontsize=7)
            ax.set_ylabel("dev Micro-F1@7")
            ax.legend(fontsize=7)
            ax.set_title("Phase 7 prior case retrieval: dev Micro-F1@7")
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "dev_score.png"), dpi=130)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in ok:
                try:
                    ax.scatter(float(r["trainable_pct"]), _score(r, "dev_dense_f1"), s=25)
                    ax.annotate(f"{r['experiment']}/{r['model_id']}",
                                (float(r["trainable_pct"]), _score(r, "dev_dense_f1")), fontsize=6)
                except (TypeError, ValueError, KeyError):
                    pass
            ax.set_xscale("log")
            ax.set_xlabel("trainable parameters (%)")
            ax.set_ylabel("dev Micro-F1@7 (dense only)")
            fig.tight_layout()
            fig.savefig(os.path.join(rep_dir, "quality_vs_trainable.png"), dpi=130)
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
