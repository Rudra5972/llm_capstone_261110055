"""
aggregate.py -- one CSV row per (experiment, job) from the files a job leaves behind, the
per-experiment CSVs, ALL_RESULTS.csv + pivots, and the E8 report (ensemble, error analysis).

    python aggregate.py [--out-root /kaggle/working/phase4_out] [--report]

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
        rc.get("doc_repr") != "headtail" else 1024, "head_type": "linear(768->2)",
        "pooling": "chunk-mean + attention", "class_weight": rc.get("class_weight"),
        "status": status, "fail_reason": "" if status in ("DONE", "PENDING", "RUNNING",
                                                           "SMOKE_OK") else
        st.get("reason", ""), "selfcheck": met.get("selfcheck", chk.get("selfcheck", "")),
        "train_docs": plan.get("train_docs"), "train_fraction": met.get(
            "train_fraction", plan.get("train_fraction")),
        "dev_docs": rc.get("n_dev"), "test_docs": rc.get("n_test"),
        "epochs_planned": met.get("epochs_planned", plan.get("epochs")),
        "epochs_run": met.get("epochs_run", prog.get("epochs_run")),
        "best_epoch": met.get("best_epoch", prog.get("best_epoch")),
        "lr_backbone": rc.get("lr"), "lr_head": rc.get("head_lr"),
        "batch_effective": rc.get("effective_batch"), "tau": met.get("tau"),
        "dev_macro_f1": met.get("dev_macro_f1", prog.get("best_dev_macro_f1")
                                if status == "RUNNING" else None),
        "dev_macro_p": dev.get("macro_p"), "dev_macro_r": dev.get("macro_r"),
        "dev_accuracy": dev.get("accuracy"), "dev_f1_class0": dev.get("f1_class0"),
        "dev_f1_class1": dev.get("f1_class1"), "dev_p_class0": dev.get("p_class0"),
        "dev_p_class1": dev.get("p_class1"), "dev_r_class0": dev.get("r_class0"),
        "dev_r_class1": dev.get("r_class1"),
        "dev_macro_f1_at_0p5": met.get("dev_macro_f1_at_0p5"),
        "tp": dev.get("tp"), "fp": dev.get("fp"), "fn": dev.get("fn"), "tn": dev.get("tn"),
        "test_pred_pos_rate": met.get("test_pred_pos_rate"),
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
def _load_probs(run_dir: str):
    p = os.path.join(run_dir, "dev", "probs.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def ensemble(rows: List[Dict], out_root: str, log=print) -> Optional[Dict]:
    """Average p1 of the best DONE run of each counted experiment; tau tuned on dev.
    NOTE: optimistic dev score (dev chose the members). CPU only."""
    from cjpe_eval import binary_scores, tune_tau
    from cjpe_predict import write_submission
    members = []
    for e in COUNTED:
        c = [r for r in rows if r.get("experiment") == e and r.get("status") == "DONE"
             and not r.get("technique", "").endswith("(teacher)")]
        if c:
            members.append(max(c, key=lambda r: float(r.get("dev_macro_f1") or -1)))
    probs = [(m, _load_probs(m["_run_dir"])) for m in members]
    probs = [(m, p) for m, p in probs if p is not None]
    if len(probs) < 2:
        log("   [ensemble] fewer than 2 members with saved probabilities -> skipped")
        return None
    dev_ids = [str(x) for x in probs[0][1]["dev_ids"]]
    test_ids = [str(x) for x in probs[0][1]["test_ids"]]
    P_dev = np.zeros(len(dev_ids))
    P_test = np.zeros(len(test_ids))
    for m, p in probs:
        di = {str(i): k for k, i in enumerate(p["dev_ids"])}
        ti = {str(i): k for k, i in enumerate(p["test_ids"])}
        P_dev += np.array([p["dev_p1"][di[i]] for i in dev_ids])
        P_test += np.array([p["test_p1"][ti[i]] for i in test_ids])
    P_dev /= len(probs)
    P_test /= len(probs)
    gold_map = {}
    dp = os.path.join(probs[0][0]["_run_dir"], "dev", "dev_probs.csv")
    for r in read_csv(dp):
        gold_map[r["id"]] = int(r["label"])
    gold = np.array([gold_map[i] for i in dev_ids])
    tt = tune_tau(P_dev, gold)
    s = binary_scores(gold, (P_dev >= tt["tau"]).astype(int))
    ens_dir = os.path.join(out_root, "E8_report", "ensemble")
    os.makedirs(os.path.join(ens_dir, "dev"), exist_ok=True)
    with open(os.path.join(ens_dir, "dev", "dev_probs.csv"), "w") as f:
        f.write("id,label,p1\n" + "".join(f"{i},{g},{p:.6f}\n" for i, g, p in
                                         zip(dev_ids, gold, P_dev)))
    with open(os.path.join(ens_dir, "dev", "errors.csv"), "w") as f:
        f.write("id,label,p1\n" + "".join(f"{i},{g},{p:.6f}\n" for i, g, p in
                                         zip(dev_ids, gold, P_dev)
                                         if int(p >= tt["tau"]) != g) + ",,\n")
    v = write_submission(ens_dir, test_ids, P_test, tt["tau"])
    rep = {"members": [(m.get("experiment"), m.get("job"), m.get("dev_macro_f1"))
                       for m, _ in probs], "tau": tt["tau"], "dev": s, "validation_ok": v["ok"],
           "note": "dev score is optimistic: dev was used to choose the members and tau"}
    write_json(os.path.join(ens_dir, "ensemble_report.json"), rep)
    log(f"   [ensemble] {len(probs)} members, dev macro-F1 {s['macro_f1']:.4f} (optimistic), "
        f"submission valid: {v['ok']}")
    return rep


def error_analysis(rows: List[Dict], out_root: str, dev_texts: Optional[Dict[str, str]] = None,
                   log=print) -> None:
    """FP/FN by length bucket, confidence histogram, discriminative n-grams (if dev texts)."""
    import re
    from collections import Counter
    rep_dir = os.path.join(out_root, "E8_report")
    os.makedirs(rep_dir, exist_ok=True)
    out = {}
    for e in COUNTED:
        c = [r for r in rows if r.get("experiment") == e and r.get("status") == "DONE"
             and not r.get("technique", "").endswith("(teacher)")]
        if not c:
            continue
        b = max(c, key=lambda r: float(r.get("dev_macro_f1") or -1))
        rd = b["_run_dir"]
        rep = read_json(os.path.join(rd, "dev", "dev_report.json"), {}) or {}
        errs = [r for r in read_csv(os.path.join(rd, "dev", "errors.csv"))
                if r.get("error_type") in ("FP", "FN")]
        hist = Counter()
        for r in errs:
            p = float(r["p1"])
            hist[f"{r['error_type']} p1 {int(p * 10) / 10:.1f}"] += 1
        ent = {"job": b.get("job"), "by_length": rep.get("by_length"),
               "reliability": rep.get("reliability"), "ece": rep.get("ece"),
               "n_fp": sum(1 for r in errs if r["error_type"] == "FP"),
               "n_fn": sum(1 for r in errs if r["error_type"] == "FN"),
               "confidence_histogram": dict(sorted(hist.items()))}
        if dev_texts:
            def grams(ids):
                cnt = Counter()
                for i in ids:
                    t = dev_texts.get(i, "")[-4000:].lower()
                    w = re.findall(r"[a-z]{3,}", t)
                    cnt.update(set(" ".join(w[k:k + 2]) for k in range(len(w) - 1)))
                return cnt
            fp = grams([r["id"] for r in errs if r["error_type"] == "FP"])
            fn = grams([r["id"] for r in errs if r["error_type"] == "FN"])
            nfp, nfn = max(1, ent["n_fp"]), max(1, ent["n_fn"])
            score = {g: fp[g] / nfp - fn[g] / nfn for g in set(fp) | set(fn)
                     if fp[g] + fn[g] >= 5}
            ent["bigrams_more_in_FP"] = sorted(score, key=lambda g: -score[g])[:25]
            ent["bigrams_more_in_FN"] = sorted(score, key=lambda g: score[g])[:25]
        out[e] = ent
    write_json(os.path.join(rep_dir, "error_analysis.json"), out)
    log(f"   [report] error analysis for {sorted(out)} -> E8_report/error_analysis.json")
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
            ax.set_title("Phase 4 CJPE: dev macro-F1 per experiment and model")
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
