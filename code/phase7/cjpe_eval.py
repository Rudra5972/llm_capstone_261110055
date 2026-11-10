"""
cjpe_eval.py -- binary Macro-F1 (implemented here, no sklearn), threshold tuning on dev, and the
per-run analysis tables (section 11.1).

Macro-F1 = mean of the F1 of class 0 and class 1 (both always counted: a class that is never
predicted gets F1 = 0, so predicting only the majority class is punished).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from cjpe_data import BUCKETS, bucket_of


def confusion(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, int]:
    g = np.asarray(gold, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    return {"tp": int(((g == 1) & (p == 1)).sum()), "fp": int(((g == 0) & (p == 1)).sum()),
            "fn": int(((g == 1) & (p == 0)).sum()), "tn": int(((g == 0) & (p == 0)).sum())}


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def binary_scores(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    c = confusion(gold, pred)
    p1, r1, f1 = _prf(c["tp"], c["fp"], c["fn"])            # class 1 = positive
    p0, r0, f0 = _prf(c["tn"], c["fn"], c["fp"])            # class 0 viewed as positive
    n = max(1, sum(c.values()))
    return {"macro_f1": (f0 + f1) / 2, "macro_p": (p0 + p1) / 2, "macro_r": (r0 + r1) / 2,
            "accuracy": (c["tp"] + c["tn"]) / n, "f1_class0": f0, "f1_class1": f1,
            "p_class0": p0, "p_class1": p1, "r_class0": r0, "r_class1": r1,
            "precision": p1, "recall": r1, "f1_positive": f1, **c}


def macro_f1(gold, pred) -> float:
    return binary_scores(gold, pred)["macro_f1"]


def tune_tau(p1: Sequence[float], gold: Sequence[int], lo: float = 0.05, hi: float = 0.95,
             step: float = 0.01, min_frac: float = 0.05) -> Dict:
    """Best macro-F1 threshold on dev, keeping >= min_frac of predictions in each class.
    Falls back to 0.5 when no threshold beats it."""
    p = np.asarray(p1, dtype=np.float64)
    g = np.asarray(gold, dtype=np.int64)
    base = macro_f1(g, (p >= 0.5).astype(int))
    best_t, best_f = 0.5, base
    n = len(p)
    for t in np.arange(lo, hi + 1e-9, step):
        pr = (p >= t).astype(int)
        pos = pr.mean() if n else 0
        if pos < min_frac or pos > 1 - min_frac:
            continue
        f = macro_f1(g, pr)
        if f > best_f + 1e-12:
            best_t, best_f = float(round(t, 4)), f
    return {"tau": best_t, "f1_tuned": best_f, "f1_at_0p5": base,
            "used": "tuned" if best_t != 0.5 else "0.5"}


def length_table(gold, pred, n_tokens) -> List[Dict]:
    rows = []
    by: Dict[str, List[int]] = {}
    for i, n in enumerate(n_tokens):
        by.setdefault(bucket_of(n), []).append(i)
    for _, name in BUCKETS:
        idx = by.get(name, [])
        if not idx:
            continue
        s = binary_scores([gold[i] for i in idx], [pred[i] for i in idx])
        rows.append({"bucket": name, "n": len(idx), "macro_f1": s["macro_f1"],
                     "accuracy": s["accuracy"], "pos_rate_gold": float(np.mean([gold[i] for i in idx]))})
    return rows


def reliability(p1, gold, bins: int = 10) -> List[Dict]:
    p = np.asarray(p1, dtype=np.float64)
    g = np.asarray(gold, dtype=np.float64)
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & ((p < hi) if b < bins - 1 else (p <= hi))
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "mean_p1": float(p[m].mean()), "observed_pos_rate": float(g[m].mean())})
    return rows


def ece(p1, gold, bins: int = 10) -> float:
    rel = reliability(p1, gold, bins)
    n = sum(r["n"] for r in rel)
    return float(sum(r["n"] * abs(r["mean_p1"] - r["observed_pos_rate"]) for r in rel) / max(1, n))


def full_report(ids, gold, p1, tau: float, n_tokens) -> Dict:
    pred = [int(x >= tau) for x in p1]
    s = binary_scores(gold, pred)
    s05 = binary_scores(gold, [int(x >= 0.5) for x in p1])
    return {"tau": tau, "scores": s, "scores_at_0p5": s05,
            "by_length": length_table(gold, pred, n_tokens),
            "reliability": reliability(p1, gold), "ece": ece(p1, gold),
            "n": len(ids), "pred_pos_rate": float(np.mean(pred)) if pred else 0.0}
