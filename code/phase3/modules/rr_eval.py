"""
rr_eval.py -- Macro-Precision / Recall / F1 from scratch (numpy only), reports, bias tuning.

Macro rule (documented, unit-tested in selftest.py): a class that has ZERO gold support AND
ZERO predictions in the evaluated set is skipped (excluded from the macro average) and listed
in `classes_skipped`. Every other class is included; a class that is predicted but absent
from gold, or present in gold but never predicted, scores P = R = F1 = 0 (0/0 -> 0). This is
what sklearn's f1_score(average="macro") does with labels=None (labels = union of gold and
predicted), which is the most likely behaviour of the official scorer. Macro-F1 is the mean
of per-class F1 (not the harmonic mean of macro-P and macro-R).

Pooling: the brief does not say whether CL and IT are pooled or averaged per domain, so we
report `pooled` (all dev sentences together; the selection metric), `per_domain` and
`domain_avg_macro_f1`.

Post-hoc bias tuning (section 8): an additive per-class logit bias found by coordinate ascent
on a bounded grid. Because dev is also used for early stopping, the real gain is estimated by
2-fold cross-fitting over a 50/50 document split (per domain): fit on one half, measure on the
other, both ways. Biases fitted on the FULL dev are applied to test only if the mean held-out
gain is positive; the decision is recorded.

Inputs : gold / predicted label indices (per document), logits (per document)
Outputs: metric dicts, confusion matrix, bias decision dict
"""
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BIAS_GRID = np.round(np.arange(-3.0, 3.0001, 0.25), 4)


# =============================================================================================
# metrics
# =============================================================================================
def confusion(gold: Sequence[int], pred: Sequence[int], n_classes: int) -> np.ndarray:
    """cm[g, p] = number of sentences with gold g predicted as p."""
    g = np.asarray(gold, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    assert g.shape == p.shape, f"gold {g.shape} vs pred {p.shape}"
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (g, p), 1)
    return cm


def scores_from_confusion(cm: np.ndarray, labels: Sequence[str]) -> Dict:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)          # gold count per class
    n_pred = cm.sum(0).astype(np.float64)           # predicted count per class
    fp, fn = n_pred - tp, support - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    used = (support > 0) | (n_pred > 0)
    total = cm.sum()
    per = {labels[c]: {"precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]),
                       "support": int(support[c]), "predicted": int(n_pred[c]),
                       "in_macro": bool(used[c])} for c in range(len(labels))}
    if not used.any():
        return {"macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "accuracy": 0.0,
                "n": 0, "per_class": per, "classes_used": [], "classes_skipped": list(labels)}
    return {"macro_f1": float(f1[used].mean()), "macro_precision": float(prec[used].mean()),
            "macro_recall": float(rec[used].mean()),
            "accuracy": float(tp.sum() / total) if total else 0.0, "n": int(total),
            "per_class": per,
            "classes_used": [labels[c] for c in range(len(labels)) if used[c]],
            "classes_skipped": [labels[c] for c in range(len(labels)) if not used[c]]}


def macro_scores(gold: Sequence[int], pred: Sequence[int], labels: Sequence[str]) -> Dict:
    return scores_from_confusion(confusion(gold, pred, len(labels)), labels)


def macro_f1(gold: Sequence[int], pred: Sequence[int], n_classes: int) -> float:
    """Fast path (used inside bias tuning): same rule as macro_scores."""
    cm = confusion(gold, pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    sup, npred = cm.sum(1), cm.sum(0)
    denom = sup + npred
    used = denom > 0
    if not used.any():
        return 0.0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)   # = 2PR/(P+R)
    return float(f1[used].mean())


def full_report(gold_by_doc: Dict[str, np.ndarray], pred_by_doc: Dict[str, np.ndarray],
                domain_of: Dict[str, str], labels: Sequence[str]) -> Dict:
    """Pooled + per-domain macro scores and the pooled confusion matrix."""
    keys = sorted(gold_by_doc)
    for k in keys:
        assert len(gold_by_doc[k]) == len(pred_by_doc[k]), f"{k}: length mismatch"
    g = np.concatenate([gold_by_doc[k] for k in keys])
    p = np.concatenate([pred_by_doc[k] for k in keys])
    cm = confusion(g, p, len(labels))
    rep = {"pooled": scores_from_confusion(cm, labels), "per_domain": {},
           "confusion": cm.tolist(), "labels": list(labels)}
    for dom in sorted(set(domain_of[k] for k in keys)):
        ks = [k for k in keys if domain_of[k] == dom]
        rep["per_domain"][dom] = macro_scores(np.concatenate([gold_by_doc[k] for k in ks]),
                                              np.concatenate([pred_by_doc[k] for k in ks]),
                                              labels)
    doms = rep["per_domain"]
    rep["domain_avg_macro_f1"] = float(np.mean([v["macro_f1"] for v in doms.values()])) \
        if doms else 0.0
    return rep


def headline(rep: Dict) -> float:
    """The selection metric: pooled Macro-F1."""
    return float(rep["pooled"]["macro_f1"])


def write_confusion_csv(path: str, cm: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gold\\pred"] + list(labels))
        for lab, row in zip(labels, cm):
            w.writerow([lab] + [int(x) for x in row])


# =============================================================================================
# post-hoc bias tuning
# =============================================================================================
def tune_bias(logits: np.ndarray, gold: np.ndarray, grid: np.ndarray = BIAS_GRID,
              rounds: int = 3) -> Tuple[np.ndarray, float]:
    """Coordinate ascent over an additive per-class bias (bounded grid). Returns (bias, F1).
    Ties keep the value closest to 0, so an uninformative class keeps bias 0."""
    C = logits.shape[1]
    b = np.zeros(C, dtype=np.float64)
    best = macro_f1(gold, (logits + b).argmax(1), C)
    order = np.argsort(np.abs(grid), kind="stable")
    for _ in range(rounds):
        changed = False
        for c in range(C):
            cur_v, cur_f = b[c], best
            for v in grid[order]:
                if v == b[c]:
                    continue
                b2 = b.copy()
                b2[c] = v
                f = macro_f1(gold, (logits + b2).argmax(1), C)
                if f > cur_f + 1e-12:
                    cur_v, cur_f = v, f
            if cur_v != b[c]:
                b[c], best, changed = cur_v, cur_f, True
        if not changed:
            break
    return b.astype(np.float32), float(best)


def halves(keys: Sequence[str], domain_of: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Deterministic 50/50 document split, alternating within each domain."""
    a, b = [], []
    for dom in sorted(set(domain_of[k] for k in keys)):
        ks = sorted(k for k in keys if domain_of[k] == dom)
        a += ks[0::2]
        b += ks[1::2]
    return a, b


def bias_decision(logits_by_doc: Dict[str, np.ndarray], gold_by_doc: Dict[str, np.ndarray],
                  domain_of: Dict[str, str], n_classes: int) -> Dict:
    keys = sorted(gold_by_doc)

    def cat(ks):
        return (np.concatenate([logits_by_doc[k] for k in ks]).astype(np.float64),
                np.concatenate([gold_by_doc[k] for k in ks]))
    A, B = halves(keys, domain_of)
    folds = []
    if A and B:
        for fit, ev in ((A, B), (B, A)):
            lf, gf = cat(fit)
            le, ge = cat(ev)
            bias, _ = tune_bias(lf, gf)
            before = macro_f1(ge, le.argmax(1), n_classes)
            after = macro_f1(ge, (le + bias).argmax(1), n_classes)
            folds.append({"fit_docs": len(fit), "eval_docs": len(ev), "before": before,
                          "after": after, "gain": after - before})
    lf, gf = cat(keys)
    bias_full, f_full = tune_bias(lf, gf)
    before_full = macro_f1(gf, lf.argmax(1), n_classes)
    gain = float(np.mean([f["gain"] for f in folds])) if folds else 0.0
    apply = bool(folds) and gain > 0.0
    return {"apply": apply, "heldout_gain_mean": gain, "folds": folds,
            "bias": bias_full.tolist() if apply else [0.0] * n_classes,
            "bias_fitted_full_dev": bias_full.tolist(), "dev_before": before_full,
            "dev_after_in_sample": f_full,
            "rule": "apply the full-dev biases only if the mean 2-fold held-out gain > 0"}


def labels_from_logits(logits_by_doc: Dict[str, np.ndarray],
                       bias: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    b = np.zeros(1) if bias is None else np.asarray(bias, dtype=np.float64)
    return {k: (v.astype(np.float64) + b).argmax(1).astype(np.int64)
            for k, v in logits_by_doc.items()}
