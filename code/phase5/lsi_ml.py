"""
lsi_ml.py -- Phase 5 (Legal Statute Identification) multi-label core: targets, losses,
metrics, dev-only decision rules and the per-run analysis (brief sections 3.3, 3.5, 5).

Labels inside the pipeline are multi-hot uint8 rows over the TRAIN label map (`label_ids`,
sorted statute ids seen in train.json). Statute ids that occur only in dev are never predicted
(the evaluator only accepts valid ids) and are reported as `dev_only_labels`.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

GRID = np.round(np.arange(0.02, 0.901, 0.02), 4)
RARE, FREQ = 5, 20                       # train-support buckets: <5 rare, 5-20 medium, >20 freq


# =============================================================================================
# labels / targets
# =============================================================================================
def multihot(label_lists: Sequence, lab2i: Dict[int, int], L: int) -> np.ndarray:
    Y = np.zeros((len(label_lists), L), np.uint8)
    for r, labs in enumerate(label_lists):
        for x in labs or []:
            j = lab2i.get(int(x))
            if j is not None:
                Y[r, j] = 1
    return Y


def targets(labels, idx, device) -> torch.Tensor:
    return torch.from_numpy(np.stack([labels[i] for i in idx]).astype(np.float32)).to(device)


def pos_weight(Y: np.ndarray, mode: str = "sqrt_inv", cap: float = 30.0) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    pos = Y.sum(0).astype(np.float64)
    neg = len(Y) - pos
    r = neg / np.maximum(pos, 1.0)
    w = np.sqrt(r) if mode == "sqrt_inv" else r
    w = np.clip(w, 1.0, cap)
    w[pos == 0] = 1.0
    return w.astype(np.float32)


def prior_logit(Y: np.ndarray) -> np.ndarray:
    p = (Y.sum(0).astype(np.float64) + 0.5) / (len(Y) + 1.0)
    p = np.clip(p, 1e-4, 0.5)
    return np.log(p / (1 - p)).astype(np.float32)


# =============================================================================================
# losses (per document = sum over labels; fp32)
# =============================================================================================
def ml_loss(logits: torch.Tensor, Y: torch.Tensor, pw: Optional[torch.Tensor] = None,
            kind: str = "bce") -> torch.Tensor:
    x = logits.float()
    if kind == "asl":                    # asymmetric loss (Ridnik et al. 2021): g-=4, g+=0, clip
        p = torch.sigmoid(x)
        pm = (p - 0.05).clamp(min=0)
        lp = Y * torch.log(p.clamp(min=1e-8))
        ln = (1 - Y) * torch.log((1 - pm).clamp(min=1e-8)) * pm.pow(4)
        return -(lp + ln).sum(1)
    if kind == "focal":                  # focal BCE, gamma 2, with pos_weight on positives
        b = F.binary_cross_entropy_with_logits(x, Y, reduction="none")
        p = torch.sigmoid(x)
        pt = Y * p + (1 - Y) * (1 - p)
        w = (1 - pt).pow(2)
        if pw is not None:
            w = w * (1 + (pw - 1) * Y)
        return (w * b).sum(1)
    return F.binary_cross_entropy_with_logits(x, Y, pos_weight=pw, reduction="none").sum(1)


def kd_loss_ml(s_logits, t_logits, Y, alpha=0.5, T=2.0, pw=None) -> torch.Tensor:
    """a * BCE(student, gold) + (1-a) * T^2 * BCE(student/T, sigmoid(teacher/T)); mean/doc."""
    s = s_logits.float()
    hard = ml_loss(s, Y, pw).mean()
    soft = F.binary_cross_entropy_with_logits(s / T, torch.sigmoid(t_logits.float() / T),
                                              reduction="none").sum(1).mean()
    return alpha * hard + (1 - alpha) * T * T * soft


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))).astype(np.float32)


# =============================================================================================
# metrics
# =============================================================================================
def counts(G: np.ndarray, P: np.ndarray):
    G = G.astype(bool)
    P = P.astype(bool)
    tp = (G & P).sum(0).astype(np.int64)
    fp = (~G & P).sum(0).astype(np.int64)
    fn = (G & ~P).sum(0).astype(np.int64)
    return tp, fp, fn


def _f1(tp, fp, fn):
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f


def ml_scores(G: np.ndarray, P: np.ndarray, train_support: Optional[np.ndarray] = None,
              n_dev_only_pos: int = 0) -> Dict:
    """macro_f1 = mean F1 over labels with dev support (what an evaluator built from the dev
    reference would see); macro_f1_all = over every train label (0 for unseen+unpredicted)."""
    tp, fp, fn = counts(G, P)
    p, r, f = _f1(tp, fp, fn)
    sup = G.sum(0) > 0
    out = {"macro_f1": float(f[sup].mean()) if sup.any() else 0.0,
           "macro_p": float(p[sup].mean()) if sup.any() else 0.0,
           "macro_r": float(r[sup].mean()) if sup.any() else 0.0,
           "macro_f1_all": float(f.mean()) if len(f) else 0.0}
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum()) + int(n_dev_only_pos)
    mp = TP / max(1, TP + FP)
    mr = TP / max(1, TP + FN)
    out.update({"micro_f1": 2 * mp * mr / max(1e-12, mp + mr), "micro_p": mp, "micro_r": mr,
                "subset_acc": float((G.astype(bool) == P.astype(bool)).all(1).mean())
                if len(G) else 0.0,
                "mean_pred_labels": float(P.sum(1).mean()) if len(P) else 0.0,
                "mean_gold_labels": float(G.sum(1).mean()) if len(G) else 0.0,
                "n_labels_with_dev_support": int(sup.sum()), "tp": TP, "fp": FP, "fn": FN})
    if train_support is not None:
        for name, m in (("rare", train_support < RARE),
                        ("medium", (train_support >= RARE) & (train_support <= FREQ)),
                        ("frequent", train_support > FREQ)):
            mm = m & sup
            out[f"macro_f1_{name}"] = float(f[mm].mean()) if mm.any() else None
            out[f"n_{name}"] = int(mm.sum())
    return out


def macro(G, P) -> float:
    tp, fp, fn = counts(G, P)
    f = _f1(tp, fp, fn)[2]
    sup = G.sum(0) > 0
    return float(f[sup].mean()) if sup.any() else 0.0


# =============================================================================================
# decision rules (dev only)
# =============================================================================================
def apply_rule(Pr: np.ndarray, taus: np.ndarray, kmax: int) -> np.ndarray:
    """threshold per label, then >= 1 label per document (top-1) and <= kmax labels."""
    pred = Pr >= taus[None, :]
    if len(Pr) == 0:
        return pred
    top1 = Pr.argmax(1)
    pred[np.arange(len(Pr)), top1] = True
    over = np.where(pred.sum(1) > kmax)[0]
    for i in over:
        keep = np.argsort(-np.where(pred[i], Pr[i], -1.0))[:kmax]
        pred[i] = False
        pred[i, keep] = True
    return pred


def fit_global(Pr, G, kmax) -> (float, float):
    best_t, best_f = 0.5, -1.0
    L = Pr.shape[1]
    for t in GRID:
        f = macro(G, apply_rule(Pr, np.full(L, t, np.float32), kmax))
        if f > best_f + 1e-12:
            best_t, best_f = float(t), f
    return best_t, best_f


def fit_perlabel(Pr, G, tg: float, m: int = 5, lam: float = 0.5) -> np.ndarray:
    L = Pr.shape[1]
    taus = np.full(L, tg, np.float32)
    sup = G.sum(0)
    cand = np.where(sup >= m)[0]
    if not len(cand):
        return taus
    best_f = np.full(len(cand), -1.0)
    best_t = np.full(len(cand), tg)
    Gc = G[:, cand].astype(bool)
    for t in GRID:
        Pc = Pr[:, cand] >= t
        tp = (Gc & Pc).sum(0)
        fp = (~Gc & Pc).sum(0)
        fn = (Gc & ~Pc).sum(0)
        f = _f1(tp, fp, fn)[2]
        better = f > best_f + 1e-12
        best_f[better] = f[better]
        best_t[better] = t
    taus[cand] = lam * best_t + (1 - lam) * tg
    return taus


def choose_rule(Pr: np.ndarray, G: np.ndarray, kmax: int, seed: int = 0,
                allow_perlabel: bool = True) -> Dict:
    L = Pr.shape[1]
    tg, fg = fit_global(Pr, G, kmax)
    taus_pl = fit_perlabel(Pr, G, tg)
    f_pl_in = macro(G, apply_rule(Pr, taus_pl, kmax))
    # honest estimate: 2-fold cross-fitting on dev
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Pr))
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    cv_g, cv_p = [], []
    for a, b in ((0, 1), (1, 0)):
        A, B = halves[a], halves[b]
        if len(A) < 4 or len(B) < 4:
            continue
        t_a, _ = fit_global(Pr[A], G[A], kmax)
        pl_a = fit_perlabel(Pr[A], G[A], t_a)
        cv_g.append(macro(G[B], apply_rule(Pr[B], np.full(L, t_a, np.float32), kmax)))
        cv_p.append(macro(G[B], apply_rule(Pr[B], pl_a, kmax)))
    cvg = float(np.mean(cv_g)) if cv_g else None
    cvp = float(np.mean(cv_p)) if cv_p else None
    use_pl = bool(allow_perlabel and cvg is not None and cvp is not None and cvp > cvg + 0.002)
    taus = taus_pl if use_pl else np.full(L, tg, np.float32)
    return {"name": "perlabel" if use_pl else "global", "tau_global": tg, "kmax": int(kmax),
            "taus": [round(float(x), 4) for x in taus],
            "dev_macro_global": fg, "dev_macro_perlabel_insample": f_pl_in,
            "cv_macro_global": cvg, "cv_macro_perlabel": cvp}


def rule_taus(rule: Dict, L: int) -> np.ndarray:
    t = np.asarray(rule.get("taus") or [rule.get("tau_global", 0.5)] * L, np.float32)
    return t if len(t) == L else np.full(L, float(rule.get("tau_global", 0.5)), np.float32)


# =============================================================================================
# per-run analysis
# =============================================================================================
def per_label_rows(G, P, taus, label_ids, train_support) -> List[Dict]:
    tp, fp, fn = counts(G, P)
    p, r, f = _f1(tp, fp, fn)
    return [{"label": int(label_ids[j]), "train_support": int(train_support[j]),
             "dev_support": int(G[:, j].sum()), "precision": float(p[j]), "recall": float(r[j]),
             "f1": float(f[j]), "predicted_count": int(P[:, j].sum()), "tau": float(taus[j])}
            for j in range(G.shape[1])]


def confused_pairs(G, P, label_ids, top: int = 30) -> List[Dict]:
    from collections import Counter
    c = Counter()
    Gb, Pb = G.astype(bool), P.astype(bool)
    for i in range(len(G)):
        miss = np.where(Gb[i] & ~Pb[i])[0]
        spur = np.where(Pb[i] & ~Gb[i])[0]
        for a in miss[:20]:
            for b in spur[:20]:
                c[(int(label_ids[a]), int(label_ids[b]))] += 1
    return [{"missed": a, "predicted_instead": b, "docs": n} for (a, b), n in c.most_common(top)]


def length_rows(G, P, n_tokens) -> List[Dict]:
    from cjpe_data import BUCKETS, bucket_of
    by: Dict[str, List[int]] = {}
    for i, n in enumerate(n_tokens):
        by.setdefault(bucket_of(n), []).append(i)
    rows = []
    for _, name in BUCKETS:
        idx = by.get(name, [])
        if idx:
            s = ml_scores(G[idx], P[idx])
            rows.append({"bucket": name, "n": len(idx), "macro_f1": s["macro_f1"],
                         "micro_f1": s["micro_f1"], "mean_gold_labels": s["mean_gold_labels"]})
    return rows


def ece_ml(Pr, G, bins: int = 10, max_n: int = 2_000_000) -> float:
    p = Pr.ravel()
    g = G.ravel().astype(np.float64)
    if len(p) > max_n:
        sel = np.random.default_rng(0).choice(len(p), max_n, replace=False)
        p, g = p[sel], g[sel]
    e, n = 0.0, len(p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & ((p < hi) if b < bins - 1 else (p <= hi))
        if m.any():
            e += m.sum() * abs(float(p[m].mean()) - float(g[m].mean()))
    return float(e / max(1, n))


def full_report(G, Pr, rule, n_tokens, label_ids, train_support, n_dev_only_pos=0) -> Dict:
    L = Pr.shape[1]
    taus = rule_taus(rule, L)
    P = apply_rule(Pr, taus, rule["kmax"])
    P05 = apply_rule(Pr, np.full(L, 0.5, np.float32), rule["kmax"])
    Pg = apply_rule(Pr, np.full(L, rule["tau_global"], np.float32), rule["kmax"])
    return {"rule": {k: v for k, v in rule.items() if k != "taus"},
            "scores": ml_scores(G, P, train_support, n_dev_only_pos),
            "scores_at_0p5": ml_scores(G, P05, train_support, n_dev_only_pos),
            "scores_global": ml_scores(G, Pg, train_support, n_dev_only_pos),
            "by_length": length_rows(G, P, n_tokens),
            "confused_pairs": confused_pairs(G, P, label_ids),
            "ece": ece_ml(Pr, G), "n": int(len(G)),
            "_P": P, "_per_label": per_label_rows(G, P, taus, label_ids, train_support)}
