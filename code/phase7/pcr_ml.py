"""
pcr_ml.py -- torch-free retrieval utilities for Phase 7 (Prior Case Retrieval).

  * word tokenizer + BM25 (Robertson/Sparck-Jones idf, k1/b saturation, log(1+qtf) query
    weights, optional bigrams) over ONE split's candidate pool, as a dense (n_query, n_pool)
    float32 score matrix (scipy.sparse product)
  * micro_prf_at_k: Micro-P/R/F1@k exactly as the PDF defines it (pooled over every query's
    top-k against the FULL gold set; gold sets are neither truncated nor padded)
  * per-query z-normalisation, score fusion (dense x BM25), top-k with self-exclusion,
    reciprocal-rank fusion (E8 ensemble), average precision (dev proxy)
"""
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

K_EVAL = 7
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an the and or of to in on at by for with from as is are was were be been being
it its this that these those which who whom whose what when where why how not no nor but if then
than so such into upon under over said shall may can could would should will has have had do does
did also any all each other there their them they he she his her him we our you your i me my
herein thereof therein whereas thus hence very only same own between before after above below""".split())


def words(text: str, ngram: int = 1) -> List[str]:
    w = [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in STOP]
    if ngram >= 2:
        w = w + [f"{a}_{b}" for a, b in zip(w, w[1:])]
    return w


class BM25:
    """Fit on one pool; score(query texts) -> (nq, n_pool) float32."""

    def __init__(self, pool_texts: Sequence[str], k1: float = 1.2, b: float = 0.75,
                 ngram: int = 1, pool_tokens: Optional[List[List[str]]] = None):
        from scipy import sparse
        self.k1, self.b, self.ngram = float(k1), float(b), int(ngram)
        toks = pool_tokens if pool_tokens is not None else [words(t, ngram) for t in pool_texts]
        self.vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, ws in enumerate(toks):
            for w, c in Counter(ws).items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n, V = len(toks), max(1, len(self.vocab))
        tf = sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(n, V))
        dl = np.asarray([len(ws) for ws in toks], np.float32)
        avg = float(dl.mean()) if n else 1.0
        df = np.bincount(np.asarray(cols, np.int64), minlength=V).astype(np.float32)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * dl[tf.row] / max(avg, 1e-9))
        w = tf.data * (self.k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        self.D = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=(n, V))

    def query_matrix(self, texts: Sequence[str], tokens: Optional[List[List[str]]] = None):
        from scipy import sparse
        rows, cols, vals = [], [], []
        toks = tokens if tokens is not None else [words(t, self.ngram) for t in texts]
        for i, ws in enumerate(toks):
            for w, c in Counter(ws).items():
                j = self.vocab.get(w)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(np.log1p(c))
        return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                                 shape=(len(toks), self.D.shape[1]))

    def score(self, texts: Sequence[str], tokens=None) -> np.ndarray:
        return np.asarray((self.query_matrix(texts, tokens) @ self.D.T).todense(), np.float32)


# =============================================================================================
# ranking + metrics
# =============================================================================================
def mask_self(S: np.ndarray, q_ids: Sequence[str], pool_ids: Sequence[str]) -> np.ndarray:
    """A query never retrieves itself (when it also sits in the pool)."""
    pos = {p: j for j, p in enumerate(pool_ids)}
    S = S.astype(np.float32, copy=True)
    for i, q in enumerate(q_ids):
        j = pos.get(q)
        if j is not None:
            S[i, j] = -np.inf
    return S


def topk(S: np.ndarray, k: int = K_EVAL) -> np.ndarray:
    """(nq, k) pool indices, best first (ties broken by index -> deterministic)."""
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(S, idx, 1)
    order = np.lexsort((idx, -part), axis=1)
    return np.take_along_axis(idx, order, 1)


def micro_prf_at_k(ranked: Sequence[Sequence], gold: Sequence[Iterable], k: int = K_EVAL
                   ) -> Dict[str, float]:
    tp = n_pred = n_gold = 0
    for r, g in zip(ranked, gold):
        g = set(g)
        r = list(r)[:k]
        tp += len(set(r) & g)
        n_pred += len(r)
        n_gold += len(g)
    p = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold if n_gold else 0.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"micro_p": p, "micro_r": rc, "micro_f1": f, "tp": tp, "n_pred": n_pred,
            "n_gold": n_gold}


def eval_scores(S: np.ndarray, gold: Sequence[Iterable], k: int = K_EVAL) -> Dict[str, float]:
    return micro_prf_at_k(topk(S, k).tolist(), gold, k)


def average_precision(scores: np.ndarray, gold: Iterable) -> float:
    g = set(gold)
    if not g:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits, s = 0, 0.0
    for rnk, j in enumerate(order, 1):
        if int(j) in g:
            hits += 1
            s += hits / rnk
            if hits == len(g):
                break
    return s / len(g)


def zrow(S: np.ndarray) -> np.ndarray:
    """Per-query z-score over the finite entries (masked entries stay -inf)."""
    S = S.astype(np.float32, copy=True)
    fin = np.isfinite(S)
    X = np.where(fin, S, np.nan)
    mu = np.nanmean(X, 1, keepdims=True)
    sd = np.nanstd(X, 1, keepdims=True)
    Z = (X - mu) / np.maximum(sd, 1e-6)
    return np.where(fin, Z, -np.inf).astype(np.float32)


def fuse(dense: np.ndarray, bm25: Optional[np.ndarray], w: float) -> np.ndarray:
    """w * z(dense) + (1 - w) * z(bm25); w = 1 -> dense only."""
    if bm25 is None or w >= 1.0:
        return zrow(dense)
    if w <= 0.0:
        return zrow(bm25)
    zd, zb = zrow(dense), zrow(bm25)
    fin = np.isfinite(zd) & np.isfinite(zb)
    return np.where(fin, w * np.where(fin, zd, 0) + (1 - w) * np.where(fin, zb, 0),
                    -np.inf).astype(np.float32)


def tune_fusion(dense: np.ndarray, bm25: Optional[np.ndarray], gold, weights: Sequence[float],
                k: int = K_EVAL) -> Tuple[float, Dict, List[Dict]]:
    curve = []
    for w in weights:
        m = eval_scores(fuse(dense, bm25, w), gold, k)
        curve.append({"dense_weight": float(w), **m})
    best = max(curve, key=lambda r: (r["micro_f1"], r["dense_weight"]))
    return best["dense_weight"], best, curve


def rrf(rank_lists: List[List[List[int]]], n_pool: int, k_rrf: int = 60, depth: int = 100
        ) -> np.ndarray:
    """Reciprocal-rank fusion of several systems' rankings -> (nq, n_pool) scores."""
    nq = len(rank_lists[0])
    S = np.zeros((nq, n_pool), np.float32)
    for sysr in rank_lists:
        for i, r in enumerate(sysr):
            for rnk, j in enumerate(r[:depth], 1):
                S[i, j] += 1.0 / (k_rrf + rnk)
    return S


def full_rank(S: np.ndarray, depth: int = 100) -> List[List[int]]:
    return topk(S, min(depth, S.shape[1])).tolist()
