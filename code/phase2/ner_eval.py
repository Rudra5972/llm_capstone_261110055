"""
ner_eval.py -- decoding + strict span metrics + error analysis (numpy only, no torch).

Tag layout (shared with ner_data.make_tag_names): 0 = O, B-X = 1 + 2k, I-X = 2 + 2k.

Decoding options (section 5.5 of the brief), all applied to document-level scores:
  * argmax
  * constrained Viterbi: only legal IOB2 transitions (O->{O,B-*}, B-X->{O,B-*,I-X},
    I-X->{O,B-*,I-X}; a sequence starts with O or B-*). For a linear head the scores are
    log-probabilities and the transition matrix is 0 / -inf; for a CRF head the learned
    transitions are added.
  * O-shift `delta`: subtract delta from the O score (delta > 0 -> more entities). Tuned on dev.

Metric (section 2): an entity counts only if (start, end, label) match exactly.
  per label P/R/F1, then two macro averages: over all labels, and over labels present in gold.
"""
import csv
import os
from collections import Counter, defaultdict

import numpy as np

NEG = -1e9


# =============================================================================================
# decoding
# =============================================================================================
def transition_constraints(n_labels):
    """allowed[prev, cur] and allowed_start[cur] for IOB2 with n_labels entity types."""
    C = 1 + 2 * n_labels
    allowed = np.ones((C, C), dtype=bool)
    for k in range(n_labels):
        i_tag = 2 + 2 * k
        allowed[:, i_tag] = False
        allowed[1 + 2 * k, i_tag] = True               # B-X -> I-X
        allowed[i_tag, i_tag] = True                   # I-X -> I-X
    start = np.ones(C, dtype=bool)
    start[2::2] = False                                # cannot start with I-*
    return allowed, start


def viterbi(scores, allowed, allowed_start, trans=None, start_scores=None, end_scores=None):
    """Best tag path for one document. scores: (T, C) float. Returns int64 (T,)."""
    T, C = scores.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    tr = np.where(allowed, 0.0 if trans is None else trans, NEG)
    s0 = np.where(allowed_start, 0.0, NEG)
    if start_scores is not None:
        s0 = s0 + start_scores
    score = s0 + scores[0]
    back = np.zeros((T, C), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + tr                     # (prev, cur)
        back[t] = cand.argmax(0)
        score = cand.max(0) + scores[t]
    if end_scores is not None:
        score = score + end_scores
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_scores(scores, td, labels, mode="viterbi", o_shift=0.0, is_logprob=True,
                  crf=None, snap=None, text=None):
    """Document scores (T, C) -> [(start, end, label)].

    is_logprob : True for averaged probabilities converted to log (linear head);
                 crf : dict with numpy "transitions", "start", "end" (CRF head) or None.
    """
    from ner_data import decode_tags
    sc = np.array(scores, dtype=np.float64, copy=True)
    if o_shift:
        sc[:, 0] -= o_shift
    if mode == "argmax":
        tags = sc.argmax(1)
    else:
        allowed, a_start = transition_constraints(len(labels))
        if crf is not None:
            tags = viterbi(sc, allowed, a_start, crf["transitions"], crf["start"], crf["end"])
        else:
            tags = viterbi(sc, allowed, a_start)
    spans = decode_tags(tags, td, labels)
    if snap is not None and text is not None:
        spans = snap.apply(spans, text)
    return spans


# =============================================================================================
# metrics
# =============================================================================================
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def strict_scores(pred, gold, labels):
    """pred/gold: {doc_id: [(start, end, label)]} -> metrics dict."""
    cnt = {l: Counter() for l in labels}
    for did in set(gold) | set(pred):
        p = {tuple(x[:3]) for x in pred.get(did, [])}
        g = {tuple(x[:3]) for x in gold.get(did, [])}
        for x in p & g:
            cnt[x[2]]["tp"] += 1
        for x in p - g:
            if x[2] in cnt:
                cnt[x[2]]["fp"] += 1
        for x in g - p:
            cnt[x[2]]["fn"] += 1
    per = {}
    for l in labels:
        c = cnt[l]
        p, r, f = _prf(c["tp"], c["fp"], c["fn"])
        per[l] = {"precision": p, "recall": r, "f1": f, "tp": c["tp"], "fp": c["fp"],
                  "fn": c["fn"], "support": c["tp"] + c["fn"]}
    present = [l for l in labels if per[l]["support"] > 0]

    def macro(ls):
        if not ls:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {k: float(np.mean([per[l][k] for l in ls])) for k in ("precision", "recall", "f1")}

    tp = sum(c["tp"] for c in cnt.values())
    fp = sum(c["fp"] for c in cnt.values())
    fn = sum(c["fn"] for c in cnt.values())
    mp, mr, mf = _prf(tp, fp, fn)
    return {"macro_all": macro(labels), "macro_present": macro(present),
            "micro": {"precision": mp, "recall": mr, "f1": mf}, "per_label": per,
            "n_gold": tp + fn, "n_pred": tp + fp}


def headline(m):
    """The selection metric: strict macro-F1 over all labels."""
    return m["macro_all"]["f1"]


# =============================================================================================
# error analysis (section 10)
# =============================================================================================
def error_analysis(pred, gold, texts, labels, n_tokens=None):
    """Per-entity taxonomy, boundary offsets, span-level confusion, F1 by length bucket."""
    tax = Counter()
    d_start, d_end = Counter(), Counter()
    conf = defaultdict(Counter)                    # gold label -> predicted label ("O" = none)
    rows = []
    for did in sorted(set(gold) | set(pred)):
        g = sorted({tuple(x[:3]) for x in gold.get(did, [])})
        p = sorted({tuple(x[:3]) for x in pred.get(did, [])})
        pset = set(p)
        used = set()
        text = texts.get(did, "")
        for (s, e, l) in g:
            if (s, e, l) in pset:
                tax["exact"] += 1
                conf[l][l] += 1
                used.add((s, e, l))
                continue
            same_span = [q for q in p if q[0] == s and q[1] == e]
            ov_same = [q for q in p if q[0] < e and s < q[1] and q[2] == l]
            ov_any = [q for q in p if q[0] < e and s < q[1]]
            if same_span:
                kind = "right_span_wrong_label"
                q = same_span[0]
                conf[l][q[2]] += 1
            elif ov_same:
                kind = "right_label_wrong_boundary"
                q = min(ov_same, key=lambda q: abs(q[0] - s) + abs(q[1] - e))
                d_start[q[0] - s] += 1
                d_end[q[1] - e] += 1
                conf[l][l + "~boundary"] += 1
            elif ov_any:
                kind = "wrong_label_and_boundary"
                q = ov_any[0]
                conf[l][q[2] + "~boundary"] += 1
            else:
                kind = "missed"
                q = None
                conf[l]["O"] += 1
            tax[kind] += 1
            if q is not None:
                used.add(q)
            rows.append({"doc": did, "kind": kind, "gold_start": s, "gold_end": e,
                         "gold_label": l, "gold_text": text[s:e],
                         "pred_start": q[0] if q else "", "pred_end": q[1] if q else "",
                         "pred_label": q[2] if q else "", "pred_text": text[q[0]:q[1]] if q else ""})
        for q in p:
            if q in used or q in set(g):
                continue
            if not any(q[0] < e and s < q[1] for (s, e, _l) in g):
                tax["spurious"] += 1
                conf["O"][q[2]] += 1
                rows.append({"doc": did, "kind": "spurious", "gold_start": "", "gold_end": "",
                             "gold_label": "", "gold_text": "", "pred_start": q[0],
                             "pred_end": q[1], "pred_label": q[2], "pred_text": text[q[0]:q[1]]})
    # F1 by document length bucket (characters), micro, because buckets hold few documents
    buckets = [(0, 20_000), (20_000, 50_000), (50_000, 100_000), (100_000, 10 ** 12)]
    by_len = {}
    for lo, hi in buckets:
        ds = [d for d in gold if lo <= len(texts.get(d, "")) < hi]
        if not ds:
            continue
        sc = strict_scores({d: pred.get(d, []) for d in ds}, {d: gold[d] for d in ds}, labels)
        by_len[f"{lo}-{hi if hi < 10 ** 12 else 'inf'}"] = {"docs": len(ds),
                                                             "micro_f1": sc["micro"]["f1"],
                                                             "macro_f1": sc["macro_all"]["f1"]}
    return {"taxonomy": dict(tax),
            "boundary_start_offsets": {str(k): v for k, v in sorted(d_start.items())},
            "boundary_end_offsets": {str(k): v for k, v in sorted(d_end.items())},
            "confusion": {k: dict(v) for k, v in conf.items()},
            "f1_by_doc_length_chars": by_len}, rows


def write_errors_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["doc", "kind", "gold_start", "gold_end", "gold_label", "gold_text", "pred_start",
            "pred_end", "pred_label", "pred_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (v.replace("\n", "\\n") if isinstance(v, str) else v)
                        for k, v in r.items()})


def tune_decoding(dev_scores, dev_docs, gold, labels, is_logprob=True, crf=None, snap=None,
                  shifts=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """Pick (mode, o_shift) maximising dev strict macro-F1. Returns (best, table)."""
    from ner_tok import TokenizedDoc
    table = []
    tds = {d: TokenizedDoc.from_dict(dev_docs[d]) for d in dev_scores}
    for mode in ("argmax", "viterbi"):
        for delta in (shifts if mode == "viterbi" else (0.0,)):
            pred = {d: decode_scores(dev_scores[d], tds[d], labels, mode, delta, is_logprob,
                                     crf, snap, dev_docs[d]["text"]) for d in dev_scores}
            m = strict_scores(pred, gold, labels)
            table.append({"mode": mode, "o_shift": delta, "macro_f1": headline(m),
                          "micro_f1": m["micro"]["f1"], "macro_present_f1": m["macro_present"]["f1"]})
    best = max(table, key=lambda r: (r["macro_f1"], r["mode"] == "viterbi", -abs(r["o_shift"])))
    return best, table
