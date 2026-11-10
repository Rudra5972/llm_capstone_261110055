"""
gen_ml.py -- Phase-6 (abstractive summarisation) text utilities and metrics. Torch-free.

  sentence_spans   split a summary into sentences at TOKEN-SAFE boundaries (whitespace preceded
                   by non-whitespace), so encode(s1)+encode(s2)+... == encode(summary)
  align_sentences  DANCER-style alignment (Gidiotis & Tsoumakas 2020): every reference sentence
                   goes to the document chunk with the highest content-word recall
  rouge_n / rouge_l  pure-Python ROUGE-1/2/L F1 (lower-cased alphanumeric tokens, no stemmer);
                   LCS by bit-parallel big-integer arithmetic (Hyyroe 2004) -> fast on long texts
  bertscore        BERTScore F1 via the `bert-score` package if importable (roberta-large, the
                   package default for lang="en"); returns None + reason otherwise (never raises)
  diagnostics      length, compression, repeated-trigram rate (redundancy), novel bigrams
                   (abstractiveness), source support of content words, number / citation
                   precision against the source (factuality proxy)
  length_curve     dev-tuned word cap: truncating at sentence boundaries to r x median train
                   summary length; the best r on dev is applied to test (generation length control)
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

WORD_RE = re.compile(r"[a-z0-9]+")
SENT_END_RE = re.compile(r"[.!?;:][\"')\]]*(\s+)|\n\s*\n(\s*)")
NUM_RE = re.compile(r"\b\d[\d,./-]*\d\b|\b\d\b")
CITE_RE = re.compile(
    r"(?i)\b(?:section|sec\.|s\.|sections|article|art\.|rule|order)\s*\d+[a-z]?\b"
    r"|\b(?:i\.?\s?p\.?\s?c|cr\.?\s?p\.?\s?c|c\.?\s?p\.?\s?c)\b|\bact,?\s*(?:18|19|20)\d\d\b")
STOP = set("""a an the of to in and or for on at by with from as is are was were be been being
it its this that these those which who whom whose what when where why how not no nor but if then
than so such into onto upon over under about above below between through during before after
he she they them his her their we our you your i me my also any all each other some can could
would should may might must shall will has have had do does did there here very more most""".split())


def words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> List[str]:
    return [w for w in words(text) if w not in STOP and (len(w) > 2 or w.isdigit())]


# =============================================================================================
# sentences + alignment
# =============================================================================================
def sentence_spans(text: str) -> List[str]:
    """Pieces whose concatenation is exactly `text`; every cut is at a whitespace character that
    follows a non-whitespace character (a BPE pre-token boundary)."""
    cuts = []
    for m in SENT_END_RE.finditer(text):
        g = 1 if m.group(1) is not None else 2
        p = m.start(g) if m.group(g) else m.start()
        while p > 0 and text[p - 1].isspace():
            p -= 1
        if 0 < p < len(text) and text[p].isspace() and not text[p - 1].isspace():
            if not cuts or p > cuts[-1]:
                cuts.append(p)
    out, s = [], 0
    for c in cuts:
        out.append(text[s:c])
        s = c
    out.append(text[s:])
    return [x for x in out if x]


def align_sentences(sent_texts: Sequence[str], chunk_texts: Sequence[str]) -> List[int]:
    """Chunk index for every sentence: argmax of content-word recall (ties -> earliest chunk);
    sentences without content words follow the previous sentence."""
    csets = [set(content_words(t)) for t in chunk_texts]
    out, prev = [], 0
    for s in sent_texts:
        w = set(content_words(s))
        if not w or not csets:
            out.append(prev)
            continue
        sc = [len(w & c) for c in csets]
        best = max(range(len(sc)), key=lambda i: (sc[i], -i))
        prev = best if sc[best] > 0 else prev
        out.append(prev)
    return out


# =============================================================================================
# ROUGE
# =============================================================================================
def _ngrams(t: List[str], n: int) -> Counter:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def _prf(hit: float, n_c: int, n_r: int) -> Dict[str, float]:
    p = hit / n_c if n_c else 0.0
    r = hit / n_r if n_r else 0.0
    return {"p": p, "r": r, "f": (2 * p * r / (p + r)) if p + r > 0 else 0.0}


def rouge_n(cand: str, ref: str, n: int = 1) -> Dict[str, float]:
    c, r = _ngrams(words(cand), n), _ngrams(words(ref), n)
    hit = sum((c & r).values())
    return _prf(hit, sum(c.values()), sum(r.values()))


def lcs_len(a: List[str], b: List[str]) -> int:
    """Bit-parallel LCS length (Hyyroe 2004): V <- (V + (V & M_t)) | (V - (V & M_t))."""
    if not a or not b:
        return 0
    masks: Dict[str, int] = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    full = (1 << len(a)) - 1
    V = full
    for t in b:
        U = V & masks.get(t, 0)
        V = ((V + U) | (V - U)) & full
    return len(a) - bin(V).count("1")


def rouge_l(cand: str, ref: str) -> Dict[str, float]:
    c, r = words(cand), words(ref)
    return _prf(lcs_len(c, r), len(c), len(r))


def rouge_all(cands: List[str], refs: List[str]) -> Dict[str, float]:
    if not cands:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    r1 = [rouge_n(c, r, 1)["f"] for c, r in zip(cands, refs)]
    r2 = [rouge_n(c, r, 2)["f"] for c, r in zip(cands, refs)]
    rl = [rouge_l(c, r)["f"] for c, r in zip(cands, refs)]
    m = lambda x: float(sum(x) / len(x))  # noqa: E731
    return {"rouge1": m(r1), "rouge2": m(r2), "rougeL": m(rl), "rouge1_each": r1,
            "rougeL_each": rl}


# =============================================================================================
# BERTScore
# =============================================================================================
_SCORER = {}


def bertscore(cands: List[str], refs: List[str], device: str = "cpu", batch_size: int = 16,
              log=print) -> Tuple[Optional[List[float]], str]:
    """(per-pair F1 list, info) or (None, reason). Uses the package defaults for lang='en'
    (roberta-large, layer 17, no baseline rescaling) like a default evaluator would."""
    if not cands:
        return None, "no candidates"
    try:
        from bert_score import BERTScorer
    except Exception as e:                                   # package missing
        return None, f"bert-score not importable ({type(e).__name__})"
    try:
        key = (device, batch_size)
        if key not in _SCORER:
            _SCORER[key] = BERTScorer(lang="en", device=device, batch_size=batch_size)
        c = [x if x.strip() else "." for x in cands]
        r = [x if x.strip() else "." for x in refs]
        _p, _r, f = _SCORER[key].score(c, r, verbose=False)
        vals = [float(v) for v in f.tolist()]
        if any(math.isnan(v) for v in vals):
            return None, "BERTScore returned NaN"
        return vals, "bert-score roberta-large (lang=en)"
    except Exception as e:                                   # no internet / OOM / ...
        return None, f"bert-score failed: {type(e).__name__}: {str(e)[:200]}"


def free_bertscore() -> None:
    _SCORER.clear()


# =============================================================================================
# length control + diagnostics
# =============================================================================================
def cap_words(text: str, n_words: Optional[int]) -> str:
    """Keep whole sentences while the word count stays <= n_words (always >= 1 sentence)."""
    if not n_words or len(words(text)) <= n_words:
        return text
    out, n = [], 0
    for s in sentence_spans(text):
        k = len(words(s))
        if out and n + k > n_words:
            break
        out.append(s)
        n += k
    return "".join(out).strip()


LENGTH_RATIOS = (0.5, 0.75, 1.0, 1.25, None)          # x median train summary words; None = no cap


def diagnostics(summ: str, source: Optional[str] = None, ref: Optional[str] = None) -> Dict:
    w = words(summ)
    tri = _ngrams(w, 3)
    n_tri = sum(tri.values())
    d = {"words": len(w), "sentences": len(sentence_spans(summ.strip())) if summ.strip() else 0,
         "repeated_trigram_rate": (1.0 - len(tri) / n_tri) if n_tri else 0.0}
    if source is not None:
        sw = words(source)
        sb = set(_ngrams(sw, 2))
        cb = list(_ngrams(w, 2))
        d["compression"] = len(w) / max(1, len(sw))
        d["novel_bigram_rate"] = (sum(1 for b in cb if b not in sb) / len(cb)) if cb else 0.0
        cw = content_words(summ)
        sset = set(sw)
        d["content_support"] = (sum(1 for x in cw if x in sset) / len(cw)) if cw else 1.0
        nums = set(NUM_RE.findall(summ))
        snums = set(NUM_RE.findall(source))
        d["number_precision"] = (len(nums & snums) / len(nums)) if nums else None
        cites = {re.sub(r"\s+", " ", c.lower()) for c in CITE_RE.findall(summ)}
        scites = {re.sub(r"\s+", " ", c.lower()) for c in CITE_RE.findall(source)}
        d["citation_precision"] = (len(cites & scites) / len(cites)) if cites else None
    if ref is not None:
        rw = set(content_words(ref))
        d["ref_content_recall"] = (len(rw & set(content_words(summ))) / len(rw)) if rw else None
        d["ref_words"] = len(words(ref))
    return d


def mean_of(rows: List[Dict], key: str) -> Optional[float]:
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(sum(v) / len(v)) if v else None


def primary(rec: Dict) -> Tuple[Optional[float], str]:
    """The model-selection score of a dev evaluation: BERTScore F1 when available, else the mean
    of ROUGE-1 and ROUGE-L F1 (reported with its name so runs are never mixed silently)."""
    if rec.get("bertscore_f1") is not None:
        return float(rec["bertscore_f1"]), "bertscore_f1"
    if rec.get("rouge1") is not None:
        return 0.5 * (float(rec["rouge1"]) + float(rec["rougeL"])), "rouge1_rougeL_mean"
    return None, "none"


def consensus_pick(cands: List[str]) -> int:
    """MBR-style ensemble: index of the candidate with the highest mean ROUGE-L F1 to the others."""
    if len(cands) <= 1:
        return 0
    best, bi = -1.0, 0
    for i, c in enumerate(cands):
        s = sum(rouge_l(c, o)["f"] for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
        if s > best:
            best, bi = s, i
    return bi
