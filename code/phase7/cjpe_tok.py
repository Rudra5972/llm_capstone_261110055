"""
cjpe_tok.py -- fast head/tail tokenisation, the token cache and the long-document
representations (section 5 of the brief).

Why slices: the Phase-1 BPE is pure Python. A judgment can be 50k+ tokens but every
representation we train on only needs the first and/or last few 511-token chunks, so we cut
character slices at *token boundaries* and tokenise only those.

Token-boundary rule (proved in the docstring of `cut_head` and tested in selftest): the
pre-tokeniser regex never lets a non-whitespace run continue into whitespace, so position p is
a pre-token boundary of the FULL text whenever text[p] is whitespace and text[p-1] is not. The
regex only looks forward, so tokenising text[:p] gives exactly the full-text tokens before p,
and tokenising text[p:] gives exactly the full-text tokens from p on.

Representations (`doc_repr`), each chunk = [EOS] + <= 511 body tokens (<= 512 total):
  hier     floor(K/2) first chunks + ceil(K/2) last chunks of the document, in order (K=1 ->
           the LAST chunk: the court's conclusion is at the end; Malik et al. 2021 (ILDC) found
           the last 512 tokens the most informative single window)
  first_k  first K chunks          last_k   last K chunks
  headtail ONE sequence of <= 1024 tokens: [EOS] first 383 [EOS] last 639
  sections first chunk + 2 chunks starting at the last "conclusion-like" heading in the last
           70% of the text (ORDER / CONCLUSION / HELD / RESULT / "for the foregoing reasons"
           ...), falling back to `hier` when there is none
  full     consecutive chunks from the start, up to 16 (8176 tokens; ablation only)
  spread   (Phase 5) K chunks evenly spaced over the cached head (up to 16 chunks) + the last
           chunk of the document, in order
  cite     (Phase 5) first chunk + 2 chunks starting at the densest window of statute citations
           ("section 302", "Article 14", "I.P.C.", "Act, 1956" ...) + last chunk; falls back to
           `hier` when the document has no citation
"""
import hashlib
import json
import os
import pickle
import re
import time
from multiprocessing import get_context
from typing import Dict, List, Optional

import numpy as np

from cjpe_common import file_md5, write_json

BODY = 511                     # body tokens per chunk; + EOS = 512
EOS_ID, PAD_ID = 0, 1
HT_HEAD, HT_TAIL = 383, 639    # headtail: 1 + 383 + 1 + 639 = 1024
CACHE_VERSION = 3
CITE_RE = re.compile(
    r"(?i)\b(?:section|sec\.|s\.|sections|article|art\.|rule|order)\s*\d+[a-z]?\b"
    r"|\b(?:i\.?\s?p\.?\s?c|cr\.?\s?p\.?\s?c|c\.?\s?p\.?\s?c|constitution)\b"
    r"|\bact,?\s*(?:18|19|20)\d\d\b")
SECTION_RE = re.compile(
    r"(?im)^\s*(?:ORDER|CONCLUSION|CONCLUSIONS|HELD|RESULT|DECISION|OPERATIVE PORTION)\b"
    r"|for the (?:foregoing|aforesaid|above|reasons) (?:reasons|stated|discussed)"
    r"|in the (?:result|circumstances)\b|we (?:accordingly )?(?:allow|dismiss|set aside)\b")

_TOK = None                    # per-process tokenizer (pool workers)


# =============================================================================================
# boundary-safe slicing
# =============================================================================================
def _ws(c: str) -> bool:
    return c.isspace()


def cut_head(text: str, j: int) -> int:
    """Smallest p >= j that is a safe boundary (text[p] whitespace, text[p-1] not); len(text)
    if none. text[:p] then tokenises exactly like the first tokens of the full text."""
    n = len(text)
    p = max(1, j)
    while p < n and not (_ws(text[p]) and not _ws(text[p - 1])):
        p += 1
    return min(p, n)


def cut_tail(text: str, i: int) -> int:
    """Largest p <= i that is a safe boundary; 0 if none (then the whole text is used)."""
    p = min(i, len(text) - 1)
    while p > 0 and not (_ws(text[p]) and not _ws(text[p - 1])):
        p -= 1
    return max(p, 0)


def _init_worker(bpe_dir: str) -> None:
    global _TOK
    from bpe_tokenizer import BPETokenizer
    _TOK = BPETokenizer.from_pretrained(bpe_dir)


def _encode(text: str) -> List[int]:
    return _TOK.encode(text)


def encode_doc(text: str, need_head: int, need_tail: int, want_sections: bool = False,
               need_sect: int = 2 * BODY, tok=None) -> Dict:
    """Head/tail token ids of one document; exact token count when the doc was fully encoded."""
    enc = (tok or _TOK).encode
    L = len(text)
    need = need_head + need_tail
    out = {"n_chars": L, "sect": None}
    full_thr = need * 9                              # short documents: encode everything
    if L <= full_thr:
        ids = enc(text)
        out.update(complete=True, head=np.asarray(ids, np.uint16), tail=None, n_tokens=len(ids))
    else:
        hc = cut_head(text, need_head * 6)
        while True:
            head = enc(text[:hc])
            if len(head) >= need_head or hc >= L:
                break
            hc = cut_head(text, hc * 2)
        tc = cut_tail(text, L - need_tail * 6)
        while True:
            tail = enc(text[tc:])
            if len(tail) >= need_tail or tc <= 0:
                break
            tc = cut_tail(text, L - (L - tc) * 2)
        if hc >= tc:                                 # slices meet: encode the whole document
            ids = enc(text)
            out.update(complete=True, head=np.asarray(ids, np.uint16), tail=None,
                       n_tokens=len(ids))
        else:
            seen_chars = hc + (L - tc)
            est = (len(head) + len(tail)) / max(1, seen_chars) * L
            out.update(complete=False, head=np.asarray(head[:need_head], np.uint16),
                       tail=np.asarray(tail[-need_tail:] if need_tail else [], np.uint16),
                       n_tokens=int(round(est)))
    if want_sections:
        pos = None
        for m in SECTION_RE.finditer(text):
            if m.start() >= 0.3 * L:
                pos = m.start()
        if pos is not None:
            s = cut_tail(text, pos) if pos > 0 else 0
            e = cut_head(text, s + need_sect * 6)
            ids = enc(text[s:e])
            out["sect"] = np.asarray(ids[:need_sect], np.uint16)
        # Phase 5: densest window of statute citations (window = ~2 chunks of characters)
        pos = [m.start() for m in CITE_RE.finditer(text)]
        out["cite"] = None
        if pos:
            win = need_sect * 5
            best, bi, j = 0, 0, 0
            for i in range(len(pos)):
                while pos[i] - pos[j] > win:
                    j += 1
                if i - j + 1 > best:
                    best, bi = i - j + 1, j
            s = cut_tail(text, max(0, pos[bi] - 200)) if pos[bi] > 200 else 0
            e = cut_head(text, s + need_sect * 6)
            ids = enc(text[s:e])
            out["cite"] = np.asarray(ids[:need_sect], np.uint16)
            out["n_cites"] = len(pos)
    return out


def _encode_doc_star(args):
    return encode_doc(*args)


def _count(text: str) -> int:
    return len(_TOK.encode(text))


def _pool(bpe_dir: str, n_proc: int):
    return get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(bpe_dir,))


def count_tokens_many(texts: List[str], tok, n_proc: int = 1) -> List[int]:
    if n_proc <= 1 or len(texts) < 50 or not getattr(tok, "_bpe_dir", None):
        return [len(tok.encode(t)) for t in texts]
    with _pool(tok._bpe_dir, n_proc) as p:
        return p.map(_count, texts, chunksize=8)


# =============================================================================================
# the cache
# =============================================================================================
def cache_key(file_md5s: Dict[str, str], tok_md5: str) -> str:
    """Base key: data files + tokenizer (+ CACHE_VERSION). The head/tail sizes and the sections
    flag are part of the FILE name, so a larger cache (e.g. from 00) serves smaller needs."""
    s = json.dumps({"v": CACHE_VERSION, "files": file_md5s, "tok": tok_md5}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def cache_name(key: str, need_head: int, need_tail: int, sections: bool) -> str:
    return f"cache_{key}_h{need_head}_t{need_tail}_s{int(bool(sections))}.pkl"


def build_cache(data: Dict[str, List[Dict]], bpe_dir: str, out_dir: str, key: str,
                need_head: int, need_tail: int, sections: bool, n_proc: int, log=print) -> str:
    """Tokenise every document once; writes <out_dir>/cache_<key>.pkl + cache_meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, cache_name(key, need_head, need_tail, sections))
    t0 = time.time()
    res = {}
    for split, docs in data.items():
        args = [(d["text"], need_head, need_tail, sections) for d in docs]
        if n_proc > 1 and len(args) >= 64:
            with _pool(bpe_dir, n_proc) as p:
                enc = p.map(_encode_doc_star, args, chunksize=16)
        else:
            _init_worker(bpe_dir)
            enc = [_encode_doc_star(a) for a in args]
        rows = []
        for d, e in zip(docs, enc):
            e["id"], e["label"] = d["id"], d["label"]
            rows.append(e)
        res[split] = rows
        log(f"   [cache] {split}: {len(rows)} docs tokenised ({time.time() - t0:.0f}s so far)")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, path)
    write_json(os.path.join(out_dir, "cache_meta.json"),
               {"key": key, "file": os.path.basename(path), "need_head": need_head,
                "need_tail": need_tail, "sections": sections, "seconds": time.time() - t0,
                "n": {s: len(v) for s, v in res.items()}})
    return path


def find_cache(key: str, need_head: int, need_tail: int, sections: bool,
               roots: List[str]) -> Optional[str]:
    """Any cache of the same data + tokenizer that holds at least the needed tokens."""
    pat = re.compile(rf"^cache_{re.escape(key)}_h(\d+)_t(\d+)_s([01])\.pkl$")
    best = None
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for dp, _dn, fns in os.walk(r):
            for fn in fns:
                m = pat.match(fn)
                if not m:
                    continue
                h, t, sc = int(m.group(1)), int(m.group(2)), m.group(3) == "1"
                if h >= need_head and t >= need_tail and (sc or not sections):
                    size = h + t
                    if best is None or size < best[0]:
                        best = (size, os.path.join(dp, fn))
    return best[1] if best else None


def load_cache(path: str) -> Dict[str, List[Dict]]:
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================================
# representations
# =============================================================================================
def _with_eos(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, np.int64), a.astype(np.int64)])


def _first_chunks(ids: np.ndarray, n: int) -> List[np.ndarray]:
    return [ids[i * BODY:(i + 1) * BODY] for i in range(n) if len(ids[i * BODY:(i + 1) * BODY])]


def _last_chunks(ids: np.ndarray, n: int) -> List[np.ndarray]:
    out = []
    L = len(ids)
    for j in range(n, 0, -1):                  # in document order
        s, e = max(0, L - j * BODY), L - (j - 1) * BODY
        if e > s:
            out.append(ids[s:e])
    return out


def doc_chunks(e: Dict, repr_: str, K: int) -> List[np.ndarray]:
    """The list of model inputs (each starts with EOS, <= 512 tokens; headtail <= 1024)."""
    head = e["head"]
    tail = e["tail"] if e["tail"] is not None else head
    complete = bool(e["complete"])
    n_tok = len(head) if complete else None
    if repr_ == "headtail":
        if complete and n_tok <= 1023:
            return [_with_eos(head)]
        h = head[:HT_HEAD]
        t = tail[-HT_TAIL:]
        return [np.concatenate([_with_eos(h), _with_eos(t)])]
    if repr_ == "cite" and e.get("cite") is not None and len(e["cite"]):
        parts = _first_chunks(head, 1) + _first_chunks(e["cite"], 2) + _last_chunks(tail, 1)
        return [_with_eos(p) for p in parts if len(p)]
    if repr_ == "spread":
        n_all = -(-len(head) // BODY)
        if complete and n_all <= K:
            return [_with_eos(p) for p in _first_chunks(head, n_all)]
        avail = _first_chunks(head, min(n_all, 16))
        whole = complete and n_all <= 16             # avail already ends at the document end
        k = max(1, min(K if whole else K - 1, len(avail)))
        pick = sorted({int(round(i * (len(avail) - 1) / max(1, k - 1))) for i in range(k)})
        parts = [avail[i] for i in pick] + ([] if whole else _last_chunks(tail, 1))
        return [_with_eos(p) for p in parts if len(p)]
    if repr_ == "sections" and e.get("sect") is not None and len(e["sect"]):
        parts = _first_chunks(head, 1) + _first_chunks(e["sect"], 2)
        return [_with_eos(p) for p in parts]
    if repr_ in ("hier", "sections", "cite"):
        a, b = K // 2, K - K // 2
    elif repr_ == "first_k":
        a, b = K, 0
    elif repr_ == "last_k":
        a, b = 0, K
    elif repr_ == "full":
        return [_with_eos(p) for p in _first_chunks(head, K)]
    else:
        raise ValueError(f"unknown doc_repr {repr_!r}")
    if complete:
        ids = head
        n_all = -(-len(ids) // BODY)
        if n_all <= a + b:                       # short document: every chunk, in order
            return [_with_eos(p) for p in _first_chunks(ids, n_all)]
        parts = _first_chunks(ids, a) + _last_chunks(ids, b)
    else:
        parts = _first_chunks(head, a) + _last_chunks(tail, b)
    return [_with_eos(p) for p in parts if len(p)]


def needs_for(repr_: str, K: int):
    """(need_head, need_tail) tokens the cache must hold for a representation."""
    if repr_ == "full":
        return K * BODY, 0
    if repr_ == "headtail":
        return 2 * BODY, 2 * BODY
    if repr_ == "first_k":
        return K * BODY, BODY
    if repr_ == "last_k":
        return BODY, K * BODY
    a, b = K // 2, K - K // 2
    return max(a, 1) * BODY, max(b, 1) * BODY
