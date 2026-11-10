"""
ner_tok.py -- character offsets for the Phase-1 byte-level BPE tokenizer.

bpe_tokenizer.py (copied verbatim, never modified) only returns token ids. NER needs, for every
token, which characters of the ORIGINAL string it covers. This module re-runs exactly the
tokenizer's own steps (regex pre-tokenisation -> 512-char pieces -> BPE merges per piece) and
tracks byte positions, so the ids are guaranteed identical to `tok.encode(text)` (asserted).

Per token we return
  raw span  [raw_start, raw_end)  : the characters the token *owns* (see rule below)
  eff span  [eff_start, eff_end)  : the raw span with leading/trailing whitespace trimmed
                                    (pre-tokens carry a leading space: " Court" -> "Court")
  -1 / -1                         : the token owns no character / is whitespace only

Ownership rule (byte-level BPE can split a multi-byte UTF-8 character across two tokens):
a character is owned by the token that contains its FIRST byte. A token holding only
continuation bytes owns no character (raw span -1, -1); ner_data.py gives such a token the
label of the token before it.
"""
import bisect

import numpy as np

from bpe_tokenizer import MAX_PRETOKEN_CHARS, NUM_SPECIAL, PRETOKEN_RE


class TokenizedDoc:
    """Token ids + character spans for one document (numpy int32 arrays, -1 = none)."""
    __slots__ = ("ids", "raw_start", "raw_end", "eff_start", "eff_end")

    def __init__(self, ids, raw_start, raw_end, eff_start, eff_end):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.raw_start = np.asarray(raw_start, dtype=np.int32)
        self.raw_end = np.asarray(raw_end, dtype=np.int32)
        self.eff_start = np.asarray(eff_start, dtype=np.int32)
        self.eff_end = np.asarray(eff_end, dtype=np.int32)

    def __len__(self):
        return int(self.ids.shape[0])

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        return cls(*(d[k] for k in cls.__slots__))


def _piece_tokens(tok, piece):
    """BPE one piece (<= MAX_PRETOKEN_CHARS chars).

    Returns a tuple of (id, raw_s, raw_e, eff_s, eff_e) in PIECE-LOCAL character coordinates.
    """
    b = piece.encode("utf-8", "replace")
    ids = tok._bpe_word(b)                                   # public ids, same call as encode()
    # byte offset at which each character starts ("replace" turns a lone surrogate into b"?",
    # exactly like the whole-string encode above, so lengths agree)
    char_byte_start, pos = [], 0
    for ch in piece:
        char_byte_start.append(pos)
        pos += len(ch.encode("utf-8", "replace"))
    assert pos == len(b), "per-character UTF-8 lengths disagree with the whole-piece encoding"

    out, bpos = [], 0
    for tid in ids:
        nb = len(tok._bytes[tid - NUM_SPECIAL])
        b0, b1 = bpos, bpos + nb
        bpos = b1
        c0 = bisect.bisect_left(char_byte_start, b0)        # first char starting at/after b0
        c1 = bisect.bisect_left(char_byte_start, b1)        # first char starting at/after b1
        if c0 >= c1:                                         # only continuation bytes
            out.append((tid, -1, -1, -1, -1))
            continue
        e0, e1 = c0, c1
        while e0 < e1 and piece[e0].isspace():
            e0 += 1
        while e1 > e0 and piece[e1 - 1].isspace():
            e1 -= 1
        if e0 >= e1:                                         # whitespace-only token
            out.append((tid, c0, c1, -1, -1))
        else:
            out.append((tid, c0, c1, e0, e1))
    assert bpos == len(b), "token bytes do not add up to the piece's bytes"
    return tuple(out)


class OffsetEncoder:
    """Wraps a BPETokenizer; caches per-pre-token results (pre-tokens repeat a lot)."""

    def __init__(self, tok, cache_max=500_000):
        self.tok = tok
        self._cache = {}
        self._cache_max = cache_max

    def _pretoken(self, w):
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        # identical piece rule to BPETokenizer._encode_pretoken
        if len(w) > MAX_PRETOKEN_CHARS:
            res, base = [], 0
            for s in range(0, len(w), MAX_PRETOKEN_CHARS):
                for (tid, a, b, c, d) in _piece_tokens(self.tok, w[s:s + MAX_PRETOKEN_CHARS]):
                    shift = lambda v: v + s if v >= 0 else -1
                    res.append((tid, shift(a), shift(b), shift(c), shift(d)))
            res = tuple(res)
        else:
            res = _piece_tokens(self.tok, w)
        if len(self._cache) >= self._cache_max:
            self._cache.clear()
        self._cache[w] = res
        return res

    def encode(self, text, check=True):
        ids, rs, re_, es, ee = [], [], [], [], []
        for m in PRETOKEN_RE.finditer(text):
            base = m.start()
            for (tid, a, b, c, d) in self._pretoken(m.group()):
                ids.append(tid)
                rs.append(a + base if a >= 0 else -1)
                re_.append(b + base if b >= 0 else -1)
                es.append(c + base if c >= 0 else -1)
                ee.append(d + base if d >= 0 else -1)
        if check:
            ref = self.tok.encode(text)
            if ref != ids:
                raise AssertionError("offset encoder produced different ids than tok.encode()")
        return TokenizedDoc(ids, rs, re_, es, ee)


def encode_with_offsets(tok, text, check=True):
    """Convenience one-shot API (builds a throw-away cache)."""
    return OffsetEncoder(tok).encode(text, check=check)


def verify_offsets(text, td):
    """Raw spans must partition the text in order: every character owned exactly once.

    Returns the number of problems found (0 = OK). Also checks that eff spans lie inside raw
    spans and contain no leading/trailing whitespace.
    """
    problems = 0
    nxt = 0
    for i in range(len(td)):
        a, b = int(td.raw_start[i]), int(td.raw_end[i])
        if a < 0:
            continue
        if a != nxt or b <= a:
            problems += 1
        nxt = b
        c, d = int(td.eff_start[i]), int(td.eff_end[i])
        if c >= 0:
            if not (a <= c < d <= b) or text[c].isspace() or text[d - 1].isspace():
                problems += 1
        elif not text[a:b].isspace():
            problems += 1
    if nxt != len(text):
        problems += 1
    return problems
