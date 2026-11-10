"""
bpe_tokenizer.py -- from-scratch byte-level BPE (pure Python, stdlib + optional numpy/torch).

No `tokenizers`, `transformers`, `sentencepiece`, `tiktoken` or any other tokenizer library.
Only: re, heapq, json, collections, random, time, gc.

Token-id layout (matches the original notebook's special-token ids):
    0 <|endoftext|>   1 <|pad|>   2 <|unk|>   3..258 = raw bytes 0..255   259.. = merges
Internally training works with "internal ids" (byte b -> b, merge k -> 256+k);
the public API adds NUM_SPECIAL (=3) to every id.
"""
import gc
import heapq
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
NUM_SPECIAL = len(SPECIALS)

# GPT-2 style pre-tokenisation, expressed with the stdlib `re` module
# (\p{L} -> [^\W\d_],  \p{N} -> \d,  punctuation/symbols -> [^\s\w] or '_').
PRETOKEN_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"      # English contractions
    r"| ?[^\W\d_]+"               # optional space + letters
    r"| ?\d+"                     # optional space + digits
    r"| ?(?:[^\s\w]|_)+"          # optional space + punctuation / symbols
    r"|\s+(?!\S)"                 # whitespace run not followed by a non-space
    r"|\s+"                       # remaining whitespace
)

MAX_PRETOKEN_CHARS = 512          # pathological runs (e.g. 50k dashes) are cut into pieces


# =============================================================================
# 1. TRAINING
# =============================================================================
def count_pretokens(texts, max_unique=6_000_000, max_word_chars=64, log_every=20000,
                    log=print):
    """Stream documents -> Counter of pre-tokens, with a hard memory ceiling.

    When the Counter grows past `max_unique`, singletons (then rarer and rarer words) are
    pruned, so RAM stays bounded no matter how large the corpus is.
    """
    counter = Counter()
    findall = PRETOKEN_RE.findall
    n_docs = n_chars = 0
    t0 = time.time()
    for text in texts:
        counter.update(findall(text))
        n_docs += 1
        n_chars += len(text)
        if len(counter) > max_unique:
            thr = 2
            while len(counter) > int(max_unique * 0.7):
                counter = Counter({k: v for k, v in counter.items() if v >= thr})
                thr += 1
            gc.collect()
            log(f"   [count] pruned to {len(counter):,} unique pre-tokens (min count {thr - 1})")
        if n_docs % log_every == 0:
            log(f"   [count] {n_docs:,} docs | {n_chars/1e6:,.0f}M chars | "
                f"{len(counter):,} unique | {time.time()-t0:.0f}s")
    # drop absurdly long pre-tokens (junk); they carry no useful merge statistics
    if max_word_chars:
        for k in [k for k in counter if len(k) > max_word_chars]:
            del counter[k]
    return counter


def train_bpe(word_counts, vocab_size, max_words=2_000_000, min_word_count=2,
              min_pair_count=2, n_verify=5000, seed=0, log_every=1000, log=print):
    """Learn BPE merges from a Counter of pre-tokens.

    Classic incremental algorithm: exact pair counts + inverted index (pair -> words that
    contain it) + lazy max-heap, so each merge only touches the words that contain the pair.

    Returns (merges, verify) where
        merges : list[(a, b)] in rank order, in INTERNAL ids (byte b -> b, merge k -> 256+k)
        verify : list[(pretoken_str, final_internal_ids)] used to cross-check the encoder
    """
    num_merges = vocab_size - NUM_SPECIAL - 256
    assert num_merges > 0, "vocab_size too small"

    # ---- 1. pick the training vocabulary of words ---------------------------------
    items = [(w, c) for w, c in word_counts.items() if c >= min_word_count]
    if len(items) > max_words:
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:max_words]
    words, freqs, strs = [], [], []
    for w, c in items:
        ids = list(w.encode("utf-8", "replace"))
        if len(ids) > 1:
            words.append(ids); freqs.append(c); strs.append(w)
    del items
    log(f"   [bpe] training on {len(words):,} unique words "
        f"({sum(freqs)/1e6:,.1f}M word occurrences)")

    # ---- 2. initial pair statistics ------------------------------------------------
    pc = {}        # pair -> total count
    where = {}     # pair -> set(word index)
    for idx, ids in enumerate(words):
        f = freqs[idx]
        for p in zip(ids, ids[1:]):
            if p in pc:
                pc[p] += f
                where[p].add(idx)
            else:
                pc[p] = f
                where[p] = {idx}
    heap = [(-c, p) for p, c in pc.items()]
    heapq.heapify(heap)

    merges = []
    t0 = time.time()
    next_id = 256
    while next_id < 256 + num_merges:
        # ---- pick the most frequent pair (lazy heap: skip stale entries) -----------
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pc.get(p, 0) == -negc:
                best = (-negc, p)
                break
        if best is None:
            log("   [bpe] no more pairs to merge -- stopping early")
            break
        count, (a, b) = best
        if count < min_pair_count:
            log(f"   [bpe] best pair count {count} < {min_pair_count} -- stopping early")
            break

        merges.append((a, b))
        new = next_id
        next_id += 1

        # ---- apply the merge to every word that contains it ------------------------
        idxs = where.pop((a, b), ())
        changed = {}                      # pair -> count before this merge
        for idx in idxs:
            ids = words[idx]
            n = len(ids)
            out = []
            i = 0
            hit = False
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new); i += 2; hit = True
                else:
                    out.append(ids[i]); i += 1
            if not hit:
                continue
            f = freqs[idx]
            old_pairs = list(zip(ids, ids[1:]))
            new_pairs = list(zip(out, out[1:]))
            for q in old_pairs:
                if q not in changed:
                    changed[q] = pc.get(q, 0)
                pc[q] -= f
            for q in new_pairs:
                if q not in changed:
                    changed[q] = pc.get(q, 0)
                pc[q] = pc.get(q, 0) + f
            new_set = set(new_pairs)
            for q in set(old_pairs) - new_set:
                s = where.get(q)
                if s is not None:
                    s.discard(idx)
            for q in new_set:
                s = where.get(q)
                if s is None:
                    where[q] = {idx}
                else:
                    s.add(idx)
            words[idx] = out

        # ---- publish the changed pair counts ---------------------------------------
        for q, orig in changed.items():
            c = pc.get(q, 0)
            if c <= 0:
                pc.pop(q, None)
                where.pop(q, None)
            elif c != orig:
                heapq.heappush(heap, (-c, q))
        # keep the heap from growing without bound
        if len(heap) > max(4_000_000, 8 * len(pc)):
            heap = [(-c, q) for q, c in pc.items()]
            heapq.heapify(heap)

        k = len(merges)
        if k % log_every == 0 or k == num_merges:
            log(f"   [bpe] merge {k:,}/{num_merges:,} | last pair count {count:,} | "
                f"{len(pc):,} live pairs | {time.time()-t0:.0f}s")

    # ---- 3. sample of final segmentations (to verify the encoder) ----------------
    rng = random.Random(seed)
    pick = rng.sample(range(len(words)), min(n_verify, len(words)))
    verify = [(strs[i], list(words[i])) for i in pick]
    del words, freqs, strs, pc, where, heap
    gc.collect()
    return merges, verify


# =============================================================================
# 2. TOKENIZER (encode / decode / save / load)
# =============================================================================
class BPETokenizer:
    """Byte-level BPE tokenizer with a small HuggingFace-like surface:
    tok(text, add_special_tokens=False)["input_ids"], tok.encode, tok.decode,
    len(tok), eos_token_id / pad_token_id, padding_side / truncation_side.
    """

    def __init__(self, merges, cache_max=2_000_000):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = NUM_SPECIAL + 256 + len(self.merges)
        self._merge_map = {m: 256 + r for r, m in enumerate(self.merges)}   # pair -> internal id
        # id -> bytes (internal ids)
        self._bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._special_bytes = [s.encode("utf-8") for s in SPECIALS]

        self._cache = {}
        self._cache_max = cache_max

        self.eos_token, self.bos_token = SPECIALS[0], SPECIALS[0]
        self.pad_token, self.unk_token = SPECIALS[1], SPECIALS[2]
        self.eos_token_id = self.bos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        self.padding_side = "left"        # decoder-only batched generation
        self.truncation_side = "left"     # an over-long prompt loses its beginning

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        v = {s: i for i, s in enumerate(SPECIALS)}
        for i, bs in enumerate(self._bytes):
            v[bs.decode("latin-1") + f"#{i}"] = i + NUM_SPECIAL   # unique readable key
        return v

    # ------------------------------------------------------------------ encoding
    def _bpe_word(self, b):
        """BPE-merge one pre-token given as bytes -> tuple of PUBLIC ids."""
        ids = list(b)
        mm = self._merge_map
        while len(ids) > 1:
            best = None
            bp = None
            for p in zip(ids, ids[1:]):
                m = mm.get(p)
                if m is not None and (best is None or m < best):
                    best, bp = m, p
            if best is None:
                break
            a, c = bp
            out = []
            i, n = 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == c:
                    out.append(best); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        return tuple(i + NUM_SPECIAL for i in ids)

    def _encode_pretoken(self, w):
        if len(w) > MAX_PRETOKEN_CHARS:
            res = ()
            for s in range(0, len(w), MAX_PRETOKEN_CHARS):
                res += self._bpe_word(w[s:s + MAX_PRETOKEN_CHARS].encode("utf-8", "replace"))
            return res
        return self._bpe_word(w.encode("utf-8", "replace"))

    def encode(self, text, add_special_tokens=False):
        cache = self._cache
        out = []
        ext = out.extend
        for w in PRETOKEN_RE.findall(text):
            t = cache.get(w)
            if t is None:
                t = self._encode_pretoken(w)
                if len(cache) >= self._cache_max:
                    cache.clear()                 # bounded RAM
                cache[w] = t
            ext(t)
        return out

    def __call__(self, text, add_special_tokens=False, padding=False, truncation=False,
                 max_length=None, return_tensors=None, **_ignored):
        single = isinstance(text, str)
        batch = [text] if single else list(text)
        enc = [self.encode(t) for t in batch]
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                enc = [e[-max_length:] if len(e) > max_length else e for e in enc]
            else:
                enc = [e[:max_length] for e in enc]
        masks = [[1] * len(e) for e in enc]
        if padding:
            target = max_length if (padding == "max_length" and max_length) else max(len(e) for e in enc)
            for k, e in enumerate(enc):
                gap = target - len(e)
                if gap > 0:
                    if self.padding_side == "left":
                        enc[k] = [self.pad_token_id] * gap + e
                        masks[k] = [0] * gap + masks[k]
                    else:
                        enc[k] = e + [self.pad_token_id] * gap
                        masks[k] = masks[k] + [0] * gap
        if return_tensors in ("pt", "np"):
            if return_tensors == "pt":
                import torch
                mk = lambda x: torch.tensor(x, dtype=torch.long)
            else:
                import numpy as np
                mk = lambda x: np.asarray(x, dtype=np.int64)
            return {"input_ids": mk(enc), "attention_mask": mk(masks)}
        if single:
            return {"input_ids": enc[0], "attention_mask": masks[0]}
        return {"input_ids": enc, "attention_mask": masks}

    # ------------------------------------------------------------------ decoding
    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        parts = []
        nb = len(self._bytes)
        for i in ids:
            i = int(i)
            if i < NUM_SPECIAL:
                if not skip_special_tokens and i >= 0:
                    parts.append(self._special_bytes[i])
            elif i - NUM_SPECIAL < nb:
                parts.append(self._bytes[i - NUM_SPECIAL])
        return b"".join(parts).decode("utf-8", errors="replace")

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens) for s in seqs]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        obj = {"type": "byte_level_bpe", "version": 1, "specials": SPECIALS,
               "num_special": NUM_SPECIAL, "vocab_size": self.vocab_size,
               "pretoken_regex": PRETOKEN_RE.pattern, "merges": self.merges}
        with open(path, "w") as f:
            json.dump(obj, f)

    def save_pretrained(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save(out_dir / "bpe_model.json")
        with open(out_dir / "tokenizer_config.json", "w") as f:
            json.dump({"eos_token": self.eos_token, "pad_token": self.pad_token,
                       "unk_token": self.unk_token, "eos_token_id": 0, "pad_token_id": 1,
                       "unk_token_id": 2, "vocab_size": self.vocab_size,
                       "padding_side": self.padding_side,
                       "truncation_side": self.truncation_side}, f, indent=2)

    @classmethod
    def load(cls, path, **kw):
        with open(path) as f:
            obj = json.load(f)
        assert obj.get("type") == "byte_level_bpe", "not a byte_level_bpe file"
        assert obj["pretoken_regex"] == PRETOKEN_RE.pattern, "pre-tokenizer regex mismatch"
        return cls(obj["merges"], **kw)

    @classmethod
    def from_pretrained(cls, directory, **kw):
        return cls.load(Path(directory) / "bpe_model.json", **kw)
