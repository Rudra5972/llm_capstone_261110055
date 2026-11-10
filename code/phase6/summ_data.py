"""
summ_data.py -- Phase-6 data: schema-tolerant loading, leak guard, held-out dev split, the token
cache and the long-document representations.

Data (PDF 12.3): Train/train.json [{id, document, summary}], Test/test.json [{id, document}].
No dev split is provided -> a deterministic dev split is carved from train by a hash of the
NORMALISED document text (exact duplicates always land on the same side; same split in every
notebook and account). `dev_gen` = the first N dev documents in hash order: the subset that is
actually decoded for ROUGE / BERTScore (generation is the slow part).

Representations (`doc_repr`), every sequence = [EOS] + document part + separator + summary [EOS]
<= 1024 - n_virtual tokens:
  head      first D document tokens, one sequence, summary budget S
  headtail  first 60% + last 40% of D (the judgment's conclusion is at the end)
  dancer    divide-and-conquer (Gidiotis & Tsoumakas 2020): the document is cut into chunks of C
            tokens; every reference sentence is aligned to the chunk with the highest
            content-word recall; chunk i is trained to produce ITS sentences ("" -> EOS only),
            separator "SUMMARY (part i of n):"; at test time the chunk outputs are concatenated
            in order. Lets a 1024-token model write long summaries of long judgments.
  auto      dancer when the median train summary is longer than `auto_long` tokens, else headtail
"""
import hashlib
import json
import os
import pickle
import re
import time
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import numpy as np

from cjpe_common import file_md5, write_json
from cjpe_data import (DataError, _detect_keys, _records, discover_files, leak_guard,
                       length_stats, norm_hash)
from gen_ml import align_sentences, sentence_spans

SUMMARY_KEYS = ("summary", "summaries", "abstract", "headnote", "reference", "target",
                "ref_summary", "gold_summary")
MAX_DOC_CHARS = 250_000
CACHE_VERSION = 1
SEP_TEXT = "\n\nSUMMARY:\n"
PART_TEXT = "\n\nSUMMARY (part {i} of {n}):\n"
_TOK = None


# =============================================================================================
# loading
# =============================================================================================
def load_summ_split(path: str, split: str, log=print) -> Tuple[List[Dict], Dict]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    recs, keyed = _records(obj)
    if not recs:
        raise DataError(f"{split}: no records in {path}")
    for r in recs:                                    # keep the summary out of the text guess
        if not isinstance(r, dict):
            raise DataError(f"{split}: records are not JSON objects")
    sk = next((k for k in SUMMARY_KEYS if all(k in r for r in recs[:200])), None)
    view = [{k: v for k, v in r.items() if k != sk} for r in recs[:200]]
    idk, tk, _ = _detect_keys(view, keyed)
    if split == "train" and sk is None:
        raise DataError(f"train: no summary key among {sorted(recs[0].keys())}")
    out = []
    for r in recs:
        t, s = r.get(tk), r.get(sk) if sk else None
        if isinstance(t, list):
            t = "\n".join(str(x) for x in t)
        if isinstance(s, list):
            s = " ".join(str(x) for x in s)
        if not isinstance(t, str):
            raise DataError(f"{split}: document field {tk!r} is {type(t).__name__}")
        out.append({"id": str(r[idk]), "text": t, "summary": s if isinstance(s, str) else None})
    if split == "train":
        out = [d for d in out if d["summary"] and d["summary"].strip() and d["text"].strip()]
    schema = {"container": "dict keyed by id" if keyed else "list of objects", "id_key": idk,
              "text_key": tk, "summary_key": sk, "n": len(out), "fields": sorted(recs[0].keys()),
              "md5": file_md5(path), "path": path}
    return out, schema


def split_dev(train: List[Dict], dev_frac: float, dev_max: int, dev_min: int = 20):
    keyed = sorted(train, key=lambda d: hashlib.md5(norm_hash(d["text"]).encode()).hexdigest())
    n_dev = int(min(dev_max, max(dev_min, round(dev_frac * len(train)))))
    n_dev = min(n_dev, max(1, len(train) // 4))
    dev_h = {norm_hash(d["text"]) for d in keyed[:n_dev]}
    dev = [d for d in keyed if norm_hash(d["text"]) in dev_h]
    tr = [d for d in train if norm_hash(d["text"]) not in dev_h]
    return tr, dev


def load_all(input_roots, dev_frac=0.06, dev_max=300, log=print) -> Tuple[Dict, Dict, Dict]:
    found = discover_files(input_roots, log=log)
    if found["source"] is None:
        raise FileNotFoundError(
            "Phase-6 data not found. Add the Kaggle secret `HF` (token of a Hugging Face account "
            "that accepted the dataset terms) and turn Internet ON -- or attach a dataset with "
            f"Phase_6_Summ/Train/train.json and Test/test.json. Found: {found['paths']}")
    data, schemas = {}, {}
    for s in ("train", "test"):
        data[s], schemas[s] = load_summ_split(found["paths"][s], s, log)
        log(f"   [data] {s}: {schemas[s]['n']} records; keys id={schemas[s]['id_key']!r} "
            f"document={schemas[s]['text_key']!r} summary={schemas[s]['summary_key']!r}")
    data["dev"] = []
    leak = leak_guard(data, log)                         # drops train docs that occur in test
    data["train"], data["dev"] = split_dev(data["train"], dev_frac, dev_max)
    log(f"   [data] held-out dev split: {len(data['dev'])} docs (hash of normalised text); "
        f"train {len(data['train'])}")
    return data, schemas, {"leak_guard": leak, "source": found["source"]}


# =============================================================================================
# token cache
# =============================================================================================
def _init_worker(bpe_dir: str) -> None:
    global _TOK
    from bpe_tokenizer import BPETokenizer
    _TOK = BPETokenizer.from_pretrained(bpe_dir)


def _encode_rec(args):
    text, summary = args
    t = text[:MAX_DOC_CHARS]
    doc = np.asarray(_TOK.encode(t), np.int32)
    n_full = len(doc) if len(t) == len(text) else int(len(doc) * len(text) / max(1, len(t)))
    sents = None
    if summary:
        pieces = sentence_spans(summary.strip())
        sents = [np.asarray(_TOK.encode(p), np.int32) for p in pieces]
    return {"doc": doc, "n_tokens": n_full, "sents": sents,
            "sent_text": pieces if summary else None}


def cache_key(file_md5s: Dict[str, str], tok_md5: str, dev_n: int) -> str:
    s = json.dumps({"v": CACHE_VERSION, "files": file_md5s, "tok": tok_md5, "dev": dev_n,
                    "max_chars": MAX_DOC_CHARS}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def find_cache(key: str, roots: List[str]) -> Optional[str]:
    name = f"summ_cache_{key}.pkl"
    for r in roots:
        if r and os.path.isdir(r):
            for dp, _dn, fns in os.walk(r):
                if name in fns:
                    return os.path.join(dp, name)
    return None


def build_cache(data: Dict[str, List[Dict]], bpe_dir: str, out_dir: str, key: str,
                n_proc: int, log=print) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"summ_cache_{key}.pkl")
    t0 = time.time()
    res = {}
    for split, docs in data.items():
        args = [(d["text"], d.get("summary")) for d in docs]
        if n_proc > 1 and len(args) >= 32:
            with get_context("fork").Pool(n_proc, initializer=_init_worker,
                                          initargs=(bpe_dir,)) as p:
                enc = p.map(_encode_rec, args, chunksize=8)
        else:
            _init_worker(bpe_dir)
            enc = [_encode_rec(a) for a in args]
        for d, e in zip(docs, enc):
            e["id"] = d["id"]
            e["summary"] = d.get("summary")
            if split != "train":
                e["text"] = d["text"][:MAX_DOC_CHARS]
        res[split] = enc
        log(f"   [cache] {split}: {len(enc)} docs tokenised ({time.time() - t0:.0f}s so far)")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.replace(path + ".tmp", path)
    return path


def load_cache(path: str) -> Dict[str, List[Dict]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def summary_stats(cache: Dict[str, List[Dict]]) -> Dict:
    tr = cache["train"]
    st = [sum(len(s) for s in r["sents"]) for r in tr if r.get("sents")]
    dt = [r["n_tokens"] for r in tr]
    sw = [len((r.get("summary") or "").split()) for r in tr]
    ns = [len(r["sents"]) for r in tr if r.get("sents")]
    return {"summary_tokens": length_stats([float(x) for x in st]),
            "doc_tokens": length_stats([float(x) for x in dt]),
            "summary_words": length_stats([float(x) for x in sw]),
            "summary_sentences": length_stats([float(x) for x in ns]),
            "median_summary_words": float(np.median(sw)) if sw else 200.0}


# =============================================================================================
# representations
# =============================================================================================
class Layout:
    """Token budgets of one run (all sequences <= ctx = 1024 - n_virtual)."""

    def __init__(self, tok, cfg: Dict, stats: Dict, n_virtual: int = 0):
        self.ctx = 1024 - int(n_virtual)
        self.repr = cfg.get("doc_repr", "auto")
        med = stats["summary_tokens"].get("median", 200.0)
        if self.repr == "auto":
            self.repr = "dancer" if med > float(cfg.get("auto_long", 350)) else "headtail"
        self.sep = np.asarray(tok.encode(SEP_TEXT), np.int32)
        self.tok = tok
        self._parts = {}
        if self.repr == "dancer":
            self.C = min(int(cfg.get("chunk_tokens", 640)), self.ctx - 200)
            self.max_chunks = int(cfg.get("max_chunks", 16))
            self.S = self.ctx - 2 - self.C - len(self.part_sep(99, 99))
        else:
            p90 = stats["summary_tokens"].get("p90", 256.0)
            cap = int(cfg.get("summary_frac", 0.5) * self.ctx)
            self.S = int(min(max(64, p90), cap))
            self.D = self.ctx - 2 - len(self.sep) - self.S
        self.median_words = stats.get("median_summary_words", 200.0)

    def part_sep(self, i: int, n: int) -> np.ndarray:
        k = (i, n)
        if k not in self._parts:
            self._parts[k] = np.asarray(self.tok.encode(PART_TEXT.format(i=i, n=n)), np.int32)
        return self._parts[k]

    def describe(self) -> Dict:
        if self.repr == "dancer":
            return {"doc_repr": "dancer", "chunk_tokens": self.C, "max_chunks": self.max_chunks,
                    "target_budget_per_chunk": self.S, "ctx": self.ctx}
        return {"doc_repr": self.repr, "doc_tokens": self.D, "summary_budget": self.S,
                "ctx": self.ctx}

    # ---------------------------------------------------------------- document parts
    def single_prompt(self, doc: np.ndarray) -> np.ndarray:
        D = self.D
        if len(doc) <= D:
            body = doc
        elif self.repr == "head":
            body = doc[:D]
        else:
            h = int(0.6 * D)
            body = np.concatenate([doc[:h], doc[len(doc) - (D - h):]])
        return np.concatenate([[0], body, self.sep]).astype(np.int64)

    def chunk_bounds(self, n_doc: int) -> List[Tuple[int, int]]:
        C = self.C
        n = max(1, int(np.ceil(n_doc / C)))
        b = [(i * C, min(n_doc, (i + 1) * C)) for i in range(n)]
        if len(b) > self.max_chunks:                     # keep first, last, evenly spaced middle
            pick = np.unique(np.round(np.linspace(0, len(b) - 1, self.max_chunks)).astype(int))
            b = [b[i] for i in pick]
        return b

    def chunk_prompts(self, doc: np.ndarray) -> List[np.ndarray]:
        bs = self.chunk_bounds(len(doc))
        return [np.concatenate([[0], doc[s:e], self.part_sep(i + 1, len(bs))]).astype(np.int64)
                for i, (s, e) in enumerate(bs)]

    def prompts(self, rec: Dict) -> List[np.ndarray]:
        if self.repr == "dancer":
            return self.chunk_prompts(rec["doc"])
        return [self.single_prompt(rec["doc"])]

    # ---------------------------------------------------------------- training examples
    def examples(self, rec: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
        sents = rec.get("sents") or []
        if self.repr != "dancer":
            tgt = np.concatenate(sents) if sents else np.zeros(0, np.int32)
            tgt = tgt[:self.S] if len(tgt) >= self.S else np.concatenate([tgt, [0]])
            return [(self.single_prompt(rec["doc"]), tgt.astype(np.int64))]
        doc = rec["doc"]
        bs = self.chunk_bounds(len(doc))
        texts = [self.tok.decode(doc[s:e].tolist()) for s, e in bs]
        owner = align_sentences(rec.get("sent_text") or [], texts)
        out = []
        for i, (s, e) in enumerate(bs):
            mine = [sents[j] for j, o in enumerate(owner) if o == i]
            tgt = np.concatenate(mine) if mine else np.zeros(0, np.int32)
            tgt = tgt[:self.S] if len(tgt) >= self.S else np.concatenate([tgt, [0]])
            prompt = np.concatenate([[0], doc[s:e], self.part_sep(i + 1, len(bs))])
            out.append((prompt.astype(np.int64), tgt.astype(np.int64)))
        return out

    def max_new(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> int:
        """Generation cap: p95 of the training target lengths (incl. EOS), within the budget."""
        L = [len(t) for _, t in train_examples if len(t) > 1]
        if not L:
            return min(self.S, 64)
        return int(min(self.S, max(16, np.percentile(L, 95))))

    def min_new(self, train_examples, frac: float) -> int:
        """Single-pass layouts: forbid EOS before frac x median target length (an early EOS
        costs recall in both ROUGE and BERTScore). Dancer chunks may legitimately be empty."""
        if self.repr == "dancer":
            return 0
        L = [len(t) for _, t in train_examples if len(t) > 1]
        return int(frac * np.median(L)) if L else 0

    def join(self, pieces: List[str]) -> str:
        return " ".join(p.strip() for p in pieces if p and p.strip())
