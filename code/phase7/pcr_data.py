"""
pcr_data.py -- Phase-7 data: discovery (attached dataset, else the HF download), schema-tolerant
loading of the six files, the chunk token cache and the BM25 lexical channel.

Phase_7_PCR/{Train,Dev,Test}/{split}.json             queries: id, text (sentences),
                                                      relevant_candidates (Train/Dev only)
Phase_7_PCR/{Train,Dev,Test}/{split}_candidates.json  that split's candidate pool
A split's queries are only ever matched against THAT split's pool.

Chunks: every document becomes <= max_chunks token chunks of <= chunk_tokens tokens (EOS last).
Short documents are tokenised whole and cut into consecutive chunks; long ones are cut into
max_chunks equal-character sentence groups and the start of each group is tokenised -> the
chunks cover the WHOLE case at bounded tokenisation cost.
BM25 (pcr_ml.BM25, word level, the split's own pool): k1 / b / bigrams tuned on Dev
Micro-F1@7; the dev + test score matrices and the train hard-negative lists are cached.
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

from cjpe_common import SPEC, file_md5, find_files
from cjpe_data import DataError, hf_download
from pcr_ml import BM25, eval_scores, mask_self, topk, words

CACHE_VERSION = 1
SPLITS = ("train", "dev", "test")
CH_PER_TOK = 4.2
ID_KEYS = ("id", "query_id", "qid", "case_id", "doc_id", "candidate_id", "cid", "name")
TEXT_KEYS = ("text", "sentences", "document", "content", "body", "case_text", "judgment")
REL_KEYS = ("relevant_candidates", "relevant", "relevant_ids", "positives", "labels",
            "gold", "citations")
EOS_ID = 0
_TOK = None


# =============================================================================================
# discovery + parsing
# =============================================================================================
def discover(input_roots: List[str], log=print) -> Dict:
    found = {}
    for key, rel in SPEC["files"].items():
        name = os.path.basename(rel)
        sub = rel.split("/")[0].lower()
        hits = [h for h in find_files(name, input_roots, max_hits=50) if os.path.isfile(h)]
        hits.sort(key=lambda h: (-(f"/{sub}/" in h.lower()) - ("phase_7" in h.lower()), h))
        if hits:
            found[key] = hits[0]
    if len(found) == len(SPEC["files"]):
        return {"source": "kaggle_dataset", "paths": found}
    log(f"   [data] attached dataset incomplete (found {sorted(found)}); trying the HF download")
    dl = hf_download(log=log)
    if dl and len(dl) == len(SPEC["files"]):
        return {"source": "huggingface", "paths": dl}
    return {"source": None, "paths": found}


def _items(obj) -> List[Dict]:
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(obj) <= 3 and lists:
            return lists[0]
        out = []
        for k, v in obj.items():
            out.append({"id": k, **v} if isinstance(v, dict) else {"id": k, "text": v})
        return out
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    raise DataError(f"unsupported JSON container {type(obj).__name__}")


def _pick(r: Dict, keys) -> Optional[str]:
    return next((k for k in keys if k in r), None)


def _sentences(t) -> List[str]:
    if isinstance(t, list):
        out = []
        for x in t:
            out += _sentences(x) if isinstance(x, list) else [str(x)]
        return [s for s in out if s.strip()]
    if isinstance(t, str):
        s = [x for x in re.split(r"\n+|(?<=[.;:?!])\s+(?=[A-Z(\"'])", t) if x.strip()]
        return s or ([t] if t.strip() else [])
    return [] if t is None else [str(t)]


def load_file(path: str, kind: str, split: str) -> Tuple[List[Dict], Dict]:
    with open(path, encoding="utf-8") as f:
        recs = _items(json.load(f))
    if not recs:
        raise DataError(f"{split} {kind}: no records in {path}")
    s0 = recs[:200]
    idk = next((k for k in ID_KEYS if all(k in r for r in s0)), None)
    tk = next((k for k in TEXT_KEYS if all(k in r for r in s0)), None)
    if idk is None or tk is None:
        raise DataError(f"{split} {kind}: cannot find id/text keys among {sorted(recs[0])}")
    rk = next((k for k in REL_KEYS if all(k in r for r in s0)), None) if kind == "q" else None
    if kind == "q" and split != "test" and rk is None:
        raise DataError(f"{split}: no relevance key among {sorted(recs[0])}")
    out = []
    for r in recs:
        d = {"id": str(r[idk]), "sents": _sentences(r[tk])}
        if rk:
            v = r.get(rk) or []
            d["rel_ids"] = [str(x) for x in (v if isinstance(v, list) else [v])]
        out.append(d)
    schema = {"id_key": idk, "text_key": tk, "rel_key": rk, "n": len(out), "path": path,
              "fields": sorted(recs[0]), "md5": file_md5(path)}
    return out, schema


def _id_lookup(pool_ids: List[str]) -> Dict[str, int]:
    m = {p: j for j, p in enumerate(pool_ids)}
    for j, p in enumerate(pool_ids):                     # "0000021652" == "21652"
        m.setdefault(p.lstrip("0") or "0", j)
    return m


def load_all(input_roots, log=print) -> Tuple[Dict, Dict, Dict]:
    found = discover(input_roots, log=log)
    if found["source"] is None:
        raise FileNotFoundError(
            "Phase-7 data not found. Either attach a dataset holding Phase_7_PCR/{Train,Dev,Test}"
            "/{split}.json + {split}_candidates.json, or add the Kaggle secret `HF` and turn "
            f"Internet ON. Found: {found['paths']}")
    data, schemas, audit = {}, {}, {}
    for s in SPLITS:
        q, sq = load_file(found["paths"][s], "q", s)
        pool, sp = load_file(found["paths"][f"{s}_candidates"], "pool", s)
        seen, uniq = set(), []
        for d in pool:
            if d["id"] not in seen:
                seen.add(d["id"])
                uniq.append(d)
        pool = uniq
        look = _id_lookup([d["id"] for d in pool])
        missing = n_rel = self_in = 0
        for d in q:
            if "rel_ids" in d:
                idx = []
                for x in d["rel_ids"]:
                    j = look.get(x, look.get(x.lstrip("0") or "0"))
                    if j is None:
                        missing += 1
                    elif pool[j]["id"] != d["id"] and j not in idx:
                        idx.append(j)
                n_rel += len(d["rel_ids"])
                d["rel"] = idx
            self_in += d["id"] in look
        data[s] = {"q": q, "pool": pool}
        schemas[s], schemas[f"{s}_candidates"] = sq, sp
        rel = [len(d.get("rel", [])) for d in q]
        audit[s] = {"queries": len(q), "pool": len(pool), "relevant_total": n_rel,
                    "relevant_missing_from_pool": missing, "queries_in_own_pool": self_in,
                    "rel_per_query_mean": float(np.mean(rel)) if rel else 0.0,
                    "rel_per_query_max": int(max(rel)) if rel else 0,
                    "queries_without_relevant": int(sum(1 for x in rel if x == 0)),
                    "query_chars_median": float(np.median([sum(map(len, d["sents"])) for d in q])),
                    "pool_chars_median": float(np.median([sum(map(len, d["sents"])) for d in pool]))}
        log(f"   [data] {s}: {len(q)} queries ({sq['id_key']!r}/{sq['text_key']!r}/"
            f"{sq['rel_key']!r}), pool {len(pool)} candidates; relevant per query mean "
            f"{audit[s]['rel_per_query_mean']:.2f}, {missing} relevant ids not in the pool, "
            f"{self_in} queries inside their own pool")
    ids = {s: {d["id"] for d in data[s]["q"]} for s in SPLITS}
    audit["query_overlap"] = {"train_dev": len(ids["train"] & ids["dev"]),
                              "train_test": len(ids["train"] & ids["test"]),
                              "dev_test": len(ids["dev"] & ids["test"])}
    return data, schemas, {"audit": audit, "source": found["source"]}


# =============================================================================================
# chunk token cache
# =============================================================================================
def _init_worker(bpe_dir: str) -> None:
    global _TOK
    from bpe_tokenizer import BPETokenizer
    _TOK = BPETokenizer.from_pretrained(bpe_dir)


def doc_chunk_texts(sents: List[str], C: int, K: int) -> Tuple[List[str], bool]:
    full = " ".join(sents)
    if len(full) <= K * C * CH_PER_TOK:
        return [full], True
    lens = np.cumsum([len(s) + 1 for s in sents])
    tot = float(lens[-1])
    groups, start = [], 0
    for k in range(1, K + 1):
        end = int(np.searchsorted(lens, tot * k / K, side="left")) + 1
        end = min(max(end, start + 1), len(sents))
        if start < len(sents):
            groups.append(" ".join(sents[start:end]))
        start = end
    lim = int(C * CH_PER_TOK * 1.4)
    return [g[:lim] for g in groups if g.strip()], False


def _encode(args):
    sents, C, K = args
    texts, whole = doc_chunk_texts(sents, C, K)
    body = C - 1
    if whole:
        ids = _TOK.encode(texts[0]) if texts[0] else []
        pieces = [ids[i:i + body] for i in range(0, max(1, len(ids)), body)] or [[]]
        if len(pieces) > K:
            pick = np.linspace(0, len(pieces) - 1, K).round().astype(int)
            pieces = [pieces[i] for i in sorted(set(pick.tolist()))]
    else:
        pieces = [_TOK.encode(t)[:body] for t in texts]
    return [np.asarray(list(p) + [EOS_ID], np.int32) for p in pieces if len(p) or len(pieces) == 1]


def cache_key(md5s: Dict[str, str], tok_md5: str, C: int, K: int) -> str:
    s = json.dumps({"v": CACHE_VERSION, "files": md5s, "tok": tok_md5, "C": C, "K": K},
                   sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def find_cache(key: str, roots: List[str]) -> Optional[str]:
    name = f"pcr_cache_{key}.pkl"
    for r in roots:
        if r and os.path.isdir(r):
            for dp, _dn, fns in os.walk(r):
                if name in fns:
                    return os.path.join(dp, name)
    return None


def build_cache(data: Dict, bpe_dir: str, out_dir: str, key: str, C: int, K: int, n_proc: int,
                log=print, bm25_grid=None, hard_k: int = 60) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pcr_cache_{key}.pkl")
    t0 = time.time()
    res = {"C": C, "K": K, "splits": {}}
    pool_ctx = get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(bpe_dir,)) \
        if n_proc > 1 else None
    if pool_ctx is None:
        _init_worker(bpe_dir)
    try:
        for s in SPLITS:
            out = {}
            for part in ("q", "pool"):
                docs = data[s][part]
                args = [(d["sents"], C, K) for d in docs]
                enc = pool_ctx.map(_encode, args, chunksize=4) if pool_ctx else \
                    [_encode(a) for a in args]
                out[part] = [{"id": d["id"], "chunks": e, "rel": d.get("rel"),
                              "n_chars": sum(map(len, d["sents"]))} for d, e in zip(docs, enc)]
            res["splits"][s] = out
            log(f"   [cache] {s}: {len(out['q'])} queries + {len(out['pool'])} candidates "
                f"tokenised ({time.time() - t0:.0f}s so far)")
    finally:
        if pool_ctx:
            pool_ctx.close()
    res["bm25"] = bm25_channel(data, log, bm25_grid, hard_k)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.replace(path + ".tmp", path)
    return path


def bm25_channel(data: Dict, log=print, grid=None, hard_k: int = 60) -> Dict:
    """Tune BM25 on dev Micro-F1@7; dev/test score matrices + train hard negatives."""
    t0 = time.time()
    grid = grid or [(k1, b) for k1 in (0.9, 1.5, 2.5) for b in (0.5, 0.75, 0.9)]
    txt = {s: {p: [" ".join(d["sents"]) for d in data[s][p]] for p in ("q", "pool")}
           for s in SPLITS}
    gold_dev = [d.get("rel", []) for d in data["dev"]["q"]]
    curve, best = [], None
    for ng in (1, 2):
        tk = {s: {p: [words(t, ng) for t in txt[s][p]] for p in ("q", "pool")}
              for s in ("dev",)}
        for k1, b in grid:
            bm = BM25(None, k1, b, ng, pool_tokens=tk["dev"]["pool"])
            S = mask_self(bm.score(None, tk["dev"]["q"]), [d["id"] for d in data["dev"]["q"]],
                          [d["id"] for d in data["dev"]["pool"]])
            m = eval_scores(S, gold_dev)
            curve.append({"k1": k1, "b": b, "ngram": ng, **m})
            if best is None or m["micro_f1"] > best["micro_f1"]:
                best = curve[-1]
    out = {"params": {k: best[k] for k in ("k1", "b", "ngram")}, "dev_curve": curve,
           "dev_micro_f1": best["micro_f1"], "dev_micro_p": best["micro_p"],
           "dev_micro_r": best["micro_r"]}
    p = out["params"]
    for s in SPLITS:
        bm = BM25(txt[s]["pool"], p["k1"], p["b"], p["ngram"])
        S = mask_self(bm.score(txt[s]["q"]), [d["id"] for d in data[s]["q"]],
                      [d["id"] for d in data[s]["pool"]])
        if s == "train":
            out["train_top"] = topk(S, min(hard_k, S.shape[1])).astype(np.int32)
        else:
            out[s] = S.astype(np.float32)
    log(f"   [bm25] tuned on dev: k1={p['k1']} b={p['b']} ngram={p['ngram']} -> dev Micro-F1@7 "
        f"{out['dev_micro_f1']:.4f} (P {out['dev_micro_p']:.4f} R {out['dev_micro_r']:.4f}); "
        f"{time.time() - t0:.0f}s")
    return out


def load_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def chunk_stats(cache: Dict) -> Dict:
    out = {}
    for s, d in cache["splits"].items():
        for p in ("q", "pool"):
            n = [len(x["chunks"]) for x in d[p]]
            out[f"{s}_{p}_chunks_mean"] = float(np.mean(n)) if n else 0.0
            out[f"{s}_{p}_chunks_full_frac"] = float(np.mean([k >= cache["K"] for k in n])) \
                if n else 0.0
    return out
