"""
cjpe_data.py -- Phase-5 data: discovery, Hugging Face fallback download, schema-tolerant loading,
data audit and the leakage guard (sections 2, 3, 4.2, 4.3 of the brief).

Data integrity rules implemented here:
  * training uses train.json ONLY; dev.json only for selection; test.json text only for
    inference (enforced again by cjpe_train at start-up);
  * any train case whose id or text hash also occurs in test is DROPPED from training, any dev
    case that occurs in test is dropped from dev; every count is written to data_audit.json;
  * PROVENANCE.json records md5s of the three files and what each one was used for.
"""
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

from cjpe_common import RAW_DIR, SPEC, file_md5, find_files, log_to, write_json

ID_KEYS = ("id", "case_id", "doc_id", "Id", "ID", "uid", "name")
TEXT_KEYS = ("text", "case_text", "document", "doc", "content", "judgment_text", "raw_text")
LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "label_ids", "pred", "y", "target")


class DataError(RuntimeError):
    """The data cannot be loaded unambiguously: the pipeline stops and asks (section 13)."""


# =============================================================================================
# Hugging Face token (Kaggle secret) + download
# =============================================================================================
def get_hf_token() -> Optional[str]:
    """Kaggle secret `HF` (or HF_TOKEN / HUGGINGFACE_TOKEN), else the HF_TOKEN env var.
    The token is never printed, logged or written anywhere."""
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        for name in ("HF", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
            try:
                tok = us.get_secret(name)
                if tok:
                    return tok.strip()
            except Exception:
                pass
    except Exception:
        pass
    return (os.environ.get("HF_TOKEN") or "").strip() or None


def _redact(msg: str, token: Optional[str]) -> str:
    return msg.replace(token, "<redacted>") if token else msg


def hf_download(dest_root: str = RAW_DIR, log=print) -> Optional[Dict[str, str]]:
    """Download the three files with urllib (no huggingface_hub). Returns {split: path} or None."""
    token = get_hf_token()
    log(f"   [data] HF token found: {'yes' if token else 'no'}"
        + (f" (length {len(token)})" if token else ""))
    if not token:
        return None
    out = {}
    for split, rel in SPEC["files"].items():
        url = (f"https://huggingface.co/datasets/{SPEC['hf_repo']}/resolve/main/"
               f"{SPEC['hf_folder']}/{rel}")
        dest = os.path.join(dest_root, SPEC["hf_folder"], rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            try:
                with open(dest, encoding="utf-8") as f:
                    json.load(f)
                out[split] = dest
                continue
            except Exception:
                pass
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                                           "User-Agent": "cs781-phase5"})
                with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".tmp", "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                with open(dest + ".tmp", encoding="utf-8") as f:
                    json.load(f)                       # validate before accepting
                os.replace(dest + ".tmp", dest)
                out[split] = dest
                log(f"   [data] downloaded {rel} ({os.path.getsize(dest) / 1e6:.1f} MB)")
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (401, 403):
                    log("   [data] HTTP %d: token missing, or this Hugging Face account has not "
                        "accepted the dataset terms" % e.code)
                    return None
            except Exception as e:                      # network / JSON errors: retry
                last = _redact(f"{type(e).__name__}: {e}", token)
            time.sleep(5 * (attempt + 1))
        if split not in out:
            log(f"   [data] download of {rel} failed after 3 attempts ({last})")
            return None
    return out


# =============================================================================================
# discovery
# =============================================================================================
def _peek_records(path: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """(id_key, text_key, label_key) if `path` looks like a CJPE split, else None."""
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        recs, keyed = _records(obj)
        if not recs:
            return None
        return _detect_keys(recs[:50], keyed)
    except Exception:
        return None


def discover_files(input_roots: List[str], log=print) -> Dict:
    """Find train/dev/test among the attached datasets; fall back to the HF download."""
    found, cands = {}, {}
    for split, rel in SPEC["files"].items():
        name = os.path.basename(rel)
        hits = find_files(name, input_roots, max_hits=200)
        good = []
        for h in hits:
            low = h.lower()
            score = (("phase_4" in low) + ("cjpe" in low) * 2 + (("/test/" in low) if split == "test"
                     else ("/train/" in low)))
            keys = _peek_records(h)
            if keys is not None:
                good.append((score, h))
        good.sort(key=lambda t: (-t[0], t[1]))
        cands[split] = [g[1] for g in good]
        if good:
            found[split] = good[0][1]
    for split, c in cands.items():
        if len(c) > 1:
            log(f"   [data] {len(c)} candidates for {split}: {c[:5]} -> using {found[split]}")
    if len(found) == 3:
        return {"source": "kaggle_dataset", "paths": found, "candidates": cands}
    log(f"   [data] attached dataset incomplete (found {sorted(found)}); trying the HF download")
    dl = hf_download(log=log)
    if dl and len(dl) == 3:
        return {"source": "huggingface", "paths": dl, "candidates": cands}
    return {"source": None, "paths": found, "candidates": cands}


# =============================================================================================
# schema-tolerant parsing
# =============================================================================================
def _records(obj) -> Tuple[List, bool]:
    """Return (records, keyed). keyed=True when the file was a dict keyed by id."""
    if isinstance(obj, list):
        return obj, False
    if isinstance(obj, dict):
        for k in ("data", "examples", "records", "cases", "rows"):
            if isinstance(obj.get(k), list):
                return obj[k], False
        # dict keyed by id: values are dicts or plain strings
        vals = list(obj.values())
        if vals and all(isinstance(v, (dict, str)) for v in vals[:50]):
            recs = []
            for k, v in obj.items():
                if isinstance(v, str):
                    recs.append({"id": k, "text": v})
                else:
                    r = dict(v)
                    r.setdefault("id", k)
                    recs.append(r)
            return recs, True
    raise DataError(f"unrecognised JSON container: {type(obj).__name__}")


def _detect_keys(recs: List, keyed: bool) -> Tuple[str, Optional[str], Optional[str]]:
    if not all(isinstance(r, dict) for r in recs):
        raise DataError("records are not JSON objects")
    keys = Counter(k for r in recs for k in r.keys())
    n = len(recs)
    idk = next((k for k in ID_KEYS if keys.get(k, 0) == n), None)
    tk = next((k for k in TEXT_KEYS if keys.get(k, 0) == n), None)
    if tk is None:                       # fall back: the single long-string field
        strf = [k for k in keys if keys[k] == n and all(isinstance(r[k], str) for r in recs)]
        strf = [k for k in strf if k not in ID_KEYS]
        if len(strf) == 1:
            tk = strf[0]
    lk = next((k for k in LABEL_KEYS if keys.get(k, 0) == n), None)
    if idk is None or tk is None:
        raise DataError(f"cannot find id/text keys among {dict(keys)}")
    return idk, tk, lk


def _to_label(v) -> List[int]:
    """Statute ids as a sorted unique list of ints (accepts int, numeric str, list, "1,7")."""
    if v is None:
        return []
    if isinstance(v, bool):
        raise DataError(f"label value {v!r} is a bool, not a statute id")
    if isinstance(v, (int, float)) and float(v) == int(v):
        return [int(v)]
    if isinstance(v, str):
        parts = [x for x in v.replace(";", ",").replace(" ", ",").split(",") if x.strip()]
        try:
            return sorted({int(float(x)) for x in parts})
        except ValueError:
            raise DataError(f"label value {v!r} is not a list of statute ids")
    if isinstance(v, (list, tuple)):
        out = set()
        for x in v:
            out.update(_to_label(x))
        return sorted(out)
    raise DataError(f"label value {v!r} is not a list of statute ids")


def load_split(path: str, split: str, log=print) -> Tuple[List[Dict], Dict]:
    """[{id, text, label|None}] + schema description. Raises DataError when ambiguous."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    recs, keyed = _records(obj)
    if not recs:
        raise DataError(f"{split}: no records in {path}")
    idk, tk, lk = _detect_keys(recs, keyed)
    if split != "test" and lk is None:
        raise DataError(f"{split}: no label key among {sorted(recs[0].keys())}")
    out = []
    for r in recs:
        t = r[tk]
        if isinstance(t, list):                          # list of sentences/paragraphs
            t = "\n".join(str(x) for x in t)
        if not isinstance(t, str):
            raise DataError(f"{split}: text field {tk!r} is {type(t).__name__}")
        lab = _to_label(r[lk]) if (lk is not None and split != "test") else None
        out.append({"id": str(r[idk]), "text": t, "label": lab})
    schema = {"container": "dict keyed by id" if keyed else "list of objects",
              "id_key": idk, "text_key": tk, "label_key": lk, "n": len(out),
              "fields": sorted(recs[0].keys()), "md5": file_md5(path), "path": path}
    return out, schema


# =============================================================================================
# audit + leak guard
# =============================================================================================
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def norm_hash(s: str) -> str:
    return sha1(re.sub(r"\s+", " ", s).strip().lower())


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))])


def length_stats(vals: List[float]) -> Dict:
    if not vals:
        return {}
    return {"mean": sum(vals) / len(vals), "median": _pct(vals, 0.5), "p90": _pct(vals, 0.9),
            "p99": _pct(vals, 0.99), "max": max(vals), "min": min(vals)}


def leak_guard(data: Dict[str, List[Dict]], log=print) -> Dict:
    """Drop train/dev cases that also occur in test (by id or raw / normalised text hash)."""
    h = {s: [(d["id"], sha1(d["text"]), norm_hash(d["text"])) for d in data[s]] for s in data}
    sets = {s: {"id": {a for a, _, _ in v}, "sha1": {b for _, b, _ in v},
                "norm": {c for _, _, c in v}} for s, v in h.items()}
    rep = {}
    for a, b in (("train", "test"), ("dev", "test"), ("train", "dev")):
        rep[f"{a}_x_{b}"] = {k: len(sets[a][k] & sets[b][k]) for k in ("id", "sha1", "norm")}
    dropped = {}
    for s in ("train", "dev"):
        keep, drop = [], []
        for d, (i, x, y) in zip(data[s], h[s]):
            if i in sets["test"]["id"] or x in sets["test"]["sha1"] or y in sets["test"]["norm"]:
                drop.append(i)
            else:
                keep.append(d)
        dropped[s] = drop
        if drop:
            log(f"   !!! [leak-guard] {len(drop)} {s} case(s) also occur in test -> DROPPED from "
                f"{s} (ids {drop[:5]}...)")
        data[s] = keep
    rep["dropped_from_train"] = len(dropped["train"])
    rep["dropped_from_dev"] = len(dropped["dev"])
    rep["dropped_ids"] = {k: v[:200] for k, v in dropped.items()}
    n_train = len(data["train"]) + len(dropped["train"])
    rep["train_test_overlap_pct"] = 100.0 * len(dropped["train"]) / max(1, n_train)
    rep["status"] = "passed"
    return rep


def basic_audit(data: Dict[str, List[Dict]], schemas: Dict) -> Dict:
    a = {"schema": schemas, "splits": {}}
    for s, docs in data.items():
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        e = {"n": len(docs), "empty_text": sum(1 for t in texts if not t.strip()),
             "duplicate_ids": len(ids) - len(set(ids)),
             "duplicate_texts": len(texts) - len({sha1(t) for t in texts}),
             "chars": length_stats([float(len(t)) for t in texts]),
             "words": length_stats([float(len(t.split())) for t in texts])}
        if s != "test":
            c = Counter(x for d in docs for x in (d["label"] or []))
            nl = [len(d["label"] or []) for d in docs]
            freq = sorted(c.values())
            e["n_labels"] = len(c)
            e["labels_per_doc"] = length_stats([float(x) for x in nl])
            e["docs_without_labels"] = sum(1 for x in nl if x == 0)
            e["label_freq_quantiles"] = {q: _pct([float(x) for x in freq], q)
                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if freq else {}
            e["pct_labels_lt5"] = 100.0 * sum(1 for x in freq if x < 5) / max(1, len(freq))
            e["top_labels"] = c.most_common(10)
        a["splits"][s] = e
    return a


def token_audit(data: Dict[str, List[Dict]], tok, sample: int = 2000, seed: int = 42,
                n_proc: int = 1) -> Dict:
    """BPE length statistics on a seeded sample (full tokenisation); chars-per-token ratio."""
    from cjpe_tok import count_tokens_many
    rng = random.Random(seed)
    out = {}
    pool = []
    for s, docs in data.items():
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        pool += [(s, i) for i in idx[:max(1, sample // 3)]]
    counts = count_tokens_many([data[s][i]["text"] for s, i in pool], tok, n_proc)
    by = {}
    for (s, i), n in zip(pool, counts):
        by.setdefault(s, []).append((n, len(data[s][i]["text"]), data[s][i]["label"]))
    tot_c = sum(c for v in by.values() for _, c, _ in v)
    tot_t = sum(n for v in by.values() for n, _, _ in v)
    ratio = tot_c / max(1, tot_t)
    out["chars_per_token"] = ratio
    out["sample_size"] = len(pool)
    for s, v in by.items():
        toks = [float(n) for n, _, _ in v]
        e = {"tokens": length_stats(toks)}
        for thr in (512, 1024, 2048, 4096, 8192):
            e[f"pct_over_{thr}"] = 100.0 * sum(1 for t in toks if t > thr) / max(1, len(toks))
        if s != "test":
            buckets = {}
            for n, _, lab in v:
                b = bucket_of(n)
                buckets.setdefault(b, []).append(lab)
            e["labels_per_doc_by_length"] = {b: {"n": len(l), "mean_labels":
                                                 sum(len(x or []) for x in l) / len(l)}
                                         for b, l in sorted(buckets.items())}
        out[s] = e
    return out


BUCKETS = [(512, "<=512"), (1024, "<=1024"), (2048, "<=2048"), (4096, "<=4096"),
           (10 ** 12, ">4096")]


def bucket_of(n_tokens: float) -> str:
    for thr, name in BUCKETS:
        if n_tokens <= thr:
            return name
    return BUCKETS[-1][1]


def provenance(schemas: Dict, leak: Dict, n_train_used: int, source: str) -> Dict:
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": source,
        "md5": {s: schemas[s]["md5"] for s in schemas},
        "paths": {s: schemas[s]["path"] for s in schemas},
        "used_for": {"train.json": "training (the ONLY file whose labels are used for gradient "
                                   "updates)",
                     "dev.json": "model selection, early stopping, threshold tuning and local "
                                 "evaluation only (never trained on, no train+dev refit)",
                     "test.json": "inference only (text); no test-time training, no pseudo "
                                  "labels, nothing tuned on test predictions"},
        "train_cases_used": n_train_used,
        "train_cases_dropped_by_leak_guard": leak.get("dropped_from_train", 0),
        "dev_cases_dropped_by_leak_guard": leak.get("dropped_from_dev", 0),
        "leak_guard": leak.get("status", "unknown"),
    }


def print_audit(a: Dict, log=print) -> None:
    log("   [audit] split   n      empty dup_id dup_text  pos_rate  chars(median)")
    for s, e in a["splits"].items():
        pr = e.get("positive_rate")
        log(f"   [audit] {s:<6} {e['n']:>6} {e['empty_text']:>6} {e['duplicate_ids']:>6} "
            f"{e['duplicate_texts']:>8}  {'' if pr is None else f'{pr:.3f}':>8}  "
            f"{e['chars'].get('median', 0):,.0f}")
    t = a.get("tokens")
    if t:
        log(f"   [audit] chars/token {t['chars_per_token']:.2f} (sample of {t['sample_size']})")
        for s in ("train", "dev", "test"):
            if s in t:
                e = t[s]
                log(f"   [audit] {s} tokens: median {e['tokens'].get('median', 0):,.0f}, p90 "
                    f"{e['tokens'].get('p90', 0):,.0f}, max {e['tokens'].get('max', 0):,.0f}; "
                    f">512 {e['pct_over_512']:.0f}%  >1024 {e['pct_over_1024']:.0f}%  "
                    f">2048 {e['pct_over_2048']:.0f}%  >4096 {e['pct_over_4096']:.0f}%")
    lg = a.get("leak_guard")
    if lg:
        log(f"   [leak-guard] train x test {lg['train_x_test']}, dev x test {lg['dev_x_test']}, "
            f"train x dev {lg['train_x_dev']}; dropped train {lg['dropped_from_train']}, "
            f"dev {lg['dropped_from_dev']}")


def print_examples(data: Dict[str, List[Dict]], log=print) -> None:
    for s, docs in data.items():
        if docs:
            d = docs[0]
            log(f"   [example] {s}: id={d['id']!r} label={d['label']!r} text={d['text'][:300]!r}")
