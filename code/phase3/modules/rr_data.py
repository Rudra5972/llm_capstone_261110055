"""
rr_data.py -- Phase-3 data: loading, gates G0/G1/G2, tokenised cache, window packing.

Torch-free (numpy + stdlib). Run on CPU:

    python rr_data.py build [--out /kaggle/working/phase3_data]

Gates (logged to GATES.json in the cache dir; any failure raises GateFailure = stop and ask):
  G0 schema   every doc has an id and a list of sentences; train/dev: len(text) == len(labels),
              no empty docs, every label in label_vocab.json, label_vocab == SPEC["labels"];
              no duplicate ids; CL_test and IT_test ids must not collide (a single
              prediction.json dict would be ambiguous -> STOP, ask the TAs on Discord).
              Overlaps of test ids with train/dev ids are printed loudly (not fatal).
  G1 tokens   every sentence is encoded; length stats (mean/p50/p95/p99/max per sentence,
              tokens per doc); a sentence that encodes to zero tokens gets [unk] (counted);
              truncations at max_sent_tokens are counted.
  G2 output   (a) window packing round trip on every train/dev/test doc for every input mode
              (each sentence covered, markers at the right positions, order and count
              recovered); (b) the submission validator of rr_predict accepts a dummy all-"None"
              prediction built from the real test ids and rejects 5 corrupted variants.

Input representations (section 4 of the code-generation prompt):
  doc    sentences packed into windows of <= context tokens; every sentence is followed by the
         EOS token (id 0) = its MARKER; the sentence vector is the hidden state at the marker.
         Long documents use overlapping windows (train stride 0.75, inference stride 0.5 of the
         budget, in tokens, snapped to sentence boundaries). Each sentence is predicted from
         the window where it has the most context (bidir: centre-most = max(min(left, right));
         causal: most left context), or from the mean over its windows (`window_combine`).
  sent   one sentence per window (context ablation baseline).
  ctx_k  target sentence + k neighbours on each side; only the target's marker is labelled.
A domain prefix ("[CL]" / "[IT]", tokenised as normal text: the tokenizer has no spare
special tokens) starts every window. Sentences are truncated to `max_sent_tokens` (256).
"""
import argparse
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rr_common import (SPEC, GateFailure, default_input_roots, file_md5, find_files, read_json,
                       write_json)

IGNORE = -100
CACHE_VERSION = 3


# =============================================================================================
# 1. locating / downloading the official files
# =============================================================================================
def _hf_read_token() -> Optional[str]:
    """Optional READ token for downloading the official data (never used for writing)."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient          # only exists on Kaggle
        return UserSecretsClient().get_secret("HF")
    except Exception:
        return None


def _download(rel_path: str, dest: str, token: Optional[str]) -> None:
    url = f"https://huggingface.co/datasets/{SPEC['hf_repo']}/resolve/main/{rel_path}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(dest + ".part", dest)


def locate_raw_files(raw_dir: str, roots: Optional[List[str]] = None, allow_download: bool = True,
                     log=print) -> Dict[str, str]:
    """{key: path} for every file in SPEC["files"], found under the input roots (attached Kaggle
    dataset) or downloaded once, read-only, from the Hugging Face dataset repo."""
    roots = roots or default_input_roots()
    found = {}
    for key, rel in SPEC["files"].items():
        name = os.path.basename(rel)
        hits = [h for h in find_files(name, roots + [raw_dir]) if "phase3_out" not in h]
        if hits:
            sub = os.path.dirname(rel)                         # "Train" / "Test"
            hits.sort(key=lambda p: (SPEC["hf_folder"] not in p, sub not in p, len(p)))
            found[key] = hits[0]
    missing = [k for k in SPEC["files"] if k not in found]
    if missing and allow_download:
        os.makedirs(raw_dir, exist_ok=True)
        token = _hf_read_token()
        for key in missing:
            rel = f"{SPEC['hf_folder']}/{SPEC['files'][key]}"
            dest = os.path.join(raw_dir, key + ".json")
            try:
                log(f"   downloading {rel} ...")
                _download(rel, dest, token)
                found[key] = dest
            except (urllib.error.URLError, OSError) as e:
                log(f"   could not download {rel}: {e}")
    missing = [SPEC["files"][k] for k in SPEC["files"] if k not in found]
    if missing:
        raise FileNotFoundError(
            "Phase-3 files not found: " + ", ".join(missing) + f". Attach the official "
            f"{SPEC['hf_folder']} folder as a Kaggle dataset (or add a read-only HF token as the "
            f"Kaggle secret 'HF' and enable internet).")
    return found


# =============================================================================================
# 2. tolerant loaders
# =============================================================================================
def load_split_file(path: str, has_labels: bool) -> List[Dict]:
    """[{"id": str, "text": [str], "labels": [str] | None}] from either
    [{"id", "text", "labels"?}]  or  {id: [sentences]}  or  {id: {"text", "labels"?}}."""
    obj = read_json(path)
    docs = []
    if isinstance(obj, list):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict) or "id" not in rec or "text" not in rec:
                raise GateFailure(f"G0: {path}: element {i} lacks 'id'/'text': "
                                  f"{str(rec)[:200]}")
            docs.append({"id": str(rec["id"]), "text": rec["text"],
                         "labels": rec.get("labels")})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                docs.append({"id": str(k), "text": v, "labels": None})
            elif isinstance(v, dict) and "text" in v:
                docs.append({"id": str(k), "text": v["text"], "labels": v.get("labels")})
            else:
                raise GateFailure(f"G0: {path}: document {k!r} has unexpected value "
                                  f"{str(v)[:200]}")
    else:
        raise GateFailure(f"G0: {path}: top level must be a list or an object")
    for d in docs:
        if not isinstance(d["text"], list) or not all(isinstance(s, str) for s in d["text"]):
            raise GateFailure(f"G0: {path}: document {d['id']!r}: 'text' must be a list of "
                              f"strings")
        if has_labels and d["labels"] is None:
            raise GateFailure(f"G0: {path}: document {d['id']!r} has no labels")
        if not has_labels:
            d["labels"] = None
    return docs


def load_label_vocab(path: str) -> List[str]:
    """label_vocab.json {label: index} -> labels in index order; must equal SPEC["labels"]."""
    obj = read_json(path)
    if isinstance(obj, dict):
        labels = [k for k, _ in sorted(obj.items(), key=lambda kv: int(kv[1]))]
        if sorted(int(v) for v in obj.values()) != list(range(len(obj))):
            raise GateFailure(f"G0: label_vocab indices are not 0..{len(obj) - 1}: {obj}")
    elif isinstance(obj, list):
        labels = [str(x) for x in obj]
    else:
        raise GateFailure(f"G0: label_vocab.json has an unexpected layout: {str(obj)[:200]}")
    if labels != SPEC["labels"]:
        raise GateFailure(f"G0: label_vocab.json {labels} differs from SPEC['labels'] "
                          f"{SPEC['labels']} (spelling/order). The brief may have changed: "
                          f"update rr_common.SPEC and regenerate the notebooks.")
    return labels


# =============================================================================================
# 3. window packing (pure functions, unit-tested in selftest.py and gate G2)
# =============================================================================================
def sentence_piece(ids: np.ndarray, max_sent_tokens: int, eos_id: int) -> np.ndarray:
    """Sentence ids truncated to max_sent_tokens, followed by the EOS marker."""
    return np.concatenate([ids[:max_sent_tokens], [eos_id]]).astype(np.int32)


def doc_ranges(lengths: Sequence[int], budget: int, stride: int) -> List[Tuple[int, int]]:
    """Sentence ranges [a, b) whose pieces (lengths incl. marker) fit in `budget` tokens.
    Consecutive ranges overlap: the next range starts at the first sentence at least `stride`
    tokens after the current start (never after the current end, so nothing is skipped)."""
    n = len(lengths)
    if n == 0:
        return []
    assert max(lengths) <= budget, f"a sentence piece ({max(lengths)}) exceeds the budget {budget}"
    off = np.concatenate([[0], np.cumsum(lengths)])
    out, a = [], 0
    while True:
        b = a + 1
        while b < n and off[b + 1] - off[a] <= budget:
            b += 1
        out.append((a, b))
        if b >= n:
            break
        a2 = a + 1
        while a2 < b and off[a2] - off[a] < stride:
            a2 += 1
        a = a2
    return out


def parse_mode(mode: str) -> Tuple[str, int]:
    if mode in ("doc", "sent"):
        return mode, 0
    if mode.startswith("ctx_"):
        k = int(mode.split("_", 1)[1])
        assert k >= 1
        return "ctx", k
    raise ValueError(f"unknown input mode {mode!r}")


def window_budget(cfg: Dict, meta: Dict, domain: str) -> Tuple[List[int], int]:
    prefix = list(meta["domain_prefix_ids"][domain]) if cfg.get("domain_prefix", True) else []
    budget = int(cfg.get("max_len", SPEC["context"])) - int(cfg.get("n_virtual", 0)) - len(prefix)
    return prefix, budget


def build_doc_windows(doc: Dict, cfg: Dict, meta: Dict, purpose: str) -> List[Dict]:
    """Windows of one document. Each window: tokens (int32), markers / starts / sent_idx /
    labels (int64, one per sentence slot), score (float32 context score per slot)."""
    mode, k = parse_mode(cfg.get("input_mode", "doc"))
    eos = int(meta["eos_id"])
    max_sent = int(cfg.get("max_sent_tokens", 256))
    prefix, budget = window_budget(cfg, meta, doc["domain"])
    pieces = [sentence_piece(s, max_sent, eos) for s in doc["ids"]]
    lens = [len(p) for p in pieces]
    labels = doc["labels"]
    causal = cfg.get("attn_mode", "bidir") == "causal"
    wins = []

    def make(sent_ids: List[int], labelled: List[bool]) -> Dict:
        toks = list(prefix)
        markers, starts = [], []
        for i in sent_ids:
            starts.append(len(toks))
            toks.extend(pieces[i].tolist())
            markers.append(len(toks) - 1)
        toks = np.asarray(toks, dtype=np.int32)
        markers = np.asarray(markers, dtype=np.int64)
        L = len(toks)
        score = (markers.astype(np.float32) if causal else
                 np.minimum(markers, L - 1 - markers).astype(np.float32))
        lab = np.full(len(sent_ids), IGNORE, dtype=np.int64)
        if labels is not None:
            for j, i in enumerate(sent_ids):
                if labelled[j]:
                    lab[j] = int(labels[i])
        keep = [j for j in range(len(sent_ids)) if labelled[j]]
        return {"key": doc["key"], "tokens": toks, "markers": markers[keep],
                "starts": np.asarray(starts, dtype=np.int64)[keep],
                "sent_idx": np.asarray(sent_ids, dtype=np.int64)[keep], "labels": lab[keep],
                "score": score[keep]}

    if mode == "doc":
        frac = float(cfg.get("train_stride_frac" if purpose == "train" else "infer_stride_frac",
                             0.75 if purpose == "train" else 0.5))
        stride = max(1, int(frac * budget))
        for a, b in doc_ranges(lens, budget, stride):
            ids = list(range(a, b))
            wins.append(make(ids, [True] * len(ids)))
    elif mode == "sent":
        for i in range(len(pieces)):
            wins.append(make([i], [True]))
    else:
        n = len(pieces)
        for i in range(n):
            left = list(range(max(0, i - k), i))
            right = list(range(i + 1, min(n, i + 1 + k)))
            while sum(lens[j] for j in left + [i] + right) > budget and (left or right):
                if len(left) >= len(right):            # drop the farthest neighbour first
                    left = left[1:]
                else:
                    right = right[:-1]
            ids = left + [i] + right
            wins.append(make(ids, [j == i for j in ids]))
    return wins


def build_windows(docs: Dict[str, Dict], keys: Sequence[str], cfg: Dict, meta: Dict,
                  purpose: str) -> List[Dict]:
    out = []
    for key in keys:
        for w in build_doc_windows(docs[key], cfg, meta, purpose):
            out.append(w)
    return out


def assign_sentences(windows: Sequence[Dict], n_sent: Dict[str, int]) -> Dict[str, List[List]]:
    """{doc_key: [[(window_idx, slot_idx, score), ...] per sentence]}. Raises if a sentence is
    not covered by any window (this is the 'unpack' of the packing round trip)."""
    occ = {k: [[] for _ in range(n)] for k, n in n_sent.items()}
    for wi, w in enumerate(windows):
        for j, si in enumerate(w["sent_idx"].tolist()):
            occ[w["key"]][si].append((wi, j, float(w["score"][j])))
    for k, lst in occ.items():
        miss = [i for i, o in enumerate(lst) if not o]
        if miss:
            raise AssertionError(f"{k}: sentences {miss[:5]} are not covered by any window")
    return occ


def combine_slot_scores(windows: Sequence[Dict], slot_scores: Sequence[np.ndarray],
                        n_sent: Dict[str, int], how: str = "center") -> Dict[str, np.ndarray]:
    """Per-slot scores (one (n_slots, C) array per window) -> {doc_key: (n_sent, C)}.
    center: the slot with the highest context score (ties: earliest window); mean: average."""
    occ = assign_sentences(windows, n_sent)
    C = slot_scores[0].shape[1] if len(slot_scores) else len(SPEC["labels"])
    out = {}
    for key, lst in occ.items():
        arr = np.zeros((len(lst), C), dtype=np.float32)
        for i, o in enumerate(lst):
            if how == "mean":
                arr[i] = np.mean([slot_scores[wi][j] for wi, j, _ in o], axis=0)
            else:
                wi, j, _ = max(o, key=lambda t: (t[2], -t[0]))
                arr[i] = slot_scores[wi][j]
        out[key] = arr
    return out


def check_packing(doc: Dict, cfg: Dict, meta: Dict, purpose: str) -> Dict:
    """Round-trip check of one document's windows (gate G2a, selftest #2)."""
    wins = build_doc_windows(doc, cfg, meta, purpose)
    eos = int(meta["eos_id"])
    max_sent = int(cfg.get("max_sent_tokens", 256))
    ctx = int(cfg.get("max_len", SPEC["context"])) - int(cfg.get("n_virtual", 0))
    for w in wins:
        assert len(w["tokens"]) <= ctx, f"window of {len(w['tokens'])} tokens > {ctx}"
        assert np.all(np.diff(w["sent_idx"]) > 0) or len(w["sent_idx"]) <= 1, "order broken"
        for j, si in enumerate(w["sent_idx"].tolist()):
            m, s = int(w["markers"][j]), int(w["starts"][j])
            assert w["tokens"][m] == eos, "marker is not EOS"
            want = doc["ids"][si][:max_sent]
            assert np.array_equal(w["tokens"][s:m], want), "marker maps to the wrong sentence"
    occ = assign_sentences(wins, {doc["key"]: len(doc["ids"])})[doc["key"]]
    chosen = [max(o, key=lambda t: (t[2], -t[0])) for o in occ]
    recovered = [int(wins[wi]["sent_idx"][j]) for wi, j, _ in chosen]
    assert recovered == list(range(len(doc["ids"]))), "unpack(pack(doc)) != sentence order"
    return {"windows": len(wins), "max_window_tokens": max(len(w["tokens"]) for w in wins),
            "max_occurrences": max(len(o) for o in occ)}


# =============================================================================================
# 4. class statistics
# =============================================================================================
def class_counts(docs: Dict[str, Dict], keys: Sequence[str], n_labels: int) -> np.ndarray:
    labs = [docs[k]["labels"] for k in keys if docs[k]["labels"] is not None]
    if not labs:
        return np.zeros(n_labels, dtype=np.int64)
    return np.bincount(np.concatenate(labs).astype(np.int64), minlength=n_labels)


def class_weights(counts: np.ndarray, kind: str = "inv_sqrt") -> np.ndarray:
    """Inverse-sqrt-frequency weights, normalised to mean 1 over the classes seen in train;
    unseen classes get the largest weight (they cannot be learned, but must not be 0)."""
    c = counts.astype(np.float64)
    w = np.zeros_like(c)
    seen = c > 0
    w[seen] = 1.0 / np.sqrt(c[seen])
    if seen.any():
        w[seen] /= w[seen].mean()
        w[~seen] = w[seen].max()
    else:
        w[:] = 1.0
    return w.astype(np.float32)


# =============================================================================================
# 5. cache build (gates G0 / G1 / G2)
# =============================================================================================
def load_tokenizer(roots: Optional[List[str]] = None, explicit: Optional[str] = None):
    """bpe_model.json from the Phase-1 final/ dirs (all copies must be identical)."""
    from bpe_tokenizer import BPETokenizer
    paths = [explicit] if explicit else [p for p in find_files("bpe_model.json",
                                                               roots or default_input_roots())
                                         if "phase3_out" not in p]
    if not paths:
        raise FileNotFoundError("bpe_model.json not found; attach the Phase-1 output datasets")
    hashes = {file_md5(p) for p in paths}
    if len(hashes) != 1:
        raise GateFailure(f"found {len(hashes)} DIFFERENT bpe_model.json files: {paths}")
    return BPETokenizer.load(paths[0]), paths[0], hashes.pop()


def _pct(a: Sequence[int]) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    q = np.percentile(a, [50, 95, 99])
    return {"mean": float(a.mean()), "p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2]),
            "max": float(a.max())}


def build_cache(out_dir: str, roots: Optional[List[str]] = None,
                tokenizer_path: Optional[str] = None, max_sent_tokens: int = 256,
                log=print) -> Dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    gates: Dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    paths = locate_raw_files(os.path.join(out_dir, "raw"), roots, log=log)
    gates["raw_files"] = paths
    gates["raw_md5"] = {k: file_md5(v) for k, v in paths.items()}

    # ---- G0 ---------------------------------------------------------------------------------
    labels = load_label_vocab(paths["label_vocab"])
    lab2i = {l: i for i, l in enumerate(labels)}
    raw: Dict[str, List[Dict]] = {}
    for dom in SPEC["domains"]:
        for split in ("train", "dev", "test"):
            raw[f"{dom}_{split}"] = load_split_file(paths[f"{dom}_{split}"], split != "test")
    g0 = {"docs": {}, "sentences": {}, "class_hist": {}, "warnings": []}
    for name, docs in raw.items():
        ids = [d["id"] for d in docs]
        dup = [i for i, c in Counter(ids).items() if c > 1]
        if dup:
            raise GateFailure(f"G0: duplicate document ids in {name}: {dup[:5]}")
        for d in docs:
            if not d["text"]:
                raise GateFailure(f"G0: {name}: document {d['id']!r} has no sentences")
            if d["labels"] is not None:
                if len(d["labels"]) != len(d["text"]):
                    raise GateFailure(f"G0: {name}: document {d['id']!r} has "
                                      f"{len(d['text'])} sentences but {len(d['labels'])} labels")
                bad = sorted({l for l in d["labels"] if l not in lab2i})
                if bad:
                    raise GateFailure(f"G0: {name}: document {d['id']!r} uses labels outside "
                                      f"label_vocab.json: {bad}")
        g0["docs"][name] = len(docs)
        g0["sentences"][name] = int(sum(len(d["text"]) for d in docs))
        if not name.endswith("test"):
            hist = Counter(l for d in docs for l in d["labels"])
            g0["class_hist"][name] = {l: int(hist.get(l, 0)) for l in labels}
    ids_of = {n: {d["id"] for d in docs} for n, docs in raw.items()}
    clash = ids_of["CL_test"] & ids_of["IT_test"]
    g0["id_overlap"] = {}
    for a in ids_of:
        for b in ids_of:
            if a < b and ids_of[a] & ids_of[b]:
                g0["id_overlap"][f"{a}&{b}"] = sorted(ids_of[a] & ids_of[b])[:10]
    for pair, ex in g0["id_overlap"].items():
        if "test" in pair:
            msg = f"!!! LOUD WARNING: document ids shared by {pair}: {ex[:5]}"
            g0["warnings"].append(msg)
            log(msg)
    if clash:
        write_json(os.path.join(out_dir, "GATES.json"), {**gates, "G0": g0, "status": "FAILED"})
        raise GateFailure(
            f"G0: CL_test and IT_test share {len(clash)} document ids (e.g. {sorted(clash)[:3]}). "
            f"A single prediction.json dict would be ambiguous. STOP and ask the TAs on Discord; "
            f"do not guess a prefix.")
    log(f"   G0 passed: docs {g0['docs']} | sentences {g0['sentences']}")
    for n, h in g0["class_hist"].items():
        log(f"   class histogram {n}: {h}")

    # ---- G1: tokenisation -------------------------------------------------------------------
    tok, tok_path, tok_md5 = load_tokenizer(roots, tokenizer_path)
    gates["tokenizer"] = {"path": tok_path, "md5": tok_md5, "vocab": len(tok)}
    eos, pad, unk = int(tok.eos_token_id), int(tok.pad_token_id), int(tok.unk_token_id)
    if (eos, pad, unk) != (SPEC["eos_id"], SPEC["pad_id"], SPEC["unk_id"]):
        raise GateFailure(f"G1: tokenizer specials (eos, pad, unk) = {(eos, pad, unk)} differ "
                          f"from the spec")
    docs: Dict[str, Dict] = {}
    split_keys: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    test_ids: Dict[str, List[str]] = {}
    sent_len, doc_len = [], []
    n_unk = n_trunc = 0
    for dom in SPEC["domains"]:
        for split in ("train", "dev", "test"):
            for d in raw[f"{dom}_{split}"]:
                key = f"{dom}/{d['id']}"
                ids = []
                for s in d["text"]:
                    e = tok.encode(s.strip())
                    if not e:
                        e = [unk]
                        n_unk += 1
                    if len(e) > max_sent_tokens:
                        n_trunc += 1
                    ids.append(np.asarray(e, dtype=np.int32))
                    sent_len.append(len(e))
                doc_len.append(int(sum(len(x) + 1 for x in ids)))
                docs[key] = {"key": key, "id": d["id"], "domain": dom, "split": split,
                             "text": d["text"], "ids": ids,
                             "labels": (np.asarray([lab2i[l] for l in d["labels"]], dtype=np.int64)
                                        if d["labels"] is not None else None)}
                split_keys[split].append(key)
                if split == "test":
                    test_ids.setdefault(dom, []).append(d["id"])
    assert all(len(x) > 0 for d in docs.values() for x in d["ids"]), "zero-token sentence"
    prefix_ids = {dom: tok.encode(f"[{dom}]") for dom in SPEC["domains"]}
    g1 = {"tokens_per_sentence": _pct(sent_len), "tokens_per_doc": _pct(doc_len),
          "empty_sentences_unk_fallback": n_unk, "max_sent_tokens": max_sent_tokens,
          "sentences_truncated": n_trunc, "total_tokens": int(sum(sent_len)),
          "domain_prefix_ids": prefix_ids}
    log(f"   G1 passed: tokens/sentence {g1['tokens_per_sentence']} | tokens/doc "
        f"{g1['tokens_per_doc']} | unk fallbacks {n_unk} | truncated at {max_sent_tokens}: "
        f"{n_trunc}")

    meta = {"version": CACHE_VERSION, "labels": labels, "split": split_keys,
            "test_ids": test_ids, "eos_id": eos, "pad_id": pad, "unk_id": unk,
            "vocab_size": len(tok), "tokenizer_md5": tok_md5, "tokenizer_path": tok_path,
            "domain_prefix_ids": prefix_ids, "raw_md5": gates["raw_md5"],
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- G2a: packing round trip ------------------------------------------------------------
    g2 = {"packing": {}}
    for mode in ("doc", "sent", "ctx_1"):
        for attn in ("bidir", "causal"):
            cfg = {"input_mode": mode, "attn_mode": attn, "max_len": SPEC["context"],
                   "max_sent_tokens": max_sent_tokens, "n_virtual": 20}
            st = Counter()
            for key, d in docs.items():
                for purpose in ("train", "infer"):
                    r = check_packing(d, cfg, meta, purpose)
                    st["windows_" + purpose] += r["windows"]
                    st["max_window_tokens"] = max(st["max_window_tokens"], r["max_window_tokens"])
            g2["packing"][f"{mode}/{attn}"] = dict(st)
    log(f"   G2a passed: window packing round trip for doc/sent/ctx_1 x bidir/causal "
        f"({g2['packing']['doc/bidir']})")

    # ---- G2b: submission validator round trip -------------------------------------------------
    from rr_predict import g2_validator_roundtrip
    n_sent = {d["id"]: len(d["text"]) for d in docs.values() if d["split"] == "test"}
    g2["validator"] = g2_validator_roundtrip(n_sent, labels, out_dir)
    log(f"   G2b passed: validator accepts the dummy all-'None' prediction and rejects "
        f"{len(g2['validator']['rejected'])} corrupted variants")

    with open(os.path.join(out_dir, "docs.pkl.tmp"), "wb") as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(os.path.join(out_dir, "docs.pkl.tmp"), os.path.join(out_dir, "docs.pkl"))
    write_json(os.path.join(out_dir, "meta.json"), meta)
    gates.update({"G0": g0, "G1": g1, "G2": g2, "status": "PASSED",
                  "seconds": round(time.time() - t0, 1)})
    write_json(os.path.join(out_dir, "GATES.json"), gates)
    log(f"   data cache written to {out_dir} in {gates['seconds']} s")
    return meta


def load_cache(data_dir: str) -> Tuple[Dict, Dict[str, Dict]]:
    meta = read_json(os.path.join(data_dir, "meta.json"))
    if meta is None:
        raise FileNotFoundError(f"no data cache in {data_dir}")
    with open(os.path.join(data_dir, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    return meta, docs


def find_or_build_cache(data_dir: str, roots: Optional[List[str]] = None, log=print,
                        **kw) -> str:
    """Reuse a cache (here or attached) built with the SAME tokenizer and raw files, else build."""
    roots = roots or default_input_roots()
    try:
        _tok, _p, tok_md5 = load_tokenizer(roots)
    except FileNotFoundError:
        tok_md5 = None

    def ok(d):
        meta = read_json(os.path.join(d, "meta.json"))
        return (isinstance(meta, dict) and meta.get("version") == CACHE_VERSION
                and os.path.exists(os.path.join(d, "docs.pkl"))
                and (tok_md5 is None or meta.get("tokenizer_md5") == tok_md5)
                and read_json(os.path.join(d, "GATES.json"), {}).get("status") == "PASSED")
    if ok(data_dir):
        return data_dir
    for m in find_files("meta.json", roots):
        d = os.path.dirname(m)
        if ok(d):
            log(f"   using the data cache attached at {d}")
            return d
    build_cache(data_dir, roots=roots, log=log, **kw)
    return data_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--out", default="/kaggle/working/phase3_data")
    ap.add_argument("--tokenizer", default=None, help="explicit path to bpe_model.json")
    ap.add_argument("--max-sent-tokens", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="rebuild even if a valid cache exists")
    a = ap.parse_args()
    try:
        if a.force or a.tokenizer:
            build_cache(a.out, tokenizer_path=a.tokenizer, max_sent_tokens=a.max_sent_tokens)
        else:
            d = find_or_build_cache(a.out, max_sent_tokens=a.max_sent_tokens)
            g = read_json(os.path.join(d, "GATES.json"), {})
            print(f"   data cache: {d} | gates {g.get('status')} | G0 docs "
                  f"{(g.get('G0') or {}).get('docs')} | G1 tokens/sentence "
                  f"{(g.get('G1') or {}).get('tokens_per_sentence')}")
            for w in (g.get("G0") or {}).get("warnings", []):
                print(w)
    except GateFailure as e:
        print("\nSTOP AND ASK:", e, file=sys.stderr)
        sys.exit(3)
    except FileNotFoundError as e:
        print("\nMISSING INPUT:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
