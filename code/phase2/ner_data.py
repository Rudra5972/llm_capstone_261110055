"""
ner_data.py -- Phase-2 data: loading, gates G0/G1/G2, IOB2 tagging, chunking, dev split, cache.

Torch-free (numpy + stdlib). Run on CPU:

    python ner_data.py build --out /kaggle/working/phase2_data

Gates (section 11 of the brief):
  G0  every gold entity satisfies text[start:end] == entity_text  (end exclusive)
  G1  gold spans -> token tags -> decoded spans round trip ("ceiling") strict F1 >= 0.97 (real data: 0.9825, the rest is gold ending mid-BPE-token, unreachable by any token tagger)
  G2  token tags -> chunks -> overlap merge -> decode gives exactly the G1 spans

Text is NEVER normalised: offsets are Python str indices into the raw document string.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import numpy as np

from ner_common import default_input_roots, file_md5, find_files, read_json, write_json

EXPECTED_LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT",
                   "STAT", "PREC", "DATE", "CASENO"]
PHASE2_FILES = {
    "fold1_text": "fold1-text.json", "fold1_gold": "fold1-gold.json",
    "fold2_text": "fold2-text.json", "fold2_gold": "fold2-gold.json",
    "labels": "labels.txt", "fold3_text": "fold3-text.json",
}
HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_PATHS = {   # tried in order when the files are not attached as a Kaggle dataset
    "fold1_text": ["Phase_2_LNER/Train/fold1-text.json"],
    "fold1_gold": ["Phase_2_LNER/Train/fold1-gold.json"],
    "fold2_text": ["Phase_2_LNER/Train/fold2-text.json"],
    "fold2_gold": ["Phase_2_LNER/Train/fold2-gold.json"],
    "labels": ["Phase_2_LNER/Train/labels.txt", "Phase_2_LNER/labels.txt"],
    "fold3_text": ["Phase_2_LNER/Test/fold3-text.json", "Phase_2_LNER/Train/fold3-text.json"],
}
IGNORE = -100


class GateFailure(RuntimeError):
    """A 'stop and ask' condition from section 12 of the brief."""


# =============================================================================================
# 1. locating / downloading the raw files
# =============================================================================================
def _hf_read_token():
    """Optional READ token for downloading the official data (never used for writing)."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient          # only exists on Kaggle
        return UserSecretsClient().get_secret("HF")
    except Exception:
        return None


def _download(rel_path, dest, token):
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{rel_path}"
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


def locate_raw_files(raw_dir, roots=None, allow_download=True, log=print):
    """Find the six Phase-2 files under the input roots, else download them (read-only)."""
    roots = roots or default_input_roots()
    found = {}
    for key, name in PHASE2_FILES.items():
        hits = [h for h in find_files(name, roots + [raw_dir]) if "phase2_out" not in h]
        if hits:
            # prefer paths that look like the official layout
            hits.sort(key=lambda p: ("Phase_2" not in p, len(p)))
            found[key] = hits[0]
    missing = [k for k in PHASE2_FILES if k not in found]
    if missing and allow_download:
        os.makedirs(raw_dir, exist_ok=True)
        token = _hf_read_token()
        for key in missing:
            dest = os.path.join(raw_dir, PHASE2_FILES[key])
            err = None
            for rel in HF_PATHS[key]:
                try:
                    log(f"   downloading {rel} ...")
                    _download(rel, dest, token)
                    found[key] = dest
                    break
                except (urllib.error.URLError, OSError) as e:
                    err = e
            if key not in found:
                log(f"   could not download {PHASE2_FILES[key]}: {err}")
    missing = [PHASE2_FILES[k] for k in PHASE2_FILES if k not in found]
    if missing:
        raise FileNotFoundError(
            "Phase-2 files not found: " + ", ".join(missing) + ". Attach the official "
            "Phase_2_LNER folder as a Kaggle dataset (or add a read-only HF token as the Kaggle "
            "secret 'HF').")
    return found


# =============================================================================================
# 2. tolerant loaders
# =============================================================================================
_ID_KEYS = ("id", "doc_id", "docid", "document_id", "case_id", "uid", "name", "file")
_TEXT_KEYS = ("text", "Text", "content", "body", "document", "judgment", "raw_text")
_START_KEYS = ("start", "start_offset", "start_char", "begin", "char_start", "startOffset",
               "start_idx")
_END_KEYS = ("end", "end_offset", "end_char", "stop", "char_end", "endOffset", "end_idx")
_LABEL_KEYS = ("label", "labels", "type", "entity_type", "tag", "entity_label", "category",
               "ner_label", "entity_group")
_ETEXT_KEYS = ("text", "entity_text", "entity", "span", "mention", "word", "surface")
_LIST_KEYS = ("entities", "annotations", "spans", "ner", "labels", "result", "gold", "entity",
              "ents", "mentions")


def load_texts(path):
    """{doc_id(str): text(str)} from {id: text}, {id: {"text":..}} or [{"id":..,"text":..}]."""
    obj = read_json(path)
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, dict):
                t = next((v[t] for t in _TEXT_KEYS if isinstance(v.get(t), str)), None)
                if t is None and isinstance(v.get("data"), dict):
                    t = next((v["data"][t] for t in _TEXT_KEYS
                              if isinstance(v["data"].get(t), str)), None)
                if t is None:
                    raise ValueError(f"{path}: cannot find the text of document {k!r}")
                out[str(k)] = t
            else:
                raise ValueError(f"{path}: unexpected value type {type(v)} for {k!r}")
    elif isinstance(obj, list):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict):
                raise ValueError(f"{path}: list element {i} is not an object")
            src = rec.get("data") if isinstance(rec.get("data"), dict) else rec
            did = next((rec[k] for k in _ID_KEYS if k in rec), None)
            t = next((src[k] for k in _TEXT_KEYS if isinstance(src.get(k), str)), None)
            if did is None or t is None:
                raise ValueError(f"{path}: element {i} has no id/text field: {list(rec)[:8]}")
            out[str(did)] = t
    else:
        raise ValueError(f"{path}: top level must be an object or a list")
    return out


def load_labels(path):
    labs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for part in s.replace(",", " ").split():
                if part[:2] in ("B-", "I-", "E-", "S-"):
                    part = part[2:]
                if part != "O" and part not in labs:
                    labs.append(part)
    return labs


def _ann_from_seq(a):
    """[s, e, label(, text)] or [label, s, e(, text)] -> (s, e, label, text|None)."""
    if len(a) >= 3 and isinstance(a[0], int) and isinstance(a[1], int) and isinstance(a[2], str):
        return int(a[0]), int(a[1]), a[2], (a[3] if len(a) > 3 and isinstance(a[3], str) else None)
    if len(a) >= 3 and isinstance(a[0], str) and isinstance(a[1], int) and isinstance(a[2], int):
        return int(a[1]), int(a[2]), a[0], (a[3] if len(a) > 3 and isinstance(a[3], str) else None)
    return None


def _ann_from_dict(d):
    if isinstance(d.get("value"), dict):                     # Label-Studio style result
        merged = dict(d)
        merged.update(d["value"])
        d = merged
    s = next((d[k] for k in _START_KEYS if isinstance(d.get(k), int)), None)
    e = next((d[k] for k in _END_KEYS if isinstance(d.get(k), int)), None)
    lab = None
    for k in _LABEL_KEYS:
        v = d.get(k)
        if isinstance(v, str):
            lab = v
            break
        if isinstance(v, list) and v and isinstance(v[0], str):
            lab = v[0]
            break
    if s is None or e is None or lab is None:
        return None
    txt = next((d[k] for k in _ETEXT_KEYS if isinstance(d.get(k), str)), None)
    return int(s), int(e), lab, txt


def _parse_ann_list(v):
    """Parse whatever holds one document's annotations -> list of tuples (or None)."""
    if isinstance(v, dict):
        for k in _LIST_KEYS:
            if k in v and isinstance(v[k], list):
                r = _parse_ann_list(v[k])
                if r is not None:
                    return r
        return None
    if not isinstance(v, list):
        return None
    out = []
    for a in v:
        if isinstance(a, (list, tuple)):
            r = _ann_from_seq(a)
        elif isinstance(a, dict):
            if "result" in a and isinstance(a["result"], list):      # Label-Studio annotation
                sub = _parse_ann_list(a["result"])
                if sub is None:
                    return None
                out.extend(sub)
                continue
            r = _ann_from_dict(a)
        else:
            r = None
        if r is None:
            return None
        out.append(r)
    return out


def parse_gold(path):
    """{doc_id: [(start, end, label, text|None)]} + a human-readable schema description."""
    obj = read_json(path)
    gold = {}
    example = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            anns = _parse_ann_list(v)
            if anns is None:
                raise GateFailure(f"{path}: cannot parse the annotations of document {k!r}; "
                                  f"value starts with {json.dumps(v)[:300]}")
            gold[str(k)] = anns
            if example is None and anns:
                example = json.dumps(v if not isinstance(v, list) else v[:1])[:300]
        schema = "object {doc_id: annotations}"
    elif isinstance(obj, list):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict):
                raise GateFailure(f"{path}: list element {i} is not an object")
            did = next((rec[k] for k in _ID_KEYS if k in rec), None)
            if did is None:
                raise GateFailure(f"{path}: element {i} has no document id field")
            anns = _parse_ann_list({k: v for k, v in rec.items() if k not in _ID_KEYS})
            if anns is None:
                raise GateFailure(f"{path}: cannot parse annotations of element {i}: "
                                  f"{json.dumps(rec)[:300]}")
            gold[str(did)] = gold.get(str(did), []) + anns
            if example is None and anns:
                example = json.dumps(rec)[:300]
        schema = "list of {id, annotations} records"
    else:
        raise GateFailure(f"{path}: top level must be an object or a list")
    return gold, {"layout": schema, "example": example}


# =============================================================================================
# 3. gate G0 + overlap resolution
# =============================================================================================
def _ws_norm(s):
    return " ".join(s.split())


def check_g0(texts, gold, labels, allow_inclusive_ends=False, log=print):
    """Verify offsets against entity texts. Returns (clean_gold, report). Raises GateFailure."""
    rep = Counter()
    bad_examples, clean = [], {}
    unknown_labels = Counter()
    for did, anns in gold.items():
        if did not in texts:
            raise GateFailure(f"gold document {did!r} has no text")
        t = texts[did]
        out = []
        for (s, e, lab, et) in anns:
            rep["total"] += 1
            if lab not in labels:
                unknown_labels[lab] += 1
                continue
            if not (0 <= s < e <= len(t)):
                rep["out_of_bounds"] += 1
                bad_examples.append((did, s, e, lab, et, "out of bounds"))
                continue
            if et is None:
                rep["no_text"] += 1
            elif t[s:e] == et:
                rep["exact"] += 1
            elif t[s:e + 1] == et:
                rep["inclusive_end"] += 1
                if allow_inclusive_ends:
                    e = e + 1
                else:
                    bad_examples.append((did, s, e, lab, et, repr(t[s:e])))
                    continue
            elif _ws_norm(t[s:e]) == _ws_norm(et):
                rep["ws_normalised_match"] += 1
            else:
                rep["mismatch"] += 1
                bad_examples.append((did, s, e, lab, et, repr(t[s:e])))
                continue
            out.append((s, e, lab))
        clean[did] = out
    report = dict(rep)
    report["unknown_labels"] = dict(unknown_labels)
    report["examples_of_problems"] = [list(x) for x in bad_examples[:15]]
    if unknown_labels:
        raise GateFailure(f"G0: gold uses labels not in labels.txt: {dict(unknown_labels)}")
    if rep["inclusive_end"] and not allow_inclusive_ends:
        raise GateFailure(
            f"G0: {rep['inclusive_end']} entities only match with an INCLUSIVE end offset "
            f"(text[s:e+1]). The brief assumes exclusive ends; confirm the convention with the "
            f"official evaluator, then rerun with --allow-inclusive-ends. Examples: "
            f"{bad_examples[:3]}")
    if rep["mismatch"] or rep["out_of_bounds"]:
        raise GateFailure(f"G0 failed: {rep['mismatch']} mismatches, {rep['out_of_bounds']} "
                          f"out-of-bounds. Examples: {bad_examples[:5]}")
    log(f"   G0 passed: {rep['total']} entities | exact {rep['exact']} | "
        f"ws-normalised {rep['ws_normalised_match']} | without entity text {rep['no_text']}")
    return clean, report


def resolve_overlaps(anns):
    """Keep the longer (outer) entity when two overlap. Returns (kept_sorted, n_nested, n_cross)."""
    order = sorted(set(anns), key=lambda a: (-(a[1] - a[0]), a[0], a[2]))
    kept, nested, cross = [], 0, 0
    for a in order:
        clash = None
        for k in kept:
            if a[0] < k[1] and k[0] < a[1]:
                clash = k
                break
        if clash is None:
            kept.append(a)
        elif clash[0] <= a[0] and a[1] <= clash[1]:
            nested += 1
        else:
            cross += 1
    kept.sort()
    return kept, nested, cross


# =============================================================================================
# 4. IOB2 tagging / decoding
# =============================================================================================
def make_tag_names(labels):
    """index 0 = O, then B-X = 1 + 2k, I-X = 2 + 2k for label k."""
    names = ["O"]
    for lab in labels:
        names += [f"B-{lab}", f"I-{lab}"]
    return names


def tag_doc(td, ents, n_labels_map):
    """Token IOB2 tags for one document.

    td   : ner_tok.TokenizedDoc
    ents : sorted, non-overlapping [(start, end, label)]
    n_labels_map : {label: k}
    Returns (tags int16 array, stats dict).
    """
    n = len(td)
    owner = np.full(n, -1, dtype=np.int64)              # entity index per token (-1 = none)
    es_, ee_ = td.eff_start, td.eff_end
    straddle = 0
    j = 0
    for i in range(n):
        ts, te = int(es_[i]), int(ee_[i])
        if ts < 0:
            continue
        while j < len(ents) and ents[j][1] <= ts:
            j += 1
        best, best_ov = -1, 0
        k = j
        while k < len(ents) and ents[k][0] < te:
            ov = min(te, ents[k][1]) - max(ts, ents[k][0])
            if ov > best_ov:
                best, best_ov = k, ov
            k += 1
        if best >= 0:
            if best_ov * 2 > (te - ts):                 # majority of the token inside the entity
                owner[i] = best
                if best_ov < te - ts:
                    straddle += 1
            else:
                straddle += 1
    # whitespace-only tokens strictly inside an entity join it
    nonws = np.nonzero(es_ >= 0)[0]
    for i in range(n):
        if es_[i] >= 0 or td.raw_start[i] < 0:
            continue
        p = np.searchsorted(nonws, i) - 1
        q = p + 1
        if p >= 0 and q < len(nonws):
            a, b = owner[nonws[p]], owner[nonws[q]]
            if a >= 0 and a == b:
                owner[i] = a
    # continuation-byte tokens (own no character) follow the previous token
    for i in range(1, n):
        if td.raw_start[i] < 0 and owner[i - 1] >= 0:
            owner[i] = owner[i - 1]
    tags = np.zeros(n, dtype=np.int16)
    seen = set()
    for i in range(n):
        o = owner[i]
        if o < 0:
            continue
        k = n_labels_map[ents[o][2]]
        if o in seen:
            tags[i] = 2 + 2 * k
        else:
            tags[i] = 1 + 2 * k
            seen.add(o)
    lost = len(ents) - len(seen)
    return tags, {"straddling_tokens": straddle, "entities_without_tokens": lost}


def decode_tags(tags, td, labels):
    """IOB2 tags -> [(start, end, label)] using effective (whitespace-trimmed) token spans.

    A stray I-X (no open X span) starts a new span, the usual lenient convention; constrained
    Viterbi never produces one.
    """
    spans = []
    cur = None                                          # [k, start, end]
    es_, ee_ = td.eff_start, td.eff_end

    def close():
        if cur is not None and cur[1] >= 0 and cur[2] > cur[1]:
            spans.append((cur[1], cur[2], labels[cur[0]]))

    for i in range(len(tags)):
        t = int(tags[i])
        if t <= 0:
            close()
            cur = None
            continue
        k, is_b = (t - 1) // 2, (t % 2 == 1)
        s, e = int(es_[i]), int(ee_[i])
        if is_b or cur is None or cur[0] != k:
            close()
            cur = [k, s, e]
        else:
            if s >= 0 and cur[1] < 0:
                cur[1] = s
            if e >= 0:
                cur[2] = e
    close()
    return spans


# =============================================================================================
# 5. learned boundary snapping (used only if it raises the train ceiling; gate G1)
# =============================================================================================
def _is_punct(c):
    return not c.isalnum() and not c.isspace()


class SnapRules:
    """Single-character boundary fixes learned from the gold round trip.

    rules[label] = {"end_trim": set, "end_extend": set, "start_trim": set, "start_extend": set}
    end_trim  c : drop a final character c       end_extend  c : include a following char c
    start_trim c: drop a leading character c     start_extend c: include a preceding char c
    A rule is kept only if, on training documents, it fixes more spans than it breaks.
    """
    KINDS = ("end_trim", "end_extend", "start_trim", "start_extend")

    def __init__(self, rules=None):
        self.rules = rules or {}

    def to_json(self):
        return {lab: {k: sorted(v) for k, v in r.items()} for lab, r in self.rules.items()}

    @classmethod
    def from_json(cls, obj):
        return cls({lab: {k: set(v) for k, v in r.items()} for lab, r in (obj or {}).items()})

    @staticmethod
    def _candidate(kind, s, e, text):
        if kind == "end_trim" and e - s > 1 and _is_punct(text[e - 1]):
            return text[e - 1], (s, e - 1)
        if kind == "end_extend" and e < len(text) and _is_punct(text[e]):
            return text[e], (s, e + 1)
        if kind == "start_trim" and e - s > 1 and _is_punct(text[s]):
            return text[s], (s + 1, e)
        if kind == "start_extend" and s > 0 and _is_punct(text[s - 1]):
            return text[s - 1], (s - 1, e)
        return None, None

    def fit(self, decoded, gold, texts):
        """decoded/gold: {doc: [(s, e, label)]}."""
        self.rules = {}
        for kind in self.KINDS:
            score = defaultdict(lambda: [0, 0])          # (label, char) -> [fixes, breaks]
            for did, spans in decoded.items():
                gset = set(gold.get(did, []))
                text = texts[did]
                for (s, e, lab) in spans:
                    c, new = self._candidate(kind, s, e, text)
                    if c is None:
                        continue
                    now_ok = (s, e, lab) in gset
                    new_ok = (new[0], new[1], lab) in gset
                    if new_ok and not now_ok:
                        score[(lab, c)][0] += 1
                    elif now_ok and not new_ok:
                        score[(lab, c)][1] += 1
            for (lab, c), (fix, brk) in score.items():
                if fix > brk:
                    self.rules.setdefault(lab, {k: set() for k in self.KINDS})[kind].add(c)
            # apply this kind before learning the next one, mirroring apply()
            decoded = {d: [self._apply_kind(kind, sp, texts[d]) for sp in v]
                       for d, v in decoded.items()}
        return self

    def _apply_kind(self, kind, span, text):
        s, e, lab = span
        r = self.rules.get(lab)
        if not r or not r.get(kind):
            return span
        c, new = self._candidate(kind, s, e, text)
        if c is not None and c in r[kind]:
            return (new[0], new[1], lab)
        return span

    def apply(self, spans, text):
        out = []
        for sp in spans:
            for kind in self.KINDS:
                sp = self._apply_kind(kind, sp, text)
            out.append(sp)
        return out


def strict_f1(pred, gold):
    """Micro strict F1 over {doc: [(s,e,label)]} (used for the ceiling)."""
    tp = fp = fn = 0
    for did in set(pred) | set(gold):
        p, g = set(pred.get(did, [])), set(gold.get(did, []))
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


# =============================================================================================
# 6. chunking + merging (gate G2)
# =============================================================================================
def make_windows(n_tokens, chunk_len, stride):
    """Token windows [a, b) of at most chunk_len-1 tokens (one slot is the EOS prefix)."""
    w = chunk_len - 1
    assert w >= 1 and 1 <= stride <= w, "need 1 <= stride <= chunk_len - 1"
    if n_tokens <= w:
        return [(0, n_tokens)]
    out, a = [], 0
    while True:
        b = min(a + w, n_tokens)
        out.append((a, b))
        if b >= n_tokens:
            break
        a += stride
    return out


def centre_weights(length):
    """Weight of each position in a window: distance to the nearer edge (+1), so tokens near
    the centre of a chunk dominate where chunks overlap."""
    i = np.arange(length, dtype=np.float32)
    return np.minimum(i + 1, length - i)


def merge_window_scores(n_tokens, windows, window_scores):
    """Weighted average of per-window scores (each (len, C)) onto the document (n, C)."""
    C = window_scores[0].shape[1]
    acc = np.zeros((n_tokens, C), dtype=np.float64)
    wsum = np.zeros((n_tokens, 1), dtype=np.float64)
    for (a, b), sc in zip(windows, window_scores):
        w = centre_weights(b - a)[:, None]
        acc[a:b] += sc * w
        wsum[a:b] += w
    assert (wsum > 0).all(), "some tokens are not covered by any window"
    return (acc / wsum).astype(np.float32)


def build_train_chunks(docs, chunk_len, stride, neg_keep, seed, eos_id=0):
    """Training chunks: [EOS] + tokens, labels -100 on EOS. Keeps every chunk with an entity
    and a seeded fraction `neg_keep` of entity-free chunks."""
    rng = random.Random(seed)
    xs, ys = [], []
    n_pos = n_neg_total = n_neg_kept = 0
    for d in docs:
        ids, tags = d["ids"], d["tags"]
        for (a, b) in make_windows(len(ids), chunk_len, stride):
            y = tags[a:b]
            if (y > 0).any():
                n_pos += 1
            else:
                n_neg_total += 1
                if rng.random() >= neg_keep:
                    continue
                n_neg_kept += 1
            xs.append(np.concatenate([[eos_id], ids[a:b]]).astype(np.int32))
            ys.append(np.concatenate([[IGNORE], y]).astype(np.int16))
    return xs, ys, {"chunks": len(xs), "with_entities": n_pos, "empty_total": n_neg_total,
                    "empty_kept": n_neg_kept}


# =============================================================================================
# 7. dev split
# =============================================================================================
def make_dev_split(doc_ids, gold, labels, frac=0.15, seed=42, tries=2000):
    """Document-level split with every label present in dev (and in train)."""
    ids = sorted(doc_ids)
    n_dev = max(1, int(round(frac * len(ids))))
    labs_of = {d: {a[2] for a in gold.get(d, [])} for d in ids}
    rng = random.Random(seed)
    best, best_cov = None, -1
    for _ in range(tries):
        perm = ids[:]
        rng.shuffle(perm)
        dev, train = sorted(perm[:n_dev]), sorted(perm[n_dev:])
        dev_l = set().union(*(labs_of[d] for d in dev)) if dev else set()
        tr_l = set().union(*(labs_of[d] for d in train)) if train else set()
        cov = len(dev_l & set(labels)) + len(tr_l & set(labels))
        if cov > best_cov:
            best, best_cov = (train, dev), cov
        if cov == 2 * len(labels):
            break
    train, dev = best
    dev_l = set().union(*(labs_of[d] for d in dev))
    return {"train": train, "dev": dev, "seed": seed, "frac": frac,
            "labels_missing_in_dev": sorted(set(labels) - dev_l)}


# =============================================================================================
# 8. cache build (S0 + S1 + S2)
# =============================================================================================
def load_tokenizer(roots=None, explicit=None, log=print):
    """Find bpe_model.json (from the Phase-1 final/ dirs) and load it with the verbatim
    bpe_tokenizer.py. All copies found must be identical."""
    from bpe_tokenizer import BPETokenizer
    if explicit:
        paths = [explicit]
    else:
        paths = [p for p in find_files("bpe_model.json", roots or default_input_roots())
                 if "phase2_out" not in p]
    if not paths:
        raise FileNotFoundError("bpe_model.json not found; attach the Phase-1 output datasets")
    hashes = {file_md5(p) for p in paths}
    if len(hashes) != 1:
        raise GateFailure(f"found {len(hashes)} DIFFERENT bpe_model.json files: {paths}")
    tok = BPETokenizer.load(paths[0])
    return tok, paths[0], hashes.pop()


def build_cache(out_dir, roots=None, tokenizer_path=None, allow_inclusive_ends=False,
                max_overlap_frac=0.02, ceiling_min=0.97, dev_frac=0.15, seed=42,
                chunk_len=512, stride=256, log=print):
    from ner_tok import OffsetEncoder, verify_offsets
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- S0: files, schema, stats ---------------------------------------------------------
    paths = locate_raw_files(os.path.join(out_dir, "raw"), roots, log=log)
    report["raw_files"] = {k: v for k, v in paths.items()}
    report["raw_md5"] = {k: file_md5(v) for k, v in paths.items()}
    labels = load_labels(paths["labels"])
    if sorted(labels) != sorted(EXPECTED_LABELS):
        log(f"   WARNING: labels.txt {labels} differs from the 12 expected labels")
    texts = {}
    fold_of = {}
    orig_id = {}                                          # cache key -> id as in the JSON file
    for fold in ("fold1", "fold2", "fold3"):
        t = load_texts(paths[f"{fold}_text"])
        for did, txt in t.items():
            key = did
            if key in texts:                              # id collision across folds
                key = f"{fold}:{did}"
            texts[key] = txt
            fold_of[key] = fold
            orig_id[key] = did
    gold_raw, schemas = {}, {}
    for fold in ("fold1", "fold2"):
        g, sch = parse_gold(paths[f"{fold}_gold"])
        schemas[fold] = sch
        ids_this_fold = {k for k, f in fold_of.items() if f == fold}
        for did, anns in g.items():
            key = did if did in ids_this_fold else f"{fold}:{did}"
            if key not in ids_this_fold:
                raise GateFailure(f"{fold}-gold has document {did!r} that is not in {fold}-text")
            gold_raw[key] = anns
        for did in ids_this_fold:
            gold_raw.setdefault(did, [])
    report["gold_schema"] = schemas
    log(f"   gold schema: {json.dumps(schemas)[:600]}")

    # ---- G0 ---------------------------------------------------------------------------------
    gold, g0 = check_g0(texts, gold_raw, labels, allow_inclusive_ends, log)
    report["G0"] = g0

    # ---- overlaps ---------------------------------------------------------------------------
    n_nested = n_cross = n_total = 0
    for did in gold:
        kept, ne, cr = resolve_overlaps(gold[did])
        n_total += len(set(gold[did]))
        n_nested += ne
        n_cross += cr
        gold[did] = kept
    frac = (n_nested + n_cross) / max(1, n_total)
    report["overlaps"] = {"nested_dropped": n_nested, "crossing_dropped": n_cross,
                          "fraction": frac}
    log(f"   overlaps: {n_nested} nested + {n_cross} crossing dropped ({frac:.2%})")
    if frac > max_overlap_frac:
        raise GateFailure(f"{frac:.2%} of gold entities are nested/overlapping (> "
                          f"{max_overlap_frac:.0%}); IOB2 cannot represent them -- stop and ask")

    # ---- S1: tokenise with offsets, tag, ceiling ---------------------------------------------
    tok, tok_path, tok_md5 = load_tokenizer(roots, tokenizer_path, log)
    report["tokenizer"] = {"path": tok_path, "md5": tok_md5, "vocab": len(tok)}
    enc = OffsetEncoder(tok)
    lab2k = {l: k for k, l in enumerate(labels)}
    docs = {}
    tag_stats = Counter()
    bad_offsets = 0
    for did, text in texts.items():
        td = enc.encode(text, check=True)
        bad_offsets += verify_offsets(text, td)
        rec = {"doc_id": did, "orig_id": orig_id[did], "fold": fold_of[did], "text": text,
               **td.to_dict()}
        if did in gold:
            tags, st = tag_doc(td, gold[did], lab2k)
            rec["tags"] = tags
            rec["gold"] = gold[did]
            tag_stats.update(st)
        docs[did] = rec
    if bad_offsets:
        raise GateFailure(f"offset encoder inconsistency: {bad_offsets} problems")
    report["tagging"] = dict(tag_stats)

    from ner_tok import TokenizedDoc
    decoded = {d: decode_tags(r["tags"], TokenizedDoc.from_dict(r), labels)
               for d, r in docs.items() if "tags" in r}
    ceiling_raw = strict_f1(decoded, gold)
    snap = SnapRules().fit(decoded, gold, texts)
    snapped = {d: snap.apply(v, texts[d]) for d, v in decoded.items()}
    ceiling_snap = strict_f1(snapped, gold)
    use_snap = ceiling_snap > ceiling_raw
    ceiling = max(ceiling_raw, ceiling_snap)
    # mismatch categories for the report
    cats = Counter()
    examples = []
    final = snapped if use_snap else decoded
    for did, g in gold.items():
        p = set(final.get(did, []))
        for (s, e, lab) in g:
            if (s, e, lab) in p:
                continue
            ov = [q for q in p if q[0] < e and s < q[1] and q[2] == lab]
            txt = texts[did]
            if not ov:
                cats["no_tokens_for_entity"] += 1
            elif txt[s].isspace() or txt[e - 1].isspace():
                cats["gold_starts_or_ends_on_whitespace"] += 1
            elif any(q[0] != s for q in ov):
                cats["start_inside_token"] += 1
            else:
                cats["end_inside_token"] += 1
            if len(examples) < 20:
                examples.append({"doc": did, "gold": [s, e, lab], "gold_text": txt[s:e],
                                 "decoded": [list(q) + [txt[q[0]:q[1]]] for q in ov][:2]})
    report["G1"] = {"ceiling_raw": ceiling_raw, "ceiling_with_snapping": ceiling_snap,
                    "use_snapping": use_snap, "snap_rules": snap.to_json(),
                    "mismatch_categories": dict(cats), "examples": examples}
    log(f"   G1 ceiling: raw {ceiling_raw:.4f} | with snapping {ceiling_snap:.4f} "
        f"(snapping {'ON' if use_snap else 'off'}) | mismatches {dict(cats)}")
    if ceiling < ceiling_min:
        write_json(os.path.join(out_dir, "data_report.json"), report)
        raise GateFailure(f"G1 failed: ceiling {ceiling:.4f} < {ceiling_min}. See "
                          f"data_report.json (G1.examples) -- stop and ask")

    # ---- dev split ---------------------------------------------------------------------------
    labeled = sorted(d for d in docs if "tags" in docs[d])
    split = make_dev_split(labeled, gold, labels, dev_frac, seed)
    split["test"] = sorted(d for d in docs if fold_of[d] == "fold3")
    if split["labels_missing_in_dev"]:
        log(f"   WARNING: labels missing in dev: {split['labels_missing_in_dev']}")
    report["split"] = {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}

    # ---- G2: tags -> windows -> merge -> decode == G1 spans -----------------------------------
    n_tags = 1 + 2 * len(labels)
    g2_bad = 0
    for did in labeled:
        r = docs[did]
        td = TokenizedDoc.from_dict(r)
        wins = make_windows(len(td), chunk_len, stride)
        onehot = np.eye(n_tags, dtype=np.float32)[r["tags"].astype(np.int64)]
        merged = merge_window_scores(len(td), wins, [onehot[a:b] for a, b in wins])
        if decode_tags(merged.argmax(1), td, labels) != decoded[did]:
            g2_bad += 1
    report["G2"] = {"docs_checked": len(labeled), "docs_mismatched": g2_bad,
                    "chunk_len": chunk_len, "stride": stride}
    if g2_bad:
        raise GateFailure(f"G2 failed on {g2_bad} documents")
    log(f"   G2 passed on {len(labeled)} documents (chunk {chunk_len}, stride {stride})")

    # ---- stats + save -------------------------------------------------------------------------
    stats = {"docs": Counter(fold_of.values()),
             "chars": {f: int(sum(len(texts[d]) for d in texts if fold_of[d] == f))
                       for f in ("fold1", "fold2", "fold3")},
             "tokens": {f: int(sum(len(docs[d]["ids"]) for d in docs if fold_of[d] == f))
                        for f in ("fold1", "fold2", "fold3")},
             "max_doc_tokens": int(max(len(r["ids"]) for r in docs.values())),
             "entities_per_label": dict(Counter(a[2] for g in gold.values() for a in g))}
    ent_tok_len = []
    for did in labeled:
        run = 0
        for v in docs[did]["tags"].tolist() + [0]:
            if v > 0 and v % 2 == 1:                      # B- starts a new entity
                if run:
                    ent_tok_len.append(run)
                run = 1
            elif v > 0 and run:                           # I- continues it
                run += 1
            else:
                if run:
                    ent_tok_len.append(run)
                run = 0
    if ent_tok_len:
        q = np.percentile(ent_tok_len, [50, 90, 99, 100])
        stats["entity_token_length_p50_p90_p99_max"] = [float(x) for x in q]
    report["stats"] = stats
    log(f"   stats: {json.dumps(stats, default=str)[:800]}")

    with open(os.path.join(out_dir, "docs.pkl.tmp"), "wb") as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(os.path.join(out_dir, "docs.pkl.tmp"), os.path.join(out_dir, "docs.pkl"))
    meta = {"labels": labels, "tag_names": make_tag_names(labels), "split": split,
            "snap_rules": snap.to_json() if use_snap else {}, "tokenizer_md5": tok_md5,
            "tokenizer_path": tok_path, "eos_id": int(tok.eos_token_id),
            "pad_id": int(tok.pad_token_id), "vocab_size": len(tok),
            "raw_md5": report["raw_md5"], "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": 1}
    write_json(os.path.join(out_dir, "meta.json"), meta)
    report["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(out_dir, "data_report.json"), report)
    log(f"   data cache written to {out_dir} in {report['seconds']} s")
    return meta


def load_cache(data_dir):
    meta = read_json(os.path.join(data_dir, "meta.json"))
    if meta is None:
        raise FileNotFoundError(f"no data cache in {data_dir}")
    with open(os.path.join(data_dir, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    return meta, docs


def find_or_build_cache(data_dir, roots=None, log=print, **kw):
    """Reuse a cache in data_dir or under the input roots, else build it. A cache is reused only
    if it was built with the SAME tokenizer (bpe_model.json md5) as the attached Phase-1 models."""
    roots = roots or default_input_roots()
    try:
        _tok, _p, tok_md5 = load_tokenizer(roots)
    except FileNotFoundError:
        tok_md5 = None

    def ok(d):
        meta = read_json(os.path.join(d, "meta.json"))
        return (isinstance(meta, dict) and meta.get("version") == 1 and "tag_names" in meta
                and os.path.exists(os.path.join(d, "docs.pkl"))
                and (tok_md5 is None or meta.get("tokenizer_md5") == tok_md5))
    if ok(data_dir):
        return data_dir
    for m in find_files("meta.json", roots):
        d = os.path.dirname(m)
        if ok(d):
            log(f"   using the data cache attached at {d}")
            return d
    build_cache(data_dir, roots=roots, log=log, **kw)
    return data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--out", default="/kaggle/working/phase2_data")
    ap.add_argument("--tokenizer", default=None, help="explicit path to bpe_model.json")
    ap.add_argument("--allow-inclusive-ends", action="store_true")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    a = ap.parse_args()
    try:
        build_cache(a.out, tokenizer_path=a.tokenizer, allow_inclusive_ends=a.allow_inclusive_ends,
                    chunk_len=a.chunk_len, stride=a.stride)
    except GateFailure as e:
        print("\nSTOP AND ASK:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
