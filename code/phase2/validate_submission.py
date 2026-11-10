"""
validate_submission.py -- block a bad Codabench upload before it costs a submission slot.

    python validate_submission.py PATH/submission.zip [--fold3 fold3-text.json] [--labels labels.txt]

Checks (doc sections 6.3, 6.4, 19):
  * the file is a zip whose ONLY member is `prediction.json`, at the root (no folder)
  * the keys are exactly the fold3-text.json ids (no missing, no extra), as strings
  * every span is [start, end, label] (an optional 4th string field is tolerated) with integer
    0 <= start < end <= len(text) and a label from labels.txt
  * spans in a document do not overlap
Without --fold3, the fold3 texts are taken from the default data cache.
"""
import argparse
import json
import sys
import zipfile


def validate(zip_path, texts, labels):
    errors, warnings = [], []
    n_spans = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if names != ["prediction.json"]:
                errors.append(f"zip members must be exactly ['prediction.json'], got {names}")
            if "prediction.json" not in names:
                return {"ok": False, "errors": errors, "warnings": warnings, "n_spans": 0}
            pred = json.loads(z.read("prediction.json").decode("utf-8"))
    except zipfile.BadZipFile as e:
        return {"ok": False, "errors": [f"not a zip file: {e}"], "warnings": [], "n_spans": 0}
    if not isinstance(pred, dict):
        errors.append("prediction.json must be a JSON object {doc_id: [...]}")
        return {"ok": False, "errors": errors, "warnings": warnings, "n_spans": 0}
    want, got = set(map(str, texts)), set(pred)
    if want - got:
        errors.append(f"{len(want - got)} test ids missing, e.g. {sorted(want - got)[:5]}")
    if got - want:
        errors.append(f"{len(got - want)} unknown ids, e.g. {sorted(got - want)[:5]}")
    labset = set(labels)
    for did, spans in pred.items():
        if did not in texts:
            continue
        if not isinstance(spans, list):
            errors.append(f"{did}: value must be a list")
            continue
        n = len(texts[did])
        prev_end = -1
        for sp in sorted(spans, key=lambda x: (x[0], x[1]) if isinstance(x, list) and len(x) >= 2 else (0, 0)):
            n_spans += 1
            if not (isinstance(sp, list) and len(sp) in (3, 4)):
                errors.append(f"{did}: span must be [start, end, label]: {sp}")
                continue
            s, e, lab = sp[0], sp[1], sp[2]
            if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool):
                errors.append(f"{did}: offsets must be integers: {sp}")
                continue
            if not (0 <= s < e <= n):
                errors.append(f"{did}: bad offsets {sp} for a text of {n} chars")
            if lab not in labset:
                errors.append(f"{did}: unknown label {lab!r}")
            if len(sp) == 4 and not isinstance(sp[3], str):
                errors.append(f"{did}: the optional 4th field must be a string")
            if s < prev_end:
                errors.append(f"{did}: overlapping spans near {sp}")
            prev_end = max(prev_end, e)
    if n_spans == 0:
        warnings.append("no spans at all -- almost certainly a broken run")
    return {"ok": not errors, "errors": errors[:50], "warnings": warnings, "n_spans": n_spans,
            "n_docs": len(pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--fold3", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--data-dir", default="/kaggle/working/phase2_data")
    a = ap.parse_args()
    if a.fold3:
        from ner_data import load_labels, load_texts
        texts = load_texts(a.fold3)
        labels = load_labels(a.labels) if a.labels else None
    else:
        from ner_data import load_cache
        meta, docs = load_cache(a.data_dir)
        texts = {docs[d]["orig_id"]: docs[d]["text"] for d in meta["split"]["test"]}
        labels = meta["labels"]
    if labels is None:
        from ner_data import EXPECTED_LABELS
        labels = EXPECTED_LABELS
    r = validate(a.zip, texts, labels)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
