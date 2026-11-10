"""
validate_submission.py -- block any invalid Phase-5 (LSI) submission (rule R4 of the brief).

Checks: the archive is a zip; `prediction.json` sits at its ROOT (no folder); it is a non-empty
JSON list of {"id": str, "pred": [int, ...]}; ids are exactly the test.json ids (no missing, no extra,
no duplicates); every pred is a non-empty list of unique ints (not bool / float / str) from the valid statute
ids; plus a degenerate check (every document gets the same label set).

    python validate_submission.py submission.zip --test-ids test_ids.json
"""
import argparse
import json
import sys
import zipfile
from typing import Dict, Iterable, Optional


def validate(zip_path: str, test_ids: Iterable[str], valid_ids: Optional[Iterable[int]] = None
             ) -> Dict:
    test_ids = [str(i) for i in test_ids]
    valid = set(int(x) for x in valid_ids) if valid_ids is not None else None
    rep = {"zip": zip_path, "ok": False, "errors": [], "degenerate": False}
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            rep["entries"] = names
            if "prediction.json" not in names:
                rep["errors"].append(f"prediction.json not at the zip root (entries: {names[:5]})")
                return rep
            if len(names) != 1:
                rep["errors"].append(f"extra entries in the zip: {names}")
            raw = z.read("prediction.json")
    except zipfile.BadZipFile as e:
        rep["errors"].append(f"not a zip file: {e}")
        return rep
    if not raw.strip():
        rep["errors"].append("prediction.json is empty")
        return rep
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:
        rep["errors"].append(f"prediction.json is not valid JSON: {e}")
        return rep
    if not isinstance(obj, list) or not obj:
        rep["errors"].append("prediction.json must be a non-empty JSON list")
        return rep
    seen, preds = set(), []
    for k, r in enumerate(obj):
        if not isinstance(r, dict) or set(r.keys()) != {"id", "pred"}:
            rep["errors"].append(f"item {k} must be exactly {{'id', 'pred'}}: {r!r}"[:200])
            break
        if not isinstance(r["id"], str):
            rep["errors"].append(f"item {k}: id must be a string")
            break
        p = r["pred"]
        if (not isinstance(p, list) or not p or any(type(x) is not int for x in p)
                or len(set(p)) != len(p)):
            rep["errors"].append(f"item {k}: pred must be a non-empty list of unique ints, "
                                 f"got {p!r}"[:200])
            break
        if valid is not None and not set(p) <= valid:
            rep["errors"].append(f"item {k}: invalid statute ids {sorted(set(p) - valid)[:5]}")
            break
        if r["id"] in seen:
            rep["errors"].append(f"duplicate id {r['id']!r}")
            break
        seen.add(r["id"])
        preds.append(p)
    want = set(test_ids)
    missing, extra = want - seen, seen - want
    if missing:
        rep["errors"].append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        rep["errors"].append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    rep["n"] = len(obj)
    rep["n_test"] = len(want)
    rep["pos_rate"] = (sum(len(x) for x in preds) / len(preds)) if preds else None
    rep["mean_labels"] = rep["pos_rate"]
    rep["degenerate"] = len(preds) > 1 and len({tuple(sorted(x)) for x in preds}) == 1
    rep["ok"] = not rep["errors"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--test-ids", required=True, help="JSON list of the official test ids")
    ap.add_argument("--valid-ids", default=None, help="JSON list of valid statute ids")
    a = ap.parse_args()
    with open(a.test_ids) as f:
        ids = json.load(f)
    valid = json.load(open(a.valid_ids)) if a.valid_ids else None
    rep = validate(a.zip, ids, valid)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["ok"] else 4)


if __name__ == "__main__":
    main()
