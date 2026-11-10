"""
validate_submission.py -- block any invalid Phase-4 submission (rule R4 of the brief).

Checks: the archive is a zip; `prediction.json` sits at its ROOT (no folder); it is a non-empty
JSON list of {"id": str, "pred": int}; ids are exactly the test.json ids (no missing, no extra,
no duplicates); every pred is an int (not bool / float / str) in {0, 1}; plus the class check
(all predictions identical -> "degenerate", reported separately because it is valid for
Codabench but useless for Macro-F1).

    python validate_submission.py submission.zip --test-ids test_ids.json
"""
import argparse
import json
import sys
import zipfile
from typing import Dict, Iterable


def validate(zip_path: str, test_ids: Iterable[str]) -> Dict:
    test_ids = [str(i) for i in test_ids]
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
        if type(p) is not int or p not in (0, 1):
            rep["errors"].append(f"item {k}: pred must be the integer 0 or 1, got {p!r}")
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
    rep["pos_rate"] = (sum(preds) / len(preds)) if preds else None
    rep["degenerate"] = bool(preds) and len(set(preds)) == 1
    rep["ok"] = not rep["errors"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--test-ids", required=True, help="JSON list of the official test ids")
    a = ap.parse_args()
    with open(a.test_ids) as f:
        ids = json.load(f)
    rep = validate(a.zip, ids)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["ok"] else 4)


if __name__ == "__main__":
    main()
