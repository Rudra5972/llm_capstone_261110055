"""
validate_submission.py -- block any invalid Phase-6 (summarisation) submission.

Checks: the archive is a zip whose only entry is `prediction.json` at its ROOT; it is a JSON list
of {"id": str, "summary": str} (PDF 12.3.5, list format); ids are exactly the test.json ids (no
missing, extra or duplicate); every summary is a non-empty string; plus a degenerate check
(every document got the same summary).

    python validate_submission.py submission.zip --test-ids test_ids.json
"""
import argparse
import json
import sys
import zipfile
from typing import Dict, Iterable, Optional


def validate(zip_path: str, test_ids: Iterable[str], valid_ids: Optional[Iterable] = None) -> Dict:
    test_ids = [str(i) for i in test_ids]
    rep = {"zip": zip_path, "ok": False, "errors": [], "degenerate": False}
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            rep["entries"] = names
            if names != ["prediction.json"]:
                rep["errors"].append(f"zip must hold exactly prediction.json at its root: {names[:5]}")
                return rep
            items = json.loads(z.read("prediction.json").decode("utf-8"))
    except Exception as e:
        rep["errors"].append(f"unreadable: {type(e).__name__}: {e}")
        return rep
    if not isinstance(items, list) or not items:
        rep["errors"].append("prediction.json must be a non-empty JSON list")
        return rep
    seen, bad = [], 0
    for it in items:
        if not (isinstance(it, dict) and isinstance(it.get("id"), str)
                and isinstance(it.get("summary"), str) and it["summary"].strip()):
            bad += 1
            continue
        seen.append(it["id"])
    if bad:
        rep["errors"].append(f"{bad} items are not {{'id': str, 'summary': non-empty str}}")
    if len(seen) != len(set(seen)):
        rep["errors"].append(f"{len(seen) - len(set(seen))} duplicate ids")
    miss, extra = set(test_ids) - set(seen), set(seen) - set(test_ids)
    if miss:
        rep["errors"].append(f"{len(miss)} test ids missing, e.g. {sorted(miss)[:3]}")
    if extra:
        rep["errors"].append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    summ = [it.get("summary", "") for it in items if isinstance(it, dict)]
    rep["n"] = len(items)
    rep["mean_words"] = sum(len(str(s).split()) for s in summ) / max(1, len(summ))
    rep["degenerate"] = len(summ) > 1 and len(set(summ)) == 1
    rep["ok"] = not rep["errors"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--test-ids", required=True, help="JSON list of test ids")
    a = ap.parse_args()
    with open(a.test_ids) as f:
        ids = json.load(f)
    r = validate(a.zip, ids)
    print(json.dumps({k: v for k, v in r.items() if k != "entries"}, indent=1))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
