"""
validate_submission.py -- Phase-7 submission check (PDF 13.3.5).

submission.zip must hold exactly `prediction.json` at its root: ONE JSON object
{"<test query id>": ["<cand id>", ... 7 ids]} -- exactly 7 unique ids per query, every id from
the test candidate pool, a query never lists itself, one entry for every test query.

    python validate_submission.py submission.zip [--test-ids test_ids.json --pool pool_ids.json]
"""
import argparse
import json
import zipfile
from typing import Dict, List, Optional

K = 7


def validate(zip_path: str, test_ids: Optional[List[str]] = None,
             pool_ids: Optional[List[str]] = None) -> Dict:
    errors: List[str] = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if names != ["prediction.json"]:
                errors.append(f"zip must contain only prediction.json at its root, has {names}")
                if "prediction.json" not in names:
                    return {"ok": False, "errors": errors}
            obj = json.loads(z.read("prediction.json").decode("utf-8"))
    except Exception as e:
        return {"ok": False, "errors": [f"unreadable: {type(e).__name__}: {e}"]}
    if not isinstance(obj, dict):
        return {"ok": False, "errors": [f"prediction.json must be an object, got {type(obj).__name__}"]}
    pool = set(pool_ids) if pool_ids is not None else None
    for q, r in obj.items():
        if not isinstance(r, list) or len(r) != K:
            errors.append(f"{q}: needs a list of exactly {K} ids")
            continue
        if not all(isinstance(x, str) and x for x in r):
            errors.append(f"{q}: ids must be non-empty strings")
        if len(set(r)) != K:
            errors.append(f"{q}: duplicate ids")
        if q in r:
            errors.append(f"{q}: lists itself")
        if pool is not None and not set(r) <= pool:
            errors.append(f"{q}: {len(set(r) - pool)} ids not in the test candidate pool")
        if len(errors) > 20:
            break
    if test_ids is not None:
        miss = set(test_ids) - set(obj)
        extra = set(obj) - set(test_ids)
        if miss:
            errors.append(f"{len(miss)} test queries missing (e.g. {sorted(miss)[:3]})")
        if extra:
            errors.append(f"{len(extra)} unknown query ids (e.g. {sorted(extra)[:3]})")
    lists = {tuple(v) for v in obj.values() if isinstance(v, list)}
    return {"ok": not errors, "errors": errors[:20], "n": len(obj),
            "degenerate": len(obj) > 3 and len(lists) == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--test-ids")
    ap.add_argument("--pool")
    a = ap.parse_args()
    t = json.load(open(a.test_ids)) if a.test_ids else None
    p = json.load(open(a.pool)) if a.pool else None
    print(json.dumps(validate(a.zip, t, p), indent=1))


if __name__ == "__main__":
    main()
