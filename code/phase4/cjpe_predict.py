"""
cjpe_predict.py -- batched inference, the dev/test output files, the submission zip and its
validation (sections 9.1, 9.3 R4, 11.1).

Every finished model writes:
  dev/dev_probs.csv     id,label,p1,pred           dev/dev_report.json (full analysis)
  dev/errors.csv        misclassified dev cases (FP / FN flagged, p1, n_tokens, snippet)
  submission/test_probs.csv  id,p1,pred            submission/prediction.json
  submission/submission.zip  (prediction.json at the zip root)
  submission/validation_report.json
"""
import csv
import json
import os
import zipfile
from typing import Dict, List

import numpy as np
import torch

from cjpe_common import write_json
from cjpe_eval import full_report
from cjpe_models import collate_docs, forward_batch
from validate_submission import validate


def batches_by_chunks(chunk_lists: List[List[np.ndarray]], max_chunks: int,
                      order=None) -> List[List[int]]:
    """Group document indices so that each group has <= max_chunks chunks (>= 1 doc)."""
    order = list(range(len(chunk_lists))) if order is None else list(order)
    out, cur, n = [], [], 0
    for i in order:
        c = len(chunk_lists[i])
        if cur and n + c > max_chunks:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += c
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def predict_logits(model, chunk_lists, device, amp_ctx, max_chunks: int = 32) -> np.ndarray:
    """(n_docs, 2) fp32 logits; documents sorted by length for less padding, then unsorted."""
    model.eval()
    n = len(chunk_lists)
    out = np.zeros((n, 2), np.float32)
    order = sorted(range(n), key=lambda i: (len(chunk_lists[i]),
                                            max(len(c) for c in chunk_lists[i])))
    mc = max_chunks
    groups = batches_by_chunks(chunk_lists, mc, order)
    gi = 0
    while gi < len(groups):
        g = groups[gi]
        try:
            b = collate_docs([chunk_lists[i] for i in g], device)
            with amp_ctx():
                logits, _ = forward_batch(model, b)
            out[g] = logits.float().cpu().numpy()
            gi += 1
        except RuntimeError as e:                      # OOM: smaller groups, same order
            if "out of memory" not in str(e).lower() or mc <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mc = max(1, mc // 2)
            rest = [i for gg in groups[gi:] for i in gg]
            groups = groups[:gi] + batches_by_chunks(chunk_lists, mc, rest)
    return out


def softmax1(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(1)).astype(np.float64)


def _write_rows(path: str, header: List[str], rows: List[List]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def write_dev_outputs(run_dir: str, ids, gold, p1, tau, n_tokens, texts_snip: Dict[str, str]
                      ) -> Dict:
    rep = full_report(ids, gold, p1, tau, n_tokens)
    pred = [int(x >= tau) for x in p1]
    _write_rows(os.path.join(run_dir, "dev", "dev_probs.csv"), ["id", "label", "p1", "pred"],
                [[i, g, f"{p:.6f}", q] for i, g, p, q in zip(ids, gold, p1, pred)])
    err = []
    for i, g, p, q, n in zip(ids, gold, p1, pred, n_tokens):
        if g != q:
            err.append([i, "FP" if q == 1 else "FN", g, q, f"{p:.6f}", int(n),
                        texts_snip.get(i, "")])
    if not err:                      # keep the file informative even for a perfect dev score
        err.append(["", "NONE", "", "", "", "", "no misclassified dev cases"])
    _write_rows(os.path.join(run_dir, "dev", "errors.csv"),
                ["id", "error_type", "label", "pred", "p1", "n_tokens", "snippet"], err)
    write_json(os.path.join(run_dir, "dev", "dev_report.json"), rep)
    return rep


def write_submission(run_dir: str, test_ids: List[str], p1, tau: float) -> Dict:
    pred = [int(x >= tau) for x in p1]
    sub = os.path.join(run_dir, "submission")
    os.makedirs(sub, exist_ok=True)
    _write_rows(os.path.join(sub, "test_probs.csv"), ["id", "p1", "pred"],
                [[i, f"{p:.6f}", q] for i, p, q in zip(test_ids, p1, pred)])
    items = [{"id": str(i), "pred": int(q)} for i, q in zip(test_ids, pred)]
    pj = os.path.join(sub, "prediction.json")
    with open(pj + ".tmp", "w", encoding="utf-8") as f:
        json.dump(items, f)
    os.replace(pj + ".tmp", pj)
    zp = os.path.join(sub, "submission.zip")
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    os.replace(zp + ".tmp", zp)
    rep = validate(zp, test_ids)
    rep["tau"] = tau
    for f in ("test_probs.csv",):
        with open(os.path.join(sub, f), encoding="utf-8") as fh:
            rep[f"{f}_rows"] = sum(1 for _ in fh) - 1
    for f in ("dev_probs.csv", "errors.csv"):
        p = os.path.join(run_dir, "dev", f)
        rep[f"{f}_rows"] = (sum(1 for _ in open(p, encoding="utf-8")) - 1) if os.path.exists(p) \
            else 0
    empty = [k for k in ("test_probs.csv_rows", "dev_probs.csv_rows", "errors.csv_rows")
             if not rep.get(k)]
    if empty:
        rep["ok"] = False
        rep["errors"].append(f"empty output files: {empty}")
    write_json(os.path.join(sub, "validation_report.json"), rep)
    return rep
