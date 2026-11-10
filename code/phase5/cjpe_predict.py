"""
cjpe_predict.py -- batched inference, the dev/test output files, the submission zip and its
validation (Phase 5, multi-label).

Every finished model writes:
  dev/dev_probs.npz       ids, gold (uint8), probs (float16)      dev/dev_report.json
  dev/per_label_dev.csv   label, supports, P/R/F1, predicted count, tau
  dev/errors.csv          dev docs whose label set is wrong (missed / spurious ids, snippet)
  dev/dev_metrics.json    scores of the chosen rule (+ global, 0.5)
  submission/test_probs.npz, submission/prediction.json, submission/submission.zip,
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
from lsi_ml import apply_rule, full_report, rule_taus
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
    """(n_docs, L) fp32 logits; documents sorted by length for less padding, then unsorted."""
    model.eval()
    n = len(chunk_lists)
    out = np.zeros((n, model.classifier.out.out_features), np.float32)
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


def _strip(rep: Dict) -> Dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def write_dev_outputs(run_dir: str, ids, G, Pr, rule, n_tokens, texts_snip: Dict[str, str],
                      label_ids, train_support, n_dev_only_pos: int = 0) -> Dict:
    rep = full_report(G, Pr, rule, n_tokens, label_ids, train_support, n_dev_only_pos)
    P = rep["_P"]
    dd = os.path.join(run_dir, "dev")
    os.makedirs(dd, exist_ok=True)
    np.savez_compressed(os.path.join(dd, "dev_probs.npz"), ids=np.asarray(ids),
                        gold=G.astype(np.uint8), probs=Pr.astype(np.float16),
                        label_ids=np.asarray(label_ids))
    pl = rep["_per_label"]
    _write_rows(os.path.join(dd, "per_label_dev.csv"), list(pl[0].keys()),
                [list(r.values()) for r in pl])
    lab = np.asarray(label_ids)
    err = []
    for k, i in enumerate(ids):
        g, p = G[k].astype(bool), P[k]
        if (g != p).any():
            err.append([i, " ".join(map(str, lab[g & ~p])), " ".join(map(str, lab[p & ~g])),
                        int(g.sum()), int(p.sum()), int(n_tokens[k]), texts_snip.get(i, "")])
    if not err:
        err.append(["", "", "", "", "", "", "no dev document with a wrong label set"])
    _write_rows(os.path.join(dd, "errors.csv"),
                ["id", "missed", "spurious", "n_gold", "n_pred", "n_tokens", "snippet"], err)
    out = _strip(rep)
    write_json(os.path.join(dd, "dev_report.json"), out)
    write_json(os.path.join(dd, "dev_metrics.json"),
               {"rule": out["rule"], "scores": out["scores"], "scores_global": out["scores_global"],
                "scores_at_0p5": out["scores_at_0p5"]})
    return rep


def write_submission(run_dir: str, test_ids: List[str], Pr, rule: Dict, label_ids) -> Dict:
    L = Pr.shape[1]
    P = apply_rule(Pr, rule_taus(rule, L), int(rule["kmax"]))
    lab = np.asarray(label_ids, dtype=np.int64)
    sub = os.path.join(run_dir, "submission")
    os.makedirs(sub, exist_ok=True)
    np.savez_compressed(os.path.join(sub, "test_probs.npz"), ids=np.asarray(test_ids),
                        probs=Pr.astype(np.float16), label_ids=lab)
    items = [{"id": str(i), "pred": sorted(int(x) for x in set(lab[P[k]].tolist()))}
             for k, i in enumerate(test_ids)]
    pj = os.path.join(sub, "prediction.json")
    with open(pj + ".tmp", "w", encoding="utf-8") as f:
        json.dump(items, f)
    os.replace(pj + ".tmp", pj)
    zp = os.path.join(sub, "submission.zip")
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    os.replace(zp + ".tmp", zp)
    rep = validate(zp, test_ids, [int(x) for x in lab])
    rep["rule"] = rule.get("name")
    for f in ("dev_probs.npz", "per_label_dev.csv", "errors.csv"):
        p = os.path.join(run_dir, "dev", f)
        rep[f"{f}_bytes"] = os.path.getsize(p) if os.path.exists(p) else 0
    rep["test_probs.npz_bytes"] = os.path.getsize(os.path.join(sub, "test_probs.npz"))
    empty = [k for k in rep if k.endswith("_bytes") and not rep[k]]
    if empty:
        rep["ok"] = False
        rep["errors"].append(f"empty output files: {empty}")
    write_json(os.path.join(sub, "validation_report.json"), rep)
    return rep
