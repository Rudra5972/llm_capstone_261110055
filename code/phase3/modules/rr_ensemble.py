"""
rr_ensemble.py -- E11 ensemble over the saved logits of every BEST model, and the per-domain
combination used by the E10 ablation (separate CL and IT models -> one submission).

    python rr_ensemble.py [--out /kaggle/working/phase3_out]

Members: every <exp>/BEST (E0..E9) found under --out or under an attached `phase3_out` folder,
with raw dev/test logits saved by rr_predict (no model is re-run). Logits are averaged after
log-softmax normalisation (= geometric mean of the probabilities; raw logits of different
techniques live on different scales). Candidates: uniform over all members, dev-F1-weighted
over all members, uniform over the top-3 by dev F1; the best on dev is kept. CAUTION: dev is
used both to pick the members/weights and to report, so the ensemble's dev score is optimistic.
The usual 2-fold held-out check decides whether post-hoc biases are applied.

Inputs : BEST/logits/{dev,test}_logits.npy + *_index.json, BEST/DONE.json, data cache
Outputs: E11_ensemble/ensemble/{DONE.json, dev/, submission/, logits/}, E11_ensemble/BEST/
"""
import argparse
import os
from typing import Dict, List, Optional

import numpy as np

from rr_common import (DATA_DIR, EXIT, EXPERIMENTS, OUT_MARKER, OUT_ROOT, default_input_roots,
                       find_files, read_json, write_json)
from rr_data import load_cache
from rr_eval import full_report, headline, labels_from_logits
from rr_predict import done_record, finalize_from_logits, load_logits


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def find_best_dirs(out_root: str, roots: List[str]) -> Dict[str, str]:
    """{exp_id: BEST dir} for E0..E9 (local output first, then attached inputs)."""
    found = {}
    for exp, spec in EXPERIMENTS.items():
        if spec["technique"] in ("ablation", "ensemble"):
            continue
        cands = [os.path.join(out_root, spec["folder"], "BEST")]
        for s in find_files("SELECTION.json", roots, max_hits=5000):
            parts = s.split(os.sep)
            if OUT_MARKER in parts and spec["folder"] in parts and parts[-2] == "BEST":
                cands.append(os.path.dirname(s))
        for c in cands:
            if os.path.exists(os.path.join(c, "logits", "dev_logits.npy")) and \
                    os.path.exists(os.path.join(c, "DONE.json")):
                found[exp] = c
                break
    return found


def average(members: List[Dict], weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    w = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=np.float64)
    w = w / w.sum()
    keys = members[0].keys()
    return {k: sum(wi * _log_softmax(m[k]) for wi, m in zip(w, members)).astype(np.float32)
            for k in keys}


def run_ensemble(out_root: str, roots: List[str], exp_dir: str, data_dir: str, log=print) -> int:
    meta, docs = load_cache(data_dir)
    best = find_best_dirs(out_root, roots)
    log(f"   ensemble candidates: {sorted(best)}")
    mem = []
    for exp, d in sorted(best.items()):
        dev, test = load_logits(os.path.join(d, "logits"), "dev"), \
            load_logits(os.path.join(d, "logits"), "test")
        done = read_json(os.path.join(d, "DONE.json"))
        if dev is None or test is None or set(test) != set(meta["split"]["test"]):
            log(f"   {exp}: incomplete logits, skipped")
            continue
        mem.append({"exp": exp, "dir": d, "dev": dev, "test": test, "f1": done["dev_macro_f1"],
                    "model": done.get("model"), "technique": done.get("technique")})
    if len(mem) < 2:
        log("   fewer than 2 BEST models with saved logits -- nothing to ensemble")
        return EXIT["missing_dependency"]
    dev_keys = sorted(mem[0]["dev"])
    gold = {k: docs[k]["labels"] for k in dev_keys}
    dom = {k: docs[k]["domain"] for k in dev_keys}
    order = sorted(range(len(mem)), key=lambda i: -mem[i]["f1"])
    cands = {"uniform_all": (list(range(len(mem))), None),
             "f1_weighted_all": (list(range(len(mem))), [m["f1"] for m in mem]),
             "uniform_top3": (order[:3], None)}
    scores = {}
    for name, (idx, w) in cands.items():
        dev = average([mem[i]["dev"] for i in idx], w)
        scores[name] = headline(full_report(gold, labels_from_logits(dev), dom, meta["labels"]))
        log(f"   {name:<16} members {[mem[i]['exp'] for i in idx]} -> dev pooled Macro-F1 "
            f"{scores[name]:.4f}")
    pick = max(scores, key=lambda n: (scores[n], n == "uniform_all"))
    idx, w = cands[pick]
    dev = average([mem[i]["dev"] for i in idx], w)
    test = average([mem[i]["test"] for i in idx], w)
    run_dir = os.path.join(exp_dir, "ensemble")
    fin = finalize_from_logits(dev, test, meta, docs, run_dir, log)
    cfg = {"exp": "E11", "job": "ensemble", "model": "+".join(mem[i]["model"] for i in idx),
           "technique": "ensemble", "variant": pick}
    done = done_record(cfg, fin, {"members": [{k: mem[i][k] for k in ("exp", "dir", "f1", "model",
                                                                     "technique")} for i in idx],
                                  "weights": w, "candidates": scores,
                                  "efficiency": {"trainable_params": 0},
                                  "note": "dev score is optimistic (dev used for selection)"})
    write_json(os.path.join(run_dir, "DONE.json"), done)
    from rr_select import select
    select(exp_dir, log=log)
    return EXIT["ok"] if fin["validation"] and fin["validation"]["ok"] else \
        EXIT["validation_failure"]


def combine_domains(src_dirs: List[str], run_dir: str, data_dir: str, job: Dict,
                    log=print) -> Dict:
    """E10 per-domain variant: CL-only and IT-only runs -> one pooled report + submission."""
    meta, docs = load_cache(data_dir)
    dev, test, eff = {}, {}, {"trainable_params": 0, "total_params": 0, "train_minutes": 0.0}
    for d in src_dirs:
        dv, ts = load_logits(os.path.join(d, "logits"), "dev"), \
            load_logits(os.path.join(d, "logits"), "test")
        if dv is None or ts is None:
            raise FileNotFoundError(f"no saved logits in {d}")
        dev.update(dv)
        test.update(ts)
        e = (read_json(os.path.join(d, "DONE.json")) or {}).get("efficiency") or {}
        for k in eff:
            eff[k] += e.get(k) or 0
    fin = finalize_from_logits(dev, test, meta, docs, run_dir, log)
    first = read_json(os.path.join(src_dirs[0], "logs", "run_config.json")) or {}
    cfg = {**{k: first.get(k) for k in ("exp", "model", "arch", "technique", "input_mode",
                                        "head", "attn_mode", "loss")},
           "job": os.path.basename(run_dir), "variant": job.get("variant", "per_domain"),
           "domains": ["CL", "IT"]}
    done = done_record(cfg, fin, {"sources": src_dirs, "efficiency": eff})
    write_json(os.path.join(run_dir, "DONE.json"), done)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()
    exp_dir = os.path.join(a.out, EXPERIMENTS["E11"]["folder"])
    raise SystemExit(run_ensemble(a.out, default_input_roots(), exp_dir, a.data_dir))


if __name__ == "__main__":
    main()
