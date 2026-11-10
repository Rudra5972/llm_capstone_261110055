"""
cjpe_kd.py -- knowledge distillation for E6 (section 7.2).

Teacher : full fine-tuning of TEACHER_MODEL (default M4) trained inside E6 (job
          `teacher_<model>`), or a finished E1 run of that model found in the attached inputs
          whose fingerprint (data, tokenizer, representation) matches.
          After training, the teacher's logits for every train / dev / test document are cached
          once in `teacher_logits.npz` (eval mode, no dropout, no_grad).
Students: the four Phase-1 models pruned to every second layer (Trunk(layer_ids=...)), full
          fine-tuning, same `hier` representation, so the logits align per DOCUMENT.
Loss    : L = a * BCE(student, y) + (1 - a) * T^2 * BCE(s/T, sigmoid(t/T))  (multi-label),
          a = 0.5, T = 2 (Hinton et al., 2015, "Distilling the Knowledge in a Neural Network").
"""
import glob
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cjpe_common import read_json


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, labels: torch.Tensor,
            alpha: float = 0.5, T: float = 2.0, weight: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
    """Phase 5 (multi-label): a * BCE(student, gold) + (1-a) * T^2 * soft-target BCE against
    sigmoid(teacher / T); `weight` = per-label pos_weight."""
    from lsi_ml import kd_loss_ml
    return kd_loss_ml(s_logits, t_logits, labels, alpha, T, weight)


def save_teacher_logits(path: str, logits: Dict[str, Dict[str, np.ndarray]]) -> None:
    """logits[split] = {"ids": array(str), "logits": (n, 2) float32}."""
    flat = {}
    for s, d in logits.items():
        flat[f"{s}_ids"] = np.asarray(d["ids"])
        flat[f"{s}_logits"] = np.asarray(d["logits"], np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_teacher_logits(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        if k.endswith("_ids"):
            s = k[:-4]
            out[s] = {"ids": z[k], "logits": z[f"{s}_logits"]}
    return out


def teacher_map(path: str, split: str = "train") -> Dict[str, np.ndarray]:
    d = load_teacher_logits(path)[split]
    return {str(i): l for i, l in zip(d["ids"], d["logits"])}


def find_reusable_teacher(input_roots, model_folder: str, fingerprint: Dict) -> Optional[str]:
    """A finished E1 run of the teacher model with an identical fingerprint, or None."""
    for r in input_roots:
        if not r or not os.path.isdir(r):
            continue
        pat = os.path.join(r, "**", "phase5_out", "E1_full_ft", model_folder, "DONE.json")
        for done in glob.glob(pat, recursive=True):
            run = os.path.dirname(done)
            rc = read_json(os.path.join(run, "logs", "run_config.json"), {}) or {}
            if not os.path.exists(os.path.join(run, "checkpoints", "best.pt")):
                continue
            if all(rc.get(k) == v for k, v in fingerprint.items()):
                return run
    return None
