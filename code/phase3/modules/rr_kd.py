"""
rr_kd.py -- knowledge distillation for Phase 3 (experiment E6).

Teacher: by default SELF-CONTAINED: the E6 notebook first trains a full fine-tuned teacher per
Phase-1 model (same recipe as the E0 reference, job `teacher_<M>`), then distils it.
`--teacher-dir` instead loads existing teachers: either a folder with per-model run folders
(<dir>/M?_*/checkpoints/best.pt, e.g. phase3_out/E0_full_ft) -> the student of model M learns
from teacher M; or one BEST / run folder (best.pt + run_config.json) -> every student learns
from that teacher. Missing teacher -> exit code 2 (missing dependency).

Student: the same Phase-1 model pruned to every second layer (M1/M4: layers 0,2,..,12;
M2/M3: 0,2,..,10), same width, initialised from those layers' Phase-1 weights, all parameters
+ a fresh head trained. Asserted: student params < teacher params (both printed). A pruned
student is a Phase-3 artifact, so the 50M lower bound of the brief does not apply; its size is
reported honestly.

Loss (Hinton, Vinyals & Dean (2015), "Distilling the Knowledge in a Neural Network",
arXiv:1503.02531):
    L = alpha * CE(student, gold) + (1 - alpha) * T^2 * KL(teacher_T || student_T)
alpha 0.5, T 2.0. Optional (off by default, `kd_hidden_weight` > 0): MSE between the student's
and the teacher's sentence vectors (through a learned projection when the widths differ).
Teacher outputs are computed once per training window and cached on disk (fp16).

Inputs : teacher run(s), student run_config
Outputs: teacher_cache.pt, KD loss, the teacher/student comparison (rr_predict writes it)
"""
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rr_common import MODELS, read_json
from rr_models import rebuild_trained_model


def resolve_teacher(teacher_dir: str, mid: str) -> Optional[Tuple[str, str]]:
    """(run_config.json, best.pt) of the teacher for model `mid`, or None."""
    if not teacher_dir or not os.path.isdir(teacher_dir):
        return None
    for cfg_rel, pt_rel in (("logs/run_config.json", "checkpoints/best.pt"),
                            ("run_config.json", "best.pt")):
        sub = os.path.join(teacher_dir, MODELS[mid]["folder"])
        for base in (sub, teacher_dir):
            c, p = os.path.join(base, cfg_rel), os.path.join(base, pt_rel)
            if os.path.exists(c) and os.path.exists(p):
                return c, p
    return None


def load_teacher(cfg: Dict, device, phase1_dirs: Dict[str, str]) -> Tuple[nn.Module, Dict]:
    found = resolve_teacher(cfg["teacher_dir"], cfg["model"])
    if not found:
        raise FileNotFoundError(f"KD teacher for {cfg['model']} not found in {cfg['teacher_dir']}")
    tcfg = read_json(found[0])
    p1 = dict(phase1_dirs)
    if tcfg.get("phase1_dir"):
        p1.setdefault(tcfg["model"], tcfg["phase1_dir"])
    teacher = rebuild_trained_model(tcfg, found[1], p1, device)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval(), tcfg


def attach_projection(model: nn.Module, cfg: Dict) -> None:
    """Learned projection student-width -> teacher-width (only when the widths differ)."""
    d_s = model.trunk.hidden_size
    d_t = int(cfg.get("teacher_hidden", d_s))
    if d_t != d_s and model.kd_proj is None:
        model.kd_proj = nn.Linear(d_s, d_t)


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, labels: torch.Tensor, hard_fn,
            alpha: float = 0.5, T: float = 2.0, s_vec: Optional[torch.Tensor] = None,
            t_vec: Optional[torch.Tensor] = None, hidden_w: float = 0.0,
            proj: Optional[nn.Module] = None) -> torch.Tensor:
    """s_logits / t_logits (N, C) for valid sentences, labels (N,)."""
    s, t = s_logits.float(), t_logits.float()
    hard = hard_fn(s, labels)
    soft = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True,
                    reduction="batchmean") * (T * T)
    loss = alpha * hard + (1 - alpha) * soft
    if hidden_w > 0 and s_vec is not None and t_vec is not None:
        sv = proj(s_vec.float()) if proj is not None else s_vec.float()
        loss = loss + hidden_w * F.mse_loss(sv, t_vec.float())
    return loss


@torch.no_grad()
def build_teacher_cache(teacher: nn.Module, windows, cfg: Dict, meta: Dict, device, amp_ctx,
                        path: str, with_vectors: bool = False, log=print) -> Dict:
    """Teacher logits (and optionally sentence vectors) for every training window, fp16."""
    from rr_predict import window_outputs
    if os.path.exists(path):
        cache = torch.load(path, map_location="cpu")
        if cache.get("n_windows") == len(windows) and (cache.get("vectors") is not None
                                                       or not with_vectors):
            log(f"   [kd] teacher cache reused: {path}")
            return cache
    logits, vecs = window_outputs(teacher, windows, meta, device, amp_ctx,
                                  int(cfg.get("eval_batch", 16)), return_vectors=with_vectors)
    cache = {"n_windows": len(windows), "logits": [l.half() for l in logits],
             "vectors": [v.half() for v in vecs] if with_vectors else None}
    torch.save(cache, path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"   [kd] teacher outputs cached for {len(windows)} windows -> {path}")
    return cache
