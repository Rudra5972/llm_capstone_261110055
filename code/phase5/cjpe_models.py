"""
cjpe_models.py -- the document classifier (section 6 of the brief).

    chunks of one document --Phase-1 Trunk (shared, reused submodules)--> hidden states
    -> masked mean over each chunk's tokens (fp32)            = chunk vectors c_1..c_n
    -> learned attention pooling  a_i = softmax_i(w . tanh(W c_i)) over the REAL chunks
    -> dropout 0.1 -> Linear(H, 2)                              = 2 logits (fp32)

All new parameters live in `model.classifier` so that ner_peft.apply_technique (which trains
every `classifier.*` parameter as "the head") works unchanged for every technique.
Padding chunks never reach the trunk: only real chunks are encoded, then scattered into a
(B, n_max, H) grid whose missing slots get exactly zero attention weight.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_models import Trunk, count_params, load_phase1_lm, torch_load

PAD_ID = 1


class DocHead(nn.Module):
    def __init__(self, d: int, n_classes: int = 2, dropout: float = 0.1, pool_dim: int = 256):
        super().__init__()
        self.att_proj = nn.Linear(d, pool_dim)
        self.att_vec = nn.Linear(pool_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d, n_classes)
        for lin in (self.att_proj, self.att_vec, self.out):
            nn.init.normal_(lin.weight, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def attention(self, c: torch.Tensor, cmask: torch.Tensor) -> torch.Tensor:
        """c (B, n, H) fp32, cmask (B, n) bool -> weights (B, n); masked slots exactly 0."""
        s = self.att_vec(torch.tanh(self.att_proj(c))).squeeze(-1)
        s = s.masked_fill(~cmask, float("-inf"))
        w = torch.softmax(s, dim=-1)
        return torch.where(cmask, w, torch.zeros_like(w))

    def forward(self, c, cmask):
        w = self.attention(c, cmask)
        doc = (w.unsqueeze(-1) * c).sum(1)
        return self.out(self.dropout(doc)), w


class DocClassifier(nn.Module):
    def __init__(self, trunk: Trunk, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.trunk = trunk
        self.classifier = DocHead(trunk.hidden_size, n_classes, dropout)

    def chunk_vectors(self, input_ids, attention_mask):
        h = self.trunk(input_ids, attention_mask)                      # (C, T, H)
        m = attention_mask.unsqueeze(-1).float()
        return (h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)        # (C, H) fp32

    def forward(self, input_ids, attention_mask, doc_index, slot, n_docs: int, n_slots: int):
        v = self.chunk_vectors(input_ids, attention_mask)
        H = v.shape[-1]
        grid = torch.zeros(n_docs, n_slots, H, device=v.device, dtype=v.dtype)
        cmask = torch.zeros(n_docs, n_slots, device=v.device, dtype=torch.bool)
        grid[doc_index, slot] = v
        cmask[doc_index, slot] = True
        # the head runs in fp32 even under autocast (section 6: logits and loss in fp32)
        with torch.autocast(device_type=v.device.type, enabled=False):
            logits, w = self.classifier(grid.float(), cmask)
        return logits.float(), w


def collate_docs(chunk_lists: List[List[np.ndarray]], device=None) -> Dict:
    """Flatten the chunks of several documents into one right-padded (C, T) batch."""
    flat, doc_index, slot = [], [], []
    for d, chunks in enumerate(chunk_lists):
        for s, c in enumerate(chunks):
            flat.append(c)
            doc_index.append(d)
            slot.append(s)
    T = max(len(c) for c in flat)
    assert T <= 1024, f"sequence of {T} tokens > 1024"
    ids = np.full((len(flat), T), PAD_ID, np.int64)
    mask = np.zeros((len(flat), T), np.int64)
    for i, c in enumerate(flat):
        ids[i, :len(c)] = c
        mask[i, :len(c)] = 1
    b = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
         "doc_index": torch.tensor(doc_index, dtype=torch.long),
         "slot": torch.tensor(slot, dtype=torch.long),
         "n_docs": len(chunk_lists), "n_slots": max(len(c) for c in chunk_lists),
         "n_tokens": int(mask.sum())}
    if device is not None:
        for k in ("input_ids", "attention_mask", "doc_index", "slot"):
            b[k] = b[k].to(device, non_blocking=True)
    return b


def forward_batch(model: DocClassifier, b: Dict):
    return model(b["input_ids"], b["attention_mask"], b["doc_index"], b["slot"], b["n_docs"],
                 b["n_slots"])


def build_doc_model(phase1_dir: str, attn_mode: str = "bidir", layer_ids=None,
                    dropout: float = 0.1, n_labels: int = 2, bias_init=None):
    """Phase 5: Linear(H, n_labels) multi-label head; bias initialised to the label log-odds
    prior (rare labels start near 0 probability: stable BCE)."""
    lm, cfg = load_phase1_lm(phase1_dir)
    trunk = Trunk(lm, attn_mode, layer_ids)
    del lm
    model = DocClassifier(trunk, int(n_labels), dropout)
    if bias_init is not None:
        with torch.no_grad():
            model.classifier.out.bias.copy_(torch.as_tensor(np.asarray(bias_init, np.float32)))
    return model, cfg


def model_size(model: nn.Module) -> Dict[str, float]:
    tot = count_params(model)
    tr = count_params(model, trainable_only=True)
    return {"total_params": tot, "trainable_params": tr,
            "trainable_pct": 100.0 * tr / max(1, tot)}


def device_report(model: nn.Module) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in list(model.parameters()) + list(model.buffers()):
        out[str(p.device)] = out.get(str(p.device), 0) + 1
    return out


def assert_one_device(model: nn.Module, device) -> None:
    """Lesson L1: after .to(device) EVERY parameter and buffer must be on that device."""
    want = torch.device(device)
    bad = [n for n, p in list(model.named_parameters()) + list(model.named_buffers())
           if p.device.type != want.type or (want.index is not None and p.device.index not in
                                             (None, want.index))]
    if bad:
        raise AssertionError(f"{len(bad)} tensors not on {want}: {bad[:5]}")


# ---------------------------------------------------------------------------------------------
# checkpoints of trained models
# ---------------------------------------------------------------------------------------------
STATE_BUFFER_TAGS = (".ada_mask", ".ada_ipt_", ".ada_unc_")


def trainable_state_dict(model: nn.Module, half: bool = False) -> Dict[str, torch.Tensor]:
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    for n, b in model.named_buffers():
        if any(t in n for t in STATE_BUFFER_TAGS):
            sd[n] = b.detach().cpu()
    if half:
        sd = {n: (t.half() if t.is_floating_point() and ".ada_mask" not in n else t)
              for n, t in sd.items()}
    return sd


def load_trainable_state(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    unknown = [n for n in sd if n not in params and n not in bufs]
    if unknown:
        raise KeyError(f"checkpoint has unknown tensors: {unknown[:5]}")
    with torch.no_grad():
        for n, t in sd.items():
            dst = params.get(n, bufs.get(n))
            dst.copy_(t.to(dst.dtype))
    absent = {n for n, p in params.items() if p.requires_grad} - set(sd)
    if absent:
        raise KeyError(f"checkpoint lacks trainable parameters: {sorted(absent)[:5]}")


def rebuild_trained(run_config: Dict, best_pt: str, phase1_dir: Optional[str] = None,
                    device="cpu"):
    """Recreate a finished run's module structure and load its best.pt (teacher reuse, E8)."""
    from cjpe_peft import apply_technique
    p1 = phase1_dir or run_config["phase1_dir"]
    model, _ = build_doc_model(p1, run_config["attn_mode"], run_config.get("layer_ids"),
                               n_labels=len(run_config["label_ids"]))
    tech = "full_ft" if run_config["technique"] in ("kd", "student_hard") else \
        run_config["technique"]
    apply_technique(model, tech, run_config, log=lambda *a: None)
    load_trainable_state(model, torch_load(best_pt))
    return model.to(device).eval()
