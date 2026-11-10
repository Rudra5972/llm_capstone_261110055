"""
cjpe_peft.py -- fine-tuning techniques for the document classifier (pure PyTorch, no `peft`).

  full_ft / kd / student_hard   every trunk parameter + head       (ner_peft, Phase 2)
  lorma      LoRMA+ y = (I + s*B*A) W0 x, rank 16, alpha 32       (ner_peft, Phase 2; compulsory)
             Bihany, Patel & Modi (2025), arXiv:2506.07621
  lora       LoRA r=16, alpha 32                                  (ner_peft; extra E7)
  adapters   Houlsby bottleneck 64 after attention-out and MLP-out + all norms (ner_peft)
  partial_ft top-6 blocks + final norm + head, layer-wise LR decay 0.85 (ner_peft)
  bitfit     bias terms only (M1/M4)                              (ner_peft; extra E7)
  adalora    dW = P diag(lambda) Q with an orthogonality regulariser, sensitivity importance
             (EMA + uncertainty) and a cubic budget schedule 16 -> 8 average rank.
             Zhang et al. (2023), ICLR, arXiv:2303.10512. Code copied from Phase 3 rr_peft.py
             (AdaLoRALinear + AdaLoRA), adapted only in where the head lives (`classifier.*`).

The head (`classifier.*`: attention pooling + output layer) is always trainable.

Lesson L1 (device-order bug of the Phase-3 pilot): the reused ner_peft modules create their
parameters on the default (CPU) device. `apply_technique` therefore moves the whole model to the
device of the frozen base (`trunk.embed.weight.device`) after inserting the modules, and then
asserts that every parameter and buffer sits on that one device, so the result is correct in
both orders ("apply then .to(device)" in production, ".to(device) then apply" in checks).
"""
import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cjpe_common import DEFAULT_LR, HEAD_LR
from cjpe_models import assert_one_device, model_size
from ner_peft import (FUSED_QKV, TARGETS, LoRALinear, LoRMALinear, AdapterWrapped, _get,
                      _groups, _set)
from ner_peft import apply_technique as phase2_apply

PHASE2 = ("full_ft", "kd", "student_hard", "lorma", "lora", "adapters", "partial_ft", "bitfit")
ALL_TECHNIQUES = PHASE2 + ("adalora",)


# =============================================================================================
# AdaLoRA (copied from Phase 3 rr_peft.py)
# =============================================================================================
class AdaLoRALinear(nn.Module):
    """y = W0 x + b + (alpha/r) * P diag(lambda * mask) Q dropout(x); lambda = 0 at init."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.r, self.scale = r, alpha / r
        self.ada_P = nn.Parameter(torch.randn(base.out_features, r, device=dev) * 0.02)
        self.ada_E = nn.Parameter(torch.zeros(r, device=dev))
        self.ada_Q = nn.Parameter(torch.randn(r, base.in_features, device=dev) * 0.02)
        self.register_buffer("ada_mask", torch.ones(r, device=dev))
        for n in ("P", "E", "Q"):
            self.register_buffer(f"ada_ipt_{n}", torch.zeros_like(getattr(self, f"ada_{n}")))
            self.register_buffer(f"ada_unc_{n}", torch.zeros_like(getattr(self, f"ada_{n}")))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.base(x)
        lam = (self.ada_E * self.ada_mask).to(x.dtype)
        u = F.linear(self.drop(x), self.ada_Q.to(x.dtype)) * lam
        return y + self.scale * F.linear(u, self.ada_P.to(x.dtype))

    @torch.no_grad()
    def merged_weight(self):
        d = self.ada_P @ torch.diag(self.ada_E * self.ada_mask) @ self.ada_Q
        return self.base.weight + self.scale * d.to(self.base.weight.dtype)

    def orth_reg(self) -> torch.Tensor:
        P, Q = self.ada_P.float(), self.ada_Q.float()
        I = torch.eye(self.r, device=P.device)
        return ((P.t() @ P - I) ** 2).sum() + ((Q @ Q.t() - I) ** 2).sum()


class Technique:
    """No-op hooks (Phase-2 techniques)."""
    name = "base"

    def extra_loss(self, model) -> Optional[torch.Tensor]:
        return None

    def after_backward(self, model, step: int, total: int) -> None:
        return None

    def selectable(self, step: int, total: int) -> bool:
        return True

    def force_final(self, model) -> None:
        return None

    def final_report(self, model) -> Dict:
        return {}


class AdaLoRA(Technique):
    name = "adalora"

    def setup(self, model, cfg, log=print):
        trunk = model.trunk
        self.r = int(cfg.get("adalora_init_r", 16))
        self.target_r = int(cfg.get("adalora_target_r", 8))
        alpha = float(cfg.get("adalora_alpha", 32))
        self.tinit_frac = float(cfg.get("adalora_tinit_frac", 0.1))
        self.tfinal_frac = float(cfg.get("adalora_tfinal_frac", 0.2))
        self.delta_t = int(cfg.get("adalora_delta_t", 10))
        self.beta1 = float(cfg.get("adalora_beta1", 0.85))
        self.beta2 = float(cfg.get("adalora_beta2", 0.85))
        self.orth_w = float(cfg.get("adalora_orth", 0.1))
        for p in model.parameters():
            p.requires_grad = False
        for blk in trunk.blocks:
            for path in TARGETS[trunk.kind]:
                _set(blk, path, AdaLoRALinear(_get(blk, path), self.r, alpha,
                                              float(cfg.get("lora_dropout", 0.05))))
        self.modules_ = [(n, m) for n, m in model.named_modules() if isinstance(m, AdaLoRALinear)]
        self.b0 = self.r * len(self.modules_)
        self.bT = self.target_r * len(self.modules_)
        named = [(n, p) for n, p in model.named_parameters() if ".ada_" in n]
        for _, p in named:
            p.requires_grad = True
        log(f"   [adalora] {len(self.modules_)} matrices, rank {self.r} -> target "
            f"{self.target_r} (budget {self.b0} -> {self.bT} singular values)")
        return _groups(named, float(cfg["lr"]), float(cfg.get("weight_decay", 0.01)), "adalora")

    def extra_loss(self, model):
        if not self.modules_ or self.orth_w <= 0:
            return None
        return self.orth_w * sum(m.orth_reg() for _, m in self.modules_) / len(self.modules_)

    def phases(self, total: int):
        tf_start = max(0, min(total - int(self.tfinal_frac * total), total - 1))
        return min(int(self.tinit_frac * total), tf_start), tf_start

    def budget(self, step: int, total: int) -> int:
        ti, tf_start = self.phases(total)
        if step < ti:
            return self.b0
        if step >= tf_start:
            return self.bT
        mul = 1.0 - (step - ti) / max(1, tf_start - ti)
        return int(self.bT + (self.b0 - self.bT) * mul ** 3)

    def selectable(self, step, total):
        return step > self.phases(total)[1]

    @torch.no_grad()
    def _update_importance(self):
        for _, m in self.modules_:
            for n in ("P", "E", "Q"):
                p = getattr(m, f"ada_{n}")
                if p.grad is None:
                    continue
                ipt = (p * p.grad).abs()
                ema = getattr(m, f"ada_ipt_{n}")
                unc = getattr(m, f"ada_unc_{n}")
                ema.mul_(self.beta1).add_((1 - self.beta1) * ipt)
                unc.mul_(self.beta2).add_((1 - self.beta2) * (ipt - ema).abs())

    @torch.no_grad()
    def _scores(self) -> List[torch.Tensor]:
        out = []
        for _, m in self.modules_:
            s = lambda n: getattr(m, f"ada_ipt_{n}") * getattr(m, f"ada_unc_{n}")  # noqa: E731
            out.append(s("E") + s("P").mean(0) + s("Q").mean(1))
        return out

    @torch.no_grad()
    def _prune_to(self, b: int):
        scores = self._scores()
        flat = torch.cat(scores)
        if b >= flat.numel():
            return
        thr = torch.kthvalue(flat.float().cpu(), flat.numel() - b).values.item()
        for (_, m), s in zip(self.modules_, scores):
            keep = (s > thr).float()
            m.ada_mask.copy_(keep)
            m.ada_E.mul_(keep)

    @torch.no_grad()
    def after_backward(self, model, step, total):
        ok = all(p.grad is None or torch.isfinite(p.grad).all()
                 for _, m in self.modules_ for p in (m.ada_P, m.ada_E, m.ada_Q))
        if not ok:
            return
        self._update_importance()
        ti, tf_start = self.phases(total)
        if step < ti or step > tf_start:
            return
        if step != tf_start and (step - ti) % self.delta_t != 0:
            return
        self._prune_to(self.budget(step, total))

    def force_final(self, model):
        """A run stopped early by the time guard still ends at the target budget."""
        kept = sum(int(m.ada_mask.sum()) for _, m in self.modules_)
        if kept > self.bT:
            self._prune_to(self.bT)

    def final_report(self, model):
        ranks = {n.replace("trunk.blocks.", "L"): int(m.ada_mask.sum().item())
                 for n, m in model.named_modules() if isinstance(m, AdaLoRALinear)}
        per_layer: Dict[str, int] = {}
        for n, r in ranks.items():
            per_layer[n.split(".")[0]] = per_layer.get(n.split(".")[0], 0) + r
        return {"adalora_rank_per_layer": per_layer,
                "adalora_budget_final": int(sum(ranks.values())),
                "adalora_budget_initial": self.b0, "adalora_budget_target": self.bT}


# =============================================================================================
# entry point
# =============================================================================================
def default_hparams(technique: str) -> Dict:
    lr = DEFAULT_LR.get(technique, 1e-3)
    head = 10 * lr if technique in ("full_ft", "kd", "student_hard") else HEAD_LR
    return {"lr": lr, "head_lr": head}


def apply_technique(model: nn.Module, technique: str, cfg: Dict, log=print
                    ) -> Tuple[List[Dict], Technique]:
    """Insert the technique, set requires_grad, return (optimizer groups, hook object)."""
    if technique not in ALL_TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r}")
    dev = model.trunk.embed.weight.device
    if technique == "adalora":
        tech = AdaLoRA()
        groups = tech.setup(model, cfg, log)
        head = [(n, p) for n, p in model.named_parameters() if n.startswith("classifier.")]
        for _, p in head:
            p.requires_grad = True
        groups += _groups(head, float(cfg.get("head_lr", HEAD_LR)),
                          float(cfg.get("weight_decay", 0.01)), "head")
        expected = {n for n, _ in head} | {n for n, p in model.named_parameters() if ".ada_" in n}
        actual = {n for n, p in model.named_parameters() if p.requires_grad}
        if actual != expected:
            raise AssertionError(f"[adalora] trainable mismatch {sorted(actual ^ expected)[:5]}")
    else:
        p2 = "full_ft" if technique in ("kd", "student_hard") else technique
        groups = phase2_apply(model, p2, cfg, log=lambda *a: None)
        tech = Technique()
        tech.name = technique
    model.to(dev)                                         # lesson L1
    assert_one_device(model, dev)
    s = model_size(model)
    log(f"   [{technique}] trainable {s['trainable_params']:,} / {s['total_params']:,} "
        f"({s['trainable_pct']:.3f}%)")
    return groups, tech


# =============================================================================================
# merge (used by the self-checks: merged == unmerged)
# =============================================================================================
def _clone_linear(base: nn.Linear) -> nn.Linear:
    lin = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                    device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(base.weight)
        if base.bias is not None:
            lin.bias.copy_(base.bias)
    return lin


def merge_model(model: nn.Module) -> int:
    """Fold LoRA / LoRMA / AdaLoRA updates into plain Linear layers. Returns the count."""
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (LoRALinear, LoRMALinear, AdaLoRALinear))]
    for n, m in hits:
        lin = _clone_linear(m.base)
        with torch.no_grad():
            lin.weight.copy_(m.merged_weight())
        _set(model, n, lin)
    return len(hits)
