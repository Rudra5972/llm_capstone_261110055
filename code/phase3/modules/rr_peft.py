"""
rr_peft.py -- fine-tuning / PEFT techniques for Phase 3, pure PyTorch (no `peft` library).

Every technique has the same interface (class `Technique`):
    apply(model, cfg, log) -> optimizer param groups (the trainable parameters, grouped)
    extra_loss(model)      -> regulariser added to the loss (AdaLoRA) or None
    after_backward(model, step, total_steps)   (AdaLoRA importance + budget; grads unscaled)
    merge(model) -> bool   folds the update into plain layers where meaningful
    state_dict_trainable(model), count(model), final_report(model)
`apply` freezes everything, inserts the modules, unfreezes exactly what the method trains and
ASSERTS that the set of trainable parameter names equals the expected set.

  lorma    LoRMA+  y = (I + s*B*A) W0 x     -- Phase-2 code (ner_peft), compulsory
           Bihany, Patel & Modi (2025), "LoRMA: Low-Rank Multiplicative Adaptation for LLMs",
           arXiv:2506.07621.
  lora     LoRA    y = W0 x + (a/r) B A x  -- Phase-2 code (ner_peft)
           Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models",
           arXiv:2106.09685.
  rslora   LoRA with the rank-stabilised scale a/sqrt(r) (shares the LoRA code; run as its own
           experiment with a higher rank, where the scale matters).
           Kalajdzievski (2023), "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA",
           arXiv:2312.03732.
  adalora  dW = P diag(lambda) Q, orthogonality regulariser ||P^T P - I||^2 + ||Q Q^T - I||^2,
           sensitivity importance |w * dL/dw| smoothed by EMA (+ uncertainty), triplet score
           s(lambda_i) + mean s(P_:i) + mean s(Q_i:), pruning to a cubic budget schedule
           (warm-up, then decay to the target budget, then a final phase with a fixed mask).
           Zhang et al. (2023), "Adaptive Budget Allocation for Parameter-Efficient
           Fine-Tuning", ICLR 2023, arXiv:2303.10512.
  qlora    frozen base linears stored as block-wise NF4 (block 64, packed uint8 + absmax
           scales, optional double quantisation of the scales: int8, block 256, mean-centred),
           de-quantised on the fly in the forward pass; trainable LoRA on top. Simulated
           without bitsandbytes. Dettmers et al. (2023), "QLoRA: Efficient Finetuning of
           Quantized LLMs", NeurIPS 2023, arXiv:2305.14314.
  ia3      learned scaling vectors (init 1) on keys, values and the FFN intermediate activation
           (input of the down projection, i.e. the SwiGLU product for gated FFNs).
           Liu et al. (2022), "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper
           than In-Context Learning", NeurIPS 2022, arXiv:2205.05638.
  prefix   per-layer key/value prefixes (length 20), via a reparameterisation MLP that is
           dropped after training (`merge`). Li & Liang (2021), "Prefix-Tuning: Optimizing
           Continuous Prompts for Generation", ACL 2021, arXiv:2101.00190.
  prompt   20 soft input embeddings, initialised from random vocabulary embeddings.
           Lester, Al-Rfou & Constant (2021), "The Power of Scale for Parameter-Efficient Prompt
           Tuning", EMNLP 2021, arXiv:2104.08691.
  full_ft  every trunk parameter (reference only; also the KD teacher)
  kd       every parameter of the layer-pruned student (see rr_kd.py)
The sentence head (head.*) is always trainable.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_peft import TARGETS, LoRALinear, LoRMALinear, _get, _groups, _set
from ner_peft import apply_technique as phase2_apply_technique
from rr_models import model_size, trainable_state_dict

# IA3 sites per architecture family: (path, kind); kind "kv_fused" scales the k and v slices of
# a fused QKV output, "out" scales a whole projection output, "in" scales a projection input.
IA3_SITES = {
    "gpt2": [("attn.qkv", "kv_fused"), ("mlp.down_proj", "in")],
    "llama": [("attn.k_proj", "out"), ("attn.v_proj", "out"), ("mlp.down_proj", "in")],
    "bloom": [("self_attention.query_key_value", "kv_fused"), ("mlp.down_proj", "in")],
}

# NF4 code book (QLoRA, Appendix E / bitsandbytes `create_normal_map`), checked in selftest
NF4_CODE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


def nf4_codebook_from_normal(offset: float = 0.9677083) -> torch.Tensor:
    """Re-derive the NF4 code book from normal quantiles (8 positive, 7 negative, exact 0)."""
    def ppf(p):
        return torch.special.ndtri(torch.as_tensor(p, dtype=torch.float64))
    pos = ppf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -ppf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    v = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (v / v.abs().max()).float()


def work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.is_cuda and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):
            return torch.get_autocast_gpu_dtype()
    return x.dtype


# =============================================================================================
# modules
# =============================================================================================
class RSLoRALinear(LoRALinear):
    """LoRA with the rank-stabilised scale alpha / sqrt(r)."""

    def __init__(self, base: nn.Linear, r: int = 32, alpha: float = 16, dropout: float = 0.05):
        super().__init__(base, r, alpha, dropout)
        self.scale = alpha / math.sqrt(r)


def quantize_nf4(w: torch.Tensor, block: int = 64):
    """Flat fp tensor -> (packed uint8 (ceil(N/2) padded), absmax float32 per block, N)."""
    code = torch.tensor(NF4_CODE, dtype=torch.float32, device=w.device)
    n = w.numel()
    flat = w.detach().float().flatten()
    pad = (-n) % block
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(1).clamp(min=1e-12)
    normed = (blocks / absmax[:, None]).flatten()
    mids = (code[1:] + code[:-1]) / 2
    idx = torch.bucketize(normed, mids).to(torch.uint8)       # nearest code
    if idx.numel() % 2:
        idx = F.pad(idx, (0, 1))
    packed = idx[0::2] | (idx[1::2] << 4)
    return packed, absmax, n


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, n: int, block: int,
                   code: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    idx = torch.stack([packed & 15, packed >> 4], -1).flatten()[: absmax.numel() * block]
    vals = code.to(dtype)[idx.long()].view(-1, block) * absmax.to(dtype)[:, None]
    return vals.flatten()[:n]


def quantize_absmax_int8(x: torch.Tensor, block: int = 256):
    """Double quantisation of the NF4 scales: mean-centred, int8, per-block absmax."""
    mean = x.mean()
    c = x - mean
    pad = (-c.numel()) % block
    cp = F.pad(c, (0, pad)).view(-1, block)
    s = (cp.abs().amax(1) / 127).clamp(min=1e-12)
    q = torch.round(cp / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q.flatten()[: x.numel()], s, mean


def dequantize_absmax_int8(q, s, mean, block: int = 256):
    pad = (-q.numel()) % block
    qp = F.pad(q.float(), (0, pad)).view(-1, block)
    return (qp * s[:, None]).flatten()[: q.numel()] + mean


class NF4Linear(nn.Module):
    """Frozen linear whose weight is stored as block-wise NF4 and de-quantised per forward."""

    def __init__(self, base: nn.Linear, block: int = 64, double_quant: bool = True,
                 dq_block: int = 256):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.block, self.double_quant, self.dq_block = block, double_quant, dq_block
        packed, absmax, n = quantize_nf4(base.weight, block)
        self.n_quantized = n
        self.register_buffer("qweight", packed)
        self.register_buffer("code", torch.tensor(NF4_CODE, dtype=torch.float32))
        if double_quant:
            q, s, mean = quantize_absmax_int8(absmax, dq_block)
            self.register_buffer("qabs", q)
            self.register_buffer("qabs_scale", s)
            self.register_buffer("qabs_mean", mean.reshape(1))
        else:
            self.register_buffer("absmax", absmax)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

    def scales(self) -> torch.Tensor:
        if self.double_quant:
            return dequantize_absmax_int8(self.qabs, self.qabs_scale, self.qabs_mean[0],
                                          self.dq_block)
        return self.absmax

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        return dequantize_nf4(self.qweight, self.scales(), self.n_quantized, self.block,
                              self.code, dtype).view(self.out_features, self.in_features)

    def forward(self, x):
        dt = work_dtype(x)
        b = self.bias.to(dt) if self.bias is not None else None
        return F.linear(x.to(dt), self.dequantize(dt), b)

    def storage_bytes(self) -> int:
        n = self.qweight.numel()
        if self.double_quant:
            n += self.qabs.numel() + 4 * (self.qabs_scale.numel() + 1)
        else:
            n += 4 * self.absmax.numel()
        return int(n)


class IA3Out(nn.Module):
    """y = base(x) * l, where l scales chosen output slices (others stay 1)."""

    def __init__(self, base: nn.Linear, slices):
        super().__init__()
        self.base = base
        self.slices = list(slices)
        self.ia3_l = nn.ParameterList([nn.Parameter(torch.ones(e - s)) for s, e in self.slices])

    def scale_vector(self, dtype, device) -> torch.Tensor:
        v = torch.ones(self.base.out_features, dtype=dtype, device=device)
        for (s, e), p in zip(self.slices, self.ia3_l):
            v = torch.cat([v[:s], p.to(dtype), v[e:]])
        return v

    def forward(self, x):
        y = self.base(x)
        return y * self.scale_vector(y.dtype, y.device)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        l = self.scale_vector(torch.float32, self.base.weight.device)
        lin = _clone_linear(self.base)
        lin.weight.mul_(l[:, None].to(lin.weight.dtype))
        if lin.bias is not None:
            lin.bias.mul_(l.to(lin.bias.dtype))
        return lin


class IA3In(nn.Module):
    """y = base(x * l): scales the FFN intermediate activation before the down projection."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.ia3_l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.ia3_l.to(x.dtype))

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = _clone_linear(self.base)
        lin.weight.mul_(self.ia3_l[None, :].to(lin.weight.dtype))
        return lin


class AdaLoRALinear(nn.Module):
    """y = W0 x + b + (alpha/r) * P diag(lambda * mask) Q dropout(x); lambda = 0 at init."""

    def __init__(self, base: nn.Linear, r: int = 12, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.ada_P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)   # left vectors
        self.ada_E = nn.Parameter(torch.zeros(r))                             # singular values
        self.ada_Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)    # right vectors
        self.register_buffer("ada_mask", torch.ones(r))
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


class PrefixEncoder(nn.Module):
    """Per-layer key/value prefixes (n_layers, 2, n_kv, length, head_dim)."""

    def __init__(self, n_layers: int, n_kv: int, head_dim: int, d_model: int, length: int = 20,
                 reparam: bool = True, hidden: int = 256, init_emb: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_layers, self.n_kv, self.head_dim, self.length = n_layers, n_kv, head_dim, length
        self.reparam = reparam
        out = n_layers * 2 * n_kv * head_dim
        if reparam:
            emb = init_emb if init_emb is not None else torch.randn(length, d_model) * 0.02
            self.prefix_emb = nn.Parameter(emb.detach().clone().float())
            self.prefix_mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(),
                                            nn.Linear(hidden, out))
            self.prefix_kv = None
        else:
            self.prefix_kv = nn.Parameter(torch.randn(length, out) * 0.02)

    def table(self) -> torch.Tensor:
        return self.prefix_mlp(self.prefix_emb) if self.reparam else self.prefix_kv

    def forward(self, B: int, dtype, device):
        t = self.table().view(self.length, self.n_layers, 2, self.n_kv, self.head_dim)
        t = t.permute(1, 2, 3, 0, 4).to(dtype)          # (L, 2, n_kv, length, hd)
        return [(t[i, 0].unsqueeze(0).expand(B, -1, -1, -1),
                 t[i, 1].unsqueeze(0).expand(B, -1, -1, -1)) for i in range(self.n_layers)]

    @torch.no_grad()
    def materialize(self) -> None:
        """Drop the reparameterisation MLP: keep only the computed prefix table."""
        if not self.reparam:
            return
        kv = self.table().detach().clone()
        del self.prefix_mlp
        del self.prefix_emb
        self.reparam = False
        self.prefix_kv = nn.Parameter(kv, requires_grad=False)


def _clone_linear(base: nn.Linear) -> nn.Linear:
    lin = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
    lin = lin.to(device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(base.weight)
        if base.bias is not None:
            lin.bias.copy_(base.bias)
    lin.weight.requires_grad = False
    if lin.bias is not None:
        lin.bias.requires_grad = False
    return lin


def _replace_modules(model: nn.Module, cls, make) -> int:
    """Replace every submodule of type `cls` by make(module). Returns the count."""
    hits = [(n, m) for n, m in model.named_modules() if isinstance(m, cls)]
    for n, m in hits:
        _set(model, n, make(m))
    return len(hits)


# =============================================================================================
# techniques
# =============================================================================================
class Technique:
    name = "base"

    def apply(self, model: nn.Module, cfg: Dict, log=print) -> List[Dict]:
        raise NotImplementedError

    def extra_loss(self, model: nn.Module) -> Optional[torch.Tensor]:
        return None

    def after_backward(self, model: nn.Module, step: int, total_steps: int) -> None:
        return None

    def merge(self, model: nn.Module) -> bool:
        return False

    def state_dict_trainable(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return trainable_state_dict(model)

    def count(self, model: nn.Module) -> Dict[str, float]:
        return model_size(model)

    def final_report(self, model: nn.Module) -> Dict:
        return {}

    def selectable(self, step: int, total_steps: int) -> bool:
        """May a checkpoint taken after `step` completed steps become best.pt? (AdaLoRA: only
        once the final budget is in place.) Early stopping is also held back until then."""
        return True

    # ---------------------------------------------------------------- shared tail of apply()
    @staticmethod
    def _freeze(model):
        for p in model.parameters():
            p.requires_grad = False

    def _finish(self, model, groups, expected: set, cfg, log) -> List[Dict]:
        head = [(n, p) for n, p in model.named_parameters() if n.startswith("head.")
                or n.startswith("kd_proj.")]
        for _, p in head:
            p.requires_grad = True
        expected = set(expected) | {n for n, _ in head}
        groups = list(groups) + _groups(head, float(cfg.get("head_lr", 1e-3)),
                                        float(cfg.get("weight_decay", 0.01)), "head")
        actual = {n for n, p in model.named_parameters() if p.requires_grad}
        if actual != expected:
            raise AssertionError(f"[{self.name}] trainable-parameter mismatch: unexpected "
                                 f"{sorted(actual - expected)[:5]}, missing "
                                 f"{sorted(expected - actual)[:5]}")
        c = self.count(model)
        log(f"   [{self.name}] trainable {c['trainable_params']:,} / {c['total_params']:,} "
            f"params ({c['trainable_pct']:.3f}%) in {len(actual)} tensors")
        return groups


class FullFT(Technique):
    name = "full_ft"

    def apply(self, model, cfg, log=print):
        groups = phase2_apply_technique(model, "full_ft", cfg, log=lambda *a: None)
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        return self._finish(model, groups, expected, cfg, log)


class KDStudent(FullFT):
    name = "kd"


class Phase2LowRank(Technique):
    """LoRA / LoRMA+: the Phase-2 implementation (ner_peft.apply_technique), wrapped."""

    def __init__(self, which: str):
        self.name = which

    def apply(self, model, cfg, log=print):
        groups = phase2_apply_technique(model, self.name, cfg, log=lambda *a: None)
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        assert expected and all((".lora_" in n) or (".lorma_" in n) for n in expected)
        return self._finish(model, groups, expected, cfg, log)

    def merge(self, model):
        cls = LoRALinear if self.name == "lora" else LoRMALinear

        def make(m):
            lin = _clone_linear(m.base)
            with torch.no_grad():
                lin.weight.copy_(m.merged_weight())
            return lin
        return _replace_modules(model, cls, make) > 0


class RSLoRA(Technique):
    name = "rslora"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        r, alpha = int(cfg.get("rslora_r", 32)), float(cfg.get("rslora_alpha", 16))
        for blk in trunk.blocks:
            for path in TARGETS[trunk.kind]:
                _set(blk, path, RSLoRALinear(_get(blk, path), r, alpha,
                                             float(cfg.get("lora_dropout", 0.05))))
        named = [(n, p) for n, p in model.named_parameters() if ".lora_" in n]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, float(cfg["lr"]), float(cfg.get("weight_decay", 0.01)), "rslora")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)

    def merge(self, model):
        def make(m):
            lin = _clone_linear(m.base)
            with torch.no_grad():
                lin.weight.copy_(m.merged_weight())
            return lin
        return _replace_modules(model, RSLoRALinear, make) > 0


class QLoRA(Technique):
    name = "qlora"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        r, alpha = int(cfg.get("qlora_r", 16)), float(cfg.get("qlora_alpha", 32))
        block, dq = int(cfg.get("qlora_block", 64)), bool(cfg.get("qlora_double_quant", True))
        fp16_bytes = nf4_bytes = 0
        for blk in trunk.blocks:
            for path in TARGETS[trunk.kind]:
                base = _get(blk, path)
                q = NF4Linear(base, block, dq)
                fp16_bytes += 2 * base.weight.numel()
                nf4_bytes += q.storage_bytes()
                _set(blk, path, LoRALinear(q, r, alpha, float(cfg.get("lora_dropout", 0.05))))
        self.memory = {"base_linear_fp16_mb": fp16_bytes / 2**20, "base_linear_nf4_mb":
                       nf4_bytes / 2**20, "ratio": fp16_bytes / max(1, nf4_bytes),
                       "block": block, "double_quant": dq}
        log(f"   [qlora] base linears: {self.memory['base_linear_fp16_mb']:.1f} MB fp16 -> "
            f"{self.memory['base_linear_nf4_mb']:.1f} MB NF4 ({self.memory['ratio']:.2f}x)")
        named = [(n, p) for n, p in model.named_parameters() if ".lora_" in n]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, float(cfg["lr"]), float(cfg.get("weight_decay", 0.01)), "qlora")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)

    def final_report(self, model):
        return {"qlora_memory": getattr(self, "memory", {})}


class IA3(Technique):
    name = "ia3"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        d = trunk.hidden_size
        for blk in trunk.blocks:
            for path, kind in IA3_SITES[trunk.kind]:
                base = _get(blk, path)
                if kind == "kv_fused":
                    assert base.out_features == 3 * d
                    new = IA3Out(base, [(d, 2 * d), (2 * d, 3 * d)])
                elif kind == "out":
                    new = IA3Out(base, [(0, base.out_features)])
                else:
                    new = IA3In(base)
                _set(blk, path, new)
        named = [(n, p) for n, p in model.named_parameters() if ".ia3_l" in n]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, float(cfg["lr"]), 0.0, "ia3")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)

    def merge(self, model):
        n = _replace_modules(model, IA3Out, lambda m: m.merged())
        return (n + _replace_modules(model, IA3In, lambda m: m.merged())) > 0


class AdaLoRA(Technique):
    name = "adalora"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        self.r = int(cfg.get("adalora_init_r", 12))
        self.target_r = int(cfg.get("adalora_target_r", 8))
        alpha = float(cfg.get("adalora_alpha", 32))
        self.tinit_frac = float(cfg.get("adalora_tinit_frac", 0.1))
        self.tfinal_frac = float(cfg.get("adalora_tfinal_frac", 0.2))
        self.delta_t = int(cfg.get("adalora_delta_t", 10))
        self.beta1 = float(cfg.get("adalora_beta1", 0.85))
        self.beta2 = float(cfg.get("adalora_beta2", 0.85))
        self.orth_w = float(cfg.get("adalora_orth", 0.1))
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
        groups = _groups(named, float(cfg["lr"]), float(cfg.get("weight_decay", 0.01)), "adalora")
        log(f"   [adalora] {len(self.modules_)} matrices, rank {self.r} -> target "
            f"{self.target_r} (budget {self.b0} -> {self.bT} singular values)")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)

    # ---------------------------------------------------------------- regulariser
    def extra_loss(self, model):
        if not self.modules_ or self.orth_w <= 0:
            return None
        reg = sum(m.orth_reg() for _, m in self.modules_) / len(self.modules_)
        return self.orth_w * reg

    # ---------------------------------------------------------------- budget schedule
    def phases(self, total: int):
        """(t_init, t_final_start): no pruning before t_init; the final budget is set at
        t_final_start (always <= the last step, so short runs reach it too)."""
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

    def selectable(self, step, total_steps):
        return step > self.phases(total_steps)[1]

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
    def after_backward(self, model, step, total_steps):
        grads_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                       for _, m in self.modules_ for p in (m.ada_P, m.ada_E, m.ada_Q))
        if not grads_ok:
            return
        self._update_importance()
        ti, tf_start = self.phases(total_steps)
        if step < ti or step > tf_start:
            return                                      # warm-up / final phase: mask fixed
        if step != tf_start and (step - ti) % self.delta_t != 0:
            return
        b = self.budget(step, total_steps)
        scores = self._scores()
        flat = torch.cat(scores)
        if b >= flat.numel():
            return
        thr = torch.kthvalue(flat.float().cpu(), flat.numel() - b).values.item()
        for (_, m), s in zip(self.modules_, scores):
            keep = (s > thr).float()
            m.ada_mask.copy_(keep)
            m.ada_E.mul_(keep)

    def merge(self, model):
        def make(m):
            lin = _clone_linear(m.base)
            with torch.no_grad():
                lin.weight.copy_(m.merged_weight())
            return lin
        return _replace_modules(model, AdaLoRALinear, make) > 0

    def final_report(self, model):
        return {**adalora_ranks(model), "adalora_budget_initial": self.b0,
                "adalora_budget_target": self.bT}


def adalora_ranks(model: nn.Module) -> Dict:
    """Rank kept per adapted matrix and per layer (from the saved masks)."""
    ranks = {n.replace("trunk.blocks.", "L"): int(m.ada_mask.sum().item())
             for n, m in model.named_modules() if isinstance(m, AdaLoRALinear)}
    per_layer: Dict[str, int] = {}
    for n, r in ranks.items():
        layer = n.split(".")[0]
        per_layer[layer] = per_layer.get(layer, 0) + r
    return {"adalora_rank_per_matrix": ranks, "adalora_rank_per_layer": per_layer,
            "adalora_budget_final": int(sum(ranks.values()))}


class PrefixTuning(Technique):
    name = "prefix"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        P = int(cfg.get("prefix_len", 20))
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        idx = torch.randint(259, trunk.embed.weight.shape[0], (P,), generator=g)
        trunk.prefix = PrefixEncoder(len(trunk.blocks), trunk.n_kv, trunk.head_dim,
                                     trunk.hidden_size, P, bool(cfg.get("prefix_reparam", True)),
                                     int(cfg.get("prefix_hidden", 256)),
                                     trunk.embed.weight[idx].detach().float())
        named = [(n, p) for n, p in model.named_parameters() if n.startswith("trunk.prefix.")]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, float(cfg["lr"]), 0.0, "prefix")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)

    def merge(self, model):
        if model.trunk.prefix is None or not model.trunk.prefix.reparam:
            return False
        model.trunk.prefix.materialize()
        return True


class PromptTuning(Technique):
    name = "prompt"

    def apply(self, model, cfg, log=print):
        self._freeze(model)
        trunk = model.trunk
        P = int(cfg.get("prompt_len", 20))
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        idx = torch.randint(259, trunk.embed.weight.shape[0], (P,), generator=g)
        trunk.prompt = nn.Parameter(trunk.embed.weight[idx].detach().clone().float())
        named = [(n, p) for n, p in model.named_parameters() if n == "trunk.prompt"]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, float(cfg["lr"]), 0.0, "prompt")
        return self._finish(model, groups, {n for n, _ in named}, cfg, log)


def get_technique(name: str) -> Technique:
    if name in ("lora", "lorma"):
        return Phase2LowRank(name)
    table = {"full_ft": FullFT, "kd": KDStudent, "rslora": RSLoRA, "qlora": QLoRA, "ia3": IA3,
             "adalora": AdaLoRA, "prefix": PrefixTuning, "prompt": PromptTuning}
    if name not in table:
        raise ValueError(f"unknown technique {name!r}")
    return table[name]()


def n_virtual_tokens(cfg: Dict) -> int:
    """Context positions a technique consumes (prompt tuning only)."""
    return int(cfg.get("prompt_len", 20)) if cfg.get("technique") == "prompt" else 0
