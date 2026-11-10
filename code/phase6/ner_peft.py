"""
ner_peft.py -- fine-tuning / PEFT techniques, implemented from scratch (no `peft` library).

  full_ft      every trunk parameter + head            (E1; also KD students E6 / X2)
  lorma        LoRMA+  h = (I + s*B*A) W0 x  (+ bias)   (E2, compulsory)
  lora         LoRA    h = W0 x + s*B*A x   (+ bias)    (E3)
  adapters     Houlsby bottleneck after attention-out and MLP-out, + norms (E4)
  partial_ft   top-6 blocks + final norm + head, layer-wise LR decay (E5)
  bitfit       bias terms only (M1/M4)                  (extra X1)

`apply_technique(model, technique, cfg)` freezes everything, inserts the modules, unfreezes
exactly what the method trains, ASSERTS that the set of trainable parameter names equals the
expected set, and returns optimizer parameter groups (no weight decay on 1-D tensors).

LoRMA reference: Bihany, Patel & Modi (2025), "LoRMA: Low-Rank Multiplicative Adaptation for
LLMs", arXiv:2506.07621. We implement the additive-identity rank inflation (LoRMA+): the
frozen output y0 = W0 x is multiplied by (I + s*B*A) with B (d_out x r) = 0 at init, so the
layer starts as the exact frozen layer. The d_out x d_out product is never materialised:
y = y0 + s * (y0 @ A^T) @ B^T. For fused QKV projections each of q, k, v gets its own B, A.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_models import RMSNorm

# Linear layers adapted by LoRA / LoRMA, per architecture family (paths inside each block).
TARGETS = {
    "gpt2": ["attn.qkv", "attn.o_proj", "mlp.up_proj", "mlp.down_proj"],
    "llama": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "bloom": ["self_attention.query_key_value", "self_attention.o_proj",
              "mlp.up_proj", "mlp.down_proj"],
}
FUSED_QKV = {"gpt2": "attn.qkv", "bloom": "self_attention.query_key_value"}
ADAPTER_SITES = {
    "gpt2": ["attn.o_proj", "mlp.down_proj"],
    "llama": ["attn.o_proj", "mlp.down_proj"],
    "bloom": ["self_attention.o_proj", "mlp.down_proj"],
}


def _get(mod, path):
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set(mod, path, new):
    parts = path.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], new)


# =============================================================================================
# modules
# =============================================================================================
class LoRALinear(nn.Module):
    """y = W0 x + b + (alpha/r) * B(A(dropout(x)))."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)),
                                         self.lora_B.to(x.dtype))

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A).to(self.base.weight.dtype)


class LoRMALinear(nn.Module):
    """LoRMA+ : y = (I + s*B*A) (W0 x) + b, applied separately to each output slice."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, slices=None):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        d_out = base.out_features
        self.slices = slices or [(0, d_out)]
        assert self.slices[0][0] == 0 and self.slices[-1][1] == d_out
        assert all(a[1] == b[0] for a, b in zip(self.slices, self.slices[1:])), "slices must tile"
        self.lorma_A = nn.ParameterList()
        self.lorma_B = nn.ParameterList()
        for (s, e) in self.slices:
            A = nn.Parameter(torch.empty(r, e - s))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lorma_A.append(A)
            self.lorma_B.append(nn.Parameter(torch.zeros(e - s, r)))

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)                  # frozen W0 x, bias added after
        parts = []
        for (s, e), A, B in zip(self.slices, self.lorma_A, self.lorma_B):
            ys = y0[..., s:e]
            parts.append(ys + self.scale * ((ys @ A.to(ys.dtype).t()) @ B.to(ys.dtype).t()))
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if self.base.bias is not None:
            y = y + self.base.bias
        return y

    @torch.no_grad()
    def merged_weight(self):
        W0 = self.base.weight
        W = W0.clone()
        for (s, e), A, B in zip(self.slices, self.lorma_A, self.lorma_B):
            W[s:e] = W0[s:e] + self.scale * (B @ (A @ W0[s:e]))
        return W


class Adapter(nn.Module):
    """Houlsby bottleneck: h + up(gelu(down(h))); up ~ N(0, 1e-3) so it starts near identity."""

    def __init__(self, d, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


class AdapterWrapped(nn.Module):
    """Wraps a sub-layer's output projection: adapter(base(x)), i.e. BEFORE the residual add."""

    def __init__(self, base: nn.Linear, bottleneck=64):
        super().__init__()
        self.base = base
        self.adapter = Adapter(base.out_features, bottleneck)

    def forward(self, x):
        return self.adapter(self.base(x))


# =============================================================================================
# applying a technique
# =============================================================================================
def _groups(named, lr, wd, tag):
    dec = [p for n, p in named if p.ndim >= 2]
    nodec = [p for n, p in named if p.ndim < 2]
    out = []
    if dec:
        out.append({"params": dec, "lr": lr, "weight_decay": wd, "name": f"{tag}/decay"})
    if nodec:
        out.append({"params": nodec, "lr": lr, "weight_decay": 0.0, "name": f"{tag}/no_decay"})
    return out


def apply_technique(model, technique, cfg, log=print):
    """Mutates `model` in place; returns optimizer param groups."""
    trunk = model.trunk
    kind = trunk.kind
    lr = float(cfg.get("lr", 1e-3))
    head_lr = float(cfg.get("head_lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.01))
    for p in model.parameters():
        p.requires_grad = False

    head_named = [(n, p) for n, p in model.named_parameters()
                  if n.startswith("classifier.") or n.startswith("crf.")]
    expected = {n for n, _ in head_named}
    groups = []

    if technique in ("full_ft", "kd", "student_hard"):
        named = [(n, p) for n, p in model.named_parameters() if n.startswith("trunk.")]
        expected |= {n for n, _ in named}
        for _, p in named:
            p.requires_grad = True
        groups += _groups(named, lr, wd, "trunk")

    elif technique in ("lora", "lorma"):
        r = int(cfg.get(f"{technique}_r", 16))
        alpha = float(cfg.get(f"{technique}_alpha", 32))
        for bi, blk in enumerate(trunk.blocks):
            for path in TARGETS[kind]:
                base = _get(blk, path)
                assert isinstance(base, nn.Linear), f"{path} is {type(base)}"
                if technique == "lora":
                    new = LoRALinear(base, r, alpha, float(cfg.get("lora_dropout", 0.05)))
                else:
                    slices = None
                    if FUSED_QKV.get(kind) == path:
                        d = trunk.hidden_size
                        assert base.out_features == 3 * d
                        slices = [(0, d), (d, 2 * d), (2 * d, 3 * d)]
                    new = LoRMALinear(base, r, alpha, slices)
                _set(blk, path, new)
        named = [(n, p) for n, p in model.named_parameters()
                 if ".lora_A" in n or ".lora_B" in n or ".lorma_A" in n or ".lorma_B" in n]
        for _, p in named:
            p.requires_grad = True
        expected |= {n for n, _ in named}
        groups += _groups(named, lr, wd, technique)

    elif technique == "adapters":
        bott = int(cfg.get("adapter_bottleneck", 64))
        for blk in trunk.blocks:
            for path in ADAPTER_SITES[kind]:
                _set(blk, path, AdapterWrapped(_get(blk, path), bott))
        named = [(n, p) for n, p in model.named_parameters() if ".adapter." in n]
        norm_ids = {id(p) for m in trunk.norm_modules() for p in m.parameters()}
        named += [(n, p) for n, p in model.named_parameters() if id(p) in norm_ids]
        for _, p in named:
            p.requires_grad = True
        expected |= {n for n, _ in named}
        groups += _groups(named, lr, wd, "adapters")

    elif technique == "bitfit":
        named = [(n, p) for n, p in model.named_parameters()
                 if n.startswith("trunk.") and n.endswith(".bias")]
        if not named:
            raise ValueError(f"BitFit needs bias terms; the {kind} architecture has none")
        for _, p in named:
            p.requires_grad = True
        expected |= {n for n, _ in named}
        groups += _groups(named, lr, wd, "bitfit")

    elif technique == "partial_ft":
        k = int(cfg.get("partial_top_k", 6))
        decay = float(cfg.get("layer_decay", 0.85))
        L = len(trunk.blocks)
        k = min(k, L)
        for i in range(L - k, L):
            named = [(f"trunk.blocks.{i}.{n}", p) for n, p in trunk.blocks[i].named_parameters()]
            for _, p in named:
                p.requires_grad = True
            expected |= {n for n, _ in named}
            groups += _groups(named, lr * decay ** (L - 1 - i), wd, f"block{i}")
        named = [(f"trunk.norm.{n}", p) for n, p in trunk.norm.named_parameters()]
        for _, p in named:
            p.requires_grad = True
        expected |= {n for n, _ in named}
        groups += _groups(named, lr, wd, "final_norm")
    else:
        raise ValueError(f"unknown technique {technique!r}")

    for _, p in head_named:
        p.requires_grad = True
    groups += _groups(head_named, head_lr, wd, "head")

    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != expected:
        raise AssertionError(f"trainable-parameter mismatch: unexpected "
                             f"{sorted(actual - expected)[:5]}, missing {sorted(expected - actual)[:5]}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"   [{technique}] trainable {n_tr:,} / {n_all:,} params ({100 * n_tr / n_all:.3f}%) "
        f"in {len(actual)} tensors")
    return groups


def summarize_trainable(model, max_lines=40):
    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    lines = [f"{n}  {k:,}" for n, k in rows[:max_lines]]
    if len(rows) > max_lines:
        lines.append(f"... and {len(rows) - max_lines} more tensors")
    return lines
