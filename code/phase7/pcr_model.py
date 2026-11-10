"""
pcr_model.py -- the Phase-1 LM as a shared-weight bi-encoder for Prior Case Retrieval, the
Phase-7 techniques, and batched chunk encoding.

PcrEncoder = Trunk (the Phase-1 embed -> blocks -> final norm, reused unchanged from Phase 2;
causal attention by default, `bidir` optional) -> pooling over the real tokens of ONE chunk
(`mean` or `last` = the EOS appended to every chunk) -> `classifier` head: a d x d projection
initialised to the IDENTITY (so every technique starts from the raw Phase-1 representation)
+ a learned logit scale (temperature, init 1/0.05) -> L2-normalised chunk vector.
A document vector is the normalised mean of its chunk vectors (pcr_train). Queries and
candidates go through the SAME encoder.

Techniques (PDF 13.2.1 list):
  full_ft / partial_ft / lorma / lora   cjpe_peft.apply_technique (Phases 2-6, unchanged)
  qlora     LoRA r=16 on NF4 4-bit frozen linears (summ_model.NF4Linear, Phase 6)
  prefix    prefix tuning (Li & Liang 2021): the activations of P virtual positions at EVERY
            block input are free parameters (initialised from the frozen model's own
            activations of real text); text tokens attend to them in every layer
  prompt    prompt tuning (Lester et al. 2021): P learned input embeddings only
  ia3       IA^3 (Liu et al. 2022): learned vectors l_k, l_v rescale the attention keys and
            values, l_ff rescales the FFN inner activation (input of down_proj); init 1
  kd / student_hard   full FT of a layer-pruned student (KD loss in pcr_train)
The virtual positions are prepended as P dummy tokens; hooks overwrite their states, pooling
ignores them.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_models import Trunk, load_phase1_lm, torch_load
from ner_peft import FUSED_QKV, LoRALinear, _get, _groups, _set

EOS_ID = 0
NEW_TECHNIQUES = ("qlora", "prefix", "prompt", "ia3")
IA3_KV = {"gpt2": ["attn.qkv"], "bloom": ["self_attention.query_key_value"],
          "llama": ["attn.k_proj", "attn.v_proj"]}


class Head(nn.Module):
    def __init__(self, d: int, scale: float = 20.0):
        super().__init__()
        self.proj = nn.Linear(d, d)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d))
            self.proj.bias.zero_()
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(scale))))

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=float(np.log(100.0))).exp()


class IA3Out(nn.Module):
    """y = base(x) * l on the output slice [lo, hi) (keys / values); l initialised to 1."""

    def __init__(self, base: nn.Linear, lo: int, hi: int):
        super().__init__()
        self.base, self.lo, self.hi = base, lo, hi
        self.ia3_l = nn.Parameter(torch.ones(hi - lo, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        s = torch.ones(y.shape[-1], device=y.device, dtype=y.dtype)
        s = torch.cat([s[:self.lo], self.ia3_l.to(y.dtype), s[self.hi:]])
        return y * s


class IA3In(nn.Module):
    """y = base(x * l_ff) -- rescales the FFN inner activation."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.ia3_l = nn.Parameter(torch.ones(base.in_features, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x * self.ia3_l.to(x.dtype))


class Virtual(nn.Module):
    """P virtual positions: states[0] replaces the embedding output, states[l] (prefix only)
    the input of block l."""

    def __init__(self, states: List[torch.Tensor], mode: str):
        super().__init__()
        self.mode = mode
        self.states = nn.ParameterList([nn.Parameter(s.detach().float().clone()) for s in states])
        self.length = int(states[0].shape[0])


def _overwrite(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    P = s.shape[0]
    return torch.cat([s.to(x.dtype)[None].expand(x.shape[0], -1, -1), x[:, P:]], 1)


class PcrEncoder(nn.Module):
    def __init__(self, lm, attn_mode: str = "causal", layer_ids=None, pooling: str = "mean"):
        super().__init__()
        self.trunk = Trunk(lm, attn_mode, layer_ids)
        self.classifier = Head(self.trunk.hidden_size)
        self.virtual: Optional[Virtual] = None
        self.pooling = pooling
        self._hooks = []

    @property
    def n_virtual(self) -> int:
        return 0 if self.virtual is None else self.virtual.length

    def install_virtual(self, v: Virtual) -> None:
        for h in self._hooks:
            h.remove()
        self.virtual = v
        st = v.states
        if v.mode == "prompt":
            self._hooks = [self.trunk.embed.register_forward_hook(
                lambda m, i, o: _overwrite(o, st[0]))]
            return
        self._hooks = []
        for li in range(len(st)):
            self._hooks.append(self.trunk.blocks[li].register_forward_pre_hook(
                lambda m, a, li=li: (_overwrite(a[0], st[li]),) + tuple(a[1:])))

    def forward(self, input_ids, attention_mask):
        P = self.n_virtual
        if P:
            B = input_ids.shape[0]
            input_ids = torch.cat([input_ids.new_full((B, P), EOS_ID), input_ids], 1)
            attention_mask = torch.cat([attention_mask.new_ones(B, P), attention_mask], 1)
        h = self.trunk(input_ids, attention_mask)[:, P:]
        m = attention_mask[:, P:].to(h.dtype)
        if self.pooling == "last":
            last = attention_mask[:, P:].long().sum(1).clamp(min=1) - 1
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
        else:
            pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.classifier.proj(pooled).float(), dim=-1)


def build_pcr_model(phase1_dir: str, layer_ids=None, attn_mode="causal", pooling="mean"):
    lm, cfg = load_phase1_lm(phase1_dir)
    return PcrEncoder(lm, attn_mode, layer_ids, pooling), cfg


# =============================================================================================
# techniques
# =============================================================================================
def _sample_ids(sample_ids, n):
    ids = [int(x) for x in (sample_ids if sample_ids is not None else [])][:n]
    while len(ids) < n:
        ids.append(3 + len(ids) % 200)
    return ids


def apply_pcr_technique(model: PcrEncoder, technique: str, cfg: Dict, log=print,
                        sample_ids=None):
    """Returns (optimizer groups, technique hooks)."""
    from cjpe_models import assert_one_device, model_size
    from cjpe_peft import Technique, apply_technique
    if technique not in NEW_TECHNIQUES:
        return apply_technique(model, technique, cfg, log=log)
    dev = model.trunk.embed.weight.device
    lr, wd = float(cfg["lr"]), float(cfg.get("weight_decay", 0.01))
    if technique == "qlora":
        from summ_model import NF4Linear
        groups, tech = apply_technique(model, "lora", cfg, log=lambda *a: None)
        n = 0
        for _, m in list(model.named_modules()):
            if isinstance(m, LoRALinear) and isinstance(m.base, nn.Linear):
                m.base = NF4Linear(m.base, int(cfg.get("qlora_block", 64)))
                n += 1
        tech.name = "qlora"
        log(f"   [qlora] {n} frozen linears stored as NF4 (4-bit) + LoRA r={cfg.get('lora_r', 16)}")
    else:
        for p in model.parameters():
            p.requires_grad = False
        trunk = model.trunk
        if technique == "ia3":
            d = trunk.hidden_size
            for blk in trunk.blocks:
                for path in IA3_KV[trunk.kind]:
                    base = _get(blk, path)
                    if FUSED_QKV.get(trunk.kind) == path:
                        _set(blk, path, IA3Out(base, d, 3 * d))           # keys + values
                    else:
                        _set(blk, path, IA3Out(base, 0, base.out_features))
                _set(blk, "mlp.down_proj", IA3In(_get(blk, "mlp.down_proj")))
            named = [(n, p) for n, p in model.named_parameters() if n.endswith("ia3_l")]
        else:
            P = int(cfg.get("n_virtual", 20))
            ids = torch.tensor([_sample_ids(sample_ids, P)], dtype=torch.long, device=dev)
            with torch.no_grad():
                if technique == "prompt":
                    states = [trunk.embed(ids)[0]]
                else:
                    got = {}
                    hs = [b.register_forward_pre_hook(
                        lambda m, a, li=li: got.__setitem__(li, a[0][0].detach()))
                        for li, b in enumerate(trunk.blocks)]
                    try:
                        trunk(ids, torch.ones_like(ids))
                    finally:
                        for h in hs:
                            h.remove()
                    states = [got[li] for li in range(len(trunk.blocks))]
            model.install_virtual(Virtual(states, technique).to(dev))
            named = [(n, p) for n, p in model.named_parameters() if n.startswith("virtual.")]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, lr, 0.0, technique)
        tech = Technique()
        tech.name = technique
    head = [(n, p) for n, p in model.named_parameters() if n.startswith("classifier.")]
    for _, p in head:
        p.requires_grad = True
    if technique != "qlora":
        groups += _groups(head, float(cfg.get("head_lr", lr)), wd, "head")
        actual = {n for n, p in model.named_parameters() if p.requires_grad}
        if actual != {n for n, _ in named} | {n for n, _ in head}:
            raise AssertionError(f"[{technique}] trainable-parameter mismatch")
    model.to(dev)
    assert_one_device(model, dev)
    s = model_size(model)
    log(f"   [{technique}] trainable {s['trainable_params']:,} / {s['total_params']:,} "
        f"({s['trainable_pct']:.3f}%)")
    return groups, tech


def rebuild_pcr(run_config: Dict, best_pt: str, device="cpu", sample_ids=None) -> PcrEncoder:
    """A finished run's encoder (KD teacher, E8)."""
    from cjpe_models import load_trainable_state
    model, _ = build_pcr_model(run_config["phase1_dir"], run_config.get("layer_ids"),
                               run_config.get("attn_mode", "causal"),
                               run_config.get("pooling", "mean"))
    tech = "full_ft" if run_config["technique"] in ("kd", "student_hard") else \
        run_config["technique"]
    apply_pcr_technique(model, tech, run_config, log=lambda *a: None, sample_ids=sample_ids)
    load_trainable_state(model, torch_load(best_pt))
    return model.to(device).eval()


# =============================================================================================
# batching + encoding
# =============================================================================================
def collate(chunks: List[np.ndarray], device):
    L = max(len(c) for c in chunks)
    ids = np.full((len(chunks), L), EOS_ID, np.int64)
    mask = np.zeros((len(chunks), L), np.int64)
    for i, c in enumerate(chunks):
        ids[i, :len(c)] = c
        mask[i, :len(c)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def micro_batches(chunks: List[np.ndarray], micro_tokens: int, n_virtual: int = 0):
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
    out, cur, mx = [], [], 0
    for i in order:
        L = len(chunks[i]) + n_virtual
        if cur and max(mx, L) * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


def encode_chunks(model, chunks: List[np.ndarray], device, amp_ctx, micro_tokens: int,
                  grad: bool = False) -> torch.Tensor:
    """(N, d) float32 L2-normalised chunk vectors, in input order."""
    d = model.trunk.hidden_size
    out = torch.empty(len(chunks), d, device=device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for g in micro_batches(chunks, micro_tokens, model.n_virtual):
            ids, mask = collate([chunks[i] for i in g], device)
            with amp_ctx():
                e = model(ids, mask)
            out[torch.as_tensor(g, device=device)] = e.float()
    return out


def doc_vectors(chunk_vecs: torch.Tensor, owner: torch.Tensor, n_docs: int) -> torch.Tensor:
    """Normalised mean of each document's chunk vectors (differentiable)."""
    s = torch.zeros(n_docs, chunk_vecs.shape[1], device=chunk_vecs.device,
                    dtype=chunk_vecs.dtype).index_add(0, owner, chunk_vecs)
    return F.normalize(s, dim=-1)
