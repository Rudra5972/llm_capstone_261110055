"""
rr_models.py -- RRModel = Phase-1 trunk + sentence head (pure PyTorch).

`RRTrunk` subclasses the Phase-2 `ner_models.Trunk` (byte-identical, reused): with no virtual
tokens it calls Trunk.forward unchanged (causal/bidir masks, RoPE, ALiBi, sliding window, GQA
as verified in Phase 2). Two optional additions serve Prompt and Prefix Tuning:

  prompt  (P, d) soft embeddings prepended to the input embeddings (Lester et al., 2021).
          Position ids / masks are built for P + T tokens; prompt columns are visible to every
          query (also outside M3's sliding window); outputs are returned for the T real tokens.
  prefix  per-layer key/value prefixes (Li & Liang, 2021), passed through the Phase-1 blocks'
          own `past` argument, i.e. concatenated AFTER RoPE (prefix keys are not rotated), at
          the KV-head count (GQA), with mask columns that are True for every query (also under
          the sliding window and causal masks) and a zero ALiBi bias.

Sentence vectors are gathered at the EOS marker positions (or mean-pooled over the sentence's
tokens, `pool="mean"`). Heads: `linear` = LayerNorm -> Dropout -> Linear(d, 13); `seqctx` = a
1-layer BiGRU over the ordered sentence vectors of a window (residual) before the linear head,
to model label transitions (Fact -> Issue -> Argument -> Ratio -> Ruling).

Inputs : input_ids / attention_mask (B, T), markers / starts / marker_mask (B, M)
Outputs: logits (B, M, 13) float32, sentence vectors (B, M, d)
"""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ner_models import MAX_POSITIONS, Trunk, load_phase1_lm, torch_load
from phase1_models import causal_allowed, positions_from_mask

STATE_BUFFER_TAGS = (".ada_mask", ".ada_ipt", ".ada_unc")   # non-parameter state to checkpoint


# =============================================================================================
# trunk with virtual tokens
# =============================================================================================
class RRTrunk(Trunk):
    def __init__(self, lm, attn_mode: str = "bidir", layer_ids: Optional[List[int]] = None):
        super().__init__(lm, attn_mode, layer_ids)
        self.prompt: Optional[nn.Parameter] = None   # set by rr_peft (prompt tuning)
        self.prefix: Optional[nn.Module] = None      # set by rr_peft (prefix tuning)
        self.force_virtual = False                   # test hook: use the virtual path even if empty
        if self.kind == "gpt2":
            self.n_kv, self.head_dim = lm.config.n_head, lm.config.n_embd // lm.config.n_head
        elif self.kind == "llama":
            self.n_kv = lm.config.num_key_value_heads
            self.head_dim = lm.config.hidden_size // lm.config.num_attention_heads
        else:
            self.n_kv, self.head_dim = lm.config.n_head, lm.config.hidden_size // lm.config.n_head

    @property
    def n_virtual(self) -> int:
        """Positions consumed by the soft prompt (prefix tuning consumes none)."""
        return 0 if self.prompt is None else int(self.prompt.shape[0])

    def forward(self, input_ids, attention_mask=None):
        if self.prompt is None and self.prefix is None and not self.force_virtual:
            return super().forward(input_ids, attention_mask)
        return self._forward_virtual(input_ids, attention_mask)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _work_dtype(x: torch.Tensor) -> torch.dtype:
        if x.is_cuda and torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):
                return torch.get_autocast_gpu_dtype()
        return x.dtype

    @staticmethod
    def _bool_mask(mask, is_causal, T, dev):
        """(mask, is_causal) from Trunk -> explicit bool (B|1, 1, T, T) or None (= all True)."""
        if mask is None:
            return causal_allowed(T, 0, None, None, dev) if is_causal else None
        return mask

    def _forward_virtual(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        dev = input_ids.device
        P = self.n_virtual
        T2 = P + T
        assert T2 <= MAX_POSITIONS, f"{P} virtual + {T} tokens > {MAX_POSITIONS}"
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        am = attention_mask
        if P:
            am = torch.cat([torch.ones(B, P, dtype=attention_mask.dtype, device=dev),
                            attention_mask], 1)
        pos = positions_from_mask(am, 0, T2, B, dev)
        x = self.embed(input_ids)
        if P:
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(B, P, -1), x], 1)
        past = self.prefix(B, self._work_dtype(x), dev) if self.prefix is not None else None
        K = past[0][0].shape[2] if past is not None else 0

        if self.kind in ("gpt2", "llama"):
            mask, is_causal = self._sdpa_mask(T2, am, dev)
            if P or K:
                mask = self._bool_mask(mask, is_causal, T2, dev)
                is_causal = False
                if mask is not None and P:
                    mask = mask.clone()
                    mask[..., :P] = True                    # soft prompt visible to every query
                if mask is not None and K:
                    mask = torch.cat([torch.ones(*mask.shape[:-1], K, dtype=torch.bool,
                                                 device=dev), mask], -1)
            if self.kind == "gpt2":
                x = x + self.wpe(pos.clamp(max=self.n_positions - 1))
                for i, blk in enumerate(self.blocks):
                    x, _ = blk(x, mask, is_causal, past[i] if past is not None else None)
            else:
                cos, sin = self.rotary(pos)
                for i, blk in enumerate(self.blocks):
                    x, _ = blk(x, cos, sin, mask, is_causal, past[i] if past is not None else None)
        else:
            bias = self._alibi(T2, am, dev)                  # (B, H, T2, T2), -inf = masked
            if P:
                bias = bias.clone()
                bias[..., :P] = torch.where(torch.isinf(bias[..., :P]),
                                            torch.zeros_like(bias[..., :P]), bias[..., :P])
            if K:
                bias = torch.cat([torch.zeros(*bias.shape[:-1], K, dtype=bias.dtype, device=dev),
                                  bias], -1)              # ALiBi bias 0 for prefix keys
            for i, blk in enumerate(self.blocks):
                x, _ = blk(x, bias, past[i] if past is not None else None)
        return self.norm(x)[:, P:]


# =============================================================================================
# heads
# =============================================================================================
class SentenceHead(nn.Module):
    def __init__(self, d: int, n_labels: int, head: str = "linear", dropout: float = 0.1):
        super().__init__()
        assert head in ("linear", "seqctx")
        self.kind = head
        self.ctx = None
        if head == "seqctx":
            self.ctx = nn.GRU(d, d // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.ctx_norm = nn.LayerNorm(d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, n_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, v: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
        if self.ctx is not None:
            lens = marker_mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(self.ctx_norm(v.float()), lens,
                                                       batch_first=True, enforce_sorted=False)
            out, _ = self.ctx(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                      total_length=v.shape[1])
            v = v.float() + out
        return self.classifier(self.dropout(self.norm(v.float()))).float()


class RRModel(nn.Module):
    """Trunk -> sentence vectors (marker / mean pooling) -> head."""

    def __init__(self, trunk: RRTrunk, n_labels: int, head: str = "linear",
                 dropout: float = 0.1, pool: str = "marker"):
        super().__init__()
        assert pool in ("marker", "mean")
        self.trunk = trunk
        self.n_labels = n_labels
        self.pool = pool
        self.head = SentenceHead(trunk.hidden_size, n_labels, head, dropout)
        self.kd_proj: Optional[nn.Module] = None     # optional, KD intermediate term

    def sentence_vectors(self, input_ids, attention_mask, markers, starts):
        h = self.trunk(input_ids, attention_mask)                   # (B, T, d)
        d = h.shape[-1]
        if self.pool == "marker":
            return torch.gather(h, 1, markers.unsqueeze(-1).expand(-1, -1, d))
        cs = torch.cumsum(h.float(), 1)
        cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)        # cs[t] = sum h[:t]
        hi = torch.gather(cs, 1, (markers + 1).unsqueeze(-1).expand(-1, -1, d))
        lo = torch.gather(cs, 1, starts.unsqueeze(-1).expand(-1, -1, d))
        n = (markers - starts + 1).clamp(min=1).unsqueeze(-1).float()
        return ((hi - lo) / n).to(h.dtype)

    def forward(self, input_ids, attention_mask, markers, starts, marker_mask):
        v = self.sentence_vectors(input_ids, attention_mask, markers, starts)
        return self.head(v, marker_mask), v


def collate(windows, idx, pad_id: int, device=None, labels_override=None) -> Dict[str, torch.Tensor]:
    """Right-padded batch of windows. Padded marker slots point at position 0 with
    marker_mask False and label -100. `labels_override` = {window_idx: labels array}."""
    B = len(idx)
    L = max(len(windows[i]["tokens"]) for i in idx)
    M = max(len(windows[i]["markers"]) for i in idx)
    x = torch.full((B, L), pad_id, dtype=torch.long)
    am = torch.zeros((B, L), dtype=torch.long)
    mk = torch.zeros((B, M), dtype=torch.long)
    st = torch.zeros((B, M), dtype=torch.long)
    mm = torch.zeros((B, M), dtype=torch.bool)
    y = torch.full((B, M), -100, dtype=torch.long)
    for r, i in enumerate(idx):
        w = windows[i]
        n, m = len(w["tokens"]), len(w["markers"])
        x[r, :n] = torch.from_numpy(w["tokens"].astype("int64"))
        am[r, :n] = 1
        mk[r, :m] = torch.from_numpy(w["markers"])
        st[r, :m] = torch.from_numpy(w["starts"])
        mm[r, :m] = True
        lab = labels_override.get(i, w["labels"]) if labels_override else w["labels"]
        y[r, :m] = torch.from_numpy(lab.astype("int64"))
    out = {"input_ids": x, "attention_mask": am, "markers": mk, "starts": st,
           "marker_mask": mm, "labels": y}
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def build_rr_model(phase1_dir: str, n_labels: int, attn_mode: str = "bidir",
                   head: str = "linear", layer_ids: Optional[List[int]] = None,
                   dropout: float = 0.1, pool: str = "marker") -> Tuple[RRModel, Dict]:
    lm, cfg = load_phase1_lm(phase1_dir)
    trunk = RRTrunk(lm, attn_mode, layer_ids)
    del lm                                          # unused blocks / lm_head are freed here
    return RRModel(trunk, n_labels, head, dropout, pool), cfg


# =============================================================================================
# checkpoint state: trainable parameters + PEFT state buffers (AdaLoRA masks / importance)
# =============================================================================================
def _is_state_buffer(name: str) -> bool:
    return any(t in name for t in STATE_BUFFER_TAGS)


def trainable_state_dict(model: nn.Module, half: bool = False) -> Dict[str, torch.Tensor]:
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    sd.update({n: b.detach().cpu() for n, b in model.named_buffers() if _is_state_buffer(n)})
    if half:
        sd = {n: (t.half() if t.is_floating_point() and not _is_state_buffer(n) else t)
              for n, t in sd.items()}
    return sd


def load_trainable_state(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """Every key must exist; every trainable parameter / state buffer must be present."""
    params = dict(model.named_parameters())
    bufs = {n: b for n, b in model.named_buffers() if _is_state_buffer(n)}
    unknown = [n for n in sd if n not in params and n not in bufs]
    if unknown:
        raise KeyError(f"checkpoint has unknown tensors: {unknown[:5]}")
    with torch.no_grad():
        for n, t in sd.items():
            dst = params.get(n, bufs.get(n))
            dst.copy_(t.to(dst.dtype))
    expected = {n for n, p in params.items() if p.requires_grad} | set(bufs)
    absent = expected - set(sd)
    if absent:
        raise KeyError(f"checkpoint lacks trainable tensors: {sorted(absent)[:5]}")


def rebuild_trained_model(run_config: Dict, best_pt: str, phase1_dirs: Dict[str, str],
                          device="cpu") -> RRModel:
    """Recreate the exact module structure of a finished run and load its best.pt."""
    from rr_peft import get_technique
    mid = run_config["model"]
    p1 = phase1_dirs.get(mid) or run_config.get("phase1_dir")
    if not p1 or not os.path.exists(os.path.join(p1, "model.pt")):
        raise FileNotFoundError(f"Phase-1 weights for {mid} not found")
    model, _ = build_rr_model(p1, run_config["n_labels"], run_config["attn_mode"],
                              run_config["head"], run_config.get("layer_ids"),
                              run_config.get("dropout", 0.1), run_config.get("pool", "marker"))
    tech = get_technique(run_config["technique"])
    tech.apply(model, run_config, log=lambda *a: None)
    if run_config.get("kd_hidden_weight", 0) and run_config["technique"] == "kd":
        from rr_kd import attach_projection
        attach_projection(model, run_config)
    load_trainable_state(model, torch_load(best_pt))
    return model.to(device).eval()


def model_size(model: nn.Module) -> Dict[str, float]:
    """Parameter counts; quantised (QLoRA) base weights count as parameters of the model."""
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    n_q = sum(int(getattr(m, "n_quantized", 0)) for m in model.modules())
    total = n_all + n_q
    return {"total_params": int(total), "trainable_params": int(n_tr),
            "trainable_pct": 100.0 * n_tr / max(1, total), "quantized_params": int(n_q)}
