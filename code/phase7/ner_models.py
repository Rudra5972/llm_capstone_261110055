"""
ner_models.py -- Phase-1 trunk wrappers + NER heads.

The Phase-1 models' forward() only returns LM logits and is always causal. For NER we need the
final hidden states, optionally with bidirectional attention. `Trunk` re-implements only the
tiny "embed -> blocks -> final norm" loop and REUSES the original submodule objects (no copies;
lm_head is dropped, which also avoids a (B, T, 32000) logits tensor).

Attention modes (section 5.4):
  causal : exactly the Phase-1 masks (verified equal to the original forward in tests)
  bidir  : M1/M2 key-padding mask only; M3 symmetric sliding window |q-k| < window;
           M4 ALiBi with symmetric distance -slope*|q-k|. Padding is on the right.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_common import MODELS
from phase1_models import (MODEL_CLASSES, RMSNorm, alibi_bias, build_attn_mask, causal_allowed,
                           positions_from_mask)

MAX_POSITIONS = 1024        # every Phase-1 model was trained on 1024-token blocks


def torch_load(path, map_location="cpu"):
    """torch.load that works on old and new torch (weights_only appeared in 1.13)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # weights_only=True refuses non-tensor pickles (our resumable checkpoints)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def load_phase1_lm(phase1_dir):
    """Rebuild a Phase-1 model from final/{config.json, model.pt}; strict key match."""
    with open(os.path.join(phase1_dir, "config.json")) as f:
        cfg = json.load(f)
    cls, cfg_cls = MODEL_CLASSES[cfg["model_class"]]
    lm = cls(cfg_cls(**cfg["config"]))
    sd = torch_load(os.path.join(phase1_dir, "model.pt"))
    lm.load_state_dict(sd, strict=True)
    return lm, cfg


def kind_of(lm):
    name = type(lm).__name__
    return {"GPT2StyleLM": "gpt2", "LlamaStyleLM": "llama", "BloomStyleLM": "bloom"}[name]


class Trunk(nn.Module):
    """embed -> selected blocks -> final norm, reusing the Phase-1 submodules."""

    def __init__(self, lm, attn_mode="bidir", layer_ids=None):
        super().__init__()
        assert attn_mode in ("causal", "bidir")
        self.kind = kind_of(lm)
        self.attn_mode = attn_mode
        c = lm.config
        self.window = None
        if self.kind == "gpt2":
            self.embed, self.wpe, blocks, self.norm = lm.wte, lm.wpe, lm.blocks, lm.ln_f
            self.n_positions = c.n_positions
            self.hidden_size = c.n_embd
        elif self.kind == "llama":
            self.embed, self.rotary, blocks, self.norm = lm.embed_tokens, lm.rotary, lm.layers, lm.norm
            self.window = c.sliding_window
            self.hidden_size = c.hidden_size
        else:
            self.embed, blocks, self.norm = lm.word_embeddings, lm.blocks, lm.ln_f
            self.register_buffer("slopes", lm.slopes.detach().clone(), persistent=False)
            self.hidden_size = c.hidden_size
        self.n_layers_original = len(blocks)
        self.layer_ids = list(range(len(blocks))) if layer_ids is None else list(layer_ids)
        assert all(0 <= i < len(blocks) for i in self.layer_ids)
        self.blocks = nn.ModuleList([blocks[i] for i in self.layer_ids])

    # ---------------------------------------------------------------- masks
    def _sdpa_mask(self, T, attention_mask, dev):
        """(mask, is_causal) for M1/M2/M3 blocks."""
        if self.attn_mode == "causal":
            return build_attn_mask(T, 0, attention_mask, self.window, dev)
        no_pad = bool(attention_mask.all())
        win = self.window if (self.window is not None and self.window < T) else None
        if no_pad and win is None:
            return None, False
        q = torch.arange(T, device=dev)[:, None]
        k = torch.arange(T, device=dev)[None, :]
        allowed = torch.ones(1, 1, T, T, dtype=torch.bool, device=dev)
        if win is not None:
            allowed = allowed & ((q - k).abs() < win)[None, None]
        if not no_pad:
            allowed = allowed & attention_mask[:, None, None, :].bool()
            allowed = allowed | (q == k)[None, None]          # no all-masked rows (pad queries)
        B = attention_mask.size(0)
        return allowed.expand(B, 1, T, T).contiguous(), False

    def _alibi(self, T, attention_mask, dev):
        pos_k = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        if self.attn_mode == "causal":                          # identical to BloomStyleLM.forward
            allowed = causal_allowed(T, 0, attention_mask, None, dev)
            return alibi_bias(pos_k[:, -T:], pos_k, self.slopes, allowed)
        dist = (pos_k[:, None, :, None] - pos_k[:, None, None, :]).abs().float()
        bias = -self.slopes.view(1, -1, 1, 1) * dist
        q = torch.arange(T, device=dev)[:, None]
        k = torch.arange(T, device=dev)[None, :]
        allowed = attention_mask[:, None, None, :].bool() | (q == k)[None, None]
        return bias.masked_fill(~allowed, float("-inf"))

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        assert T <= MAX_POSITIONS, f"sequence of {T} tokens > {MAX_POSITIONS} (Phase-1 context)"
        dev = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pos = positions_from_mask(attention_mask, 0, T, B, dev)
        if self.kind == "gpt2":
            pos = pos.clamp(max=self.n_positions - 1)
            mask, is_causal = self._sdpa_mask(T, attention_mask, dev)
            x = self.embed(input_ids) + self.wpe(pos)
            for blk in self.blocks:
                x, _ = blk(x, mask, is_causal, None)
        elif self.kind == "llama":
            cos, sin = self.rotary(pos)
            mask, is_causal = self._sdpa_mask(T, attention_mask, dev)
            x = self.embed(input_ids)
            for blk in self.blocks:
                x, _ = blk(x, cos, sin, mask, is_causal, None)
        else:
            bias = self._alibi(T, attention_mask, dev)
            x = self.embed(input_ids)
            for blk in self.blocks:
                x, _ = blk(x, bias, None)
        return self.norm(x)

    def norm_modules(self):
        return [m for m in self.modules() if isinstance(m, (nn.LayerNorm, RMSNorm))]


class NERModel(nn.Module):
    """Trunk + dropout + Linear(H, C) [+ CRF]."""

    def __init__(self, trunk, n_labels, head="linear", dropout=0.1):
        super().__init__()
        self.trunk = trunk
        self.n_labels = n_labels
        self.num_tags = 1 + 2 * n_labels
        self.head_type = head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(trunk.hidden_size, self.num_tags)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.crf = None
        if head == "crf":
            from ner_crf import CRF
            self.crf = CRF(n_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
        h = self.trunk(input_ids, attention_mask)
        logits = self.classifier(self.dropout(h)).float()
        loss = None
        if labels is not None:
            if self.crf is None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                       weight=class_weights, ignore_index=-100)
            else:                                   # drop the EOS position, mask = real labels
                lab = labels[:, 1:]
                loss = self.crf.neg_log_likelihood(logits[:, 1:], lab, lab != -100)
        return logits, loss


def build_ner_model(phase1_dir, n_labels, attn_mode="bidir", head="linear", layer_ids=None,
                    dropout=0.1):
    lm, cfg = load_phase1_lm(phase1_dir)
    trunk = Trunk(lm, attn_mode, layer_ids)
    del lm                                          # unused blocks / lm_head are freed here
    return NERModel(trunk, n_labels, head, dropout), cfg


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


# ---------------------------------------------------------------------------------------------
# trained-model reconstruction (teacher for E6, prediction, ensembles)
# ---------------------------------------------------------------------------------------------
def trainable_state_dict(model, half=False):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    if half:
        sd = {n: (t.half() if t.is_floating_point() else t) for n, t in sd.items()}
    return sd


def load_trainable_state(model, sd):
    """Load a trainable-only state dict; every key must exist and match a trainable param."""
    params = dict(model.named_parameters())
    missing = [n for n in sd if n not in params]
    if missing:
        raise KeyError(f"checkpoint has unknown parameters: {missing[:5]}")
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].dtype))
    expected = {n for n, p in params.items() if p.requires_grad}
    absent = expected - set(sd)
    if absent:
        raise KeyError(f"checkpoint lacks trainable parameters: {sorted(absent)[:5]}")


def rebuild_trained_model(run_config, best_pt, phase1_dirs, device="cpu"):
    """Recreate the exact module structure of a finished run and load its best.pt."""
    from ner_peft import apply_technique
    mid = run_config["model"]
    p1 = phase1_dirs.get(mid) or run_config.get("phase1_dir")
    if not p1 or not os.path.exists(os.path.join(p1, "model.pt")):
        raise FileNotFoundError(f"Phase-1 weights for {mid} ({MODELS[mid]['arch']}) not found")
    model, _ = build_ner_model(p1, run_config["n_labels"], run_config["attn_mode"],
                               run_config["head"], run_config.get("layer_ids"))
    apply_technique(model, run_config["technique"], run_config, log=lambda *a: None)
    load_trainable_state(model, torch_load(best_pt))
    return model.to(device).eval()
