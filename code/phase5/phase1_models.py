"""
phase1_models.py -- the Phase-1 model code, copied VERBATIM from the Phase-1 training notebooks.

  * cell 13 of every notebook (identical in all four): shared core
    (CausalLMOutput, positions_from_mask, causal_allowed, build_attn_mask, init_weights,
     decoding helpers, CausalLMBase). The only change: its trailing print() was removed so
     that importing this module is silent.
  * cell 15, class definitions only (everything before `MODEL_CLASS, CONFIG_CLASS = ...`):
      - 01_train_gpt2style       -> GPT2StyleConfig, GPT2Attention, GELUMLP, GPT2Block, GPT2StyleLM
      - 02_train_llamastyle      -> LlamaStyleConfig, RMSNorm, RotaryEmbedding, GQAttention, SwiGLU,
                                    LlamaBlock, LlamaStyleLM
        (03_train_mistral_sliding defines the SAME classes; it differs only by a module-level
         SLIDING_WINDOW constant and one comment. The window comes from config.json, so one
         copy serves both M2 and M3.)
      - 04_train_bloom_alibi     -> BloomStyleConfig, alibi_slopes, alibi_bias, BloomAttention,
                                    BloomMLP, BloomBlock, BloomStyleLM

DO NOT EDIT the copied code: state_dict keys of the Phase-1 `model.pt` files depend on it.
Everything NER-specific lives in ner_models.py.
"""

# =============================================================================================
# notebook cell 13 (shared core)
# =============================================================================================
import math
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[list] = None      # per layer: (k, v), each (B, n_kv_heads, S, head_dim)


def positions_from_mask(attention_mask, past_len, T, B, device):
    """Position ids for the T new tokens. attention_mask (if given) covers ALL keys: (B, past_len+T)."""
    if attention_mask is None:
        return torch.arange(past_len, past_len + T, device=device).unsqueeze(0).expand(B, T)
    pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    return pos[:, -T:]


def causal_allowed(T, past_len, attention_mask, window, device):
    """Bool (B|1, 1, T, S): True where query t may attend key s. Always materialised."""
    S = past_len + T
    q = torch.arange(past_len, S, device=device)[:, None]      # absolute index of each query
    k = torch.arange(S, device=device)[None, :]
    allowed = k <= q                                             # causal
    if window is not None:
        allowed = allowed & (q - k < window)                     # sliding window: last `window` keys incl. self
    allowed = allowed[None, None]
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()   # drop padding keys
        allowed = allowed | (q == k)[None, None]                 # a pad query would see nothing -> NaN; let it see itself
    return allowed


def build_attn_mask(T, past_len, attention_mask, window, device):
    """Returns (mask, is_causal) for F.scaled_dot_product_attention."""
    S = past_len + T
    window_active = window is not None and window < S
    no_pad = attention_mask is None or bool(attention_mask.all())
    if no_pad and not window_active:
        if past_len == 0:
            return None, True        # fast fused causal kernel (training path)
        if T == 1:
            return None, False       # single cached query sees every key
    return causal_allowed(T, past_len, None if no_pad else attention_mask,
                          window if window_active else None, device), False


def init_weights(model, n_layer, std=0.02):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
    for name, p in model.named_parameters():       # residual-branch output projections
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layer))


# ------------------------------------------------------------------ decoding helpers
def _sample(logits, temperature, top_p, top_k):
    logits = logits / max(float(temperature), 1e-5)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        probs = sl.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p           # keep the smallest set with mass >= top_p
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(1)


def _banned_ngram_tokens(row, n):
    """Tokens that would complete an n-gram already present in `row` (list of ints)."""
    if n < 2 or len(row) < n:
        return []
    arr = np.asarray(row)
    win = sliding_window_view(arr[:-1], n - 1)           # window i = arr[i:i+n-1], followed by arr[i+n-1]
    hit = np.nonzero((win == arr[-(n - 1):]).all(1))[0]
    return arr[hit + n - 1].tolist()


class CausalLMBase(nn.Module):
    """Loss + generation shared by all four architectures."""

    @staticmethod
    def lm_loss(logits, labels):
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    def _process_logits(self, logits, seq, rows, repetition_penalty, no_repeat_ngram_size, suppress):
        if repetition_penalty and repetition_penalty != 1.0:
            sc = torch.gather(logits, 1, seq)
            sc = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
            logits = logits.scatter(1, seq, sc)
        if no_repeat_ngram_size and no_repeat_ngram_size > 1:
            for b, row in enumerate(rows):
                banned = _banned_ngram_tokens(row, no_repeat_ngram_size)
                if banned:
                    logits[b, banned] = float("-inf")
        if suppress:
            logits[:, list(suppress)] = float("-inf")
        return logits

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=256, do_sample=False,
                 temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                 no_repeat_ngram_size=0, num_beams=1, suppress_token_ids=(),
                 stop_fn=None, stop_check_every=16, **_ignored):
        """Returns ONLY the new tokens, shape (B, n_new). stop_fn(new_tokens) -> True ends early."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.long()
        rows = [ids[m.bool()].tolist() for ids, m in zip(input_ids, attention_mask)]   # real tokens only
        if num_beams > 1:
            return self._beam_search(input_ids, attention_mask, rows, max_new_tokens, num_beams,
                                     repetition_penalty, no_repeat_ngram_size, suppress_token_ids,
                                     stop_fn, stop_check_every)
        out = self(input_ids, attention_mask=attention_mask, use_cache=True, last_only=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
        seq, new = input_ids, []
        for step in range(max_new_tokens):
            logits = self._process_logits(logits, seq, rows, repetition_penalty,
                                          no_repeat_ngram_size, suppress_token_ids)
            nxt = _sample(logits, temperature, top_p, top_k) if do_sample else logits.argmax(-1)
            new.append(nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(nxt)[:, None]], 1)
            for row, t in zip(rows, nxt.tolist()):
                row.append(t)
            if step == max_new_tokens - 1:
                break
            if stop_fn is not None and (step + 1) % stop_check_every == 0 and stop_fn(torch.stack(new, 1)):
                break
            out = self(nxt[:, None], attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1].float()
        return torch.stack(new, 1)

    def _beam_search(self, input_ids, attention_mask, rows, max_new_tokens, K, repetition_penalty,
                     no_repeat_ngram_size, suppress, stop_fn, stop_check_every):
        B = input_ids.size(0)
        out = self(input_ids, attention_mask=attention_mask, use_cache=True, last_only=True)
        past = [(k.repeat_interleave(K, 0), v.repeat_interleave(K, 0)) for k, v in out.past_key_values]
        logits = out.logits[:, -1].float().repeat_interleave(K, 0)
        seq = input_ids.repeat_interleave(K, 0)
        am = attention_mask.repeat_interleave(K, 0)
        rows = [list(r) for r in rows for _ in range(K)]
        scores = torch.zeros(B, K, device=input_ids.device)
        scores[:, 1:] = -1e9                       # all beams start identical: expand from beam 0 only
        gen = seq[:, :0]
        for step in range(max_new_tokens):
            logits = self._process_logits(logits, seq, rows, repetition_penalty, no_repeat_ngram_size, suppress)
            logp = logits.log_softmax(-1)
            V = logp.size(-1)
            cand = (scores.view(-1, 1) + logp).view(B, K * V)
            top, idx = cand.topk(K, dim=1)
            src = (torch.arange(B, device=idx.device)[:, None] * K + idx // V).view(-1)
            tok = (idx % V).view(-1)
            scores = top
            gen = torch.cat([gen[src], tok[:, None]], 1)
            seq = torch.cat([seq[src], tok[:, None]], 1)
            am = torch.cat([am[src], torch.ones_like(tok)[:, None]], 1)
            rows = [rows[i] + [t] for i, t in zip(src.tolist(), tok.tolist())]
            past = [(k[src], v[src]) for k, v in past]
            if step == max_new_tokens - 1:
                break
            if stop_fn is not None and (step + 1) % stop_check_every == 0 and stop_fn(gen):
                break
            out = self(tok[:, None], attention_mask=am, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1].float()
        best = scores.argmax(1)
        return gen.view(B, K, -1)[torch.arange(B, device=best.device), best]


# =============================================================================================
# 01_train_gpt2style, cell 15 (classes)
# =============================================================================================
@dataclass
class GPT2StyleConfig:
    vocab_size: int
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 14
    n_head: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02


class GPT2Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.head_dim, self.d = c.n_head, c.n_embd // c.n_head, c.n_embd
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.o_proj = nn.Linear(c.n_embd, c.n_embd)

    def forward(self, x, mask, is_causal, past=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, H, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(B, T, self.d)                        # merge heads
        return self.o_proj(y), (k, v)


class GELUMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up_proj = nn.Linear(d, hidden)
        self.down_proj = nn.Linear(hidden, d)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class GPT2Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.n_embd, eps=c.layer_norm_eps)
        self.attn = GPT2Attention(c)
        self.ln_2 = nn.LayerNorm(c.n_embd, eps=c.layer_norm_eps)
        self.mlp = GELUMLP(c.n_embd, c.mlp_ratio * c.n_embd)

    def forward(self, x, mask, is_causal, past=None):
        a, present = self.attn(self.ln_1(x), mask, is_causal, past)       # pre-LN residual
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT2StyleLM(CausalLMBase):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.n_positions, c.n_embd)
        self.blocks = nn.ModuleList([GPT2Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd, eps=c.layer_norm_eps)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        init_weights(self, c.n_layer, c.initializer_range)
        self.lm_head.weight = self.wte.weight                              # tied input/output embeddings

    def get_input_embeddings(self):
        return self.wte

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None,
                use_cache=False, last_only=False):
        B, T = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        pos = positions_from_mask(attention_mask, past_len, T, B, input_ids.device)
        pos = pos.clamp(max=self.config.n_positions - 1)
        mask, is_causal = build_attn_mask(T, past_len, attention_mask, None, input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        presents = []
        for i, blk in enumerate(self.blocks):
            x, pr = blk(x, mask, is_causal, past_key_values[i] if past_key_values else None)
            presents.append(pr)
        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = self.lm_loss(logits, labels) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents if use_cache else None)


# =============================================================================================
# 02_train_llamastyle, cell 15 (classes) -- also used for 03_train_mistral_sliding
# =============================================================================================
@dataclass
class LlamaStyleConfig:
    vocab_size: int
    max_position_embeddings: int = 2048
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    sliding_window: Optional[int] = None      # None = full causal attention
    initializer_range: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pos):                                   # pos: (B, T) int
        freqs = pos[..., None].float() * self.inv_freq        # (B, T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)               # (B, T, hd): angle for dim i and i+hd/2
        return emb.cos()[:, None], emb.sin()[:, None]         # (B, 1, T, hd) -> broadcast over heads


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


class GQAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.hidden_size % c.num_attention_heads == 0
        assert c.num_attention_heads % c.num_key_value_heads == 0, "KV heads must divide query heads"
        self.h, self.kv = c.num_attention_heads, c.num_key_value_heads
        self.hd = c.hidden_size // c.num_attention_heads
        self.n_rep = self.h // self.kv
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, c.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask, is_causal, past=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)    # (B, 12, T, hd)
        k = self.k_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)   # (B,  4, T, hd)
        v = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)           # rotate Q and K only
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        present = (k, v)                                                  # cache stays at 4 heads
        if self.n_rep > 1:                                                # kv head j serves q heads j*3..j*3+2
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.h * self.hd)), present


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = GQAttention(c)
        self.post_attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = SwiGLU(c.hidden_size, c.intermediate_size)

    def forward(self, x, cos, sin, mask, is_causal, past=None):
        a, present = self.attn(self.input_norm(x), cos, sin, mask, is_causal, past)
        x = x + a
        x = x + self.mlp(self.post_attn_norm(x))
        return x, present


class LlamaStyleLM(CausalLMBase):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.rotary = RotaryEmbedding(c.hidden_size // c.num_attention_heads, c.rope_theta)
        self.layers = nn.ModuleList([LlamaBlock(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        init_weights(self, c.num_hidden_layers, c.initializer_range)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None,
                use_cache=False, last_only=False):
        B, T = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        pos = positions_from_mask(attention_mask, past_len, T, B, input_ids.device)
        cos, sin = self.rotary(pos)
        mask, is_causal = build_attn_mask(T, past_len, attention_mask,
                                          self.config.sliding_window, input_ids.device)
        x = self.embed_tokens(input_ids)
        presents = []
        for i, layer in enumerate(self.layers):
            x, pr = layer(x, cos, sin, mask, is_causal, past_key_values[i] if past_key_values else None)
            presents.append(pr)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = self.lm_loss(logits, labels) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents if use_cache else None)


# =============================================================================================
# 04_train_bloom_alibi, cell 15 (classes)
# =============================================================================================
@dataclass
class BloomStyleConfig:
    vocab_size: int
    hidden_size: int = 768
    n_layer: int = 14
    n_head: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02


def alibi_slopes(n_head):
    def pow2_slopes(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(n_head).is_integer():
        return pow2_slopes(n_head)
    closest = 2 ** math.floor(math.log2(n_head))
    return pow2_slopes(closest) + alibi_slopes(2 * closest)[0::2][: n_head - closest]


def alibi_bias(pos_q, pos_k, slopes, allowed):
    """pos_q (B,T), pos_k (B,S), slopes (H,), allowed bool (B|1,1,T,S) -> float bias (B,H,T,S)."""
    dist = (pos_q[:, None, :, None] - pos_k[:, None, None, :]).clamp(min=0).float()
    bias = -slopes.view(1, -1, 1, 1) * dist
    return bias.masked_fill(~allowed, float("-inf"))


class BloomAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.hidden_size % c.n_head == 0
        self.h, self.hd, self.d = c.n_head, c.hidden_size // c.n_head, c.hidden_size
        self.query_key_value = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x, bias, past=None):
        B, T, _ = x.shape
        q, k, v = self.query_key_value(x).split(self.d, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))   # additive ALiBi bias
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.d)), (k, v)


class BloomMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up_proj = nn.Linear(d, hidden)
        self.down_proj = nn.Linear(hidden, d)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class BloomBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.self_attention = BloomAttention(c)
        self.post_attention_layernorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.mlp = BloomMLP(c.hidden_size, c.mlp_ratio * c.hidden_size)

    def forward(self, x, bias, past=None):
        a, present = self.self_attention(self.input_layernorm(x), bias, past)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class BloomStyleLM(CausalLMBase):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList([BloomBlock(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.register_buffer("slopes", torch.tensor(alibi_slopes(c.n_head), dtype=torch.float32),
                             persistent=False)
        init_weights(self, c.n_layer, c.initializer_range)
        self.lm_head.weight = self.word_embeddings.weight

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None,
                use_cache=False, last_only=False):
        B, T = input_ids.shape
        dev = input_ids.device
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        S = past_len + T
        if attention_mask is None:                          # all rows identical -> batch dim 1
            pos_k = torch.arange(S, device=dev)[None]
            allowed = causal_allowed(T, past_len, None, None, dev)
        else:
            pos_k = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
            allowed = causal_allowed(T, past_len, attention_mask, None, dev)
        bias = alibi_bias(pos_k[:, -T:], pos_k, self.slopes, allowed)    # computed once, shared by all layers
        x = self.word_embeddings(input_ids)                  # no position embedding, no embedding LayerNorm (see note)
        presents = []
        for i, blk in enumerate(self.blocks):
            x, pr = blk(x, bias, past_key_values[i] if past_key_values else None)
            presents.append(pr)
        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = self.lm_loss(logits, labels) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents if use_cache else None)


# ---------------------------------------------------------------------------------------------
# Registry (added for Phase 2; not part of the copied code)
# ---------------------------------------------------------------------------------------------
MODEL_CLASSES = {
    "GPT2StyleLM": (GPT2StyleLM, GPT2StyleConfig),
    "LlamaStyleLM": (LlamaStyleLM, LlamaStyleConfig),
    "BloomStyleLM": (BloomStyleLM, BloomStyleConfig),
}
