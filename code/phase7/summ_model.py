"""
summ_model.py -- the Phase-1 causal LM as a summariser + the Phase-6 techniques + decoding.

SummLM keeps the Phase-1 LM intact (its own forward, KV cache, positions, ALiBi, sliding window)
and registers a `Trunk` VIEW of the same blocks so the tested Phase-2..5 PEFT code
(cjpe_peft.apply_technique: LoRMA+, LoRA, AdaLoRA, Houlsby adapters, full FT, ...) inserts its
modules into the very blocks the LM runs. `classifier` is an empty module (no new head: the
tied LM head generates the summary).

New techniques (Phase-6 list, PDF 12.2.1):
  pfeiffer  Pfeiffer adapter: ONE bottleneck (d/16) after the FFN output of every block + norms
  qlora     LoRA r=16 on top of NF4-quantised frozen linears (blockwise absmax, block 64, two
            4-bit codes per byte; Dettmers et al. 2023), dequantised on the fly
  prefix    prefix tuning (Li & Liang 2021): P learned key/value vectors per layer, used as the
            KV-cache "past"; initialised from the frozen model's own keys/values of real text
  prompt    prompt tuning (Lester et al. 2021): P learned input embeddings; their keys/values are
            computed through the frozen network on every forward (exactly "prepend P embeddings"
            for a causal decoder), initialised from real token embeddings
Training loss: token cross-entropy on the SUMMARY tokens only; logits are computed only at the
positions that predict a summary token (SelectHead) -> no (B, T, V) tensor.
"""
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_models import Trunk, count_params, load_phase1_lm, torch_load
from ner_peft import ADAPTER_SITES, AdapterWrapped, LoRALinear, _get, _groups, _set
from phase1_models import _banned_ngram_tokens

EOS_ID, PAD_ID = 0, 1
CTX = 1024
BLOCK_ATTR = {"gpt2": "blocks", "llama": "layers", "bloom": "blocks"}
NEW_TECHNIQUES = ("pfeiffer", "qlora", "prefix", "prompt")
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


class SelectHead(nn.Module):
    """lm_head that (when `sel` is set) only computes logits at the selected positions."""

    def __init__(self, head: nn.Linear):
        super().__init__()
        self.head = head
        self.sel = None

    def forward(self, x):
        if self.sel is not None:
            x = x[self.sel]
        return self.head(x)


# =============================================================================================
# virtual tokens (prefix / prompt tuning)
# =============================================================================================
def _attn_dims(trunk) -> Tuple[int, int]:
    a = trunk.blocks[0]
    if trunk.kind == "gpt2":
        return a.attn.n_head, a.attn.head_dim
    if trunk.kind == "llama":
        return a.attn.kv, a.attn.hd
    return a.self_attention.h, a.self_attention.hd


class Prefix(nn.Module):
    def __init__(self, past: List[Tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.k = nn.ParameterList([nn.Parameter(k[0].detach().float().clone()) for k, _ in past])
        self.v = nn.ParameterList([nn.Parameter(v[0].detach().float().clone()) for _, v in past])
        self.length = int(past[0][0].shape[2])

    def past(self, model, B, dtype):
        return [(k.to(dtype)[None].expand(B, -1, -1, -1), v.to(dtype)[None].expand(B, -1, -1, -1))
                for k, v in zip(self.k, self.v)]


class Prompt(nn.Module):
    def __init__(self, emb: torch.Tensor):
        super().__init__()
        self.emb = nn.Parameter(emb.detach().float().clone())
        self.length = int(emb.shape[0])

    def past(self, model, B, dtype):
        e = self.emb
        h = model.trunk.embed.register_forward_hook(lambda m, i, o: e[None].to(o.dtype))
        try:
            ids = torch.zeros(1, self.length, dtype=torch.long, device=e.device)
            out = model.lm(ids, use_cache=True)
        finally:
            h.remove()
        return [(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1))
                for k, v in out.past_key_values]


# =============================================================================================
# NF4 linear (QLoRA base)
# =============================================================================================
class NF4Linear(nn.Module):
    def __init__(self, base: nn.Linear, block: int = 64):
        super().__init__()
        W = base.weight.detach().float()
        self.in_features, self.out_features = base.in_features, base.out_features
        n = W.numel()
        pad = (-n) % block
        flat = torch.cat([W.reshape(-1), W.new_zeros(pad)]).view(-1, block)
        absmax = flat.abs().amax(1).clamp(min=1e-8)
        table = torch.tensor(NF4, device=W.device)
        codes = (flat / absmax[:, None]).unsqueeze(-1).sub(table).abs().argmin(-1).to(torch.uint8)
        codes = codes.view(-1)
        if codes.numel() % 2:
            codes = torch.cat([codes, codes.new_zeros(1)])
        self.register_buffer("q_packed", (codes[0::2] << 4) | codes[1::2])
        self.register_buffer("q_absmax", absmax.half())
        self.register_buffer("q_table", table)
        self.n, self.block = n, block
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(),
                                                                requires_grad=False)

    @property
    def weight(self):
        p = self.q_packed
        codes = torch.stack([p >> 4, p & 15], 1).view(-1).long()
        vals = self.q_table[codes][: self.q_absmax.numel() * self.block].view(-1, self.block)
        W = (vals * self.q_absmax.float()[:, None]).view(-1)[: self.n]
        return W.view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else
                        self.bias.to(x.dtype))


# =============================================================================================
# the model
# =============================================================================================
class SummLM(nn.Module):
    def __init__(self, lm, layer_ids=None):
        super().__init__()
        kind = {"GPT2StyleLM": "gpt2", "LlamaStyleLM": "llama", "BloomStyleLM": "bloom"}[
            type(lm).__name__]
        if layer_ids is not None:
            blocks = getattr(lm, BLOCK_ATTR[kind])
            setattr(lm, BLOCK_ATTR[kind], nn.ModuleList([blocks[i] for i in layer_ids]))
        self.trunk = Trunk(lm, "causal")                 # PEFT view: the SAME block modules
        self.lm = lm
        lm.lm_head = SelectHead(lm.lm_head)
        self.classifier = nn.Module()                     # no new head
        self.virtual = None
        self.kind = kind

    @property
    def n_virtual(self) -> int:
        return 0 if self.virtual is None else self.virtual.length

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                last_only=False, sel=None):
        B = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.virtual is not None:
            if past_key_values is None:
                dt = self.trunk.embed.weight.dtype
                if input_ids.is_cuda and torch.is_autocast_enabled():
                    try:
                        dt = torch.get_autocast_dtype("cuda")
                    except AttributeError:                       # torch < 2.4
                        dt = torch.get_autocast_gpu_dtype()
                past_key_values = self.virtual.past(self, B, dt)
            attention_mask = torch.cat([attention_mask.new_ones(B, self.n_virtual),
                                        attention_mask], 1)
        self.lm.lm_head.sel = sel
        try:
            return self.lm(input_ids, attention_mask=attention_mask,
                           past_key_values=past_key_values, use_cache=use_cache,
                           last_only=last_only)
        finally:
            self.lm.lm_head.sel = None

    def lm_logits_at_targets(self, input_ids, attention_mask, labels):
        """(logits (N, V) fp32, targets (N,)) at the positions that predict a summary token."""
        tgt = labels[:, 1:]
        sel = torch.zeros_like(labels, dtype=torch.bool)
        sel[:, :-1] = tgt != -100
        out = self(input_ids, attention_mask, sel=sel)
        return out.logits.float(), tgt[tgt != -100]


def build_summ_model(phase1_dir: str, layer_ids=None):
    lm, cfg = load_phase1_lm(phase1_dir)
    return SummLM(lm, layer_ids), cfg


# =============================================================================================
# techniques
# =============================================================================================
def _sample_ids(sample_ids, n):
    ids = [int(x) for x in (sample_ids if sample_ids is not None else [])][:n]
    while len(ids) < n:
        ids.append(3 + len(ids) % 200)
    return ids


def apply_summ_technique(model: SummLM, technique: str, cfg: Dict, log=print, sample_ids=None):
    """Returns (optimizer groups, technique hooks). `sample_ids`: real tokens for prefix/prompt
    initialisation."""
    from cjpe_models import assert_one_device, model_size
    from cjpe_peft import Technique, apply_technique
    if technique not in NEW_TECHNIQUES:
        return apply_technique(model, technique, cfg, log=log)
    dev = model.trunk.embed.weight.device
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.01))
    if technique == "qlora":
        groups, tech = apply_technique(model, "lora", cfg, log=lambda *a: None)
        n = 0
        for name, m in list(model.named_modules()):
            if isinstance(m, LoRALinear) and isinstance(m.base, nn.Linear):
                m.base = NF4Linear(m.base, int(cfg.get("qlora_block", 64)))
                n += 1
        tech.name = "qlora"
        log(f"   [qlora] {n} frozen linears stored as NF4 (4-bit) + LoRA r={cfg.get('lora_r', 16)}")
    else:
        for p in model.parameters():
            p.requires_grad = False
        trunk = model.trunk
        if technique == "pfeiffer":
            bott = int(cfg.get("pfeiffer_bottleneck") or max(8, trunk.hidden_size // 16))
            for blk in trunk.blocks:
                path = ADAPTER_SITES[trunk.kind][1]                     # FFN output only
                _set(blk, path, AdapterWrapped(_get(blk, path), bott))
            named = [(n, p) for n, p in model.named_parameters() if ".adapter." in n]
            norm_ids = {id(p) for m in trunk.norm_modules() for p in m.parameters()}
            named += [(n, p) for n, p in model.named_parameters() if id(p) in norm_ids]
        else:
            P = int(cfg.get("n_virtual", 20))
            ids = torch.tensor([_sample_ids(sample_ids, P)], dtype=torch.long, device=dev)
            with torch.no_grad():
                if technique == "prefix":
                    model.virtual = Prefix(model.lm(ids, use_cache=True).past_key_values)
                else:
                    model.virtual = Prompt(trunk.embed(ids)[0])
            named = [(n, p) for n, p in model.named_parameters() if n.startswith("virtual.")]
        for _, p in named:
            p.requires_grad = True
        groups = _groups(named, lr, wd if technique == "pfeiffer" else 0.0, technique)
        tech = Technique()
        tech.name = technique
        actual = {n for n, p in model.named_parameters() if p.requires_grad}
        if actual != {n for n, _ in named}:
            raise AssertionError(f"[{technique}] trainable-parameter mismatch")
    model.to(dev)
    assert_one_device(model, dev)
    s = model_size(model)
    log(f"   [{technique}] trainable {s['trainable_params']:,} / {s['total_params']:,} "
        f"({s['trainable_pct']:.3f}%)")
    return groups, tech


def rebuild_summ(run_config: Dict, best_pt: str, phase1_dir: Optional[str] = None,
                 device="cpu") -> SummLM:
    """A finished run's model (KD teacher)."""
    from cjpe_models import load_trainable_state
    model, _ = build_summ_model(phase1_dir or run_config["phase1_dir"],
                                run_config.get("layer_ids"))
    tech = "full_ft" if run_config["technique"] in ("kd", "student_hard") else \
        run_config["technique"]
    apply_summ_technique(model, tech, run_config, log=lambda *a: None)
    load_trainable_state(model, torch_load(best_pt))
    return model.to(device).eval()


# =============================================================================================
# batching + decoding
# =============================================================================================
def collate_train(seqs: List[Tuple[np.ndarray, np.ndarray]], device=None) -> Dict:
    """seqs: (prompt incl. EOS-start and separator, target) -> right-padded batch with labels
    -100 everywhere except the target tokens."""
    T = max(len(p) + len(t) for p, t in seqs)
    assert T <= CTX, f"sequence of {T} tokens > {CTX}"
    ids = np.full((len(seqs), T), PAD_ID, np.int64)
    mask = np.zeros((len(seqs), T), np.int64)
    lab = np.full((len(seqs), T), -100, np.int64)
    for i, (p, t) in enumerate(seqs):
        n = len(p) + len(t)
        ids[i, :len(p)] = p
        ids[i, len(p):n] = t
        mask[i, :n] = 1
        lab[i, len(p):n] = t
    b = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
         "labels": torch.from_numpy(lab), "n_tokens": int(mask.sum()),
         "n_targets": int((lab != -100).sum())}
    if device is not None:
        for k in ("input_ids", "attention_mask", "labels"):
            b[k] = b[k].to(device, non_blocking=True)
    return b


def collate_prompts(prompts: List[np.ndarray], device) -> Tuple[torch.Tensor, torch.Tensor]:
    T = max(len(p) for p in prompts)
    ids = np.full((len(prompts), T), PAD_ID, np.int64)
    mask = np.zeros((len(prompts), T), np.int64)
    for i, p in enumerate(prompts):                         # LEFT padding for generation
        ids[i, T - len(p):] = p
        mask[i, T - len(p):] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def _ban(logits, rows, finished, n_ngram):
    if n_ngram and n_ngram > 1:
        for b, row in enumerate(rows):
            if not finished[b] and len(row) >= n_ngram:
                bad = _banned_ngram_tokens(row, n_ngram)
                if bad:
                    logits[b, bad] = float("-inf")


@torch.inference_mode()
def decode_batch(model, ids, mask, max_new: int, min_new: int = 0, num_beams: int = 1,
                 no_repeat_ngram: int = 4, length_penalty: float = 1.0) -> List[List[int]]:
    """Greedy (num_beams=1) or beam search with n-gram blocking over the GENERATED tokens only
    (copying phrases from the document stays possible). Returns token lists without EOS."""
    B = ids.shape[0]
    K = max(1, int(num_beams))
    out = model(ids, attention_mask=mask, use_cache=True, last_only=True)
    past, logits = out.past_key_values, out.logits[:, -1].float()
    if K > 1:
        past = [(k.repeat_interleave(K, 0), v.repeat_interleave(K, 0)) for k, v in past]
        logits = logits.repeat_interleave(K, 0)
        mask = mask.repeat_interleave(K, 0)
    R = B * K
    rows = [[] for _ in range(R)]
    fin = [False] * R
    scores = torch.zeros(B, K, device=ids.device)
    if K > 1:
        scores[:, 1:] = float("-inf")
    for step in range(max_new):
        logits[:, PAD_ID] = float("-inf")
        if step < min_new:
            logits[:, EOS_ID] = float("-inf")
        _ban(logits, rows, fin, no_repeat_ngram)
        if K == 1:
            nxt = logits.argmax(-1)
            nl = nxt.tolist()
            for b in range(R):
                if not fin[b]:
                    if nl[b] == EOS_ID:
                        fin[b] = True
                    else:
                        rows[b].append(nl[b])
            nxt = torch.where(torch.tensor(fin, device=nxt.device), torch.full_like(nxt, EOS_ID),
                              nxt)
        else:
            logp = logits.log_softmax(-1)
            ft = torch.tensor(fin, device=logp.device)
            if ft.any():
                logp[ft] = float("-inf")
                logp[ft, EOS_ID] = 0.0
            V = logp.shape[-1]
            cand = (scores.view(-1, 1) + logp).view(B, K * V)
            top, idx = cand.topk(K, dim=1)
            src = (torch.arange(B, device=idx.device)[:, None] * K + idx // V).view(-1)
            tok = (idx % V).view(-1)
            scores = top
            sl, tl = src.tolist(), tok.tolist()
            rows = [list(rows[s]) for s in sl]
            fin = [fin[s] for s in sl]
            for r in range(R):
                if not fin[r]:
                    if tl[r] == EOS_ID:
                        fin[r] = True
                    else:
                        rows[r].append(tl[r])
            past = [(k[src], v[src]) for k, v in past]
            mask = mask[src]
            nxt = tok
        if all(fin) or step == max_new - 1:
            break
        mask = torch.cat([mask, mask.new_ones(R, 1)], 1)
        out = model(nxt[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
    if K == 1:
        return rows
    lens = torch.tensor([max(1, len(r)) for r in rows], device=scores.device,
                        dtype=torch.float32).view(B, K)
    best = (scores / lens ** length_penalty).argmax(1).tolist()
    return [rows[b * K + best[b]] for b in range(B)]


def generate_many(model, prompts: List[np.ndarray], device, amp_ctx, max_new: int,
                  min_new: int = 0, num_beams: int = 1, no_repeat_ngram: int = 4,
                  batch: int = 16, log=None) -> List[List[int]]:
    """Length-sorted batches; OOM -> batch halves. Output order == input order."""
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    res: List[Optional[List[int]]] = [None] * len(prompts)
    i = 0
    bs = max(1, int(batch))
    while i < len(order):
        g = order[i:i + bs]
        try:
            ids, mask = collate_prompts([prompts[j] for j in g], device)
            with amp_ctx():
                outs = decode_batch(model, ids, mask, max_new, min_new, num_beams,
                                    no_repeat_ngram)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            if log:
                log(f"   [oom] generation batch -> {bs}")
            continue
        for j, o in zip(g, outs):
            res[j] = o
        i += len(g)
    return res
