"""
selftest.py -- the tests the Phase-3 code must pass (section 11 of the code-generation prompt).

    python selftest.py [--out SELFTEST.json] [--data-dir DIR] [--full-size]

CPU only, tiny random models of the four Phase-1 architectures (no weights needed), synthetic
documents (plus the real dev/test documents when a data cache is given):
  1  Macro-P/R/F1 against hand-computed cases; the empty-class rule
  2  window packing: every sentence covered, markers at the right positions, unpack(pack(doc))
     recovers order and count -- incl. a document > 1024 tokens and a one-sentence document,
     for doc / sent / ctx_k, bidir / causal, with and without virtual tokens
  3  output length == number of sentences for every dev doc; the validator rejects corrupted
     submissions
  4  every PEFT technique: trainable count > 0 (and <= MAX_TRAINABLE_PCT with --full-size);
     zero-init variants reproduce the base logits (LoRA/LoRMA/rsLoRA/AdaLoRA B or lambda = 0,
     IA3 ones, prefix/prompt P = 0, QLoRA = its NF4 base); merge == unmerged within 1e-4;
     rsLoRA scale; AdaLoRA cubic budget + pruning to the target; NF4 code book, round-trip
     error bound and memory
  5  prefix (and soft-prompt) positions are visible to every query under all four attention
     variants, causal and bidir (incl. M3's sliding window)
  6  overfit: 32 sentences reach > 95% train accuracy within 300 steps (rr_train, subprocess)
  7  checkpoint -> kill -> resume reproduces the step counter and a finite loss (subprocess)

`job_checks(cfg, meta, docs, device)` runs the technique-specific checks on the REAL Phase-1
model; rr_train calls it in smoke mode before the throughput measurement.
"""
import argparse
import copy
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, List

import numpy as np
import torch

from rr_common import MAX_TRAINABLE_PCT, MODELS, SPEC, write_json
from rr_data import (build_doc_windows, build_windows, check_packing, combine_slot_scores,
                     load_cache)
from rr_eval import macro_f1, macro_scores, tune_bias
from rr_models import build_rr_model, collate, model_size
from rr_peft import (NF4_CODE, NF4Linear, RSLoRALinear, dequantize_nf4, get_technique,
                     nf4_codebook_from_normal, quantize_nf4)

HERE = os.path.dirname(os.path.abspath(__file__))
V_TINY = 400
TINY = {
    "M1": ("GPT2StyleLM", {"vocab_size": V_TINY, "n_positions": 1024, "n_embd": 32, "n_layer": 4,
                           "n_head": 4, "mlp_ratio": 4}),
    "M2": ("LlamaStyleLM", {"vocab_size": V_TINY, "max_position_embeddings": 1024,
                            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                            "num_attention_heads": 4, "num_key_value_heads": 2,
                            "sliding_window": None}),
    "M3": ("LlamaStyleLM", {"vocab_size": V_TINY, "max_position_embeddings": 1024,
                            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                            "num_attention_heads": 4, "num_key_value_heads": 2,
                            "sliding_window": 8}),
    "M4": ("BloomStyleLM", {"vocab_size": V_TINY, "hidden_size": 32, "n_layer": 4, "n_head": 4,
                            "mlp_ratio": 4}),
}
FULL = {
    "M1": ("GPT2StyleLM", {"vocab_size": 32000, "n_positions": 2048, "n_embd": 768, "n_layer": 14,
                           "n_head": 12, "mlp_ratio": 4}),
    "M2": ("LlamaStyleLM", {"vocab_size": 32000, "max_position_embeddings": 2048,
                            "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                            "num_attention_heads": 12, "num_key_value_heads": 4,
                            "sliding_window": None}),
    "M3": ("LlamaStyleLM", {"vocab_size": 32000, "max_position_embeddings": 2048,
                            "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                            "num_attention_heads": 12, "num_key_value_heads": 4,
                            "sliding_window": 512}),
    "M4": ("BloomStyleLM", {"vocab_size": 32000, "hidden_size": 768, "n_layer": 14, "n_head": 12,
                            "mlp_ratio": 4}),
}
PEFT = ["lorma", "lora", "rslora", "adalora", "qlora", "ia3", "prefix", "prompt"]


# =============================================================================================
# fixtures
# =============================================================================================
def write_phase1(root: str, mid: str, spec=TINY, seed: int = 0) -> str:
    """A Phase-1-style final/ folder (config.json + model.pt) with random weights."""
    from phase1_models import MODEL_CLASSES
    cls_name, conf = spec[mid]
    d = os.path.join(root, f"p1_{mid}", "final")
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(seed)
    cls, ccls = MODEL_CLASSES[cls_name]
    lm = cls(ccls(**conf))
    torch.save(lm.state_dict(), os.path.join(d, "model.pt"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_class": cls_name, "arch": MODELS[mid]["arch"], "config": conf}, f)
    return d


def synthetic_cache(root: str, n_train: int = 6, n_dev: int = 2, n_test: int = 2,
                    seed: int = 0) -> str:
    """meta.json + docs.pkl with random token ids; labels follow a learnable rule."""
    rng = np.random.default_rng(seed)
    labels = SPEC["labels"]
    docs, split = {}, {"train": [], "dev": [], "test": []}
    test_ids = {}
    for dom in SPEC["domains"]:
        for sp, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
            for i in range(n):
                did = f"{dom}_{sp}_{i}"
                key = f"{dom}/{did}"
                ns = int(rng.integers(8, 30))
                lab = rng.integers(0, len(labels), ns)
                ids = []
                for j in range(ns):
                    L = int(rng.integers(3, 25))
                    s = rng.integers(3 + 30 * (lab[j] % 10), 33 + 30 * (lab[j] % 10), L)
                    ids.append(s.astype(np.int32))
                docs[key] = {"key": key, "id": did, "domain": dom, "split": sp,
                             "text": [f"s{j}" for j in range(ns)], "ids": ids,
                             "labels": None if sp == "test" else lab.astype(np.int64)}
                split[sp].append(key)
                if sp == "test":
                    test_ids.setdefault(dom, []).append(did)
    meta = {"version": 3, "labels": labels, "split": split, "test_ids": test_ids, "eos_id": 0,
            "pad_id": 1, "unk_id": 2, "vocab_size": V_TINY, "tokenizer_md5": "synthetic",
            "domain_prefix_ids": {"CL": [350, 351], "IT": [352, 353]}}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    write_json(os.path.join(root, "meta.json"), meta)
    write_json(os.path.join(root, "GATES.json"), {"status": "PASSED", "synthetic": True})
    return root


def base_cfg(technique: str, mid: str, phase1_dir: str, data_dir: str, **over) -> Dict:
    """A complete run_config (the orchestrator's defaults) for tests."""
    import run_experiment as rx
    a = rx.build_parser().parse_args(["--exp", "E1"])
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    job = {"model": mid, "folder": f"test_{technique}_{mid}", "technique": technique,
           "overrides": over}
    cfg = rx.make_run_config(a, "TEST", "TEST", job, meta, data_dir, phase1_dir, "bidir", {})
    cfg["n_virtual"] = int(cfg["prompt_len"]) if technique == "prompt" else 0
    return cfg


def fake_meta(prefix=(350, 351)) -> Dict:
    return {"eos_id": 0, "pad_id": 1, "domain_prefix_ids": {"CL": list(prefix),
                                                            "IT": list(prefix)}}


# =============================================================================================
# 1. metrics
# =============================================================================================
def t1_metrics() -> Dict:
    labs = ["A", "B", "C"]
    r = macro_scores([0, 0, 1, 1], [0, 1, 1, 1], labs)
    assert abs(r["macro_f1"] - (2 / 3 + 0.8) / 2) < 1e-9, r["macro_f1"]
    assert abs(r["macro_precision"] - (1 + 2 / 3) / 2) < 1e-9
    assert abs(r["macro_recall"] - (0.5 + 1.0) / 2) < 1e-9
    assert r["classes_skipped"] == ["C"] and r["classes_used"] == ["A", "B"]
    r2 = macro_scores([0, 0], [0, 1], labs)               # predicted-but-absent class counts as 0
    assert abs(r2["macro_f1"] - (2 / 3 + 0.0) / 2) < 1e-9, r2["macro_f1"]
    r3 = macro_scores([0, 1, 2], [0, 1, 2], labs)
    assert r3["macro_f1"] == 1.0
    r4 = macro_scores([0, 0, 0], [1, 1, 1], labs)
    assert r4["macro_f1"] == 0.0 and r4["classes_used"] == ["A", "B"]
    # sklearn-documented example: y=[0,1,2,0,1,2], p=[0,2,1,0,0,1] -> macro F1 0.2667
    r5 = macro_scores([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], labs)
    assert abs(r5["macro_f1"] - 0.26666666666666666) < 1e-9, r5["macro_f1"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        g, p = rng.integers(0, 5, 50), rng.integers(0, 5, 50)
        assert abs(macro_f1(g, p, 13) - macro_scores(g, p, [str(i) for i in range(13)])
                   ["macro_f1"]) < 1e-12
    lg = rng.normal(size=(200, 4))
    gold = rng.integers(0, 4, 200)
    b, f = tune_bias(lg, gold)
    assert f >= macro_f1(gold, lg.argmax(1), 4) - 1e-12 and np.all(np.abs(b) <= 3)
    return {"cases": 6, "rule": "skip classes with zero support and zero predictions"}


# =============================================================================================
# 2. packing
# =============================================================================================
def _doc(key, lens, rng, labels=True):
    ids = [rng.integers(3, 300, L).astype(np.int32) for L in lens]
    return {"key": key, "domain": "CL", "ids": ids,
            "labels": rng.integers(0, 13, len(lens)).astype(np.int64) if labels else None}


def t2_packing() -> Dict:
    rng = np.random.default_rng(1)
    meta = fake_meta()
    docs = [_doc("CL/one", [12], rng), _doc("CL/long", [40] * 60, rng),
            _doc("CL/trunc", [10, 300, 7, 1, 256, 255], rng),
            _doc("CL/var", list(rng.integers(1, 200, 80)), rng)]
    assert sum(len(x) + 1 for x in docs[1]["ids"]) > 1024
    n = 0
    for mode in ("doc", "sent", "ctx_1", "ctx_2"):
        for attn in ("bidir", "causal"):
            for nv in (0, 20):
                cfg = {"input_mode": mode, "attn_mode": attn, "max_len": 1024,
                       "max_sent_tokens": 256, "n_virtual": nv}
                for d in docs:
                    for purpose in ("train", "infer"):
                        r = check_packing(d, cfg, meta, purpose)
                        n += 1
                        if mode == "doc" and d["key"] == "CL/long":
                            assert r["windows"] >= 3 and r["max_occurrences"] >= 2
    cfg = {"input_mode": "doc", "attn_mode": "bidir", "max_len": 1024, "max_sent_tokens": 256}
    wins = build_doc_windows(docs[1], cfg, meta, "infer")
    slot = [w["sent_idx"].astype(np.float32)[:, None].repeat(13, 1) for w in wins]
    for how in ("center", "mean"):
        comb = combine_slot_scores(wins, slot, {"CL/long": 60}, how)["CL/long"]
        assert np.allclose(comb[:, 0], np.arange(60)), how
    return {"configurations_checked": n}


# =============================================================================================
# 3. output length + validator
# =============================================================================================
def t3_outputs(data_dir: str) -> Dict:
    from rr_predict import g2_validator_roundtrip, official_test_counts
    meta, docs = load_cache(data_dir)
    cfg = {"input_mode": "doc", "attn_mode": "bidir", "max_len": 1024, "max_sent_tokens": 256,
           "n_virtual": 0}
    rng = np.random.default_rng(2)
    checked = 0
    for split in ("dev", "test"):
        keys = meta["split"][split]
        wins = build_windows(docs, keys, cfg, meta, "infer")
        slot = [rng.normal(size=(len(w["markers"]), 13)).astype(np.float32) for w in wins]
        comb = combine_slot_scores(wins, slot, {k: len(docs[k]["ids"]) for k in keys})
        for k in keys:
            assert comb[k].shape == (len(docs[k]["text"]), 13), k
            checked += 1
    with tempfile.TemporaryDirectory() as td:
        g2 = g2_validator_roundtrip(official_test_counts(meta, docs), meta["labels"], td)
    return {"docs_checked": checked, "rejected_variants": sorted(g2["rejected"])}


# =============================================================================================
# 4. PEFT
# =============================================================================================
def _batch(meta, docs, n=2, device="cpu", attn="bidir", n_virtual=0, max_tokens=None):
    cfg = {"input_mode": "doc", "attn_mode": attn, "max_len": max_tokens or 1024,
           "max_sent_tokens": 256, "n_virtual": n_virtual}
    keys = meta["split"]["dev"][:2]
    wins = build_windows(docs, keys, cfg, meta, "infer")
    wins = sorted(wins, key=lambda w: len(w["tokens"]))[:n]
    return collate(wins, list(range(len(wins))), meta["pad_id"], device)


def _fwd(m, b):
    with torch.no_grad():
        return m(b["input_ids"], b["attention_mask"], b["markers"], b["starts"],
                 b["marker_mask"])[0]


def _randomize(m, tech: str, scale: float = 0.05) -> None:
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not p.requires_grad or n.startswith("head."):
                continue
            noise = torch.randn(p.shape, generator=g).to(p.device, p.dtype) * scale
            if tech == "ia3":
                p.copy_(1 + noise)
            else:
                p.add_(noise)


def _qlora_reference(m0, cfg):
    """The model whose target linears are NF4 (no LoRA): QLoRA with B = 0 must equal it."""
    from ner_peft import TARGETS, _get, _set
    ref = copy.deepcopy(m0)
    for blk in ref.trunk.blocks:
        for path in TARGETS[ref.trunk.kind]:
            _set(blk, path, NF4Linear(_get(blk, path), cfg["qlora_block"],
                                      cfg["qlora_double_quant"]))
    return ref


def peft_checks(m0, cfg: Dict, b, tech: str, check_pct: bool, tol_zero: float = 1e-5,
                tol_merge: float = 1e-4) -> Dict:
    out0 = _fwd(m0, b)
    res = {}
    # zero-init reproduces the base (P = 0 for prefix / prompt; NF4 base for QLoRA)
    zc = dict(cfg, technique=tech, prefix_len=0, prompt_len=0)
    mz = copy.deepcopy(m0)
    get_technique(tech).apply(mz, zc, log=lambda *x: None)
    mz.eval()
    ref = _fwd(_qlora_reference(m0, zc), b) if tech == "qlora" else out0
    dz = float((_fwd(mz, b) - ref).abs().max())
    res["zero_init_max_abs_diff"] = dz
    assert dz <= tol_zero * max(1.0, float(ref.abs().max())), f"{tech}: zero-init diff {dz}"
    # trainable count / pct, and a training step produces finite grads
    m1 = copy.deepcopy(m0)
    t = get_technique(tech)
    t.apply(m1, dict(cfg, technique=tech), log=lambda *x: None)
    size = model_size(m1)
    res.update(trainable=size["trainable_params"], trainable_pct=size["trainable_pct"])
    assert size["trainable_params"] > 0
    if check_pct and tech in MAX_TRAINABLE_PCT:
        assert size["trainable_pct"] <= MAX_TRAINABLE_PCT[tech], \
            f"{tech}: {size['trainable_pct']:.2f}% > {MAX_TRAINABLE_PCT[tech]}%"
    m1.train()
    logits, _ = m1(b["input_ids"], b["attention_mask"], b["markers"], b["starts"],
                   b["marker_mask"])
    loss = logits[b["marker_mask"]].float().logsumexp(-1).mean()
    extra = t.extra_loss(m1)
    if extra is not None:
        loss = loss + extra
    loss.backward()
    gr = [p.grad for n, p in m1.named_parameters() if p.requires_grad and not n.startswith("head.")]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in gr), f"{tech}: no gradient"
    assert all(g is None or bool(torch.isfinite(g).all()) for g in gr)
    m1.zero_grad(set_to_none=True)
    # merge == unmerged
    m1.eval()
    _randomize(m1, tech)
    out_u = _fwd(m1, b)
    if t.merge(m1):
        out_m = _fwd(m1, b)
        dm = float((out_m - out_u).abs().max())
        res["merge_max_abs_diff"] = dm
        assert dm <= tol_merge * max(1.0, float(out_u.abs().max())), f"{tech}: merge diff {dm}"
    else:
        res["merge"] = "not applicable"
    return res


def t4_peft(data_dir: str, root: str) -> Dict:
    meta, docs = load_cache(data_dir)
    out = {}
    for mid in MODELS:
        p1 = write_phase1(root, mid)
        m0, _ = build_rr_model(p1, 13, "bidir", "linear")
        m0.eval()
        for tech in PEFT:
            cfg = base_cfg(tech, mid, p1, data_dir)
            b = _batch(meta, docs, n_virtual=cfg["n_virtual"])
            out[f"{mid}/{tech}"] = peft_checks(m0, cfg, b, tech, False)
    # rsLoRA scale formula
    lin = torch.nn.Linear(16, 16)
    for r, al in ((4, 8), (32, 16), (64, 16)):
        assert abs(RSLoRALinear(lin, r, al).scale - al / math.sqrt(r)) < 1e-12
    out["rslora_scale"] = "alpha/sqrt(r) verified"
    out["adalora"] = t4_adalora(root, data_dir)
    out["nf4"] = t4_nf4()
    return out


def t4_adalora(root: str, data_dir: str) -> Dict:
    p1 = write_phase1(root, "M2")
    m, _ = build_rr_model(p1, 13)
    cfg = base_cfg("adalora", "M2", p1, data_dir)
    t = get_technique("adalora")
    t.apply(m, cfg, log=lambda *x: None)
    T = 200
    bs = [t.budget(s, T) for s in range(T)]
    assert bs[0] == t.b0 and bs[-1] == t.bT
    assert all(x >= y for x, y in zip(bs, bs[1:])), "budget must not increase"
    g = torch.Generator().manual_seed(0)
    for step in range(T):
        for _, mod in t.modules_:
            for p in (mod.ada_P, mod.ada_E, mod.ada_Q):
                p.grad = torch.randn(p.shape, generator=g)
            mod.ada_E.data += 0.01 * torch.randn(mod.ada_E.shape, generator=g)
        t.after_backward(m, step, T)
    rep = t.final_report(m)
    assert rep["adalora_budget_final"] == t.bT, rep["adalora_budget_final"]
    x = torch.randn(2, 5, m.trunk.hidden_size)
    m.eval()
    for _, mod in t.modules_[:3]:               # pruned singular values contribute nothing
        y = mod(x[..., :mod.base.in_features]) if mod.base.in_features == x.shape[-1] else None
        if y is not None:
            with torch.no_grad():
                e = mod.ada_E.clone()
                mod.ada_E.mul_(mod.ada_mask)
                y2 = mod(x)
                mod.ada_E.copy_(e)
            assert torch.allclose(y, y2, atol=1e-6)
    return {"b0": t.b0, "bT": t.bT, "final": rep["adalora_budget_final"],
            "per_layer": rep["adalora_rank_per_layer"]}


def t4_nf4() -> Dict:
    code = nf4_codebook_from_normal()
    assert torch.allclose(code, torch.tensor(NF4_CODE), atol=1e-6), (code, NF4_CODE)
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 256)
    w = lin.weight.detach()
    packed, absmax, n = quantize_nf4(w, 64)
    deq = dequantize_nf4(packed, absmax, n, 64, torch.tensor(NF4_CODE)).view_as(w)
    half_gap = float(torch.diff(torch.tensor(NF4_CODE)).max()) / 2
    bound = absmax.repeat_interleave(64)[:n].view_as(w) * (half_gap + 1e-6)
    assert torch.all((w - deq).abs() <= bound), "NF4 round-trip error above the bound"
    rel = float((w - deq).norm() / w.norm())
    assert rel < 0.12, rel
    q = NF4Linear(lin, 64, True)
    rel_dq = float((q.dequantize() - w).norm() / w.norm())
    assert rel_dq < 0.13, rel_dq
    fp16 = 2 * w.numel()
    assert q.storage_bytes() * 3 < fp16, (q.storage_bytes(), fp16)
    x = torch.randn(3, 512)
    assert torch.allclose(q(x), torch.nn.functional.linear(x, q.dequantize(), lin.bias),
                          atol=1e-5)
    return {"relative_error": rel, "relative_error_double_quant": rel_dq,
            "bytes_fp16": fp16, "bytes_nf4_dq": q.storage_bytes(),
            "ratio": fp16 / q.storage_bytes()}


# =============================================================================================
# 5. prefix / prompt visibility
# =============================================================================================
def visibility(m, b, param: torch.Tensor, positions: List[int]) -> List[int]:
    """Positions whose output does NOT depend on `param` (must be empty)."""
    h = m.trunk(b["input_ids"], b["attention_mask"])
    # a fixed random projection (the plain sum of a LayerNorm output is constant)
    r = torch.randn(h.shape[-1], generator=torch.Generator().manual_seed(3)).to(h.device)
    blind = []
    for t in positions:
        g, = torch.autograd.grad((h[0, t].float() * r).sum(), param, retain_graph=True,
                                 allow_unused=True)
        if g is None or float(g.abs().sum()) == 0:
            blind.append(t)
    return blind


def t5_visibility(data_dir: str, root: str) -> Dict:
    out = {}
    x = torch.randint(3, V_TINY, (1, 24))
    b = {"input_ids": x, "attention_mask": torch.ones_like(x)}
    for mid in MODELS:
        p1 = write_phase1(root, mid)
        for attn in ("causal", "bidir"):
            for tech in ("prefix", "prompt"):
                m, _ = build_rr_model(p1, 13, attn)
                cfg = base_cfg(tech, mid, p1, data_dir, prefix_len=4, prompt_len=4,
                               prefix_reparam=False)
                get_technique(tech).apply(m, cfg, log=lambda *a: None)
                m.eval()
                param = m.trunk.prefix.prefix_kv if tech == "prefix" else m.trunk.prompt
                blind = visibility(m, b, param, list(range(24)))
                assert not blind, f"{mid}/{attn}/{tech}: positions {blind} cannot see it"
                out[f"{mid}/{attn}/{tech}"] = "visible to all 24 queries"
    return out


# =============================================================================================
# 6 + 7. overfit and kill/resume (subprocesses running rr_train.py)
# =============================================================================================
def _run(cmd, cwd=HERE, timeout=1200):
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="",
               PHASE3_INPUT_ROOTS=os.environ.get("PHASE3_INPUT_ROOTS", "/nonexistent"))
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def t6_overfit(data_dir: str, root: str) -> Dict:
    p1 = write_phase1(root, "M2")
    run = os.path.join(root, "run_overfit")
    cfg = base_cfg("lorma", "M2", p1, data_dir)
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    code, out = _run([sys.executable, "rr_train.py", "--run-dir", run, "--mode", "overfit",
                      "--max-steps", "300"])
    ov = json.load(open(os.path.join(run, "logs", "overfit.json"))) if code == 0 else None
    assert ov is not None, out[-2000:]
    assert ov["passed"] and ov["n_sentences"] == 32, ov
    return {"train_accuracy": ov["train_accuracy"], "steps": ov["global_step"]}


def t7_resume(data_dir: str, root: str) -> Dict:
    p1 = write_phase1(root, "M4")
    run = os.path.join(root, "run_resume")
    cfg = base_cfg("lora", "M4", p1, data_dir, epochs=2, effective_batch=2, micro_batch=2,
                   watchdog_min_f1=-1.0)
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    code, out = _run([sys.executable, "rr_train.py", "--run-dir", run, "--mode", "train",
                      "--kill-at-step", "3"])
    assert code == 137, f"expected the simulated kill (137), got {code}: {out[-1500:]}"
    assert os.path.exists(os.path.join(run, "checkpoints", "last.pt"))
    code, out = _run([sys.executable, "rr_train.py", "--run-dir", run, "--mode", "train"])
    assert code == 0, out[-2000:]
    rj = json.load(open(os.path.join(run, "logs", "resume.json")))
    s = json.load(open(os.path.join(run, "logs", "train_summary.json")))
    assert rj["resumed_at_step"] == 3, rj
    assert s["resumed_from_step"] == 3 and s["global_step"] > 3
    assert s["last_loss"] is not None and math.isfinite(s["last_loss"])
    return {"resumed_at_step": 3, "final_step": s["global_step"], "last_loss": s["last_loss"]}


def t_full_size(root: str, data_dir: str) -> Dict:
    """Trainable % of every technique on the REAL-size architectures (random weights)."""
    out = {}
    for mid in MODELS:
        p1 = write_phase1(root, mid, FULL)
        m0, _ = build_rr_model(p1, 13)
        for tech in PEFT:
            m = copy.deepcopy(m0)
            cfg = base_cfg(tech, mid, p1, data_dir)
            get_technique(tech).apply(m, cfg, log=lambda *a: None)
            pct = model_size(m)["trainable_pct"]
            assert pct <= MAX_TRAINABLE_PCT[tech], f"{mid}/{tech}: {pct:.2f}%"
            out[f"{mid}/{tech}"] = round(pct, 4)
            del m
        del m0
    return out


# =============================================================================================
# job checks on the REAL model (called by rr_train --mode smoke)
# =============================================================================================
def job_checks(cfg: Dict, meta: Dict, docs: Dict, device) -> Dict:
    res: Dict = {"checks": {}, "failed": []}

    def run(name: str, fn: Callable) -> None:
        try:
            res["checks"][name] = fn()
        except Exception as e:
            res["failed"].append(name)
            res["checks"][name] = f"FAILED: {e}\n{traceback.format_exc()[-1500:]}"

    def outputs():
        keys = [k for k in meta["split"]["dev"] + meta["split"]["test"]
                if docs[k]["domain"] in set(cfg["domains"])]
        wins = build_windows(docs, keys, cfg, meta, "infer")
        slot = [np.zeros((len(w["markers"]), cfg["n_labels"]), np.float32) for w in wins]
        comb = combine_slot_scores(wins, slot, {k: len(docs[k]["ids"]) for k in keys})
        assert all(comb[k].shape[0] == len(docs[k]["text"]) for k in keys)
        return {"docs": len(keys), "windows": len(wins)}
    run("output_lengths", outputs)
    tech = cfg["technique"]
    if tech in PEFT:
        def peft():
            m0, _ = build_rr_model(cfg["phase1_dir"], cfg["n_labels"], cfg["attn_mode"],
                                   cfg["head"], cfg.get("layer_ids"), cfg.get("dropout", 0.1),
                                   cfg.get("pool", "marker"))
            m0.to(device).eval()
            b = _batch(meta, docs, 2, device, cfg["attn_mode"], cfg["n_virtual"], 512)
            real_size = model_size(m0)["total_params"] >= SPEC["rules"]["foundation_params"][0]
            r = peft_checks(m0, cfg, b, tech, real_size, 1e-5, 1e-3)
            r["trainable_pct_checked"] = real_size
            del m0
            return r
        run("peft_zero_merge_pct", peft)
        if tech == "prefix" or tech == "prompt":
            def vis():
                m, _ = build_rr_model(cfg["phase1_dir"], cfg["n_labels"], cfg["attn_mode"])
                get_technique(tech).apply(m, dict(cfg, prefix_reparam=False), log=lambda *a: None)
                m.to(device).eval()
                b = _batch(meta, docs, 1, device, cfg["attn_mode"], cfg["n_virtual"])
                T = int(b["attention_mask"][0].sum())
                pos = sorted({0, 1, T // 2, T - 1} | ({min(T - 1, 600)} if T > 600 else set()))
                param = m.trunk.prefix.prefix_kv if tech == "prefix" else m.trunk.prompt
                blind = visibility(m, b, param, pos)
                assert not blind, f"positions {blind} cannot see the {tech}"
                return {"positions": pos}
            run("virtual_token_visibility", vis)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    res["ok"] = not res["failed"]
    return res


# =============================================================================================
def run_all(data_dir: str = None, full_size: bool = False, log=print) -> Dict:
    res = {"tests": {}, "failed": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with tempfile.TemporaryDirectory() as root:
        syn = synthetic_cache(os.path.join(root, "data"))
        real = data_dir if data_dir and os.path.exists(os.path.join(data_dir, "meta.json")) \
            else None
        tests = [("1_macro_f1", lambda: t1_metrics()),
                 ("2_window_packing", lambda: t2_packing()),
                 ("3_output_lengths_and_validator", lambda: t3_outputs(real or syn)),
                 ("4_peft_techniques", lambda: t4_peft(syn, root)),
                 ("5_prefix_prompt_visibility", lambda: t5_visibility(syn, root)),
                 ("6_overfit_32", lambda: t6_overfit(syn, root)),
                 ("7_kill_resume", lambda: t7_resume(syn, root))]
        if full_size:
            tests.append(("8_full_size_trainable_pct", lambda: t_full_size(root, syn)))
        for name, fn in tests:
            t0 = time.time()
            try:
                res["tests"][name] = {"result": fn(), "seconds": round(time.time() - t0, 1)}
                log(f"   PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:
                res["failed"].append(name)
                res["tests"][name] = {"error": str(e), "trace": traceback.format_exc()[-3000:]}
                log(f"   FAIL {name}: {e}")
    res["ok"] = not res["failed"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--full-size", action="store_true")
    a = ap.parse_args()
    res = run_all(a.data_dir, a.full_size)
    if a.out:
        write_json(a.out, res)
    print(json.dumps({"ok": res["ok"], "failed": res["failed"]}))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
