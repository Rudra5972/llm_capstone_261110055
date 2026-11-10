"""
selftest.py -- CPU tests with tiny random-initialised models of the four Phase-1 architectures
and a tiny BPE trained on synthetic legal-ish text (no real weights or data needed).

    python selftest.py [--out SELFTEST.json]

  1 metrics      binary macro-F1 / P / R vs hand-computed values; tau scan + min-class guard
  2 loader       list / dict-keyed / wrapped containers, multi-label variants, ambiguity -> DataError;
                 leak guard drops planted train/dev copies of test cases (id and text)
  3 tokenisation head/tail slices tokenise exactly like the full document (prefix / suffix);
                 every representation: <= 512 (headtail <= 1024) tokens, EOS first, K chunks
  4 model        causal trunk == original LM (lm_head(trunk) == logits); bidir: a future token
                 changes earlier states, padding changes nothing; masked chunks get weight 0;
                 batched == single-document logits
  5 peft         every technique on every architecture: trainable set, identity at init
                 (LoRMA / LoRA / AdaLoRA exact), merged == unmerged, all tensors on one device
                 in BOTH orders (lesson L1, simulated with the `meta` device)
  6 overfit      32 documents reach loss < 0.1 (cjpe_train --mode overfit, subprocess)
  7 resume       checkpoint -> kill (137) -> resume from the same step -> DONE (subprocess)
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cjpe_common import MODELS, write_json  # noqa: E402

TINY = {
    "M1": ("GPT2StyleLM", {"vocab_size": 600, "n_positions": 1024, "n_embd": 32, "n_layer": 4,
                           "n_head": 4, "mlp_ratio": 4}),
    "M2": ("LlamaStyleLM", {"vocab_size": 600, "max_position_embeddings": 1024,
                            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                            "num_attention_heads": 4, "num_key_value_heads": 2,
                            "sliding_window": None}),
    "M3": ("LlamaStyleLM", {"vocab_size": 600, "max_position_embeddings": 1024,
                            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                            "num_attention_heads": 4, "num_key_value_heads": 2,
                            "sliding_window": 8}),
    "M4": ("BloomStyleLM", {"vocab_size": 600, "hidden_size": 32, "n_layer": 4, "n_head": 4,
                            "mlp_ratio": 4}),
}
WORDS = ("the court held that appellant respondent section act order petition dismissed "
         "allowed appeal judgment high supreme evidence witness learned counsel state "
         "accused conviction sentence bail facts issue reasons therefore accordingly").split()


# =============================================================================================
# fixtures
# =============================================================================================
def synth_text(rng: random.Random, n_words: int, label) -> str:
    """label: list of statute ids (Phase 5); an int is treated as [int]."""
    labs = label if isinstance(label, (list, tuple)) else [int(label)]
    words = []
    for k in range(n_words):
        w = rng.choice(WORDS)
        if rng.random() < 0.08:
            w = w.upper()
        words.append(w)
        if rng.random() < 0.1:
            words.append(rng.choice([",", ".", ";", "(1)", "12/2003", "\n", "  "]))
    cue = {1: "murder penal", 2: "cheating fraud", 3: "dowry cruelty", 4: "bail custody",
           5: "property land", 6: "contract breach"}
    tail = "".join(f" the accused was charged under section {x} for {cue.get(x, 'offence')} ."
                   for x in labs)
    return " ".join(words) + "\n\nORDER\n" + tail * 3


def write_tokenizer(d: str, seed: int = 0) -> str:
    from bpe_tokenizer import BPETokenizer, count_pretokens, train_bpe
    rng = random.Random(seed)
    texts = [synth_text(rng, 200, [i % 6 + 1]) for i in range(40)]
    merges, _ = train_bpe(count_pretokens(texts, log=lambda *a: None), 3 + 256 + 300,
                          log=lambda *a: None)
    tok = BPETokenizer(merges)
    tok.save_pretrained(d)
    return d


def write_phase1(root: str, mid: str, seed: int = 0, tok_dir: str = None) -> str:
    from phase1_models import MODEL_CLASSES
    cls_name, conf = TINY[mid]
    d = os.path.join(root, f"p1_{mid}", "final")
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(seed)
    cls, ccls = MODEL_CLASSES[cls_name]
    torch.save(cls(ccls(**conf)).state_dict(), os.path.join(d, "model.pt"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_class": cls_name, "arch": MODELS[mid]["arch"], "config": conf}, f)
    if tok_dir:
        import shutil
        for fn in os.listdir(tok_dir):
            shutil.copy2(os.path.join(tok_dir, fn), os.path.join(d, fn))
    return d


def write_dataset(root: str, n_train=48, n_dev=16, n_test=16, seed=0, shape="list",
                  label_key="labels", leak: int = 0) -> str:
    """Phase_5_LSI/{Train/train.json, Train/dev.json, Test/test.json}; `leak` test cases are
    planted into train (to exercise the leak guard)."""
    rng = random.Random(seed)
    base = os.path.join(root, "Phase_5_LSI")
    os.makedirs(os.path.join(base, "Train"), exist_ok=True)
    os.makedirs(os.path.join(base, "Test"), exist_ok=True)
    splits = {}
    for s, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        recs = []
        for i in range(n):
            lab = sorted(rng.sample(range(1, 7), rng.choice([1, 1, 2, 3])))
            nw = rng.choice([30, 120, 400, 900, 2500])
            r = {"id": f"{s}_{i:04d}", "text": synth_text(rng, nw, lab)}
            if s != "test":
                r[label_key] = lab
            recs.append(r)
        splits[s] = recs
    for k in range(leak):
        t = dict(splits["test"][k])
        t["id"] = f"train_leak_{k}"
        t[label_key] = [1]
        splits["train"].append(t)
    for s, rel in (("train", "Train/train.json"), ("dev", "Train/dev.json"),
                   ("test", "Test/test.json")):
        obj = splits[s]
        if shape == "dict":
            obj = {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in obj}
        elif shape == "wrapped":
            obj = {"data": obj}
        with open(os.path.join(base, rel), "w") as f:
            json.dump(obj, f)
    return base


# =============================================================================================
# 1 metrics
# =============================================================================================
def t1_metrics() -> Dict:
    from lsi_ml import apply_rule, choose_rule, macro, ml_scores, multihot
    G = multihot([[1, 2], [2], [3], [1]], {1: 0, 2: 1, 3: 2, 4: 3}, 4)
    P = multihot([[1], [2, 3], [3], [4]], {1: 0, 2: 1, 3: 2, 4: 3}, 4)
    sc = ml_scores(G, P)
    # label1: tp1 fn1 -> F=2/3 ; label2: tp1 fn1 -> 2/3 ; label3: tp1 fp1 -> 2/3 ; label4 no
    # dev support -> excluded from macro_f1, counted (F=0) in macro_f1_all
    assert abs(sc["macro_f1"] - 2 / 3) < 1e-12, sc
    assert abs(sc["macro_f1_all"] - 0.5) < 1e-12, sc
    assert abs(sc["subset_acc"] - 0.25) < 1e-12 and sc["tp"] == 3 and sc["fp"] == 2
    Pr = np.array([[0.1, 0.05], [0.9, 0.95], [0.6, 0.2]], np.float32)
    pr = apply_rule(Pr, np.array([0.5, 0.5], np.float32), 1)
    assert pr.sum(1).tolist() == [1, 1, 1] and pr[1].tolist() == [False, True]  # top-1, kmax
    rng = np.random.default_rng(0)
    Gb = (rng.random((200, 8)) < 0.2).astype(np.uint8)
    Prb = np.clip(0.5 * Gb + rng.normal(0.2, 0.15, Gb.shape), 0, 1).astype(np.float32)
    r = choose_rule(Prb, Gb, 4, 0)
    assert r["dev_macro_global"] >= macro(Gb, apply_rule(Prb, np.full(8, 0.5, np.float32), 4))
    assert r["name"] in ("global", "perlabel") and len(r["taus"]) == 8
    return {"hand_cases": 3, "rule": r["name"], "tau": r["tau_global"]}


# =============================================================================================
# 2 loader + leak guard
# =============================================================================================
def t2_loader(root: str) -> Dict:
    from cjpe_data import DataError, leak_guard, load_split, provenance
    out = {}
    for shape, lk in (("list", "labels"), ("dict", "statutes"), ("wrapped", "label")):
        base = write_dataset(os.path.join(root, f"ds_{shape}"), 10, 4, 4, shape=shape,
                             label_key=lk, leak=2)
        data, sch = {}, {}
        for s, rel in (("train", "Train/train.json"), ("dev", "Train/dev.json"),
                       ("test", "Test/test.json")):
            data[s], sch[s] = load_split(os.path.join(base, rel), s, log=lambda *a: None)
        assert all(isinstance(d["label"], list) and d["label"] for d in data["train"])
        assert all(d["label"] is None for d in data["test"])
        rep = leak_guard(data, log=lambda *a: None)
        assert rep["dropped_from_train"] == 2, rep
        assert not ({d["id"] for d in data["train"]} & {d["id"] for d in data["test"]})
        prov = provenance(sch, rep, len(data["train"]), "test")
        assert prov["train_cases_dropped_by_leak_guard"] == 2
        out[shape] = {"label_key": sch["train"]["label_key"], "dropped": 2}
    # string labels / bools
    p = os.path.join(root, "lab.json")
    json.dump([{"id": "a", "text": "x y", "labels": "7, 1"}, {"id": "b", "text": "y",
                                                               "labels": ["3", 3, 2]},
               {"id": "c", "text": "z", "labels": 5}], open(p, "w"))
    d, _ = load_split(p, "train", log=lambda *a: None)
    assert [x["label"] for x in d] == [[1, 7], [2, 3], [5]], [x["label"] for x in d]
    json.dump([{"id": "a", "text": "x", "labels": [True]}], open(p, "w"))
    try:
        load_split(p, "train", log=lambda *a: None)
        raise AssertionError("bool label accepted")
    except DataError:
        pass
    json.dump([{"id": "a", "body": "x", "label": 1}, {"id": "b", "title": "y", "label": 0}],
              open(p, "w"))
    try:
        load_split(p, "train", log=lambda *a: None)
        raise AssertionError("ambiguous text key accepted")
    except DataError:
        pass
    return out


# =============================================================================================
# 3 tokenisation
# =============================================================================================
def t3_tok(tok_dir: str) -> Dict:
    from bpe_tokenizer import BPETokenizer
    from cjpe_tok import BODY, doc_chunks, encode_doc
    tok = BPETokenizer.from_pretrained(tok_dir)
    rng = random.Random(3)
    n = 0
    for i in range(30):
        text = synth_text(rng, rng.choice([50, 800, 3000, 6000]), [i % 6 + 1])
        if i % 5 == 0:
            text = "   " + text + "  \n "
        full = tok.encode(text)
        for nh, nt in ((2 * BODY, 2 * BODY), (BODY, 4 * BODY), (8 * BODY, 0)):
            e = encode_doc(text, nh, nt, want_sections=True, tok=tok)
            if e["complete"]:
                assert e["head"].tolist() == full, "complete encode differs"
            else:
                assert e["head"].tolist() == full[:len(e["head"])], "head is not a prefix"
                if nt:
                    assert e["tail"].tolist() == full[len(full) - len(e["tail"]):], "tail"
                assert len(e["head"]) == nh and len(e["tail"]) == nt
            for rep_, K in (("hier", 4), ("hier", 1), ("first_k", 2), ("last_k", 3),
                            ("headtail", 1), ("sections", 2), ("spread", 4), ("cite", 4)):
                if rep_ in ("first_k",) and nh < K * BODY and not e["complete"]:
                    continue
                if rep_ in ("last_k", "hier") and nt < BODY and not e["complete"]:
                    continue
                if rep_ == "headtail" and (nh < 383 or nt < 639) and not e["complete"]:
                    continue
                ch = doc_chunks(e, rep_, K)
                assert ch and all(c[0] == 0 for c in ch)
                lim = 1024 if rep_ == "headtail" else 512
                assert all(len(c) <= lim for c in ch), rep_
                if rep_ not in ("sections", "cite"):
                    assert len(ch) <= K or rep_ == "headtail"
                n += 1
            if not e["complete"] and nt:
                ch = doc_chunks(e, "hier", 1)
                assert ch[0][1:].tolist() == full[-BODY:], "K=1 must be the LAST chunk"
    return {"representation_checks": n}


# =============================================================================================
# 4 model
# =============================================================================================
def t4_model(root: str) -> Dict:
    from cjpe_models import build_doc_model, collate_docs, forward_batch
    from ner_models import load_phase1_lm
    out = {}
    for mid in MODELS:
        p1 = write_phase1(root, mid)
        lm, _ = load_phase1_lm(p1)
        lm.eval()
        m, _ = build_doc_model(p1, "causal")
        m.eval()
        x = torch.randint(3, 600, (2, 20))
        with torch.no_grad():
            ref = lm(x).logits if hasattr(lm(x), "logits") else lm(x)[0]
            h = m.trunk(x, torch.ones_like(x))
            got = lm.lm_head(h)
        assert torch.allclose(ref, got, atol=1e-4), f"{mid}: causal trunk != LM"
        mb, _ = build_doc_model(p1, "bidir")
        mb.eval()
        with torch.no_grad():
            a = mb.trunk(x[:1], torch.ones_like(x[:1]))
            y = x[:1].clone()
            y[0, -1] = (y[0, -1] + 1) % 590 + 3
            b = mb.trunk(y, torch.ones_like(y))
            assert (a[0, 0] - b[0, 0]).abs().max() > 1e-6, f"{mid}: bidir not bidirectional"
            pad = torch.cat([x[:1], torch.full((1, 5), 1)], 1)
            msk = torch.cat([torch.ones(1, 20, dtype=torch.long), torch.zeros(1, 5,
                                                                             dtype=torch.long)], 1)
            c = mb.trunk(pad, msk)[:, :20]
            pad2 = pad.clone()
            pad2[0, 22] = 77
            c2 = mb.trunk(pad2, msk)[:, :20]
            assert torch.allclose(c, c2, atol=1e-5), f"{mid}: padding leaks"
        rng = np.random.default_rng(0)
        docs = [[np.concatenate([[0], rng.integers(3, 600, L)]) for L in Ls]
                for Ls in ([30, 12, 7], [25], [40, 40])]
        with torch.no_grad():
            bt = collate_docs(docs)
            lg, w = forward_batch(mb, bt)
            assert float(w[1, 1:].abs().sum()) == 0.0 and float(w[2, 2:].abs().sum()) == 0.0
            for k, d in enumerate(docs):
                l1, _ = forward_batch(mb, collate_docs([d]))
                assert torch.allclose(l1[0], lg[k], atol=1e-4), f"{mid}: batched != single"
        out[mid] = "ok"
    return out


# =============================================================================================
# 5 peft
# =============================================================================================
def t5_peft(root: str) -> Dict:
    from cjpe_models import assert_one_device, build_doc_model, model_size
    from cjpe_peft import apply_technique, merge_model
    out = {}
    x = torch.randint(3, 600, (2, 16))
    am = torch.ones_like(x)
    for mid in MODELS:
        p1 = write_phase1(root, mid)
        for tech in ("full_ft", "lorma", "lora", "adalora", "adapters", "partial_ft", "bitfit",
                     "kd"):
            if tech == "bitfit" and mid in ("M2", "M3"):
                continue
            m0, _ = build_doc_model(p1, "bidir", [0, 2] if tech == "kd" else None)
            m0.eval()
            m = copy.deepcopy(m0)
            cfg = {"lr": 1e-3, "head_lr": 1e-3, "partial_top_k": 2}
            groups, t = apply_technique(m, tech, cfg, log=lambda *a: None)
            m.eval()
            s = model_size(m)
            assert s["trainable_params"] > 0
            names = [n for n, p in m.named_parameters() if p.requires_grad]
            assert any(n.startswith("classifier.") for n in names)
            if tech in ("lorma", "lora", "adalora"):
                assert all(n.startswith("classifier.") or any(t_ in n for t_ in (
                    "lora_", "lorma_", "ada_")) for n in names), names[:5]
                with torch.no_grad():
                    a0 = m0.chunk_vectors(x, am)
                    a1 = m.chunk_vectors(x, am)
                assert torch.allclose(a0, a1, atol=1e-6), f"{mid}/{tech}: not identity at init"
                mm = copy.deepcopy(m)
                g = torch.Generator().manual_seed(1)
                with torch.no_grad():
                    for n, p in mm.named_parameters():
                        if any(t_ in n for t_ in ("lora_B", "lorma_B", "ada_E")):
                            p.add_(torch.randn(p.shape, generator=g) * 0.05)
                    u = mm.chunk_vectors(x, am)
                    k = merge_model(mm)
                    v = mm.chunk_vectors(x, am)
                assert k > 0 and torch.allclose(u, v, atol=1e-4), f"{mid}/{tech}: merge"
            # lesson L1: base on another device FIRST, then apply -> everything follows it
            md = copy.deepcopy(m0).to("meta")
            apply_technique(md, tech, cfg, log=lambda *a: None)
            assert_one_device(md, "meta")
            out[f"{mid}/{tech}"] = round(s["trainable_pct"], 3)
    return out


# =============================================================================================
# 6 overfit (subprocess, the real trainer)
# =============================================================================================
def t6_overfit(root: str, tok_dir: str) -> Dict:
    from bpe_tokenizer import BPETokenizer
    from cjpe_data import load_split
    from cjpe_tok import build_cache
    base = write_dataset(os.path.join(root, "ds_overfit"), 40, 8, 8, seed=5)
    data = {s: load_split(os.path.join(base, rel), s, log=lambda *a: None)[0]
            for s, rel in (("train", "Train/train.json"), ("dev", "Train/dev.json"),
                           ("test", "Test/test.json"))}
    cache = build_cache(data, tok_dir, os.path.join(root, "cache_of"), "of", 2 * 511, 2 * 511,
                        False, 1, log=lambda *a: None)
    p1 = write_phase1(root, "M2", tok_dir=tok_dir)
    run = os.path.join(root, "run_overfit")
    cfg = {"experiment": "T", "job": "overfit", "model": "M2", "phase1_dir": p1,
           "technique": "full_ft", "attn_mode": "bidir", "doc_repr": "hier", "K": 2,
           "class_weight": "none", "lr": 3e-3, "head_lr": 3e-3, "effective_batch": 8,
           "micro_chunks": 16, "infer_chunks": 32, "seed": 1, "budget_s": 600,
           "cache_path": cache, "skip_job_checks": False,
           "label_ids": sorted({x for d in data["train"] for x in d["label"]}), "kmax": 3}
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, os.path.join(HERE, "cjpe_train.py"), "--run-dir", run,
                        "--mode", "overfit"], env=env, capture_output=True, text=True,
                       timeout=900)
    ov = json.load(open(os.path.join(run, "logs", "overfit.json"))) if os.path.exists(
        os.path.join(run, "logs", "overfit.json")) else None
    assert ov is not None, (p.stdout + p.stderr)[-2000:]
    assert ov["passed"], ov
    return ov


def t7_resume(root: str, tok_dir: str) -> Dict:
    """checkpoint -> kill (exit 137) -> resume reproduces the step counter and finishes DONE."""
    run_of = os.path.join(root, "run_overfit")
    cfg = json.load(open(os.path.join(run_of, "logs", "run_config.json")))
    run = os.path.join(root, "run_resume")
    cfg.update(job="resume", technique="lorma", lr=1e-3, budget_s=900, kill_at_step=3,
               epochs_max=2, effective_batch=8)
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(HERE, "cjpe_train.py"), "--run-dir", run]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 137, (p.returncode, (p.stdout + p.stderr)[-1500:])
    assert os.path.exists(os.path.join(run, "checkpoints", "last.pt"))
    cfg["kill_at_step"] = -1
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    rj = json.load(open(os.path.join(run, "logs", "resume.json")))
    assert rj["resumed_at_step"] == 3, rj
    st = json.load(open(os.path.join(run, "status.json")))
    # a 2-epoch LoRMA run on 8 synthetic test docs may legitimately give every doc the same
    # label set (flagged DEGENERATE); either way the job must finish and write a valid zip
    assert os.path.exists(os.path.join(run, "DONE.json")) or st.get("state") == "degenerate", st
    assert os.path.exists(os.path.join(run, "submission", "submission.zip"))
    assert not os.path.exists(os.path.join(run, "checkpoints", "last.pt"))
    return {"resumed_at_step": 3, "state": st.get("state")}


def run_all(log=print) -> Dict:
    res = {"tests": {}, "failed": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with tempfile.TemporaryDirectory() as root:
        tok_dir = write_tokenizer(os.path.join(root, "tok"))
        tests = [("1_metrics", t1_metrics), ("2_loader_leak_guard", lambda: t2_loader(root)),
                 ("3_tokenisation", lambda: t3_tok(tok_dir)), ("4_model", lambda: t4_model(root)),
                 ("5_peft_device", lambda: t5_peft(root)),
                 ("6_overfit_32", lambda: t6_overfit(root, tok_dir)),
                 ("7_kill_resume", lambda: t7_resume(root, tok_dir))]
        for name, fn in tests:
            t0 = time.time()
            try:
                res["tests"][name] = {"result": fn(), "seconds": round(time.time() - t0, 1)}
                log(f"   PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:
                res["failed"].append(name)
                res["tests"][name] = {"error": str(e)[:2000],
                                      "trace": traceback.format_exc()[-3000:]}
                log(f"   FAIL {name}: {str(e)[:500]}")
    res["ok"] = not res["failed"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run_all()
    if a.out:
        write_json(a.out, res)
    print(json.dumps({"ok": res["ok"], "failed": res["failed"]}))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
