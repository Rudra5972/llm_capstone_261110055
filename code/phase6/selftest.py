"""
selftest.py -- Phase-6 CPU tests with tiny random-initialised models of the four Phase-1
architectures and a tiny BPE trained on synthetic legal-ish text (no real weights or data).

    python selftest.py [--out SELFTEST.json]

  1 metrics     ROUGE-1/2/L vs hand values, bit-parallel LCS == DP LCS, token-exact sentence
                spans, word caps, submission validator (good + 5 kinds of bad zips)
  2 data        loader (list / dict-keyed / wrapped), leak guard drops a planted test copy,
                deterministic held-out dev split disjoint from train, token cache
  3 layout      headtail + dancer: every sequence <= ctx, EOS start, separator, targets = the
                aligned summary sentences (all sentences used exactly once), labels only on targets
  4 model       for all 4 architectures x 9 techniques: trainable set, identity at init
                (LoRMA / LoRA / AdaLoRA exact), SelectHead logits == full logits at the targets,
                gradients reach every trainable tensor, KV-cache greedy == no-cache greedy (incl.
                prefix / prompt virtual tokens), left-padded batch == one-by-one decoding,
                beam search runs, NF4 dequantisation error small
  5 overfit     8 sequences: summary-token loss falls below 25% of its start (subprocess)
  6 resume      train -> kill (137) at step 3 -> resume at step 3 -> DONE + valid submission
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


def write_dataset(root: str, n_train=40, n_test=8, seed=0, shape="list", leak: int = 0,
                  long_summ=False) -> str:
    """Phase_6_Summ/{Train/train.json, Test/test.json}; `leak` test documents planted in train."""
    rng = random.Random(seed)
    base = os.path.join(root, "Phase_6_Summ")
    os.makedirs(os.path.join(base, "Train"), exist_ok=True)
    os.makedirs(os.path.join(base, "Test"), exist_ok=True)
    splits = {}
    for s, n in (("train", n_train), ("test", n_test)):
        recs = []
        for i in range(n):
            lab = sorted(rng.sample(range(1, 7), rng.choice([1, 2])))
            doc = synth_text(rng, rng.choice([60, 200, 600]), lab)
            k = 6 if long_summ else 2
            summ = " ".join(f"The court held that section {x} applies and the appeal was "
                            f"{'allowed' if x % 2 else 'dismissed'}." for x in lab * k)
            r = {"id": f"{s}_{i:04d}", "document": doc}
            if s == "train":
                r["summary"] = summ
            recs.append(r)
        splits[s] = recs
    for k in range(leak):
        t = dict(splits["test"][k])
        t["id"] = f"train_leak_{k}"
        t["summary"] = "The court held that the appeal was dismissed."
        splits["train"].append(t)
    for s, rel in (("train", "Train/train.json"), ("test", "Test/test.json")):
        obj = splits[s]
        if shape == "dict":
            obj = {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in obj}
        elif shape == "wrapped":
            obj = {"data": obj}
        with open(os.path.join(base, rel), "w") as f:
            json.dump(obj, f)
    return base


def _data(root, tok_dir, name, **kw):
    """write a dataset, load it through the real loader, build the cache -> (cache, path)."""
    from summ_data import build_cache, load_all, load_cache
    base = write_dataset(os.path.join(root, name), **kw)
    os.environ["PHASE6_INPUT_ROOTS"] = os.path.dirname(base)
    data, _sch, info = load_all([os.path.dirname(base)], 0.2, 50, log=lambda *a: None)
    p = build_cache(data, tok_dir, os.path.join(root, name + "_cache"), name, 1,
                    log=lambda *a: None)
    return load_cache(p), p, data, info


# =============================================================================================
def t1_metrics(tok_dir: str) -> Dict:
    from gen_ml import cap_words, lcs_len, rouge_l, rouge_n, sentence_spans
    from bpe_tokenizer import BPETokenizer
    r = rouge_n("the cat sat on the mat", "the cat lay on the mat", 1)
    assert abs(r["f"] - 5 / 6) < 1e-9, r
    r2 = rouge_n("the cat sat on the mat", "the cat lay on the mat", 2)
    assert abs(r2["f"] - 3 / 5) < 1e-9, r2
    assert abs(rouge_l("a b c d e", "a x c y e")["f"] - 3 / 5) < 1e-9
    rng = random.Random(3)

    def dp(a, b):
        M = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                M[i + 1][j + 1] = M[i][j] + 1 if a[i] == b[j] else max(M[i][j + 1], M[i + 1][j])
        return M[-1][-1]
    for _ in range(200):
        a = [rng.choice("abcde") for _ in range(rng.randint(0, 40))]
        b = [rng.choice("abcde") for _ in range(rng.randint(0, 40))]
        assert lcs_len(a, b) == dp(a, b), (a, b)
    tok = BPETokenizer.from_pretrained(tok_dir)
    for k in range(30):
        t = synth_text(rng, 80, [k % 6 + 1]) + " Held: appeal allowed! Costs? None; order:  done."
        sp = sentence_spans(t)
        assert "".join(sp) == t
        assert sum((tok.encode(x) for x in sp), []) == tok.encode(t), "span tokenisation"
    s = "One two three. Four five six. Seven eight nine."
    assert cap_words(s, 6) == "One two three. Four five six." and cap_words(s, 2) == "One two three."
    # validator
    import zipfile
    from validate_submission import validate
    d = tempfile.mkdtemp()

    def mk(name, obj, entry="prediction.json"):
        p = os.path.join(d, name)
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(entry, json.dumps(obj))
        return p
    good = [{"id": "a", "summary": "x y"}, {"id": "b", "summary": "z"}]
    assert validate(mk("g.zip", good), ["a", "b"])["ok"]
    bad = [mk("b1.zip", good[:1]), mk("b2.zip", good + [good[0]]),
           mk("b3.zip", [{"id": "a", "summary": ""}, good[1]]), mk("b4.zip", good, "x/prediction.json"),
           mk("b5.zip", {"a": "x", "b": "y"})]
    assert not any(validate(p, ["a", "b"])["ok"] for p in bad)
    return {"rouge_lcs_spans_validator": "ok"}


def t2_data(root: str, tok_dir: str) -> Dict:
    from summ_data import load_all, split_dev
    out = {}
    for shape in ("list", "dict", "wrapped"):
        base = write_dataset(os.path.join(root, f"ds_{shape}"), 30, 6, shape=shape, leak=2)
        data, sch, info = load_all([os.path.dirname(base)], 0.2, 50, log=lambda *a: None)
        assert info["leak_guard"]["dropped_from_train"] == 2, info["leak_guard"]
        assert sch["train"]["summary_key"] == "summary" and sch["train"]["text_key"] == "document"
        tr, dv, te = ({d["id"] for d in data[s]} for s in ("train", "dev", "test"))
        assert not (tr & dv) and not (tr & te) and len(dv) >= 1 and len(te) == 6
        out[shape] = {"train": len(tr), "dev": len(dv)}
    a = split_dev(data["train"] + data["dev"], 0.2, 50)
    b = split_dev(list(reversed(data["train"] + data["dev"])), 0.2, 50)
    assert [d["id"] for d in a[1]] and sorted(d["id"] for d in a[1]) == sorted(d["id"] for d in b[1])
    cache, _p, _d, _i = _data(root, tok_dir, "ds_cache", n_train=20, n_test=4)
    r = cache["train"][0]
    assert sum(len(s) for s in r["sents"]) > 0 and len(r["doc"]) == r["n_tokens"]
    return out


def t3_layout(root: str, tok_dir: str) -> Dict:
    from bpe_tokenizer import BPETokenizer
    from summ_data import Layout, summary_stats
    from summ_model import collate_train
    tok = BPETokenizer.from_pretrained(tok_dir)
    cache, _p, _d, _i = _data(root, tok_dir, "ds_layout", n_train=24, n_test=4, long_summ=True)
    st = summary_stats(cache)
    res = {}
    for rep, extra in (("headtail", {}), ("head", {}), ("dancer", {"chunk_tokens": 120}),
                       ("auto", {"auto_long": 10, "chunk_tokens": 120})):
        lay = Layout(tok, {"doc_repr": rep, **extra}, st, 0)
        n = 0
        for rec in cache["train"]:
            ex = lay.examples(rec)
            allt = []
            for p, t in ex:
                assert p[0] == 0 and len(p) + len(t) <= lay.ctx
                allt.append(t[:-1] if len(t) and t[-1] == 0 else t)
                n += 1
            full = np.concatenate([np.asarray(s) for s in rec["sents"]])
            if lay.repr == "dancer" and all(len(t) < lay.S for _, t in ex):
                got = sorted(np.concatenate(allt).tolist())
                assert got == sorted(full.tolist()), "dancer must use every summary token once"
        b = collate_train(ex[:2])
        for i, (p, t) in enumerate(ex[:2]):
            assert int((b["labels"][i] != -100).sum()) == len(t)
        res[rep] = {"repr": lay.repr, "seqs": n}
    assert res["auto"]["repr"] == "dancer" and res["dancer"]["seqs"] > res["headtail"]["seqs"]
    return res


def t4_model(root: str) -> Dict:
    from summ_model import (NF4Linear, apply_summ_technique, build_summ_model, collate_prompts,
                            decode_batch, generate_many)
    import contextlib
    techs = ["lorma", "lora", "adalora", "adapters", "pfeiffer", "qlora", "prefix", "prompt",
             "full_ft"]
    res = {}
    rng = np.random.default_rng(0)
    for mid in ("M1", "M2", "M3", "M4"):
        p1 = write_phase1(root, mid, seed=3)
        for tech in techs:
            torch.manual_seed(0)
            m, _ = build_summ_model(p1)
            base = copy.deepcopy(m).eval()
            cfg = {"lr": 1e-3, "n_virtual": 4}
            groups, hook = apply_summ_technique(m, tech, cfg, log=lambda *a: None,
                                                sample_ids=list(range(5, 40)))
            m.eval()
            ids = torch.from_numpy(rng.integers(3, 500, (2, 30))).long()
            mask = torch.ones_like(ids)
            mask[1, 25:] = 0
            lab = ids.clone()
            lab[:, :12] = -100
            lab[mask == 0] = -100
            with torch.no_grad():
                a, tg = m.lm_logits_at_targets(ids, mask, lab)
                full = m(ids, mask).logits
            sel = torch.zeros_like(lab, dtype=torch.bool)
            sel[:, :-1] = lab[:, 1:] != -100
            assert torch.allclose(a, full[sel], atol=1e-5), "SelectHead"
            if tech in ("lorma", "lora", "adalora"):
                with torch.no_grad():
                    b0, _ = base.lm_logits_at_targets(ids, mask, lab)
                assert float((a - b0).abs().max()) < 1e-4, f"{mid}/{tech} not identity at init"
            if tech == "qlora":
                lin = [x for x in m.modules() if isinstance(x, NF4Linear)][0]
                assert lin.q_packed.dtype == torch.uint8
            m.train()
            lt, tg = m.lm_logits_at_targets(ids, mask, lab)
            torch.nn.functional.cross_entropy(lt, tg).backward()
            tr = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
            assert tr and sum(int(p.grad is not None and p.grad.abs().sum() > 0) for _, p in tr) \
                >= (0.3 if tech == "adalora" else 0.5) * len(tr), f"{mid}/{tech}: gradients"
            m.eval()
            # KV-cache greedy == no-cache greedy
            pr = rng.integers(3, 500, 9).astype(np.int64)
            x, xm = collate_prompts([pr], "cpu")
            nul = contextlib.nullcontext
            with torch.no_grad():
                g = decode_batch(m, x, xm, 6, 6, 1, 0)[0]
                seq = list(pr)
                ref = []
                for _ in range(6):
                    t = torch.tensor([seq])
                    lo = m(t, torch.ones_like(t)).logits[0, -1].clone()
                    lo[[0, 1]] = float("-inf")
                    nx = int(lo.argmax())
                    ref.append(nx)
                    seq.append(nx)
            assert g == ref, f"{mid}/{tech}: cached {g} != uncached {ref}"
            if tech in ("lorma", "prefix", "prompt"):
                p2 = rng.integers(3, 500, 5).astype(np.int64)
                both = generate_many(m, [pr, p2], "cpu", nul, 5, 5, 1, 0, 2)
                one = [generate_many(m, [q], "cpu", nul, 5, 5, 1, 0, 1)[0] for q in (pr, p2)]
                assert both == one, f"{mid}/{tech}: left padding changes the output"
                bm = generate_many(m, [pr, p2], "cpu", nul, 5, 0, 3, 3, 2)
                assert len(bm) == 2
            res[f"{mid}/{tech}"] = len(tr)
    W = torch.randn(64, 128)
    lin = torch.nn.Linear(128, 64)
    lin.weight.data.copy_(W)
    q = NF4Linear(lin)
    rel = float((q.weight - W).norm() / W.norm())
    assert rel < 0.15, rel
    res["nf4_rel_error"] = rel
    return res


def _job_cfg(root, tok_dir, cache_path, tech, run, **kw):
    p1 = write_phase1(root, "M2", tok_dir=tok_dir)
    cfg = {"experiment": "T", "job": os.path.basename(run), "model": "M2", "phase1_dir": p1,
           "bpe_dir": tok_dir, "technique": tech, "doc_repr": "dancer", "chunk_tokens": 120,
           "lr": 3e-3, "head_lr": 3e-3, "effective_batch": 8, "micro_tokens": 4096,
           "gen_batch": 8, "dev_gen": 8, "seed": 1, "budget_s": 900, "cache_path": cache_path,
           "bertscore": False, "skip_job_checks": False, "n_virtual": 4, "epochs_max": 2}
    cfg.update(kw)
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    return cfg


def t5_overfit(root: str, tok_dir: str) -> Dict:
    _c, path, _d, _i = _data(root, tok_dir, "ds_job", n_train=30, n_test=6, long_summ=True)
    run = os.path.join(root, "run_overfit")
    _job_cfg(root, tok_dir, path, "full_ft", run)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, os.path.join(HERE, "summ_train.py"), "--run-dir", run,
                        "--mode", "overfit"], env=env, capture_output=True, text=True, timeout=900)
    f = os.path.join(run, "logs", "overfit.json")
    assert os.path.exists(f), (p.stdout + p.stderr)[-2500:]
    ov = json.load(open(f))
    assert ov["passed"], ov
    return ov


def t6_resume(root: str, tok_dir: str) -> Dict:
    from validate_submission import validate
    path = json.load(open(os.path.join(root, "run_overfit", "logs", "run_config.json")))["cache_path"]
    run = os.path.join(root, "run_resume")
    cfg = _job_cfg(root, tok_dir, path, "lorma", run, kill_at_step=3, lr=1e-3)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(HERE, "summ_train.py"), "--run-dir", run]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 137, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert os.path.exists(os.path.join(run, "checkpoints", "last.pt"))
    cfg["kill_at_step"] = -1
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    log = open(os.path.join(run, "logs", "console.log")).read()
    assert p.returncode == 0, log[-2500:]
    assert json.load(open(os.path.join(run, "logs", "resume.json")))["resumed_at_step"] == 3
    assert os.path.exists(os.path.join(run, "DONE.json"))
    tids = [r["id"] for r in __import__("summ_data").load_cache(path)["test"]]
    v = validate(os.path.join(run, "submission", "submission.zip"), tids)
    assert v["ok"], v
    met = json.load(open(os.path.join(run, "metrics.json")))
    assert met["dev_rouge1"] is not None and met["score_metric"] == "rouge1_rougeL_mean"
    for f in ("dev/dev_predictions.jsonl", "dev/length_curve.csv", "dev/dev_report.json",
              "dev/length_buckets.csv"):
        assert os.path.exists(os.path.join(run, f)), f
    return {"resumed_at_step": 3, "dev_rouge1": met["dev_rouge1"], "test_words": v["mean_words"]}


def run_all(log=print) -> Dict:
    res = {"tests": {}, "failed": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with tempfile.TemporaryDirectory() as root:
        tok_dir = write_tokenizer(os.path.join(root, "tok"))
        tests = [("1_metrics", lambda: t1_metrics(tok_dir)),
                 ("2_data_leak_split_cache", lambda: t2_data(root, tok_dir)),
                 ("3_layout", lambda: t3_layout(root, tok_dir)),
                 ("4_model_techniques_decoding", lambda: t4_model(root)),
                 ("5_overfit", lambda: t5_overfit(root, tok_dir)),
                 ("6_kill_resume", lambda: t6_resume(root, tok_dir))]
        only = os.environ.get("SELFTEST_ONLY")
        for name, fn in tests:
            if only and not any(name.startswith(o) for o in only.split(",")):
                continue
            t0 = time.time()
            try:
                res["tests"][name] = {"result": fn(), "seconds": round(time.time() - t0, 1)}
                log(f"   PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:
                res["failed"].append(name)
                res["tests"][name] = {"error": str(e)[:2000],
                                      "trace": traceback.format_exc()[-3000:]}
                log(f"   FAIL {name}: {str(e)[:500]}\n{traceback.format_exc()[-1500:]}")
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
