"""
selftest.py -- Phase-7 CPU tests with tiny random-initialised models of the four Phase-1
architectures and a tiny BPE trained on synthetic legal-ish text (no real weights or data).

    python selftest.py [--out SELFTEST.json]

  1 metrics     Micro-P/R/F1@7 vs hand values (gold sets longer / shorter than 7), top-k order and
                ties, BM25 ranks an identical case first, fusion keeps masked entries masked,
                submission validator (good + 6 kinds of bad zips)
  2 data        loader (list / dict-keyed pools, zero-padded ids), relevant ids mapped into the
                split's OWN pool, self-retrieval masked, chunk cache bounds, BM25 tuned on dev
  3 model       for all 4 architectures x 8 techniques: trainable set, identity at init
                (LoRMA / LoRA / IA3 / full / partial exact), GradCache gradient == exact gradient,
                gradients reach the trainable tensors, padding invariance, virtual positions
                excluded from pooling, NF4 dequantisation error small
  4 overfit     4 pairs: contrastive loss falls below 30% of its start (subprocess)
  5 resume      train -> kill (137) at step 3 -> resume at step 3 -> DONE + valid submission
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


def write_dataset(root: str, n_q=(30, 10, 10), seed=0, shape="list") -> str:
    """Phase_7_PCR/{Train,Dev,Test}: topic-structured cases; relevant = same topic."""
    rng = random.Random(seed)
    base = os.path.join(root, "Phase_7_PCR")
    nid = [1000]

    def new_id():
        nid[0] += rng.randint(1, 9)
        return f"{nid[0]:010d}"

    def case(topic):
        sents = []
        for _ in range(rng.choice([3, 8, 20])):
            sents.append(synth_text(rng, rng.randint(8, 25), [topic % 6 + 1]).split("\n")[0]
                         + f" the doctrine of topic{topic} and rule {100 + topic} applied .")
        return sents
    for split, nq in zip(("train", "dev", "test"), n_q):
        os.makedirs(os.path.join(base, split.capitalize()), exist_ok=True)
        pool, queries = [], []
        for k in range(nq):
            t = rng.randint(0, 11)
            rel = [new_id() for _ in range(rng.randint(1, 4))]
            pool += [{"id": r, "text": case(t)} for r in rel]
            q = {"id": new_id(), "text": case(t)}
            if split != "test":
                q["relevant_candidates"] = rel if k % 5 else [r.lstrip("0") for r in rel]
            queries.append(q)
        pool += [{"id": new_id(), "text": case(rng.randint(12, 20))} for _ in range(2 * nq)]
        if split == "dev":                                    # a query inside its own pool
            pool.append({"id": queries[0]["id"], "text": queries[0]["text"]})
        rng.shuffle(pool)
        if shape == "dict":
            pool = {p["id"]: p["text"] for p in pool}
        for name, obj in ((f"{split}.json", queries), (f"{split}_candidates.json", pool)):
            with open(os.path.join(base, split.capitalize(), name), "w") as f:
                json.dump(obj, f)
    return base


def _data(root, tok_dir, name, **kw):
    from pcr_data import build_cache, load_all, load_cache
    base = write_dataset(os.path.join(root, name), **kw)
    data, sch, info = load_all([os.path.dirname(base)], log=lambda *a: None)
    p = build_cache(data, tok_dir, os.path.join(root, name + "_cache"), name, 48, 4, 1,
                    log=lambda *a: None)
    return load_cache(p), p, data, info


# =============================================================================================
def t1_metrics(tok_dir: str) -> Dict:
    import zipfile
    from pcr_ml import BM25, fuse, mask_self, micro_prf_at_k, topk
    from validate_submission import validate
    m = micro_prf_at_k([[1, 2, 3, 4, 5, 6, 7], [9, 8, 7, 6, 5, 4, 3]],
                       [{1, 2, 99}, set(range(3, 13))])
    # tp = 2 + 7 = 9; pred = 14; gold = 3 + 10 = 13
    assert m["tp"] == 9 and abs(m["micro_p"] - 9 / 14) < 1e-12 and abs(m["micro_r"] - 9 / 13) < 1e-12
    S = np.array([[0.1, 0.5, 0.5, 0.9, -np.inf]], np.float32)
    assert topk(S, 3).tolist() == [[3, 1, 2]]
    docs = ["the appellant was convicted under section 302 for murder",
            "contract breach damages awarded to plaintiff", "bail granted to accused under 437"]
    bm = BM25(docs, 1.2, 0.75, 2)
    assert bm.score(["contract breach damages awarded"]).argmax() == 1
    Sm = mask_self(np.ones((1, 3), np.float32), ["b"], ["a", "b", "c"])
    F = fuse(Sm, np.arange(3, dtype=np.float32)[None], 0.5)
    assert not np.isfinite(F[0, 1]) and np.isfinite(F[0, [0, 2]]).all()
    tmp = tempfile.mkdtemp()
    ids, pool = ["q1", "q2"], [f"c{i}" for i in range(10)]

    def mk(name, obj, entry="prediction.json"):
        p = os.path.join(tmp, name)
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(entry, json.dumps(obj))
        return validate(p, ids, pool)
    good = {"q1": pool[:7], "q2": pool[1:8]}
    assert mk("good.zip", good)["ok"]
    bad = {"short": {"q1": pool[:6], "q2": pool[1:8]},
           "dup": {"q1": pool[:6] + [pool[0]], "q2": pool[1:8]},
           "foreign": {"q1": pool[:6] + ["zz"], "q2": pool[1:8]},
           "missing": {"q1": pool[:7]},
           "list_format": [{"id": "q1", "candidates": pool[:7]}]}
    for k, v in bad.items():
        assert not mk(f"{k}.zip", v)["ok"], k
    assert not mk("entry.zip", good, "pred.json")["ok"]
    return {"micro": m, "validator_bad_cases": len(bad) + 1}


def t2_data(root: str, tok_dir: str) -> Dict:
    cache, path, data, info = _data(root, tok_dir, "ds_dict", shape="dict")
    a = info["audit"]
    for s in ("train", "dev", "test"):
        assert a[s]["relevant_missing_from_pool"] == 0, a[s]
    assert a["dev"]["queries_in_own_pool"] == 1
    sp = cache["splits"]
    for s in ("train", "dev"):
        for q in sp[s]["q"]:
            assert q["rel"] and all(0 <= j < len(sp[s]["pool"]) for j in q["rel"])
            assert all(sp[s]["pool"][j]["id"] != q["id"] for j in q["rel"])
    for s in sp:
        for x in sp[s]["q"] + sp[s]["pool"]:
            assert 1 <= len(x["chunks"]) <= 4 and all(1 <= len(c) <= 48 and c[-1] == 0
                                                     for c in x["chunks"])
    bm = cache["bm25"]
    qid, pool = sp["dev"]["q"][0]["id"], [x["id"] for x in sp["dev"]["pool"]]
    assert not np.isfinite(bm["dev"][0, pool.index(qid)]), "self not masked"
    assert bm["dev_micro_f1"] > 0.2, bm["dev_micro_f1"]
    assert bm["train_top"].shape[0] == len(sp["train"]["q"])
    return {"bm25_dev_f1": bm["dev_micro_f1"], "params": bm["params"]}


def t3_model(root: str) -> Dict:
    from cjpe_train import make_optimizer, make_scaler
    from pcr_model import NEW_TECHNIQUES, apply_pcr_technique, build_pcr_model, encode_chunks
    from pcr_train import ZERO_INIT, Stepper, build_data, job_checks
    tok = os.path.join(root, "tok")
    cache, _p, _d, _i = _data(root, tok, "ds_model")
    out = {}
    nullctx = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
    for mid in MODELS:
        p1 = write_phase1(root, mid, seed=3)
        for tech in ("full_ft", "partial_ft", "lorma", "lora", "qlora", "prefix", "prompt", "ia3"):
            torch.manual_seed(0)
            model, _ = build_pcr_model(p1)
            base = copy.deepcopy(model)
            cfg = {"lr": 1e-3, "n_virtual": 3, "partial_top_k": 2, "seed": 1, "batch_pairs": 4,
                   "hard_negs": 2, "k_train": 2, "micro_tokens": 4096, "chunk_tokens": 48}
            groups, techo = apply_pcr_technique(model, tech, cfg, lambda *a: None, [5, 6, 7])
            if tech in ("qlora",):                                   # NF4 error small
                a_ = encode_chunks(base, cache["splits"]["train"]["q"][0]["chunks"], "cpu",
                                   nullctx, 10 ** 6)
                b_ = encode_chunks(model, cache["splits"]["train"]["q"][0]["chunks"], "cpu",
                                   nullctx, 10 ** 6)
                assert float((a_ - b_).abs().max()) < 0.25, f"{mid}/qlora NF4 error"
            ctxo = type("C", (), {"device": torch.device("cpu"), "log": print})()
            d = build_data(cfg, cache)
            st = Stepper(ctxo, model, make_optimizer(groups), None, make_scaler(False), techo,
                         nullctx, cfg, cache, d)
            st.ctx_cache = cache
            res = job_checks(base if tech in ZERO_INIT else None, model, st, cfg, d,
                             torch.device("cpu"), nullctx)
            assert not res["critical_failed"], (mid, tech, res)
            bad_adv = [c for c in res["checks"] if c["verdict"] == "fail"]
            assert not bad_adv, (mid, tech, bad_adv)
            if tech in ("prefix", "prompt"):
                assert model.n_virtual == 3
            out[f"{mid}/{tech}"] = {c["name"]: round(c.get("value", 0) or 0, 6) for c in res["checks"]}
    return {"n": len(out), "sample": dict(list(out.items())[:3])}


def _job_cfg(root, tok_dir, cache_path, tech, run, **kw):
    p1 = write_phase1(root, "M2", tok_dir=tok_dir)
    cfg = {"experiment": "T", "job": os.path.basename(run), "model": "M2", "phase1_dir": p1,
           "bpe_dir": tok_dir, "technique": tech, "doc_repr": "hier", "pooling": "mean",
           "attn_mode": "causal", "chunk_tokens": 48, "k_train": 2, "k_infer": 4,
           "batch_pairs": 8, "hard_negs": 2, "lr": 3e-3, "head_lr": 3e-3, "micro_tokens": 4096,
           "seed": 1, "budget_s": 900, "cache_path": cache_path, "skip_job_checks": False,
           "n_virtual": 4, "epochs_max": 2, "n_dev": 10, "n_test": 10}
    cfg.update(kw)
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    return cfg


def t4_overfit(root: str, tok_dir: str) -> Dict:
    _c, path, _d, _i = _data(root, tok_dir, "ds_job")
    run = os.path.join(root, "run_overfit")
    _job_cfg(root, tok_dir, path, "full_ft", run)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, os.path.join(HERE, "pcr_train.py"), "--run-dir", run,
                        "--mode", "overfit"], env=env, capture_output=True, text=True, timeout=900)
    f = os.path.join(run, "logs", "overfit.json")
    assert os.path.exists(f), (p.stdout + p.stderr)[-2500:]
    ov = json.load(open(f))
    assert ov["passed"], ov
    return ov


def t5_resume(root: str, tok_dir: str) -> Dict:
    from pcr_data import load_cache
    from validate_submission import validate
    path = json.load(open(os.path.join(root, "run_overfit", "logs", "run_config.json")))["cache_path"]
    run = os.path.join(root, "run_resume")
    cfg = _job_cfg(root, tok_dir, path, "lorma", run, kill_at_step=3, lr=1e-3,
                   batch_pairs=4)
    env = dict(os.environ, PYTHONPATH=HERE, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(HERE, "pcr_train.py"), "--run-dir", run]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 137, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert os.path.exists(os.path.join(run, "checkpoints", "last.pt"))
    cfg["kill_at_step"] = -1
    write_json(os.path.join(run, "logs", "run_config.json"), cfg)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    log = open(os.path.join(run, "logs", "console.log")).read()
    assert p.returncode == 0, log[-2500:]
    assert "[resume] from step 3" in log
    assert os.path.exists(os.path.join(run, "DONE.json"))
    sp = load_cache(path)["splits"]["test"]
    v = validate(os.path.join(run, "submission", "submission.zip"), [x["id"] for x in sp["q"]],
                 [x["id"] for x in sp["pool"]])
    assert v["ok"], v
    met = json.load(open(os.path.join(run, "metrics.json")))
    assert met["dev_micro_f1"] is not None and met["dev_bm25_f1"] is not None
    for f in ("dev/dev_predictions.jsonl", "dev/fusion_curve.csv", "dev/dev_report.json",
              "dev/scores.npz"):
        assert os.path.exists(os.path.join(run, f)), f
    return {"resumed_at_step": 3, "dev_micro_f1": met["dev_micro_f1"],
            "dev_dense_f1": met["dev_dense_f1"], "dev_bm25_f1": met["dev_bm25_f1"]}


def run_all(log=print) -> Dict:
    res = {"tests": {}, "failed": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with tempfile.TemporaryDirectory() as root:
        tok_dir = write_tokenizer(os.path.join(root, "tok"))
        tests = [("1_metrics", lambda: t1_metrics(tok_dir)),
                 ("2_data_pools_cache_bm25", lambda: t2_data(root, tok_dir)),
                 ("3_model_techniques_gradcache", lambda: t3_model(root)),
                 ("4_overfit", lambda: t4_overfit(root, tok_dir)),
                 ("5_kill_resume", lambda: t5_resume(root, tok_dir))]
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
