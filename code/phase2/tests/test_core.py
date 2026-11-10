"""
tests/test_core.py -- CPU unit tests for S1-S4 (run: python tests/test_core.py  or  pytest tests).

Uses tiny random-init models of all four Phase-1 classes, built and saved exactly like the
Phase-1 notebooks, plus a small BPE trained with the verbatim trainer.
"""
import itertools
import os
import random
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synth import make_phase1_models, make_tokenizer  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="p2test_")
_TOK = make_tokenizer(os.path.join(_TMP, "tok", "bpe_model.json"), vocab_size=400)
_DIRS = make_phase1_models(os.path.join(_TMP, "p1"), os.path.join(_TMP, "tok", "bpe_model.json"),
                           vocab_size=len(_TOK))
V = len(_TOK)


def _lm(arch):
    from ner_models import load_phase1_lm
    return load_phase1_lm(_DIRS[arch])[0].eval()


# ------------------------------------------------------------------------------------- tokens
def test_offsets_and_tags_roundtrip():
    from ner_data import decode_tags, tag_doc
    from ner_tok import OffsetEncoder, verify_offsets
    enc = OffsetEncoder(_TOK)
    text = "IN THE  SUPREME COURT OF INDIA\n\nShri Rāma  Kumar ...Appellant; Zoë 😀Das, J."
    ents = [(text.index("SUPREME"), text.index("INDIA") + 5, "COURT"),
            (text.index("Shri"), text.index("Kumar") + 5, "APP"),
            (text.index("Zoë"), text.index("Das") + 3, "JUDGE")]
    td = enc.encode(text)
    assert verify_offsets(text, td) == 0
    labels = ["COURT", "APP", "JUDGE"]
    tags, st = tag_doc(td, ents, {l: i for i, l in enumerate(labels)})
    assert decode_tags(tags, td, labels) == ents, (decode_tags(tags, td, labels), ents)


def test_snap_rules_fix_inside_token_boundary():
    """Real 32k BPE merges '.,' into one token, so an entity ending in '.' loses its final char.
    Feed such one-char-short spans directly and check the learned rule repairs them without
    breaking spans that were already right."""
    from ner_data import SnapRules, strict_f1
    r = random.Random(0)
    gold, dec, texts = {}, {}, {}
    for i in range(30):
        name = f"{r.choice(['Acme', 'Tata', 'Bajaj'])} Pvt. Ltd."
        text = f"The company {name}, filed. Ram Kumar, J. said so."
        s = text.index(name)
        rk = text.index("Ram Kumar")
        d = f"d{i}"
        gold[d] = [(s, s + len(name), "AUTH"), (rk, rk + 9, "JUDGE")]
        dec[d] = [(s, s + len(name) - 1, "AUTH"), (rk, rk + 9, "JUDGE")]   # AUTH lost its '.'
        texts[d] = text
    before = strict_f1(dec, gold)
    snap = SnapRules().fit(dec, gold, texts)
    after = strict_f1({d: snap.apply(v, texts[d]) for d, v in dec.items()}, gold)
    assert before < 0.8 and after == 1.0, (before, after, snap.to_json())
    assert "JUDGE" not in snap.rules            # never learned a rule that breaks correct spans


def test_windows_merge_identity():
    from ner_data import make_windows, merge_window_scores
    for n in (1, 5, 511, 512, 1000, 3001):
        w = make_windows(n, 512, 256)
        assert w[0][0] == 0 and w[-1][1] == n and all(b - a <= 511 for a, b in w)
        x = np.random.rand(n, 5).astype(np.float32)
        m = merge_window_scores(n, w, [x[a:b] for a, b in w])
        assert np.allclose(m, x, atol=1e-5)


# ------------------------------------------------------------------------------------- trunks
def _hidden_via_hook(lm, ids, mask):
    store = {}
    h = lm.lm_head.register_forward_hook(lambda m, i, o: store.__setitem__("x", i[0]))
    lm(ids, attention_mask=mask)
    h.remove()
    return store["x"]


def test_causal_trunk_equals_original():
    from ner_models import Trunk
    torch.manual_seed(0)
    for arch in _DIRS:
        lm = _lm(arch)
        trunk = Trunk(lm, "causal").eval()
        ids = torch.randint(3, V, (3, 40))
        for mask in (torch.ones_like(ids), torch.cat([torch.ones(3, 30), torch.zeros(3, 10)], 1).long()):
            with torch.no_grad():
                ref = _hidden_via_hook(lm, ids, mask)
                got = trunk(ids, mask)
            assert torch.allclose(ref, got, atol=1e-6), (arch, (ref - got).abs().max())


def test_bidir_sees_future_and_ignores_padding():
    from ner_models import Trunk
    torch.manual_seed(0)
    for arch in _DIRS:
        lm = _lm(arch)
        for mode in ("causal", "bidir"):
            tr = Trunk(lm, mode).eval()
            ids = torch.randint(3, V, (1, 30))
            ids2 = ids.clone()
            ids2[0, 20] = (ids2[0, 20] - 3 + 1) % (V - 3) + 3
            with torch.no_grad():
                a, b = tr(ids), tr(ids2)
            changed_before = not torch.allclose(a[0, :20], b[0, :20], atol=1e-6)
            assert changed_before == (mode == "bidir"), (arch, mode)
            # padding: change pad tokens -> real positions unchanged; batched == single
            pad = torch.cat([ids, torch.full((1, 8), 1)], 1)
            pad2 = torch.cat([ids, torch.randint(3, V, (1, 8))], 1)
            m = torch.cat([torch.ones(1, 30), torch.zeros(1, 8)], 1).long()
            with torch.no_grad():
                p1, p2 = tr(pad, m), tr(pad2, m)
            assert torch.allclose(p1[0, :30], p2[0, :30], atol=1e-5), (arch, mode, "pad")
            assert torch.allclose(p1[0, :30], a[0], atol=1e-5), (arch, mode, "batched vs single")


def test_mistral_bidir_window():
    from ner_models import Trunk
    lm = _lm("mistral-sliding")                     # tiny window = 16
    tr = Trunk(lm, "bidir", layer_ids=[0]).eval()   # one layer -> receptive field = window
    ids = torch.randint(3, V, (1, 40))
    ids2 = ids.clone()
    ids2[0, 0] = (ids2[0, 0] - 3 + 1) % (V - 3) + 3
    with torch.no_grad():
        a, b = tr(ids), tr(ids2)
    diff = (a - b).abs().amax(-1)[0]
    assert diff[1:16].min() > 0 and diff[16:].max() == 0, diff


def test_student_pruning_counts():
    from ner_models import build_ner_model, count_params
    full, _ = build_ner_model(_DIRS["llamastyle"], 12)
    stud, _ = build_ner_model(_DIRS["llamastyle"], 12, layer_ids=[0, 2])
    assert count_params(stud) < count_params(full)
    assert len(stud.trunk.blocks) == 2


# ------------------------------------------------------------------------------------- PEFT
def _model(arch, technique, cfg=None):
    from ner_models import build_ner_model
    from ner_peft import apply_technique
    m, _ = build_ner_model(_DIRS[arch], 12)
    m.eval()
    ref = m
    groups = apply_technique(m, technique, cfg or {}, log=lambda *a: None)
    return m, groups


def test_peft_identity_and_merge():
    from ner_models import build_ner_model
    from ner_peft import LoRALinear, LoRMALinear
    torch.manual_seed(0)
    ids = torch.randint(3, V, (2, 25))
    for arch in _DIRS:
        base, _ = build_ner_model(_DIRS[arch], 12)
        base.eval()
        with torch.no_grad():
            h0 = base.trunk(ids)
        for tech in ("lora", "lorma", "adapters"):
            m, _ = _model(arch, tech)
            with torch.no_grad():
                h1 = m.trunk(ids)
            tol = 1e-5 if tech != "adapters" else 5e-2
            assert torch.allclose(h0, h1, atol=tol), (arch, tech, (h0 - h1).abs().max())
            # merge equality after random B
            for mod in m.modules():
                if isinstance(mod, (LoRALinear, LoRMALinear)):
                    with torch.no_grad():
                        for p in mod.parameters():
                            if p.requires_grad:
                                p.normal_(std=0.05)
                    x = torch.randn(4, mod.base.in_features)
                    with torch.no_grad():
                        y = mod.eval()(x)
                        W = mod.merged_weight()
                        y2 = torch.nn.functional.linear(x, W, mod.base.bias)
                    assert torch.allclose(y, y2, atol=1e-4), (arch, tech)


def test_lorma_param_count_and_slices():
    m, _ = _model("gpt2style", "lorma", {"lorma_r": 4})
    d = m.trunk.hidden_size
    blk = m.trunk.blocks[0]
    assert [tuple(s) for s in blk.attn.qkv.slices] == [(0, d), (d, 2 * d), (2 * d, 3 * d)]
    n = sum(p.numel() for n_, p in m.named_parameters() if p.requires_grad and "lorma" in n_)
    per_block = 3 * (2 * 4 * d) + 2 * 4 * d + 2 * 4 * (4 * d) + 2 * 4 * d
    assert n == per_block * len(m.trunk.blocks), (n, per_block)


def test_technique_trainable_sets():
    for arch in _DIRS:
        m, _ = _model(arch, "partial_ft", {"partial_top_k": 2})
        tr = {n for n, p in m.named_parameters() if p.requires_grad}
        L = len(m.trunk.blocks)
        assert all(n.startswith(("classifier.", "trunk.norm.", f"trunk.blocks.{L - 1}.",
                                 f"trunk.blocks.{L - 2}.")) for n in tr), arch
        assert not any(n.startswith("trunk.embed") for n in tr)
        m, _ = _model(arch, "full_ft")
        assert all(p.requires_grad for p in m.parameters())
    for arch in ("gpt2style", "bloom-alibi"):
        m, _ = _model(arch, "bitfit")
        assert all(n.endswith(".bias") or n.startswith("classifier.")
                   for n, p in m.named_parameters() if p.requires_grad)
    try:
        _model("llamastyle", "bitfit")
        raise AssertionError("BitFit on llama should fail (no biases)")
    except ValueError:
        pass


def test_partial_ft_layer_decay():
    m, groups = _model("llamastyle", "partial_ft", {"partial_top_k": 3, "lr": 1e-4,
                                                    "layer_decay": 0.5})
    lrs = {g["name"].split("/")[0]: g["lr"] for g in groups}
    L = len(m.trunk.blocks)
    assert abs(lrs[f"block{L - 1}"] - 1e-4) < 1e-12 and abs(lrs[f"block{L - 3}"] - 2.5e-5) < 1e-12


def test_every_trainable_param_learns():
    """After two optimizer steps every trainable tensor has received a gradient and changed
    (also required by DDP find_unused_parameters=False)."""
    torch.manual_seed(0)
    ids = torch.randint(3, V, (4, 30))
    labels = torch.randint(0, 25, (4, 30))
    labels[:, 0] = -100
    for arch in _DIRS:
        for tech in ("full_ft", "lora", "lorma", "adapters", "partial_ft") + \
                (("bitfit",) if arch in ("gpt2style", "bloom-alibi") else ()):
            m, groups = _model(arch, tech, {"partial_top_k": 2})
            m.train()
            opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups])
            before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
            for _ in range(2):
                _, loss = m(ids, torch.ones_like(ids), labels=labels)
                opt.zero_grad()
                loss.backward()
                no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
                assert not no_grad, (arch, tech, no_grad[:3])
                opt.step()
            frozen_changed = [n for n, p in m.named_parameters() if not p.requires_grad
                              and p.grad is not None]
            assert not frozen_changed, (arch, tech, frozen_changed[:3])
            unchanged = [n for n, p in m.named_parameters()
                         if p.requires_grad and torch.equal(p.detach(), before[n])]
            assert not unchanged, (arch, tech, unchanged[:3])


# ------------------------------------------------------------------------------------- CRF
def test_crf_against_brute_force():
    from ner_crf import CRF
    torch.manual_seed(0)
    crf = CRF(1)                                     # tags O, B, I
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
    T, C = 4, 3
    em = torch.randn(1, T, C)
    mask = torch.ones(1, T, dtype=torch.bool)
    tr = (crf.transitions + crf.trans_pen).detach()
    st = crf.start_transitions.detach()             # window starts are unconstrained
    scores = {}
    for path in itertools.product(range(C), repeat=T):
        s = st[path[0]] + em[0, 0, path[0]]
        for t in range(1, T):
            s = s + tr[path[t - 1], path[t]] + em[0, t, path[t]]
        s = s + crf.end_transitions[path[-1]].detach()
        scores[path] = s
    logz = torch.logsumexp(torch.stack(list(scores.values())), 0)
    gold = (0, 1, 2, 0)
    nll = crf.neg_log_likelihood(em, torch.tensor([gold]), mask, reduction="sum")
    assert torch.allclose(nll, logz - scores[gold], atol=1e-4)
    best = max(scores, key=lambda k: scores[k])
    assert tuple(crf.viterbi(em, mask)[0]) == best
    # numpy doc-level viterbi (which additionally forbids starting a DOCUMENT with I-X) agrees
    # with brute force restricted to legal starts
    from ner_eval import transition_constraints, viterbi
    allowed, a_start = transition_constraints(1)
    ex = crf.export()
    path = viterbi(em[0].numpy(), allowed, a_start, ex["transitions"], ex["start"], ex["end"])
    best_doc = max((k for k in scores if a_start[k[0]]), key=lambda k: scores[k])
    assert tuple(path.tolist()) == best_doc
    # a window starting inside an entity (I tag first) has a finite, sensible NLL
    nll_i = crf.neg_log_likelihood(em, torch.tensor([(2, 2, 0, 0)]), mask, reduction="sum")
    assert float(nll_i) < 100, float(nll_i)


# ------------------------------------------------------------------------------------- eval
def test_viterbi_legal_and_metrics():
    from ner_eval import strict_scores, transition_constraints, viterbi
    allowed, a_start = transition_constraints(3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        path = viterbi(rng.normal(size=(30, 7)), allowed, a_start)
        assert a_start[path[0]] and all(allowed[a, b] for a, b in zip(path, path[1:]))
    gold = {"d": [(0, 5, "A"), (10, 12, "B")]}
    pred = {"d": [(0, 5, "A"), (10, 13, "B"), (20, 22, "A")]}
    m = strict_scores(pred, gold, ["A", "B", "C"])
    assert m["per_label"]["A"]["precision"] == 0.5 and m["per_label"]["A"]["recall"] == 1.0
    assert m["per_label"]["B"]["f1"] == 0.0
    assert abs(m["macro_all"]["f1"] - (2 / 3) / 3) < 1e-9
    assert abs(m["macro_present"]["f1"] - (2 / 3) / 2) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"\nALL {len(fns)} CORE TESTS PASSED")
