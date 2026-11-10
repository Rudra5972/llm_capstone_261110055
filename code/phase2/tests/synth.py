"""
tests/synth.py -- synthetic stand-ins for the Phase-2 data and the Phase-1 checkpoints, so every
code path can be exercised on CPU without the real files.

  make_raw(root, gold_format)   -> writes Phase_2_LNER/{Train,Test}/... in one of several formats
  make_tokenizer(path)          -> a small BPE trained with the verbatim train_bpe()
  make_phase1_models(root, vocab) (needs torch) -> tiny GPT2/Llama/Mistral/BLOOM checkpoints
                                    saved exactly like the Phase-1 notebooks (model.pt +
                                    config.json + bpe files in a final/ dir)
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FIRST = ["Ram", "Sita", "Arjun", "Meera", "Rahul", "Anita", "Vikram", "Lakshmi", "Suresh",
         "Kavita", "Mohan", "Priyá", "Zoë", "Rāma"]
LAST = ["Kumar", "Sharma", "Iyer", "Reddy", "Gupta", "Singh", "Nair", "Rao", "Das", "Patil"]
COURTS = ["SUPREME COURT OF INDIA", "High Court of Bombay", "High Court of Delhi",
          "Madras High Court", "National Green Tribunal"]
STATS = ["Indian Penal Code", "Code of Criminal Procedure, 1973", "Constitution of India",
         "Arbitration and Conciliation Act, 1996", "Income Tax Act, 1961"]
AUTHS = ["State of Maharashtra", "Union of India", "Central Bureau of Investigation",
         "Reserve Bank of India", "Election Commission"]
FILLER = ["The learned counsel submitted that", "It is well settled that",
          "Having heard the parties at length,", "In view of the above discussion,",
          "We are unable to accept the contention that", "The impugned order —",
          "the evidence on record clearly shows", "§ 12(1)(a) was relied upon;",
          "the appeal is accordingly disposed of.", "(see para 14 “supra”)",
          "Learned Senior Counsel  argued\n\nthat", "the Tribunal erred in law."]


def _name(r):
    return f"{r.choice(['Shri ', 'Smt. ', 'Mr. ', ''])}{r.choice(FIRST)} {r.choice(LAST)}"


def make_doc(r, n_paras):
    """Return (text, [(start, end, label, text)]) with every label used."""
    parts, ents = [], []
    pos = 0

    def add(s, label=None):
        nonlocal pos
        if label:
            ents.append((pos, pos + len(s), label, s))
        parts.append(s)
        pos += len(s)

    add("REPORTABLE\nIN THE ")
    add(r.choice(COURTS), "COURT")
    add("\nCIVIL APPEAL ")
    add(f"No. {r.randint(100, 9999)} of {r.randint(1990, 2023)}", "CASENO")
    add("\n\n")
    add(_name(r), "APP")
    add(" ...Appellant(s)\nVERSUS\n")
    if r.random() < 0.5:
        add(r.choice(AUTHS), "RESP")
    else:
        add(_name(r), "RESP")
    add(" ...Respondent(s)\n\nJUDGMENT\n\n")
    add(_name(r), "JUDGE")
    add(", J.\n\n")
    for _ in range(n_paras):
        k = r.random()
        add(r.choice(FILLER) + " ")
        if k < 0.15:
            add("Mr. ")
            add(f"{r.choice(FIRST)} {r.choice(LAST)}", "A.COUNSEL")
            add(", learned counsel for the appellant, and ")
            add(f"{r.choice(FIRST)} {r.choice(LAST)}", "R.COUNSEL")
            add(" for the respondent. ")
        elif k < 0.3:
            add("Section ")
            add(f"{r.randint(1, 500)}", None)
            add(" of the ")
            add(r.choice(STATS), "STAT")
            add(". ")
        elif k < 0.45:
            add("In ")
            add(f"{r.choice(LAST)} v. {r.choice(AUTHS)}, ({r.randint(1950, 2023)}) "
                f"{r.randint(1, 12)} SCC {r.randint(1, 900)}", "PREC")
            add(", this Court held so. ")
        elif k < 0.6:
            add("The order dated ")
            add(f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.{r.randint(1990, 2023)}", "DATE")
            add(" of the ")
            add(r.choice(AUTHS), "AUTH")
            add(" was challenged. ")
        elif k < 0.7:
            add("PW-1 ")
            add(_name(r), "WIT")
            add(" deposed that the incident took place. ")
        else:
            add("The ")
            add(r.choice(COURTS), "COURT")
            add(" dismissed the petition. ")
        if r.random() < 0.2:
            add("\n\n")
    add("\n\n")
    add(_name(r), "JUDGE")
    add("\nNew Delhi;\n")
    add(f"{r.choice(['January', 'March', 'July'])} {r.randint(1, 28)}, {r.randint(1995, 2023)}",
        "DATE")
    add(".")
    return "".join(parts), ents


def make_raw(root, gold_format="list", n_docs=(12, 12, 6), paras=(20, 60), seed=0,
             nested=False):
    """Write the six Phase-2 files under root/Phase_2_LNER in a chosen gold format.

    gold_format: "list"   {doc: [[s, e, label, text], ...]}
                 "dict"   {doc: [{"start","end","label","text"}, ...]}
                 "nested" {doc: {"entities": [...dicts...]}}
                 "studio" [{"id": doc, "annotations": [{"result": [{"value": {...}}]}]}]
    """
    r = random.Random(seed)
    base = os.path.join(root, "Phase_2_LNER")
    os.makedirs(os.path.join(base, "Train"), exist_ok=True)
    os.makedirs(os.path.join(base, "Test"), exist_ok=True)
    did = 1128800
    all_ents = {}
    for fold, n in zip(("fold1", "fold2", "fold3"), n_docs):
        texts, gold = {}, {}
        for _ in range(n):
            did += 1
            t, e = make_doc(r, r.randint(*paras))
            if nested and e:
                s0, e0, _, _ = e[1]                          # a nested entity inside CASENO
                e.append((s0 + 4, s0 + 6, "DATE", t[s0 + 4:s0 + 6]))
            texts[str(did)] = t
            gold[str(did)] = e
            all_ents[str(did)] = e
        json.dump(texts, open(os.path.join(base, "Test" if fold == "fold3" else "Train",
                                           f"{fold}-text.json"), "w"), ensure_ascii=False)
        if fold == "fold3":
            continue
        if gold_format == "list":
            g = {k: [list(x) for x in v] for k, v in gold.items()}
        elif gold_format == "dict":
            g = {k: [{"start": s, "end": e, "label": l, "text": t} for s, e, l, t in v]
                 for k, v in gold.items()}
        elif gold_format == "nested":
            g = {k: {"entities": [{"start_offset": s, "end_offset": e, "type": l, "text": t}
                                  for s, e, l, t in v]} for k, v in gold.items()}
        elif gold_format == "studio":
            g = [{"id": k, "annotations": [{"result": [
                {"value": {"start": s, "end": e, "text": t, "labels": [l]}} for s, e, l, t in v]}]}
                for k, v in gold.items()]
        else:
            raise ValueError(gold_format)
        json.dump(g, open(os.path.join(base, "Train", f"{fold}-gold.json"), "w"),
                  ensure_ascii=False)
    with open(os.path.join(base, "Train", "labels.txt"), "w") as f:
        f.write("\n".join(["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH",
                           "COURT", "STAT", "PREC", "DATE", "CASENO"]) + "\n")
    return all_ents


def make_tokenizer(path, vocab_size=600, seed=0):
    """Small BPE trained with the verbatim Phase-1 trainer on synthetic legal text."""
    from bpe_tokenizer import BPETokenizer, count_pretokens, train_bpe
    r = random.Random(seed)
    texts = [make_doc(r, 30)[0] for _ in range(40)]
    wc = count_pretokens(texts, log=lambda *a: None)
    merges, _ = train_bpe(wc, vocab_size, log=lambda *a: None, min_pair_count=2)
    tok = BPETokenizer(merges)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


TINY = {
    "gpt2style": ("GPT2StyleLM", dict(n_positions=2048, n_embd=32, n_layer=4, n_head=4)),
    "llamastyle": ("LlamaStyleLM", dict(max_position_embeddings=2048, hidden_size=32,
                                         intermediate_size=64, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2,
                                         sliding_window=None)),
    "mistral-sliding": ("LlamaStyleLM", dict(max_position_embeddings=2048, hidden_size=32,
                                              intermediate_size=64, num_hidden_layers=4,
                                              num_attention_heads=4, num_key_value_heads=2,
                                              sliding_window=16)),
    "bloom-alibi": ("BloomStyleLM", dict(hidden_size=32, n_layer=4, n_head=4)),
}


def make_phase1_models(root, tokenizer_json, vocab_size, seed=0):
    """Tiny random-init Phase-1-format checkpoints: root/<arch>/final/{model.pt, config.json,...}."""
    import shutil
    from dataclasses import asdict

    import torch

    from phase1_models import MODEL_CLASSES
    out = {}
    for i, (arch, (cls_name, kw)) in enumerate(TINY.items()):
        cls, cfg_cls = MODEL_CLASSES[cls_name]
        torch.manual_seed(seed + i)
        cfg = cfg_cls(vocab_size=vocab_size, **kw)
        m = cls(cfg)
        d = os.path.join(root, f"legal-lm-{arch}", "final")
        os.makedirs(d, exist_ok=True)
        torch.save(m.state_dict(), os.path.join(d, "model.pt"))
        json.dump({"model_class": cls_name, "arch": arch, "config": asdict(cfg)},
                  open(os.path.join(d, "config.json"), "w"), indent=2)
        shutil.copy(tokenizer_json, os.path.join(d, "bpe_model.json"))
        json.dump({"eval_loss": 5.0, "arch": arch}, open(os.path.join(d, "run_summary.json"), "w"))
        out[arch] = d
    return out
