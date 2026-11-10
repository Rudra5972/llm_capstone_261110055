"""
make_notebooks.py -- generate the standalone Kaggle notebooks.

    python make_notebooks.py [--out notebooks]

Writes 00_data_and_tests.ipynb (CPU is enough) and E1_full_ft.ipynb ... E6_kd.ipynb. Every
notebook inlines ALL modules with %%writefile cells (byte-identical in all of them; an MD5
table is embedded and checked at run time), so a notebook can be uploaded to any Kaggle account
and only needs the input datasets listed in its first cell.
"""
import argparse
import hashlib
import json
import os

from ner_common import EXPERIMENTS, REQUIRED_EXPS

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["bpe_tokenizer.py", "phase1_models.py", "ner_common.py", "ner_tok.py", "ner_data.py",
           "ner_eval.py", "ner_crf.py", "ner_models.py", "ner_peft.py", "ner_predict.py",
           "ner_train.py", "validate_submission.py", "select_best.py", "aggregate.py",
           "run_experiment.py"]
TEST_MODULES = ["tests/synth.py", "tests/test_core.py"]
CODE_DIR = "/kaggle/working/phase2"

INPUTS_MD = """**Attach these inputs** (Add Input -> Datasets / Notebook Output):
1. The four Phase-1 outputs (each has `final/model.pt`, `final/config.json`, `final/bpe_model.json`).
   Models are recognised by the `arch` field in `config.json`, so dataset names do not matter.
2. The official Phase-2 files (`Phase_2_LNER`: `fold1/2-text.json`, `fold1/2-gold.json`,
   `labels.txt`, `fold3-text.json`) as a Kaggle dataset. *Alternative:* a read-only Hugging Face
   token as the Kaggle secret `HF`; the files are then downloaded once with plain `urllib`
   (nothing is ever written to the Hub).
3. Optional: the output of `00_data_and_tests` (the `phase2_data` cache) -- otherwise the cache is
   rebuilt here in about a minute on CPU.
4. Optional: the output of an earlier session of THIS notebook: finished models are skipped and an
   unfinished one resumes from its `last.pt`."""


def _id(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


def md_cell(text):
    return {"cell_type": "markdown", "id": _id("md" + text), "metadata": {},
            "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "id": _id("code" + text),
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def writefile_cells(paths):
    cells, hashes = [], {}
    for rel in paths:
        src = open(os.path.join(HERE, rel), encoding="utf-8").read()
        if not src.endswith("\n"):
            src += "\n"
        hashes[rel] = hashlib.md5(src.encode("utf-8")).hexdigest()
        cells.append(code_cell(f"%%writefile {CODE_DIR}/{rel}\n{src}"))
    return cells, hashes


def header_cells(title, body, hashes):
    setup = f"""import os, sys, json, hashlib, subprocess, glob
os.makedirs("{CODE_DIR}/tests", exist_ok=True)
print(sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(),
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as e:
    print("torch not importable:", e)
print("inputs:", sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else "none")
EXPECTED_MD5 = {json.dumps(hashes, indent=1)}
"""
    return [md_cell(f"# {title}\n\n{body}\n\n{INPUTS_MD}\n"), code_cell(setup)]


def verify_cell():
    return code_cell(f"""# every module must be byte-identical to the generated version (same code in all notebooks)
bad = []
for rel, h in EXPECTED_MD5.items():
    p = os.path.join("{CODE_DIR}", rel)
    if hashlib.md5(open(p, "rb").read()).hexdigest() != h:
        bad.append(rel)
assert not bad, f"modules differ from the generated version: {{bad}}"
print(f"all {{len(EXPECTED_MD5)}} modules verified")
sys.path.insert(0, "{CODE_DIR}")
from ner_common import find_phase1_models, find_files
print("Phase-1 weights:", json.dumps(find_phase1_models(), indent=1))
print("bpe_model.json:", find_files("bpe_model.json", ["/kaggle/input"])[:4])
""")


def tail_cells():
    return [code_cell("""print(open("/kaggle/working/phase2_out/INDEX.md").read()
      if os.path.exists("/kaggle/working/phase2_out/INDEX.md") else "no INDEX.md yet")
for dp, dn, fn in sorted(os.walk("/kaggle/working/phase2_out")):
    depth = dp.count(os.sep) - "/kaggle/working/phase2_out".count(os.sep)
    if depth <= 2:
        size = sum(os.path.getsize(os.path.join(dp, f)) for f in fn) / 1e6
        print("  " * depth + os.path.basename(dp) + f"/  ({size:.1f} MB in files here)")
"""),
            md_cell("**Run this notebook as a commit** (Save Version -> Save & Run All): it keeps "
                    "running without the browser, and `/kaggle/working/phase2_out` becomes the "
                    "notebook's output. Attach that output to the next session of this notebook "
                    "to skip finished models / resume an unfinished one.\n\n"
                    "Upload `phase2_out/<experiment>/BEST/submission.zip` to Codabench, then run "
                    "`python select_best.py --coverage` once all six are done.")]


def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def make_experiment_nb(exp):
    spec = EXPERIMENTS[exp]
    files, hashes = writefile_cells(MODULES)
    extra = ""
    if exp == "E6":
        extra = ("\n\n**E6 needs E1's output attached** (it reads `phase2_out/E1_full_ft/BEST/`: "
                 "the teacher's `best.pt` + `run_config.json`).")
    body = (f"**{exp}: {spec['title']}** on all four Phase-1 models (order M4, M2, M3, M1). "
            f"Accelerator: **GPU T4 x2**. Each model: pre-flight (2-GPU smoke test, falls back "
            f"to 1 GPU), overfit-64 check (first model), LR probe, training with 30-minute "
            f"checkpoints, dev report, validated `submission.zip`. Then best-of-4 into "
            f"`phase2_out/{spec['folder']}/BEST/`.{extra}\n\n"
            f"To split the four models across two accounts, change the run cell to e.g. "
            f"`--models M4,M2` here and `--models M3,M1` on the other account.")
    cells = header_cells(f"CS-781 Phase 2 (L-NER) -- {exp} {spec['title']}", body, hashes)
    cells += files + [verify_cell()]
    cells.append(code_cell(f"!cd {CODE_DIR} && python run_experiment.py --exp {exp} "
                           f"--nproc auto --resume auto"))
    cells += tail_cells()
    return notebook(cells)


def make_prep_nb():
    files, hashes = writefile_cells(MODULES + TEST_MODULES)
    body = ("Runs on CPU (no GPU quota needed). Builds the shared data cache and checks the "
            "gates before any GPU time is spent:\n\n"
            "* **G0** every gold entity: `text[start:end] == entity text` (end exclusive)\n"
            "* **G1** gold -> token tags -> spans round trip (ceiling) >= 0.97, with learned "
            "boundary snapping if it helps\n"
            "* **G2** chunking + overlap merging reproduces the same spans\n\n"
            "Then runs the CPU unit tests (tiny random models of all four architectures). "
            "If a gate fails, the cell prints STOP AND ASK with the exact problem.")
    cells = header_cells("CS-781 Phase 2 (L-NER) -- 00 data cache, gates, CPU tests", body, hashes)
    cells += files + [verify_cell()]
    cells.append(code_cell(f"!cd {CODE_DIR} && python ner_data.py build --out /kaggle/working/phase2_data"))
    cells.append(code_cell("""rep = json.load(open("/kaggle/working/phase2_data/data_report.json"))
for k in ("gold_schema", "G0", "overlaps", "G2", "split", "stats"):
    print(k, ":", json.dumps(rep.get(k), indent=1)[:1500])
g1 = rep["G1"]
print("G1 ceiling raw", g1["ceiling_raw"], "| with snapping", g1["ceiling_with_snapping"],
      "| mismatch categories", g1["mismatch_categories"])
for ex in g1["examples"][:10]:
    print("  ", ex)
"""))
    cells.append(code_cell(f"!cd {CODE_DIR} && python tests/test_core.py"))
    cells.append(md_cell("Save this notebook's output and attach `phase2_data` to the experiment "
                         "notebooks (optional -- they rebuild it if absent)."))
    return notebook(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "notebooks"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nbs = {"00_data_and_tests.ipynb": make_prep_nb()}
    for exp in REQUIRED_EXPS:
        nbs[f"{EXPERIMENTS[exp]['folder']}.ipynb"] = make_experiment_nb(exp)
    for name, nb in nbs.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("wrote", os.path.join(a.out, name), f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
