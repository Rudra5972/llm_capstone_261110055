"""
make_notebooks.py -- generate the Phase-3 Kaggle notebooks (dev tool, not a shared module).

Every notebook embeds the SAME module block (%%writefile cells) and the same EXPECTED_MD5,
then runs one experiment family. Usage: python tools/make_notebooks.py OUT_DIR
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from rr_common import EXPERIMENTS, MODULE_FILES  # noqa: E402

CODE = "/kaggle/working/phase3"
OUT = "/kaggle/working/phase3_out"
DATA = "/kaggle/working/phase3_data"

ATTACH_COMMON = """**Attach these inputs** (Add Input -> Datasets / Notebook Output):
1. The four Phase-1 outputs (`rudra1610/01-train-gpt2style`, `02-train-llamastyle`,
   `03-train-mistral-sliding`, `04-train-bloom-alibi`; each has `final/model.pt`,
   `final/config.json`, `final/bpe_model.json`). Models are recognised by the `arch` field of
   `config.json`, so the dataset names do not matter.
2. The official Phase-3 data (`Phase_3_RR`: `Train/CL_train.json`, `CL_dev.json`,
   `IT_train.json`, `IT_dev.json`, `label_vocab.json`, `Test/CL_test.json`, `IT_test.json`) as a
   Kaggle dataset. *Alternative:* a read-only Hugging Face token as the Kaggle secret `HF` (+
   internet on); the files are then downloaded once with plain `urllib`.
3. Recommended: the output of `00_data_tests_pilot` (`PILOT.json` = causal-vs-bidir winner, and
   the `phase3_data` cache). Without it the attention mode defaults to `bidir`.
4. Optional: the output of an earlier session of THIS notebook: finished jobs are skipped and an
   unfinished one resumes from its `last.pt`.
"""

NOTEBOOKS = {
    "00_data_tests_pilot": {
        "title": "Data gates, self-tests and the causal-vs-bidir pilot",
        "what": "Gates G0-G2 on the real data (schema, ID collisions, tokenisation, packing and "
                "validator round trips), the CPU selftest incl. the full-size trainable-% check, "
                "then the pilot: M2 + LoRA for one epoch with causal and with bidirectional "
                "attention -> `PILOT.json` (attach this output to every other notebook). "
                "About 20-30 min on T4 x2.",
        "pre": [f"!cd {CODE} && python selftest.py --full-size --data-dir {DATA} "
                f"--out {OUT}/SELFTEST.json"],
        "launch": "--exp P0", "folder": "P0_pilot", "gpu": True},
    "E11_ensemble_report": {
        "title": "Ensemble of all BEST models + final report tables",
        "what": "CPU only. Averages the saved dev/test logits of every `<exp>/BEST` (log-softmax "
                "space; uniform / F1-weighted / top-3, best on dev), validates the submission and "
                "writes `REPORT_TABLES.md` + `README_PHASE3.md` from all results. **Attach the "
                "outputs of all other notebooks.** The ensemble's dev score is optimistic (dev "
                "chose the members).",
        "pre": [], "launch": "--exp E11", "folder": "E11_ensemble", "gpu": False},
}
for e, spec in EXPERIMENTS.items():
    if e == "E11":
        continue
    nb = {"title": spec["title"], "folder": spec["folder"], "gpu": True, "pre": [],
          "launch": f"--exp {e}"}
    t = spec["technique"]
    nb["what"] = {
        "full_ft": "Full fine-tuning of every trunk parameter (reference baseline: NOT on the "
                   "brief's technique menu, never counted toward the 5 techniques).",
        "lorma": "LoRMA+ (compulsory): y = (I + s*B*A) W0 x on q/k/v/o and the FFN projections, "
                 "Phase-2 implementation reused unchanged.",
        "lora": "LoRA r=16, alpha=32 on q/k/v/o and the FFN projections (Phase-2 code).",
        "rslora": "rsLoRA: LoRA with scale alpha/sqrt(r), r=32, alpha=16. Shares the LoRA code; "
                  "listed by the brief as its own technique.",
        "ia3": "IA3: learned scaling vectors on keys, values and the FFN intermediate activation "
               "(lr 3e-3, 12 epochs).",
        "adalora": "AdaLoRA: SVD-form update, orthogonality regulariser, sensitivity-based "
                   "importance with EMA, cubic budget 12 -> 8 average rank; logs the final rank "
                   "per layer.",
        "kd": "Knowledge distillation. Self-contained: per model a full fine-tuned TEACHER "
              "(`teacher_M*`) is trained first, then a STUDENT pruned to every second layer learns "
              "from the cached teacher logits (alpha 0.5, T 2). To reuse E0's runs as teachers "
              "instead, attach E0's output and add `--teacher-dir <.../phase3_out/E0_full_ft>`.",
        "qlora": "QLoRA, simulated without bitsandbytes: NF4 block-wise base weights (block 64, "
                 "double-quantised scales), de-quantised on the fly, LoRA r=16 on top.",
        "prefix": "Prefix Tuning: 20 key/value prefix vectors per layer through a reparameterisation "
                  "MLP (dropped after training); visible to every query under every mask.",
        "prompt": "Prompt Tuning: 20 soft input embeddings initialised from vocabulary embeddings.",
        "ablation": "Ablations on the strongest model (default M4) for LoRMA+ and LoRA: sentence-"
                    "only vs document windows vs +-1 neighbour context, linear vs BiGRU sequence "
                    "head, loss variants (CE, weighted CE, focal, logit-adjusted), and separate "
                    "CL / IT models combined vs one joint model. Not a counted experiment.",
    }[t]
    if t == "ablation":
        nb["launch"] = f"--exp {e} --models M4 --ablation-techniques lorma,lora --skip-probe"
    NOTEBOOKS[f"{e}_{spec['folder'].split('_', 1)[1]}"] = nb


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(text, i):
    return {"cell_type": "markdown", "id": f"md{i:02d}", "metadata": {}, "source": src(text)}


def code(text, i):
    return {"cell_type": "code", "execution_count": None, "id": f"c{i:02d}", "metadata": {},
            "outputs": [], "source": src(text)}


def build(name, nb, modules, md5):
    cells = []
    head = (f"# CS-781 Phase 3 (Rhetorical Role Prediction) -- {name.split('_')[0]}: "
            f"{nb['title']}\n\n{nb['what']}\n\n"
            + ("Accelerator: **GPU T4 x2**. Jobs run as a per-GPU queue (two models in parallel); "
               "a GPU that fails its sanity check is dropped (fallback to one GPU). Each job: "
               "smoke test with real-model self-checks -> overfit-32 (first job) -> LR probe "
               "x{1/3,1,3} -> training with 30-minute checkpoints -> dev report + bias decision "
               "-> validated `submission.zip`. Then best-of-4 into "
               f"`phase3_out/{nb['folder']}/BEST/`.\n\n"
               "To split the four models across two accounts, add e.g. `--models M4,M2` here "
               "and `--models M3,M1` on the other account.\n\n" if nb["gpu"] else
               "Accelerator: **None** (CPU).\n\n")
            + ATTACH_COMMON)
    cells.append(md(head, 0))
    c1 = ("import os, sys, json, hashlib, subprocess, glob\n"
          f"os.makedirs(\"{CODE}\", exist_ok=True)\n"
          "print(sys.version.split()[0])\n"
          "try:\n"
          "    import torch\n"
          "    print(\"torch\", torch.__version__, \"| GPUs:\", torch.cuda.device_count(),\n"
          "          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
          "except Exception as e:\n"
          "    print(\"torch not importable:\", e)\n"
          "print(\"inputs:\", sorted(os.listdir(\"/kaggle/input\")) if os.path.isdir(\"/kaggle/input\") else \"none\")\n"
          "EXPECTED_MD5 = " + json.dumps(md5, indent=1) + "\n")
    cells.append(code(c1, 1))
    for i, f in enumerate(MODULE_FILES):
        cells.append(code(f"%%writefile {CODE}/{f}\n" + modules[f], 2 + i))
    k = 2 + len(MODULE_FILES)
    verify = ("# every module must be byte-identical to the generated version (same code in all notebooks)\n"
              "bad = []\n"
              "for rel, h in EXPECTED_MD5.items():\n"
              f"    p = os.path.join(\"{CODE}\", rel)\n"
              "    if hashlib.md5(open(p, \"rb\").read()).hexdigest() != h:\n"
              "        bad.append(rel)\n"
              "assert not bad, f\"modules differ from the generated version: {bad}\"\n"
              "print(f\"all {len(EXPECTED_MD5)} modules verified\")\n"
              f"sys.path.insert(0, \"{CODE}\")\n"
              "from rr_common import SPEC, default_input_roots, find_files, find_phase1_models\n"
              "print(\"Phase-1 weights:\", json.dumps(find_phase1_models(default_input_roots()), indent=1))\n"
              "print(\"bpe_model.json:\", find_files(\"bpe_model.json\", default_input_roots())[:4])\n"
              "print(\"labels (\", len(SPEC['labels']), \"):\", SPEC['labels'])\n")
    cells.append(code(verify, k))
    gates = ("# data discovery + gates G0 (schema, ID collisions), G1 (tokenisation), G2 (packing and\n"
             "# validator round trips); reuses an attached phase3_data cache when it is valid\n"
             f"!cd {CODE} && python rr_data.py build --out {DATA}\n")
    cells.append(code(gates, k + 1))
    for j, p in enumerate(nb["pre"]):
        cells.append(code(p, k + 2 + j))
    launch = f"!cd {CODE} && python run_experiment.py {nb['launch']} --resume auto"
    if not nb["gpu"]:
        launch += " --gpus cpu"
    cells.append(code(launch, k + 5))
    show = (f"OUT = \"{OUT}\"\nFOLDER = \"{nb['folder']}\"\n"
            "print(open(f\"{OUT}/INDEX.md\").read() if os.path.exists(f\"{OUT}/INDEX.md\") else \"no INDEX.md yet\")\n"
            "p = f\"{OUT}/{FOLDER}/RESULTS.json\"\n"
            "rows = json.load(open(p)) if os.path.exists(p) else []\n"
            "print(f\"{'job':<30}{'dev F1':>8}{'CL':>8}{'IT':>8}{'trainable':>13}{'total':>13}{'min':>7}\")\n"
            "for r in rows:\n"
            "    d = r.get('dev_per_domain_macro_f1') or {}\n"
            "    print(f\"{r['job']:<30}{r['dev_macro_f1']:>8.4f}{d.get('CL', float('nan')):>8.4f}{d.get('IT', float('nan')):>8.4f}\"\n"
            "          f\"{r.get('trainable_params') or 0:>13,}{r.get('total_params') or 0:>13,}{r.get('train_minutes') or 0:>7.1f}\")\n"
            "for f in (\"SELECTION.json\", \"validation_report.json\"):\n"
            "    q = f\"{OUT}/{FOLDER}/BEST/{f}\"\n"
            "    if os.path.exists(q):\n"
            "        print(f, json.dumps(json.load(open(q)), indent=1)[:1500])\n"
            "for f in (\"PILOT.json\", \"REPORT_TABLES.md\"):\n"
            "    q = f\"{OUT}/{f}\"\n"
            "    if os.path.exists(q):\n"
            "        print(open(q).read()[:6000])\n")
    cells.append(code(show, k + 6))
    files = (f"OUT = \"{OUT}\"\n"
             "for dp, dn, fn in sorted(os.walk(OUT)):\n"
             "    depth = dp.count(os.sep) - OUT.count(os.sep)\n"
             "    if depth <= 2:\n"
             "        size = sum(os.path.getsize(os.path.join(dp, f)) for f in fn) / 1e6\n"
             "        print(\"  \" * depth + os.path.basename(dp) + f\"/  ({size:.1f} MB in files here)\")\n"
             "print(\"\\nUpload to Codabench (validated):\")\n"
             "for z in sorted(glob.glob(f\"{OUT}/*/BEST/submission.zip\")):\n"
             "    print(\"  \", z)\n")
    cells.append(code(files, k + 7))
    tail = ("**Run this notebook as a commit** (Save Version -> Save & Run All): it keeps running "
            "without the browser, and `/kaggle/working/phase3_out` (+ `phase3_data`) becomes the "
            "notebook's output. Attach that output to the next session of this notebook to skip "
            "finished jobs / resume an unfinished one, and to E11.\n\n"
            f"Upload `phase3_out/{nb['folder']}/BEST/submission.zip` to Codabench. "
            "`python rr_select.py --coverage` lists the winners, the rule compliance and any extra "
            "upload needed so that all four Phase-1 models are represented.")
    cells.append(md(tail, k + 8))
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4" if nb["gpu"] else "none",
                   "isGpuEnabled": nb["gpu"], "isInternetEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    modules = {f: open(os.path.join(HERE, f), encoding="utf-8").read() for f in MODULE_FILES}
    md5 = {f: hashlib.md5(modules[f].encode("utf-8")).hexdigest() for f in MODULE_FILES}
    for name, nb in NOTEBOOKS.items():
        with open(os.path.join(out, name + ".ipynb"), "w", encoding="utf-8") as f:
            json.dump(build(name, nb, modules, md5), f, indent=1, ensure_ascii=False)
    with open(os.path.join(out, "MODULE_MD5.json"), "w") as f:
        json.dump(md5, f, indent=1)
    print(f"{len(NOTEBOOKS)} notebooks -> {out}")
    for k, v in md5.items():
        print(f"  {v}  {k}")


if __name__ == "__main__":
    main()
