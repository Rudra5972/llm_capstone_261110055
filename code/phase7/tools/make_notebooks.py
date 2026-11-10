"""
make_notebooks.py -- generate the Phase-7 Kaggle notebooks (dev tool, not a shared module).

Every notebook embeds the SAME module block (%%writefile cells) with the same EXPECTED_MD5,
then runs one experiment. No cell ever raises (rule R1): a notebook that errors during
"Save & Run All" saves NO output, which looks like an empty result.

    python tools/make_notebooks.py OUT_DIR
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from cjpe_common import EXPERIMENTS, MODULE_FILES  # noqa: E402

CODE = "/kaggle/working/phase7"
OUT = "/kaggle/working/phase7_out"

ATTACH = """### Inputs to attach (Add Input)
1. **The four Phase-1 datasets** (`01-train-gpt2style`, `02-train-llamastyle`,
   `03-train-mistral-sliding`, `04-train-bloom-alibi`: `final/model.pt` + `config.json`) --
   the SAME weights as every phase since Phase 2. Models are recognised by the `arch` field of
   `config.json`, so dataset names do not matter.
2. **The `bpe_model` dataset** (`bpe_model.json`; found anywhere under /kaggle/input).
3. **Data (one of):** (a) Add-ons -> Secrets -> `HF` (tick it for THIS notebook) **and**
   Settings -> Internet **On** -- the six Phase_7_PCR files are downloaded; or (b) a Kaggle
   dataset holding `Phase_7_PCR/{Train,Dev,Test}/{split}.json` + `{split}_candidates.json`
   (then Internet can stay Off: nothing else is downloaded in Phase 7).
4. *Optional:* the output of an earlier session of THIS notebook (finished models are skipped,
   an unfinished one resumes from `last.pt`). For **E6** you may also attach a finished
   **E2_full_ft** output: its M4 run is reused as the KD teacher (saves ~40% of E6).
"""

NOTEBOOKS = {
    "E8_report": {
        "exp": "E8", "gpu": False, "budget": 0.5, "minutes": "2-10",
        "what": "CPU only. Attach the OUTPUTS of E1-E6 (and E7 if run). Writes ALL_RESULTS.csv "
                "(+ pivots), UPLOAD_PLAN.csv (the six uploads: one per technique, all four "
                "Phase-1 models covered, pairwise different predictions), a score-fusion ensemble "
                "submission (optimistic dev score, not a counted experiment), the retrieval "
                "error analysis and plots.",
        "extra": ""},
}
COMMON = (" Every model is a shared-weight bi-encoder (query and candidate through the same "
          "Phase-1 trunk, mean-pooled chunk vectors, identity-initialised projection, learned "
          "temperature), trained with InfoNCE over in-batch + BM25 hard negatives (GradCache); "
          "the final ranking fuses the dense score with BM25 (weight tuned on Dev, >= 0.2).")
WHAT = {
    "lorma": "LoRMA+ (compulsory): y = (I + s*B*A) W0 x on q/k/v/o and the FFN projections, "
             "rank 16, alpha 32 (Phase-2 implementation reused unchanged)." + COMMON,
    "full_ft": "Full fine-tuning of every Phase-1 parameter (lr 3e-5)." + COMMON,
    "partial_ft": "Partial / layer-wise fine-tuning: top 6 blocks + final norm, layer-wise LR "
                  "decay 0.85 (lower layers frozen)." + COMMON,
    "qlora": "QLoRA: every adapted linear stored as NF4 4-bit (blockwise absmax 64) + LoRA r=16 "
             "trained on top." + COMMON,
    "prefix": "Prefix tuning: 20 virtual positions whose activations at EVERY block input are "
              "free parameters (Li & Liang 2021), initialised from real text." + COMMON,
    "kd": "Knowledge distillation: a full fine-tuned TEACHER (M4, all layers) is trained first "
          "(or reused from an attached E2 output); then each Phase-1 model pruned to every "
          "second layer is a STUDENT trained ONLINE on 0.5 InfoNCE + 0.5 T^2 KL(teacher || "
          "student) over the same query x candidate score matrix (T=2)." + COMMON,
    "extras": "Extras (not counted), all on M4: IA3 and prompt tuning (techniques), the zero-shot "
              "Phase-1 encoder (no training: how Phase-1 representations transfer), and with "
              "LoRMA on a FIXED plan (1 epoch, 50% of pairs): last-token vs mean pooling, "
              "bidirectional vs causal attention, first-chunk-only vs whole-case chunks, "
              "in-batch negatives only (no BM25 hard negatives).",
}
for e, s in EXPERIMENTS.items():
    if e == "E8":
        continue
    b = 4.0 if e == "E6" else 3.5
    NOTEBOOKS[s["folder"]] = {"exp": e, "gpu": True, "budget": b, "minutes": f"<= {int(60 * b + 10)}",
                              "what": WHAT[s["technique"]], "extra": ""}


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(text, i):
    return {"cell_type": "markdown", "id": f"md{i:02d}", "metadata": {}, "source": src(text)}


def code(text, i):
    return {"cell_type": "code", "execution_count": None, "id": f"c{i:02d}", "metadata": {},
            "outputs": [], "source": src(text)}


HEALTHY = """### What a healthy run prints in the first minutes
* `all 25 modules verified`
* `[phase1] M1 gpt2style -> ...` (all four found), `[bpe] using ...`
* `[data] train: N queries ..., pool M candidates; ... 0 relevant ids not in the pool` (same for
  dev / test), `[cache] ...`, `[bm25] tuned on dev: ... Micro-F1@7 0.xxxx`, the `[audit]` line
* per job: `[check] ... pass` (4 lines), `[preflight] train ... s/step, encode ... tok/s`,
  `[zero-shot] ...`, `[plan] CHOSEN: epochs=... train_fraction=...`, then `[eval] epoch 0.50: dev
  proxy MAP ...` rising above the zero-shot value
* at the end of a job: `DONE dev Micro-F1@7 0.xxxx ... dense only ..., BM25 only ...; submission
  validated`
"""


def build(name, nb, modules, md5):
    cells = []
    exp = nb["exp"]
    folder = EXPERIMENTS[exp]["folder"]
    head = (f"# CS-781 Phase 7 (Prior Case Retrieval) -- {name}\n\n{nb['what']}\n\n"
            + (f"**Accelerator: GPU T4 x2. Internet: ON if the data comes from the HF secret.** "
               f"Expected runtime: {nb['minutes']} min (time budget {nb['budget']} h; every "
               "job plans its epochs / data fraction / inference chunks from measured throughput and "
               "stops cleanly at its share of the budget). Two models train at the same time, "
               "one per GPU (no DDP).\n\n" if nb["gpu"] else
               "**Accelerator: None (CPU).**\n\n")
            + ("Run it with **Save Version -> Save & Run All (Commit)** so the output persists.\n\n"
               + ATTACH + "\n" + HEALTHY))
    if exp == "E8":
        head += ("\n**E8 inputs:** the four Phase-1 datasets are NOT needed. Attach the saved "
                 "outputs of E1_lorma ... E6_kd (and E7_extras). No data, no GPU, no secret.\n")
    cells.append(md(head, 0))
    c1 = ("import os, sys, json, hashlib, subprocess, glob, time, traceback\n"
          f"CODE = \"{CODE}\"; OUT = \"{OUT}\"\n"
          "os.makedirs(CODE, exist_ok=True); os.makedirs(OUT, exist_ok=True)\n"
          "print(sys.version.split()[0], time.strftime('%Y-%m-%d %H:%M:%S'))\n"
          "try:\n"
          "    import torch\n"
          "    print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count(),\n"
          "          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
          "except Exception as e:\n"
          "    print('torch not importable:', e)\n"
          "print('inputs:', sorted(os.listdir('/kaggle/input')) if os.path.isdir('/kaggle/input') "
          "else 'none')\n"
          "EXPECTED_MD5 = " + json.dumps(md5, indent=1) + "\n")
    cells.append(code(c1, 1))
    for i, f in enumerate(MODULE_FILES):
        cells.append(code(f"%%writefile {CODE}/{f}\n" + modules[f], 2 + i))
    k = 2 + len(MODULE_FILES)
    verify = ("# every module must be byte-identical to the generated version (same code in all "
              "notebooks)\n"
              "try:\n"
              "    bad = [rel for rel, h in EXPECTED_MD5.items()\n"
              "           if hashlib.md5(open(os.path.join(CODE, rel), 'rb').read()).hexdigest() != h]\n"
              "    if bad:\n"
              "        print('!' * 80 + f'\\n!!! modules differ from the generated version: {bad}\\n' + '!' * 80)\n"
              "    else:\n"
              "        print(f'all {len(EXPECTED_MD5)} modules verified')\n"
              "except Exception as e:\n"
              "    print('module check failed:', e)\n")
    cells.append(code(verify, k))
    disc = ("# what is attached (never raises)\n"
            "try:\n"
            "    sys.path.insert(0, CODE)\n"
            "    from cjpe_common import default_input_roots, find_files, find_phase1_models\n"
            "    from cjpe_data import get_hf_token\n"
            "    print('Phase-1 weights:', json.dumps(find_phase1_models(default_input_roots()), indent=1))\n"
            "    for n in ('train.json', 'train_candidates.json', 'dev.json', 'dev_candidates.json', 'test.json', 'test_candidates.json'):\n"
            "        print(n, '->', find_files(n, default_input_roots())[:4])\n"
            "    t = get_hf_token()\n"
            "    print('HF secret found:', 'yes (length %d)' % len(t) if t else 'no')\n"
            "    print('earlier phase7_out inputs:', glob.glob('/kaggle/input/**/phase7_out', recursive=True)[:6])\n"
            "except Exception as e:\n"
            "    print('discovery failed:', e); traceback.print_exc()\n")
    cells.append(code(disc, k + 1))
    args = f"--exp {exp} --resume auto --time-budget-h {nb['budget']}"
    if exp in ("E1", "E2", "E3", "E4", "E5", "E6"):
        args = f"--exp {exp} --models M4,M2,M3,M1 --gpu-mode jobs --resume auto " \
               f"--time-budget-h {nb['budget']}"
    run = ("# the run (streams its log; a non-zero exit code is PRINTED, never raised -- rule R1)\n"
           "# to split the four models over two accounts: add e.g. '--models M4,M2' here and\n"
           "# '--models M3,M1' on the other account (then attach both outputs to E8).\n"
           f"ARGS = \"{args}\"\n"
           "RC = None\n"
           "try:\n"
           "    p = subprocess.Popen([sys.executable, '-u', os.path.join(CODE, 'run_experiment.py')] + ARGS.split(),\n"
           "                         cwd=CODE, env=dict(os.environ, PYTHONPATH=CODE, PYTHONUNBUFFERED='1'),\n"
           "                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
           "    for line in p.stdout:\n"
           "        print(line, end='', flush=True)\n"
           "    RC = p.wait()\n"
           "except Exception as e:\n"
           "    print('launch failed:', e); traceback.print_exc()\n"
           "print(f'\\n===== run_experiment.py exit code: {RC} '\n"
           "      '(0 ok, 1 nothing finished, 2 data / weights missing) =====')\n")
    cells.append(code(run, k + 2))
    final = (
        "# R7: final verification + aggregation (never raises)\n"
        f"FOLDER = \"{folder}\"\n"
        "try:\n"
        "    sys.path.insert(0, CODE)\n"
        "    from cjpe_common import read_csv\n"
        "    from validate_submission import validate\n"
        "    ld = lambda f: json.load(open(f'{OUT}/{f}')) if os.path.exists(f'{OUT}/{f}') else None\n"
        "    test_ids, pool = ld('test_ids.json'), ld('test_pool_ids.json')\n"
        "    rows = read_csv(f'{OUT}/{FOLDER}/results_{FOLDER}.csv')\n"
        "    print(f\"{'job':<34}{'status':<13}{'F1@7':>8}{'dense':>8}{'BM25':>8}{'w':>5}{'frac':>6}{'min':>7}  check\")\n"
        "    for r in rows:\n"
        "        z = f\"{OUT}/{FOLDER}/{r['job']}/submission/submission.zip\"\n"
        "        verdict = ''\n"
        "        if r['status'] in ('DONE', 'DEGENERATE'):\n"
        "            v = validate(z, test_ids, pool) if (test_ids and os.path.exists(z)) else {'ok': False, 'errors': ['zip missing']}\n"
        "            files_ok = all(os.path.exists(f\"{OUT}/{FOLDER}/{r['job']}/{f}\")\n"
        "                           for f in ('dev/dev_predictions.jsonl', 'dev/fusion_curve.csv', 'dev/dev_report.json', 'dev/scores.npz'))\n"
        "            verdict = 'PASS' if (v['ok'] and files_ok and r.get('dev_score')) else f\"FAIL {v.get('errors')}\"\n"
        "        g = lambda k: f'{float(r[k]):.4f}' if r.get(k) else ''\n"
        "        print(f\"{r['job']:<34}{r['status']:<13}{g('dev_micro_f1'):>8}{g('dev_dense_f1'):>8}{g('dev_bm25_f1'):>8}{(r.get('dense_weight') or '')[:4]:>5}\"\n"
        "              f\"{(r.get('train_fraction') or '')[:4]:>6}{(r.get('train_minutes') or '')[:5]:>7}  {verdict} {r.get('fail_reason', '')[:80]}\")\n"
        "    best = f'{OUT}/{FOLDER}/BEST/submission.zip'\n"
        "    if os.path.exists(best):\n"
        "        print('\\nBEST:', json.dumps({k: v for k, v in json.load(open(f'{OUT}/{FOLDER}/BEST/SELECTION.json')).items() if k != 'ranking'}))\n"
        "        print('UPLOAD THIS:', best)\n"
        "    elif FOLDER != 'E8_report':\n"
        "        print('!' * 80 + '\\n!!! NO MODEL FINISHED in this notebook. Read the fail_reason column above and\\n'\n"
        "              f'!!! {OUT}/INDEX.md / BLOCKED.json; the logs are in {OUT}/{FOLDER}/<job>/logs/console.log\\n' + '!' * 80)\n"
        "    for f in ('INDEX.md', 'UPLOAD_PLAN.csv', 'BLOCKED.json'):\n"
        "        if os.path.exists(f'{OUT}/{f}'):\n"
        "            print(f'\\n----- {f} -----'); print(open(f'{OUT}/{f}').read()[:6000])\n"
        "    tot = 0\n"
        "    for dp, dn, fn in sorted(os.walk(OUT)):\n"
        "        depth = dp.count(os.sep) - OUT.count(os.sep)\n"
        "        size = sum(os.path.getsize(os.path.join(dp, f)) for f in fn)\n"
        "        tot += size\n"
        "        if depth <= 2:\n"
        "            print('  ' * depth + os.path.basename(dp) + f'/  ({size / 1e6:.1f} MB in files here)')\n"
        "    print(f'total output {tot / 1e9:.2f} GB')\n"
        "except Exception as e:\n"
        "    print('final check failed:', e); traceback.print_exc()\n"
        "print('\\nREMINDER: Save Version -> Save & Run All (Commit) keeps /kaggle/working/phase7_out as this '\n"
        "      'notebook\\'s output. Download BEST/submission.zip and results_*.csv from the Output tab.')\n")
    cells.append(code(final, k + 3))
    cells.append(md("Upload `phase7_out/" + folder + "/BEST/submission.zip` to Codabench (or any "
                    "`<job>/submission/submission.zip`). After E1-E6 are done, run `E8_report` "
                    "with all outputs attached: it writes `UPLOAD_PLAN.csv` (six distinct "
                    "uploads, all four Phase-1 models).", k + 4))
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


if __name__ == "__main__":
    main()
