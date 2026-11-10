"""
tests/test_e2e.py -- every code path end to end on CPU with tiny models (S5 of the brief).

    python tests/test_e2e.py            (~10-15 min on 2 CPU cores)

Covers: data build + gates, E1..E6 on all four architectures, extras X1 (BitFit) and X2 (KD
control), the CRF head, 2-process DDP (gloo), resume after the time budget, restoring finished
models from an attached earlier output, best-of-4 selection, coverage, aggregate, ensemble,
and submission validation.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from synth import make_phase1_models, make_raw, make_tokenizer  # noqa: E402

W = os.environ.get("E2E_WORKDIR") or tempfile.mkdtemp(prefix="p2e2e_")
FROM = os.environ.get("E2E_FROM", "start")         # start | extras  (reuse an existing workdir)
INP, OUT, DATA = os.path.join(W, "input"), os.path.join(W, "out"), os.path.join(W, "data")
ENV = dict(os.environ, PHASE2_INPUT_ROOTS=INP, PYTHONPATH=ROOT)
FAST = ["--cpu-test", "--out", OUT, "--data-dir", DATA, "--chunk-len", "64", "--train-stride",
        "48", "--infer-stride", "32", "--epochs", "2", "--micro-batch", "8", "--effective-batch",
        "16", "--probe-steps", "3", "--watchdog-min-f1", "-1", "--log-every", "1000"]


def run(args, check=True, env=None):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env or ENV, capture_output=True,
                       text=True)
    if check and p.returncode != 0:
        print(p.stdout[-4000:], p.stderr[-4000:])
        raise AssertionError(f"command failed ({p.returncode}): {args}")
    return p


def done(exp_folder, model_folder, out=OUT):
    return json.load(open(os.path.join(out, exp_folder, model_folder, "DONE.json")))


def check_zip(z):
    with zipfile.ZipFile(z) as f:
        assert f.namelist() == ["prediction.json"]


def main():
    folders = ["M4_bloom_alibi", "M2_llamastyle", "M3_mistral_sliding", "M1_gpt2style"]
    if FROM == "start":
        stage_six(folders)
    stage_rest(folders)


def stage_six(folders):
    tok = make_tokenizer(os.path.join(W, "tok", "bpe_model.json"), vocab_size=500)
    make_raw(os.path.join(INP, "phase2-data"), "dict", n_docs=(10, 10, 4), paras=(25, 60), seed=5)
    make_phase1_models(os.path.join(INP, "p1"), os.path.join(W, "tok", "bpe_model.json"), len(tok))

    # ---- the six notebooks ------------------------------------------------------------------
    for exp, folder, models in [("E1", "E1_full_ft", None), ("E2", "E2_lorma", None),
                                ("E3", "E3_lora", "M4,M1"), ("E4", "E4_adapters", "M2,M3"),
                                ("E5", "E5_partial_ft", "M4,M2"), ("E6", "E6_kd", "M3,M1")]:
        args = ["run_experiment.py", "--exp", exp, *FAST]
        if models:
            args += ["--models", models]
        run(args)
        sel = json.load(open(os.path.join(OUT, folder, "BEST", "SELECTION.json")))
        check_zip(os.path.join(OUT, folder, "BEST", "submission.zip"))
        n = len(models.split(",")) if models else 4
        assert sel["n_models_finished"] == n, (exp, sel)
        print(f"ok  {exp}: {n} models finished, BEST = {sel['winner']}")
    assert all(os.path.exists(os.path.join(OUT, "E1_full_ft", f, "DONE.json")) for f in folders)
    # no resumable checkpoint left behind for finished models
    assert not any(os.path.exists(os.path.join(OUT, "E1_full_ft", f, "checkpoints", "last.pt"))
                   for f in folders)
    # KD student is smaller than its Phase-1 model (layer-pruned)
    assert done("E6_kd", "M3_mistral_sliding")["total_params"] < \
        done("E1_full_ft", "M3_mistral_sliding")["total_params"]
    print("ok  KD students are layer-pruned")


def stage_rest(folders):

    # ---- extras -------------------------------------------------------------------------------
    run(["run_experiment.py", "--exp", "X1", *FAST])                       # BitFit on M4, M1
    assert os.path.exists(os.path.join(OUT, "X1_bitfit", "BEST", "submission.zip"))
    run(["run_experiment.py", "--exp", "X2", "--models", "M2", *FAST])     # KD control
    run(["run_experiment.py", "--exp", "X3_crf", "--technique", "full_ft", "--head", "crf",
         "--models", "M2", *FAST])
    d = done("X3_crf", "M2_llamastyle")
    print(f"ok  extras X1/X2/X3 (CRF dev F1 {d['dev_macro_f1']:.3f})")

    # ---- DDP with 2 processes (gloo on CPU) ---------------------------------------------------
    run(["run_experiment.py", "--exp", "DDP2", "--technique", "lorma", "--models", "M4",
         "--nproc", "2", *FAST])
    d = done("DDP2", "M4_bloom_alibi")
    assert d["train"]["world"] == 2, d["train"]
    print("ok  2-process DDP run finished (world=2)")

    # ---- resume after the time budget ----------------------------------------------------------
    run(["run_experiment.py", "--exp", "RES", "--technique", "lora", "--models", "M2",
         "--epochs", "30", *[x for x in FAST if x not in ("--epochs", "2")], "--skip-probe"],
        check=True)
    rd = os.path.join(OUT, "RES", "M2_llamastyle")
    os.remove(os.path.join(rd, "DONE.json"))
    shutil.rmtree(os.path.join(rd, "checkpoints"))
    cfgp = os.path.join(rd, "logs", "run_config.json")
    cfg = json.load(open(cfgp))
    import time
    cfg["deadline_unix"], cfg["deadline_margin_s"], cfg["patience"] = time.time() + 8, 0, 1000
    json.dump(cfg, open(cfgp, "w"))
    p = run(["ner_train.py", "--run-dir", rd, "--mode", "train"], check=False)
    assert p.returncode == 2, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    assert os.path.exists(os.path.join(rd, "checkpoints", "last.pt"))
    cfg["deadline_unix"] = 0
    json.dump(cfg, open(cfgp, "w"))
    p = run(["ner_train.py", "--run-dir", rd, "--mode", "train"])
    assert "[resume]" in p.stdout, p.stdout[-2000:]
    print("ok  time-budget exit (code 2) + resume from last.pt")

    # ---- restore finished models from an attached earlier output ------------------------------
    inp2 = os.path.join(W, "input2")
    shutil.rmtree(inp2, ignore_errors=True)
    shutil.rmtree(os.path.join(W, "out2"), ignore_errors=True)
    shutil.copytree(INP, inp2)
    shutil.copytree(os.path.join(OUT, "E1_full_ft"), os.path.join(inp2, "old-run", "phase2_out",
                                                                  "E1_full_ft"))
    out2 = os.path.join(W, "out2")
    env2 = dict(ENV, PHASE2_INPUT_ROOTS=inp2)
    fast2 = [x if x != OUT else out2 for x in FAST]
    p = run(["run_experiment.py", "--exp", "E1", *fast2], env=env2)
    assert p.stdout.count("already finished") == 4, p.stdout[-3000:]
    print("ok  finished models restored from an attached dataset and skipped")

    # ---- coverage, aggregate, ensemble, validator ----------------------------------------------
    p = run(["select_best.py", "--coverage", "--out", OUT])
    assert "Winners" in p.stdout
    run(["aggregate.py", "--out", OUT])
    assert os.path.exists(os.path.join(OUT, "results.md"))
    ens_out = os.path.join(OUT, "X5_ensemble")
    run(["ner_predict.py", "--ensemble",
         os.path.join(OUT, "E1_full_ft", "BEST") + "," + os.path.join(OUT, "E2_lorma", "BEST"),
         "--out", ens_out])
    check_zip(os.path.join(ens_out, "submission", "submission.zip"))
    p = run(["validate_submission.py", os.path.join(OUT, "E2_lorma", "BEST", "submission.zip"),
             "--data-dir", DATA])
    assert json.loads(p.stdout)["ok"]
    print("ok  coverage, aggregate, ensemble (X5), validator")
    print(f"\nALL END-TO-END CHECKS PASSED  (workdir {W})")


if __name__ == "__main__":
    main()
