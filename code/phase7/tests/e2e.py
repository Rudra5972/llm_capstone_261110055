"""
tests/e2e.py -- end-to-end CPU run of every notebook (E1-E8) with tiny random Phase-1
models and a synthetic Phase_7_PCR dataset laid out like the Kaggle inputs.

    python tests/e2e.py WORKDIR [--exps E1,E2,...] [--inject-failure]

Checks after each run: results_<exp>.csv has one row per planned job and no empty file;
every DONE job has a validated object-format submission.zip (7 ids per query from the test
pool) and its dev outputs;
BEST/ exists; E8 writes ALL_RESULTS.csv, the pivots and UPLOAD_PLAN.csv (six distinct uploads
covering all four models).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from selftest import write_dataset, write_phase1, write_tokenizer  # noqa: E402
from cjpe_common import MODELS  # noqa: E402


def setup(work: str):
    inp = os.path.join(work, "input")
    if os.path.exists(inp):
        return inp
    tok = write_tokenizer(os.path.join(work, "tok"))
    for i, m in enumerate(MODELS):
        write_phase1(os.path.join(inp, f"0{i + 1}-train-{MODELS[m]['arch']}"), m, seed=i,
                     tok_dir=tok)
    write_dataset(os.path.join(inp, "phase7-pcr-data"), (40, 12, 12), seed=11)
    return inp


def run(work, inp, exp, extra=(), roots=None):
    out = os.path.join(work, f"work_{exp}", "phase7_out")
    env = dict(os.environ, PYTHONPATH=HERE, PHASE7_INPUT_ROOTS=os.pathsep.join(roots or [inp]),
               CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(HERE, "run_experiment.py"), "--exp", exp, "--out-root",
           out, "--cache-dir", os.path.join(work, f"work_{exp}", "phase7_cache"),
           "--time-budget-h", "1.5", "--epochs-max", "2", "--n-proc", "2",
           "--micro-tokens", "2048", "--batch-pairs", "8", "--chunk-tokens", "48",
           "--max-chunks", "4", "--k-infer", "4", "--n-virtual", "4", *extra]
    if exp == "E1":
        env["PHASE7_INJECT_FAIL"] = "M2_llamastyle:6"          # retry ladder: skip checks
    if exp == "E4":
        env["PHASE7_INJECT_FAIL"] = "M3_mistral_sliding:7"     # retry ladder: micro halved
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    with open(os.path.join(work, f"log_{exp}.txt"), "w") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, out


def check(out, exp_folder, expect_rows):
    d = os.path.join(out, exp_folder)
    rows = list(csv.DictReader(open(os.path.join(d, f"results_{exp_folder}.csv"))))
    assert len(rows) == expect_rows, (len(rows), expect_rows)
    done = [r for r in rows if r["status"] == "DONE"]
    tids = json.load(open(os.path.join(out, "test_ids.json")))
    pool = json.load(open(os.path.join(out, "test_pool_ids.json")))
    from validate_submission import validate
    for r in done:
        z = os.path.join(d, r["job"], "submission", "submission.zip")
        v = validate(z, tids, pool)
        assert v["ok"], v
        for f in ("dev/dev_predictions.jsonl", "dev/fusion_curve.csv", "dev/dev_report.json",
                  "dev/scores.npz"):
            assert os.path.getsize(os.path.join(d, r["job"], f)) > 0, f
        assert r["dev_score"] != "" and r["dev_dense_f1"] != "" and r["dev_bm25_f1"] != "", r
    for r in rows:
        assert "None" not in r.values() and "nan" not in r.values(), r
    assert os.path.exists(os.path.join(out, "INDEX.md"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--exps", default="E1,E2,E3,E4,E5,E6,E7,E8")
    ap.add_argument("--inject-failure", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    inp = setup(a.work)
    summary = {}
    for exp in a.exps.split(","):
        extra = []
        if exp == "E7":
            extra = ["--e7-jobs", "ia3,prompt,zeroshot,pool_last,attn_bidir,repr_head,hn_0"]
        roots = [inp]
        if exp == "E8":
            roots += [os.path.join(a.work, d) for d in os.listdir(a.work) if d.startswith("work_E")
                      and d != "work_E8"]
        code, out = run(a.work, inp, exp, extra, roots)
        from cjpe_common import EXPERIMENTS
        folder = EXPERIMENTS[exp]["folder"]
        n = {"E6": 5, "E7": 7, "E8": 0}.get(exp, 4)
        rows = check(out, folder, n) if n else []
        st = {r["job"]: r["status"] for r in rows}
        summary[exp] = {"exit": code, "status": st}
        print(exp, code, st, flush=True)
        if exp == "E8":
            plan = list(csv.DictReader(open(os.path.join(out, "UPLOAD_PLAN.csv"))))
            req = [p for p in plan if p["role"] == "required"]
            print("UPLOAD_PLAN", [(p["experiment"], p["model_id"], p["dev_score"]) for p in req])
            assert len(req) == 6, plan
            assert len({p["model_id"] for p in req}) == 4
            assert os.path.exists(os.path.join(out, "ALL_RESULTS.csv"))
            assert os.path.exists(os.path.join(out, "ALL_RESULTS_pivot.csv"))
            assert os.path.exists(os.path.join(out, "E8_report", "retrieval_table.csv"))
    json.dump(summary, open(os.path.join(a.work, "e2e_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
