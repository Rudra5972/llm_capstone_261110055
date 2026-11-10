"""
tests/e2e.py -- end-to-end CPU run of every notebook (P0, E1-E8) with tiny random Phase-1
models and a synthetic Phase_5_LSI dataset laid out like the Kaggle inputs.

    python tests/e2e.py WORKDIR [--exps E1,E2,...] [--inject-failure]

Checks after each run: results_<exp>.csv has one row per planned job and no empty file;
every DONE job has a validated submission.zip, non-empty dev_probs / errors / test_probs;
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
    write_dataset(os.path.join(inp, "phase5-cjpe-data"), 60, 20, 20, seed=11, leak=1)
    return inp


def run(work, inp, exp, extra=(), roots=None):
    out = os.path.join(work, f"work_{exp}", "phase5_out")
    env = dict(os.environ, PYTHONPATH=HERE, PHASE5_INPUT_ROOTS=os.pathsep.join(roots or [inp]),
               CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(HERE, "run_experiment.py"), "--exp", exp, "--out-root",
           out, "--cache-dir", os.path.join(work, f"work_{exp}", "phase5_cache"),
           "--time-budget-h", "1.5", "--epochs-max", "2", "--n-proc", "2",
           "--audit-sample", "60", "--micro-chunks", "16", "--effective-batch", "8",
           *extra]
    if exp == "E1":
        env["PHASE5_INJECT_FAIL"] = "M2_llamastyle:6"          # retry ladder: skip checks
    if exp == "E4":
        env["PHASE5_INJECT_FAIL"] = "M3_mistral_sliding:7"     # retry ladder: micro halved
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    with open(os.path.join(work, f"log_{exp}.txt"), "w") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, out


def check(out, exp_folder, expect_rows):
    d = os.path.join(out, exp_folder)
    rows = list(csv.DictReader(open(os.path.join(d, f"results_{exp_folder}.csv"))))
    assert len(rows) == expect_rows, (len(rows), expect_rows)
    done = [r for r in rows if r["status"] == "DONE"]
    for r in done:
        z = os.path.join(d, r["job"], "submission", "submission.zip")
        with zipfile.ZipFile(z) as zz:
            assert zz.namelist() == ["prediction.json"]
            items = json.loads(zz.read("prediction.json"))
            assert items and all(isinstance(i["pred"], list) and i["pred"] and
                                 all(type(x) is int for x in i["pred"]) and
                                 len(set(i["pred"])) == len(i["pred"]) for i in items)
            valid = set(json.load(open(os.path.join(out, "label_map.json")))["label_ids"])
            assert all(set(i["pred"]) <= valid for i in items)
        for f in ("dev/per_label_dev.csv", "dev/errors.csv"):
            assert sum(1 for _ in open(os.path.join(d, r["job"], f))) > 1, f
        for f in ("dev/dev_probs.npz", "submission/test_probs.npz"):
            assert os.path.getsize(os.path.join(d, r["job"], f)) > 0, f
        assert r["dev_macro_f1"] != "", r
    for r in rows:
        assert "None" not in r.values() and "nan" not in r.values(), r
    assert os.path.exists(os.path.join(out, "INDEX.md"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--exps", default="P0,E1,E2,E3,E4,E5,E6,E7,E8")
    ap.add_argument("--inject-failure", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    inp = setup(a.work)
    summary = {}
    for exp in a.exps.split(","):
        extra = []
        if exp == "E7":
            extra = ["--e7-jobs", "bitfit,repr_spread,repr_cite,loss_asl,loss_focal,cw_none"]
        roots = [inp]
        if exp == "E8":
            roots += [os.path.join(a.work, d) for d in os.listdir(a.work) if d.startswith("work_E")
                      and d != "work_E8"]
        code, out = run(a.work, inp, exp, extra, roots)
        from cjpe_common import EXPERIMENTS
        folder = EXPERIMENTS[exp]["folder"]
        n = {"P0": 5, "E6": 5, "E7": 7, "E8": 0}.get(exp, 4)
        rows = check(out, folder, n) if n else []
        st = {r["job"]: r["status"] for r in rows}
        summary[exp] = {"exit": code, "status": st}
        print(exp, code, st, flush=True)
        if exp == "E8":
            plan = list(csv.DictReader(open(os.path.join(out, "UPLOAD_PLAN.csv"))))
            req = [p for p in plan if p["role"] == "required"]
            print("UPLOAD_PLAN", [(p["experiment"], p["model_id"], p["dev_macro_f1"]) for p in req])
            assert len(req) == 6, plan
            assert len({p["model_id"] for p in req}) == 4
            assert os.path.exists(os.path.join(out, "ALL_RESULTS.csv"))
            assert os.path.exists(os.path.join(out, "ALL_RESULTS_pivot.csv"))
    json.dump(summary, open(os.path.join(a.work, "e2e_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
