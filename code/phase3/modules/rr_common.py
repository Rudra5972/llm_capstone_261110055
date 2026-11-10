"""
rr_common.py -- the Phase-3 (Rhetorical Role prediction) spec, registries and small helpers.

Torch-free. Every other rr_* module imports its constants from here, so a change in the brief
(it is a dynamic document) is a one-line edit of `SPEC` (and a regeneration of the notebooks,
whose md5 check then covers the new value everywhere).

Reused byte-identical from Phase 2 (never edited): ner_common.py (file discovery, atomic JSON,
disk guard, Phase-1 model discovery by the `arch` field), ner_models.py (Trunk), ner_peft.py
(LoRA, LoRMA+), phase1_models.py, bpe_tokenizer.py.

Inputs : nothing (constants) / environment variable PHASE3_INPUT_ROOTS (os.pathsep-separated)
Outputs: SPEC, EXPERIMENTS, TECHNIQUES, defaults, helpers
"""
import glob
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

from ner_common import (MODELS, STUDENT_LAYERS, DiskFull, dir_size_gb, disk_guard, file_md5,
                        find_files, find_phase1_models, free_gb, read_json, write_json)

__all__ = ["SPEC", "MODELS", "STUDENT_LAYERS", "EXPERIMENTS", "TECHNIQUES", "DEFAULT_LR",
           "DEFAULT_EPOCHS", "MAX_TRAINABLE_PCT", "DEFAULT_MODEL_ORDER", "OUT_ROOT", "DATA_DIR",
           "CODE_DIR", "EXIT", "GateFailure", "default_input_roots", "find_files",
           "find_phase1_models", "read_json", "write_json", "file_md5", "disk_guard", "DiskFull",
           "dir_size_gb", "free_gb", "module_md5s", "iter_finished_runs", "log_to"]

# =============================================================================================
# The brief (CS-781 Capstone, Phase 3 -- Rhetorical Role Prediction). Edit HERE only.
# =============================================================================================
SPEC: Dict = {
    "phase": 3,
    "task": "rhetorical_role_prediction",
    "hf_repo": "Exploration-Lab/CS-781-Capstone",
    "hf_folder": "Phase_3_RR",
    # key -> path inside hf_folder. Train/dev: [{"id", "text": [..], "labels": [..]}];
    # test: {id: [sentences]} (the loader also accepts [{"id", "text"}]).
    "files": {
        "CL_train": "Train/CL_train.json", "CL_dev": "Train/CL_dev.json",
        "IT_train": "Train/IT_train.json", "IT_dev": "Train/IT_dev.json",
        "label_vocab": "Train/label_vocab.json",
        "CL_test": "Test/CL_test.json", "IT_test": "Test/IT_test.json",
    },
    "domains": ["CL", "IT"],
    # the 13 official labels, in label_vocab.json index order, exact spelling ("None" is a string)
    "labels": ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent",
               "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled",
               "RulingByLowerCourt", "RatioOfTheDecision", "RulingByPresentCourt", "None"],
    "metric": "macro_f1",                 # primary; macro precision / recall are secondary
    "optimise_pooling": "pooled",         # brief does not say pooled vs per-domain: log both
    "submission_entry": "prediction.json",  # must sit at the ROOT of submission.zip
    "submission_zip": "submission.zip",
    "submissions": {"min": 6, "max": 40, "max_per_day": 10},
    "rules": {
        "min_experiments": 6, "min_peft_techniques": 5, "compulsory": ["lorma"],
        "min_kd_experiments": 1, "all_models_required": True,
        "foundation_params": [50e6, 300e6],   # Phase-1 models; pruned KD students may be smaller
    },
    # technique menu of the brief (full_ft is NOT on it: reference only)
    "peft_menu": ["lora", "qlora", "adalora", "rslora", "lorma", "prefix", "prompt", "ia3"],
    "kd_technique": "kd",
    "context": 1024,                      # Phase-1 models were trained on 1024-token blocks
    "eos_id": 0, "pad_id": 1, "unk_id": 2,
    "deadline": "2026-10-16 23:59 (strict)",
    "codabench_user_format": "a_123456",
    "allowed_imports": ["torch", "numpy", "stdlib"],
}

# =============================================================================================
# Registries
# =============================================================================================
DEFAULT_MODEL_ORDER: List[str] = ["M4", "M2", "M3", "M1"]   # strongest Phase-1 LM first

# technique -> description; `peft` marks techniques that count toward the 5-technique rule
TECHNIQUES: Dict[str, Dict] = {
    "lorma":   {"peft": True,  "title": "LoRMA+ (multiplicative low-rank, compulsory)"},
    "lora":    {"peft": True,  "title": "LoRA"},
    "rslora":  {"peft": True,  "title": "rsLoRA (rank-stabilised LoRA)"},
    "ia3":     {"peft": True,  "title": "IA3"},
    "adalora": {"peft": True,  "title": "AdaLoRA (SVD + importance-based rank allocation)"},
    "qlora":   {"peft": True,  "title": "QLoRA (simulated NF4 base + LoRA)"},
    "prefix":  {"peft": True,  "title": "Prefix Tuning"},
    "prompt":  {"peft": True,  "title": "Prompt Tuning"},
    "kd":      {"peft": False, "title": "Knowledge distillation (pruned student)"},
    "full_ft": {"peft": False, "title": "Full fine-tuning (reference only)"},
}

EXPERIMENTS: Dict[str, Dict] = {
    "E0":  {"folder": "E0_full_ft", "technique": "full_ft", "counts": False, "priority": 3,
            "title": "Full fine-tuning (reference baseline, not counted)"},
    "E1":  {"folder": "E1_lorma", "technique": "lorma", "counts": True, "priority": 1,
            "title": "LoRMA+ (compulsory)"},
    "E2":  {"folder": "E2_lora", "technique": "lora", "counts": True, "priority": 1,
            "title": "LoRA"},
    "E3":  {"folder": "E3_rslora", "technique": "rslora", "counts": True, "priority": 2,
            "title": "rsLoRA"},
    "E4":  {"folder": "E4_ia3", "technique": "ia3", "counts": True, "priority": 2,
            "title": "IA3"},
    "E5":  {"folder": "E5_adalora", "technique": "adalora", "counts": True, "priority": 2,
            "title": "AdaLoRA"},
    "E6":  {"folder": "E6_kd", "technique": "kd", "counts": True, "priority": 1,
            "title": "Knowledge distillation (self-contained full-FT teacher -> pruned student)"},
    "E7":  {"folder": "E7_qlora", "technique": "qlora", "counts": True, "priority": 3,
            "title": "QLoRA (simulated NF4)"},
    "E8":  {"folder": "E8_prefix", "technique": "prefix", "counts": True, "priority": 3,
            "title": "Prefix Tuning"},
    "E9":  {"folder": "E9_prompt", "technique": "prompt", "counts": True, "priority": 3,
            "title": "Prompt Tuning"},
    "E10": {"folder": "E10_ablations", "technique": "ablation", "counts": False, "priority": 3,
            "title": "Ablations: context, head, loss, domain handling"},
    "E11": {"folder": "E11_ensemble", "technique": "ensemble", "counts": False, "priority": 9,
            "title": "Ensemble of BEST models + final report"},
}
PILOT_FOLDER = "P0_pilot"

# Section 6 starting points; the LR probe multiplies `lr` by {1/3, 1, 3}.
DEFAULT_LR: Dict[str, float] = {
    "lorma": 1e-3, "lora": 1e-3, "rslora": 1e-3, "adalora": 1e-3, "qlora": 1e-3,
    "ia3": 3e-3, "prefix": 2e-3, "prompt": 3e-3, "full_ft": 5e-5, "kd": 1e-4,
}
DEFAULT_EPOCHS: Dict[str, int] = {
    "lorma": 10, "lora": 10, "rslora": 10, "adalora": 10, "qlora": 10,
    "ia3": 12, "prefix": 15, "prompt": 15, "full_ft": 6, "kd": 8,
}
# upper bound on trainable % of the (real-size) model, asserted by selftest --full-size
MAX_TRAINABLE_PCT: Dict[str, float] = {
    "lorma": 5.0, "lora": 5.0, "rslora": 8.0, "adalora": 5.0, "qlora": 5.0,
    "ia3": 0.5, "prefix": 6.0, "prompt": 0.5,
}

# =============================================================================================
# Paths / exit codes
# =============================================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/kaggle/working/phase3_out"
DATA_DIR = "/kaggle/working/phase3_data"
OUT_MARKER = "phase3_out"                 # folder name used to find earlier outputs in inputs
EXIT = {"ok": 0, "error": 1, "missing_dependency": 2, "gate_failure": 3,
        "validation_failure": 4}

MODULE_FILES = ["bpe_tokenizer.py", "phase1_models.py", "ner_common.py", "ner_models.py",
                "ner_peft.py", "rr_common.py", "rr_data.py", "rr_eval.py", "rr_models.py",
                "rr_peft.py", "rr_kd.py", "rr_train.py", "rr_predict.py", "rr_select.py",
                "rr_aggregate.py", "rr_ensemble.py", "run_experiment.py", "selftest.py"]


class GateFailure(RuntimeError):
    """A 'stop and ask' condition: the pipeline must not produce a submission."""


def default_input_roots() -> List[str]:
    """Where attached Kaggle datasets live; PHASE3_INPUT_ROOTS (os.pathsep-separated) overrides."""
    env = os.environ.get("PHASE3_INPUT_ROOTS")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    return ["/kaggle/input"]


def module_md5s(code_dir: str = CODE_DIR) -> Dict[str, Optional[str]]:
    """md5 of every shared module (recorded in each run_config.json; no git needed)."""
    out = {}
    for f in MODULE_FILES:
        p = os.path.join(code_dir, f)
        out[f] = file_md5(p) if os.path.exists(p) else None
    return out


def iter_finished_runs(out_root: Optional[str], input_roots=()) -> Iterator[Tuple[str, str, str]]:
    """Yield (exp_folder, job_folder, run_dir) for every finished run (DONE.json) directly under
    `out_root` (<exp>/<job>/DONE.json) and under any `phase3_out` folder inside the attached
    inputs. BEST folders are skipped. First occurrence wins."""
    seen = set()
    if out_root and os.path.isdir(out_root):
        for d in sorted(glob.glob(os.path.join(os.path.abspath(out_root), "*", "*", "DONE.json"))):
            exp_f, job_f = d.split(os.sep)[-3], d.split(os.sep)[-2]
            if job_f != "BEST" and (exp_f, job_f) not in seen:
                seen.add((exp_f, job_f))
                yield exp_f, job_f, os.path.dirname(d)
    for d in find_files("DONE.json", list(input_roots), max_hits=20000):
        parts = d.split(os.sep)
        if OUT_MARKER not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index(OUT_MARKER)
        if len(parts) - i != 4 or parts[-2] == "BEST":
            continue
        if (parts[i + 1], parts[-2]) not in seen:
            seen.add((parts[i + 1], parts[-2]))
            yield parts[i + 1], parts[-2], os.path.dirname(d)


def student_layers(mid: str, phase1_dir: str) -> List[int]:
    """Every second layer of the Phase-1 model (= STUDENT_LAYERS for the real 12/14-layer
    models; derived from config.json so that tiny test models work too)."""
    cfg = read_json(os.path.join(phase1_dir, "config.json"))["config"]
    n = cfg.get("n_layer", cfg.get("num_hidden_layers"))
    ids = list(range(0, n, 2))
    if n in (12, 14):
        assert ids == STUDENT_LAYERS[mid], (ids, STUDENT_LAYERS[mid])
    return ids


def log_to(path: Optional[str], msg: str, prefix: str = "") -> None:
    """Print a line and append it (timestamped) to a log file."""
    line = f"{prefix}{msg}"
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + line + "\n")
