"""
cjpe_common.py -- Phase-7 (Prior Case Retrieval) spec, registries and small helpers.

Torch-free. Every other cjpe_* module imports its constants from here.

Reused byte-identical from Phases 2/3 (never edited): ner_common.py (file discovery, atomic
JSON, disk guard, Phase-1 model discovery by the `arch` field), ner_models.py (Trunk),
ner_peft.py (LoRA, LoRMA+, adapters, BitFit, partial FT), phase1_models.py, bpe_tokenizer.py.
"""
import csv
import glob
import io
import json
import math
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

from ner_common import (MODELS, STUDENT_LAYERS, DiskFull, dir_size_gb, disk_guard, file_md5,
                        find_files, find_phase1_models, free_gb, read_json, write_json)

__all__ = ["SPEC", "MODELS", "STUDENT_LAYERS", "EXPERIMENTS", "DEFAULT_LR", "DEFAULT_EPOCHS",
           "MAX_TRAINABLE_PCT", "DEFAULT_MODEL_ORDER", "OUT_ROOT", "CACHE_DIR", "RAW_DIR",
           "CODE_DIR", "EXIT", "JOB_EXIT", "CSV_COLUMNS", "EPOCH_COLUMNS", "default_input_roots",
           "find_files", "find_phase1_models", "read_json", "write_json", "file_md5",
           "disk_guard", "DiskFull", "dir_size_gb", "free_gb", "module_md5s", "log_to",
           "write_csv", "read_csv", "now_str", "student_layers", "iter_out_dirs", "fmt",
           "MODULE_FILES"]

# =============================================================================================
# The brief (CS-781 Capstone, Phase 7 -- Prior Case Retrieval). Edit HERE only.
# =============================================================================================
SPEC: Dict = {
    "phase": 7,
    "task": "prior_case_retrieval",
    "hf_repo": "Exploration-Lab/CS-781-Capstone",
    "hf_folder": "Phase_7_PCR",
    "files": {"train": "Train/train.json", "train_candidates": "Train/train_candidates.json",
              "dev": "Dev/dev.json", "dev_candidates": "Dev/dev_candidates.json",
              "test": "Test/test.json", "test_candidates": "Test/test_candidates.json"},
    "metric": "micro_f1@7",                  # primary; Micro-P@7 / Micro-R@7 secondary
    "submission_entry": "prediction.json",   # {"query_id": [7 candidate ids]}, zip root
    "submission_zip": "submission.zip",
    "submissions": {"min": 6, "max": 40, "max_per_day": 10},
    "deadline": "2026-11-07 23:59 IST",
    "rules": {"min_experiments": 6, "min_peft_techniques": 5, "compulsory": ["lorma"],
              "min_kd_experiments": 1, "all_models_required": True,
              "foundation_params": [50e6, 300e6]},
}

DEFAULT_MODEL_ORDER = ["M4", "M2", "M3", "M1"]

# The six counted experiments (PDF 13.2.1 list; the technique is the ONLY difference) + extras.
EXPERIMENTS: Dict[str, Dict] = {
    "E1": {"folder": "E1_lorma", "technique": "lorma", "counts": True,
           "title": "LoRMA+ (compulsory)"},
    "E2": {"folder": "E2_full_ft", "technique": "full_ft", "counts": True,
           "title": "Full fine-tuning"},
    "E3": {"folder": "E3_partial_ft", "technique": "partial_ft", "counts": True,
           "title": "Partial / layer-wise fine-tuning (top 6 blocks, layer-wise LR decay)"},
    "E4": {"folder": "E4_qlora", "technique": "qlora", "counts": True,
           "title": "QLoRA (NF4 4-bit frozen base + LoRA)"},
    "E5": {"folder": "E5_prefix", "technique": "prefix", "counts": True,
           "title": "Prefix tuning (per-layer virtual activations)"},
    "E6": {"folder": "E6_kd", "technique": "kd", "counts": True,
           "title": "Knowledge distillation (full-FT teacher -> layer-pruned students, online "
                    "score-distribution KD)"},
    "E7": {"folder": "E7_extras", "technique": "extras", "counts": False,
           "title": "Extras: IA3, prompt tuning, zero-shot Phase-1, pooling / attention / "
                    "long-document / hard-negative ablations"},
    "E8": {"folder": "E8_report", "technique": "report", "counts": False,
           "title": "Report: ALL_RESULTS, upload plan, score-fusion ensemble, analysis (CPU)"},
}
COUNTED = [e for e, s in EXPERIMENTS.items() if s["counts"]]

DEFAULT_LR: Dict[str, float] = {
    "full_ft": 3e-5, "kd": 5e-5, "student_hard": 5e-5, "partial_ft": 1e-4, "lorma": 5e-4,
    "lora": 5e-4, "qlora": 5e-4, "adalora": 1e-3, "adapters": 5e-4, "prefix": 1e-3,
    "prompt": 5e-3, "ia3": 3e-3, "bitfit": 2e-3,
}
HEAD_LR = 1e-3
DEFAULT_EPOCHS: Dict[str, int] = {
    "full_ft": 4, "kd": 4, "student_hard": 4, "partial_ft": 4, "lorma": 5, "lora": 5,
    "qlora": 5, "adalora": 5, "adapters": 5, "prefix": 6, "prompt": 8, "ia3": 6, "bitfit": 5,
}
MAX_TRAINABLE_PCT: Dict[str, float] = {"lorma": 5.0, "lora": 5.0, "qlora": 5.0}

# =============================================================================================
# Paths / exit codes
# =============================================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/kaggle/working/phase7_out"
CACHE_DIR = "/kaggle/working/phase7_cache"
RAW_DIR = "/kaggle/working/phase7_raw"
OUT_MARKER = "phase7_out"
EXIT = {"ok": 0, "error": 1, "missing_dependency": 2, "gate_failure": 3,
        "validation_failure": 4}
# exit codes of one training job (cjpe_train.py)
JOB_EXIT = {0: "ok", 1: "error", 2: "does not fit the time budget", 4: "too many NaN steps",
            5: "early-fail watchdog", 6: "critical self-check failed", 7: "OOM at micro-batch 1",
            137: "killed"}

MODULE_FILES = ["bpe_tokenizer.py", "phase1_models.py", "ner_common.py", "ner_models.py",
                "ner_peft.py", "cjpe_common.py", "cjpe_data.py", "cjpe_tok.py", "cjpe_models.py",
                "cjpe_peft.py", "cjpe_eval.py", "cjpe_kd.py", "lsi_ml.py", "cjpe_predict.py",
                "cjpe_train.py", "summ_model.py", "pcr_ml.py", "pcr_data.py", "pcr_model.py",
                "pcr_train.py", "validate_submission.py", "select_best.py", "aggregate.py",
                "run_experiment.py", "selftest.py"]

CSV_COLUMNS = [
    "experiment", "technique", "model_id", "arch", "attn_mode", "pooling", "doc_repr",
    "chunk_tokens", "k_train", "k_infer", "hard_negs", "status", "fail_reason", "selfcheck",
    "train_queries", "train_pairs", "train_fraction", "dev_queries", "test_queries",
    "epochs_planned", "epochs_run", "best_epoch", "lr", "batch_pairs", "score_metric",
    "dev_score", "dev_micro_f1", "dev_micro_p", "dev_micro_r", "dev_dense_f1",
    "dev_dense_mean_f1", "dev_dense_max_f1", "dev_bm25_f1", "dense_weight", "doc_agg",
    "dev_dense_recall_at_100", "dev_proxy_map", "dev_proxy_map_init", "trainable_params",
    "total_params", "trainable_pct", "params_ratio_vs_teacher", "projected_minutes",
    "train_minutes", "tokens_per_s", "infer_tokens_per_s", "ms_per_doc_infer", "peak_mem_gb",
    "gpu_slot", "attempts", "leak_guard", "seed", "job", "submission_zip", "started_at",
    "finished_at"]
EPOCH_COLUMNS = ["experiment", "model_id", "job", "epoch", "step", "train_loss",
                 "dev_proxy_map", "dev_proxy_f1", "lr", "minutes", "skipped_steps"]


def default_input_roots() -> List[str]:
    """Where attached Kaggle datasets live; PHASE7_INPUT_ROOTS (os.pathsep-separated) overrides."""
    env = os.environ.get("PHASE7_INPUT_ROOTS")
    if env is not None:
        return [p for p in env.split(os.pathsep) if p]
    return ["/kaggle/input"]


def module_md5s(code_dir: str = CODE_DIR) -> Dict[str, Optional[str]]:
    out = {}
    for f in MODULE_FILES:
        p = os.path.join(code_dir, f)
        out[f] = file_md5(p) if os.path.exists(p) else None
    return out


def student_layers(mid: str, phase1_dir: str) -> List[int]:
    """Every second layer (== STUDENT_LAYERS for the real 12/14-layer models)."""
    cfg = read_json(os.path.join(phase1_dir, "config.json"))["config"]
    n = cfg.get("n_layer", cfg.get("num_hidden_layers"))
    ids = list(range(0, n, 2))
    if n in (12, 14):
        assert ids == STUDENT_LAYERS[mid], (ids, STUDENT_LAYERS[mid])
    return ids


def iter_out_dirs(out_root: Optional[str], input_roots=()) -> List[str]:
    """Every phase5_out folder: the working one first, then any inside attached inputs."""
    outs = []
    if out_root and os.path.isdir(out_root):
        outs.append(os.path.abspath(out_root))
    for r in input_roots:
        if r and os.path.isdir(r):
            for d in glob.glob(os.path.join(r, "**", OUT_MARKER), recursive=True):
                if os.path.isdir(d) and os.path.abspath(d) not in outs:
                    outs.append(os.path.abspath(d))
    return outs


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_to(path: Optional[str], msg: str, prefix: str = "") -> None:
    line = f"{prefix}{msg}"
    print(line, flush=True)
    if path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + line + "\n")
        except OSError:
            pass


def fmt(v) -> str:
    """CSV cell: empty string for missing values, never None / nan."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.6g}" if abs(v) < 1e5 else f"{v:.1f}"
    return str(v)


def write_csv(path: str, rows: List[Dict], columns: List[str]) -> None:
    """Atomic CSV write (tmp + os.replace); unknown keys are ignored, missing ones are ''."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in columns])
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
