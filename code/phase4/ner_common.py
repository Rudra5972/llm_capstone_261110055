"""
ner_common.py -- small torch-free helpers shared by every Phase-2 module.

  * registries: the four Phase-1 models (M1..M4) and the experiments (E1..E6, extras)
  * file discovery under /kaggle/input (or any root), atomic JSON writes
  * disk guard and output-size accounting (no Hugging Face Hub anywhere in this project)
"""
import glob
import hashlib
import json
import os
import shutil
import time

# ---------------------------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------------------------
# `arch` must equal the "arch" field the Phase-1 notebooks wrote into final/config.json.
MODELS = {
    "M1": {"arch": "gpt2style", "folder": "M1_gpt2style", "kind": "gpt2"},
    "M2": {"arch": "llamastyle", "folder": "M2_llamastyle", "kind": "llama"},
    "M3": {"arch": "mistral-sliding", "folder": "M3_mistral_sliding", "kind": "llama"},
    "M4": {"arch": "bloom-alibi", "folder": "M4_bloom_alibi", "kind": "bloom"},
}
DEFAULT_MODEL_ORDER = ["M4", "M2", "M3", "M1"]          # strongest Phase-1 LM first

# Layer subsets for the layer-pruned KD students (every second layer; section 8 of the brief).
STUDENT_LAYERS = {
    "M1": [0, 2, 4, 6, 8, 10, 12],      # 7 of 14
    "M2": [0, 2, 4, 6, 8, 10],          # 6 of 12
    "M3": [0, 2, 4, 6, 8, 10],          # 6 of 12
    "M4": [0, 2, 4, 6, 8, 10, 12],      # 7 of 14
}

# The six required experiments + extras. `technique` selects the code path in ner_peft.py.
EXPERIMENTS = {
    "E1": {"folder": "E1_full_ft", "technique": "full_ft", "title": "Full fine-tuning"},
    "E2": {"folder": "E2_lorma", "technique": "lorma", "title": "LoRMA+ (compulsory)"},
    "E3": {"folder": "E3_lora", "technique": "lora", "title": "LoRA"},
    "E4": {"folder": "E4_adapters", "technique": "adapters", "title": "Houlsby adapters"},
    "E5": {"folder": "E5_partial_ft", "technique": "partial_ft", "title": "Partial / layer-wise FT"},
    "E6": {"folder": "E6_kd", "technique": "kd", "title": "Knowledge distillation"},
    # extras (not counted among the six)
    "X1": {"folder": "X1_bitfit", "technique": "bitfit", "title": "BitFit (M1/M4 only)",
           "models": ["M4", "M1"]},
    "X2": {"folder": "X2_student_no_kd", "technique": "student_hard",
           "title": "Pruned student, hard labels only (KD control)"},
}
REQUIRED_EXPS = ["E1", "E2", "E3", "E4", "E5", "E6"]

# Default learning rates (section 9.3). The LR probe multiplies `lr` by {1/3, 1, 3}.
DEFAULT_LR = {
    "full_ft": 5e-5, "lorma": 1e-3, "lora": 1e-3, "adapters": 1e-3, "bitfit": 2e-3,
    "partial_ft": 1e-4, "kd": 1e-4, "student_hard": 1e-4,
}
DEFAULT_EPOCHS = {
    "full_ft": 6, "kd": 6, "student_hard": 6, "partial_ft": 8,
    "lorma": 10, "lora": 10, "adapters": 10, "bitfit": 10,
}


# ---------------------------------------------------------------------------------------------
# JSON / files
# ---------------------------------------------------------------------------------------------
def write_json(path, obj, indent=2):
    """Atomic JSON write: a crash never leaves a half-written file behind."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_files(name, roots, max_hits=50):
    """All files called `name` below any of `roots` (recursive), sorted for determinism."""
    hits = []
    for r in roots:
        if r and os.path.isdir(r):
            hits.extend(glob.glob(os.path.join(r, "**", name), recursive=True))
    return sorted(set(hits))[:max_hits]


def default_input_roots():
    """Where attached Kaggle datasets live; PHASE2_INPUT_ROOTS (os.pathsep-separated) overrides."""
    env = os.environ.get("PHASE2_INPUT_ROOTS")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    return ["/kaggle/input"]


def iter_finished_runs(out_root, input_roots=()):
    """Yield (exp_folder, model_folder, run_dir) for every finished run (DONE.json):
    directly under `out_root` (<exp>/<model>/DONE.json, whatever out_root is called) and under
    any `phase2_out` folder inside the attached input datasets. First occurrence wins."""
    seen = set()
    out_root = os.path.abspath(out_root) if out_root else None
    if out_root and os.path.isdir(out_root):
        for d in sorted(glob.glob(os.path.join(out_root, "*", "*", "DONE.json"))):
            exp_f, mod_f = d.split(os.sep)[-3], d.split(os.sep)[-2]
            if mod_f != "BEST" and (exp_f, mod_f) not in seen:
                seen.add((exp_f, mod_f))
                yield exp_f, mod_f, os.path.dirname(d)
    for d in find_files("DONE.json", list(input_roots), max_hits=5000):
        parts = d.split(os.sep)
        if "phase2_out" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("phase2_out")
        if len(parts) - i != 4 or parts[-2] == "BEST":
            continue
        if (parts[i + 1], parts[-2]) not in seen:
            seen.add((parts[i + 1], parts[-2]))
            yield parts[i + 1], parts[-2], os.path.dirname(d)


def find_phase1_models(roots=None):
    """Map M1..M4 -> directory holding that model's Phase-1 `model.pt` + `config.json`.

    Identified by the "arch" field the Phase-1 notebooks wrote into config.json, so the
    dataset/folder names do not matter.
    """
    roots = roots or default_input_roots()
    by_arch = {}
    for cfg_path in find_files("config.json", roots, max_hits=10000):
        d = os.path.dirname(cfg_path)
        if not os.path.exists(os.path.join(d, "model.pt")):
            continue
        try:
            cfg = read_json(cfg_path)
        except Exception:
            continue
        if isinstance(cfg, dict) and "arch" in cfg and "model_class" in cfg:
            by_arch.setdefault(cfg["arch"], []).append(d)
    out = {}
    for mid, m in MODELS.items():
        cands = by_arch.get(m["arch"], [])
        if cands:
            # prefer a directory literally called "final" (what the Phase-1 notebooks saved)
            cands = sorted(cands, key=lambda p: (os.path.basename(p) != "final", p))
            out[mid] = cands[0]
    return out


# ---------------------------------------------------------------------------------------------
# Disk guard (section 9.5)
# ---------------------------------------------------------------------------------------------
def free_gb(path="/kaggle/working"):
    p = path
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 1e9


def dir_size_gb(path):
    tot = 0
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if not os.path.islink(fp):
                try:
                    tot += os.path.getsize(fp)
                except OSError:
                    pass
    return tot / 1e9


class DiskFull(RuntimeError):
    pass


def disk_guard(out_root, min_free_gb=4.0, log=print, keep=None):
    """Make sure at least `min_free_gb` is free under `out_root`.

    Deletion order: leftover last.pt / last.tmp of models that are finished (DONE.json) or not
    the one currently running (`keep` = its checkpoint dir), then tmp files. Never touches
    best.pt, submissions or reports. Raises DiskFull if still short (callers stop cleanly).
    """
    if free_gb(out_root) >= min_free_gb:
        return
    removed = []
    for pat in ("**/checkpoints/last.pt", "**/checkpoints/last.tmp", "**/*.tmp*"):
        for f in glob.glob(os.path.join(out_root, pat), recursive=True):
            if keep and os.path.abspath(f).startswith(os.path.abspath(keep)):
                continue
            try:
                sz = os.path.getsize(f) / 1e9
                os.remove(f)
                removed.append((f, sz))
            except OSError:
                pass
        if free_gb(out_root) >= min_free_gb:
            break
    for f, sz in removed:
        log(f"   [disk] removed {f} ({sz:.2f} GB)")
    if free_gb(out_root) < min_free_gb:
        raise DiskFull(f"only {free_gb(out_root):.1f} GB free under {out_root} "
                       f"(need {min_free_gb} GB) even after cleanup")


# ---------------------------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------------------------
def now():
    return time.time()


def fmt_h(seconds):
    return f"{seconds / 3600:.2f} h"


def log_line(path, msg, also_print=True):
    """Append a timestamped line to a text log (and print it)."""
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    if also_print:
        print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
