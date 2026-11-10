# Phase 4 (CJPE): everything to create or attach on Kaggle

The deadline is **21 Oct 2026, 23:59 IST**, and Codabench takes 15-30 minutes to score each upload. Start your last upload by about 21:30.

## 1. Kaggle datasets

| # | Dataset (any name) | Exact files and folder layout inside it | Size | Needed by | Required? |
|---|---|---|---|---|---|
| A | `rudra1610/01-train-gpt2style` (already exists) | `final/model.pt`, `final/config.json`, `final/bpe_model.json` (+ `final/tokenizer_config.json` if present) | ~0.5 GB | 00, E1-E7 | **required** |
| B | `rudra1610/02-train-llamastyle` (already exists) | same three files under `final/` | ~0.5 GB | 00, E1-E7 | **required** |
| C | `rudra1610/03-train-mistral-sliding` (already exists) | same | ~0.5 GB | 00, E1-E7 | **required** |
| D | `rudra1610/04-train-bloom-alibi` (already exists) | same | ~0.5 GB | 00, E1-E7 | **required** |
| E | **new: `phase4-cjpe-data`** | `Phase_4_CJPE/Train/train.json`, `Phase_4_CJPE/Train/dev.json`, `Phase_4_CJPE/Test/test.json` | depends on the release | 00, E1-E7 (E8 optional) | **required** (or use the `HF` secret, see 1.1) |
| F | output of `00_data_audit_cache` | `phase4_cache/cache_<key>.pkl`, `phase4_cache/cache_meta.json`, `phase4_out/...` | ~0.1-0.5 GB | E1-E6 | optional (saves 2-5 min of tokenisation per notebook) |
| G | output of an earlier session of the same notebook | `phase4_out/<Exp>/<job>/...` | small | the same notebook | optional (finished models are skipped; an unfinished one resumes from `last.pt`) |
| H | outputs of E1 ... E6 (and E7) | `phase4_out/E1_full_ft/...` etc. | small (best.pt of full-FT runs ~250 MB each) | **E8** | required for E8 |

Notes
- Models are recognised by the `arch` field of `final/config.json` (`gpt2style`, `llamastyle`, `mistral-sliding`, `bloom-alibi`), not by dataset or folder names, so the names do not matter.
- `bpe_model.json` is the same tokenizer in all four datasets. The code uses the first one it finds.
- The three data files are recognised by their names (`train.json`, `dev.json`, `test.json`) and by their content: every record must have an id and a text field. Paths that contain `Phase_4` or `CJPE` are preferred. Several attached datasets can contain files with the same names without breaking this.

### 1.1 How to create dataset E (Phase-4 data)
1. On a machine where you can log in to Hugging Face (an account that has **accepted the terms** of `Exploration-Lab/CS-781-Capstone`), open the dataset page, go to **Files and versions -> Phase_4_CJPE** and download `Train/train.json`, `Train/dev.json` and `Test/test.json`. **Use the CURRENT files.** The organisers rebuilt dev/test after the leak, so files downloaded before the notice are stale.
2. On your computer, recreate the layout `Phase_4_CJPE/Train/train.json`, `Phase_4_CJPE/Train/dev.json`, `Phase_4_CJPE/Test/test.json`.
3. Kaggle -> **Datasets -> New Dataset**. Drag the `Phase_4_CJPE` folder in (Kaggle keeps sub-folders), name it `phase4-cjpe-data`, keep it **Private** and press Create.
4. *Alternative with no dataset:* in each notebook use **Add-ons -> Secrets -> Add secret**, with label `HF` and value = a *read* token from huggingface.co/settings/tokens. Tick the secret for the notebook and set **Settings -> Internet -> On**. The notebook then downloads the three files with plain `urllib`. The token is never printed or saved.

## 2. Per-notebook checklist

All GPU notebooks: **Settings -> Accelerator -> GPU T4 x2**, **Persistence: Files only** (optional), then **Save Version -> Save & Run All (Commit)**.

| Notebook | Accelerator | Internet | Secret `HF` | Attach | Run order | Expected time |
|---|---|---|---|---|---|---|
| `00_data_audit_cache` (optional) | GPU T4 x2 | off (on only with `HF`) | only without E | A, B, C, D, E | anytime; nothing waits for it | 15-30 min |
| `E1_full_ft` | GPU T4 x2 | off | only without E | A-E (+F, +G) | round 1 | <= 3.5 h |
| `E2_lorma` | GPU T4 x2 | off | only without E | A-E (+F, +G) | round 1 | <= 3.5 h |
| `E3_adalora` | GPU T4 x2 | off | only without E | A-E (+F, +G) | round 1 | <= 3.5 h |
| `E6_kd` | GPU T4 x2 | off | only without E | A-E (+F, +G; + E1 output to reuse its M4 as teacher) | round 1 (slowest: teacher, then 4 students) | <= 3.5 h |
| `E4_adapters` | GPU T4 x2 | off | only without E | A-E (+F, +G) | round 2 (first free account) | <= 3.5 h |
| `E5_partial_ft` | GPU T4 x2 | off | only without E | A-E (+F, +G) | round 2 | <= 3.5 h |
| `E7_extras` (optional) | GPU T4 x2 | off | only without E | A-E | only if an account is free | <= 3 h |
| `E8_report` | **None (CPU)** | off | no | H (+E for the n-gram analysis) | after E1-E6 | 2-10 min |

**Tight on time?** Put `--time-budget-h 3.0` (or less) in the `ARGS` line of the run cell. Every job plans its epochs, chunks and data fraction from its share of that budget. To split one notebook over two accounts, use `--models M4,M2` on one and `--models M3,M1` on the other, then attach both outputs to E8.

### Lines that prove a run is healthy (first ~5 minutes of the log)
1. `all 19 modules verified`
2. `Phase-1 weights: {"M1": ".../final", "M2": ..., "M3": ..., "M4": ...}` (all four), and `[attn] bidir`
3. `[data] source: kaggle_dataset` (or `huggingface`), then for each split a line like `train: N records, list of objects, keys id='id' text='text' label='label'`
4. the `[audit]` table, the `[audit] train tokens: median ...` lines, and `[leak-guard] ... dropped train X, dev Y`
5. `[cache] building ...` (or `reusing`), then `[queue] START M4_bloom_alibi (...) on slot 0` and `... on slot 1`
6. per job: `[check] all_tensors_on_one_device critical pass`, `[check] forward_backward_finite critical pass`, `[preflight] train N tok/s`, the `[plan]` table, `CHOSEN: K=... epochs=... train_fraction=...`, and after a while `[eval] epoch 0.50: dev macro-F1 ...`

## 3. Saving and using the outputs
- Commit with **Save Version -> Save & Run All (Commit)**. When it finishes, the **Output** tab of the version shows `phase4_out/` (and `phase4_cache/`).
- **Upload to Codabench:** download `phase4_out/<Exp>/BEST/submission.zip` (the best of the four models). Every model also has its own `phase4_out/<Exp>/<job>/submission/submission.zip`.
- **Analysis files:** `phase4_out/<Exp>/results_<Exp>.csv` has one row per model and about 60 columns. `phase4_out/<Exp>/epoch_log_<Exp>.csv` has one row per evaluation.
- **For E8 (and to resume):** in E8 use **Add Input -> Notebook Output Files** and pick each finished notebook (E1 ... E6, E7). Alternatively, use **Output -> New Dataset** on each version. E8 then writes `ALL_RESULTS.csv`, `ALL_RESULTS_pivot*.csv` and `UPLOAD_PLAN.csv`, which lists the **six uploads in order**.

## 4. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `HTTP 401/403: token missing, or this Hugging Face account has not accepted the dataset terms` | the `HF` secret is missing or not ticked for this notebook, or the account did not accept the gated dataset | accept the terms on the dataset page with that account, re-add the secret and tick it, set Internet On. Better: attach dataset E |
| `Phase-4 data not found ...` / `BLOCKED.json` | neither dataset E nor the secret is attached | attach E (section 1.1) and re-run |
| `Phase-1 weights missing for ['M3'] ...` | one of A-D is not attached | attach it. The other models still run |
| `!!! modules differ from the generated version` | a `%%writefile` cell was edited | re-download the notebook from the package |
| `[queue] SKIP <job>: only N min left` / status `SKIPPED_TIME` | the time budget ran out | INDEX.md prints the exact `--models ...` command; run it on another account and attach the output to E8 |
| `exit 6 (critical self-check failed)` then `RETRY ... self-checks skipped` | a self-check on the real GPU failed (see `logs/job_checks.json`) | nothing to do: it retries automatically. Send me `job_checks.json` if it ends FAILED |
| `[oom] micro-batch -> 8 chunks` | GPU memory | automatic. If a job ends with `exit 7`, it retries with K=2 |
| `DEGENERATE` | the model predicts one class for every test case | it is excluded from BEST and from the plan. Use another model's zip |
| `No space left on device` | `/kaggle/working` is full | the disk guard deletes stale `last.pt` files. Do not attach unnecessary large inputs |
| session stays **Queued** | Kaggle has no free T4 slot | wait, or start the same notebook on another account (the code is identical in all notebooks) |
