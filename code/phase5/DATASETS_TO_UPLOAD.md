# Phase 5 (LSI) — what to attach and how to run

## Attach to EVERY notebook (E1–E7)
| # | What | How |
|---|---|---|
| 1 | Four Phase-1 datasets (`01-train-gpt2style`, `02-train-llamastyle`, `03-train-mistral-sliding`, `04-train-bloom-alibi`) — `model.pt` + `config.json` | Add Input → Your datasets (same as Phase 4) |
| 2 | `bpe_model` dataset (`bpe_model.json`) | Add Input (found anywhere under `/kaggle/input`) |
| 3 | HF secret `HF` **ticked for this notebook** + Settings → Internet **On** | data is downloaded from `Exploration-Lab/CS-781-Capstone/Phase_5_LSI` |
| 4 | Accelerator **GPU T4 x2** | Settings |

No data upload is needed. E8 (CPU, no GPU, no secret) needs the **outputs of E1–E6** (+E7 if run).
Private notebook outputs cannot be attached on another account: on the owning account open the
notebook → Output → download, then upload that zip as a private dataset on the account running E8
(or run E8 on the account that ran most of them and add the rest as uploaded datasets).
E6 optionally reuses an attached E1 output (its M4 run becomes the teacher → saves ~40 min).

## Schedule (4 accounts, budget 2.5 h per notebook — set in the `ARGS` line)
| when | account A | account B | account C | account D |
|---|---|---|---|---|
| now | **E6** (slowest) | **E2** (LoRMA, compulsory) | **E1** | **E3** |
| first free | **E4** | **E5** | E7 (only if ≥3 h before 15:00 IST on the 26th) | — |
| after E1–E6 | **E8** (CPU, ~5 min) | | | |

Upload the first `BEST/submission.zip` (E2 or E1) as soon as it exists. Codabench takes 15–30 min
per upload: **start the last upload by 21:30 IST on Oct 26.** Order: `UPLOAD_PLAN.csv` from E8
(six required = one per technique E1–E6, all four Phase-1 models covered). Max 10 uploads/day.

If you start late, lower `--time-budget-h` in the `ARGS` line (e.g. 2.0); the planner shrinks
K / epochs / train fraction automatically and never skips a model silently.

## Healthy log (first ~10 min)
```
[labels] N statute ids in train; M ids only in dev (never predicted); kmax (p99 labels/doc) = k
[preflight] train 12,345 tok/s, inference 40,000 tok/s, micro-batch 16 chunks, peak 6.1 GB
CHOSEN: K=4 epochs=3 frac=1.00 ...
[check] forward_backward_finite  critical  pass
```
Then per evaluation: `dev macro-F1 …`, and at the end
`DONE dev macro-F1 0.xxxx (micro …, subset acc …, rare …, rule global|perlabel …) submission validated`.

## If something fails
Send only: the job's `logs/job_checks.json` **or** the last 40 lines of `logs/console.log`
(both inside `phase5_out/<EXP>/<job>/`), plus the `INDEX.md` row. `INDEX.md` always says why.

## Files you get per notebook (`/kaggle/working/phase5_out/<EXP>/`)
`results_<EXP>.csv` (one row per model, ~69 columns: Macro/Micro-F1, subset acc, rare/medium/frequent
Macro-F1, chosen threshold rule, cross-fitted per-label vs global, params, minutes, tok/s, …),
`INDEX.md`, `BEST/submission.zip`, and per job `dev/per_label_dev.csv`, `dev/errors.csv`
(missed / spurious statute ids per document), `dev/dev_report.json` (length buckets, confused
statute pairs, calibration). E8 adds `ALL_RESULTS.csv` + pivots, `PER_LABEL_ALL.csv`,
`E8_report/{statutes_hardest_first, frequency_buckets, threshold_rules, length_buckets,
confused_statute_pairs}.csv`, an optional ensemble and plots.
