# Phase 6 (Abstractive Summarization) — what to attach and how to run

## Attach to EVERY notebook (E1–E7)
| # | What | How |
|---|---|---|
| 1 | Four Phase-1 datasets (`01-train-gpt2style`, `02-train-llamastyle`, `03-train-mistral-sliding`, `04-train-bloom-alibi`) — `model.pt` + `config.json` | Add Input → Your datasets (same as Phases 4/5) |
| 2 | `bpe_model` dataset (`bpe_model.json`) | Add Input (found anywhere under `/kaggle/input`) |
| 3 | HF secret `HF` **ticked for this notebook** + Settings → Internet **On** | data from `Exploration-Lab/CS-781-Capstone/Phase_6_Summ`; Internet also installs `bert-score` (dev scoring only) |
| 4 | Accelerator **GPU T4 x2** | Settings |

No data upload is needed. There is no public dev split: every notebook carves the SAME held-out dev
split from train (hash of the normalised text), so dev scores are comparable across notebooks.
E8 (CPU, no GPU, no secret) needs the **outputs of E1–E6** (+E7 if run). Private notebook outputs
cannot be attached on another account: open the notebook → Output → download, upload the zip as a
private dataset on the account that runs E8.

## The six counted experiments (PDF 12.2.1 list; one notebook each, all four Phase-1 models)
E1 LoRMA+ (compulsory) · E2 LoRA · E3 AdaLoRA · E4 Houlsby adapters · E5 Pfeiffer adapters ·
E6 knowledge distillation (full-FT M4 teacher → layer-pruned students of all four models, online
token-level KD). E7 (optional extras): QLoRA, prefix tuning, prompt tuning, headtail-vs-dancer and
beam-3-vs-greedy ablations on M4.

## Schedule (5 accounts, budget 2.0 h per notebook — the `ARGS` line; ~10 min setup on top)
| when | A | B | C | D | E |
|---|---|---|---|---|---|
| now | **E6** (slowest) | **E1** (LoRMA, compulsory) | **E2** | **E3** | **E4** |
| first free | **E5** | E7 (only if ≥ 3 h before 21:00 IST Nov 2) | | | |
| after E1–E6 | **E8** (CPU, ~5 min) | | | | |

Upload the first `BEST/submission.zip` (E1 or E2) as soon as it exists. Codabench takes 15–30 min
per upload, max 10/day: **start the last upload by 21:30 IST on Nov 2.** Upload order =
`UPLOAD_PLAN.csv` from E8 (six required = one per technique E1–E6, all four Phase-1 models covered,
pairwise different predictions). If E8 cannot run in time: upload each notebook's
`BEST/submission.zip`, then swap duplicated models by hand using the `results_*.csv` rows.

If you start late, lower `--time-budget-h` in `ARGS` (e.g. 1.5); the planner shrinks epochs / train
fraction / dev decoding automatically and never skips a model silently.

## Healthy log (first ~15 min)
```
bert-score: ok
[data] train: N records; keys id='id' document='document' summary='summary'
[data] held-out dev split: 300 docs (hash of normalised text); train ...
[audit] document tokens median ... ; summary tokens median ... p90 ...
[repr] doc_repr=auto -> dancer        (or headtail when summaries are short)
[check] forward_backward_finite  critical  pass
[preflight] train 9,000 tok/s, generation 25 ms/step (batch 16), micro-batch 8192 tokens
[plan] CHOSEN: epochs=2 train_fraction=1.0 (... seqs), dev_gen 64, max_new 180, ...
[eval] epoch 0.50: dev loss 2.1 (ppl 8.2)
DONE dev BERTScore-F1 0.8xxx, ROUGE-1 0.4x, ROUGE-2 ..., ROUGE-L ... submission validated
```
`bert-score: NOT available -> ROUGE only` is not fatal: runs are then ranked by (ROUGE-1+ROUGE-L)/2.

## If something fails
Send only: the job's `logs/job_checks.json` **or** the last 40 lines of `logs/console.log`
(inside `phase6_out/<EXP>/<job>/`), plus the `INDEX.md` row. `INDEX.md` always says why.

## Files you get per notebook (`/kaggle/working/phase6_out/<EXP>/`)
`results_<EXP>.csv` (one row per model, ~62 columns: BERTScore-F1, ROUGE-1/2/L, dev loss/ppl,
chosen length cap, words vs reference, compression, repeated-trigram rate (redundancy),
novel-bigram rate (abstractiveness), content support / number & citation precision vs the source
(factuality proxies), reference content recall (coverage), params, minutes, tok/s, ms/doc …),
`epoch_log_<EXP>.csv`, `INDEX.md`, `BEST/submission.zip`, and per job
`dev/dev_predictions.jsonl` (prediction + reference + per-doc scores), `dev/length_curve.csv`
(scores at 0.5/0.75/1/1.25 × median length / uncapped), `dev/length_buckets.csv` (by document
length), `dev/dev_report.json`. E8 adds `ALL_RESULTS.csv` + pivots, `UPLOAD_PLAN.csv`,
`E8_report/{quality_table, length_curves, doc_length_buckets, worst_dev_docs}.csv`, a consensus
ensemble submission (`use_it` says whether it beat the best single run on dev) and plots.
