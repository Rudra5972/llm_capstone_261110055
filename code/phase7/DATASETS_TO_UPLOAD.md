# Phase 7 (Prior Case Retrieval): what to attach and how to run

## Weights: nothing new to train or upload
Phase 7 starts from the **same four Phase-1 checkpoints** you have attached since Phase 2. You
do not create or upload any new weight dataset. Each notebook fine-tunes the Phase-1 models
inside the run and keeps the result in its own output (`<job>/checkpoints/best.pt`, trainable
tensors only).

## Attach to EVERY GPU notebook (E1-E7)
| # | What | How |
|---|---|---|
| 1 | The four Phase-1 datasets (`01-train-gpt2style`, `02-train-llamastyle`, `03-train-mistral-sliding`, `04-train-bloom-alibi`): `final/model.pt` + `config.json` | Add Input -> Your datasets (same as Phases 2-6). Models are found by the `arch` field, so dataset names do not matter |
| 2 | The `bpe_model` dataset (`bpe_model.json`) | Add Input (found anywhere under `/kaggle/input`) |
| 3 | **Data, option A:** the HF secret `HF` **ticked for this notebook** + Settings -> Internet **On** | downloads `Exploration-Lab/CS-781-Capstone/Phase_7_PCR/{Train,Dev,Test}/{split}.json + {split}_candidates.json` (6 files) |
| 3' | **Data, option B:** a private Kaggle dataset holding `Phase_7_PCR/` with the same 6 files | then Internet can stay **Off**. Nothing else is downloaded in Phase 7 (no BERTScore model this time) |
| 4 | Accelerator **GPU T4 x2** | Settings |

* **E6 only (optional):** also attach a finished **E2_full_ft** output. Its M4 run becomes the KD
  teacher, so E6 skips training its own teacher (about 40% of its time).
* **E8** (CPU, no GPU, no data, no secret) needs the **outputs of E1-E6** (plus E7 if run).
  Private outputs cannot be attached across accounts: open the notebook -> Output -> download,
  then upload the zip as a private dataset on the account that runs E8.

## The six counted experiments (PDF 13.2.1 list; one notebook each, all four Phase-1 models)
E1 LoRMA+ (compulsory) · E2 full fine-tuning · E3 partial / layer-wise FT · E4 QLoRA ·
E5 prefix tuning · E6 knowledge distillation (full-FT M4 teacher -> layer-pruned students of all
four models, online KL over the query x candidate score matrix).
E7 (optional, not counted): IA3, prompt tuning, zero-shot Phase-1 encoder, and ablations on M4
(last-token pooling, bidirectional attention, first-chunk-only, no hard negatives).

## Schedule (deadline Sat Nov 7, 23:59 IST; 5 accounts)
Budgets are already set in each notebook's `ARGS` line: 3.5 h (E6: 4.0 h), plus about 10 min of
setup. The times below assume you start at about 03:00 IST.

| IST | Account 1 | Account 2 | Account 3 | Account 4 | Account 5 |
|---|---|---|---|---|---|
| 03:00 | **E1** LoRMA | **E2** full FT | **E3** partial FT | **E4** QLoRA | **E6** KD |
| ~06:45 | **E5** prefix (first free account) | **E7** extras (optional) | upload E1-E4 BESTs | | |
| ~07:15 | | | | | E6 done -> upload |
| ~10:30 | **E8** (CPU, ~5 min), all outputs attached | | | | |
| by 13:00 | upload the rest following `UPLOAD_PLAN.csv` (6 required = one per E1-E6, all 4 models) | | | | |

* Upload each notebook's `BEST/submission.zip` as soon as it exists. Codabench takes 15-30 min
  per upload and allows at most 10 per day. **Start the last upload by 22:30 IST.**
* If you start later, lower `--time-budget-h` in `ARGS` (for example 2.5). The planner shrinks
  epochs, train fraction and inference chunks on its own and never skips a model silently.
* If E8 cannot run in time, upload the six `BEST/submission.zip` files and check that all four
  model ids appear among them (the `results_*.csv` rows give the alternatives).

## Healthy log (first ~15 min)
```
all 25 modules verified
[data] train: N queries ('id'/'text'/'relevant_candidates'), pool M candidates; ... 0 relevant ids not in the pool ...
[cache] tokenising every case into <= 16 chunks of 256 tokens ... [bm25] tuned on dev: k1=.. b=.. ngram=.. -> dev Micro-F1@7 0.xxxx
[check] zero_init_equals_phase1 / gradcache_equals_exact_gradient_and_grads_flow / padding_invariance / pools_never_crossed  pass
[preflight] train 2.5 s/step (16 pairs + 3 hard negatives each), encode 60,000 tok/s ...
[zero-shot] Phase-1 representation (before training): dev proxy MAP 0.xx
[plan] CHOSEN: epochs=4 train_fraction=1.0 ...
[eval] epoch 0.50: dev proxy MAP 0.xx   (should rise above the zero-shot value)
DONE dev Micro-F1@7 0.xxxx (P .. R ..) with mean aggregation, dense weight 0.x; dense only 0.xx, BM25 only 0.xx; submission validated
```

## How the ranking is made (same for every job, so experiments differ only by technique)
1. Each case is cut into at most 16 chunks of 256 tokens spread over the whole judgment. A
   training step samples 2 chunks per case; inference uses up to 8, evenly spaced.
2. Query and candidate go through the same fine-tuned Phase-1 encoder (shared weights). The case
   vector is the mean of its chunk vectors. Training uses InfoNCE with in-batch negatives plus
   3 BM25 hard negatives per query from the split's own pool; other relevant cases of a query
   are never treated as negatives.
3. On Dev the job picks how chunk scores are combined (`mean` / `max` / both) and the fusion
   weight `w` (0.2-1.0) in `w * z(dense) + (1 - w) * z(BM25)`. That setting ranks Test, and the
   top 7 unique candidates from the test pool go to `prediction.json` (`{"query_id": [7 ids]}`).
   A query never retrieves itself.
4. `results_*.csv` keeps the fused score, the **dense-only** score and the **BM25-only** score
   separately, so the report can show what each technique adds on its own.

## If something fails
Send only the job's `logs/job_checks.json` **or** the last 40 lines of `logs/console.log`
(inside `phase7_out/<EXP>/<job>/`), plus its `INDEX.md` row. `INDEX.md` always states why.

## Files you get per notebook (`/kaggle/working/phase7_out/<EXP>/`)
* `results_<EXP>.csv`: one row per model. Columns include Micro-F1/P/R@7, dense-only, BM25-only,
  mean- vs max-chunk dense F1, fusion weight, dense recall@100, zero-shot vs best proxy MAP,
  trainable %, params vs teacher, minutes, tok/s and ms per case.
* `epoch_log_<EXP>.csv`, `INDEX.md`, `BEST/submission.zip`.
* Per job: `dev/dev_predictions.jsonl` (top 7, gold, hits, dense ranks of the gold cases),
  `dev/fusion_curve.csv`, `dev/dev_report.json`, `dev/scores.npz`.

E8 adds `ALL_RESULTS.csv` plus pivots, `UPLOAD_PLAN.csv`,
`E8_report/{retrieval_table, query_length_buckets, hardest_dev_queries}.csv`, plots, and an
ensemble submission (mean of the best runs' dense scores fused with BM25). Its `use_it` flag
says whether the ensemble beat the best single run on Dev.
