# CS-781 Phase 6 — Abstractive Legal Summarization (built on the tested Phase 4/5 package)

What changed versus Phase 5 (everything else — Kaggle orchestration, time planner, OOM back-off,
retry ladder, resumable checkpoints, leak guard, INDEX.md / per-experiment CSVs, BEST + upload
plan, notebook generator — is reused):

* **Task / model.** No classifier head: the Phase-1 causal LM itself writes the summary after
  `[EOS] document "SUMMARY:"`; loss on summary tokens only; logits are computed only at those
  positions (`summ_model.SelectHead`). Modules: `gen_ml.py` (ROUGE, BERTScore wrapper, length
  control, diagnostics), `summ_data.py` (loader, held-out dev split, token cache, layouts),
  `summ_model.py` (SummLM, new techniques, KV-cache greedy / beam decoding), `summ_train.py` (job).
* **Long documents.** `--doc-repr auto` = `dancer` (chunk-wise summaries aligned sentence by
  sentence, concatenated) when the median train summary is > 350 tokens, else `headtail`;
  E7 ablates both. **Length control:** the word cap (0.5/0.75/1/1.25 × median train summary, or
  none) is chosen on dev and applied to test.
* **Experiments (PDF 12.2.1 list):** E1 LoRMA+ (compulsory), E2 LoRA, E3 AdaLoRA, E4 Houlsby,
  E5 Pfeiffer, E6 KD (full-FT M4 teacher → layer-pruned students, online KL on summary tokens);
  E7 extras QLoRA (NF4), prefix tuning, prompt tuning, repr and beam ablations; E8 report +
  consensus (MBR) ensemble.
* **Metrics:** dev BERTScore-F1 (`bert-score`, roberta-large) when installable, else
  (ROUGE-1 + ROUGE-L)/2 — never mixed across runs (`aggregate.common_metric`).
* **Submission:** `prediction.json` list format `[{"id", "summary"}]` at the zip root.

Tests: `python selftest.py` (6 CPU tests), `python tests/e2e.py WORKDIR` (all notebooks, CPU).

---

# Phase 5 — Legal Statute Identification (multi-label)

Built on the tested Phase 4 package (same scheduler, time planner, retries, resume, PEFT code).
What changed for Phase 5:
- **Head**: chunk-mean + attention pooling -> `Linear(768, L)` sigmoid, bias initialised to each
  statute's train log-odds prior. `L` = statute ids seen in `train.json` (`label_map.json`).
- **Loss**: per-label BCE with `pos_weight = clip(sqrt(neg/pos), 1, 30)`; E7 adds asymmetric
  loss and focal loss. KD (E6): 0.5 BCE(gold) + 0.5 T^2 soft-target BCE vs sigmoid(teacher/T), T=2.
- **Decision rule (dev only)**: global threshold grid 0.02-0.90 maximising dev Macro-F1;
  per-label thresholds (labels with >= 5 dev positives, shrunk 50% to the global one) used only
  if 2-fold cross-fitting on dev beats the global rule; always >= 1 and <= kmax (p99 of
  train labels/doc) statutes per document.
- **Metrics**: Macro-F1 (labels with dev support; also over all train labels), Micro-F1,
  subset accuracy, rare (<5) / medium (5-20) / frequent (>20 train docs) Macro-F1.
- **Output**: `[{"id": ..., "pred": [sorted unique valid statute ids]}]` in `prediction.json` at
  the root of `submission.zip`, validated (ids, ints, uniqueness, valid ids, non-empty).
- **Representations** (E7): `spread` (evenly spaced chunks + last chunk) and `cite` (first
  chunk + densest window of statute citations + last chunk) next to the default `hier` K=4.
- Optional 00 notebook removed; run order and attachments are in `DATASETS_TO_UPLOAD.md`.

---

# CS-781 Capstone, Phase 4: Court Judgment Prediction (CJPE)

This is binary classification of long Indian court judgments, scored by Macro-F1. It is pure PyTorch, and every experiment starts from my own Phase-1 models (M1 GPT-2 style, M2 Llama style, M3 Mistral sliding window, M4 BLOOM/ALiBi).

**Before you run anything, read `DATASETS_TO_UPLOAD.md`.** It lists what to attach, the per-notebook checklist and the troubleshooting table.

## The six counted experiments (the technique is the only difference)

| Notebook | Technique | What trains |
|---|---|---|
| E1_full_ft | Full fine-tuning | all parameters (backbone LR 3e-5, head 3e-4) |
| E2_lorma | **LoRMA+** (compulsory) | multiplicative low-rank `(I + s·B·A) W0 x`, r 16, alpha 32, + head |
| E3_adalora | AdaLoRA | SVD-form `P diag(λ) Q`, importance pruning 16 -> 8 average rank, + head |
| E4_adapters | Houlsby adapters | bottleneck 64 after attention-out and MLP-out + norms + head |
| E5_partial_ft | Partial / layer-wise FT with layer freezing | top 6 layers + final norm + head, LR decay 0.85 per layer |
| E6_kd | **Knowledge distillation** (compulsory) | M4 full-FT teacher -> each model pruned to every 2nd layer, `0.5 CE + 0.5 T² KL`, T=2 |

That is five different fine-tuning techniques plus knowledge distillation, which makes six experiments. Every notebook fine-tunes **all four** Phase-1 models. The upload planner (E8, `UPLOAD_PLAN.csv`) picks one model per experiment so that all four models appear and no two predictions are identical.

The extras are E7_extras (BitFit, LoRA, the document-representation and class-weight ablations) and E8_report (CPU: ALL_RESULTS, the upload plan, the ensemble and the error analysis). `00_data_audit_cache` is optional, and nothing depends on it.

## Method

**Long documents.** `hier` is the default for E1-E6. The model sees up to K=4 chunks of 512 tokens: the first 2 and the last 2 chunks of the judgment. Each chunk starts with EOS and is encoded independently by the Phase-1 trunk (bidirectional attention, as in Phases 2 and 3). The chunk vector is the masked mean of the last hidden layer. The document vector comes from learned attention pooling (`softmax(w·tanh(W c))`) over the real chunks, followed by dropout 0.1 and `Linear(768, 2)`. The head runs in fp32.
- If K has to drop to 1, the remaining chunk is the **last** one. The court's conclusion is at the end of a judgment, and ILDC (Malik et al., 2021) found the last 512 tokens the most informative single window.
- Ablations in E7: `headtail` (one window: first 383 + last 639 tokens), `first_k`, `last_k`, `sections` (first chunk + 2 chunks starting at the last "ORDER / CONCLUSION / HELD / for the foregoing reasons ..." heading), and `full` (up to 16 consecutive chunks, off by default).

**Fast tokenisation.** The pure-Python BPE only tokenises the needed head and tail character slices. They are cut at proven token boundaries: whitespace preceded by a non-whitespace character. The selftest asserts that the head is a prefix, and the tail a suffix, of the full-document token sequence. Tokenisation runs in parallel processes and is cached once per notebook (attach the 00 output to reuse it). Token counts of long documents are estimated from the measured chars-per-token ratio of the same document.

**Class imbalance and the decision rule.**
- Cross-entropy is weighted by `sqrt_inv` class weights, normalised to a mean of 1.
- `p1 = softmax(logits)[1]`. The threshold `tau` is tuned **on dev only**, from 0.05 to 0.95 in steps of 0.01, keeping at least 5% of dev predictions in each class. It falls back to 0.5 when nothing beats 0.5.
- F1 at 0.5 and at the tuned tau are both reported.

**Data integrity (the leaked-test notice).**
- Gradient updates use only `train.json`. `dev.json` is used for selection, early stopping and the threshold. `test.json` text is used for inference only.
- The leak guard drops any train or dev case whose id, raw-text SHA-1 or normalised-text hash also occurs in test. It records the counts in `data_audit.json` and `PROVENANCE.json`, the latter with the md5s of the three files and what each was used for.
- Every job asserts at start-up that its training ids are disjoint from the dev and test ids.

**Self-calibrating time plan (section 8.4).**
- Each job measures its training and inference tokens/s on the real GPU. From its share of the notebook budget it then picks, in order: K=4 with the largest number of epochs that fits (at least 1 for full/partial FT and KD students, at least 2 for PEFT), then K=2, then K=1, then a stratified subsample of at least 30% of train.
- The plan table is printed before training. A time guard stops training so that the final dev and test inference (and the teacher's logits) still fit.

**Two GPUs without DDP.** There is one worker slot per GPU, and each job is a single-GPU subprocess (`CUDA_VISIBLE_DEVICES=<slot>`). Four models on two GPUs therefore take about half of the sequential time, with no NCCL hangs.

**Robustness.**
- OOM back-off: the micro-batch is halved, down to one document.
- Non-finite steps are skipped. More than 5% skipped ends the job (exit 4).
- Watchdog: a constant output after epoch 1 ends the job (exit 5).
- Checkpoints: atomic `last.pt` every 30 min (resume across sessions), and `best.pt` whenever dev improves.
- The disk guard deletes stale `last.pt` files when space runs low.
- Retry ladder (exit 4/5: LR ×0.3; exit 1/7: micro-batch halved; exit 6: self-checks skipped; then K=2).
- `status.json` is updated at every state change and from `finally`.
- Notebook cells never raise, so a commit always saves its output.

**Self-checks on the real device (lesson L1/L2).**
- Critical (failure triggers the retry ladder): every tensor is on one device, the forward/backward pass is finite with non-zero gradients, the output is 2-way, and masked chunks get exactly zero attention.
- Advisory (failure logs a warning and the job continues): zero-init equals the base model (LoRMA/LoRA/AdaLoRA exactly, adapters approximately), merged equals unmerged, and the trainable-% cap.
- Every value is written to `logs/job_checks.json`.
- The device-order bug of the Phase-3 pilot is fixed in `cjpe_peft.apply_technique`. After inserting adapters it moves the whole model to the device of the frozen base and asserts that everything is there. The selftest checks this in both orders, using the `meta` device.

## Outputs (`/kaggle/working/phase5_out/`)

```
phase5_out/
  E2_lorma/
    M4_bloom_alibi/   checkpoints/best.pt  logs/{run_config,plan,job_checks,model_size,integrity}.json
                      logs/{console.log,train_log.jsonl}  dev/{dev_probs.csv,dev_report.json,errors.csv,probs.npz}
                      submission/{prediction.json,submission.zip,validation_report.json,test_probs.csv}
                      status.json  metrics.json  DONE.json
    M2_llamastyle/ M3_mistral_sliding/ M1_gpt2style/
    BEST/{submission.zip,SELECTION.json,run_config.json}
    results_E2_lorma.csv  epoch_log_E2_lorma.csv  summary.json  orchestrator.log
  ALL_RESULTS.csv  ALL_RESULTS_pivot*.csv  UPLOAD_PLAN.csv  INDEX.md
  data_audit.json  PROVENANCE.json  test_ids.json
```

`results_<Exp>.csv` is written as soon as the run starts, with one `PENDING` row per planned model. It is rewritten every minute and at every job event. A failed model keeps its row, with `fail_reason`. Missing values are empty cells, never `None` or `nan`.

## Reused code and changes
- **Byte-identical to Phases 2/3** (no change): `bpe_tokenizer.py`, `phase1_models.py`, `ner_common.py`, `ner_models.py` (Trunk), `ner_peft.py` (LoRA, LoRMA+, adapters, BitFit, partial FT).
- **Copied from Phase 3:** `rr_peft.py` (AdaLoRALinear + AdaLoRA, into `cjpe_peft.py`). Two things changed: the head lives in `classifier.*`, and the module takes an explicit `device=`. `force_final()` was added so that a run cut short by the time guard still ends at the target rank budget.
- **New:** `cjpe_common`, `cjpe_data`, `cjpe_tok`, `cjpe_models`, `cjpe_peft`, `cjpe_eval`, `cjpe_kd`, `cjpe_train`, `cjpe_predict`, `validate_submission`, `select_best`, `aggregate`, `run_experiment`, `selftest`, `tools/make_notebooks.py`, `tests/e2e.py`.

## Defaults chosen (and why)
- **attention:** `bidir` (the Phase-2/3 default). A `PILOT.json` with `attn_mode` is used if one is attached.
- **chunk length:** 512 including EOS. Every model was pretrained on 1024-token blocks, and 1024 is never exceeded.
- **effective batch, warm-up and schedule:** 16 documents; 6% linear warm-up, then linear decay; AdamW (0.9, 0.98); weight decay 0.01 (none on 1-D tensors); grad clip 1.0; seed 42.
- **learning rates:** full FT 3e-5 (head ×10); LoRMA, LoRA, AdaLoRA and adapters 1e-3; partial FT 1e-4 for the top layer, ×0.85 per layer downwards; KD students 5e-5; BitFit 2e-3; head 1e-3 for the PEFT techniques.
- **epochs and evaluation:** at most 4 epochs for full FT, partial FT and KD, 6 for PEFT. Early stopping after 2 epochs without improvement, with 2 dev evaluations per epoch.
- **micro-batch:** 16 chunks for training and 48 for inference, with automatic OOM back-off.
- **knowledge distillation:** the teacher is M4 (`--teacher-model`) and gets 38% of the E6 budget. Student layers are M1 and M4 [0,2,...,12] and M2 and M3 [0,2,...,10].
- **leak guard:** if more than 2% of train overlaps test, a loud warning is printed and the overlapping cases are dropped. The run does **not** stop, because of the deadline. Report the count to the TAs.

## Commands

```bash
python selftest.py --out SELFTEST.json         # CPU, ~1 min: metrics, loader, leak guard, tokenisation,
                                               # model, PEFT + device order, overfit
python tests/e2e.py /tmp/e2e                   # CPU, ~15 min: every notebook end to end with tiny
                                               # models, incl. injected failures and the retry ladder
python tools/make_notebooks.py notebooks       # regenerate the notebooks (md5 of all 19 modules)
python run_experiment.py --exp E2 --models M4,M2,M3,M1 --gpu-mode jobs --resume auto --time-budget-h 3.5
python select_best.py --plan                   # UPLOAD_PLAN.csv from every attached output
python aggregate.py --report                   # ALL_RESULTS + ensemble + error analysis
```

## Rule compliance

| Rule | How it is met |
|---|---|
| Libraries | torch, numpy and the stdlib only. `kaggle_secrets` is used only to read the secret, matplotlib only for optional E8 plots. No HF Hub library, and no checkpoints are pushed anywhere |
| Phase-1 start | every model is loaded with `strict=True` from my `final/model.pt` |
| Experiments | 6 or more; at least 5 different techniques; LoRMA included; KD included and counted |
| All four models | every notebook trains M1-M4, and the planner requires all four among the six uploads |
| Submission | `prediction.json` at the zip root; exactly the test ids; integer 0/1 preds; validated before a job counts as DONE |
| No leaked data | train-only gradients; the leak guard; `PROVENANCE.json`; the integrity assertion in every job |
