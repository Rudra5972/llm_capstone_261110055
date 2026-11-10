# CS-781 Capstone, Phase 2: Legal NER (L-NER)

Pure-PyTorch fine-tuning of the four Phase-1 legal language models for character-span NER,
scored by strict macro-F1. Nothing from `transformers`, `peft`, `datasets`, `tokenizers`,
`accelerate`, `pytorch-crf` or `seqeval` is used; LoRA, LoRMA, adapters, BitFit, the CRF,
distillation and the metrics are implemented here.

## 1. What you run, in order

| Step | Notebook | Hardware | What it does |
|---|---|---|---|
| 0 | `00_data_and_tests.ipynb` | CPU | builds the data cache, runs gates G0/G1/G2 and the CPU unit tests |
| 1 | `E1_full_ft.ipynb` | GPU T4 x2 | full fine-tuning of M4, M2, M3, M1; best-of-4 into `BEST/` |
| 1 | `E2_lorma.ipynb` | GPU T4 x2 | LoRMA+ (compulsory) |
| 1 | `E3_lora.ipynb` | GPU T4 x2 | LoRA |
| 1 | `E4_adapters.ipynb` | GPU T4 x2 | Houlsby adapters |
| 2 | `E5_partial_ft.ipynb` | GPU T4 x2 | partial / layer-wise fine-tuning |
| 2 | `E6_kd.ipynb` | GPU T4 x2 | knowledge distillation (needs E1's output attached) |

Start E1-E4 on your four accounts at the same time. E5 and E6 go to the first accounts that free
up; E6 needs E1 finished. Upload `phase2_out/E1_full_ft/BEST/submission.zip` to Codabench as soon
as it exists, to prove the pipeline end to end before the rest.

**How to run a GPU notebook:** Settings -> Accelerator **GPU T4 x2**, then **Save Version -> Save &
Run All (Commit)**, as in Phase 1. A commit run keeps going after you close the browser, and when
it ends `/kaggle/working` becomes the notebook's output (attach it to later sessions). Turn
Internet on only if you use the Hugging Face download fallback below.

Every notebook is standalone: all modules are inlined with `%%writefile` (identical in all seven
notebooks, MD5-checked at run time). Attach to each:
1. the four Phase-1 output datasets (each with `final/model.pt`, `final/config.json`,
   `final/bpe_model.json`); models are recognised by the `arch` field, not by dataset name;
2. the official `Phase_2_LNER` files as a dataset, **or** a read-only HF token as the Kaggle
   secret `HF` (the files are then downloaded once with `urllib`; nothing is ever written to the
   Hub);
3. optionally the `phase2_data` output of notebook 00 (otherwise rebuilt in about a minute);
4. for a second session of the same notebook: that notebook's earlier output (finished models are
   skipped, an unfinished one resumes from `last.pt`).

To split one notebook over two accounts edit the run cell: `--models M4,M2` on one and
`--models M3,M1` on the other; afterwards `python select_best.py --exp E1` on the merged output.

## 2. Output layout (per notebook)

```
phase2_out/
  E1_full_ft/
    M4_bloom_alibi/
      checkpoints/   last.pt (resumable, deleted when the model finishes), best.pt (weights only)
      logs/          run_config.json, smoke.json, nproc_decision.json, overfit.json,
                     lr_probe.json, train_log.jsonl, train_summary.json, console.log
      dev/           dev_report.json (metrics, error analysis, decoding table), errors.csv
      submission/    prediction.json, submission.zip, validation_report.json, entities_debug.json
      DONE.json      final dev metrics + timings (its presence = model finished)
    M2_llamastyle/  M3_mistral_sliding/  M1_gpt2style/
    BEST/            SELECTION.json, submission.zip, best.pt, run_config.json, DONE.json
    summary.csv  summary.json
  INDEX.md           what ran / finished / failed and the exact next command
```

## 3. Rules and how the code respects them

* **Starts from the Phase-1 weights only.** `load_phase1_lm` rebuilds each model from its own
  `config.json` and loads `model.pt` with `strict=True`. The model code in `phase1_models.py` is
  copied verbatim from notebook cells 13 and 15 (only a trailing `print` removed).
* **Six methodologically different experiments, LoRMA compulsory, one KD.** E1-E6 differ only in
  the technique; head, chunking, dev split, decoding and attention mode are shared.
* **All four models in every notebook**, best-of-4 submitted. **Coverage guard:**
  `python select_best.py --coverage` lists which model won each notebook and, if a model never
  wins, which of its runs to upload too, so all four Phase-1 models appear among the submissions
  (at most 10 uploads; limits are 10/day, 40/phase).
* **Submission format** is checked by `validate_submission.py` before anything is marked done:
  only `prediction.json` at the zip root, exactly the fold3 ids, integer `0 <= start < end <=
  len(text)`, known labels, no overlaps.
* **No Hugging Face Hub writes, no remote checkpoints.** Storage: at most `last.pt` + `best.pt` per
  model, `last.pt` overwritten atomically every 30 min and deleted when the model finishes; a
  disk guard frees space or stops cleanly below 4 GB free.

## 4. Gates (run automatically; failures print STOP AND ASK)

| Gate | Check | If it fails |
|---|---|---|
| G0 | every gold entity: `text[start:end] == entity_text` (end exclusive); labels in `labels.txt` | stops with examples. If ends turn out inclusive, confirm with the official evaluator and rerun `ner_data.py build --allow-inclusive-ends` |
| overlaps | nested/crossing gold entities (IOB2 cannot represent them) | outer entity kept; stops if > 2% |
| G1 | gold -> token tags -> decoded spans round trip ("ceiling") >= 0.97 | learns single-character snapping rules (e.g. a final `.` merged with `,` into one BPE token) and keeps them only if they raise the ceiling; stops if still < 0.97 |
| G2 | tags -> overlapping windows -> centre-weighted merge -> decode == G1 spans | stops |
| pre-flight | 2-GPU smoke test (falls back to `NCCL_P2P_DISABLE=1`, then 1 GPU); projected minutes | model skipped and reported in `INDEX.md` |
| overfit-64 | 64 entity chunks, 150 steps at 10x LR must cut the loss by 80% | model skipped only if it learns almost nothing (likely label bug) |
| watchdog | dev F1 < 0.05 after ~100 steps | run stopped, reported |

`data_report.json` (written by the cache build) contains the gold schema it detected, all gate
numbers, the mismatch categories and examples, the dev split and the dataset statistics.

## 5. Design decisions (defaults) and why

| Topic | Default | Reason |
|---|---|---|
| tokenisation | Phase-1 BPE + `ner_tok.py` offsets | ids asserted identical to `tok.encode`; a character belongs to the token holding its first byte; leading spaces trimmed for spans |
| labels | IOB2, 25 tags, every token labelled | |
| chunks | 512 tokens incl. a leading EOS, train stride 384, inference stride 256 | all models were trained on 1024-token blocks; GPT-2 positions beyond 1023 are untrained |
| empty chunks | keep 30% (`--neg-keep`) | legal boilerplate drives false positives |
| attention | bidirectional (`--attn-mode bidir`) | NER needs right context; M3 keeps a symmetric 512 window, M4 uses symmetric ALiBi distance |
| head | dropout 0.1 + linear (CRF available: `--head crf`) | same head in all six so the technique is the only variable |
| decoding | constrained IOB2 Viterbi + O-shift tuned on dev | on synthetic data Viterbi beat argmax 0.995 vs 0.899 |
| dev split | document-level 15% of fold1+fold2, seed 42, every label in dev | the same split for every experiment |
| optimiser | AdamW, wd 0.01 (none on 1-D), clip 1.0, 6% warm-up then linear decay, effective batch 32 chunks | |
| LR | full FT 5e-5 (head 10x); LoRA/LoRMA/adapters 1e-3; partial FT 1e-4 top layer x0.85 per layer down; KD student 1e-4; head 1e-3 | then the LR probe tries x{1/3, 1, 3} |
| epochs | full/KD 6, partial 8, PEFT 10; early stop on dev F1, patience 3 | |
| LoRA | r 16, alpha 32, dropout 0.05, all linears (fused QKV as one) | |
| LoRMA+ | r 16, alpha 32; `h = (I + s B A) W0 x + b`, B = 0 at init; q, k, v slices separate | |
| adapters | bottleneck 64 after attention-out and MLP-out, + all norm weights | |
| partial FT | top 6 blocks + final norm + head | |
| KD | teacher = best E1 model; students = every 2nd layer of each Phase-1 model (74-77M params); loss 0.5 CE + 0.5 T^2 KL, T = 2 | same tokenizer -> token-level soft labels align exactly |
| AMP | the Phase-1 pattern: fp16 + GradScaler on T4 | T4 has no bf16 |
| DDP | `torch.distributed.run --standalone`, padded equal-length rank shards, eval on rank 0 | uneven shards would deadlock |

Selection metric everywhere: strict macro-F1 over all 12 labels (`macro_all`); the variant over
labels present in gold is reported too.

## 6. Extras (not among the six)

```
python run_experiment.py --exp X1                                     # BitFit (M4, M1 only)
python run_experiment.py --exp X2 --models M2                         # pruned student, no KD
python run_experiment.py --exp X3_crf --technique full_ft --head crf --models M4
python run_experiment.py --exp X4_ctx1024 --technique full_ft --chunk-len 1024 --models M4
python run_experiment.py --exp X6_refit --technique full_ft --refit --epochs <best_epoch> --skip-probe --models M4
python ner_predict.py --ensemble phase2_out/E1_full_ft/BEST,phase2_out/E3_lora/BEST --out phase2_out/X5_ensemble
python select_best.py --coverage
python aggregate.py                  # results.csv / results.md: technique x model table
```

## 7. Tests

* `python tests/test_core.py`: offsets on Unicode/long runs, IOB2 round trip, snapping, window
  merge, causal trunk == original Phase-1 forward (all four architectures), bidirectional
  attention sees the future and ignores padding, Mistral window, KD pruning, LoRA/LoRMA identity at
  init and merge equality, LoRMA parameter count, trainable-parameter sets, layer-wise LR, CRF
  against brute force, Viterbi legality, metrics.
* `python tests/test_e2e.py`: E1-E6 + extras on tiny models, 2-process DDP (gloo), resume after
  the time budget, restore from an attached output, coverage, aggregate, ensemble, validator.

## 8. Things to verify on the real data

1. **Gold schema.** The real `fold*-gold.json` layout was not available while writing this; the
   loader accepts the common layouts and `data_report.json` prints what it found. G0 will stop if
   it cannot parse it.
2. **Official evaluator.** Download the Phase-2 evaluation script (Drive link in section 18.1 of the
   PDF) and compare its macro-F1 on one `prediction.json` with `dev_report.json`.
3. **LoRMA.** Implemented from arXiv 2506.07621 (LoRMA+, additive identity inflation). The paper's
   shape notation is inconsistent; identity-at-init and merge tests pass. Read the paper before
   the viva.
