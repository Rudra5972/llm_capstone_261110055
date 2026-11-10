# CS781 Capstone — In-House Legal Intelligence System

Code for all seven phases of the CS781 legal-NLP capstone (IIT Kanpur). Everything is written in pure PyTorch, and only the official `Exploration-Lab/CS-781-Capstone` data is used. No pretrained weights, no `transformers`/`peft` model classes, and no external corpora. Model weights and checkpoints are **not** included.

| Folder | Task | Primary metric | Entry points |
|---|---|---|---|
| `phase1/` | Legal LM pretraining (512-word prompt → 128-word continuation) | BERTScore-F1 | `notebooks/00_bpe_tokenizer_and_data.ipynb`, then `01`–`04` |
| `phase2/` | Legal NER (12 entity types, character spans) | strict Macro-F1 | `notebooks/00_data_and_tests.ipynb`, `E1`–`E6` |
| `phase3/` | Rhetorical role prediction (13 roles, CL + IT) | Macro-F1 | `notebooks/00_data_tests_pilot.ipynb`, `E0`–`E11` |
| `phase4/` | Court judgment prediction (binary) | Macro-F1 | `notebooks/E1`–`E8` |
| `phase5/` | Legal statute identification (multi-label) | Macro-F1 | `notebooks/E1`–`E8` |
| `phase6/` | Abstractive summarization | BERTScore-F1 | `notebooks/E1`–`E8` |
| `phase7/` | Prior case retrieval (top-7) | Micro-F1@7 | `notebooks/E1`–`E8` |

Each phase folder has its own `README.md` (phases 2 and 4–7) or `RUN_ORDER_AND_CHECKLIST.md` (phase 3). These give the run order, the datasets to attach and the design decisions.

## Phase 1 — foundation models

**Tokenizer.** We wrote our own byte-level BPE (`bpe_tokenizer.py`) with 32,000 tokens (31,741 merges). It was trained on 2.0M unique words from all four sources, with no lowercasing or normalisation, and the round trip is exact. It averages 5.33 characters per token. EOS = 0, PAD = 1.

**Data.** All four sources (Acts, Supreme Court, Tribunals, High Courts) are streamed in their natural proportions. That gives 3,480,220 documents and 4.18B training tokens, stored as a `uint16` memmap. 1% of documents go to validation, and 47 prompt/continuation pairs are held out for local evaluation.

**Models.** Four from-scratch decoders, each with d = 768, 12 heads, tied embeddings and 1024-token training blocks:

| | Architecture | Layers | Positions | Attention | Norm / MLP | Params |
|---|---|---|---|---|---|---|
| M1 | GPT-2 style | 14 | learned absolute | full MHA | LayerNorm / GELU 4d | 125.38M |
| M2 | Llama style | 12 | RoPE θ=1e4 | GQA (4 KV heads) | RMSNorm / SwiGLU 3072 | 128.40M |
| M3 | Mistral style | 12 | RoPE | GQA + 512 sliding window | RMSNorm / SwiGLU 3072 | 128.40M |
| M4 | BLOOM style | 14 | ALiBi | full MHA | LayerNorm / GELU 4d | 123.81M |

**Training.** AdamW (0.9, 0.95), weight decay 0.1, peak LR 3e-4, 500 warm-up steps, cosine decay to 0. Each step is 4 × 1024 tokens with 8 accumulation steps (32,768 tokens). Gradient clipping 1.0, fp16 with GradScaler, one T4 per model. Checkpoints are saved every 500 steps and pushed to a private hub repo so a run can resume. 12,000 steps is the target, but the 9-hour budget binds first.

**Decoding.** Four settings are compared on the 47 local pairs, and the best is kept per model. Output is cut to exactly 128 words.

| Model | Steps | Final val loss | Local BERTScore | Decoding |
|---|---|---|---|---|
| M1 | 12,000 | 1.887 | 0.8239 | greedy, repetition 1.2, no repeated 4-grams |
| M2 | 11,150 | 1.852 | 0.8296 | T 0.8, top-p 0.95, repetition 1.1 |
| M3 | 10,950 | 1.866 | 0.8266 | greedy, repetition 1.2, no repeated 4-grams |
| M4 | 10,850 | 1.853 | 0.8295 | T 0.8, top-p 0.95, repetition 1.1 |

`bakeoff_results_1.csv` holds the first short bake-off of 10 architectures. It was used to narrow the choice to the four decoder-only families.

**Run (Kaggle, GPU T4 ×2, Internet on).**
1. Run `00_bpe_tokenizer_and_data.ipynb` and save its output as the dataset `cs781-phase1-data`.
2. Attach that dataset to each of `01`–`04`, set `HF_TOKEN` in Kaggle secrets for checkpoint resumption, then *Save & Run All*.
3. Each notebook writes `final/model.pt`, `final/config.json`, `final/bpe_model.json` and `submission.zip`.

## Phases 2–7 — shared transfer protocol

* Every phase loads **all four** Phase-1 checkpoints (`load_phase1_lm`, `strict=True`) and fine-tunes each of them. The best of the four is submitted, and the coverage tools make sure every foundation appears among the uploads.
* In the encoder tasks (2, 3, 4, 5 and the Phase-7 encoder), the trunk runs with a **bidirectional** mask. M3 keeps a symmetric 512 window and M4 uses symmetric ALiBi. Phase 6 keeps the causal LM.
* Within a phase, only the adaptation technique changes. Techniques implemented from scratch: full FT, partial FT (top 6 blocks, layer-wise LR decay 0.85), LoRA, rsLoRA, **LoRMA+** (compulsory), AdaLoRA, Houlsby and Pfeiffer adapters, IA3, BitFit, NF4 (QLoRA-style), prefix and prompt tuning, and **knowledge distillation**. KD uses a full-FT M4 teacher and students built from every 2nd layer of each foundation, with loss 0.5·CE + 0.5·T²·KL at T = 2.
* All inputs stay within the 1024-token pretraining length. Phase 2 uses 512-token windows. Phases 4–5 use up to 4 × 512 chunks (first 2 and last 2). Phase 6 uses head–tail or DANCER chunking. Phase 7 uses up to 16 × 256 chunks.
* Common infrastructure: a self-calibrating time plan; one job per GPU; atomic `last.pt` every 30 min with resume; OOM back-off; a validator for every `prediction.json`; and selftests (`python selftest.py`, `tests/`).

| Phase | Head / method | Decision rule |
|---|---|---|
| 2 | IOB2 (25 tags) linear head, optional CRF; character offsets from token offsets with learned punctuation snapping (span ceiling 98.25%) | constrained Viterbi + O-shift tuned on dev |
| 3 | EOS-marker sentence vectors over a sentence window, linear or BiGRU (`seqctx`) head, CL+IT joint | domain biases via 2-fold cross-fitting |
| 4 | masked-mean chunk vectors → attention pooling → 2-way head, sqrt-inverse class weights, leak guard vs test | threshold τ tuned on dev |
| 5 | same encoder, L sigmoid outputs with prior-initialised bias, BCE with pos_weight | global / cross-fitted per-label thresholds, 1 ≤ k ≤ k_max |
| 6 | causal LM on `[EOS] doc "SUMMARY:"`, loss on summary tokens | word cap chosen on dev; greedy / beam |
| 7 | shared bi-encoder, InfoNCE with BM25 hard negatives + GradCache | dense × BM25 fusion, top-7 |

## Submissions (Codabench user r_261110055, ranks recorded 9 Nov 2026)

| Phase | Task | Best primary-metric score | Rank | Valid submissions |
|---|---|---|---|---|
| 1 | Legal LM pretraining | 0.8287 (BERTScore-F1) | 94 | 4 |
| 2 | Legal NER | 0.5325 (strict Macro-F1) | 61 | 22 |
| 3 | Rhetorical roles | 0.4692 (Macro-F1) | 87 | 19 |
| 4 | Judgment prediction | 0.7348 (Macro-F1) | 60 | 7 |
| 5 | Statute identification | 0.3791 (Macro-F1) | 60 | 7 |
| 6 | Summarization | 0.8448 (BERTScore-F1) | 44 | 7 |
| 7 | Prior case retrieval | 0.3952 (Micro-F1@7) | 64 | 7 |

Total valid submissions: 73. Per-submission experiment and foundation mapping: _add from your Codabench history_.

## Report

`report/` in the submission zip contains the final PDF. GitHub: _add repository URL_.
