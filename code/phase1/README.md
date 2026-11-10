# CS781 Capstone — Phase 1: Legal LM Pretraining


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

