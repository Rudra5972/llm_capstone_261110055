# CS-781 Capstone -- Phase 3: Rhetorical Role Prediction

Sentence-level rhetorical-role classification of Indian legal judgments (Corporate Law and IT domains, 13 labels) with the four Phase-1 foundation models (M1 GPT-2 style, M2 Llama style, M3 Mistral style with sliding-window attention, M4 BLOOM style with ALiBi) and parameter-efficient fine-tuning implemented from scratch in plain PyTorch.

## Architecture

* `RRModel` = Phase-1 trunk (embeddings -> blocks -> final norm, LM head dropped) + sentence head. Documents are packed into <= 1024-token windows; every sentence ends with the EOS token, whose hidden state is the sentence vector; a domain prefix (`[CL]`/`[IT]`) starts each window. Long documents use overlapping windows; each sentence is predicted from the window where it has the most context.
* Heads: LayerNorm -> Dropout -> Linear(768, 13) (default); a BiGRU over the sentence sequence (`seqctx`) in the ablation.
* Loss: inverse-sqrt-frequency weighted cross-entropy (Macro-F1 punishes ignoring rare labels); focal / logit-adjusted / plain CE in the ablation.
* Post-hoc per-class logit biases, applied only when a 2-fold held-out check on dev shows a gain.

## Techniques

* **lorma** -- LoRMA+ (multiplicative low-rank, compulsory)
* **lora** -- LoRA
* **rslora** -- rsLoRA (rank-stabilised LoRA)
* **ia3** -- IA3
* **adalora** -- AdaLoRA (SVD + importance-based rank allocation)
* **qlora** -- QLoRA (simulated NF4 base + LoRA)
* **prefix** -- Prefix Tuning
* **prompt** -- Prompt Tuning
* **kd** -- Knowledge distillation (pruned student)
* **full_ft** -- Full fine-tuning (reference only) (reference, not counted)

rsLoRA shares the LoRA module (only the scale alpha/sqrt(r) and the rank differ) but is listed by the brief as its own technique, so it runs as its own experiment.
QLoRA is simulated without bitsandbytes: block-wise NF4 storage (block 64, double-quantised scales) with on-the-fly de-quantisation.

## How to run

1. `00_data_tests_pilot`: data gates (G0-G2), CPU selftest, causal-vs-bidir pilot -> `PILOT.json`.
2. E1, E2, E6 (priority 1) then E3, E4, E5 (priority 2), then E7, E8, E9, E0, E10 (priority 3) -- each is an independent Kaggle notebook (T4 x2): `python run_experiment.py --exp E<k>`.
3. E11: `python run_experiment.py --exp E11` (ensemble + this report) with the other outputs attached.
4. Upload `phase3_out/<exp>/BEST/submission.zip` for each experiment.

## Results

## Dev Macro-F1 per technique and model (pooled / CL / IT)

| exp | technique | M1 (gpt2style) | M2 (llamastyle) | M3 (mistral-sliding) | M4 (bloom-alibi) |
|---|---|---|---|---|---|
| E0 | Full fine-tuning (reference baseline, not counted) |  | 0.0463 / 0.070 / 0.050 |  | 0.0856 / 0.134 / 0.097 |
| E1 | LoRMA+ (compulsory) |  | 0.0020 / 0.004 / 0.000 |  | 0.0669 / 0.120 / 0.059 |
| E2 | LoRA |  | 0.0718 / 0.132 / 0.058 |  | 0.0855 / 0.134 / 0.096 |
| E3 | rsLoRA |  | 0.0442 / 0.050 / 0.065 |  | 0.0669 / 0.120 / 0.059 |
| E4 | IA3 |  | 0.0632 / 0.119 / 0.033 |  | 0.0550 / 0.056 / 0.083 |
| E5 | AdaLoRA |  | 0.0000 / 0.000 / 0.000 |  | 0.0658 / 0.088 / 0.078 |
| E6 | Knowledge distillation (self-contained full-FT teacher -> pruned student) |  | 0.0791 / 0.140 / 0.071 |  | 0.0876 / 0.149 / 0.082 |
| E7 | QLoRA (simulated NF4) |  | 0.0731 / 0.136 / 0.058 |  | 0.0669 / 0.120 / 0.059 |
| E8 | Prefix Tuning |  | 0.0680 / 0.111 / 0.057 |  | 0.0669 / 0.120 / 0.059 |
| E9 | Prompt Tuning |  | 0.0696 / 0.120 / 0.060 |  | 0.0529 / 0.050 / 0.083 |

## Best counted model: E6 / M4_bloom_alibi (dev pooled Macro-F1 0.0876)

Per-class dev scores:

| label | P | R | F1 | support | predicted |
|---|---|---|---|---|---|
| Fact | 0.4535 | 0.3900 | 0.4194 | 100 | 86 |
| Issue | 0.0000 | 0.0000 | 0.0000 | 6 | 0 |
| ArgumentPetitioner | 0.0000 | 0.0000 | 0.0000 | 37 | 1 |
| ArgumentRespondent | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| Statute | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |
| Dissent | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| PrecedentReliedUpon | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| PrecedentNotReliedUpon | 0.0000 | 0.0000 | 0.0000 | 14 | 0 |
| PrecedentOverruled | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| RulingByLowerCourt | 0.2483 | 0.7200 | 0.3692 | 50 | 145 |
| RatioOfTheDecision | 0.0000 | 0.0000 | 0.0000 | 19 | 0 |
| RulingByPresentCourt | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |
| None | 0.0000 | 0.0000 | 0.0000 | 4 | 0 |

Confusion matrix (rows gold, columns predicted):

| gold \ pred | Fact | Issue | Argume | Argume | Statut | Dissen | Preced | Preced | Preced | Ruling | RatioO | Ruling | None |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fact | 39 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 0 | 0 | 0 |
| Issue | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
| ArgumentPetitioner | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 0 | 0 |
| ArgumentRespondent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Statute | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dissent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PrecedentReliedUpon | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PrecedentNotReliedUpon | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 |
| PrecedentOverruled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RulingByLowerCourt | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 |
| RatioOfTheDecision | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| RulingByPresentCourt | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| None | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Ablations (E10)

| job | technique | model | variant | pooled F1 | CL | IT |
|---|---|---|---|---|---|---|
| lorma_base_M4 | lorma | M4 | base | 0.0646 | 0.1296 | 0.0540 |
| lorma_ce_M4 | lorma | M4 | ce | 0.1101 | 0.1871 | 0.1053 |
| lorma_ctx1_M4 | lorma | M4 | ctx1 | 0.0570 | 0.0816 | 0.0633 |
| lorma_focal_M4 | lorma | M4 | focal | 0.1091 | 0.1843 | 0.1053 |
| lorma_logitadj_M4 | lorma | M4 | logitadj | 0.0394 | 0.0723 | 0.0342 |
| lorma_onlyCL_M4 | lorma | M4 | onlyCL | 0.0534 | 0.0534 |  |
| lorma_onlyIT_M4 | lorma | M4 | onlyIT | 0.0342 |  | 0.0342 |
| lorma_perdomain_M4 | lorma | M4 | per_domain | 0.0467 | 0.0534 | 0.0342 |
| lorma_sent_M4 | lorma | M4 | sent | 0.0775 | 0.1337 | 0.0702 |
| lorma_seqctx_M4 | lorma | M4 | seqctx | 0.0191 | 0.0082 | 0.0326 |

## PEFT vs full fine-tuning gap (dev pooled Macro-F1, technique - full FT)

| exp | technique | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| E1 | lorma |  | -0.0444 |  | -0.0187 |
| E2 | lora |  | +0.0255 |  | -0.0001 |
| E3 | rslora |  | -0.0022 |  | -0.0187 |
| E4 | ia3 |  | +0.0169 |  | -0.0306 |
| E5 | adalora |  | -0.0463 |  | -0.0198 |
| E6 | kd |  | +0.0328 |  | +0.0020 |
| E7 | qlora |  | +0.0267 |  | -0.0187 |
| E8 | prefix |  | +0.0216 |  | -0.0187 |
| E9 | prompt |  | +0.0232 |  | -0.0327 |

## Knowledge distillation: teacher (full FT) vs pruned student

| model | teacher F1 | student F1 | F1 retained | teacher params | student params | size ratio | speed-up | memory ratio |
|---|---|---|---|---|---|---|---|---|
| M2 | 0.0669 | 0.0791 | 1.1820 | 225,805 | 151,821 | 0.6724 | 1.9063 |  |
| M4 | 0.0669 | 0.0876 | 1.3091 | 277,837 | 177,869 | 0.6402 | 1.4578 |  |

## Efficiency

| exp | job | technique | total params | trainable | trainable % | train min | train peak GB | train tok/s | infer tok/s |
|---|---|---|---|---|---|---|---|---|---|
| E0 | M2_llamastyle | full_ft | 225,805 | 225,805 | 100.0 | 0.0373 | 0.0000 | 6,486.9 | 29,369.1 |
| E0 | M4_bloom_alibi | full_ft | 277,837 | 277,837 | 100.0 | 0.2173 | 0.0000 | 1,114.0 | 7,877.1 |
| E1 | M2_llamastyle | lorma | 291,341 | 66,509 | 22.8 | 0.0411 | 0.0000 | 5,894.0 | 24,026.3 |
| E1 | M4_bloom_alibi | lorma | 351,565 | 74,701 | 21.2 | 0.2309 | 0.0000 | 1,048.2 | 4,589.6 |
| E2 | M2_llamastyle | lora | 291,341 | 66,509 | 22.8 | 0.0470 | 0.0000 | 5,154.8 | 27,701.4 |
| E2 | M4_bloom_alibi | lora | 343,373 | 66,509 | 19.4 | 0.2265 | 0.0000 | 1,068.6 | 6,426.7 |
| E3 | M2_llamastyle | rslora | 356,877 | 132,045 | 37.0 | 0.0483 | 0.0000 | 5,006.4 | 22,080.2 |
| E3 | M4_bloom_alibi | rslora | 408,909 | 132,045 | 32.3 | 0.3071 | 0.0000 | 788.3 | 5,165.4 |
| E4 | M2_llamastyle | ia3 | 226,573 | 1,741 | 0.7684 | 0.0349 | 0.0000 | 6,936.9 | 27,680.5 |
| E4 | M4_bloom_alibi | ia3 | 279,373 | 2,509 | 0.8981 | 0.2041 | 0.0000 | 1,186.2 | 7,277.1 |
| E5 | M2_llamastyle | adalora | 275,293 | 50,461 | 18.3 | 0.0443 | 0.0000 | 5,458.2 | 26,562.1 |
| E5 | M4_bloom_alibi | adalora | 327,181 | 50,317 | 15.4 | 0.3379 | 0.0000 | 716.3 | 5,030.4 |
| E6 | M2_llamastyle | kd | 151,821 | 151,821 | 100.0 | 0.0184 | 0.0000 | 13,181.2 | 66,422.2 |
| E6 | M4_bloom_alibi | kd | 177,869 | 177,869 | 100.0 | 0.1319 | 0.0000 | 1,835.4 | 11,420.3 |
| E7 | M2_llamastyle | qlora | 291,341 | 66,509 | 22.8 | 0.0453 | 0.0000 | 5,346.3 | 20,489.1 |
| E7 | M4_bloom_alibi | qlora | 343,373 | 66,509 | 19.4 | 0.3506 | 0.0000 | 690.4 | 5,033.6 |
| E8 | M2_llamastyle | prefix | 309,517 | 84,685 | 27.4 | 0.0332 | 0.0000 | 7,300.1 | 27,852.3 |
| E8 | M4_bloom_alibi | prefix | 427,341 | 150,477 | 35.2 | 0.1862 | 0.0000 | 1,300.0 | 6,742.8 |
| E9 | M2_llamastyle | prompt | 227,085 | 2,253 | 0.9921 | 0.0342 | 0.0000 | 6,941.8 | 26,565.7 |
| E9 | M4_bloom_alibi | prompt | 279,117 | 2,253 | 0.8072 | 0.1922 | 0.0000 | 1,234.8 | 6,389.5 |

AdaLoRA final rank per layer (M2): {'L0': 84, 'L1': 84, 'L2': 84, 'L3': 84}

AdaLoRA final rank per layer (M4): {'L0': 48, 'L1': 48, 'L2': 48, 'L3': 48}

## Rule compliance

`{'experiments_counted': 9, 'min_experiments': 6, 'peft_techniques': ['adalora', 'ia3', 'lora', 'lorma', 'prefix', 'prompt', 'qlora', 'rslora'], 'min_peft_techniques': 5, 'lorma_done': True, 'kd_done': True, 'ok': True}`


## Rule compliance

* experiments counted: 9 (min 6)
* PEFT techniques: ['adalora', 'ia3', 'lora', 'lorma', 'prefix', 'prompt', 'qlora', 'rslora'] (min 5; LoRMA done: True)
* KD experiment done: True
* every experiment starts from a Phase-1 model; no external weights, corpora or APIs; no transformers / peft / bitsandbytes / sklearn.

## References

* Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
* Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. arXiv:2305.14314.
* Zhang et al. (2023). Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (AdaLoRA). ICLR. arXiv:2303.10512.
* Kalajdzievski (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA). arXiv:2312.03732.
* Bihany, Patel & Modi (2025). LoRMA: Low-Rank Multiplicative Adaptation for LLMs. arXiv:2506.07621.
* Li & Liang (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL. arXiv:2101.00190.
* Lester, Al-Rfou & Constant (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP. arXiv:2104.08691.
* Liu et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (IA3). NeurIPS. arXiv:2205.05638.
* Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
* Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV (focal loss ablation).
* Menon et al. (2021). Long-tail learning via logit adjustment. ICLR (logit-adjust ablation).
