# Phase 3 (Rhetorical Role Prediction) — run order, checklist, open points

Deadline: **16 Oct 2026, 11:59 PM (strict)**. Codabench scores each upload in 15–30 min. You get at most 10 uploads a day.

## 1. What is in this package

| File | Purpose |
|---|---|
| `notebooks/*.ipynb` | 13 Kaggle notebooks: `00_data_tests_pilot`, `E0`…`E10`, `E11_ensemble_report`. Every notebook embeds the same 18 modules and the same `EXPECTED_MD5`. |
| `modules/*.py` | The 18 shared modules (the same text as in the notebooks), with md5 values in `MODULE_MD5.json`. |
| `tools/make_notebooks.py` | Regenerates every notebook from `modules/` after an edit, which keeps the md5 values the same everywhere. |
| `CPU_TEST_EVIDENCE/` | Evidence from runs on CPU, using your sample JSONs and tiny random models of all four architectures: every experiment (pilot, E0–E11) end to end, the selftest results, the gates, the generated report, and the `00` and `E1` notebooks executed against a fake `/kaggle` tree (00's output attached to E1, as on Kaggle). The F1 values there are meaningless. They only show that the plumbing works. |

Five modules are byte-identical to Phase 2 (same md5 as your Phase-2 `EXPECTED_MD5`): `bpe_tokenizer.py`, `phase1_models.py`, `ner_common.py`, `ner_models.py` and `ner_peft.py`. The `Trunk` and the LoRA/LoRMA+ code are imported from them unchanged.

## 2. Run order

| Step | Notebook | Accelerator | Expected wall time on T4 x2* | Upload |
|---|---|---|---|---|
| 0 | `00_data_tests_pilot` | GPU T4 x2 | 20–30 min | none. Its output (`PILOT.json`, `phase3_data`) goes to every other notebook. |
| 1 | `E1_lorma`, `E2_lora`, `E6_kd` | GPU T4 x2 each | E1/E2 45–70 min; E6 1.5–2.5 h | E1, E2, E6 `BEST/submission.zip` |
| 2 | `E3_rslora`, `E4_ia3`, `E5_adalora` | GPU T4 x2 each | 45–75 min each | E3, E4, E5 |
| 3 | `E7_qlora`, `E8_prefix`, `E9_prompt`, `E0_full_ft`, `E10_ablations` | GPU T4 x2 each | 50–90 min each; E10 1.5–2.5 h | E7, E8, E9 (E0 optional, not counted) |
| 4 | `E11_ensemble_report` (attach all outputs) | None (CPU) | 5–10 min | E11 (optional extra) |

\*These are estimates, not measurements. They assume about 0.9M training tokens, 12–20k tokens/s for PEFT on one T4 and two jobs in parallel. Each job's smoke test measures the real throughput and prints a projection (`projected … min`) before training starts.

Steps 1–3 are independent and can run in parallel on different accounts or sessions. Priority 1 + 2 gives exactly 6 counted submissions (LoRMA, LoRA, KD, rsLoRA, IA3, AdaLoRA), which meets the rules: at least 6 experiments, at least 5 PEFT techniques, LoRMA included, at least 1 KD.

If time is short, skip `00`. The other notebooks then default to `bidir` attention and log that no pilot was attached.

## 3. Manual checklist per notebook

1. Create the notebook by uploading the `.ipynb`. Set the accelerator to **GPU T4 x2** (`E11`: None). Internet is only needed if the data comes from Hugging Face.
2. **Add Input → Datasets**:
   - `rudra1610/01-train-gpt2style`, `02-train-llamastyle`, `03-train-mistral-sliding`, `04-train-bloom-alibi` (Phase-1 `final/` folders: `model.pt`, `config.json`, `bpe_model.json`).
   - The official `Phase_3_RR` folder as a dataset (`Train/CL_train.json`, `CL_dev.json`, `IT_train.json`, `IT_dev.json`, `label_vocab.json`, `Test/CL_test.json`, `IT_test.json`). The alternative is the Kaggle secret `HF` (a read-only token) with internet on.
   - Recommended: the output of `00_data_tests_pilot`.
   - To resume: the output of an earlier session of the same notebook.
   - `E11` only: the outputs of every other notebook.
3. **Save Version → Save & Run All (commit)** so the run continues without the browser.
4. Cell 3 must print `all 18 modules verified`. If it fails, a module cell was edited. Regenerate the notebooks with `tools/make_notebooks.py`.
5. The gates cell must print `G0 passed`, `G1 passed`, `G2a passed` and `G2b passed`. If it prints **STOP AND ASK**:
   - A CL/IT test-ID collision means asking the TAs on Discord. Do not upload anything.
   - A label-vocab mismatch means the brief changed. Edit `rr_common.SPEC` and regenerate.
6. After the run, upload `phase3_out/<exp>/BEST/submission.zip`. The file has already been validated (`BEST/validation_report.json`).
7. Collect the output folders `phase3_out/` and `phase3_data/` and attach them to `E11`.
8. When all are done, run `python rr_select.py --coverage` (E11 does this too). It prints the winner per experiment and the rule compliance. If one of M1–M4 never wins, it lists that model's best zip as an extra upload, so all four models appear among the submissions.

Splitting a notebook across two accounts: add `--models M4,M2` on one account and `--models M3,M1` on the other. Then attach both outputs to the later notebooks.

## 4. Exit codes of `run_experiment.py`

`0` ok · `2` missing dependency (Phase-1 weights, data, KD teacher) · `3` gate or selftest failure · `4` submission validation failure · `1` other error, or nothing finished.

Per job, `rr_train.py` exits with `2` when the time budget is reached (checkpoint saved, resumable), `4` on NaN, `5` from the watchdog and `6` when a real-model self-check fails.

## 5. Decisions taken where the prompt was ambiguous or silent

| Topic | Decision |
|---|---|
| KD notebook id | The table wins: **E6 = KD** (section 7 says "E9"). E9 is Prompt Tuning. |
| Test file schema | Test files are `{id: [sentences]}` (from SCHEMA_REPORT). The loader also accepts `[{id, text}]`. |
| Causal vs bidir pilot | `00` runs M2 + LoRA for one epoch in each mode and writes `PILOT.json`. `--attn-mode auto` (the default) uses an attached `PILOT.json`. Otherwise it falls back to `bidir`. Hard-coding the winner into a module would break md5 parity across already-launched notebooks, so the winner is not hard-coded. |
| Orchestration | One job per GPU (two models in parallel). The Phase-2 DDP code is still in `rr_train.py`. |
| Window combination | `center` by default: the window where the sentence has the most context, meaning centre-most for bidir and most left context for causal. `--window-combine mean` averages the logits instead. |
| Bias tuning | 2-fold cross-fit over a 50/50 document split per domain. Biases fitted on the full dev set are applied only if the mean held-out gain is above 0. Selection always uses the unbiased dev F1. |
| Empty-class rule | sklearn `labels=None`: a class with zero gold and zero predictions is skipped. Everything else counts, with 0/0 → 0. |
| rsLoRA | r = 32, alpha = 16 (scale 2.83). A higher rank than LoRA is where the alpha/sqrt(r) scale matters. It shares the LoRA module. |
| AdaLoRA | Rank 12 → 8 average, cubic schedule (10% warm-up, final 20% with a fixed mask). Best checkpoint and early stopping only count epochs after the final budget is in place, so the submitted model has the target budget. The rank per layer is reported from that checkpoint. |
| Prefix tuning | P = 20, reparameterisation MLP with hidden size 256, dropped (`materialize`) before inference. Prefix keys are not rotated (RoPE), ALiBi bias is 0 for them, they have the KV-head shape under GQA, and they are always visible. |
| Prompt tuning | P = 20, initialised from random vocabulary embeddings. Prompt columns are visible to every query, including under M3's sliding window. Windows reserve P positions. |
| QLoRA | NF4 block 64 with int8 double quantisation of the scales (block 256, mean-centred) as the simulation of the QLoRA DQ. |
| Ensemble (E11) | Averaging in log-softmax space, because raw logits of different techniques have different scales. Candidates are uniform-all, F1-weighted-all and top-3, picked on dev, so the dev score is optimistic. |
| KD student size | 7/14 or 6/12 layers of the same model. Student < teacher is asserted. The 50M minimum is not enforced for students (brief rule 8). |

## 6. What could not be verified here (no GPU, no real data, no real weights)

1. **Real data.** Only your 40-sentence samples were available. The schema (train/dev lists, test as `{id: [sentences]}`) and the 13-label vocab match `SCHEMA_REPORT.txt`. Real tokens-per-sentence statistics, class frequencies and whether CL/IT test IDs collide are only checked by the gates on Kaggle. Judging by the names in the report, they do not collide.
2. **Real weights and the tokenizer.** Tests used tiny random models of all four architectures and a BPE trained on the samples. Loading uses the Phase-2 code path (strict `state_dict` match), unchanged.
3. **GPU behaviour.** fp16 autocast + GradScaler, cuDNN GRU (`seqctx`), on-the-fly NF4 de-quantisation, prefix KV and soft prompts under autocast, memory per micro-batch, and throughput. Everything ran in fp32 on CPU. The smoke test of every job re-checks zero-init, merge, trainable % and prefix visibility on the **real** model on the GPU before training.
4. **DDP.** The default is one process per GPU (two models in parallel). The torchrun/DDP path is kept from Phase 2 but was not exercised here.
5. **Wall times.** These are the estimates in section 2. The smoke tests measure the real throughput.
6. **Official metric pooling.** We optimise pooled Macro-F1 and also log per-domain and domain-averaged F1. The empty-class rule is sklearn's `labels=None` behaviour: classes absent from both gold and predictions are skipped.
7. **`%%writefile` byte parity on Kaggle.** The notebook construction is the same as in Phase 2, which already passed md5 verification on Kaggle.
8. **Dev set size.** The dev set has 10 documents and is used for early stopping, the LR probe, selection and the bias check. Expect noisy rankings. The 2-fold held-out bias check guards only the bias step.
