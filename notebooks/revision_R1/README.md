# Revision R1 experiments (KNOSYS-D-26-08416)

Colab notebooks delivering the experiments requested in the first-round review.
Run on Colab (GPU). Results are written to `results/revision_R1/` **on Drive**
(resume-safe across disconnects). The submitted `results/v2/` is never touched.

## What these add to v2

| Reviewer point | Change | Where it shows up |
|----------------|--------|-------------------|
| **R1-1 (A1)** confound: LR is class-weighted, BERT is not | `MiniLM+LR-NoWeight` (unweighted LR) **and** `BERT-Weighted` (inverse-frequency CE) — equalises both directions | all three protocols |
| **R1-2 (A2)** only 3 seeds | 10 seeds (`42,123,456` reused + 7 new) for confidence intervals / permutation tests | all methods, all protocols |
| **R3-3 (A3)** bert-base is outdated | `DeBERTa-v3-base` (2021) + `ModernBERT-base` (2024) fine-tuned baselines | all three protocols |

RoBERTa is wired in `config.TRANSFORMER_METHODS` but **not run** (ModernBERT
already covers recency + strength; DeBERTa-v3 covers strongest-classic).

## Notebooks (run order is flexible; all resume-safe)

1. `R1_A_InDistribution.ipynb` — 70/10/20 per dataset
2. `R1_B_CrossDataset.ipynb` — 18 directional pairs (the protocol R1 flagged for variance)
3. `R1_C_Mixed.ipynb` — leave-one-out mixed training

Each notebook: mounts Drive → copies code locally → builds the 7 new-seed
manifests → seeds `results/revision_R1/` from v2 (so existing 3-seed cells are
skipped) → runs that protocol. Within each, **cell 3a** is the cheap methods
(embeddings, TF-IDF, BERT, BERT-Weighted); **3b/3c** are DeBERTa-v3
and ModernBERT — each can run in its own Colab session and will resume.

## Compute note

The two modern encoders at 10 seeds across all protocols are the bulk of the
cost (~tens of GPU-hours total, dataset-dependent). Everything is resume-safe:
re-running a cell skips completed `(method, dataset, seed, hparams)` rows. If a
modern encoder OOMs on a T4, lower `batch_size` in `BERT_HPARAMS`
(`experiments/config.py`) or use an A100 runtime.

## After the runs

Aggregate with `experiments/resume_utils.aggregate_results` (or read the CSVs
directly) to regenerate the paper tables and the new performance-drop figure
(R3-2). The revision manuscript lives in `paper/revision_R1/`.
