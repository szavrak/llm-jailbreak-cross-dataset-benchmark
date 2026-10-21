# Cross-Dataset Generalization in LLM Jailbreak Prompt Detection

**Paper:** *An Empirical Study of Cross-Dataset Generalization in LLM Jailbreak Prompt Detection*, Knowledge-Based Systems (accepted).
**Author:** Sultan Zavrak (Department of Computer Engineering, Duzce University)

This repository holds the code behind the paper: dataset download, the pairwise contamination audit, frozen split construction, the eight detector configurations, and the supplementary R2 analyses. The datasets and result CSVs are not redistributed. The download script fetches the public corpora, and the splits are rebuilt deterministically from fixed seeds.

## Setup

```bash
pip install -r requirements.txt   # Python 3.13; versions as recorded in results/revision_R2/provenance.txt
```

The fine-tuned transformers (BERT, ModernBERT) need a GPU. The paper's runs used Colab T4 and L4 GPUs. TF-IDF and the frozen-embedding heads run on a CPU.

## Reproducing the benchmark

```bash
# 1. Download and normalize the five public corpora (+ the diagnostic hawkins set) into data/
python data/download_datasets.py

# 2. Pairwise contamination audit (exact normalized-text overlap)
python experiments/dedup_audit.py

# 3. Frozen split manifests for the ten seeds used in the paper
python experiments/build_manifests.py --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021

# 4. Detectors, per protocol (resume-safe: completed rows are skipped on re-run)
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
OUT=results/revision_R1

# in-distribution (70/10/20)
python experiments/runners/run_tfidf.py --result_file $OUT/in_distribution_tfidf.csv --seeds $SEEDS
python experiments/runners/run_embedding.py --protocol in_distribution --result_file $OUT/in_distribution_embeddings.csv \
  --methods MiniLM+LR MiniLM+XGBoost Arctic+RF BERT-frozen+LR --seeds $SEEDS
python experiments/runners/run_bert.py --result_file $OUT/in_distribution_bert.csv --methods BERT ModernBERT --seeds $SEEDS

# source-held-out cross-dataset transfer (train on A, test on all of B)
python experiments/runners/run_cross_dataset.py --result_file $OUT/cross_dataset.csv \
  --methods BERT ModernBERT TF-IDF+LR TF-IDF+SVM --seeds $SEEDS
python experiments/runners/run_embedding.py --protocol cross_dataset --result_file $OUT/cross_dataset.csv \
  --methods MiniLM+LR MiniLM+XGBoost Arctic+RF BERT-frozen+LR --seeds $SEEDS

# leave-one-out mixed training
python experiments/runners/run_mixed.py --result_file $OUT/mixed.csv \
  --methods BERT ModernBERT TF-IDF+LR TF-IDF+SVM --seeds $SEEDS
python experiments/runners/run_embedding.py --protocol mixed --result_file $OUT/mixed.csv \
  --methods MiniLM+LR MiniLM+XGBoost Arctic+RF BERT-frozen+LR --seeds $SEEDS

# 5. Figures and table bodies from the result CSVs
python experiments/revision_R2/make_figures.py
python experiments/revision_R1/make_tables.py
```

The class-weighting and encoder ablations reported in the paper use the same runners with `BERT-Weighted`, `MiniLM+LR-NoWeight`, `MiniLM+XGBoost-Bal` and `DeBERTa-v3`. The method registry is in `experiments/config.py`.

The supplementary analyses (30-seed stability, the DAN length confound with the length-only baseline, and inference-memory profiling) live in `experiments/revision_R2/`. Their design and run order are documented in `notebooks/revision_R2/README.md`.

## Repository layout

| Path | Contents |
|------|----------|
| `data/download_datasets.py` | Fetches and normalizes AdvBench, DAN, HarmBench, JailbreakBench, jackhhao (and the derived hawkins set) |
| `experiments/dedup_audit.py`, `results/dedup_audit.txt` | Pairwise contamination audit and its output |
| `experiments/build_manifests.py`, `results/manifest_build_log.txt` | Deterministic split manifests (deduplication, test-side leakage removal) and the build log |
| `experiments/config.py` | Seeds, datasets, excluded contaminated pairs, all hyperparameters |
| `experiments/baselines/` | TF-IDF, frozen-embedding + head, and fine-tuned transformer implementations |
| `experiments/runners/` | One runner per protocol / detector family |
| `experiments/revision_R2/` | Supplementary analyses and coverage check |
| `notebooks/` | The Colab notebooks used for the runs. Clone this repository into your Drive root, or edit `CANDIDATES` in the first cell |

## License and data use

The code is released under the MIT License (see `LICENSE`). The jailbreak corpora are public, dual-use research artefacts and remain bound by the licences of their original releases. They are fetched from their sources rather than redistributed here.

## Contact

Sultan Zavrak — Department of Computer Engineering, Duzce University, Turkey (`sultanzavrak@duzce.edu.tr`).
