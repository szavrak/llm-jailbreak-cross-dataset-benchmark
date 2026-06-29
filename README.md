# Cross-Dataset Generalization in LLM Jailbreak Prompt Detection

**Paper:** *An Empirical Study of Cross-Dataset Generalization in LLM Jailbreak Prompt Detection*
**Author:** Sultan Zavrak (Department of Computer Engineering, Duzce University)
**Status:** Under review at *Knowledge-Based Systems* (Elsevier)

> ⚠️ **Code release pending.** This repository currently contains only this summary.
> The full code (embedding generation, baselines, the pairwise contamination audit,
> evaluation scripts) and the frozen split manifests for the paper above **will be
> added to this repository** on publication.

A contamination-audited benchmark of how well jailbreak-prompt detectors transfer
across datasets.

## Overview

Jailbreak detectors deployed as guardrails in front of hosted LLM APIs are routinely
reported at F1 scores above 0.95, but those numbers almost always come from a random
in-distribution split of a *single* corpus. This study asks what happens under
realistic **cross-source distribution shift**: train on one jailbreak dataset, test
on another.

Eight detector configurations are compared across five public, English-language
jailbreak datasets under three evaluation protocols, preceded by a pairwise
exact-text contamination audit that removes leakage before any transfer is measured.

## Key findings

- **In-distribution scores collapse under transfer.** Detectors that reach a mean
  F1 of roughly 0.94 in-distribution fall to **0.35–0.61** under source-held-out
  cross-dataset transfer.
- **The capacity ordering inverts.** A *frozen* MiniLM-L6 Sentence-BERT embedding
  with a linear (logistic-regression) head is the most transfer-robust detector,
  attaining a mean cross-dataset F1 of **0.608** versus **0.393** for fine-tuned
  BERT-base (**+0.215**, paired permutation test *p* < 0.001 over 10 seeds) and
  **0.519** for the December-2024 ModernBERT encoder.
- **Not a class-weighting artefact.** An equalized-weights ablation (run in both
  directions) shows the advantage persists when class weighting is held equal.
- **Mixed training recovers most of the gap.** Pooling several sources
  (leave-one-out) lifts both frozen probes and fine-tuned transformers back to
  ~0.88–0.90 F1, so the collapse is specific to single-source transfer.
- **Lightweight and practical.** The frozen-embedding classifier has 385 trainable
  parameters and fits in ~50 ms, more than three orders of magnitude cheaper to
  train and update than fine-tuning BERT or ModernBERT.

## Datasets (English-language, public)

| Dataset | Total | Jailbreak | Benign | Release |
|---|---:|---:|---:|---|
| AdvBench | 1,040 | 520 | 520 | 2023-07 |
| DAN | 6,108 | 648 | 5,460 | 2023-08 |
| HarmBench | 793 | 393 | 400 | 2024-02 |
| JailbreakBench | 200 | 100 | 100 | 2024-04 |
| jackhhao | 1,033 | 517 | 516 | 2024-04 |

Counts reflect the preprocessed splits (exact-text deduplication, lowercasing). A
pairwise contamination audit found that jackhhao's jailbreak class is near-contained
in DAN's (79.5% overlap), so the two DAN↔jackhhao directional pairs are excluded from
the cross-dataset protocol.

## Detectors compared (8 configurations)

- **Lexical:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM
- **Frozen embeddings:** MiniLM + Logistic Regression, MiniLM + XGBoost,
  Arctic-Embed + Random Forest, BERT-frozen + Logistic Regression (disentangling control)
- **Fine-tuned encoders:** BERT-base, ModernBERT

## Evaluation protocols

1. **In-distribution** — 70/10/20 split within each dataset.
2. **Cross-dataset transfer** — train on dataset A, test on all of dataset B
   (18 directional pairs after contamination exclusion).
3. **Leave-one-out mixed** — pool N−1 datasets, test on the held-out dataset.

## Citation

The paper is currently under review. A citation will be added on acceptance.

```bibtex
@unpublished{zavrak2026crossdataset,
  title  = {An Empirical Study of Cross-Dataset Generalization in LLM Jailbreak Prompt Detection},
  author = {Zavrak, Sultan},
  note   = {Under review, Knowledge-Based Systems (Elsevier)},
  year   = {2026}
}
```

## Author

Sultan Zavrak — Department of Computer Engineering, Duzce University, Turkey
(`sultanzavrak@duzce.edu.tr`).

## License and data use

The jailbreak corpora referenced here are public, dual-use research artefacts and
remain bound by the licences of their original releases; they are referenced by
identifier rather than redistributed. A licence for the code in this repository will
be added with the code release.
