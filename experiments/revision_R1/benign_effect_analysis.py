"""R3-6 diagnostic: is cross-dataset transfer governed by the *benign* pool?

Hypothesis (the "Benign Distribution Effect"): transfer F1 between two datasets
is driven by how similar their BENIGN classes are, more than how similar their
JAILBREAK (attack) classes are.

Test (no new training; uses the frozen MiniLM encoder already in the pipeline):
  1. Encode each dataset's benign and jailbreak texts with frozen MiniLM.
  2. Per dataset, take the class centroid (mean of L2-normalized embeddings).
  3. For each directional pair, compute benign-centroid distance and
     attack-centroid distance (1 - cosine).
  4. Correlate transfer F1 (MiniLM+LR) with each distance, and run a
     two-predictor OLS to see which dominates. Report on both the directional
     pairs and the symmetrized (pair-averaged) view, since centroid distance is
     symmetric while F1 is directional.

Run:  USE_TF=0 python experiments/revision_R1/benign_effect_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import CROSS_DATASET_PAIRS  # noqa: E402

DATASETS = ['dan', 'jackhhao', 'jailbreakbench', 'advbench', 'harmbench']
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
CROSS_CSV = 'results/revision_R1/cross_dataset.csv'
FOCAL = 'MiniLM+LR'


def class_centroids():
    """Return {dataset: {0: centroid_benign, 1: centroid_attack}} (L2-normed)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER)
    cents = {}
    for d in DATASETS:
        df = pd.read_csv(f'data/{d}.csv').dropna(subset=['text'])
        cents[d] = {}
        for lab in (0, 1):
            texts = df[df['label'] == lab]['text'].astype(str).tolist()
            emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
            c = emb.mean(axis=0)
            cents[d][lab] = c / (np.linalg.norm(c) + 1e-12)
    return cents


def cos_dist(a, b):
    return float(1.0 - np.dot(a, b))


def transfer_f1():
    """Mean directional transfer F1 for the focal method, per (src,tgt)."""
    df = pd.read_csv(CROSS_CSV)
    df['F1Binary'] = pd.to_numeric(df['F1Binary'], errors='coerce')
    df = df[df['Method'] == FOCAL]
    out = {}
    for _, r in df.groupby('Dataset')['F1Binary'].mean().items():
        src, tgt = _.split('->')
        out[(src, tgt)] = r
    return out


def main():
    cents = class_centroids()
    f1 = transfer_f1()

    rows = []
    for src, tgt in CROSS_DATASET_PAIRS:
        if (src, tgt) not in f1:
            continue
        rows.append({
            'pair': f'{src}->{tgt}', 'src': src, 'tgt': tgt,
            'benign_dist': cos_dist(cents[src][0], cents[tgt][0]),
            'attack_dist': cos_dist(cents[src][1], cents[tgt][1]),
            'f1': f1[(src, tgt)],
        })
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} directional pairs (focal method: {FOCAL})\n")
    print(d.round(3).to_string(index=False))

    from scipy.stats import spearmanr, pearsonr
    print("\n--- Directional correlations (transfer F1 vs distance) ---")
    for col in ('benign_dist', 'attack_dist'):
        rho, p = spearmanr(d[col], d['f1'])
        r, pp = pearsonr(d[col], d['f1'])
        print(f"  {col:12s}  Spearman rho={rho:+.3f} (p={p:.3f})   "
              f"Pearson r={r:+.3f} (p={pp:.3f})")

    # Symmetrized view: average the two directions of each unordered pair.
    sym = {}
    for _, row in d.iterrows():
        key = tuple(sorted([row['src'], row['tgt']]))
        sym.setdefault(key, []).append(row)
    srows = []
    for key, items in sym.items():
        srows.append({
            'pair': f'{key[0]}<->{key[1]}',
            'benign_dist': items[0]['benign_dist'],   # symmetric anyway
            'attack_dist': items[0]['attack_dist'],
            'f1': np.mean([it['f1'] for it in items]),
        })
    s = pd.DataFrame(srows)
    print(f"\n--- Symmetrized view ({len(s)} unordered pairs) ---")
    print(s.round(3).to_string(index=False))
    for col in ('benign_dist', 'attack_dist'):
        rho, p = spearmanr(s[col], s['f1'])
        print(f"  {col:12s}  Spearman rho={rho:+.3f} (p={p:.3f})")

    # Two-predictor OLS (standardized) to see which dominates.
    print("\n--- OLS: f1 ~ z(benign_dist) + z(attack_dist) [directional] ---")
    X = d[['benign_dist', 'attack_dist']].copy()
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, 'const', 1.0)
    y = d['f1'].values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    yhat = X.values @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    print(f"  const={beta[0]:+.3f}  benign={beta[1]:+.3f}  attack={beta[2]:+.3f}  "
          f"R^2={1 - ss_res / ss_tot:.3f}")
    print("\nInterpretation: a more negative benign coefficient / stronger "
          "benign correlation than attack supports the Benign Distribution "
          "Effect (closer benign pools -> higher transfer).")


if __name__ == '__main__':
    main()
