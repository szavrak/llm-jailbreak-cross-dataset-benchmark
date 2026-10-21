"""R3-2 figure: in-distribution vs cross-dataset F1 per method (with 95% CIs).

Makes the performance-drop figure the reviewer asked for, at 300 dpi, PNG (not
PDF). DeBERTa-v3 is excluded (not used in the paper).

Run:  python experiments/revision_R1/make_figures.py
Output: paper/figures/perf_drop.png  (+ .pdf for the LaTeX editable source)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RES = os.path.join(ROOT, 'results', 'revision_R1')
OUT_DIR = os.path.join(ROOT, 'paper', 'revision_R2', 'figures')

IN_DIST_FILES = ['in_distribution_embeddings.csv', 'in_distribution_bert.csv',
                 'in_distribution_tfidf.csv']
CROSS_FILE = 'cross_dataset.csv'

# Method order matches the paper's main tables (8 detector configurations).
METHODS = ['TF-IDF+LR', 'TF-IDF+SVM', 'MiniLM+XGBoost', 'Arctic+RF',
           'BERT-frozen+LR', 'BERT', 'ModernBERT', 'MiniLM+LR']
EXCLUDE = {'DeBERTa-v3'}


def mean_ci(df):
    """Cluster-level 95% CI. The sampling unit is the evaluation unit, not the
    individual seed: F1 is first averaged within each unit (the 'Dataset' column
    -- dataset name for in-distribution, directional pair for cross-dataset,
    held-out dataset for leave-one-out), then a normal-approximation interval is
    formed over those unit means. Pooling the pair-and-seed cells instead would
    treat correlated seeds as independent and understate the interval."""
    d = df[['Dataset', 'F1Binary']].copy()
    d['F1Binary'] = pd.to_numeric(d['F1Binary'], errors='coerce')
    unit = d.dropna(subset=['F1Binary']).groupby('Dataset')['F1Binary'].mean()
    if len(unit) == 0:
        return np.nan, np.nan
    n = len(unit)
    hw = 0.0 if n < 2 else 1.96 * unit.std(ddof=1) / np.sqrt(n)
    return unit.mean(), hw


def load_indist():
    frames = [pd.read_csv(os.path.join(RES, f)) for f in IN_DIST_FILES]
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ind = load_indist()
    cross = pd.read_csv(os.path.join(RES, CROSS_FILE))
    mixed = pd.read_csv(os.path.join(RES, 'mixed.csv'))
    for d in (ind, cross, mixed):
        d['F1Binary'] = pd.to_numeric(d['F1Binary'], errors='coerce')
    ind = ind[~ind['Method'].isin(EXCLUDE)]
    cross = cross[~cross['Method'].isin(EXCLUDE)]

    methods = [m for m in METHODS if m in set(ind['Method'])]
    ind_m, ind_e, cr_m, cr_e, lo_m, lo_e = ([] for _ in range(6))
    for m in methods:
        a, b = mean_ci(ind[ind['Method'] == m]); ind_m.append(a); ind_e.append(b)
        a, b = mean_ci(cross[cross['Method'] == m]); cr_m.append(a); cr_e.append(b)
        a, b = mean_ci(mixed[mixed['Method'] == m]); lo_m.append(a); lo_e.append(b)

    # Publication styling (colourblind-safe palette: blue / vermilion / teal-green).
    # Text rendered through LaTeX (Computer Modern serif) to match the manuscript.
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.3)
    plt.rcParams.update({'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
                         'figure.dpi': 300, 'text.usetex': True, 'font.family': 'serif',
                         'text.latex.preamble': r'\usepackage{lmodern}'})
    C_IN, C_CROSS, C_LOO = '#4C72B0', '#D55E00', '#029E73'
    err = dict(error_kw={'capsize': 2.5, 'ecolor': '#333333', 'elinewidth': 0.9, 'alpha': 0.9})

    x = np.arange(len(methods)); w = 0.27
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - w, ind_m, w, yerr=ind_e, label='In-distribution', color=C_IN, **err)
    ax.bar(x,     cr_m, w, yerr=cr_e, label='Cross-dataset (source-held-out)', color=C_CROSS, **err)
    ax.bar(x + w, lo_m, w, yerr=lo_e, label='Leave-one-out (mixed)', color=C_LOO, **err)

    ax.set_ylabel('Binary F1-score')
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha='right')
    # Legend above the plot (outside the axes) so it never overlaps the tall bars.
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3)
    ax.grid(axis='y', alpha=0.35); ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    # annotate the in-distribution-to-cross-dataset drop above each cross bar
    for xi, im, cm, ce in zip(x, ind_m, cr_m, cr_e):
        if not (np.isnan(im) or np.isnan(cm)):
            ax.annotate(f'$-${im - cm:.2f}', (xi, cm + ce),
                        textcoords='offset points', xytext=(0, 4),
                        ha='center', fontsize=12, color='#7a2f33', fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'perf_drop.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', os.path.relpath(path, ROOT))

    # print the underlying numbers for the table / caption
    print('\nMethod                 in-dist F1 (±CI)   cross F1 (±CI)   leave-one-out (±CI)   drop')
    for m, im, ie, cm, ce, lm, le in zip(methods, ind_m, ind_e, cr_m, cr_e, lo_m, lo_e):
        print(f'  {m:20s}  {im:.3f} ± {ie:.3f}   {cm:.3f} ± {ce:.3f}   {lm:.3f} ± {le:.3f}   -{im - cm:.3f}')


DS = ['dan', 'jackhhao', 'jailbreakbench', 'advbench', 'harmbench']
DISP = {'dan': 'DAN', 'jackhhao': 'jackhhao', 'jailbreakbench': 'JailbreakBench',
        'advbench': 'AdvBench', 'harmbench': 'HarmBench'}


def heatmap():
    """Cross-dataset transfer matrices (source x target) for MiniLM+LR vs ModernBERT.

    Diagonal and the excluded DAN<->jackhhao pairs are rendered as grey N/A.
    """
    import numpy.ma as ma
    df = pd.read_csv(os.path.join(RES, CROSS_FILE))
    df['F1Binary'] = pd.to_numeric(df['F1Binary'], errors='coerce')
    # Match the manuscript typography (Computer Modern serif via LaTeX).
    sns.set_theme(style='white', context='paper', font_scale=1.1)
    plt.rcParams.update({'figure.dpi': 300, 'text.usetex': True, 'font.family': 'serif',
                         'text.latex.preamble': r'\usepackage{lmodern}'})
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad('#d9d9d9')  # grey for N/A cells

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={'wspace': 0.12})
    im = None
    for ax, method in zip(axes, ['MiniLM+LR', 'ModernBERT']):
        M = np.full((5, 5), np.nan)
        sub = df[df['Method'] == method]
        for i, s in enumerate(DS):
            for j, t in enumerate(DS):
                if s == t:
                    continue
                v = sub[sub['Dataset'] == f'{s}->{t}']['F1Binary']
                if len(v):
                    M[i, j] = v.mean()
        Mm = ma.masked_invalid(M)
        im = ax.imshow(Mm, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([DISP[d] for d in DS], rotation=40, ha='right', fontsize=8.5)
        ax.set_xlabel('Test (target)', fontsize=9.5)
        if ax is axes[0]:
            # rows are the same datasets in both panels: label them once, on the left
            ax.set_yticklabels([DISP[d] for d in DS], fontsize=8.5)
            ax.set_ylabel('Train (source)', fontsize=9.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(method, fontsize=12, fontweight='bold', pad=8)
        # minor grid lines between cells
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)
        for i in range(5):
            for j in range(5):
                if np.isnan(M[i, j]):
                    if i != j:  # excluded (contaminated) pair, not the diagonal
                        ax.text(j, i, 'n/a', ha='center', va='center', fontsize=9,
                                color='#777', style='italic')
                else:
                    tc = 'white' if (M[i, j] < 0.22 or M[i, j] > 0.78) else 'black'
                    ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                            fontsize=10.5, color=tc)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('Mean cross-dataset F1 (ten seeds)', fontsize=9.5)
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'transfer_heatmap.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', os.path.relpath(path, ROOT))


if __name__ == '__main__':
    main()  # perf_drop only for R2 (heatmap unchanged)
