"""Emit LaTeX table bodies for the revision, computed from results/revision_R1/.

10 seeds. DeBERTa-v3 excluded. Per-cell entries are mean_{+-std} (sample std);
aggregate means carry 95% CIs. Prints ready-to-paste rows per table.

Run:  python experiments/revision_R1/make_tables.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RES = os.path.join(ROOT, 'results', 'revision_R1')

DS_COLS = [('advbench', 'AdvB'), ('dan', 'DAN'), ('harmbench', 'HarmB'),
           ('jailbreakbench', 'JBB'), ('jackhhao', 'jackhhao')]

# Display name -> source csv for in-distribution
INDIST_SRC = {
    'TF-IDF+LR': 'in_distribution_tfidf.csv', 'TF-IDF+SVM': 'in_distribution_tfidf.csv',
    'MiniLM+XGBoost': 'in_distribution_embeddings.csv',
    'Arctic+RF': 'in_distribution_embeddings.csv',
    'BERT-frozen+LR': 'in_distribution_embeddings.csv',
    'MiniLM+LR': 'in_distribution_embeddings.csv',
    'MiniLM+LR-NoWeight': 'in_distribution_embeddings.csv',
    'BERT': 'in_distribution_bert.csv', 'BERT-Weighted': 'in_distribution_bert.csv',
    'ModernBERT': 'in_distribution_bert.csv',
}
_cache = {}
def load(fn):
    if fn not in _cache:
        df = pd.read_csv(os.path.join(RES, fn))
        for c in ['F1Binary', 'F1Macro', 'PrecisionBinary', 'RecallBinary',
                  'NumParams', 'TrainTimeSec']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        _cache[fn] = df
    return _cache[fn]

def cell(vals):
    v = pd.Series(vals).dropna()
    return v.mean(), v.std(ddof=1)

def fmt(m, s):
    return f"{m:.3f}$_{{\\pm{s:.2f}}}$"


# RdYlGn cell shading (lightened toward white so black text stays legible),
# matching the transfer-heatmap colour scale. Used for tab:cross (R3-2).
import matplotlib
try:
    _RYG = matplotlib.colormaps['RdYlGn']
except Exception:
    from matplotlib import cm as _mcm
    _RYG = _mcm.get_cmap('RdYlGn')

def shade(v, keep=0.55):
    r, g, b, _ = _RYG(float(v))
    r = r * keep + (1 - keep)
    g = g * keep + (1 - keep)
    b = b * keep + (1 - keep)
    return f"{round(r*255):02X}{round(g*255):02X}{round(b*255):02X}"


# ── Table: in-distribution ────────────────────────────────────
def indist():
    methods = ['TF-IDF+LR', 'TF-IDF+SVM', 'MiniLM+XGBoost', 'Arctic+RF',
               'BERT-frozen+LR', 'BERT', 'ModernBERT', '__mid__', 'MiniLM+LR']
    # gather mean per (method,dataset)
    cellmean = {}; cellstr = {}; means = {}
    for m in methods:
        if m == '__mid__':
            continue
        df = load(INDIST_SRC[m]); df = df[df['Method'] == m]
        ds_means = []
        for key, _ in DS_COLS:
            mm, ss = cell(df[df['Dataset'] == key]['F1Binary'])
            cellmean[(m, key)] = mm; cellstr[(m, key)] = fmt(mm, ss)
            ds_means.append(mm)
        means[m] = np.mean(ds_means)
    best = {key: max(cellmean[(m, key)] for m in means) for key, _ in DS_COLS}
    best_mean = max(means.values())
    print("% ==== tab:indist body ====")
    for m in methods:
        if m == '__mid__':
            print("\\midrule"); continue
        disp = '\\textbf{MiniLM+LR}' if m == 'MiniLM+LR' else (
            'MiniLM\\allowbreak +XGBoost' if m == 'MiniLM+XGBoost' else m)
        cells = []
        for key, _ in DS_COLS:
            s = cellstr[(m, key)]
            if abs(cellmean[(m, key)] - best[key]) < 1e-9:
                s = s.replace(f"{cellmean[(m,key)]:.3f}", f"\\textbf{{{cellmean[(m,key)]:.3f}}}", 1)
            s = f"\\cellcolor[HTML]{{{shade(cellmean[(m,key)])}}}" + s
            cells.append(s)
        mn = f"{means[m]:.3f}"
        if abs(means[m] - best_mean) < 1e-9:
            mn = f"\\textbf{{{mn}}}"
        mn = f"\\cellcolor[HTML]{{{shade(means[m])}}}" + mn
        print(f"{disp:26s} & " + " & ".join(cells) + f" & {mn} \\\\")
    print()


# ── Table: cross-dataset ──────────────────────────────────────
CROSS_PAIRS = [
    ('advbench','dan'),('advbench','harmbench'),('advbench','jailbreakbench'),('advbench','jackhhao'),
    ('dan','advbench'),('dan','harmbench'),('dan','jailbreakbench'),
    ('harmbench','advbench'),('harmbench','dan'),('harmbench','jackhhao'),('harmbench','jailbreakbench'),
    ('jackhhao','advbench'),('jackhhao','harmbench'),('jackhhao','jailbreakbench'),
    ('jailbreakbench','advbench'),('jailbreakbench','dan'),('jailbreakbench','harmbench'),('jailbreakbench','jackhhao'),
]
DISP = {'advbench':'AdvBench','dan':'DAN','harmbench':'HarmBench',
        'jailbreakbench':'JailbreakBench','jackhhao':'jackhhao'}

def cross():
    methods = ['TF-IDF+LR','TF-IDF+SVM','MiniLM+XGBoost','Arctic+RF',
               'BERT-frozen+LR','BERT','ModernBERT','MiniLM+LR']
    df = load('cross_dataset.csv')
    cm = {}; cs = {}
    for m in methods:
        sub = df[df['Method'] == m]
        for s, t in CROSS_PAIRS:
            mm, ss = cell(sub[sub['Dataset'] == f'{s}->{t}']['F1Binary'])
            cm[(m,s,t)] = mm; cs[(m,s,t)] = fmt(mm, ss)
    print("% ==== tab:cross body ====")
    for s, t in CROSS_PAIRS:
        best = max(cm[(m,s,t)] for m in methods)
        cells = []
        for m in methods:
            c = cs[(m,s,t)]
            if abs(cm[(m,s,t)] - best) < 1e-9:
                c = c.replace(f"{cm[(m,s,t)]:.3f}", f"\\textbf{{{cm[(m,s,t)]:.3f}}}", 1)
            c = f"\\cellcolor[HTML]{{{shade(cm[(m,s,t)])}}}" + c
            cells.append(c)
        print(f"{DISP[s]} $\\to$ {DISP[t]:14s} & " + " & ".join(cells) + " \\\\")
    # mean row
    print("\\midrule")
    pmean = {m: np.mean([cm[(m,s,t)] for s,t in CROSS_PAIRS]) for m in methods}
    best = max(pmean.values())
    cells = []
    for m in methods:
        v = f"{pmean[m]:.3f}"
        if abs(pmean[m]-best) < 1e-9: v = f"\\textbf{{{v}}}"
        v = f"\\cellcolor[HTML]{{{shade(pmean[m])}}}" + v
        cells.append(v)
    print("\\textbf{Mean (18 pairs)} & " + " & ".join(cells) + " \\\\")
    print()


# ── Table: A1 equalized-weights ablation ──────────────────────
def ablation():
    print("% ==== tab:ablation (A1) — cross-dataset mean F1 +- 95% CI ====")
    cdf = load('cross_dataset.csv'); edf = load('in_distribution_embeddings.csv')
    bdf = load('in_distribution_bert.csv')
    rows = [
        ('MiniLM+LR', 'balanced', edf), ('MiniLM+LR-NoWeight', 'none', edf),
        ('BERT', 'none', bdf), ('BERT-Weighted', 'balanced', bdf),
    ]
    for disp, wt, idf in rows:
        ind = idf[idf['Method']==disp]['F1Binary']
        im, _ = cell(ind)
        # Pair-level CI: the directional pair is the sampling unit, not the
        # pair-and-seed cell. Average the ten seeds within each pair, then form
        # the normal-approximation interval over the 18 pair means.
        cr = cdf[cdf['Method']==disp][['Dataset','F1Binary']].dropna()
        pm = cr.groupby('Dataset')['F1Binary'].mean()
        cmn = pm.mean(); ci = 1.96*pm.std(ddof=1)/np.sqrt(len(pm))
        print(f"{disp:20s} & {wt:8s} & {im:.3f} & {cmn:.3f} $\\pm$ {ci:.3f} \\\\  % pair-level CI (n={len(pm)} pairs)")
    # Paired permutation + bootstrap on the cross-dataset gap MiniLM+LR vs BERT,
    # with the directional pair (not the pair-seed cell) as the sampling unit.
    print("\n% permutation test (MiniLM+LR vs BERT, cross-dataset, pair-level):")
    a = cdf[cdf['Method']=='MiniLM+LR'].groupby('Dataset')['F1Binary'].mean()
    b = cdf[cdf['Method']=='BERT'].groupby('Dataset')['F1Binary'].mean()
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    d = (j['a']-j['b']).values
    obs = d.mean(); n = len(d)
    rng = np.random.RandomState(0)
    cnt = sum(abs((d*rng.choice([1,-1],size=n)).mean()) >= abs(obs) for _ in range(100000))
    bs = [np.random.RandomState(k).choice(d, size=n, replace=True).mean() for k in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  mean paired diff = {obs:.3f}, permutation p = {cnt/100000:.4f}, "
          f"bootstrap 95% CI = [{lo:.3f}, {hi:.3f}] (n={n} directional pairs)")
    print()


# ── Table: DAN-facing breakdown ───────────────────────────────
def dan_metrics():
    methods = ['TF-IDF+LR','TF-IDF+SVM','MiniLM+XGBoost','Arctic+RF',
               'BERT-frozen+LR','BERT','ModernBERT','__mid__','MiniLM+LR']
    df = load('cross_dataset.csv')
    df['PrecisionBinary'] = pd.to_numeric(df['PrecisionBinary'], errors='coerce')
    df['RecallBinary'] = pd.to_numeric(df['RecallBinary'], errors='coerce')
    src = [f'dan->{t}' for t in ['advbench','harmbench','jailbreakbench']]
    tgt = [f'{s}->dan' for s in ['advbench','harmbench','jailbreakbench','jackhhao']]
    METR = ['F1Binary','F1Macro','PrecisionBinary','RecallBinary']
    print("% ==== tab:dan-metrics body ====")
    for m in methods:
        if m == '__mid__':
            print("\\midrule"); continue
        sub = df[df['Method']==m]
        vals = []
        for grp in (src, tgt):
            g = sub[sub['Dataset'].isin(grp)]
            for met in METR:
                vals.append(f"{pd.to_numeric(g[met],errors='coerce').mean():.3f}")
        disp = '\\textbf{MiniLM+LR}' if m=='MiniLM+LR' else (
            'MiniLM\\allowbreak +XGBoost' if m=='MiniLM+XGBoost' else m)
        print(f"{disp:26s} & " + " & ".join(vals) + " \\\\")
    print()


# ── Table: mixed leave-one-out ────────────────────────────────
def mixed():
    methods = ['TF-IDF+LR','TF-IDF+SVM','MiniLM+XGBoost','Arctic+RF',
               'BERT-frozen+LR','BERT','ModernBERT','MiniLM+LR']
    holds = [('advbench','AdvBench'),('harmbench','HarmBench'),
             ('jailbreakbench','JailbreakBench')]
    df = load('mixed.csv'); df['F1Binary']=pd.to_numeric(df['F1Binary'],errors='coerce')
    cm = {}; cs = {}
    for m in methods:
        sub = df[df['Method']==m]
        for hk,_ in holds:
            mm,ss = cell(sub[sub['Dataset']==f'mixed->{hk}']['F1Binary'])
            cm[(m,hk)]=mm; cs[(m,hk)]=fmt(mm,ss)
    print("% ==== tab:mixed body ====")
    for hk,hd in holds:
        best = max(cm[(m,hk)] for m in methods)
        cells=[]
        for m in methods:
            c = cs[(m,hk)]
            if abs(cm[(m,hk)]-best)<1e-9:
                c = c.replace(f"{cm[(m,hk)]:.3f}", f"\\textbf{{{cm[(m,hk)]:.3f}}}",1)
            c = f"\\cellcolor[HTML]{{{shade(cm[(m,hk)])}}}" + c
            cells.append(c)
        print(f"{hd:14s} & " + " & ".join(cells) + " \\\\")
    print("\\midrule")
    pmean = {m: np.mean([cm[(m,hk)] for hk,_ in holds]) for m in methods}
    best = max(pmean.values()); cells=[]
    for m in methods:
        v=f"{pmean[m]:.3f}"
        if abs(pmean[m]-best)<1e-9: v=f"\\textbf{{{v}}}"
        v = f"\\cellcolor[HTML]{{{shade(pmean[m])}}}" + v
        cells.append(v)
    print("\\textbf{Mean}  & " + " & ".join(cells) + " \\\\")
    print()


# ── Table: efficiency ─────────────────────────────────────────
def efficiency():
    rows = [('TF-IDF+LR','in_distribution_tfidf.csv'),
            ('TF-IDF+SVM','in_distribution_tfidf.csv'),
            ('MiniLM+XGBoost','in_distribution_embeddings.csv'),
            ('Arctic+RF','in_distribution_embeddings.csv'),
            ('BERT-frozen+LR','in_distribution_embeddings.csv'),
            ('BERT','in_distribution_bert.csv'),
            ('ModernBERT','in_distribution_bert.csv'),
            ('__mid__',None),
            ('MiniLM+LR','in_distribution_embeddings.csv')]
    print("% ==== tab:efficiency body ====")
    for m, fn in rows:
        if m=='__mid__':
            print("\\midrule"); continue
        df = load(fn); sub = df[df['Method']==m]
        p = int(round(pd.to_numeric(sub['NumParams'],errors='coerce').mean()))
        t = pd.to_numeric(sub['TrainTimeSec'],errors='coerce').mean()
        pstr = f"{p:,}".replace(',', '{,}')
        disp = '\\textbf{MiniLM+LR}' if m=='MiniLM+LR' else (
            'MiniLM\\allowbreak +XGBoost' if m=='MiniLM+XGBoost' else m)
        print(f"{disp:26s} & {pstr:18s} & {t:.2f} \\\\")
    print()


if __name__ == '__main__':
    indist(); cross(); dan_metrics(); mixed(); efficiency(); ablation()
