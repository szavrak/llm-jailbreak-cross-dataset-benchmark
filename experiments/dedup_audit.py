import pandas as pd
import re
from pathlib import Path
from itertools import combinations

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ["dan", "jackhhao", "jailbreakbench", "advbench", "harmbench", "hawkins"]


def norm(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df = df.dropna(subset=["text"]).copy()
    df["key"] = df["text"].map(norm)
    return df


def main():
    data = {name: load(name) for name in DATASETS}

    print("=== DATASET SIZES ===")
    for n, df in data.items():
        dup_within = len(df) - df["key"].nunique()
        counts = df["label"].value_counts().to_dict()
        print(f"  {n:16s} rows={len(df):5d}  unique_keys={df['key'].nunique():5d}  "
              f"within-dataset dups={dup_within:4d}  labels={counts}")

    print("\n=== PAIRWISE OVERLAP (row-level, normalized text) ===")
    print("Columns: total_overlap | same_label | label_conflict | "
          "overlap_as_%_of_smaller")
    header = f"  {'A -> B':<36s}  {'overlap':>8s}  {'same_lbl':>8s}  "\
             f"{'conflict':>8s}  {'% smaller':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for a, b in combinations(DATASETS, 2):
        da, db = data[a], data[b]
        ka = dict(zip(da["key"], da["label"]))
        kb = dict(zip(db["key"], db["label"]))
        shared = set(ka.keys()) & set(kb.keys())
        same_label = sum(1 for k in shared if ka[k] == kb[k])
        conflict = len(shared) - same_label
        pct = 100.0 * len(shared) / min(len(set(ka)), len(set(kb))) if shared else 0.0
        rows.append({
            "pair": f"{a} <-> {b}",
            "overlap": len(shared),
            "same_label": same_label,
            "label_conflict": conflict,
            "pct_of_smaller": pct,
        })
        print(f"  {a+' <-> '+b:<36s}  {len(shared):>8d}  {same_label:>8d}  "
              f"{conflict:>8d}  {pct:>9.1f}%")

    print("\n=== OVERLAP BY CLASS (where it matters) ===")
    print("For each pair, how many overlaps are in jailbreak (label=1) vs benign (label=0):")
    for a, b in combinations(DATASETS, 2):
        da, db = data[a], data[b]
        ka = dict(zip(da["key"], da["label"]))
        kb = dict(zip(db["key"], db["label"]))
        shared = set(ka.keys()) & set(kb.keys())
        if not shared:
            continue
        jb_a = sum(1 for k in shared if ka[k] == 1)
        bn_a = sum(1 for k in shared if ka[k] == 0)
        print(f"  {a:16s} <-> {b:16s}  overlap={len(shared):5d}  "
              f"(jailbreak_in_A={jb_a:4d}, benign_in_A={bn_a:4d})")

    print("\n=== CROSS-DATASET LEAKAGE IMPACT ===")
    print("If training on A and testing on ALL of B, how many test rows are contaminated?")
    print(f"  {'train A':<16s} {'test B':<16s} {'B size':>8s}  "
          f"{'leaked':>8s}  {'%':>6s}")
    for a in DATASETS:
        for b in DATASETS:
            if a == b:
                continue
            da, db = data[a], data[b]
            keys_a = set(da["key"])
            leaked = db["key"].isin(keys_a).sum()
            pct = 100.0 * leaked / len(db) if len(db) else 0.0
            flag = "  <-- HIGH" if pct >= 10 else ""
            print(f"  {a:<16s} {b:<16s} {len(db):>8d}  {leaked:>8d}  "
                  f"{pct:>5.1f}%{flag}")

    print("\n=== MIXED LEAVE-ONE-OUT LEAKAGE ===")
    print("Training on all-but-target, how many target rows leak through other datasets?")
    for target in DATASETS:
        others = [d for d in DATASETS if d != target]
        pool_keys = set()
        for o in others:
            pool_keys |= set(data[o]["key"])
        db = data[target]
        leaked = db["key"].isin(pool_keys).sum()
        pct = 100.0 * leaked / len(db)
        flag = "  <-- HIGH" if pct >= 10 else ""
        print(f"  holdout={target:<16s} size={len(db):5d}  leaked={leaked:5d}  "
              f"({pct:5.1f}%){flag}")


if __name__ == "__main__":
    main()
