"""Build frozen per-seed split manifests for all evaluation protocols.

Policy (confirmed with audit, see results/dedup_audit.txt):
- Main benchmark uses 5 independent datasets; hawkins is excluded.
- Within-dataset duplicates (normalized text key) are removed.
- For cross-dataset and mixed LOO, rows in train+val whose normalized key
  appears in the test set are removed (row-level leakage prevention).
- Manifests are deterministic per seed and written once.

Every detector (TF-IDF, frozen-embedding, fine-tuned transformer) reads these same files.
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"

MAIN_DATASETS = ["dan", "jackhhao", "jailbreakbench", "advbench", "harmbench"]
SEEDS = [42, 123, 456]

IN_DIST_TEST_FRAC = 0.20
IN_DIST_VAL_FRAC_OF_REMAINDER = 0.125  # 0.125 * 0.8 = 0.10 of total -> 70/10/20
CROSS_VAL_FRAC = 0.10  # of the training dataset
MIXED_VAL_FRAC = 0.10


# ── normalization (leakage-detection key only; NOT the tokenizer) ──
def norm(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_dataset(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df = df.dropna(subset=["text"]).copy()
    df["label"] = df["label"].astype(int)
    df["key"] = df["text"].map(norm)
    df["source"] = name
    return df


def dedup_within(df):
    """Resolve within-dataset duplicates by normalized text key.

    - Keys whose rows all share the same label: keep the first occurrence
      (deterministic, label-preserving).
    - Keys whose rows disagree on the label (same text, different classes):
      drop every copy entirely. Order-dependent "keep-first" on conflicting
      labels would bake source row order into the result.

    Returns (clean_df, same_label_dupes_removed, conflict_rows_removed,
             conflicts_detail).
    """
    key_labels = df.groupby("key")["label"].unique()
    conflict_keys = {k for k, labs in key_labels.items() if len(labs) > 1}
    conflicts_detail = [{"key": k, "labels": key_labels.loc[k].tolist()}
                        for k in conflict_keys]

    before = len(df)
    # Drop every row whose key is in conflict_keys
    df = df[~df["key"].isin(conflict_keys)]
    conflict_rows_removed = before - len(df)
    # Now safe to keep-first for the remaining (consistent-label) duplicates
    before2 = len(df)
    df = df.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)
    same_label_dups_removed = before2 - len(df)

    return df, same_label_dups_removed, conflict_rows_removed, conflicts_detail


def _remove_leaked(train_df, test_keys):
    before = len(train_df)
    mask = ~train_df["key"].isin(test_keys)
    out = train_df[mask].reset_index(drop=True)
    return out, before - len(out)


def save_split(out_dir, train, val, test, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["text", "label", "source"]
    train[cols].to_csv(out_dir / "train.csv", index=False)
    val[cols].to_csv(out_dir / "val.csv", index=False)
    test[cols].to_csv(out_dir / "test.csv", index=False)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def stratified_split(df, test_size, seed):
    return train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed,
    )


def _clean_split_ok(train_df, val_df):
    """Return (is_trainable, reason) — trainable means both classes appear in
    train AND in val (needed for meaningful loss + early stopping)."""
    if train_df["label"].nunique() < 2:
        return False, "train has only one class after decontamination"
    if val_df["label"].nunique() < 2:
        return False, "val has only one class after decontamination"
    return True, None


def build_in_distribution():
    out_root = SPLITS_DIR / "in_distribution"
    summary = []
    for ds in MAIN_DATASETS:
        df = load_dataset(ds)
        df, within_dups, conflict_rows, conflicts = dedup_within(df)
        for seed in SEEDS:
            remaining, test = stratified_split(df, IN_DIST_TEST_FRAC, seed)
            train, val = stratified_split(remaining, IN_DIST_VAL_FRAC_OF_REMAINDER, seed)
            train_clean, removed_t = _remove_leaked(train, set(test["key"]))
            val_clean, removed_v = _remove_leaked(val, set(test["key"]))
            trainable, reason = _clean_split_ok(train_clean, val_clean)
            manifest = {
                "protocol": "in_distribution",
                "dataset": ds,
                "seed": seed,
                "sizes": {"train": len(train_clean), "val": len(val_clean), "test": len(test)},
                "within_dataset_duplicates_removed": within_dups,
                "within_dataset_label_conflicts_removed": conflict_rows,
                "leakage_removed_from_train": removed_t,
                "leakage_removed_from_val": removed_v,
                "trainable": trainable,
                "untrainable_reason": reason,
            }
            out_dir = out_root / ds / f"seed_{seed}"
            save_split(out_dir, train_clean, val_clean, test, manifest)
            summary.append(manifest)
    return summary


def build_cross_dataset():
    """For each (src, tgt) pair:
       1. Strip rows from src whose normalized key appears in tgt (leakage).
       2. Stratified-split the cleaned source into 90/10 train/val.
    Leakage removal happens BEFORE splitting so class balance is honest and
    untrainable pairs are detectable without running a seed.
    """
    out_root = SPLITS_DIR / "cross_dataset"
    loaded = {}
    for ds in MAIN_DATASETS:
        cleaned, _dups, _conf, _ = dedup_within(load_dataset(ds))
        loaded[ds] = cleaned
    summary = []
    for src in MAIN_DATASETS:
        for tgt in MAIN_DATASETS:
            if src == tgt:
                continue
            src_df = loaded[src]
            tgt_df = loaded[tgt]
            test = tgt_df.copy()
            test_keys = set(test["key"])
            # Strip leakage FIRST
            src_clean, leakage_removed = _remove_leaked(src_df, test_keys)
            src_classes = src_clean["label"].nunique()
            for seed in SEEDS:
                if src_classes < 2:
                    # Can't stratify a single-class corpus — mark untrainable
                    # and write an empty split so the runner sees it.
                    train, val = src_clean.iloc[:0].copy(), src_clean.iloc[:0].copy()
                    trainable = False
                    reason = ("source has only one class after test-leakage "
                              "removal; stratified split is undefined")
                else:
                    try:
                        train, val = stratified_split(src_clean, CROSS_VAL_FRAC, seed)
                        trainable, reason = _clean_split_ok(train, val)
                    except ValueError as e:
                        train, val = src_clean.iloc[:0].copy(), src_clean.iloc[:0].copy()
                        trainable = False
                        reason = f"stratified split failed: {e}"
                manifest = {
                    "protocol": "cross_dataset",
                    "train_dataset": src,
                    "test_dataset": tgt,
                    "seed": seed,
                    "sizes": {"train": len(train), "val": len(val), "test": len(test)},
                    "leakage_removed_from_source": leakage_removed,
                    "leakage_removed_from_train": 0,  # already removed pre-split
                    "leakage_removed_from_val": 0,
                    "trainable": trainable,
                    "untrainable_reason": reason,
                }
                out_dir = out_root / f"{src}__to__{tgt}" / f"seed_{seed}"
                save_split(out_dir, train, val, test, manifest)
                summary.append(manifest)
    return summary


def build_mixed():
    """Mixed leave-one-out. Order of operations:
       1. Merge N-1 source datasets.
       2. Drop cross-source label conflicts (same text, different label across datasets).
       3. Drop cross-source duplicates (keep-first on consistent labels).
       4. Strip leakage against holdout test BEFORE splitting.
       5. Stratified-split into 90/10 train/val.
    """
    out_root = SPLITS_DIR / "mixed"
    loaded = {}
    for ds in MAIN_DATASETS:
        cleaned, _dups, _conf, _ = dedup_within(load_dataset(ds))
        loaded[ds] = cleaned
    summary = []
    for holdout in MAIN_DATASETS:
        sources = [d for d in MAIN_DATASETS if d != holdout]
        merged = pd.concat([loaded[d] for d in sources], ignore_index=True)
        test = loaded[holdout].copy()
        test_keys = set(test["key"])

        # Step 2-3: cross-source conflict + duplicate handling
        key_labels = merged.groupby("key")["label"].unique()
        conflict_keys = {k for k, labs in key_labels.items() if len(labs) > 1}
        merged_clean = merged[~merged["key"].isin(conflict_keys)]
        cross_source_conflicts = len(merged) - len(merged_clean)
        pool = merged_clean.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)
        cross_source_dups = len(merged_clean) - len(pool)

        # Step 4: strip holdout leakage BEFORE splitting
        pool_clean, leakage_removed = _remove_leaked(pool, test_keys)
        pool_classes = pool_clean["label"].nunique()

        for seed in SEEDS:
            if pool_classes < 2:
                train, val = pool_clean.iloc[:0].copy(), pool_clean.iloc[:0].copy()
                trainable = False
                reason = ("training pool has only one class after holdout "
                          "leakage removal")
            else:
                try:
                    train, val = stratified_split(pool_clean, MIXED_VAL_FRAC, seed)
                    trainable, reason = _clean_split_ok(train, val)
                except ValueError as e:
                    train, val = pool_clean.iloc[:0].copy(), pool_clean.iloc[:0].copy()
                    trainable = False
                    reason = f"stratified split failed: {e}"
            manifest = {
                "protocol": "mixed",
                "holdout": holdout,
                "training_sources": sources,
                "seed": seed,
                "sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "cross_source_duplicates_removed": cross_source_dups,
                "cross_source_label_conflicts_removed": cross_source_conflicts,
                "leakage_removed_from_pool": leakage_removed,
                "leakage_removed_from_train": 0,
                "leakage_removed_from_val": 0,
                "trainable": trainable,
                "untrainable_reason": reason,
            }
            out_dir = out_root / f"holdout_{holdout}" / f"seed_{seed}"
            save_split(out_dir, train, val, test, manifest)
            summary.append(manifest)
    return summary


def print_summary(tag, rows):
    print(f"\n=== {tag} ({len(rows)} splits) ===")
    for r in rows:
        extra = []
        for field, label in (("leakage_removed_from_train", "train_leak"),
                              ("leakage_removed_from_val", "val_leak"),
                              ("leakage_removed_from_source", "src_leak"),
                              ("leakage_removed_from_pool", "pool_leak")):
            if r.get(field):
                extra.append(f"{label}={r[field]}")
        if r.get("cross_source_duplicates_removed"):
            extra.append(f"cross_src_dup={r['cross_source_duplicates_removed']}")
        tag_ = r.get("dataset") or f"{r.get('train_dataset')}→{r.get('test_dataset')}" or r.get("holdout")
        if r["protocol"] == "mixed":
            tag_ = f"holdout={r['holdout']}"
        print(f"  [{r['protocol']:15s}] {tag_:32s} seed={r['seed']} "
              f"sizes={r['sizes']}  {' '.join(extra)}")


def main():
    import argparse
    global SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Seeds to build manifests for. Existing per-seed '
                             'dirs are deterministic, so passing only new seeds '
                             'leaves prior manifests untouched.')
    args = parser.parse_args()
    SEEDS = args.seeds

    SPLITS_DIR.mkdir(exist_ok=True)
    print(f"Writing manifests to {SPLITS_DIR} for seeds {SEEDS}")
    in_d = build_in_distribution()
    cross = build_cross_dataset()
    mixed = build_mixed()
    print_summary("IN-DISTRIBUTION", in_d)
    print_summary("CROSS-DATASET (first 6)", cross[:6])
    if len(cross) > 6:
        print(f"  ... {len(cross) - 6} more cross-dataset splits")
    print_summary("MIXED LOO", mixed)

    total_splits = len(in_d) + len(cross) + len(mixed)
    print(f"\nTotal: {total_splits} splits written.")

    # Aggregate leakage removal counts across all fields (pre-split + legacy)
    all_rows = in_d + cross + mixed
    fields = ("leakage_removed_from_train", "leakage_removed_from_val",
              "leakage_removed_from_source", "leakage_removed_from_pool")
    total_leak = sum(r.get(f, 0) for r in all_rows for f in fields)
    print(f"Total rows removed due to test-side leakage (all manifests): {total_leak}")


if __name__ == "__main__":
    main()
