"""Load jailbreak datasets for all evaluation protocols.

v2 (post-contamination audit):
- New manifest-based loaders read frozen splits from splits/{protocol}/.../seed_{n}/.
- Every model consumes the same train.csv / val.csv / test.csv.
- Legacy on-the-fly splitters are kept for backwards compatibility only.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SPLITS_ROOT = Path(__file__).resolve().parent / "splits"


def peek_manifest(protocol, subdir, seed):
    """Cheap read of just the manifest.json for a split (no data loading).
    Used by runners to check `trainable` before deciding whether to pay the
    cost of a full load."""
    split_dir = SPLITS_ROOT / protocol / subdir / f"seed_{seed}"
    with open(split_dir / "manifest.json") as f:
        return json.load(f)


def _read_split_dir(split_dir):
    split_dir = Path(split_dir)
    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    with open(split_dir / "manifest.json") as f:
        manifest = json.load(f)
    return train, val, test, manifest


def _int_to_str_labels(labels):
    mapping = {0: 'benign', 1: 'jailbreak'}
    return [mapping[int(x)] for x in labels]


def _bundle(train, val, test, manifest):
    """Produce the dict format that existing runners consume."""
    # Leakage can be reported under different keys depending on when it was
    # stripped: per-split (_from_train/_from_val, legacy) or pre-split
    # (_from_source for cross, _from_pool for mixed, current).
    leakage = (manifest.get("leakage_removed_from_train", 0)
               + manifest.get("leakage_removed_from_val", 0)
               + manifest.get("leakage_removed_from_source", 0)
               + manifest.get("leakage_removed_from_pool", 0))
    return {
        'train_sentences': train['text'].tolist(),
        'val_sentences': val['text'].tolist(),
        'test_sentences': test['text'].tolist(),
        'train_labels': _int_to_str_labels(train['label']),
        'val_labels': _int_to_str_labels(val['label']),
        'test_labels': _int_to_str_labels(test['label']),
        'manifest': manifest,
        'leakage_removed': leakage,
        # `trainable` defaults to True for legacy manifests; new manifests set
        # it explicitly. Runners should skip False with an "untrainable" record.
        'trainable': bool(manifest.get('trainable', True)),
        'untrainable_reason': manifest.get('untrainable_reason'),
    }


def load_manifest_in_distribution(dataset, seed):
    split_dir = SPLITS_ROOT / "in_distribution" / dataset / f"seed_{seed}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split: {split_dir}")
    return _bundle(*_read_split_dir(split_dir))


def load_manifest_cross(src, tgt, seed):
    split_dir = SPLITS_ROOT / "cross_dataset" / f"{src}__to__{tgt}" / f"seed_{seed}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split: {split_dir}")
    return _bundle(*_read_split_dir(split_dir))


def load_manifest_mixed(holdout, seed):
    split_dir = SPLITS_ROOT / "mixed" / f"holdout_{holdout}" / f"seed_{seed}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split: {split_dir}")
    return _bundle(*_read_split_dir(split_dir))


# ─────────────────────────────────────────────────────────────────
# Legacy on-the-fly splitters. Superseded by the manifest loaders
# above. Retained only for debugging the old results set.
# ─────────────────────────────────────────────────────────────────


def get_dataset(args):
    """Load and split a single dataset for in-distribution evaluation.

    Split: 80/20 stratified (train+val / test), then 10% of train+val for val.
    Effective split: ~72/8/20 (train/val/test).
    """
    dataset = args.dataset
    random_state = args.random_state
    split_values = [float(x) for x in args.split_ratio.split(":")]
    test_frac = split_values[2] / sum(split_values)

    print(f"Loading {dataset}")
    df = pd.read_csv(f"data/{dataset}.csv")
    print(f"  Total: {len(df)} samples, {df['label'].sum()} jailbreak ({df['label'].mean()*100:.1f}%)")

    # Map labels to strings for compatibility with encode_labels
    label_map = {0: 'benign', 1: 'jailbreak'}
    df['label_str'] = df['label'].map(label_map)

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df['label'],
        random_state=random_state
    )

    train_sentences = train_df['text'].tolist()
    test_sentences = test_df['text'].tolist()
    train_labels = train_df['label_str'].tolist()
    test_labels = test_df['label_str'].tolist()

    return (train_sentences, test_sentences, train_labels, test_labels,
            len(train_sentences), len(test_sentences))


def get_dataset_cross(train_dataset, test_dataset, seed):
    """Load two datasets for cross-dataset evaluation.

    Train: 90% of train_dataset (10% held out as val for early stopping).
    Test: ALL of test_dataset.
    """
    train_df = pd.read_csv(f"data/{train_dataset}.csv")
    test_df = pd.read_csv(f"data/{test_dataset}.csv")

    label_map = {0: 'benign', 1: 'jailbreak'}

    # Split train dataset into train/val (90/10)
    train_split, val_split = train_test_split(
        train_df, test_size=0.1, stratify=train_df['label'],
        random_state=seed
    )

    # For the graph pipeline, we concatenate train+test as the full corpus
    # (train portion is used for graph construction, test for inference)
    train_sentences = train_split['text'].tolist()
    val_sentences = val_split['text'].tolist()
    test_sentences = test_df['text'].tolist()

    train_labels = train_split['label'].map(label_map).tolist()
    val_labels = val_split['label'].map(label_map).tolist()
    test_labels = test_df['label'].map(label_map).tolist()

    print(f"Cross-dataset: train={train_dataset} ({len(train_sentences)}+{len(val_sentences)} val) -> test={test_dataset} ({len(test_sentences)})")

    return {
        'train_sentences': train_sentences,
        'val_sentences': val_sentences,
        'test_sentences': test_sentences,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
    }


def get_dataset_mixed(datasets, holdout_dataset, seed):
    """Load N-1 datasets for training, hold out 1 for testing (leave-one-out).

    Train: 90% of merged N-1 datasets (10% val for early stopping).
    Test: ALL of holdout dataset.
    """
    train_dfs = []
    for ds in datasets:
        if ds != holdout_dataset:
            train_dfs.append(pd.read_csv(f"data/{ds}.csv"))

    merged_train = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.read_csv(f"data/{holdout_dataset}.csv")

    label_map = {0: 'benign', 1: 'jailbreak'}

    train_split, val_split = train_test_split(
        merged_train, test_size=0.1, stratify=merged_train['label'],
        random_state=seed
    )

    train_sentences = train_split['text'].tolist()
    val_sentences = val_split['text'].tolist()
    test_sentences = test_df['text'].tolist()

    train_labels = train_split['label'].map(label_map).tolist()
    val_labels = val_split['label'].map(label_map).tolist()
    test_labels = test_df['label'].map(label_map).tolist()

    train_sources = [ds for ds in datasets if ds != holdout_dataset]
    print(f"Mixed: train={'+'.join(train_sources)} ({len(train_sentences)}+{len(val_sentences)} val) -> test={holdout_dataset} ({len(test_sentences)})")

    return {
        'train_sentences': train_sentences,
        'val_sentences': val_sentences,
        'test_sentences': test_sentences,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
    }
