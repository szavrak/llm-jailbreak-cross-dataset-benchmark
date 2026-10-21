"""Resume-safe utilities: skip completed runs, aggregate results."""

import os
import json
import pandas as pd

from experiments.config import RESULT_COLUMNS


def load_completed_set(result_file):
    """Load all completed (method, dataset, seed, hparams) keys from CSV into a set.

    Call once at startup, then use the returned set for O(1) lookups
    instead of re-reading the CSV on every iteration.
    Uses on_bad_lines='skip' to survive corrupted rows (e.g. from a killed process).
    """
    completed = set()
    if not os.path.isfile(result_file):
        return completed
    try:
        df = pd.read_csv(result_file, on_bad_lines='skip')
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return completed
    except Exception:
        return completed

    for _, row in df.iterrows():
        hp = row.get('HyperParams', '')
        if pd.isna(hp):
            hp = ''
        key = (row['Method'], row['Dataset'], int(row['Seed']), hp)
        completed.add(key)
    return completed


def make_key(method, dataset, seed, hparams=None):
    """Build a lookup key matching the format used in load_completed_set."""
    hparams_str = json.dumps(hparams) if hparams is not None else ''
    return (method, dataset, int(seed), hparams_str)


def is_done(result_file, method, dataset, seed, hparams=None):
    """Check if (method, dataset, seed, hparams) combo already exists in CSV."""
    if not os.path.isfile(result_file):
        return False
    try:
        df = pd.read_csv(result_file, on_bad_lines='skip')
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False

    mask = (
        (df['Method'] == method) &
        (df['Dataset'] == dataset) &
        (df['Seed'] == seed)
    )
    if hparams is not None:
        hparams_str = json.dumps(hparams)
        mask = mask & (df['HyperParams'] == hparams_str)

    return mask.any()


def aggregate_results(results_dir, output_file):
    """Merge all per-session CSVs in results_dir, deduplicate, write to output_file."""
    all_dfs = []
    for fname in os.listdir(results_dir):
        if fname.endswith('.csv'):
            fpath = os.path.join(results_dir, fname)
            try:
                df = pd.read_csv(fpath)
                all_dfs.append(df)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue

    if not all_dfs:
        print("No CSV files found in", results_dir)
        return None

    merged = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by (Method, Dataset, Seed, HyperParams), keep last
    dedup_cols = ['Method', 'Dataset', 'Seed', 'HyperParams']
    merged = merged.drop_duplicates(subset=dedup_cols, keep='last')
    merged = merged.sort_values(['Method', 'Dataset', 'Seed']).reset_index(drop=True)

    # Ensure column order
    for col in RESULT_COLUMNS:
        if col not in merged.columns:
            merged[col] = ''
    merged = merged[RESULT_COLUMNS]

    merged.to_csv(output_file, index=False)
    print(f"Aggregated {len(merged)} results to {output_file}")
    return merged
