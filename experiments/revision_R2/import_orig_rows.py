"""R4-P3 — import the uncapped ("orig") detector rows for the length confound.

The length-confound comparison contrasts the uncapped ("orig") numbers with the
60-char-capped ones. The uncapped numbers for the 8 detectors are BYTE-IDENTICAL
to results already produced in the first revision round (results/revision_R1),
so we IMPORT them rather than burn GPU-hours recomputing. This guarantees the
orig-vs-60 table compares against genuinely frozen R1 results.

Sources (results/revision_R1 — these carry all 10 SEEDS_EXTENDED and ModernBERT):
  - DAN in-distribution:      in_distribution_{tfidf,embeddings,bert}.csv
  - the 6 DAN-facing cross pairs: cross_dataset.csv

For every imported row we:
  * keep the metrics / sizes / timings / NumParams / Device verbatim,
  * retag HyperParams with max_chars="orig" via the SAME _hparams_for_row used by
    run_length_confound.py (so orig and 60 keys never collide and both are
    resume-safe against results/revision_R2/length_confound.csv),
  * set SplitRatio="lenconf" and Notes="maxchars=orig; ..." to match the capped
    rows' convention.

VALIDATION (hard): exactly one source row per (Method, Dataset, Seed) for all
8 detectors x 10 seeds x 7 conditions. Any missing OR duplicate source row
raises — this is the guard against a silently-incomplete comparison table.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import SEEDS_EXTENDED
from experiments.metrics import build_result_row, save_result
from experiments.resume_utils import load_completed_set
# Reuse the exact hparam/condition/method definitions the capped runner uses so
# the imported orig rows are byte-compatible with the capped rows.
from run_length_confound import (_hparams_for_row, CONDITIONS,
                                  TFIDF_METHODS, EMB_METHODS, BERT_METHODS)


R1_DIR = 'results/revision_R1'
RESULT_FILE = 'results/revision_R2/length_confound.csv'

DETECTORS = TFIDF_METHODS + EMB_METHODS + BERT_METHODS   # the 8 detectors

# Which R1 in-distribution CSV holds each detector family's DAN rows.
_INDIST_SOURCE = {m: 'in_distribution_tfidf.csv' for m in TFIDF_METHODS}
_INDIST_SOURCE.update({m: 'in_distribution_embeddings.csv' for m in EMB_METHODS})
_INDIST_SOURCE.update({m: 'in_distribution_bert.csv' for m in BERT_METHODS})

METRIC_COLS = ['F1Binary', 'RecallBinary', 'PrecisionBinary', 'Accuracy',
               'F1Macro', 'RecallMacro', 'PrecisionMacro',
               'F1Weighted', 'RecallWeighted', 'PrecisionWeighted',
               'TN', 'FP', 'FN', 'TP']


def _read_csv(r1_dir, name):
    path = os.path.join(r1_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required R1 source not found: {path}")
    return pd.read_csv(path)


def _source_frame_for(condition, method, frames):
    """Return the R1 dataframe + dataset tag that should hold (method, condition)."""
    if '->' in condition:                      # a cross pair
        return frames['cross'], condition
    # in-distribution DAN — routed by detector family
    return frames[_INDIST_SOURCE[method]], 'dan'


def _verbatim_device(value):
    """Preserve a blank source Device verbatim (write '' not the literal 'nan').

    ~300 R1 in-distribution embedding rows and ~1080 cross rows carry a blank
    Device; str(NaN) would turn those into the literal string 'nan', which is
    not verbatim. Keep blank/whitespace-only values as ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    s = str(value)
    return '' if s.strip().lower() in ('', 'nan') else s


def _lookup_row(df, method, dataset, seed):
    """Return the single matching source row, or raise on missing / duplicate."""
    m = ((df['Method'] == method) & (df['Dataset'] == dataset) &
         (df['Seed'] == seed))
    hits = df[m]
    return hits


def build_orig_rows(r1_dir=R1_DIR, seeds=SEEDS_EXTENDED):
    """Collect and validate one orig row per (Method, Dataset, Seed).

    Returns (rows, expected_count). Raises RuntimeError listing every missing or
    duplicated (Method, Dataset, Seed) so an incomplete comparison table can
    never be produced silently.
    """
    frames = {
        'cross': _read_csv(r1_dir, 'cross_dataset.csv'),
        'in_distribution_tfidf.csv': _read_csv(r1_dir, 'in_distribution_tfidf.csv'),
        'in_distribution_embeddings.csv': _read_csv(r1_dir, 'in_distribution_embeddings.csv'),
        'in_distribution_bert.csv': _read_csv(r1_dir, 'in_distribution_bert.csv'),
    }

    missing, duplicated, rows = [], [], []
    for condition in CONDITIONS:
        for method in DETECTORS:
            df, dataset = _source_frame_for(condition, method, frames)
            src_name = ('cross_dataset.csv' if '->' in condition
                        else _INDIST_SOURCE[method])
            for seed in seeds:
                hits = _lookup_row(df, method, dataset, int(seed))
                if len(hits) == 0:
                    missing.append((method, condition, int(seed), src_name))
                    continue
                if len(hits) > 1:
                    duplicated.append((method, condition, int(seed),
                                       src_name, len(hits)))
                    continue
                src = hits.iloc[0]
                metrics = {c: src[c] for c in METRIC_COLS}
                hp = _hparams_for_row(method, None)   # mc=None -> max_chars="orig"
                row = build_result_row(
                    method=method, dataset=condition, seed=int(seed),
                    metrics_dict=metrics, split_ratio='lenconf',
                    remove_limit=int(src['RemoveLimit']),
                    train_size=int(src['TrainSize']), val_size=int(src['ValSize']),
                    test_size=int(src['TestSize']),
                    leakage_removed=int(src['LeakageRemoved']),
                    preprocess_time=float(src['PreprocessTimeSec']),
                    representation_time=float(src['RepresentationTimeSec']),
                    train_time=float(src['TrainTimeSec']),
                    infer_time=float(src['InferTimeSec']),
                    total_time=float(src['TotalRunTimeSec']),
                    num_params=int(src['NumParams']),
                    hparams=hp, device=_verbatim_device(src['Device']),
                    notes=f"maxchars=orig; imported from revision_R1/{src_name}")
                rows.append(row)

    expected = len(CONDITIONS) * len(DETECTORS) * len(seeds)
    if missing or duplicated:
        lines = [f"Refusing to import an incomplete orig comparison table "
                 f"(expected {expected} rows: "
                 f"{len(DETECTORS)} detectors x {len(seeds)} seeds x "
                 f"{len(CONDITIONS)} conditions)."]
        if missing:
            lines.append(f"MISSING {len(missing)} source row(s):")
            for m, c, s, f in missing:
                lines.append(f"  - {m} | {c} | seed={s} | expected in {f}")
        if duplicated:
            lines.append(f"DUPLICATE {len(duplicated)} source row(s):")
            for m, c, s, f, n in duplicated:
                lines.append(f"  - {m} | {c} | seed={s} | {n} copies in {f}")
        raise RuntimeError("\n".join(lines))

    return rows, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1_dir', type=str, default=R1_DIR)
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS_EXTENDED)
    cli_args = parser.parse_args()

    rows, expected = build_orig_rows(cli_args.r1_dir, cli_args.seeds)
    print(f"[validate] all {expected} orig source rows found "
          f"(exactly one per Method/Dataset/Seed).")

    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    completed = load_completed_set(cli_args.result_file)

    written = 0
    for row in rows:
        # HyperParams is already the JSON string load_completed_set keys on.
        key = (row['Method'], row['Dataset'], int(row['Seed']),
               row['HyperParams'])
        if key in completed:
            continue
        save_result(row, cli_args.result_file)
        written += 1

    print(f"[import_orig_rows] wrote {written} new orig rows "
          f"({expected - written} already present) -> {cli_args.result_file}")


if __name__ == '__main__':
    main()
