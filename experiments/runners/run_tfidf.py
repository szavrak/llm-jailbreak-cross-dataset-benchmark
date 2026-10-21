#!/usr/bin/env python3
"""Runner for TF-IDF + SVM and TF-IDF + LR baselines.

Usage:
    python experiments/runners/run_tfidf.py --result_file results/session_a_tfidf.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import SEEDS, DATASETS, SPLIT_RATIO, REMOVE_LIMIT
from experiments.shared_preprocessing import load_split, get_tfidf_features
from experiments.metrics import compute_metrics, build_result_row, save_result
from experiments.resume_utils import load_completed_set, make_key
from experiments.baselines.tfidf_classifiers import run_svm, run_lr

RESULT_FILE = 'results/v2/in_distribution_tfidf.csv'


def main():
    import time as _time

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    methods = [('TF-IDF+SVM', run_svm), ('TF-IDF+LR', run_lr)]
    method_names = [name for name, _ in methods]

    # Load completed runs once
    completed = load_completed_set(args.result_file)
    total = len(methods) * len(args.datasets) * len(args.seeds)
    skipped = sum(1 for m in method_names for d in args.datasets for s in args.seeds
                  if make_key(m, d, s) in completed)
    remaining = total - skipped
    print(f"Resume: {skipped}/{total} already done, {remaining} remaining.")
    if remaining == 0:
        print("All experiments completed!")
        return

    start_time = _time.time()
    run_count = 0

    for dataset in args.datasets:
        for seed in args.seeds:
            # Load data once per (dataset, seed) — both methods share it
            skip_all = all(
                make_key(name, dataset, seed) in completed
                for name, _ in methods
            )
            if skip_all:
                print(f"[SKIP] All TF-IDF methods done for {dataset} seed={seed}")
                continue

            print(f"[LOAD] {dataset} seed={seed}")
            # Shared load across both TF-IDF methods. We start each method's
            # wall-clock BEFORE its run_fn call but AFTER the shared
            # preprocessing+vectorization; to make TotalRunTimeSec enclose the
            # preprocess/representation stages reported on the same row, we add
            # those shared times into the per-method wall-clock.
            shared_load_start = _time.time()
            data = load_split(dataset, seed, remove_limit=REMOVE_LIMIT,
                              split_ratio=SPLIT_RATIO)
            X_train, X_val, X_test, y_train, y_val, y_test, rep_time = \
                get_tfidf_features(data)
            shared_load_elapsed = _time.time() - shared_load_start

            for method_name, run_fn in methods:
                if make_key(method_name, dataset, seed) in completed:
                    print(f"  [SKIP] {method_name} {dataset} seed={seed}")
                    continue

                elapsed = _time.time() - start_time
                eta_str = ""
                if run_count > 0:
                    avg = elapsed / run_count
                    eta = avg * (remaining - run_count)
                    eta_str = f" | ETA: {int(eta // 60)}m{int(eta % 60):02d}s"

                print(f"  [RUN] ({skipped+run_count+1}/{total}){eta_str}  "
                      f"{method_name} {dataset} seed={seed}")
                method_start = _time.time()
                y_pred, train_time, infer_time, num_params = run_fn(
                    X_train, X_test, y_train, y_test, seed)
                wall_time = shared_load_elapsed + (_time.time() - method_start)

                metrics = compute_metrics(y_test, y_pred)
                row = build_result_row(
                    method=method_name, dataset=dataset, seed=seed,
                    metrics_dict=metrics, split_ratio=SPLIT_RATIO,
                    remove_limit=REMOVE_LIMIT,
                    train_size=data['n_train'], val_size=data['n_val'],
                    test_size=data['n_test'],
                    leakage_removed=data['leakage_removed'],
                    preprocess_time=data['preprocess_time'],
                    representation_time=rep_time,
                    train_time=train_time,
                    infer_time=infer_time,
                    total_time=wall_time,
                    num_params=num_params,
                    device='cpu')
                save_result(row, args.result_file)
                run_count += 1
                print(f"    F1Binary={metrics['F1Binary']:.4f} F1Macro={metrics['F1Macro']:.4f} Acc={metrics['Accuracy']:.4f}")

    total_time = _time.time() - start_time
    print(f"Done. {run_count} runs completed in {int(total_time // 60)}m. "
          f"Total: {skipped+run_count}/{total}.")


def run_all():
    sys.argv = ['run_tfidf.py', '--result_file', RESULT_FILE]
    main()


if __name__ == '__main__':
    main()
