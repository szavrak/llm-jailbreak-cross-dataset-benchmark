"""Runner for fine-tuned BERT baseline on frozen manifest splits.

BERT uses raw manifest text + HuggingFace tokenization; it does NOT call the
non-BERT TF-IDF preprocessor, so PreprocessTimeSec is reported as 0 and
tokenization time is charged to RepresentationTimeSec.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (SEEDS, DATASETS, BERT_HPARAMS,
                                 TRANSFORMER_METHODS, SPLIT_RATIO, REMOVE_LIMIT)
from experiments.shared_preprocessing import load_bert_split
from experiments.baselines.bert_classifier import train_and_evaluate_bert
from experiments.metrics import (compute_metrics, build_result_row,
                                  build_untrainable_row, save_result)
from experiments.resume_utils import load_completed_set, make_key
from preprocess import LABEL_MAP


RESULT_FILE = 'results/v2/in_distribution_bert.csv'


def _labels_to_int(labels_str):
    return [LABEL_MAP[l] for l in labels_str]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--methods', nargs='+', default=['BERT'],
                        help='Subset of TRANSFORMER_METHODS keys to fine-tune.')
    cli_args = parser.parse_args()

    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    completed = load_completed_set(cli_args.result_file)
    methods = [m for m in cli_args.methods if m in TRANSFORMER_METHODS]
    if not methods:
        print("No valid methods selected."); return

    for dataset in cli_args.datasets:
        for seed in cli_args.seeds:
            pending = [m for m in methods
                       if make_key(m, dataset, seed,
                                   hparams=TRANSFORMER_METHODS[m]) not in completed]
            if not pending:
                print(f"[SKIP] {dataset} seed={seed} (all methods done)")
                continue

            data = load_bert_split(dataset, seed)

            if not data['trainable']:
                reason = data['untrainable_reason'] or 'unknown'
                print(f"  [UNTRAINABLE] {dataset} seed={seed}: {reason}")
                for name in pending:
                    save_result(build_untrainable_row(
                        name, dataset, seed, SPLIT_RATIO, REMOVE_LIMIT,
                        data['n_train'], data['n_val'], data['n_test'],
                        data['leakage_removed'], reason,
                        hparams=TRANSFORMER_METHODS[name], device=''),
                        cli_args.result_file)
                continue

            train_labels = _labels_to_int(data['train_labels_str'])
            val_labels = _labels_to_int(data['val_labels_str'])
            test_labels = _labels_to_int(data['test_labels_str'])
            y_test = np.array(test_labels)

            for name in pending:
                hparams = TRANSFORMER_METHODS[name]
                print(f"\n[RUN] {name} {dataset} seed={seed}")
                wall_start = time.time()
                try:
                    result = train_and_evaluate_bert(
                        train_texts=data['train_sentences'],
                        train_labels=train_labels,
                        val_texts=data['val_sentences'],
                        val_labels=val_labels,
                        test_texts=data['test_sentences'],
                        test_labels=test_labels,
                        model_name=hparams['model_name'],
                        lr=hparams['lr'],
                        epochs=hparams['epochs'],
                        batch_size=hparams['batch_size'],
                        max_len=hparams['max_len'],
                        early_stopping_patience=hparams['early_stopping_patience'],
                        seed=seed,
                        class_weight=hparams.get('class_weight'),
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
                wall_time = time.time() - wall_start

                metrics = compute_metrics(y_test, result['predictions'])
                row = build_result_row(
                    method=name, dataset=dataset, seed=seed,
                    metrics_dict=metrics, split_ratio=SPLIT_RATIO,
                    remove_limit=REMOVE_LIMIT,
                    train_size=data['n_train'], val_size=data['n_val'],
                    test_size=data['n_test'],
                    leakage_removed=data['leakage_removed'],
                    preprocess_time=0.0,  # no non-HF preprocessing
                    representation_time=result['representation_time'],
                    train_time=result['train_time'],
                    infer_time=result['infer_time'],
                    total_time=wall_time,  # wall-clock around the whole run
                    num_params=result['num_params'],
                    hparams=hparams,
                    device=result['device'],
                )
                save_result(row, cli_args.result_file)
                print(f"  F1Binary={metrics['F1Binary']:.4f}  "
                      f"F1Macro={metrics['F1Macro']:.4f}  Acc={metrics['Accuracy']:.4f}")


def run_all():
    sys.argv = ['run_bert.py', '--result_file', RESULT_FILE]
    main()


if __name__ == '__main__':
    main()
