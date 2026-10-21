"""Runner for embedding + simple-classifier baselines across all three protocols.

Usage:
    python experiments/runners/run_embedding.py --protocol in_distribution
    python experiments/runners/run_embedding.py --protocol cross_dataset
    python experiments/runners/run_embedding.py --protocol mixed

Methods (from config.EMBEDDING_METHODS) run by default:
    MiniLM+LR, MiniLM+XGBoost, Arctic+RF
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (SEEDS, DATASETS, SPLIT_RATIO, REMOVE_LIMIT,
                                 EMBEDDING_METHODS, EMBEDDING_HPARAMS,
                                 CROSS_DATASET_PAIRS, MIXED_HELDOUTS)
from experiments.shared_preprocessing import (load_bert_split,
                                               load_bert_split_cross,
                                               load_bert_split_mixed)
from experiments.baselines.embedding_classifier import train_and_evaluate
from experiments.metrics import (compute_metrics, build_result_row,
                                  build_untrainable_row, save_result)
from experiments.resume_utils import load_completed_set, make_key
from preprocess import LABEL_MAP


DEFAULT_RESULTS = {
    'in_distribution': 'results/v2/in_distribution_embeddings.csv',
    'cross_dataset':   'results/v2/cross_dataset.csv',
    'mixed':           'results/v2/mixed.csv',
}


def _labels_to_int(labels_str):
    return [LABEL_MAP[l] for l in labels_str]


def _hparams_for_method(method_name, classifier_kind):
    # HyperParams column stores a stable JSON dict so resume-keys are deterministic.
    return {
        'method': method_name,
        'classifier_kind': classifier_kind,
        'classifier_hparams': EMBEDDING_HPARAMS[classifier_kind],
        'max_seq_length': EMBEDDING_HPARAMS['max_seq_length'],
        'batch_size': EMBEDDING_HPARAMS['batch_size'],
    }


def _tags_for(protocol, cli_args):
    if protocol == 'in_distribution':
        return list(cli_args.datasets)
    if protocol == 'cross_dataset':
        return [f"{s}->{t}" for s, t in CROSS_DATASET_PAIRS]
    if protocol == 'mixed':
        return [f"mixed->{h}" for h in cli_args.holdouts]
    raise ValueError(f"Unknown protocol: {protocol}")


def _load_for(protocol, tag, seed):
    if protocol == 'in_distribution':
        return load_bert_split(tag, seed)
    if protocol == 'cross_dataset':
        src, tgt = tag.split('->')
        return load_bert_split_cross(src, tgt, seed)
    if protocol == 'mixed':
        return load_bert_split_mixed(tag.split('->')[1], seed)
    raise ValueError(f"Unknown protocol: {protocol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True,
                        choices=['in_distribution', 'cross_dataset', 'mixed'])
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of method names in EMBEDDING_METHODS')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--holdouts', nargs='+', default=MIXED_HELDOUTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    cli_args = parser.parse_args()

    result_file = cli_args.result_file or DEFAULT_RESULTS[cli_args.protocol]
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # SplitRatio column uses the protocol tag for cross-dataset and mixed
    # runs so those CSVs don't mislabel embedding rows as "70:10:20".
    split_ratio_tag = {
        'in_distribution': SPLIT_RATIO,
        'cross_dataset':   'cross',
        'mixed':           'mixed',
    }[cli_args.protocol]

    methods = [(n, m, c) for n, m, c in EMBEDDING_METHODS
               if cli_args.methods is None or n in cli_args.methods]
    if not methods:
        print("No methods selected."); return

    completed = load_completed_set(result_file)

    tags = _tags_for(cli_args.protocol, cli_args)
    total = len(methods) * len(tags) * len(cli_args.seeds)
    done_already = sum(
        1 for name, _, kind in methods
        for tag in tags for seed in cli_args.seeds
        if make_key(name, tag, seed,
                    hparams=_hparams_for_method(name, kind)) in completed)
    remaining = total - done_already
    print(f"[{cli_args.protocol}] resume: {done_already}/{total} done, "
          f"{remaining} remaining.")
    if remaining == 0:
        return

    start_time = time.time()
    run_count = 0

    for tag in tags:
        for seed in cli_args.seeds:
            if all(make_key(name, tag, seed,
                            hparams=_hparams_for_method(name, kind)) in completed
                   for name, _, kind in methods):
                continue

            data = _load_for(cli_args.protocol, tag, seed)

            for name, model_name, kind in methods:
                hp = _hparams_for_method(name, kind)
                if make_key(name, tag, seed, hparams=hp) in completed:
                    continue

                elapsed = time.time() - start_time
                eta = (elapsed / run_count * (remaining - run_count)) if run_count else 0
                eta_str = f" | ETA: {int(eta // 60)}m{int(eta % 60):02d}s" if run_count else ""
                print(f"[RUN] ({done_already + run_count + 1}/{total}){eta_str}  "
                      f"{name} {tag} seed={seed}")

                wall_start = time.time()

                if not data['trainable']:
                    reason = data['untrainable_reason'] or 'unknown'
                    print(f"  [UNTRAINABLE] {reason}")
                    save_result(build_untrainable_row(
                        name, tag, seed, split_ratio_tag, REMOVE_LIMIT,
                        data['n_train'], data['n_val'], data['n_test'],
                        data['leakage_removed'], reason,
                        hparams=hp, device=''), result_file)
                    run_count += 1
                    continue

                train_labels = _labels_to_int(data['train_labels_str'])
                val_labels = _labels_to_int(data['val_labels_str'])
                test_labels = _labels_to_int(data['test_labels_str'])

                try:
                    result = train_and_evaluate(
                        train_texts=data['train_sentences'],
                        train_labels=train_labels,
                        val_texts=data['val_sentences'],
                        val_labels=val_labels,
                        test_texts=data['test_sentences'],
                        test_labels=test_labels,
                        model_name=model_name,
                        classifier_kind=kind,
                        classifier_hparams=EMBEDDING_HPARAMS[kind],
                        protocol_tag=f"{cli_args.protocol}__{tag}",
                        seed=seed,
                        max_seq_length=EMBEDDING_HPARAMS['max_seq_length'],
                        batch_size=EMBEDDING_HPARAMS['batch_size'],
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                wall_time = time.time() - wall_start
                y_test = np.asarray(test_labels)
                metrics = compute_metrics(y_test, result['predictions'])
                row = build_result_row(
                    method=name, dataset=tag, seed=seed,
                    metrics_dict=metrics, split_ratio=split_ratio_tag,
                    remove_limit=REMOVE_LIMIT,
                    train_size=data['n_train'], val_size=data['n_val'],
                    test_size=data['n_test'],
                    leakage_removed=data['leakage_removed'],
                    preprocess_time=0.0,
                    representation_time=result['representation_time'],
                    train_time=result['train_time'],
                    infer_time=result['infer_time'],
                    total_time=wall_time,
                    num_params=result['num_params'],
                    hparams=hp,
                    device='',
                )
                save_result(row, result_file)
                run_count += 1
                print(f"  F1Binary={metrics['F1Binary']:.4f}  "
                      f"F1Macro={metrics['F1Macro']:.4f}  Acc={metrics['Accuracy']:.4f}")

    total_time = time.time() - start_time
    print(f"Done. {run_count} runs in {int(total_time // 60)}m. "
          f"Total: {done_already + run_count}/{total}.")


if __name__ == '__main__':
    main()
