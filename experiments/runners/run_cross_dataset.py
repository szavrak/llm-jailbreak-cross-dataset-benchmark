"""Cross-dataset evaluation runner on frozen manifest splits.

Train on dataset A (90% train / 10% val), test on ALL of dataset B. Rows whose
normalized text appears in B have already been stripped from A's train+val by
the manifest builder.

Runs the fine-tuned transformers and the TF-IDF baselines. The frozen-embedding
baselines live in run_embedding.py (--protocol cross_dataset).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (SEEDS, REMOVE_LIMIT,
                                 BERT_HPARAMS,
                                 TRANSFORMER_METHODS, CROSS_DATASET_PAIRS)
from experiments.shared_preprocessing import (load_split_cross, load_bert_split_cross,
                                                get_tfidf_features)
from build_dataset import peek_manifest
from preprocess import LABEL_MAP
from experiments.baselines.bert_classifier import train_and_evaluate_bert
from experiments.baselines.tfidf_classifiers import run_svm, run_lr
from experiments.metrics import (compute_metrics, build_result_row,
                                  build_untrainable_row, save_result)
from experiments.resume_utils import load_completed_set, make_key


RESULT_FILE = 'results/v2/cross_dataset.csv'


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_name(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    return str(device)


def run_bert_cross(bert_data, hparams, seed):
    """BERT on a manifest split. Caller wraps wall-clock around this AND
    the preceding data load."""
    train_labels = [LABEL_MAP[l] for l in bert_data['train_labels_str']]
    val_labels = [LABEL_MAP[l] for l in bert_data['val_labels_str']]
    test_labels = [LABEL_MAP[l] for l in bert_data['test_labels_str']]

    result = train_and_evaluate_bert(
        train_texts=bert_data['train_sentences'],
        train_labels=train_labels,
        val_texts=bert_data['val_sentences'],
        val_labels=val_labels,
        test_texts=bert_data['test_sentences'],
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
    return (result['predictions'], np.array(test_labels),
            result['representation_time'], result['train_time'],
            result['infer_time'],
            result['num_params'], result['device'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--methods', nargs='+',
                        default=['BERT', 'TF-IDF+LR', 'TF-IDF+SVM'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    cli_args = parser.parse_args()

    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    completed = load_completed_set(cli_args.result_file)
    device = get_device()

    # Any requested method that is a fine-tuned transformer (BERT, BERT-Weighted,
    # RoBERTa, DeBERTa-v3, ModernBERT). All share the load_bert_split_cross load.
    transformer_methods = [m for m in cli_args.methods if m in TRANSFORMER_METHODS]
    run_lr_  = 'TF-IDF+LR'  in cli_args.methods
    run_svm_ = 'TF-IDF+SVM' in cli_args.methods
    run_tfidf = run_lr_ or run_svm_

    start_time = time.time()
    run_count = 0

    for train_ds, test_ds in CROSS_DATASET_PAIRS:
        ds_label = f"{train_ds}->{test_ds}"

        for seed in cli_args.seeds:
            pending_tf = [m for m in transformer_methods
                          if make_key(m, ds_label, seed,
                                      hparams=TRANSFORMER_METHODS[m]) not in completed]
            need_lr  = run_lr_  and make_key('TF-IDF+LR',  ds_label, seed) not in completed
            need_svm = run_svm_ and make_key('TF-IDF+SVM', ds_label, seed) not in completed
            if not (pending_tf or need_lr or need_svm):
                continue

            # Cheap manifest peek — used to detect untrainable splits without
            # paying the full preprocessing cost.
            manifest = peek_manifest('cross_dataset',
                                      f"{train_ds}__to__{test_ds}", seed)
            if not manifest.get('trainable', True):
                reason = manifest.get('untrainable_reason') or 'unknown'
                print(f"[UNTRAINABLE] {ds_label} seed={seed}: {reason}")
                sizes = manifest.get('sizes', {})
                n_train_m = sizes.get('train', 0)
                n_val_m = sizes.get('val', 0)
                n_test_m = sizes.get('test', 0)
                leak_m = (manifest.get('leakage_removed_from_source', 0)
                          + manifest.get('leakage_removed_from_train', 0)
                          + manifest.get('leakage_removed_from_val', 0))
                for name in pending_tf:
                    save_result(build_untrainable_row(
                        name, ds_label, seed, 'cross', REMOVE_LIMIT,
                        n_train_m, n_val_m, n_test_m, leak_m, reason,
                        hparams=TRANSFORMER_METHODS[name],
                        device=get_device_name(device)), cli_args.result_file)
                for name, need in [('TF-IDF+LR', need_lr), ('TF-IDF+SVM', need_svm)]:
                    if need:
                        save_result(build_untrainable_row(
                            name, ds_label, seed, 'cross', REMOVE_LIMIT,
                            n_train_m, n_val_m, n_test_m, leak_m, reason,
                            device='cpu'), cli_args.result_file)
                continue

            if pending_tf:
                # Load the manifest split once; every transformer method on this
                # (pair, seed) shares it.
                data_bert = load_bert_split_cross(train_ds, test_ds, seed)
                for name in pending_tf:
                    hp = TRANSFORMER_METHODS[name]
                    print(f"\n[RUN] {name} {ds_label} seed={seed}")
                    wall_start = time.time()
                    try:
                        y_pred, y_true, rep_t, tt, it, np_, dev = run_bert_cross(
                            data_bert, hp, seed)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        continue
                    wall_time = time.time() - wall_start
                    metrics = compute_metrics(y_true, y_pred)
                    row = build_result_row(
                        method=name, dataset=ds_label, seed=seed,
                        metrics_dict=metrics, split_ratio='cross',
                        remove_limit=REMOVE_LIMIT,
                        train_size=data_bert['n_train'], val_size=data_bert['n_val'],
                        test_size=data_bert['n_test'],
                        leakage_removed=data_bert['leakage_removed'],
                        preprocess_time=0.0,
                        representation_time=rep_t,
                        train_time=tt, infer_time=it, total_time=wall_time,
                        num_params=np_, hparams=hp, device=dev)
                    save_result(row, cli_args.result_file)
                    run_count += 1
                    print(f"  F1B={metrics['F1Binary']:.4f} F1M={metrics['F1Macro']:.4f} leak={data_bert['leakage_removed']}")

            if need_lr or need_svm:
                wall_start = time.time()
                data = load_split_cross(train_ds, test_ds, seed,
                                        remove_limit=REMOVE_LIMIT)
                X_train, X_val, X_test, y_train, y_val, y_test, rep_time = \
                    get_tfidf_features(data)
                shared_load = time.time() - wall_start
                for name, need, run_fn in [
                    ('TF-IDF+LR',  need_lr,  run_lr),
                    ('TF-IDF+SVM', need_svm, run_svm),
                ]:
                    if not need:
                        continue
                    print(f"\n[RUN] {name} {ds_label} seed={seed}")
                    m_start = time.time()
                    y_pred, train_time, infer_time, np_ = run_fn(
                        X_train, X_test, y_train, y_test, seed)
                    wall_time = shared_load + (time.time() - m_start)
                    metrics = compute_metrics(y_test, y_pred)
                    row = build_result_row(
                        method=name, dataset=ds_label, seed=seed,
                        metrics_dict=metrics, split_ratio='cross',
                        remove_limit=REMOVE_LIMIT,
                        train_size=data['n_train'], val_size=data['n_val'],
                        test_size=data['n_test'],
                        leakage_removed=data['leakage_removed'],
                        preprocess_time=data['preprocess_time'],
                        representation_time=rep_time,
                        train_time=train_time, infer_time=infer_time,
                        total_time=wall_time,
                        num_params=np_, device='cpu')
                    save_result(row, cli_args.result_file)
                    run_count += 1
                    print(f"  F1B={metrics['F1Binary']:.4f} F1M={metrics['F1Macro']:.4f} leak={data['leakage_removed']}")

    total_time = time.time() - start_time
    print(f"\nDone. {run_count} runs completed in {int(total_time // 60)}m.")


def run_all():
    sys.argv = ['run_cross_dataset.py', '--result_file', RESULT_FILE]
    main()


if __name__ == '__main__':
    main()
