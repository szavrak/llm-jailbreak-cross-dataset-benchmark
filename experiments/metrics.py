"""Unified metric computation and CSV writing (v2 schema — post-decontamination).

Binary labels: 0 = benign, 1 = jailbreak. Primary metric is F1Binary.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from experiments.config import RESULT_COLUMNS


def compute_metrics(y_true, y_pred):
    """Compute the full metric set (Binary + Macro + Weighted + confusion matrix).

    Assumes binary labels {0, 1}. For safety, if only one class is present in
    either y_true or y_pred the binary-class metrics still compute via
    zero_division=0; confusion matrix is padded to 2x2.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1Binary': f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'RecallBinary': recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'PrecisionBinary': precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'F1Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'RecallMacro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'PrecisionMacro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'F1Weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'RecallWeighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'PrecisionWeighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }
    return metrics


def build_result_row(method, dataset, seed, metrics_dict,
                     split_ratio="70:10:20", remove_limit=2,
                     train_size=0, val_size=0, test_size=0, leakage_removed=0,
                     preprocess_time=0.0, representation_time=0.0,
                     train_time=0.0, infer_time=0.0, total_time=None,
                     num_params=0, hparams=None, device='', notes=''):
    """Build a single result row dict matching RESULT_COLUMNS (v2 schema)."""
    if total_time is None:
        total_time = preprocess_time + representation_time + train_time + infer_time

    row = {
        'Method': method,
        'Dataset': dataset,
        'Seed': seed,
        'SplitRatio': split_ratio,
        'RemoveLimit': remove_limit,
        'TrainSize': train_size,
        'ValSize': val_size,
        'TestSize': test_size,
        'LeakageRemoved': leakage_removed,
        'PreprocessTimeSec': round(preprocess_time, 4),
        'RepresentationTimeSec': round(representation_time, 4),
        'TrainTimeSec': round(train_time, 4),
        'InferTimeSec': round(infer_time, 4),
        'TotalRunTimeSec': round(total_time, 4),
        'NumParams': num_params,
        'Device': device,
        'HyperParams': json.dumps(hparams) if hparams else '',
        'Notes': notes,
    }
    row.update(metrics_dict)
    return row


def build_untrainable_row(method, dataset, seed, split_ratio, remove_limit,
                           train_size, val_size, test_size, leakage_removed,
                           reason, hparams=None, device=''):
    """Row emitted when a split has only one class in train or val so the
    result is not a model failure, it's a data-availability issue.

    HyperParams is kept pristine (same JSON as a normal run with the same
    hparams) so the row matches the resume key and is not rewritten on rerun.
    The reason goes in the Notes column.
    """
    row = {
        'Method': method,
        'Dataset': dataset,
        'Seed': seed,
        'SplitRatio': split_ratio,
        'RemoveLimit': remove_limit,
        'TrainSize': train_size,
        'ValSize': val_size,
        'TestSize': test_size,
        'LeakageRemoved': leakage_removed,
        'PreprocessTimeSec': 0.0,
        'RepresentationTimeSec': 0.0,
        'TrainTimeSec': 0.0,
        'InferTimeSec': 0.0,
        'TotalRunTimeSec': 0.0,
        'NumParams': 0,
        'Device': device,
        'HyperParams': json.dumps(hparams) if hparams else '',
        'Notes': f"UNTRAINABLE: {reason}",
    }
    for col in ['F1Binary', 'RecallBinary', 'PrecisionBinary', 'Accuracy',
                'F1Macro', 'RecallMacro', 'PrecisionMacro',
                'F1Weighted', 'RecallWeighted', 'PrecisionWeighted',
                'TN', 'FP', 'FN', 'TP']:
        row[col] = ''
    return row


def save_result(row, file_path):
    """Append one result row to a CSV file. Creates file with header if needed.

    Hard-fails if the existing file has a different header (schema mismatch),
    so v2 rows can never be appended to a v1 CSV and vice-versa. Use a distinct
    output path per schema version (e.g. results/v2/).
    """
    df = pd.DataFrame([row])
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    df = df[RESULT_COLUMNS]

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
        return

    with open(file_path, 'r') as f:
        header_line = f.readline().rstrip('\n').rstrip('\r')
    existing_cols = header_line.split(',') if header_line else []
    if existing_cols != list(RESULT_COLUMNS):
        raise RuntimeError(
            f"Schema mismatch: {file_path} has {len(existing_cols)} columns "
            f"but current RESULT_COLUMNS has {len(RESULT_COLUMNS)}. "
            f"Write to a new file (e.g. under results/v2/) instead of "
            f"appending. Existing header: {existing_cols[:5]}..."
        )
    df.to_csv(file_path, mode='a', header=False, index=False)
