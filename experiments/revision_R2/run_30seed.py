"""R4-P2 — 30-seed stability on a single cross-dataset pair.

Reviewer #4 asked whether the headline cross-dataset gap is stable under many
seeds, not just ten. This runs BERT, ModernBERT and MiniLM+LR on the single
pair jackhhao -> advbench (train jackhhao, test advbench) for all 30 seeds in
config.SEEDS_30, writing per-seed rows to results/revision_R2/stability_30seed.csv.

Aggregation (mean / CI / permutation test) is done in analysis, not here — this
script only produces per-seed result rows.

The frozen manifests for the twenty NEW seeds of THIS pair are built first
(reusing experiments.build_manifests logic). The existing ten seeds' manifests
are deterministic and left untouched.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (SEEDS_30, SEEDS_EXTENDED, REMOVE_LIMIT,
                                 TRANSFORMER_METHODS, EMBEDDING_METHODS,
                                 EMBEDDING_HPARAMS)
from experiments.shared_preprocessing import load_bert_split_cross
from experiments.baselines.embedding_classifier import train_and_evaluate
from experiments.metrics import (compute_metrics, build_result_row,
                                  build_untrainable_row, save_result)
from experiments.resume_utils import load_completed_set, make_key
from experiments.runners.run_cross_dataset import (run_bert_cross, get_device,
                                                    get_device_name)
from experiments.runners.run_embedding import _hparams_for_method, _labels_to_int
from build_dataset import peek_manifest

# Single-pair manifest build (reuse build_manifests primitives).
from experiments.build_manifests import (load_dataset, dedup_within,
                                          _remove_leaked, stratified_split,
                                          _clean_split_ok, save_split,
                                          SPLITS_DIR, CROSS_VAL_FRAC)


RESULT_FILE = 'results/revision_R2/stability_30seed.csv'
R1_CROSS = 'results/revision_R1/cross_dataset.csv'
PAIR = ('jackhhao', 'advbench')          # (train, test)
METHODS = ['BERT', 'ModernBERT', 'MiniLM+LR']

# MiniLM+LR is the (embedding_model, classifier_kind) entry in EMBEDDING_METHODS.
_EMB_LOOKUP = {name: (model, kind) for name, model, kind in EMBEDDING_METHODS}


def build_pair_cross_manifests(src, tgt, seeds):
    """Build ONLY the cross_dataset/{src}__to__{tgt}/seed_{n} manifests for the
    given seeds. Mirrors experiments.build_manifests.build_cross_dataset for a
    single pair so we never rebuild the other pairs or the existing seeds."""
    src_clean, _, _, _ = dedup_within(load_dataset(src))
    tgt_clean, _, _, _ = dedup_within(load_dataset(tgt))
    test = tgt_clean.copy()
    test_keys = set(test['key'])
    src_stripped, leakage_removed = _remove_leaked(src_clean, test_keys)
    src_classes = src_stripped['label'].nunique()
    out_root = SPLITS_DIR / 'cross_dataset'
    for seed in seeds:
        out_dir = out_root / f"{src}__to__{tgt}" / f"seed_{seed}"
        if (out_dir / 'manifest.json').exists():
            continue  # deterministic; never overwrite an existing manifest
        if src_classes < 2:
            train, val = src_stripped.iloc[:0].copy(), src_stripped.iloc[:0].copy()
            trainable = False
            reason = ("source has only one class after test-leakage removal; "
                      "stratified split is undefined")
        else:
            try:
                train, val = stratified_split(src_stripped, CROSS_VAL_FRAC, seed)
                trainable, reason = _clean_split_ok(train, val)
            except ValueError as e:
                train, val = src_stripped.iloc[:0].copy(), src_stripped.iloc[:0].copy()
                trainable = False
                reason = f"stratified split failed: {e}"
        manifest = {
            "protocol": "cross_dataset",
            "train_dataset": src,
            "test_dataset": tgt,
            "seed": seed,
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "leakage_removed_from_source": leakage_removed,
            "leakage_removed_from_train": 0,
            "leakage_removed_from_val": 0,
            "trainable": trainable,
            "untrainable_reason": reason,
        }
        save_split(out_dir, train, val, test, manifest)
        print(f"  built manifest {src}->{tgt} seed={seed} "
              f"sizes={manifest['sizes']} src_leak={leakage_removed}")


FAILURE_COLS = ['Method', 'Pair', 'Seed', 'Phase', 'Reason', 'Device']


def _failures_path(result_file):
    base, ext = os.path.splitext(result_file)
    return base + '_failures' + (ext or '.csv')


def _record_failure(failures, failures_file, name, pair, seed, phase, reason,
                    device):
    """Log a failed config to an in-process list AND a SEPARATE failures CSV.

    The failure is NOT written to the resume-keyed results file, so a later
    rerun on adequate hardware retries the config instead of skipping it."""
    rec = {'Method': name, 'Pair': pair, 'Seed': seed, 'Phase': phase,
           'Reason': reason, 'Device': device}
    failures.append(rec)
    os.makedirs(os.path.dirname(failures_file), exist_ok=True)
    df = pd.DataFrame([rec])[FAILURE_COLS]
    if not os.path.isfile(failures_file):
        df.to_csv(failures_file, index=False)
    else:
        df.to_csv(failures_file, mode='a', header=False, index=False)


def _hparams_for(method):
    if method in TRANSFORMER_METHODS:
        return TRANSFORMER_METHODS[method]
    model_name, kind = _EMB_LOOKUP[method]
    return _hparams_for_method(method, kind)


def pre_seed_from_r1(result_file, r1_cross=R1_CROSS):
    """Copy this pair's ten existing SEEDS_EXTENDED rows from R1 verbatim.

    Reviewer #4 asked for 30 seeds; ten of those (SEEDS_EXTENDED) were already
    run for jackhhao->advbench in results/revision_R1/cross_dataset.csv with
    identical hparams. Copying them into stability_30seed.csv means
    load_completed_set marks them done, so only the twenty NEW seeds are
    recomputed. Missing rows are a WARNING (they will simply be recomputed),
    not a hard failure."""
    src, tgt = PAIR
    ds_label = f"{src}->{tgt}"
    if not os.path.isfile(r1_cross):
        print(f"[pre-seed] {r1_cross} not found; nothing to reuse.")
        return
    df = pd.read_csv(r1_cross)
    sub = df[(df['Dataset'] == ds_label) & (df['Method'].isin(METHODS))]
    completed = load_completed_set(result_file)

    written, missing = 0, []
    for m in METHODS:
        for s in SEEDS_EXTENDED:
            hits = sub[(sub['Method'] == m) & (sub['Seed'] == s)]
            if len(hits) != 1:
                missing.append((m, int(s), len(hits)))
                continue
            row = hits.iloc[0]
            hp = row['HyperParams']
            hp = '' if pd.isna(hp) else hp
            key = (row['Method'], row['Dataset'], int(row['Seed']), hp)
            if key in completed:
                continue
            save_result(row.to_dict(), result_file)
            written += 1
    if missing:
        print(f"[pre-seed] WARNING: {len(missing)} expected R1 row(s) "
              f"missing/duplicated (will be recomputed): {missing}")
    print(f"[pre-seed] reused {written} R1 row(s) for {ds_label} "
          f"(the {len(SEEDS_EXTENDED)} SEEDS_EXTENDED seeds).")


def run_embedding_method(data, method, seed):
    """Run one embedding+classifier method on a cross bert bundle."""
    model_name, kind = _EMB_LOOKUP[method]
    train_labels = _labels_to_int(data['train_labels_str'])
    val_labels = _labels_to_int(data['val_labels_str'])
    test_labels = _labels_to_int(data['test_labels_str'])
    result = train_and_evaluate(
        train_texts=data['train_sentences'], train_labels=train_labels,
        val_texts=data['val_sentences'], val_labels=val_labels,
        test_texts=data['test_sentences'], test_labels=test_labels,
        model_name=model_name, classifier_kind=kind,
        classifier_hparams=EMBEDDING_HPARAMS[kind],
        protocol_tag=f"cross_dataset__{PAIR[0]}->{PAIR[1]}", seed=seed,
        max_seq_length=EMBEDDING_HPARAMS['max_seq_length'],
        batch_size=EMBEDDING_HPARAMS['batch_size'])
    return result['predictions'], np.asarray(test_labels), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS_30)
    parser.add_argument('--build_only', action='store_true',
                        help="Build (and skip-if-present) this pair's frozen "
                             "manifests for every requested seed, then exit "
                             "WITHOUT running. Use this to persist the new "
                             "splits to Drive BEFORE the long run consumes the "
                             "session, so a mid-run disconnect cannot lose them.")
    cli_args = parser.parse_args()

    src, tgt = PAIR
    ds_label = f"{src}->{tgt}"

    # 1. Ensure the frozen manifests for THIS pair exist for every requested
    #    seed. build_pair_cross_manifests skips any seed whose manifest already
    #    exists, so the pre-existing SEEDS_EXTENDED (and 42/123/456) manifests
    #    are never rebuilt; only the twenty new seeds (or any that happen to be
    #    missing on a fresh Colab session) are created.
    new_seeds = [s for s in cli_args.seeds if s not in SEEDS_EXTENDED]
    print(f"[manifests] ensuring {ds_label} manifests exist for "
          f"{len(cli_args.seeds)} seeds (new seeds: {new_seeds or 'none'})")
    build_pair_cross_manifests(src, tgt, cli_args.seeds)

    if cli_args.build_only:
        print(f"[build_only] {ds_label} manifests ensured for "
              f"{len(cli_args.seeds)} seeds; exiting before the run so the "
              f"notebook can sync the frozen splits to Drive first.")
        return

    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    # 2. Reuse the ten existing SEEDS_EXTENDED rows from R1 so only the twenty
    #    new seeds are computed here.
    pre_seed_from_r1(cli_args.result_file)
    completed = load_completed_set(cli_args.result_file)
    device = get_device()
    dev_name = get_device_name(device)

    transformer_methods = [m for m in cli_args.methods if m in TRANSFORMER_METHODS]
    embedding_methods = [m for m in cli_args.methods if m in _EMB_LOOKUP]

    run_count = 0
    t_start = time.time()
    failures_file = _failures_path(cli_args.result_file)
    failures = []
    # Methods that hit CUDA OOM are disabled for all remaining seeds (a failure
    # is logged once to the separate failures file) rather than retried 30x.
    oom_methods = set()

    for seed in cli_args.seeds:
        pending_tf = [m for m in transformer_methods
                      if m not in oom_methods
                      and make_key(m, ds_label, seed,
                                   hparams=TRANSFORMER_METHODS[m]) not in completed]
        pending_emb = [m for m in embedding_methods
                       if m not in oom_methods
                       and make_key(m, ds_label, seed,
                                    hparams=_hparams_for(m)) not in completed]
        if not (pending_tf or pending_emb):
            continue

        manifest = peek_manifest('cross_dataset', f"{src}__to__{tgt}", seed)
        if not manifest.get('trainable', True):
            reason = manifest.get('untrainable_reason') or 'unknown'
            print(f"[UNTRAINABLE] {ds_label} seed={seed}: {reason}")
            sizes = manifest.get('sizes', {})
            leak = manifest.get('leakage_removed_from_source', 0)
            for name in pending_tf:
                save_result(build_untrainable_row(
                    name, ds_label, seed, 'cross', REMOVE_LIMIT,
                    sizes.get('train', 0), sizes.get('val', 0),
                    sizes.get('test', 0), leak, reason,
                    hparams=TRANSFORMER_METHODS[name], device=dev_name),
                    cli_args.result_file)
            for name in pending_emb:
                save_result(build_untrainable_row(
                    name, ds_label, seed, 'cross', REMOVE_LIMIT,
                    sizes.get('train', 0), sizes.get('val', 0),
                    sizes.get('test', 0), leak, reason,
                    hparams=_hparams_for(name), device=dev_name),
                    cli_args.result_file)
            continue

        if pending_tf:
            data_bert = load_bert_split_cross(src, tgt, seed)
            for name in pending_tf:
                hp = TRANSFORMER_METHODS[name]
                print(f"\n[RUN] {name} {ds_label} seed={seed}")
                wall_start = time.time()
                try:
                    y_pred, y_true, rep_t, tt, it, np_, dev = run_bert_cross(
                        data_bert, hp, seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [OOM] {name} {ds_label} seed={seed}: aborting method")
                    _record_failure(
                        failures, failures_file, name, ds_label, seed, 'oom',
                        "CUDA OOM during transformer fine-tuning (batch 16) / "
                        "inference; method aborted for remaining seeds", dev_name)
                    oom_methods.add(name)
                    continue
                except Exception as e:
                    print(f"  ERROR: {name} {ds_label} seed={seed}: {e}")
                    _record_failure(failures, failures_file, name, ds_label,
                                    seed, 'error', str(e), dev_name)
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
                    preprocess_time=0.0, representation_time=rep_t,
                    train_time=tt, infer_time=it, total_time=wall_time,
                    num_params=np_, hparams=hp, device=dev)
                save_result(row, cli_args.result_file)
                run_count += 1
                print(f"  F1B={metrics['F1Binary']:.4f}")

        if pending_emb:
            data_emb = load_bert_split_cross(src, tgt, seed)
            for name in pending_emb:
                hp = _hparams_for(name)
                print(f"\n[RUN] {name} {ds_label} seed={seed}")
                wall_start = time.time()
                try:
                    y_pred, y_true, result = run_embedding_method(
                        data_emb, name, seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [OOM] {name} {ds_label} seed={seed}: aborting method")
                    _record_failure(
                        failures, failures_file, name, ds_label, seed, 'oom',
                        "CUDA OOM during embedding encoding (batch 32) / head "
                        "fit; method aborted for remaining seeds", dev_name)
                    oom_methods.add(name)
                    continue
                except Exception as e:
                    print(f"  ERROR: {name} {ds_label} seed={seed}: {e}")
                    _record_failure(failures, failures_file, name, ds_label,
                                    seed, 'error', str(e), dev_name)
                    continue
                wall_time = time.time() - wall_start
                metrics = compute_metrics(y_true, y_pred)
                row = build_result_row(
                    method=name, dataset=ds_label, seed=seed,
                    metrics_dict=metrics, split_ratio='cross',
                    remove_limit=REMOVE_LIMIT,
                    train_size=data_emb['n_train'], val_size=data_emb['n_val'],
                    test_size=data_emb['n_test'],
                    leakage_removed=data_emb['leakage_removed'],
                    preprocess_time=0.0,
                    representation_time=result['representation_time'],
                    train_time=result['train_time'],
                    infer_time=result['infer_time'], total_time=wall_time,
                    num_params=result['num_params'], hparams=hp, device=dev_name)
                save_result(row, cli_args.result_file)
                run_count += 1
                print(f"  F1B={metrics['F1Binary']:.4f}")

    print(f"\nDone. {run_count} runs in {int((time.time() - t_start) // 60)}m. "
          f"-> {cli_args.result_file}")

    if failures:
        print("\n" + "=" * 72)
        print(f"[FAILURES] {len(failures)} config(s) did NOT complete:")
        for r in failures:
            print(f"  - {r['Method']} | {r['Pair']} | seed={r['Seed']} "
                  f"| {r['Phase']}: {r['Reason']}")
        print(f"Logged to {failures_file} (NOT the results file), so a rerun on "
              f"adequate hardware retries them. Exiting non-zero.")
        print("=" * 72)
        sys.exit(1)


if __name__ == '__main__':
    main()
