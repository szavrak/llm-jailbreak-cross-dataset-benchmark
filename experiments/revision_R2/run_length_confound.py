"""R4-P3 — DAN 60-character length confound.

Reviewer #4 noted that DAN jailbreak prompts are far longer than its benign
prompts, so a detector might be exploiting raw length rather than content. This
experiment truncates EVERY DAN row (both classes) to the first 60 Unicode code
points and re-runs the full detector roster, comparing against the uncapped
("orig") numbers.

Key properties
--------------
* Truncation is a LOAD-TIME transform applied AFTER manifest split membership and
  leakage removal (via the optional max_chars/truncate_datasets kwargs on the
  shared loaders). The contamination audit therefore still ran on FULL text.
* Only DAN is truncated. In the six DAN-facing cross pairs, DAN is truncated
  whether it is the train side or the test side; the partner dataset is left
  intact.
* jackhhao<->dan pairs are excluded (they are in CONTAMINATED_PAIRS).
* The MaxChars condition (60 vs "orig") is recorded inside HyperParams (so
  resume keys stay distinct for every method, including TF-IDF which otherwise
  carries no hparams) and echoed into the Notes column for easy grepping. The
  Dataset column keeps the plain tag ("dan", "dan->advbench", ...). The uncapped
  ("orig") rows for the 8 detectors are NOT recomputed here; they are imported
  verbatim from results/revision_R1 by import_orig_rows.py. Only the capped
  detector rows and the LengthOnly baseline (at both orig and 60) are new
  compute in this script.
* Capped-text embeddings use a distinct cache namespace ("lenconf_mc60__...")
  so they never overwrite the original plm_cache entries.
* A length-only baseline (LogisticRegression on the single character-count
  feature) is run on the same conditions to show directly how predictive length
  alone is.
* A collision audit counts rows whose first-60-char prefix becomes identical
  after truncation, on the actual (asymmetrically capped) model inputs: within
  train, within val, train-vs-test, and val-vs-test. Policy: keep them.

Outputs (results/revision_R2/):
    length_confound.csv             — the RESULT_COLUMNS result rows
    length_confound_collisions.csv  — the collision audit
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (SEEDS_EXTENDED, REMOVE_LIMIT,
                                 TRANSFORMER_METHODS, EMBEDDING_METHODS,
                                 EMBEDDING_HPARAMS)
from experiments.shared_preprocessing import (load_split, load_split_cross,
                                               load_bert_split,
                                               load_bert_split_cross,
                                               get_tfidf_features)
from experiments.baselines.embedding_classifier import train_and_evaluate
from experiments.baselines.tfidf_classifiers import run_lr, run_svm
from experiments.metrics import (compute_metrics, build_result_row,
                                  build_untrainable_row, save_result)
from experiments.resume_utils import load_completed_set, make_key
from experiments.runners.run_cross_dataset import (run_bert_cross, get_device,
                                                    get_device_name)
from experiments.runners.run_embedding import _hparams_for_method, _labels_to_int


RESULT_FILE = 'results/revision_R2/length_confound.csv'
COLLISION_FILE = 'results/revision_R2/length_confound_collisions.csv'
CAP = 60
TRUNCATE = {'dan'}

# DAN in-distribution + the six valid DAN-facing cross pairs (jackhhao<->dan
# excluded — CONTAMINATED_PAIRS).
CONDITIONS = [
    'dan',
    'dan->jailbreakbench', 'dan->advbench', 'dan->harmbench',
    'jailbreakbench->dan', 'advbench->dan', 'harmbench->dan',
]

TFIDF_METHODS = ['TF-IDF+LR', 'TF-IDF+SVM']
EMB_LOOKUP = {name: (model, kind) for name, model, kind in EMBEDDING_METHODS}
EMB_METHODS = ['MiniLM+LR', 'MiniLM+XGBoost', 'Arctic+RF', 'BERT-frozen+LR']
BERT_METHODS = ['BERT', 'ModernBERT']
LENGTH_METHOD = 'LengthOnly+LR'

DEFAULT_METHODS = TFIDF_METHODS + EMB_METHODS + BERT_METHODS + [LENGTH_METHOD]


def _is_cross(tag):
    return '->' in tag


def _mc_label(mc):
    return 'orig' if mc is None else str(mc)


def _parse_mc(token):
    return None if token == 'orig' else int(token)


def _hparams_for_row(method, mc):
    """HyperParams payload; embeds max_chars so orig/capped keys never collide."""
    label = _mc_label(mc)
    if method in TRANSFORMER_METHODS:
        hp = dict(TRANSFORMER_METHODS[method])
    elif method in EMB_LOOKUP:
        _, kind = EMB_LOOKUP[method]
        hp = dict(_hparams_for_method(method, kind))
    elif method in TFIDF_METHODS:
        hp = {}
    elif method == LENGTH_METHOD:
        hp = {'classifier': 'logreg', 'feature': 'char_length'}
    else:
        raise ValueError(f"Unknown method: {method}")
    hp['max_chars'] = label
    return hp


def _load_bundle(tag, seed, mc):
    if _is_cross(tag):
        s, t = tag.split('->')
        return load_bert_split_cross(s, t, seed, max_chars=mc,
                                     truncate_datasets=TRUNCATE)
    return load_bert_split(tag, seed, max_chars=mc, truncate_datasets=TRUNCATE)


def _load_tfidf_data(tag, seed, mc):
    if _is_cross(tag):
        s, t = tag.split('->')
        return load_split_cross(s, t, seed, remove_limit=REMOVE_LIMIT,
                                max_chars=mc, truncate_datasets=TRUNCATE)
    return load_split(tag, seed, remove_limit=REMOVE_LIMIT,
                      max_chars=mc, truncate_datasets=TRUNCATE)


# ── collision audit ────────────────────────────────────────────────
def _load_collision_done(path):
    done = set()
    if not os.path.isfile(path):
        return done
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception:
        return done
    for _, r in df.iterrows():
        done.add((r['Condition'], int(r['Seed']), str(r['MaxChars'])))
    return done


_COLLISION_COLS = ['Condition', 'Seed', 'MaxChars', 'TrainRows', 'ValRows',
                   'TestRows', 'TrainPrefixDup', 'ValPrefixDup',
                   'TrainVsTestPrefixDup', 'ValVsTestPrefixDup', 'Policy']


def _save_collision_row(path, row):
    import pandas as pd
    df = pd.DataFrame([row])[_COLLISION_COLS]
    if not os.path.isfile(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def collision_audit(conditions, seeds, path):
    """Count first-CAP-char prefix collisions after truncation.

    Prefixes are the ACTUAL (capped) model inputs — bundle['*_sentences'] already
    carry the correct asymmetric truncation (only the DAN side is truncated), so
    we must NOT re-slice [:CAP], which would falsely truncate the untouched
    partner dataset and report spurious collisions. Both the train and val DAN
    rows are truncated and fed to early stopping, so val-side coverage is
    audited too. Policy: keep the collisions."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _load_collision_done(path)
    label = str(CAP)
    print(f"\n[collision audit] first-{CAP}-char prefix duplicates (policy: keep)")
    for tag in conditions:
        for seed in seeds:
            if (tag, seed, label) in done:
                continue
            bundle = _load_bundle(tag, seed, CAP)
            # Use the actual (asymmetrically truncated) model inputs directly.
            train = list(bundle['train_sentences'])
            val = list(bundle['val_sentences'])
            test = list(bundle['test_sentences'])
            test_set = set(test)
            train_dup = len(train) - len(set(train))
            val_dup = len(val) - len(set(val))
            train_vs_test = sum(1 for t in train if t in test_set)
            val_vs_test = sum(1 for t in val if t in test_set)
            row = {
                'Condition': tag, 'Seed': seed, 'MaxChars': label,
                'TrainRows': len(train), 'ValRows': len(val),
                'TestRows': len(test),
                'TrainPrefixDup': train_dup, 'ValPrefixDup': val_dup,
                'TrainVsTestPrefixDup': train_vs_test,
                'ValVsTestPrefixDup': val_vs_test, 'Policy': 'keep',
            }
            _save_collision_row(path, row)
            print(f"  {tag:24s} seed={seed:<4} "
                  f"train_dup={train_dup} val_dup={val_dup} "
                  f"train_vs_test={train_vs_test} val_vs_test={val_vs_test} (kept)")


# ── length-only baseline ───────────────────────────────────────────
def run_length_only(bundle, seed):
    train_labels = _labels_to_int(bundle['train_labels_str'])
    test_labels = _labels_to_int(bundle['test_labels_str'])
    X_train = np.array([[len(t)] for t in bundle['train_sentences']], dtype=float)
    X_test = np.array([[len(t)] for t in bundle['test_sentences']], dtype=float)
    clf = LogisticRegression(random_state=seed, max_iter=1000,
                             class_weight='balanced')
    t0 = time.time()
    clf.fit(X_train, np.asarray(train_labels))
    train_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = time.time() - t0
    n_params = int(clf.coef_.size + clf.intercept_.size)
    return (np.asarray(y_pred, dtype=int), np.asarray(test_labels),
            train_time, infer_time, n_params)


def run_embedding_method(bundle, method, tag, mc, seed):
    model_name, kind = EMB_LOOKUP[method]
    train_labels = _labels_to_int(bundle['train_labels_str'])
    val_labels = _labels_to_int(bundle['val_labels_str'])
    test_labels = _labels_to_int(bundle['test_labels_str'])
    result = train_and_evaluate(
        train_texts=bundle['train_sentences'], train_labels=train_labels,
        val_texts=bundle['val_sentences'], val_labels=val_labels,
        test_texts=bundle['test_sentences'], test_labels=test_labels,
        model_name=model_name, classifier_kind=kind,
        classifier_hparams=EMBEDDING_HPARAMS[kind],
        protocol_tag=f"lenconf_mc{_mc_label(mc)}__{tag}", seed=seed,
        max_seq_length=EMBEDDING_HPARAMS['max_seq_length'],
        batch_size=EMBEDDING_HPARAMS['batch_size'])
    return result['predictions'], np.asarray(test_labels), result


def _emit_untrainable(methods, tag, seed, mc, bundle, dev_name, result_file):
    reason = bundle.get('untrainable_reason') or 'unknown'
    print(f"  [UNTRAINABLE] {tag} seed={seed} mc={_mc_label(mc)}: {reason}")
    for name in methods:
        dev = 'cpu' if name in TFIDF_METHODS or name == LENGTH_METHOD else dev_name
        save_result(build_untrainable_row(
            name, tag, seed, 'lenconf', REMOVE_LIMIT,
            bundle['n_train'], bundle['n_val'], bundle['n_test'],
            bundle['leakage_removed'], reason,
            hparams=_hparams_for_row(name, mc), device=dev), result_file)


FAILURE_COLS = ['Method', 'Condition', 'Seed', 'MaxChars', 'Phase', 'Reason',
                'Device']


def _failures_path(result_file):
    base, ext = os.path.splitext(result_file)
    return base + '_failures' + (ext or '.csv')


def _record_failure(failures, failures_file, name, tag, seed, mc, phase, reason,
                    device):
    """Log a failed config to an in-process list AND a SEPARATE failures CSV.

    Crucially the failure is NOT written to the resume-keyed results file, so a
    later rerun on adequate hardware naturally retries the config instead of
    permanently skipping it."""
    import pandas as pd
    rec = {'Method': name, 'Condition': tag, 'Seed': seed,
           'MaxChars': _mc_label(mc), 'Phase': phase, 'Reason': reason,
           'Device': device}
    failures.append(rec)
    os.makedirs(os.path.dirname(failures_file), exist_ok=True)
    df = pd.DataFrame([rec])[FAILURE_COLS]
    if not os.path.isfile(failures_file):
        df.to_csv(failures_file, index=False)
    else:
        df.to_csv(failures_file, mode='a', header=False, index=False)


def _emit_oom(failures, failures_file, name, tag, seed, mc, reason, dev_name):
    """Log a CUDA-OOM failure (separate failures file, not the results file) and
    abort the method for the remaining configs."""
    print(f"  [OOM] {name} {tag} seed={seed} mc={_mc_label(mc)}: {reason}")
    _record_failure(failures, failures_file, name, tag, seed, mc, 'oom', reason,
                    dev_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--collision_file', type=str, default=COLLISION_FILE)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS_EXTENDED)
    parser.add_argument('--max_chars', nargs='+', default=[str(CAP)],
                        help="Length settings to run: '60' (default) and/or "
                             "'orig'. The 8 detectors' 'orig' rows are imported "
                             "from results/revision_R1 via import_orig_rows.py, "
                             "so run '60' only for them; pass 'orig 60' only for "
                             "the LengthOnly baseline (not present in R1).")
    parser.add_argument('--skip_audit', action='store_true')
    cli_args = parser.parse_args()

    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    failures_file = _failures_path(cli_args.result_file)
    failures = []
    mc_list = [_parse_mc(x) for x in cli_args.max_chars]

    # Collision audit runs once for the capped setting if it is requested.
    if not cli_args.skip_audit and CAP in mc_list:
        collision_audit(cli_args.conditions, cli_args.seeds,
                        cli_args.collision_file)

    completed = load_completed_set(cli_args.result_file)
    device = get_device()
    dev_name = get_device_name(device)

    methods = [m for m in cli_args.methods if m in DEFAULT_METHODS]
    tfidf_pending_all = [m for m in methods if m in TFIDF_METHODS]
    emb_pending_all = [m for m in methods if m in EMB_METHODS]
    bert_pending_all = [m for m in methods if m in BERT_METHODS]
    length_pending_all = [m for m in methods if m == LENGTH_METHOD]

    run_count = 0
    t_start = time.time()
    # Methods that hit CUDA OOM are disabled for all remaining configs rather
    # than retried (and repeated) on every (tag, seed, mc). A failed row is
    # recorded once, at the config where the OOM occurred.
    oom_methods = set()

    for tag in cli_args.conditions:
        for seed in cli_args.seeds:
            for mc in mc_list:
                def _pending(group):
                    return [m for m in group
                            if m not in oom_methods
                            and make_key(m, tag, seed,
                                         hparams=_hparams_for_row(m, mc))
                            not in completed]
                p_tfidf = _pending(tfidf_pending_all)
                p_emb = _pending(emb_pending_all)
                p_bert = _pending(bert_pending_all)
                p_len = _pending(length_pending_all)
                if not (p_tfidf or p_emb or p_bert or p_len):
                    continue

                label = _mc_label(mc)
                # Text bundle covers embed / bert / length methods.
                need_bundle = p_emb or p_bert or p_len
                bundle = _load_bundle(tag, seed, mc) if need_bundle else None

                if bundle is not None and not bundle['trainable']:
                    _emit_untrainable(p_tfidf + p_emb + p_bert + p_len, tag,
                                      seed, mc, bundle, dev_name,
                                      cli_args.result_file)
                    continue

                # ── TF-IDF methods ──
                if p_tfidf:
                    # Wall clock starts before load + feature construction so
                    # TotalRunTimeSec covers the whole cost (matches
                    # run_cross_dataset's shared_load accounting).
                    load_start = time.time()
                    tdata = _load_tfidf_data(tag, seed, mc)
                    if not tdata['trainable']:
                        _emit_untrainable(p_tfidf, tag, seed, mc, tdata,
                                          dev_name, cli_args.result_file)
                    else:
                        (X_tr, _Xv, X_te, y_tr, _yv, y_te,
                         rep_time) = get_tfidf_features(tdata)
                        shared_load = time.time() - load_start
                        for name in p_tfidf:
                            run_fn = run_lr if name == 'TF-IDF+LR' else run_svm
                            print(f"[RUN] {name} {tag} seed={seed} mc={label}")
                            m_start = time.time()
                            y_pred, tt, it, np_ = run_fn(X_tr, X_te, y_tr, y_te, seed)
                            wall_time = shared_load + (time.time() - m_start)
                            metrics = compute_metrics(y_te, y_pred)
                            row = build_result_row(
                                method=name, dataset=tag, seed=seed,
                                metrics_dict=metrics, split_ratio='lenconf',
                                remove_limit=REMOVE_LIMIT,
                                train_size=tdata['n_train'], val_size=tdata['n_val'],
                                test_size=tdata['n_test'],
                                leakage_removed=tdata['leakage_removed'],
                                preprocess_time=tdata['preprocess_time'],
                                representation_time=rep_time,
                                train_time=tt, infer_time=it,
                                total_time=wall_time,
                                num_params=np_,
                                hparams=_hparams_for_row(name, mc),
                                device='cpu', notes=f"maxchars={label}")
                            save_result(row, cli_args.result_file)
                            run_count += 1
                            print(f"  F1B={metrics['F1Binary']:.4f}")

                # ── length-only baseline ──
                for name in p_len:
                    print(f"[RUN] {name} {tag} seed={seed} mc={label}")
                    wall_start = time.time()
                    y_pred, y_true, tt, it, np_ = run_length_only(bundle, seed)
                    metrics = compute_metrics(y_true, y_pred)
                    row = build_result_row(
                        method=name, dataset=tag, seed=seed,
                        metrics_dict=metrics, split_ratio='lenconf',
                        remove_limit=REMOVE_LIMIT,
                        train_size=bundle['n_train'], val_size=bundle['n_val'],
                        test_size=bundle['n_test'],
                        leakage_removed=bundle['leakage_removed'],
                        preprocess_time=0.0, representation_time=0.0,
                        train_time=tt, infer_time=it,
                        total_time=time.time() - wall_start,
                        num_params=np_, hparams=_hparams_for_row(name, mc),
                        device='cpu', notes=f"maxchars={label}")
                    save_result(row, cli_args.result_file)
                    run_count += 1
                    print(f"  F1B={metrics['F1Binary']:.4f}")

                # ── embedding methods ──
                for name in p_emb:
                    print(f"[RUN] {name} {tag} seed={seed} mc={label}")
                    wall_start = time.time()
                    try:
                        y_pred, y_true, result = run_embedding_method(
                            bundle, name, tag, mc, seed)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        _emit_oom(failures, failures_file, name, tag, seed, mc,
                                  "CUDA OOM during embedding encoding (batch 32) "
                                  "/ head fit; method aborted for remaining "
                                  "configs", dev_name)
                        oom_methods.add(name)
                        continue
                    except Exception as e:
                        print(f"  ERROR: {name} {tag} seed={seed} "
                              f"mc={label}: {e}")
                        _record_failure(failures, failures_file, name, tag, seed,
                                        mc, 'error', str(e), dev_name)
                        continue
                    metrics = compute_metrics(y_true, y_pred)
                    row = build_result_row(
                        method=name, dataset=tag, seed=seed,
                        metrics_dict=metrics, split_ratio='lenconf',
                        remove_limit=REMOVE_LIMIT,
                        train_size=bundle['n_train'], val_size=bundle['n_val'],
                        test_size=bundle['n_test'],
                        leakage_removed=bundle['leakage_removed'],
                        preprocess_time=0.0,
                        representation_time=result['representation_time'],
                        train_time=result['train_time'],
                        infer_time=result['infer_time'],
                        total_time=time.time() - wall_start,
                        num_params=result['num_params'],
                        hparams=_hparams_for_row(name, mc),
                        device=dev_name, notes=f"maxchars={label}")
                    save_result(row, cli_args.result_file)
                    run_count += 1
                    print(f"  F1B={metrics['F1Binary']:.4f}")

                # ── fine-tuned transformers ──
                for name in p_bert:
                    hp = TRANSFORMER_METHODS[name]
                    print(f"[RUN] {name} {tag} seed={seed} mc={label}")
                    wall_start = time.time()
                    try:
                        y_pred, y_true, rep_t, tt, it, np_, dev = run_bert_cross(
                            bundle, hp, seed)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        _emit_oom(failures, failures_file, name, tag, seed, mc,
                                  "CUDA OOM during transformer fine-tuning "
                                  "(batch 16) / inference; method aborted for "
                                  "remaining configs", dev_name)
                        oom_methods.add(name)
                        continue
                    except Exception as e:
                        print(f"  ERROR: {name} {tag} seed={seed} "
                              f"mc={label}: {e}")
                        _record_failure(failures, failures_file, name, tag, seed,
                                        mc, 'error', str(e), dev_name)
                        continue
                    metrics = compute_metrics(y_true, y_pred)
                    row = build_result_row(
                        method=name, dataset=tag, seed=seed,
                        metrics_dict=metrics, split_ratio='lenconf',
                        remove_limit=REMOVE_LIMIT,
                        train_size=bundle['n_train'], val_size=bundle['n_val'],
                        test_size=bundle['n_test'],
                        leakage_removed=bundle['leakage_removed'],
                        preprocess_time=0.0, representation_time=rep_t,
                        train_time=tt, infer_time=it,
                        total_time=time.time() - wall_start,
                        num_params=np_,
                        hparams=_hparams_for_row(name, mc),
                        device=dev, notes=f"maxchars={label}")
                    save_result(row, cli_args.result_file)
                    run_count += 1
                    print(f"  F1B={metrics['F1Binary']:.4f}")

    print(f"\nDone. {run_count} runs in {int((time.time() - t_start) // 60)}m. "
          f"-> {cli_args.result_file}")

    if failures:
        print("\n" + "=" * 72)
        print(f"[FAILURES] {len(failures)} config(s) did NOT complete:")
        for r in failures:
            print(f"  - {r['Method']} | {r['Condition']} | seed={r['Seed']} "
                  f"| mc={r['MaxChars']} | {r['Phase']}: {r['Reason']}")
        print(f"Logged to {failures_file} (NOT the results file), so a rerun on "
              f"adequate hardware retries them. Exiting non-zero.")
        print("=" * 72)
        sys.exit(1)


if __name__ == '__main__':
    main()
