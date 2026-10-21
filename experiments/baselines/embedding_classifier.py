"""Pretrained-embedding + simple-classifier baselines.

Supports three methods:
  - MiniLM+LR       : all-MiniLM-L6-v2 + LogisticRegression (diagnostic linear)
  - MiniLM+XGBoost  : all-MiniLM-L6-v2 + XGBoost  (Galinkin & Sablotny 2024 style)
  - Arctic+RF       : Snowflake Arctic Embed L + Random Forest (NemoGuard 2024)

Embeddings are cached under plm_cache/ keyed by (model_slug, protocol_tag,
split_name, seed). A text fingerprint is also stored so stale caches are
auto-invalidated when the underlying split content changes.
"""
import hashlib
import os
import time

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'plm_cache')


def _slug(model_name):
    return model_name.replace('/', '__').replace('-', '_')


def _fingerprint(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode('utf-8', errors='ignore'))
        h.update(b'\x00')
    h.update(str(len(texts)).encode())
    return h.hexdigest()[:16]


def _cache_path(model_name, protocol_tag, split_name, seed):
    fname = f"{_slug(model_name)}__{protocol_tag}__{split_name}__seed{seed}.npz"
    return os.path.join(CACHE_DIR, fname)


def encode_texts(texts, model_name, protocol_tag, split_name, seed,
                 max_seq_length=512, batch_size=32, device=None):
    """Return (embeddings, elapsed_seconds, from_cache: bool).

    Embeddings are L2-normalized. Cache is invalidated if the text fingerprint
    doesn't match.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = _cache_path(model_name, protocol_tag, split_name, seed)
    fp = _fingerprint(texts)

    if os.path.exists(cache_file):
        try:
            npz = np.load(cache_file, allow_pickle=True)
            if str(npz['fingerprint']) == fp:
                return npz['embeddings'], 0.0, True
        except Exception:
            pass  # corrupt cache, re-encode

    from sentence_transformers import SentenceTransformer
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps')
                     and torch.backends.mps.is_available() else 'cpu')

    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.time() - t0

    np.savez_compressed(cache_file, embeddings=embeddings, fingerprint=fp)
    return embeddings, elapsed, False


def _build_classifier(kind, seed, hparams):
    if kind in ('logreg', 'logreg_noweight'):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, **hparams)
    if kind in ('xgboost', 'xgboost_balanced'):
        from xgboost import XGBClassifier
        # 'auto' is a sentinel; scale_pos_weight is resolved per-fit from labels.
        hp = {k: v for k, v in hparams.items() if v != 'auto'}
        return XGBClassifier(random_state=seed, eval_metric='logloss',
                             tree_method='hist', n_jobs=-1, **hp)
    if kind == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, **hparams)
    raise ValueError(f"Unknown classifier kind: {kind}")


def _count_params(clf, kind, n_features):
    if kind in ('logreg', 'logreg_noweight'):
        return int(clf.coef_.size + clf.intercept_.size)
    if kind == 'xgboost':
        # n_estimators * (2^max_depth - 1) is an upper bound; report booster size
        booster = clf.get_booster()
        n_nodes = sum(len(t.split('\n')) for t in booster.get_dump())
        return int(n_nodes * 2)  # ~2 floats per node (threshold + value)
    if kind == 'randomforest':
        total = 0
        for est in clf.estimators_:
            total += est.tree_.node_count * 2  # threshold + value per node
        return int(total)
    return -1


def train_and_evaluate(train_texts, train_labels, val_texts, val_labels,
                       test_texts, test_labels, *, model_name, classifier_kind,
                       classifier_hparams, protocol_tag, seed,
                       max_seq_length=512, batch_size=32):
    """End-to-end: encode → fit → predict. Returns dict with metrics inputs.

    Val split is not encoded; these classifiers have no early stopping.
    """
    train_emb, t_train_enc, _ = encode_texts(
        train_texts, model_name, protocol_tag, 'train', seed,
        max_seq_length=max_seq_length, batch_size=batch_size)
    test_emb, t_test_enc, _ = encode_texts(
        test_texts, model_name, protocol_tag, 'test', seed,
        max_seq_length=max_seq_length, batch_size=batch_size)

    representation_time = t_train_enc + t_test_enc

    clf = _build_classifier(classifier_kind, seed, classifier_hparams)

    # Resolve per-fit class balancing for xgboost_balanced.
    if classifier_kind == 'xgboost_balanced' and \
            classifier_hparams.get('scale_pos_weight') == 'auto':
        y_train_arr = np.asarray(train_labels)
        n_pos = int((y_train_arr == 1).sum())
        n_neg = int((y_train_arr == 0).sum())
        # Guard against degenerate splits; XGBoost requires a positive float.
        ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
        clf.set_params(scale_pos_weight=ratio)

    t0 = time.time()
    clf.fit(train_emb, np.asarray(train_labels))
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(test_emb)
    infer_time = time.time() - t0

    return {
        'predictions': np.asarray(y_pred, dtype=int),
        'representation_time': representation_time,
        'train_time': train_time,
        'infer_time': infer_time,
        'num_params': _count_params(clf, classifier_kind, train_emb.shape[1]),
        'embedding_dim': int(train_emb.shape[1]),
    }
