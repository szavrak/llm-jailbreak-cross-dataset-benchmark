"""Manifest-based data loading and feature construction.

v2 (post-contamination audit): every method reads frozen splits from
splits/{protocol}/.../seed_{n}/{train,val,test}.csv. Preprocessing builds
vocabulary from the training portion only; no stopword removal.
"""
import sys
import os
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from build_dataset import (
    load_manifest_in_distribution,
    load_manifest_cross,
    load_manifest_mixed,
)
from preprocess import encode_labels, preprocess_data


def make_args(dataset, seed, remove_limit=2, split_ratio="70:10:20",
              train_size=1, test_size=1, use_gpu=0, shuffle_seed=None,
              hidden_dim=200, dropout=0.5, learning_rate=0.02,
              weight_decay=0, early_stopping=10, epochs=200,
              multiple_times=0, easy_copy=1, result_file="results.csv"):
    return argparse.Namespace(
        dataset=dataset,
        random_state=seed,
        split_ratio=split_ratio,
        train_size=train_size,
        test_size=test_size,
        remove_limit=remove_limit,
        use_gpu=use_gpu,
        shuffle_seed=shuffle_seed,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping=early_stopping,
        epochs=epochs,
        multiple_times=multiple_times,
        easy_copy=easy_copy,
        result_file=result_file,
    )


# ── Revision R2 (R4-P3) length-confound support ────────────────────
# Optional LOAD-TIME truncation applied AFTER manifest split membership and
# leakage removal (the contamination audit therefore still runs on FULL text).
# Byte-identical to the original loaders whenever max_chars is None.

def _truncate_bundle_sides(bundle, sides, max_chars):
    """Truncate a manifest bundle's sentences to the first `max_chars` Unicode
    code points, in place, for the named sides ('train'/'val'/'test')."""
    keymap = {'train': 'train_sentences', 'val': 'val_sentences',
              'test': 'test_sentences'}
    for side in sides:
        k = keymap[side]
        bundle[k] = [t[:max_chars] for t in bundle[k]]
    return bundle


def _cross_truncate_sides(src, tgt, truncate_datasets):
    """Sides to truncate for a cross-dataset pair: the source feeds train+val,
    the target feeds test."""
    sides = []
    if src in truncate_datasets:
        sides += ['train', 'val']
    if tgt in truncate_datasets:
        sides += ['test']
    return sides


def _assemble(bundle, dataset_tag, seed, remove_limit, split_ratio):
    """Shared post-processing with the clean inductive protocol.

    - Graph is built over TRAIN documents only (train_size == n_train).
    - Validation and test documents are evaluated inductively via
      get_test_emb / compute_inductive_logits — they are never graph nodes.
    - Vocabulary and word_doc_freq are therefore derived from train only.
    """
    train_sentences = bundle['train_sentences']
    val_sentences = bundle['val_sentences']
    test_sentences = bundle['test_sentences']
    train_labels_str = bundle['train_labels']
    val_labels_str = bundle['val_labels']
    test_labels_str = bundle['test_labels']

    n_train = len(train_sentences)
    n_val = len(val_sentences)
    n_test = len(test_sentences)
    train_size = n_train  # clean inductive: graph = train only

    args = make_args(dataset_tag, seed, remove_limit=remove_limit,
                     split_ratio=split_ratio)

    preprocess_start = time.time()
    # Vocabulary built from train only; val+test tokens reuse train vocab
    # (OOV is dropped). tokenize_sentences is ordered: train..., val..., test...
    non_train_sentences = val_sentences + test_sentences
    tokenize_sentences, word_list = preprocess_data(
        train_sentences, non_train_sentences, args)
    preprocess_time = time.time() - preprocess_start

    all_str_labels = train_labels_str + val_labels_str + test_labels_str
    labels, num_class = encode_labels(all_str_labels, [], args)

    vocab_length = len(word_list)
    word_id_map = {w: i for i, w in enumerate(word_list)}

    # Convenient tokenized slices for inductive evaluation paths.
    tokenized_train = tokenize_sentences[:n_train]
    tokenized_val = tokenize_sentences[n_train:n_train + n_val]
    tokenized_test = tokenize_sentences[n_train + n_val:]

    return {
        'train_sentences': train_sentences,
        'val_sentences': val_sentences,
        'test_sentences': test_sentences,
        'train_labels_str': train_labels_str,
        'val_labels_str': val_labels_str,
        'test_labels_str': test_labels_str,
        'labels': labels,
        'num_class': num_class,
        # tokenize_sentences keeps the legacy (train+val+test) concatenation
        # so existing code paths keep working; new code should prefer the
        # explicit tokenized_train / tokenized_val / tokenized_test below.
        'tokenize_sentences': tokenize_sentences,
        'tokenized_train': tokenized_train,
        'tokenized_val': tokenized_val,
        'tokenized_test': tokenized_test,
        'word_list': word_list,
        'word_id_map': word_id_map,
        'vocab_length': vocab_length,
        'train_size': train_size,   # == n_train (graph is train-only)
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'idx_train': np.arange(0, n_train),
        'manifest': bundle['manifest'],
        'leakage_removed': bundle['leakage_removed'],
        'trainable': bundle.get('trainable', True),
        'untrainable_reason': bundle.get('untrainable_reason'),
        'preprocess_time': preprocess_time,
        'args': args,
    }


def load_split(dataset, seed, remove_limit=2, split_ratio="70:10:20",
               max_chars=None, truncate_datasets=None):
    """In-distribution loader (reads splits/in_distribution/...).

    max_chars / truncate_datasets (R2, optional): when set, truncate every
    split to the first max_chars code points if `dataset` is in
    truncate_datasets. Unused (None) => byte-identical to the original loader.
    """
    bundle = load_manifest_in_distribution(dataset, seed)
    if max_chars is not None and truncate_datasets and dataset in truncate_datasets:
        _truncate_bundle_sides(bundle, ['train', 'val', 'test'], max_chars)
    return _assemble(bundle, dataset, seed, remove_limit, split_ratio)


def load_split_cross(src, tgt, seed, remove_limit=2, split_ratio="cross",
                     max_chars=None, truncate_datasets=None):
    """Cross-dataset loader (reads splits/cross_dataset/...).

    max_chars / truncate_datasets (R2, optional): truncate the source-fed
    train+val and/or the target-fed test to the first max_chars code points,
    depending on which dataset is in truncate_datasets. Unused (None) =>
    byte-identical to the original loader.
    """
    bundle = load_manifest_cross(src, tgt, seed)
    if max_chars is not None and truncate_datasets:
        sides = _cross_truncate_sides(src, tgt, truncate_datasets)
        if sides:
            _truncate_bundle_sides(bundle, sides, max_chars)
    tag = f"{src}->{tgt}"
    return _assemble(bundle, tag, seed, remove_limit, split_ratio)


def load_split_mixed(holdout, seed, remove_limit=2, split_ratio="mixed"):
    """Mixed leave-one-out loader (reads splits/mixed/...)."""
    bundle = load_manifest_mixed(holdout, seed)
    tag = f"mixed->{holdout}"
    return _assemble(bundle, tag, seed, remove_limit, split_ratio)


# ── BERT-only loaders ─────────────────────────────────────────────
# BERT works on raw text and uses HuggingFace tokenization. Running the
# non-BERT preprocessing (TF-IDF clean_str + vocab construction) just to
# get timing would mis-attribute that work as BERT's preprocessing time.

def _bundle_for_bert(bundle, protocol_tag, seed):
    """Thin wrapper: raw manifest + trainable flag. No non-BERT preprocessing."""
    return {
        'train_sentences': bundle['train_sentences'],
        'val_sentences': bundle['val_sentences'],
        'test_sentences': bundle['test_sentences'],
        'train_labels_str': bundle['train_labels'],
        'val_labels_str': bundle['val_labels'],
        'test_labels_str': bundle['test_labels'],
        'n_train': len(bundle['train_sentences']),
        'n_val': len(bundle['val_sentences']),
        'n_test': len(bundle['test_sentences']),
        'manifest': bundle['manifest'],
        'leakage_removed': bundle['leakage_removed'],
        'trainable': bundle.get('trainable', True),
        'untrainable_reason': bundle.get('untrainable_reason'),
        'dataset_tag': protocol_tag,
        'seed': seed,
    }


def load_bert_split(dataset, seed, max_chars=None, truncate_datasets=None):
    bundle = load_manifest_in_distribution(dataset, seed)
    if max_chars is not None and truncate_datasets and dataset in truncate_datasets:
        _truncate_bundle_sides(bundle, ['train', 'val', 'test'], max_chars)
    return _bundle_for_bert(bundle, dataset, seed)


def load_bert_split_cross(src, tgt, seed, max_chars=None, truncate_datasets=None):
    bundle = load_manifest_cross(src, tgt, seed)
    if max_chars is not None and truncate_datasets:
        sides = _cross_truncate_sides(src, tgt, truncate_datasets)
        if sides:
            _truncate_bundle_sides(bundle, sides, max_chars)
    return _bundle_for_bert(bundle, f"{src}->{tgt}", seed)


def load_bert_split_mixed(holdout, seed):
    return _bundle_for_bert(load_manifest_mixed(holdout, seed),
                            f"mixed->{holdout}", seed)


# ── Feature construction helpers (TF-IDF, sequences) ──────────────

def get_tfidf_features(data):
    """Build TF-IDF features, fit on train only (not train+val).

    Returns (X_train, X_val, X_test, y_train, y_val, y_test, rep_time).
    """
    n_train = data['n_train']
    n_val = data['n_val']
    tokenize_sentences = data['tokenize_sentences']

    train_texts = [' '.join(t) for t in tokenize_sentences[:n_train]]
    val_texts = [' '.join(t) for t in tokenize_sentences[n_train:n_train + n_val]]
    test_texts = [' '.join(t) for t in tokenize_sentences[n_train + n_val:]]

    rep_start = time.time()
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    rep_time = time.time() - rep_start

    labels = data['labels']
    y_train = np.array(labels[:n_train])
    y_val = np.array(labels[n_train:n_train + n_val])
    y_test = np.array(labels[n_train + n_val:])

    return X_train, X_val, X_test, y_train, y_val, y_test, rep_time


def get_sequence_data(data, max_len=512):
    """Integer-encode and pad tokenized sentences for CNN/LSTM."""
    n_train = data['n_train']
    n_val = data['n_val']
    tokenize_sentences = data['tokenize_sentences']
    word_list = data['word_list']

    word2idx = {w: i + 1 for i, w in enumerate(word_list)}
    vocab_size = len(word_list) + 1

    def encode_and_pad(token_lists):
        encoded = np.zeros((len(token_lists), max_len), dtype=np.int64)
        for i, tokens in enumerate(token_lists):
            for j, tok in enumerate(tokens[:max_len]):
                if tok in word2idx:
                    encoded[i, j] = word2idx[tok]
        return encoded

    X_train = encode_and_pad(tokenize_sentences[:n_train])
    X_val = encode_and_pad(tokenize_sentences[n_train:n_train + n_val])
    X_test = encode_and_pad(tokenize_sentences[n_train + n_val:])

    labels = data['labels']
    y_train = np.array(labels[:n_train])
    y_val = np.array(labels[n_train:n_train + n_val])
    y_test = np.array(labels[n_train + n_val:])

    return X_train, X_val, X_test, y_train, y_val, y_test, word2idx, vocab_size
