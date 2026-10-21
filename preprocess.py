"""Text preprocessing for non-BERT methods (TF-IDF baselines).

v2 changes (post-contamination audit):
- Fixed binary label mapping {benign: 0, jailbreak: 1} — no LabelEncoder fit
  against test labels.
- No English stopword removal — words like "ignore", "not", "you", "do" carry
  jailbreak-intent signal and should stay in the vocabulary.
- Vocabulary is built from the training documents only (the caller is
  responsible for passing the right sentences; this module does not know about
  splits).

BERT should use raw text and the HuggingFace tokenizer; it does not call this.
"""
import numpy as np
import re


LABEL_MAP = {'benign': 0, 'jailbreak': 1}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def encode_labels(original_labels_train, original_labels_test, args):
    """Encode labels using a fixed {benign: 0, jailbreak: 1} mapping.

    Accepts either string labels ('benign'/'jailbreak') or integer labels (0/1).
    Returns (labels_flat, num_class) where labels_flat is the concatenation of
    train labels followed by test labels (matches the legacy pipeline).
    """
    def _to_int(lbl):
        if isinstance(lbl, (int, np.integer)):
            return int(lbl)
        if isinstance(lbl, str):
            return LABEL_MAP[lbl]
        raise ValueError(f"Unrecognized label type: {type(lbl)} value={lbl!r}")

    train_labels = [_to_int(l) for l in original_labels_train]
    test_labels = [_to_int(l) for l in original_labels_test]

    if not getattr(args, 'easy_copy', False):
        print(f"Labels: benign=0, jailbreak=1 (fixed mapping)")

    return train_labels + test_labels, 2


def clean_str(string):
    """Tokenizer-friendly cleanup. Preserves intent-bearing tokens like
    'ignore', 'you', 'not' (no stopword removal)."""
    string = re.sub(r"[^A-Za-z0-9()\.,!?:;\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_data(original_train_sentences, original_test_sentences, args):
    """Tokenize train and test documents using a train-only vocabulary.

    Vocabulary rules:
    - Tokens are lowercased (via clean_str).
    - A token is kept iff it appears at least args.remove_limit times in the
      training sentences.
    - No stopword filtering.
    - Test documents reuse the train vocabulary; OOV tokens are dropped.
    """
    train_word_freq = {}
    for sentence in original_train_sentences:
        for word in clean_str(sentence).split():
            train_word_freq[word] = train_word_freq.get(word, 0) + 1

    vocab = {w for w, c in train_word_freq.items() if c >= args.remove_limit}

    tokenize_sentences = []
    for sentence in original_train_sentences + original_test_sentences:
        tokens = [w for w in clean_str(sentence).split() if w in vocab]
        tokenize_sentences.append(tokens)

    word_list = sorted(vocab)

    if not getattr(args, 'easy_copy', False):
        lens = [len(s) for s in tokenize_sentences[:len(original_train_sentences)]]
        if lens:
            print(f"Train vocab size: {len(word_list)}  avg train doc length: {sum(lens)/len(lens):.1f}")

    return tokenize_sentences, word_list
