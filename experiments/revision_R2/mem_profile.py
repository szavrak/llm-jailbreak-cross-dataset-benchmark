"""R4-P1 — inference (deployment) memory profiler for the 8 detector configs.

Reviewer #4 asked for the real deployment memory footprint, not just the tiny
385-parameter LR head. For each config we BUILD the full deployable artifact
using the SAME production hyperparameters as the experiment runners, then
PROFILE a batch=1 forward pass — the two phases are strictly separated so the
build cost never contaminates the measured inference peak.

Methodology (what makes the numbers trustworthy)
------------------------------------------------
* BUILD ≠ PROFILE. We construct + fit + warm up, THEN (on CUDA) call
  torch.cuda.empty_cache() and torch.cuda.reset_peak_memory_stats(), and only
  THEN run the measured batch=1 inferences. So GPUAllocated/GPUReserved reflect
  the inference-time peak (weights resident + batch=1 activations), not build
  scratch.
* Realistic artifacts. Heads are fit on a REAL manifest training split (DAN
  in-distribution, seed 42) with the runners' exact estimators, so vocab / tree
  / forest sizes are production-scale, not toy:
    - TF-IDF+LR / +SVM : the run_lr / run_svm estimators (UNWEIGHTED, matching
      the TF-IDF runner) on the TfidfVectorizer built the same way as
      get_tfidf_features.
    - embedding heads   : embedding_classifier._build_classifier with
      EMBEDDING_HPARAMS[kind] (logreg class_weight='balanced'; xgboost with the
      full n_estimators/max_depth/learning_rate/subsample/colsample_bytree;
      randomforest n_estimators=500, class_weight='balanced').
  For transformers the inference footprint is architecture-determined, so we
  load the base checkpoint + a 2-way sequence-classification head and tokenize a
  REAL batch=1 input padded to max_len; fine-tuned weights have an identical
  footprint.
* Host RSS is sampled with psutil around the measured inference only:
  HostRSSDelta_MB = peak-during-inference − RSS-immediately-before-inference.
  (Most of the artifact RSS is already resident from the build phase; the delta
  isolates the inference allocation.)
* Sizes. EstimatedFP32WeightsMB = TotalParams × 4 bytes (labelled estimate),
  computed identically for every method (encoder + fitted head for the embedding
  configs, so it is internally consistent with TotalParams). ArtifactDiskMB is
  the REAL serialized size of the deployable sklearn artifact (vectorizer+head
  for TF-IDF, the head only for embedding methods — that is the deployable
  fitted-head artifact, the encoder being a HF download).
* SeqLen truthfulness. The reported SeqLen is what was ACTUALLY fed to the
  encoder. Transformers are tokenized with padding='max_length' to `seqlen`.
  For the SentenceTransformer encoders we set enc.max_seq_length = seqlen and
  feed a sample long enough to truncate/pad to `seqlen` tokens, so the measured
  peak reflects SeqLen=512 as labelled. If a given model cannot reach the
  requested length, the ACTUAL fed length is recorded in SeqLen (the requested
  value is kept separately in RequestedSeqLen, which is what the resume key
  uses). TF-IDF is bag-of-words, so its SeqLen is nominal (footprint is
  length-independent).

Each method is profiled in an ISOLATED SUBPROCESS so peaks can't bleed between
configs. Resume key = (Method, ModelName, BatchSize, RequestedSeqLen, Device);
the CSV header is validated on resume.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.config import (EMBEDDING_METHODS, EMBEDDING_HPARAMS,
                                 TRANSFORMER_METHODS)
from experiments.baselines.embedding_classifier import (_build_classifier,
                                                         _count_params)
from experiments.shared_preprocessing import load_split, load_bert_split
from preprocess import LABEL_MAP

RESULT_FILE = 'results/revision_R2/memory_profile.csv'
COLUMNS = ['Method', 'ModelName', 'Backend', 'Device', 'BatchSize',
           'RequestedSeqLen', 'SeqLen',
           'TotalParams', 'EstimatedFP32WeightsMB', 'ArtifactDiskMB',
           'GPUAllocated_MB', 'GPUReserved_MB',
           'HostRSSBefore_MB', 'HostRSSPeak_MB', 'HostRSSDelta_MB', 'Notes']

METHODS = ['TF-IDF+LR', 'TF-IDF+SVM', 'MiniLM+LR', 'MiniLM+XGBoost',
           'Arctic+RF', 'BERT-frozen+LR', 'BERT', 'ModernBERT']
TFIDF_METHODS = ('TF-IDF+LR', 'TF-IDF+SVM')

FAILURE_COLS = ['Method', 'ModelName', 'BatchSize', 'RequestedSeqLen', 'Device',
                'Reason']

_EMB_LOOKUP = {name: (model, kind) for name, model, kind in EMBEDDING_METHODS}

# The heads are fit on this REAL manifest training split (documented choice) so
# vocab / tree sizes are production-scale, not toy.
FIT_DATASET = 'dan'
FIT_SEED = 42
N_WARMUP = 3
N_MEASURE = 10


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _device_name(dev):
    import torch
    if dev.type == 'cuda':
        return torch.cuda.get_device_name(dev)
    return str(dev)


def _model_name(method):
    if method in TFIDF_METHODS:
        return 'tfidf'
    if method in _EMB_LOOKUP:
        return _EMB_LOOKUP[method][0]
    return TRANSFORMER_METHODS[method]['model_name']


def _method_device_name(method, dev):
    """TF-IDF is CPU-only; the encoder/transformer methods run on `dev`."""
    return 'cpu' if method in TFIDF_METHODS else _device_name(dev)


def _pickle_mb(obj):
    import pickle
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(obj, f)
        path = f.name
    mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return mb


# ── real fit data (DAN in-distribution train, seed 42) ─────────────
def _tfidf_fit_data():
    """Returns (vectorizer, X_train, y_train, sample_text) matching the runner's
    get_tfidf_features construction (fit on train only)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    data = load_split(FIT_DATASET, FIT_SEED)
    n_train = data['n_train']
    n_val = data['n_val']
    toks = data['tokenize_sentences']
    train_texts = [' '.join(t) for t in toks[:n_train]]
    test_texts = [' '.join(t) for t in toks[n_train + n_val:]]
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(train_texts)
    y_train = np.array(data['labels'][:n_train])
    sample = test_texts[0] if test_texts else train_texts[0]
    return vec, X_train, y_train, sample


def _text_fit_data():
    """Returns (train_texts, train_labels, sample_text) from the DAN split."""
    data = load_bert_split(FIT_DATASET, FIT_SEED)
    train_texts = data['train_sentences']
    train_labels = [LABEL_MAP[l] for l in data['train_labels_str']]
    sample = (data['test_sentences'][0] if data['test_sentences']
              else train_texts[0])
    return train_texts, train_labels, sample


# ── builders: return (infer_callable, meta_dict) ───────────────────
def build_tfidf(method, dev, seqlen, batch):
    """TF-IDF + (LR|SVM) built with the EXACT run_lr / run_svm estimators
    (unweighted), fit on the real DAN train TF-IDF matrix."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    vec, X_train, y_train, sample = _tfidf_fit_data()
    if method == 'TF-IDF+LR':
        head = LogisticRegression(random_state=FIT_SEED, max_iter=10000,
                                  solver='lbfgs')
    else:
        head = LinearSVC(random_state=FIT_SEED, max_iter=10000)
    head.fit(X_train, y_train)

    def infer():
        head.predict(vec.transform([sample]))

    vocab = len(vec.vocabulary_)
    head_params = int(head.coef_.size + head.intercept_.size)
    artifact_mb = _pickle_mb(vec) + _pickle_mb(head)
    total = vocab + head_params
    return infer, dict(
        total_params=total, fp32_mb=total * 4 / 1e6, artifact_mb=artifact_mb,
        backend='cpu', device='cpu', model_name='tfidf', seqlen=seqlen,
        notes=f'TF-IDF vocab={vocab}; unweighted head; '
              f'SeqLen nominal (bag-of-words, length-independent); '
              f'ArtifactDiskMB=vectorizer+head pickle')


def _long_encoder_input(enc, sample, seqlen):
    """Build a text long enough to fill `seqlen` tokens under enc.max_seq_length,
    and return (long_text, actual_seqlen) where actual_seqlen is the token count
    the encoder will ACTUALLY feed after truncation/padding (batch=1 pads to the
    single sequence's length). If the model cannot reach `seqlen` (its truncation
    cap is lower), actual_seqlen is that lower value — reported truthfully."""
    tok = enc.tokenizer
    base_ids = tok(sample, add_special_tokens=False)['input_ids']
    per = max(1, len(base_ids))
    reps = seqlen // per + 4          # comfortably overshoot the target length
    long_text = (sample + ' ') * reps
    ids = tok(long_text, truncation=True, max_length=enc.max_seq_length)['input_ids']
    return long_text, len(ids)


def build_encoder_head(method, dev, seqlen, batch):
    """Sentence encoder + fitted head. Head hparams == EMBEDDING_HPARAMS[kind]
    (matches the embedding runner exactly). Encoder runs on `dev`.

    The measured input is a REAL text long enough to truncate/pad to `seqlen`
    tokens (set via enc.max_seq_length), so the profiled peak reflects the
    labelled SeqLen rather than a short ~85-word sentence. The actual fed length
    is recorded in meta['seqlen']."""
    from sentence_transformers import SentenceTransformer
    model_name, kind = _EMB_LOOKUP[method]
    train_texts, train_labels, sample = _text_fit_data()
    enc = SentenceTransformer(model_name, device=str(dev))
    enc.max_seq_length = seqlen
    long_text, actual_seqlen = _long_encoder_input(enc, sample, seqlen)
    X_train = enc.encode(train_texts, batch_size=EMBEDDING_HPARAMS['batch_size'],
                         convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
    head = _build_classifier(kind, FIT_SEED, EMBEDDING_HPARAMS[kind])
    head.fit(X_train, train_labels)

    def infer():
        emb = enc.encode([long_text] * batch, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        head.predict(emb)

    enc_params = sum(p.numel() for p in enc.parameters())
    head_params = _count_params(head, kind, X_train.shape[1])
    artifact_mb = _pickle_mb(head)      # encoder is a HF checkpoint, not pickled
    total = int(enc_params + head_params)
    return infer, dict(
        total_params=total, fp32_mb=total * 4 / 1e6, artifact_mb=artifact_mb,
        backend=dev.type, device=_device_name(dev), model_name=model_name,
        seqlen=actual_seqlen,
        notes=(f'{kind} head (EMBEDDING_HPARAMS); '
               f'EstimatedFP32WeightsMB=TotalParams*4B (encoder+head); '
               f'ArtifactDiskMB=fitted-head pickle (encoder is a HF download)'))


def build_transformer(method, dev, seqlen, batch):
    """Base checkpoint + 2-way sequence-classification head, real tokenized
    batch=1 input padded to max_len. Fine-tuned weights share this footprint."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = TRANSFORMER_METHODS[method]['model_name']
    _, _, sample = _text_fit_data()
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok([sample] * batch, padding='max_length', truncation=True,
              max_length=seqlen, return_tensors='pt')
    input_ids = enc['input_ids'].to(dev)
    attn = enc['attention_mask'].to(dev)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(dev).eval()

    def infer():
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)

    total = sum(p.numel() for p in model.parameters())
    actual_seqlen = int(input_ids.shape[1])   # padded to max_length == seqlen
    return infer, dict(
        total_params=int(total), fp32_mb=total * 4 / 1e6, artifact_mb=None,
        backend=dev.type, device=_device_name(dev), model_name=model_name,
        seqlen=actual_seqlen,
        notes='seq-classification head; EstimatedFP32WeightsMB=params*4B '
              '(architecture-determined; fine-tuned weights identical)')


def _build(method, dev, seqlen, batch):
    if method in TFIDF_METHODS:
        return build_tfidf(method, dev, seqlen, batch)
    if method in _EMB_LOOKUP:
        return build_encoder_head(method, dev, seqlen, batch)
    if method in ('BERT', 'ModernBERT'):
        return build_transformer(method, dev, seqlen, batch)
    raise ValueError(f"Unknown method: {method}")


def _rss_bytes():
    import psutil
    return psutil.Process().memory_info().rss


def profile_one(method, seqlen, batch, out_file):
    """Child entry point: BUILD then PROFILE one method, append its row."""
    import torch
    dev = get_device()
    cpu_only = method in TFIDF_METHODS
    prof_dev = torch.device('cpu') if cpu_only else dev

    # ── BUILD (not measured) ──
    infer, meta = _build(method, prof_dev, seqlen, batch)

    # ── warm-up (not measured) ──
    for _ in range(N_WARMUP):
        infer()

    # ── enter measured region: reset peaks AFTER build + warm-up ──
    if prof_dev.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rss_before = _rss_bytes()
    rss_peak = rss_before
    for _ in range(N_MEASURE):
        infer()
        rss_peak = max(rss_peak, _rss_bytes())

    gpu_alloc = gpu_resv = 0.0
    if prof_dev.type == 'cuda':
        torch.cuda.synchronize()
        gpu_alloc = torch.cuda.max_memory_allocated() / 1e6
        gpu_resv = torch.cuda.max_memory_reserved() / 1e6

    artifact_mb = meta['artifact_mb']
    actual_seqlen = meta.get('seqlen', seqlen)
    if actual_seqlen != seqlen:
        print(f"  [SeqLen] {method}: requested {seqlen} tokens but the model fed "
              f"{actual_seqlen}; recording the ACTUAL length.")
    row = {
        'Method': method,
        'ModelName': meta['model_name'],
        'Backend': meta['backend'],
        'Device': meta['device'],
        'BatchSize': batch,
        'RequestedSeqLen': seqlen,
        'SeqLen': actual_seqlen,
        'TotalParams': meta['total_params'],
        'EstimatedFP32WeightsMB': round(meta['fp32_mb'], 3),
        'ArtifactDiskMB': ('' if artifact_mb is None else round(artifact_mb, 4)),
        'GPUAllocated_MB': round(gpu_alloc, 2),
        'GPUReserved_MB': round(gpu_resv, 2),
        'HostRSSBefore_MB': round(rss_before / 1e6, 2),
        'HostRSSPeak_MB': round(rss_peak / 1e6, 2),
        'HostRSSDelta_MB': round((rss_peak - rss_before) / 1e6, 2),
        'Notes': meta['notes'],
    }
    _append_row(out_file, row)
    print(f"[{method}] GPUAlloc={row['GPUAllocated_MB']}MB "
          f"GPUResv={row['GPUReserved_MB']}MB "
          f"RSSdelta={row['HostRSSDelta_MB']}MB params={row['TotalParams']} "
          f"fp32={row['EstimatedFP32WeightsMB']}MB "
          f"disk={row['ArtifactDiskMB']} seqlen={row['SeqLen']} "
          f"dev={row['Device']}")


def _failures_path(result_file):
    base, ext = os.path.splitext(result_file)
    return base + '_failures' + (ext or '.csv')


def _record_failure(failures, failures_file, rec):
    """Log a failed profiling config to an in-process list AND a SEPARATE
    failures CSV (never the resume-keyed results file), so a rerun on adequate
    hardware retries the config instead of permanently skipping it."""
    import pandas as pd
    failures.append(rec)
    os.makedirs(os.path.dirname(failures_file), exist_ok=True)
    df = pd.DataFrame([rec])[FAILURE_COLS]
    if not os.path.isfile(failures_file):
        df.to_csv(failures_file, index=False)
    else:
        df.to_csv(failures_file, mode='a', header=False, index=False)


def _append_row(out_file, row):
    import pandas as pd
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df = pd.DataFrame([row])[COLUMNS]
    if not os.path.isfile(out_file):
        df.to_csv(out_file, index=False)
    else:
        df.to_csv(out_file, mode='a', header=False, index=False)


def _validate_header(out_file):
    if not os.path.isfile(out_file):
        return
    with open(out_file, 'r') as f:
        header = f.readline().rstrip('\n').rstrip('\r').split(',')
    if header != COLUMNS:
        raise RuntimeError(
            f"Schema mismatch in {out_file}: header has {len(header)} columns "
            f"but the profiler expects {len(COLUMNS)} ({COLUMNS}). Write to a "
            f"new file instead of appending.")


def _completed(out_file):
    """Resume set keyed on (Method, ModelName, BatchSize, RequestedSeqLen,
    Device). RequestedSeqLen (not the actually-fed SeqLen) is used so resume is
    robust even when a model cannot reach the requested length."""
    import pandas as pd
    if not os.path.isfile(out_file):
        return set()
    _validate_header(out_file)
    df = pd.read_csv(out_file)
    done = set()
    for r in df.itertuples(index=False):
        done.add((r.Method, str(r.ModelName), str(r.BatchSize),
                  str(r.RequestedSeqLen), str(r.Device)))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=RESULT_FILE)
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    # Child mode: profile exactly one method in this process, then exit.
    parser.add_argument('--method', type=str, default=None)
    cli_args = parser.parse_args()

    if cli_args.method is not None:
        profile_one(cli_args.method, cli_args.seqlen, cli_args.batch,
                    cli_args.result_file)
        return

    # Driver mode: one isolated subprocess per method.
    os.makedirs(os.path.dirname(cli_args.result_file), exist_ok=True)
    _validate_header(cli_args.result_file)
    done = _completed(cli_args.result_file)
    dev = get_device()
    failures_file = _failures_path(cli_args.result_file)
    failures = []
    for method in cli_args.methods:
        dev_name = _method_device_name(method, dev)
        key = (method, _model_name(method), str(cli_args.batch),
               str(cli_args.seqlen), dev_name)
        if key in done:
            print(f"[SKIP] {method} (already profiled for this "
                  f"batch/seqlen/device)")
            continue
        print(f"\n=== profiling {method} (isolated subprocess) ===")
        cmd = [sys.executable, os.path.abspath(__file__),
               '--method', method,
               '--result_file', cli_args.result_file,
               '--seqlen', str(cli_args.seqlen),
               '--batch', str(cli_args.batch)]
        rec = {'Method': method, 'ModelName': _model_name(method),
               'BatchSize': cli_args.batch, 'RequestedSeqLen': cli_args.seqlen,
               'Device': dev_name, 'Reason': None}
        try:
            ret = subprocess.run(cmd)
        except Exception as e:
            print(f"  [FAILED] {method} (exception: {e})")
            rec['Reason'] = f'exception: {e}'
            _record_failure(failures, failures_file, rec)
            continue
        if ret.returncode != 0:
            print(f"  [FAILED] {method} (exit {ret.returncode})")
            rec['Reason'] = f'child exited {ret.returncode}'
            _record_failure(failures, failures_file, rec)

    if failures:
        print("\n" + "=" * 60)
        print(f"[FAILURES] {len(failures)} config(s) did NOT complete:")
        for r in failures:
            print(f"  - {r['Method']} | seqlen={r['RequestedSeqLen']} | "
                  f"batch={r['BatchSize']} | dev={r['Device']} | {r['Reason']}")
        print(f"Logged to {failures_file} (NOT the results file), so a rerun on "
              f"adequate hardware retries the config.")
        print(f"\nDone. -> {cli_args.result_file}")
        sys.exit(1)

    print(f"\nDone. -> {cli_args.result_file}")


if __name__ == '__main__':
    main()
