"""Single source of truth for all experiment hyperparameters, seeds, and datasets."""

SEEDS = [42, 123, 456]

# ── Revision R1 (KNOSYS-D-26-08416) ───────────────────────────
# A2 (Reviewer #1, point 2): three seeds are insufficient given BERT's high
# per-cell variance. SEEDS_EXTENDED supplies 10 seeds for confidence intervals
# / permutation tests. The original three are a prefix so v2 runs still resume.
SEEDS_EXTENDED = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# ── Revision R2 (Reviewer #4, point R4-P2) ─────────────────────
# 30-seed stability study on a single cross-dataset pair. The first ten are
# SEEDS_EXTENDED (so their frozen manifests are reused untouched); the extra
# twenty are fixed, predefined, and distinct — never randomized at run time.
SEEDS_30 = SEEDS_EXTENDED + [
    2223, 2425, 2627, 2829, 3031, 3233, 3435, 3637, 3839, 4041,
    4243, 4445, 4647, 4849, 5051, 5253, 5455, 5657, 5859, 6061,
]

# Main benchmark datasets — hawkins excluded (derived from dan+jackhhao).
# See results/dedup_audit.txt for the contamination rationale.
DATASETS = [
    'dan',              # "Do Anything Now" — 6,142 (653 jailbreak) — ACM CCS 2024
    'jailbreakbench',   # JailbreakBench — 200 (100 jailbreak) — NeurIPS 2024
    'jackhhao',         # jackhhao/jailbreak-classification — 1,033 (517 jailbreak) — HuggingFace
    'advbench',         # AdvBench — 1,040 (520 jailbreak) — Zou et al. 2023
    'harmbench',        # HarmBench — 793 (393 jailbreak) — Mazeika et al. 2024
]

# hawkins is retained only as a diagnostic exhibit for the contamination audit,
# never as an independent dataset in the main benchmark.
DIAGNOSTIC_DATASETS = ['hawkins']

SPLIT_RATIO = "70:10:20"
REMOVE_LIMIT = 2

# BERT fine-tuning hyperparameters (new baseline for jailbreak)
BERT_HPARAMS = {
    'model_name': 'bert-base-uncased',
    'lr': 2e-5,
    'epochs': 5,
    'batch_size': 16,
    'max_len': 512,
    'early_stopping_patience': 2,
}


def transformer_hparams(model_name, class_weight=None):
    """BERT_HPARAMS clone with a swapped HF checkpoint and/or loss weighting.

    class_weight: None -> default (unweighted) cross-entropy, byte-identical to
    the original BERT baseline; 'balanced' -> inverse-frequency weighted CE,
    matching the MiniLM+LR scheme. The key is only added when set, so plain
    'BERT' keeps its original resume hash.
    """
    hp = BERT_HPARAMS.copy()
    hp['model_name'] = model_name
    if class_weight is not None:
        hp['class_weight'] = class_weight
    return hp


# ── Revision R1 fine-tuned transformer roster ─────────────────
# Addresses A1 (Reviewer #1, point 1) and A3 (Reviewer #3, point 3) with a
# single registry the runners iterate over. 'BERT' is the original baseline,
# unchanged. 'BERT-Weighted' equalises the class-weighting confound from the
# transformer side; RoBERTa / DeBERTa-v3 / ModernBERT are the modern encoders
# (2019 / 2021 / 2024) answering "bert-base-uncased is outdated".
TRANSFORMER_METHODS = {
    'BERT':          BERT_HPARAMS,
    'BERT-Weighted': transformer_hparams('bert-base-uncased', class_weight='balanced'),
    'RoBERTa':       transformer_hparams('roberta-base'),
    'DeBERTa-v3':    transformer_hparams('microsoft/deberta-v3-base'),
    'ModernBERT':    transformer_hparams('answerdotai/ModernBERT-base'),
}


# Pretrained-embedding + simple-classifier baselines (Galinkin & Sablotny 2024,
# NemoGuard 2024). Each entry is (method_name, embedding_model, classifier).
# The classifier string is decoded in experiments.baselines.embedding_classifier.
EMBEDDING_METHODS = [
    ('MiniLM+LR',          'sentence-transformers/all-MiniLM-L6-v2', 'logreg'),
    ('MiniLM+XGBoost',     'sentence-transformers/all-MiniLM-L6-v2', 'xgboost'),
    ('MiniLM+XGBoost-Bal', 'sentence-transformers/all-MiniLM-L6-v2', 'xgboost_balanced'),
    ('Arctic+RF',          'Snowflake/snowflake-arctic-embed-l',     'randomforest'),
    # Disentangling ablation: isolates the effect of freezing the encoder
    # separately from the effect of using MiniLM's sentence-similarity-trained
    # embeddings. bert-base-uncased is auto-wrapped with mean-pooling by
    # SentenceTransformer since it is not a sentence-transformers checkpoint.
    ('BERT-frozen+LR',     'bert-base-uncased',                      'logreg'),
    # A1 (Reviewer #1, point 1) — unweighted side: MiniLM+LR with no class
    # reweighting, matching fine-tuned BERT's unweighted loss. Isolates whether
    # the cross-dataset advantage is an artefact of inverse-frequency weighting.
    ('MiniLM+LR-NoWeight', 'sentence-transformers/all-MiniLM-L6-v2', 'logreg_noweight'),
]

EMBEDDING_HPARAMS = {
    'max_seq_length': 512,  # truncate at encoding time
    'batch_size': 32,
    'logreg':       {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'},
    # A1 unweighted counterpart of 'logreg' (class_weight omitted == None).
    'logreg_noweight': {'C': 1.0, 'max_iter': 1000, 'class_weight': None},
    'xgboost':      {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.1,
                     'subsample': 0.9, 'colsample_bytree': 0.9},
    # Class-balanced XGBoost ablation: scale_pos_weight is computed per-fit
    # from training labels as n_neg/n_pos; all other hparams match 'xgboost'.
    'xgboost_balanced': {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.1,
                         'subsample': 0.9, 'colsample_bytree': 0.9,
                         'scale_pos_weight': 'auto'},
    'randomforest': {'n_estimators': 500, 'max_depth': None, 'n_jobs': -1,
                     'class_weight': 'balanced'},
}


# Result CSV columns (v2 schema — post-decontamination)
RESULT_COLUMNS = [
    'Method', 'Dataset', 'Seed', 'SplitRatio', 'RemoveLimit',
    # Primary jailbreak-detection metrics
    'F1Binary', 'RecallBinary', 'PrecisionBinary',
    # Secondary aggregate metrics
    'Accuracy',
    'F1Macro', 'RecallMacro', 'PrecisionMacro',
    'F1Weighted', 'RecallWeighted', 'PrecisionWeighted',
    # Confusion matrix counts
    'TN', 'FP', 'FN', 'TP',
    # Split + leakage accounting
    'TrainSize', 'ValSize', 'TestSize', 'LeakageRemoved',
    # Timing (5-field schema)
    'PreprocessTimeSec', 'RepresentationTimeSec',
    'TrainTimeSec', 'InferTimeSec', 'TotalRunTimeSec',
    # Book-keeping
    'NumParams', 'Device', 'HyperParams',
    # Free-form annotation (e.g. "UNTRAINABLE: <reason>"); does NOT participate
    # in the resume key so untrainable rows still dedupe correctly.
    'Notes',
]

# ── Cross-dataset evaluation ──────────────────────────────────
# All directional pairs among the 5 main datasets (hawkins excluded).
# After manifest-level row dedup, every pair is clean w.r.t. test-side leakage.
#
# dan↔jackhhao is excluded: per the HuggingFace dataset card, jackhhao's
# jailbreak class is sourced from github.com/verazuo/jailbreak_llms (the DAN
# repo), so the positive class is shared by construction. Cross-dataset
# transfer between them measures memorization, not generalization.
CONTAMINATED_PAIRS = frozenset({
    ('dan', 'jackhhao'),
    ('jackhhao', 'dan'),
})

CROSS_DATASET_PAIRS = [
    (train_ds, test_ds)
    for train_ds in DATASETS
    for test_ds in DATASETS
    if train_ds != test_ds
    and (train_ds, test_ds) not in CONTAMINATED_PAIRS
]

# ── Mixed leave-one-out evaluation ────────────────────────────
# Held-outs exclude dan and jackhhao: since their jailbreak classes are
# shared (see CONTAMINATED_PAIRS), holding out one means the training set
# still contains those same jailbreak prompts under the other's name —
# so the held-out test measures memorization, not novel-attack generalization.
MIXED_HELDOUTS = [d for d in DATASETS if d not in {'dan', 'jackhhao'}]
