# Revision R2 experiments (Reviewer #4)

Colab notebook + runners for the three additions Reviewer #4 asked for. Run on
Colab (GPU). Results are written to `results/revision_R2/` **on Drive**
(resume-safe across disconnects). The submitted `results/v2/` and the first-round
`results/revision_R1/` are **never touched**.

## What these add

| Reviewer point | Experiment | Runner | Output |
|----------------|-----------|--------|--------|
| **R4-P2** — 3/10 seeds are too few to trust the cross-dataset gap | 30-seed stability on the single pair jackhhao→advbench (BERT, ModernBERT, MiniLM+LR) | `experiments/revision_R2/run_30seed.py` | `results/revision_R2/stability_30seed.csv` |
| **R4-P3** — DAN jailbreak prompts are much longer than its benign ones, so length may be the real signal | Truncate every DAN row (both classes) to the first 60 Unicode code points and re-run the full 8-detector roster + a length-only baseline. The uncapped (`orig`) detector rows are imported verbatim from `results/revision_R1` | `experiments/revision_R2/import_orig_rows.py`, `experiments/revision_R2/run_length_confound.py` | `results/revision_R2/length_confound.csv`, `length_confound_collisions.csv` |
| **R4-P1** — reported memory is just the 385-param LR head, not the deployed artifact | Inference-memory profiler for the 8 configs (encoder + fitted head / vectorizer / forest, or transformer + classification head), batch=1, isolated subprocess, BUILD separated from measured PROFILE | `experiments/revision_R2/mem_profile.py` | `results/revision_R2/memory_profile.csv` |

## Notebook

`R2_Experiments.ipynb` — one notebook: a bootstrap that pins provenance
(`provenance.txt`) and defines `sync_splits_to_drive()`, then three sections
(A/B/C mapping to R4-P2/P3/P1). Each section builds any missing frozen manifests
(persisted back to Drive so a resumed session reuses the SAME splits) → runs →
writes to `results/revision_R2/` on Drive. Every runner is resume-safe: a re-run
skips completed `(Method, Dataset, Seed, HyperParams)` rows (and, for the memory
profiler, completed `(Method, ModelName, BatchSize, RequestedSeqLen, Device)`
keys), so a
disconnected session is resumed by simply re-running the cell.

## Design notes

- **Truncation is a load-time transform** applied AFTER manifest split
  membership and leakage removal (via the new optional `max_chars` /
  `truncate_datasets` kwargs on the shared loaders). The contamination audit
  therefore still ran on full text. Only DAN is truncated; the partner dataset
  in each cross pair is left intact. jackhhao↔dan pairs stay excluded
  (`CONTAMINATED_PAIRS`).
- **Condition id.** The length setting (`60` vs `orig`) is recorded inside the
  `HyperParams` JSON (so resume keys stay distinct for every method, including
  TF-IDF, which otherwise carries no hparams) and echoed into `Notes`
  (`maxchars=60`). The `Dataset` column keeps the plain tag (`dan`,
  `dan->advbench`, …).
- **Reused `orig` rows.** The uncapped comparison for the 8 detectors is **not
  recomputed**: `import_orig_rows.py` imports those rows verbatim from
  `results/revision_R1` (DAN in-distribution from
  `in_distribution_{tfidf,embeddings,bert}.csv`, the six DAN-facing cross pairs
  from `cross_dataset.csv`), tags them `max_chars="orig"`, and writes them into
  `length_confound.csv`. It **fails loudly** if any of the 8 detectors × 10
  seeds × 7 conditions source rows is missing or duplicated. Only the capped
  detector rows and the LengthOnly baseline (absent from R1, so run at both
  `orig` and `60`) are new compute.
- **Distinct embedding cache.** Capped-text embeddings use the
  `lenconf_mc60__…` cache namespace, so they never overwrite the original
  `plm_cache/` entries.
- **Collision audit.** After truncation we count rows whose first-60-char prefix
  becomes identical, on the ACTUAL asymmetrically-capped model inputs (only the
  DAN side is truncated): within-train, within-val, train-vs-test, and
  val-vs-test. Policy: keep them; the counts are saved to
  `length_confound_collisions.csv`.
- **Memory profiler** separates BUILD from PROFILE. Artifacts are built with the
  runners' exact production hyperparameters (unweighted TF-IDF heads; full
  XGBoost/RandomForest configs) and fit on a real manifest split (DAN
  in-distribution, seed 42). It then resets CUDA peak stats AFTER build + warm-up
  and reports, at batch=1, `torch.cuda.max_memory_allocated` /
  `max_memory_reserved`, the psutil host-RSS inference delta, `TotalParams`, an
  `EstimatedFP32WeightsMB` (`TotalParams×4`, computed identically for every
  method — encoder **plus** fitted head for the embedding configs, so it is
  internally consistent with `TotalParams`), and the real serialized
  `ArtifactDiskMB` for the sklearn heads (head-only for embedding methods, the
  encoder being a HF download).
- **Truthful `SeqLen`.** The reported `SeqLen` is what was ACTUALLY fed.
  Transformers are tokenized with `padding='max_length'` to 512; the
  SentenceTransformer encoders set `max_seq_length = 512` and are fed a text long
  enough to truncate/pad to 512 tokens, so the cross-method comparison is fair
  (previously the encoders were measured at ~85 words while the row said 512). If
  a model cannot reach the requested length, the ACTUAL fed length is recorded in
  `SeqLen`; the requested value is kept in `RequestedSeqLen`, which is what the
  resume key uses: `(Method, ModelName, BatchSize, RequestedSeqLen, Device)`.
  TF-IDF is bag-of-words, so its `SeqLen` is nominal (footprint is
  length-independent).
- **Failure handling (OOM & errors).** A CUDA-OOM or other exception is logged to
  a SEPARATE `results/revision_R2/<name>_failures.csv` — **not** the resume-keyed
  results file — so a rerun on a bigger GPU naturally retries the config instead
  of permanently skipping it. Each runner also prints a LOUD end-of-run summary
  of every failed `(method, condition/pair, seed)` and **exits non-zero** if any
  failure occurred, so a notebook cell surfaces the problem instead of appearing
  to succeed.
- **Coverage gate.** `experiments/revision_R2/check_coverage.py` (also the final
  notebook cell, section E) recomputes the full expected grid for each of the
  three result CSVs and FAILS LOUDLY (non-zero exit) if any row is missing or
  marked `UNTRAINABLE`/failed: `length_confound.csv` (8 detectors + LengthOnly ×
  7 conditions × 10 seeds × {orig, 60}), `stability_30seed.csv` (3 methods × 30
  seeds), and `memory_profile.csv` (8 configs, `SeqLen` truthful). Run it before
  trusting any number in the paper.

## Compute note

The bulk of the cost is the two fine-tuned transformers. 30-seed stability adds
`2 encoders × 30 seeds` short fine-tunes on ~930 training rows (cheap). The
length confound adds `2 encoders × 7 conditions × 10 seeds` capped fine-tunes
(~140 each) — the heavy part, on the order of tens of GPU-hours on a T4/L4. TF-IDF,
embedding + head, and the length-only baseline are minutes. The memory profiler
is seconds per config.

**Do not change `batch_size` for the reused-comparison transformers.** The
capped BERT/ModernBERT rows are compared against the imported R1 `orig` rows,
which were trained at `BERT_HPARAMS['batch_size'] = 16`. Changing the batch size
changes the optimization and would make the orig-vs-60 comparison invalid. If a
modern encoder OOMs, use an A100/L4 runtime, or — if the batch size genuinely
must change — recompute the `orig` baseline at the SAME batch (do not import the
R1 rows), or use gradient accumulation to preserve the effective batch of 16.
On a CUDA OOM the runners log the config to a separate `<name>_failures.csv`
(never the results file) and disable that method for the remaining configs
rather than retrying it every seed; the run then exits non-zero. Transformer
fine-tuning uses `batch_size = 16` and embedding encoding uses `batch_size = 32`
(only the memory profiler's inference pass is `batch = 1`).

**Frozen-split persistence.** The notebook builds the 20 new 30-seed pair
manifests with `run_30seed.py --build_only` and syncs them to Drive BEFORE the
long run starts, so a mid-run disconnect cannot lose them. Even without that, a
disconnect before the sync is safe: `build_manifests` is deterministic, so a
resumed session rebuilds byte-identical splits.

## After the runs

Read the three CSVs directly (or with `experiments/resume_utils.aggregate_results`
for the metric files). For the length confound, `length_confound.csv` already
contains BOTH the `orig` rows (imported from `results/revision_R1` by
`import_orig_rows.py`) and the capped `60` rows in the same schema; group on
`(Method, Dataset, Seed)` and the `max_chars` field in `HyperParams` (or the
`maxchars=` tag in `Notes`) to build the orig-vs-60 table — no external join
needed. `provenance.txt` records the package versions and HF model revisions the
comparison relied on.
