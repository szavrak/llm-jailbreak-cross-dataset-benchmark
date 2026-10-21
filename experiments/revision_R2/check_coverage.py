"""R2 coverage assertion — fail LOUDLY if any paper-data CSV is incomplete.

The three R2 result files each have a KNOWN full grid. A silently-missing or
OOM/failed row would poison the paper (a mean over 9 of 10 seeds, an orig-vs-60
table with a hole). This script recomputes the expected grid for each file and
lists every hole:

  * length_confound.csv  — 8 detectors x 10 seeds x 7 conditions at
    max_chars in {orig, 60}  (orig imported from R1, 60 computed) PLUS the
    LengthOnly baseline at {orig, 60}. Lists any missing
    (Method, Condition, Seed, MaxChars) and any UNTRAINABLE / failed rows.
  * stability_30seed.csv — 3 methods x 30 seeds for jackhhao->advbench.
  * memory_profile.csv   — all 8 configs present, SeqLen truthful (a positive
    integer not exceeding the requested length).

A per-file '<name>_failures.csv' (written by the runners for OOM / errors) is
read ONLY to annotate configs that are STILL missing from the results file:
success supersedes a historical failure, so a config that failed once but later
SUCCEEDED (its valid result row is present) is COVERED and does NOT fail the
gate. Prints a clear PASS/FAIL per file and exits non-zero if any config is
genuinely missing.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import SEEDS_EXTENDED, SEEDS_30
from run_length_confound import (CONDITIONS, TFIDF_METHODS, EMB_METHODS,
                                  BERT_METHODS, LENGTH_METHOD)
from run_30seed import PAIR as STAB_PAIR, METHODS as STAB_METHODS
from mem_profile import METHODS as MEM_METHODS

R2_DIR = 'results/revision_R2'
DETECTORS = TFIDF_METHODS + EMB_METHODS + BERT_METHODS   # the 8 detectors
MC_LABELS = ['orig', '60']


def _failures_path(result_file):
    base, ext = os.path.splitext(result_file)
    return base + '_failures' + (ext or '.csv')


def _is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == '':
        return True
    return False


def _row_trained(notes, f1):
    """A row counts as a completed training only if it is not UNTRAINABLE and
    carries a real F1Binary."""
    if isinstance(notes, str) and notes.startswith('UNTRAINABLE'):
        return False
    return not _is_blank(f1)


def _max_chars(hp):
    if not isinstance(hp, str) or not hp.strip():
        return None
    try:
        return json.loads(hp).get('max_chars')
    except Exception:
        return None


def _failure_map(result_file, key_fn):
    """Read the '<name>_failures.csv' sidecar and map each logged failure to a
    human-readable description, keyed by the SAME config key used for the results
    grid (via key_fn(row_dict)). Used ONLY to explain configs that are still
    missing from the results file; a config whose valid result row is present is
    covered regardless of any sidecar entry. Returns (map, read_error_or_None)."""
    path = _failures_path(result_file)
    if not os.path.isfile(path):
        return {}, None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {}, f"(could not read {path}: {e})"
    fmap = {}
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        desc = ' | '.join(f"{c}={row[c]}" for c in df.columns)
        try:
            fmap[key_fn(row)] = desc
        except Exception:
            fmap.setdefault(None, desc)   # malformed row: cannot annotate a config
    return fmap, None


# ── per-file checks: return (ok, lines) ─────────────────────────────
def check_length_confound(path, seeds=SEEDS_EXTENDED, conditions=CONDITIONS):
    lines = []
    if not os.path.isfile(path):
        return False, [f"MISSING FILE: {path}"]
    df = pd.read_csv(path)

    good, untrainable = set(), []
    for _, r in df.iterrows():
        mc = _max_chars(r.get('HyperParams'))
        keybase = (r['Method'], r['Dataset'], int(r['Seed']), str(mc))
        if _row_trained(r.get('Notes'), r.get('F1Binary')):
            good.add(keybase)
        else:
            untrainable.append(keybase)

    expected = []
    for mc in MC_LABELS:
        for cond in conditions:
            for seed in seeds:
                for m in DETECTORS + [LENGTH_METHOD]:
                    expected.append((m, cond, int(seed), mc))

    missing = [k for k in expected if k not in good]
    n_exp = len(expected)
    lines.append(f"expected {n_exp} rows "
                 f"({len(DETECTORS)} detectors + LengthOnly) x {len(conditions)} "
                 f"conditions x {len(seeds)} seeds x {len(MC_LABELS)} max_chars; "
                 f"found {len(good)} completed")
    fmap, ferr = _failure_map(path, lambda row: (
        row['Method'], row['Condition'], int(row['Seed']), str(row['MaxChars'])))
    ok = True
    if missing:
        ok = False
        lines.append(f"MISSING {len(missing)} row(s) (Method,Condition,Seed,MaxChars):")
        for m, c, s, mc in missing:
            reason = fmap.get((m, c, s, mc))
            suffix = f"  [previously failed: {reason}]" if reason else ""
            lines.append(f"  - {m} | {c} | seed={s} | max_chars={mc}{suffix}")
    if untrainable:
        ok = False
        lines.append(f"UNTRAINABLE/failed {len(untrainable)} row(s) present:")
        for m, c, s, mc in untrainable:
            lines.append(f"  - {m} | {c} | seed={s} | max_chars={mc}")
    if ferr:
        lines.append(f"NOTE failures sidecar unreadable: {ferr}")
    return ok, lines


def check_stability(path, seeds=SEEDS_30, methods=STAB_METHODS):
    lines = []
    if not os.path.isfile(path):
        return False, [f"MISSING FILE: {path}"]
    df = pd.read_csv(path)
    ds_label = f"{STAB_PAIR[0]}->{STAB_PAIR[1]}"

    good, untrainable = set(), []
    for _, r in df.iterrows():
        if r['Dataset'] != ds_label:
            continue
        key = (r['Method'], int(r['Seed']))
        if _row_trained(r.get('Notes'), r.get('F1Binary')):
            good.add(key)
        else:
            untrainable.append(key)

    expected = [(m, int(s)) for m in methods for s in seeds]
    missing = [k for k in expected if k not in good]
    lines.append(f"expected {len(expected)} rows "
                 f"({len(methods)} methods x {len(seeds)} seeds) for {ds_label}; "
                 f"found {len(good)} completed")
    fmap, ferr = _failure_map(path, lambda row: (row['Method'], int(row['Seed'])))
    ok = True
    if missing:
        ok = False
        lines.append(f"MISSING {len(missing)} row(s) (Method,Seed):")
        for m, s in missing:
            reason = fmap.get((m, s))
            suffix = f"  [previously failed: {reason}]" if reason else ""
            lines.append(f"  - {m} | seed={s}{suffix}")
    if untrainable:
        ok = False
        lines.append(f"UNTRAINABLE/failed {len(untrainable)} row(s) present:")
        for m, s in untrainable:
            lines.append(f"  - {m} | seed={s}")
    if ferr:
        lines.append(f"NOTE failures sidecar unreadable: {ferr}")
    return ok, lines


def check_memory(path, methods=MEM_METHODS):
    lines = []
    if not os.path.isfile(path):
        return False, [f"MISSING FILE: {path}"]
    df = pd.read_csv(path)
    present = {r['Method']: r for _, r in df.iterrows()}

    missing = [m for m in methods if m not in present]
    lines.append(f"expected {len(methods)} configs; found {len(present)}")
    ok = True
    if missing:
        ok = False
        fmap, ferr = _failure_map(path, lambda row: row['Method'])
        lines.append(f"MISSING {len(missing)} config(s):")
        for m in missing:
            reason = fmap.get(m)
            suffix = f"  [previously failed: {reason}]" if reason else ""
            lines.append(f"  - {m}{suffix}")
        if ferr:
            lines.append(f"NOTE failures sidecar unreadable: {ferr}")
    for m in methods:
        if m not in present:
            continue
        r = present[m]
        seqlen, req = r.get('SeqLen'), r.get('RequestedSeqLen')
        try:
            sl = int(seqlen)
        except (TypeError, ValueError):
            ok = False
            lines.append(f"  - {m}: SeqLen is not a positive integer ({seqlen!r})")
            continue
        if sl <= 0:
            ok = False
            lines.append(f"  - {m}: SeqLen must be positive (got {sl})")
            continue
        try:
            rq = int(req)
        except (TypeError, ValueError):
            rq = None
        if rq is not None and sl > rq:
            ok = False
            lines.append(f"  - {m}: SeqLen {sl} exceeds RequestedSeqLen {rq} "
                         f"(impossible / inconsistent)")
        elif rq is not None and sl < rq:
            lines.append(f"  - {m}: NOTE SeqLen {sl} < requested {rq} "
                         f"(model truncation cap; recorded truthfully)")
    return ok, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=R2_DIR)
    parser.add_argument('--length_confound', type=str, default=None)
    parser.add_argument('--stability', type=str, default=None)
    parser.add_argument('--memory', type=str, default=None)
    cli_args = parser.parse_args()

    lc = cli_args.length_confound or os.path.join(cli_args.results_dir,
                                                  'length_confound.csv')
    st = cli_args.stability or os.path.join(cli_args.results_dir,
                                            'stability_30seed.csv')
    mem = cli_args.memory or os.path.join(cli_args.results_dir,
                                          'memory_profile.csv')

    checks = [('length_confound.csv', check_length_confound, lc),
              ('stability_30seed.csv', check_stability, st),
              ('memory_profile.csv', check_memory, mem)]

    all_ok = True
    for name, fn, path in checks:
        ok, lines = fn(path)
        status = 'PASS' if ok else 'FAIL'
        print(f"\n[{status}] {name}  ({path})")
        for ln in lines:
            print(f"    {ln}")
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print("COVERAGE: PASS — all three R2 files are complete.")
        sys.exit(0)
    print("COVERAGE: FAIL — see the holes listed above. Delete+rerun the "
          "affected configs (OOM rows live in the *_failures.csv and are "
          "retried automatically on rerun).")
    sys.exit(1)


if __name__ == '__main__':
    main()
