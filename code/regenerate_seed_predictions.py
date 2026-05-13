#!/usr/bin/env python3
"""Regenerate seed_predictions.csv (and per-seed splits) from the canonical
5-seed CV cache, applying NHNN de-identification.

This is the *internal* bridge script that reads the canonical cache file
(`data/cv_cache_backup/nested_cv_results_FULL_multi_5e480eef.pkl.original`)
and emits the public-bundleable seed_predictions.csv inputs that every
other downstream script consumes. The cache file is not part of the
public bundle (it is large and binary); this script makes the CSVs
reproducible from it.

NHNN de-identification: original IDs are of the form
`NHNN_<code>_<YYYY>_<Mon>` where the date suffix is the scan date — a
direct identifier under HIPAA-style criteria. We replace the date with a
chronological scan-rank suffix that preserves uniqueness without
revealing dates: `NHNN_<code>_scan<N>` where N=1 is the earliest scan
of that patient code, N=2 the next, etc. Single-scan codes get
`NHNN_<code>_scan1`. The mapping is deterministic (chronological order
of dates within each code), so it produces stable IDs across runs and
matches the scheme already used in `radiologist_df.csv` and other
already-stripped bundle files.

Public-dataset case IDs (BraTS, EGD, UCSF-PDGM, UPENN-GBM) are passed
through unchanged — those datasets are already de-identified by
construction.

Usage: `python3 code/regenerate_seed_predictions.py`

Inputs:  data/cv_cache_backup/nested_cv_results_FULL_multi_5e480eef.pkl.original
Outputs: data/source_data/figure_1/csv_v2/seed_predictions.csv (5,500 rows)
         data/source_data/figure_1/csv_v2/seed_predictions/seed_<N>.csv (×5)
"""
import os
import re
import datetime
from collections import defaultdict

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..'))
CACHE_PATH = os.path.join(
    R1_ROOT, 'data', 'cv_cache_backup',
    'nested_cv_results_FULL_multi_5e480eef.pkl.original',
)
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
PER_SEED_DIR = os.path.join(OUT_DIR, 'seed_predictions')

NHNN_DATE_RE = re.compile(r'^NHNN_(\d+)_(\d{4})_([A-Z][a-z]+)$')

COLS = [
    'fold', 'case_id', 'radiologist', 'pathology', 'cohort',
    'volume', 'radiomic_category', 'model_confidence',
    'gt', 'model_pred', 'cv_pred',
    'model_prob', 'combined_prob',
    'human_prob', 'human_confidence', 'human_agreement', 'used_human',
]


def build_nhnn_anon_mapping(all_case_ids):
    """De-identify NHNN IDs by stripping date suffixes.

    For multi-scan patient codes (≥2 dates per code), assign chronological
    scan-rank suffixes (NHNN_<code>_scan1, scan2, ...) to preserve scan-
    level uniqueness. For single-scan codes, use plain NHNN_<code> form
    (no suffix). This matches the canonical scheme already in
    radiologist_df.csv: 21 multi-scan codes (46 IDs total) plus 162
    single-scan codes (162 IDs) — 208 unique IDs across 183 patient codes.

    Returns dict[old_id -> new_id], including only NHNN entries.
    """
    by_code = defaultdict(list)
    for cid in set(all_case_ids):
        m = NHNN_DATE_RE.match(cid)
        if not m:
            continue
        code, year, mon = m.group(1), m.group(2), m.group(3)
        by_code[code].append((f'{year}_{mon}', cid))

    mapping = {}
    for code, entries in by_code.items():
        if len(entries) == 1:
            mapping[entries[0][1]] = f'NHNN_{code}'
        else:
            entries_sorted = sorted(
                entries,
                key=lambda x: datetime.datetime.strptime(x[0], '%Y_%b'),
            )
            for rank, (_date, orig) in enumerate(entries_sorted, start=1):
                mapping[orig] = f'NHNN_{code}_scan{rank}'
    return mapping


def main():
    print('─' * 78)
    print('Regenerate seed_predictions.csv from canonical cache (with NHNN de-id)')
    print('─' * 78)
    print(f'Source: {os.path.relpath(CACHE_PATH, R1_ROOT)}')

    cache = joblib.load(CACHE_PATH)
    seeds = [s['seed'] for s in cache['seed_results']]
    print(f'Seeds: {seeds}')

    all_ids = []
    for sd in cache['seed_results']:
        for p in sd['predictions']:
            all_ids.append(p['case_id'])

    nhnn_mapping = build_nhnn_anon_mapping(all_ids)
    n_codes = len({m.split('_scan')[0] for m in nhnn_mapping.values()})
    print(f'NHNN de-id mapping: {len(nhnn_mapping)} entries '
          f'(unique patient codes: {n_codes})')

    rows = []
    for sd in cache['seed_results']:
        seed = sd['seed']
        for p in sd['predictions']:
            r = {'seed': seed}
            for c in COLS:
                r[c] = p.get(c)
            r['case_id'] = nhnn_mapping.get(r['case_id'], r['case_id'])
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f'Combined dataframe: {len(df)} rows × {len(df.columns)} cols')

    n_unique = df['case_id'].nunique()
    n_pairs = df.groupby(['case_id', 'radiologist']).ngroups
    print(f'  unique case_ids: {n_unique}  (expect 564)')
    print(f'  unique (case, rad) pairs: {n_pairs}  (expect 1100)')
    assert n_unique == 564, f'case-id collision after de-id (got {n_unique}, expected 564)'
    assert n_pairs == 1100, f'(case, rad) pair count changed (got {n_pairs}, expected 1100)'

    out = os.path.join(OUT_DIR, 'seed_predictions.csv')
    df.to_csv(out, index=False, float_format='%.10g')
    print(f'Saved: {os.path.relpath(out, R1_ROOT)} ({os.path.getsize(out)/1024:.1f} KB)')

    os.makedirs(PER_SEED_DIR, exist_ok=True)
    for seed in seeds:
        sub = df[df['seed'] == seed].drop(columns=['seed'])
        p = os.path.join(PER_SEED_DIR, f'seed_{seed}.csv')
        sub.to_csv(p, index=False, float_format='%.10g')
    print(f'Saved {len(seeds)} per-seed files in {os.path.relpath(PER_SEED_DIR, R1_ROOT)}/')

    print('─' * 78)
    print('Next: run code/generate_aggregates.py to regenerate aggregates.json + '
          'best_cv_predictions.csv from the new seed_predictions.csv.')


if __name__ == '__main__':
    main()
