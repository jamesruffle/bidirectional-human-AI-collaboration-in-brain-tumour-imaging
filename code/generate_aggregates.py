#!/usr/bin/env python3
"""Generate aggregates.json and best_cv_predictions.csv from the bundled
seed-prediction CSV using the documented 5-seed optimistic-dedup,
prefer-correct tie-break procedure.

This script makes every model-side metric in Table 1 / Figure 1 a
*derived* artifact emitted by a runnable Python file from a CSV input,
satisfying the "no hardcoded literals" reproducibility criterion. After
this script runs, fig_1.py / table_1.py inputs are internally consistent
and traceable to a single computation on bundled data.

Input (bundled, public, plain-text):
  data/source_data/figure_1/csv_v2/seed_predictions.csv  (5,500 rows)
    = 5 random seeds × 1,100 (case_id, radiologist) pair predictions.
    The seeds are exported from the canonical 5-seed cross-validation
    run; the CSV is the public-shareable form of those predictions.

Procedure (mirrors `multi_radiologist_analysis.py:32070-32287`):

1. Read seed_predictions.csv (5,500 rows).
2. For each unique (case_id, radiologist) pair, dedup across the 5 seeds
   by preferring the correct cv_pred (prefer-correct tie-break).
3. Compute pair-level metrics (acc / prec / sens / spec / f1 / AUROC /
   AUPRC) from the deduped 1,100 predictions.
4. Compute case-level (n=564) AUROC / AUPRC by mean-aggregating model_prob
   and combined_prob across all reviews of each case.

Outputs:
  data/source_data/figure_1/csv_v2/aggregates.json   (derived; keys
    mirror prior schema: pair_level_model_metrics, pair_level_cv_metrics,
    unique_case_model_metrics, unique_case_cv_metrics)
  data/source_data/figure_1/csv_v2/best_cv_predictions.csv  (derived;
    1,100 deduped pair-level predictions)

Usage: `python3 code/generate_aggregates.py`
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SEED_CSV_PATH = os.path.join(
    R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2', 'seed_predictions.csv',
)
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')


def optimistic_dedup_seed_predictions(df_seeds):
    """Apply master's prefer-correct dedup across all seeds.

    For each unique (case_id, radiologist) pair, gather rows from every
    seed and pick one: prefer rows where cv_pred == gt; otherwise take
    the first row encountered (matches `multi_radiologist_analysis.py`
    line 32079-32095 logic).

    Returns a DataFrame of 1,100 deduped pair-level predictions.
    """
    pair_buckets = defaultdict(list)
    for _, row in df_seeds.iterrows():
        key = (row['case_id'], row['radiologist'])
        pair_buckets[key].append(row)
    deduped = []
    for k, rows in pair_buckets.items():
        if len(rows) == 1:
            deduped.append(rows[0])
            continue
        correct = [r for r in rows if int(r['cv_pred']) == int(r['gt'])]
        deduped.append(correct[0] if correct else rows[0])
    return pd.DataFrame(deduped).reset_index(drop=True)


def metrics_dict(gt, pred, prob=None):
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    cm = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel()
    n = tp + fp + tn + fn
    out = {
        'accuracy':    float(accuracy_score(gt, pred)),
        'precision':   float(precision_score(gt, pred, zero_division=0)),
        'recall':      float(recall_score(gt, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1':          float(f1_score(gt, pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n_pairs': int(n),
    }
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        if len(set(gt.tolist())) > 1:
            out['auroc'] = float(roc_auc_score(gt, prob))
            out['auprc'] = float(average_precision_score(gt, prob))
        else:
            out['auroc'] = float('nan')
            out['auprc'] = float('nan')
    return out


def main():
    print('─' * 78)
    print('Generating aggregates.json + best_cv_predictions.csv from seed CSV')
    print('─' * 78)
    print(f'Source: {os.path.relpath(SEED_CSV_PATH, R1_ROOT)}')

    df_seeds = pd.read_csv(SEED_CSV_PATH, float_precision='round_trip')
    seeds = sorted(df_seeds['seed'].unique().tolist())
    print(f'Seeds in CSV: {seeds}')
    print(f'Total seed-prediction rows: {len(df_seeds)}')

    df_dedup = optimistic_dedup_seed_predictions(df_seeds)
    print(f'After 5-seed optimistic-dedup: {len(df_dedup)} pairs')

    gt    = df_dedup['gt'].astype(int).values
    cv    = df_dedup['cv_pred'].astype(int).values
    mp    = df_dedup['model_pred'].astype(int).values
    cprob = df_dedup['combined_prob'].astype(float).values
    mprob = df_dedup['model_prob'].astype(float).values

    pair_level_cv_metrics    = metrics_dict(gt, cv, prob=cprob)
    pair_level_model_metrics = metrics_dict(gt, mp, prob=mprob)
    print(f"Model alone:  acc={pair_level_model_metrics['accuracy']:.4f}, "
          f"TP/FP/TN/FN={pair_level_model_metrics['tp']}/"
          f"{pair_level_model_metrics['fp']}/{pair_level_model_metrics['tn']}/"
          f"{pair_level_model_metrics['fn']}")
    print(f"Model + Rad:  acc={pair_level_cv_metrics['accuracy']:.4f}, "
          f"TP/FP/TN/FN={pair_level_cv_metrics['tp']}/"
          f"{pair_level_cv_metrics['fp']}/{pair_level_cv_metrics['tn']}/"
          f"{pair_level_cv_metrics['fn']}")

    # Case-level probability aggregation uses ALL 5,500 seed-prediction rows
    # (not the 1,100 deduped subset) — mean across every seed × radiologist
    # review of each case. This is the canonical 5-seed ensemble used in the
    # published unique_case_cv_metrics, and is independent of the dedup
    # choice which only affects pair-level binary metrics.
    cdf = (
        df_seeds
        .groupby('case_id')
        .agg(gt=('gt', 'first'),
             model_prob=('model_prob', 'mean'),
             combined_prob=('combined_prob', 'mean'))
        .reset_index()
    )
    unique_case_model_metrics = {
        'auroc': float(roc_auc_score(cdf['gt'], cdf['model_prob'])),
        'auprc': float(average_precision_score(cdf['gt'], cdf['model_prob'])),
        'n_unique_cases': int(len(cdf)),
    }
    unique_case_cv_metrics = {
        'auroc': float(roc_auc_score(cdf['gt'], cdf['combined_prob'])),
        'auprc': float(average_precision_score(cdf['gt'], cdf['combined_prob'])),
        'n_unique_cases': int(len(cdf)),
    }
    print(f'Case-level (n={len(cdf)}):  '
          f"Model AUROC={unique_case_model_metrics['auroc']:.4f}, "
          f"Model+Rad AUROC={unique_case_cv_metrics['auroc']:.4f}")

    # Build aggregates.json — preserve other top-level keys (group1_avg /
    # group2_avg / mrmc_*) carried forward from existing file when present.
    out_json_path = os.path.join(OUT_DIR, 'aggregates.json')
    old_obj = {}
    if os.path.isfile(out_json_path):
        with open(out_json_path) as fh:
            old_obj = json.load(fh)
    old_data = old_obj.get('data', {})

    new_data = dict(old_data)
    new_data['pair_level_model_metrics']  = pair_level_model_metrics
    new_data['pair_level_cv_metrics']     = pair_level_cv_metrics
    new_data['unique_case_model_metrics'] = unique_case_model_metrics
    new_data['unique_case_cv_metrics']    = unique_case_cv_metrics

    out_obj = {'data': new_data}
    if 'types' in old_obj:
        out_obj['types'] = old_obj['types']

    with open(out_json_path, 'w') as fh:
        json.dump(out_obj, fh, indent=2)
    print(f'Saved: {os.path.relpath(out_json_path, R1_ROOT)}')

    out_cols = [
        'fold', 'case_id', 'radiologist', 'pathology', 'cohort',
        'volume', 'radiomic_category', 'model_confidence',
        'gt', 'model_pred', 'cv_pred',
        'model_prob', 'combined_prob',
        'human_prob', 'human_confidence', 'human_agreement', 'used_human',
    ]
    out_csv_path = os.path.join(OUT_DIR, 'best_cv_predictions.csv')
    df_dedup[[c for c in out_cols if c in df_dedup.columns]].to_csv(
        out_csv_path, index=False, float_format='%.10g',
    )
    print(f'Saved: {os.path.relpath(out_csv_path, R1_ROOT)}')

    print('─' * 78)
    print('Done. Re-run fig_1.py / table_1.py to regenerate downstream outputs.')


if __name__ == '__main__':
    main()
