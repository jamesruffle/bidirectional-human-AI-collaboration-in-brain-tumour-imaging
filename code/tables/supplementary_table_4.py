#!/usr/bin/env python3
"""Supplementary Table 4: Female–male performance gaps Δ = female − male.

For each agent × support condition, reports the difference in each metric
(BA, AUROC, AUPRC, sensitivity, specificity, precision, F1) between the
female (n=37) and male (n=45) subsets reported in Supplementary Table 3.
Positive Δ indicates higher performance on the female subset; negative Δ
indicates higher performance on the male subset.

Reuses compute_sex_metrics() from supplementary_table_3.py to ensure both
tables are generated from the same source-of-truth computation.

Inputs:  data/source_data/figure_1/csv_v2/{radiologist_df.csv,
                                              best_cv_predictions.csv,
                                              sex_metadata.csv}
Outputs: stdout printout + data/source_data/supplementary_table_4/
                            csv/supplementary_table_4.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supplementary_table_3 import compute_sex_metrics  # noqa: E402

R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'supplementary_table_4', 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    radiologist_df = pd.read_csv(
        os.path.join(SRC_DIR, 'radiologist_df.csv'), float_precision='round_trip'
    ).astype({'with_segmentation': bool})
    bcv_df = pd.read_csv(
        os.path.join(SRC_DIR, 'best_cv_predictions.csv'), float_precision='round_trip'
    )
    sex_meta = pd.read_csv(os.path.join(SRC_DIR, 'sex_metadata.csv'))

    metrics = compute_sex_metrics(radiologist_df, bcv_df, sex_meta)

    print("─" * 78)
    print("Supplementary Table 4: Female–male performance gaps (Δ = female − male)")
    print("─" * 78)

    conditions = [
        ('Radiologist', 'Radiologist (Without model)'),
        ('Radiologist', 'Radiologist (With model)'),
        ('Model',       'Model (Without radiologist)'),
        ('Model',       'Model (With radiologist)'),
    ]
    metric_keys = ['ba', 'auroc', 'auprc', 'sens', 'spec', 'prec', 'f1']
    metric_labels = ['ΔBA', 'ΔAUROC', 'ΔAUPRC', 'ΔSens', 'ΔSpec', 'ΔPrec', 'ΔF1']
    pretty = {
        'ba':    'Δ Balanced accuracy',
        'auroc': 'Δ AUROC',
        'auprc': 'Δ AUPRC',
        'sens':  'Δ Sensitivity',
        'spec':  'Δ Specificity',
        'prec':  'Δ Precision',
        'f1':    'Δ F1 score',
    }

    print(f"{'Agent':<12} {'Condition':<32} " +
          "".join(f"{lab:>10}" for lab in metric_labels))

    rows = []
    for agent, cond in conditions:
        female = metrics.get((agent, cond, 'Female'))
        male = metrics.get((agent, cond, 'Male'))
        if female is None or male is None:
            print(f"{agent:<12} {cond:<32}  (skipped — missing female or male)")
            continue
        deltas = [female[k] - male[k] for k in metric_keys]
        print(f"{agent:<12} {cond:<32} " +
              "".join(f"{d:>+10.3f}" for d in deltas))
        # Emit deltas as printf-formatted strings with explicit '+' on
        # positive values; preserves trailing zeros and matches the docx
        # Supplementary Table 4 cells exactly (e.g. '+0.004', '-0.030').
        rows.append({
            'Agent': agent, 'Condition': cond,
            **{pretty[k]: f"{d:+.3f}" for k, d in zip(metric_keys, deltas)},
        })
    print("─" * 78)

    out_csv = os.path.join(OUT_DIR, 'supplementary_table_4.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
