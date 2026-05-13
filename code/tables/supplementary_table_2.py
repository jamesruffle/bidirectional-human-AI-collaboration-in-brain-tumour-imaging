#!/usr/bin/env python3
"""Supplementary Table 2: False-positive and false-negative geography
and pathology distribution.

Per-condition × country × pathology breakdown of model and radiologist
confusion-matrix counts and accuracy across the four conditions
(Radiologist alone, Radiologist with model, Model alone, Model with rad).
Filter (country, pathology) groups to those with >=10 cases per arm to
match the table thresholds.

Inputs:  data/source_data/figure_1/csv_v2/{radiologist_df.csv,
                                              best_cv_predictions.csv}
Outputs: stdout printout + data/source_data/supplementary_table_2/
                            csv/supplementary_table_2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'supplementary_table_2', 'csv')
os.makedirs(OUT_DIR, exist_ok=True)


def _sg_metrics(gt, pred):
    """Return (n, tp, fp, tn, fn, bal_acc, sens, spec, prec, f1)."""
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    n = tp + fp + tn + fn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    bal_acc = (sens + spec) / 2
    return n, tp, fp, tn, fn, bal_acc, sens, spec, prec, f1


def main():
    radiologist_df = pd.read_csv(
        os.path.join(SRC_DIR, 'radiologist_df.csv'), float_precision='round_trip'
    ).astype({'with_segmentation': bool})
    bcv_df = pd.read_csv(
        os.path.join(SRC_DIR, 'best_cv_predictions.csv'), float_precision='round_trip'
    )

    print("─" * 78)
    print("Supplementary Table 2: Country × Pathology subgroup metrics")
    print("─" * 78)
    print(f"{'Condition':<32} {'Country':<22} {'Pathology':<32} {'N':>4} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} "
          f"{'BalAcc':>7} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'F1':>6}")

    # Order matches the published Supplementary Table 2 row layout:
    # Model (Without radiologist) → Model (With radiologist) →
    # Radiologist (With model) → Radiologist (Without model).
    arm_specs = [
        ('Model (Without radiologist)',
         radiologist_df.dropna(subset=['model_predicted_enhancement']).drop_duplicates('case_id'),
         'has_enhancement_gt', 'model_predicted_enhancement'),
        ('Model (With radiologist)', None, None, None),
        ('Radiologist (With model)',
         radiologist_df[radiologist_df['with_segmentation'] == True],
         'has_enhancement_gt', 'predicted_enhancement'),
        ('Radiologist (Without model)',
         radiologist_df[radiologist_df['with_segmentation'] == False],
         'has_enhancement_gt', 'predicted_enhancement'),
    ]

    cvw_unique = bcv_df.drop_duplicates('case_id').merge(
        radiologist_df[['case_id', 'Country', 'Pathology']].drop_duplicates('case_id'),
        on='case_id', how='left',
    )

    rows = []
    for cond, df_arm, gt_col, pred_col in arm_specs:
        if cond == 'Model (With radiologist)':
            df_arm = cvw_unique
            gt_col = 'gt'
            pred_col = 'cv_pred'
        if df_arm is None or 'Country' not in df_arm.columns:
            continue
        for (country, pathology), group in df_arm.groupby(['Country', 'Pathology'], observed=True):
            if len(group) < 10:
                continue
            gt = group[gt_col].values
            pred = group[pred_col].values
            n, tp, fp, tn, fn, ba, sens, spec, prec, f1 = _sg_metrics(gt, pred)
            print(f"{cond:<32} {country:<22} {pathology:<32} {n:>4} "
                  f"{tp:>3} {fp:>3} {tn:>3} {fn:>3} "
                  f"{ba:>7.3f} {sens:>6.3f} {spec:>6.3f} {prec:>6.3f} {f1:>6.3f}")
            # Emit as printf-formatted strings (not floats) so trailing zeros
            # are preserved in the CSV and match the docx Supplementary Table 2
            # cell formatting exactly (e.g. '0.810', not '0.81').
            rows.append({
                'Condition': cond, 'Country': country, 'Pathology': pathology,
                'N cases': n, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'Balanced accuracy': f"{ba:.3f}",
                'Sensitivity':       f"{sens:.3f}",
                'Specificity':       f"{spec:.3f}",
                'Precision':         f"{prec:.3f}",
                'F1 score':          f"{f1:.3f}",
            })
    print("─" * 78)

    out_csv = os.path.join(OUT_DIR, 'supplementary_table_2.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
