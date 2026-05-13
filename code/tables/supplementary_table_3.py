#!/usr/bin/env python3
"""Supplementary Table 3: Patient-sex disaggregation of agent performance.

Per-sex (Female, Male) point values for each agent × support condition on the
82-case sex-metadata subset (37 female, 45 male).

Inputs:  data/source_data/figure_1/csv_v2/{radiologist_df.csv,
                                              best_cv_predictions.csv,
                                              sex_metadata.csv}
Outputs: stdout printout + data/source_data/supplementary_table_3/
                            csv/supplementary_table_3.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'supplementary_table_3', 'csv')
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


def compute_sex_metrics(radiologist_df, bcv_df, sex_meta):
    """Return dict[(agent, condition, sex)] = {metric: value}."""
    rad_sex = radiologist_df.merge(sex_meta[['case_id', 'Sex']], on='case_id', how='inner')
    bcv_sex = bcv_df.merge(sex_meta[['case_id', 'Sex']], on='case_id', how='inner')

    out = {}
    for sex in ['Female', 'Male']:
        for ws, label in [(False, 'Radiologist (Without model)'),
                          (True,  'Radiologist (With model)')]:
            sub = rad_sex[(rad_sex['with_segmentation'] == ws) & (rad_sex['Sex'] == sex)]
            if len(sub) == 0:
                continue
            gt = sub['has_enhancement_gt'].astype(int).values
            pred = sub['predicted_enhancement'].astype(int).values
            score = np.where(pred == 1, sub['confidence'] / 10.0,
                             1 - sub['confidence'] / 10.0).astype(float)
            n, tp, fp, tn, fn, ba, sens, spec, prec, f1 = _sg_metrics(gt, pred)
            try:
                au = roc_auc_score(gt, score); ap = average_precision_score(gt, score)
            except Exception:
                au = ap = float('nan')
            acc = (tp + tn) / n if n else float('nan')
            out[('Radiologist', label, sex)] = dict(
                n=n, tp=tp, fp=fp, tn=tn, fn=fn,
                acc=acc, ba=ba, auroc=au, auprc=ap,
                sens=sens, spec=spec, prec=prec, f1=f1,
            )
    for sex in ['Female', 'Male']:
        sub_uniq = bcv_sex.drop_duplicates('case_id')
        sub_sex = sub_uniq[sub_uniq['Sex'] == sex]
        if len(sub_sex) == 0:
            continue
        gt = sub_sex['gt'].astype(int).values
        # Model alone
        m_pred = sub_sex['model_pred'].astype(int).values
        m_prob = sub_sex['model_prob'].values
        n, tp, fp, tn, fn, ba, sens, spec, prec, f1 = _sg_metrics(gt, m_pred)
        try:
            au = roc_auc_score(gt, m_prob); ap = average_precision_score(gt, m_prob)
        except Exception:
            au = ap = float('nan')
        acc = (tp + tn) / n if n else float('nan')
        out[('Model', 'Model (Without radiologist)', sex)] = dict(
            n=n, tp=tp, fp=fp, tn=tn, fn=fn,
            acc=acc, ba=ba, auroc=au, auprc=ap,
            sens=sens, spec=spec, prec=prec, f1=f1,
        )
        # Model + rad
        c_pred = sub_sex['cv_pred'].astype(int).values
        c_prob = sub_sex['combined_prob'].values
        n, tp, fp, tn, fn, ba, sens, spec, prec, f1 = _sg_metrics(gt, c_pred)
        try:
            au = roc_auc_score(gt, c_prob); ap = average_precision_score(gt, c_prob)
        except Exception:
            au = ap = float('nan')
        acc = (tp + tn) / n if n else float('nan')
        out[('Model', 'Model (With radiologist)', sex)] = dict(
            n=n, tp=tp, fp=fp, tn=tn, fn=fn,
            acc=acc, ba=ba, auroc=au, auprc=ap,
            sens=sens, spec=spec, prec=prec, f1=f1,
        )
    return out


def main():
    radiologist_df = pd.read_csv(
        os.path.join(SRC_DIR, 'radiologist_df.csv'), float_precision='round_trip'
    ).astype({'with_segmentation': bool})
    bcv_df = pd.read_csv(
        os.path.join(SRC_DIR, 'best_cv_predictions.csv'), float_precision='round_trip'
    )
    sex_meta = pd.read_csv(os.path.join(SRC_DIR, 'sex_metadata.csv'))

    n_total = len(sex_meta)
    n_female = int((sex_meta['Sex'] == 'Female').sum())
    n_male = int((sex_meta['Sex'] == 'Male').sum())
    n_rad_cases = int(radiologist_df['case_id'].nunique())
    print("─" * 78)
    print("Supplementary Table 3: Patient-sex disaggregation of agent performance")
    print("─" * 78)
    print(f"Sex metadata available for {n_total} cases (Female {n_female}, Male {n_male})")
    print(f"  Coverage of {n_rad_cases}-case radiologist subset: "
          f"{n_total}/{n_rad_cases} ({n_total/n_rad_cases*100:.1f}%)")

    metrics = compute_sex_metrics(radiologist_df, bcv_df, sex_meta)

    print(f"\n{'Agent':<12} {'Condition':<28} {'Sex':<8} {'N':>4} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  "
          f"{'Acc':>6} {'BalAcc':>6} {'AUROC':>6} {'AUPRC':>6} "
          f"{'Sens':>6} {'Spec':>6} {'Prec':>6} {'F1':>6}")

    rows = []
    order = [
        ('Radiologist', 'Radiologist (Without model)', 'Female'),
        ('Radiologist', 'Radiologist (Without model)', 'Male'),
        ('Radiologist', 'Radiologist (With model)',    'Female'),
        ('Radiologist', 'Radiologist (With model)',    'Male'),
        ('Model',       'Model (Without radiologist)', 'Female'),
        ('Model',       'Model (Without radiologist)', 'Male'),
        ('Model',       'Model (With radiologist)',    'Female'),
        ('Model',       'Model (With radiologist)',    'Male'),
    ]
    for agent, cond, sex in order:
        m = metrics.get((agent, cond, sex))
        if m is None:
            continue
        print(f"{agent:<12} {cond:<28} {sex:<8} {m['n']:>4} "
              f"{m['tp']:>3} {m['fp']:>3} {m['tn']:>3} {m['fn']:>3}  "
              f"{m['acc']:>6.3f} {m['ba']:>6.3f} {m['auroc']:>6.3f} {m['auprc']:>6.3f} "
              f"{m['sens']:>6.3f} {m['spec']:>6.3f} {m['prec']:>6.3f} {m['f1']:>6.3f}")
        # Use printf-style %.3f formatting (half-up) so the CSV/docx values
        # match the log's f-string formatting; Python's built-in round() uses
        # banker's rounding which differs from %.3f on x.xxx5 values (e.g.
        # 0.7125 → round() yields 0.712 but %.3f yields 0.713).
        rows.append({
            'Agent': agent, 'Condition': cond, 'Sex': sex,
            'N': m['n'], 'TP': m['tp'], 'FP': m['fp'], 'TN': m['tn'], 'FN': m['fn'],
            'Balanced accuracy': f"{m['ba']:.3f}",
            'AUROC':       f"{m['auroc']:.3f}",
            'AUPRC':       f"{m['auprc']:.3f}",
            'Sensitivity': f"{m['sens']:.3f}",
            'Specificity': f"{m['spec']:.3f}",
            'Precision':   f"{m['prec']:.3f}",
            'F1 score':    f"{m['f1']:.3f}",
        })
    print("─" * 78)

    out_csv = os.path.join(OUT_DIR, 'supplementary_table_3.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
