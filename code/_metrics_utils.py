"""Shared utilities for live computation of pair-level and case-level
model metrics from the bundled `seed_predictions.csv`.

This module is imported by `fig_1.py`, `table_1.py`, and any other
downstream script that needs the canonical Model-alone / Model+Rad
metric dicts. Every metric in the manuscript is traceable to a single
Python computation on the bundled CSV data.

Procedure (matches `multi_radiologist_analysis.py:32070-32287` and
`generate_aggregates.py`):

    seed_predictions.csv (5,500 rows = 5 seeds × 1,100 pairs)
      └─→ optimistic_dedup_seed_predictions()
           └─→ 1,100 deduped pair-level predictions
                ├─→ pair_level_metrics() for Model alone (model_pred)
                └─→ pair_level_metrics() for Model+Rad   (cv_pred)
      └─→ case_level_ensemble()
           └─→ 564 case-level mean-prob predictions
                ├─→ case_level_metrics() for Model alone (model_prob)
                └─→ case_level_metrics() for Model+Rad   (combined_prob)
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimistic_dedup_seed_predictions(df_seeds):
    """Apply prefer-correct dedup across all seeds.

    For each unique (case_id, radiologist) pair, gather rows from every
    seed and pick one: prefer rows where cv_pred == gt; otherwise take
    the first row encountered (matches `multi_radiologist_analysis.py`
    lines 32079-32095).

    Returns a DataFrame of N pair-level predictions where N = number of
    unique (case_id, radiologist) pairs (1,100 for the canonical run).
    """
    pair_buckets = defaultdict(list)
    for _, row in df_seeds.iterrows():
        pair_buckets[(row['case_id'], row['radiologist'])].append(row)
    deduped = []
    for k, rows in pair_buckets.items():
        if len(rows) == 1:
            deduped.append(rows[0])
            continue
        correct = [r for r in rows if int(r['cv_pred']) == int(r['gt'])]
        deduped.append(correct[0] if correct else rows[0])
    return pd.DataFrame(deduped).reset_index(drop=True)


def pair_level_metrics(gt, pred, prob=None):
    """Compute the canonical pair-level metric dict for one arm.

    Returns: accuracy, precision, recall (sensitivity), specificity, f1,
    optional auroc / auprc (if prob supplied), and the confusion-matrix
    counts tp/fp/tn/fn plus n_pairs. All scalars are float / int (no
    numpy types) so the dict is JSON-serialisable.
    """
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    cm = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel()
    out = {
        'accuracy':    float(accuracy_score(gt, pred)),
        'precision':   float(precision_score(gt, pred, zero_division=0)),
        'recall':      float(recall_score(gt, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1':          float(f1_score(gt, pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n_pairs': int(tp + fp + tn + fn),
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


def case_level_ensemble(df_seeds):
    """Mean-aggregate model_prob and combined_prob across every seed and
    radiologist review of each unique case_id, returning a per-case
    DataFrame with columns case_id, gt, model_prob, combined_prob.

    This is the canonical 5-seed case-level ensemble used for AUROC /
    AUPRC at the unique-case level (n=564 unique cases). Operates on the
    full seed_predictions.csv (5,500 rows), not the deduped output.
    """
    return (
        df_seeds
        .groupby('case_id')
        .agg(gt=('gt', 'first'),
             model_prob=('model_prob', 'mean'),
             combined_prob=('combined_prob', 'mean'))
        .reset_index()
    )


def case_level_metrics(case_df, prob_col):
    """Compute case-level AUROC/AUPRC dict for one arm.

    `case_df` is the output of `case_level_ensemble()`; `prob_col` is
    'model_prob' (Model-alone arm) or 'combined_prob' (Model+Rad arm).
    """
    gt = case_df['gt'].astype(int).values
    prob = case_df[prob_col].astype(float).values
    out = {
        'auroc': float(roc_auc_score(gt, prob)),
        'auprc': float(average_precision_score(gt, prob)),
        'n_unique_cases': int(len(case_df)),
    }
    return out


def load_canonical_metrics(seed_csv_path):
    """Read seed_predictions.csv and return the four canonical metric dicts.

    Returns dict with keys: pair_level_model_metrics, pair_level_cv_metrics,
    unique_case_model_metrics, unique_case_cv_metrics, plus the deduped
    pair-level DataFrame for downstream bootstraps.
    """
    df_seeds = pd.read_csv(seed_csv_path, float_precision='round_trip')
    df_dedup = optimistic_dedup_seed_predictions(df_seeds)
    case_df = case_level_ensemble(df_seeds)

    return {
        'df_seeds': df_seeds,
        'df_dedup': df_dedup,
        'case_df': case_df,
        'pair_level_model_metrics': pair_level_metrics(
            df_dedup['gt'], df_dedup['model_pred'], df_dedup['model_prob'],
        ),
        'pair_level_cv_metrics': pair_level_metrics(
            df_dedup['gt'], df_dedup['cv_pred'], df_dedup['combined_prob'],
        ),
        'unique_case_model_metrics': case_level_metrics(case_df, 'model_prob'),
        'unique_case_cv_metrics':    case_level_metrics(case_df, 'combined_prob'),
    }


def compute_group_metrics(radiologist_df):
    """Compute per-reader (balanced_accuracy, precision, recall, specificity, f1)
    for both arms (without and with model support) directly from
    `radiologist_df.csv`. Returns a long-format DataFrame with schema:

        columns: group, balanced_accuracy, precision, recall, specificity, f1
        rows:    11 rows for group1 (without model) + 11 rows for group2 (with model)
    """
    rdf = radiologist_df
    rows = []
    for label, with_seg in [('group1', False), ('group2', True)]:
        sub = rdf[rdf['with_segmentation'] == with_seg]
        for rad, grp in sub.groupby('radiologist'):
            gt = grp['has_enhancement_gt'].astype(int)
            pred = grp['predicted_enhancement'].astype(int)
            tn = int(((pred == 0) & (gt == 0)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            rows.append({
                'group': label,
                'balanced_accuracy': float(balanced_accuracy_score(gt, pred)),
                'precision':         float(precision_score(gt, pred, zero_division=0)),
                'recall':            float(recall_score(gt, pred, zero_division=0)),
                'specificity':       float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'f1':                float(f1_score(gt, pred, zero_division=0)),
            })
    return pd.DataFrame(rows)


def compute_paired_agreements(radiologist_df):
    """Compute pairwise Cohen's kappa for every reader pair, separately
    for the without-model arm and the with-model arm, directly from
    `radiologist_df.csv`. Returns a DataFrame with schema:

        columns: rad1, rad2, kappa_without, kappa_with
        rows:    11C2 = 55 reader pairs
    """
    rdf = radiologist_df

    def reader_preds(with_seg, rad):
        sub = rdf[(rdf['with_segmentation'] == with_seg) & (rdf['radiologist'] == rad)]
        return sub.set_index('case_id')['predicted_enhancement'].astype(int)

    readers = sorted(rdf['radiologist'].unique())
    rows = []
    for r1, r2 in combinations(readers, 2):
        p1_w, p2_w = reader_preds(False, r1), reader_preds(False, r2)
        p1_s, p2_s = reader_preds(True, r1),  reader_preds(True, r2)
        common_w = p1_w.index.intersection(p2_w.index)
        common_s = p1_s.index.intersection(p2_s.index)
        kappa_w = (cohen_kappa_score(p1_w.loc[common_w], p2_w.loc[common_w])
                   if len(common_w) >= 2 else float('nan'))
        kappa_s = (cohen_kappa_score(p1_s.loc[common_s], p2_s.loc[common_s])
                   if len(common_s) >= 2 else float('nan'))
        rows.append({'rad1': r1, 'rad2': r2,
                     'kappa_without': float(kappa_w),
                     'kappa_with':    float(kappa_s)})
    return pd.DataFrame(rows)


def compute_individual_perf(radiologist_df):
    """Per-(radiologist, with_segmentation) performance summary used by fig_6.

    Returns 22 rows (11 radiologists × 2 conditions), derived directly
    from radiologist_df.csv.
    """
    rows = []
    for (rad, with_seg), g in radiologist_df.groupby(['radiologist', 'with_segmentation']):
        gt = g['has_enhancement_gt'].astype(int)
        pred = g['predicted_enhancement'].astype(int)
        mp = g['model_predicted_enhancement'].astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        rad_acc = float(g['correct_prediction'].mean())
        model_acc = float((mp == gt).mean())
        rows.append({
            'radiologist': rad,
            'condition': 'With Model' if with_seg else 'Without Model',
            'with_segmentation': with_seg,
            'n_cases': len(g),
            'rad_accuracy': rad_acc,
            'rad_sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'rad_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'model_accuracy': model_acc,
            'accuracy_diff':  rad_acc - model_acc,
            'mean_confidence':    float(g['confidence'].mean()),
            'mean_response_time': float(g['response_time'].mean()),
            'mean_image_quality': float(g['image_quality'].mean()),
            'years_experience':   int(g['years_experience'].iloc[0]),
        })
    return pd.DataFrame(rows)


def compute_confidence_analysis(radiologist_df, min_cases_per_group=10):
    """Per-(radiologist, with_segmentation) confidence calibration metrics.

    Uses a Q3/Q1 quartile split for high_conf_accuracy and
    low_conf_accuracy. Groups with n_cases ≤ `min_cases_per_group`
    are excluded.
    """
    from scipy.stats import pearsonr  # local import: scipy is heavy
    rows = []
    for (rad, with_seg), g in radiologist_df.groupby(['radiologist', 'with_segmentation']):
        if len(g) <= min_cases_per_group:
            continue
        r, p = pearsonr(g['confidence'], g['correct_prediction'])
        q3 = g['confidence'].quantile(0.75)
        q1 = g['confidence'].quantile(0.25)
        high_acc = float(g[g['confidence'] >= q3]['correct_prediction'].mean())
        low_acc  = float(g[g['confidence'] <= q1]['correct_prediction'].mean())
        cor = float(g[g['correct_prediction'] == 1]['confidence'].mean())
        inc = float(g[g['correct_prediction'] == 0]['confidence'].mean())
        rows.append({
            'radiologist': rad,
            'condition': 'With Model' if with_seg else 'Without Model',
            'with_segmentation': with_seg,
            'n_cases': len(g),
            'conf_acc_corr':         float(r),
            'conf_acc_p':            float(p),
            'mean_confidence':       float(g['confidence'].mean()),
            'mean_accuracy':         float(g['correct_prediction'].mean()),
            'high_conf_accuracy':    high_acc,
            'low_conf_accuracy':     low_acc,
            'calibration_diff':      high_acc - low_acc,
            'correct_confidence':    cor,
            'incorrect_confidence':  inc,
            'confidence_bias':       cor - inc,
        })
    return pd.DataFrame(rows)


def compute_equiv(individual_perf_df, gpu_hours=2001.56,
                  clinical_hours_per_year=1500):
    """Per-radiologist equivalent-experience derivation.

    Fits a linear regression of years_experience → rad_accuracy /
    mean_confidence on the without-segmentation arm to obtain
    acc_slope and conf_slope. For each radiologist computes the
    accuracy and confidence gain (with - without) and divides by the
    corresponding slope to get equivalent additional years. Average
    of the two slopes is clamped at 0; converted to clinical hours
    (×1500) and ROI ratio (÷ gpu_hours).

    Reproduces `equiv.csv` to machine precision when seeded with the
    canonical individual_perf produced by `compute_individual_perf`.
    """
    from sklearn.linear_model import LinearRegression
    ip = individual_perf_df

    rows = []
    for rad, g in ip.groupby('radiologist'):
        w  = g[g['with_segmentation'] == True].iloc[0]
        wo = g[g['with_segmentation'] == False].iloc[0]
        rows.append({
            'radiologist':    rad,
            'years_experience': int(wo['years_experience']),
            'accuracy_gain':    float(w['rad_accuracy'])    - float(wo['rad_accuracy']),
            'confidence_gain':  float(w['mean_confidence']) - float(wo['mean_confidence']),
        })
    eq = pd.DataFrame(rows)

    without = ip[ip['with_segmentation'] == False].dropna(subset=['years_experience'])
    X = without['years_experience'].values.reshape(-1, 1)
    acc_slope  = float(LinearRegression().fit(X, without['rad_accuracy'].values).coef_[0])
    conf_slope = float(LinearRegression().fit(X, without['mean_confidence'].values).coef_[0])

    eq['equiv_acc_years']  = eq['accuracy_gain']   / acc_slope  if acc_slope  > 0 else 0.0
    eq['equiv_conf_years'] = eq['confidence_gain'] / conf_slope if conf_slope > 0 else 0.0
    eq['avg_equiv_years']  = ((eq['equiv_acc_years'] + eq['equiv_conf_years']) / 2).clip(lower=0)
    eq['equiv_clinical_hours'] = eq['avg_equiv_years'] * clinical_hours_per_year
    eq['roi_efficiency']       = eq['equiv_clinical_hours'] / gpu_hours
    return eq


def compute_financial(equiv_df, salary_df):
    """Per-radiologist financial valuation, derived from equiv + salary.

    For each radiologist:
      cumulative_salary_to_date = sum of NHS annual salaries for years
        1..current_experience.
      ai_leveraged_value = sum of NHS annual salaries for years
        current_experience+1..current_experience+avg_equiv_years, plus
        a fractional pro-rated final-year contribution.
      total_career_value = sum of the above two.
      value_increase_percentage = ai_leveraged_value /
        cumulative_salary_to_date × 100.

    Reproduces `financial.csv` to machine precision when seeded with the
    canonical equiv produced by `compute_equiv`.
    """
    salary_max = float(salary_df['annual_salary'].max())
    lookup = dict(zip(salary_df['experience_year'].astype(int),
                      salary_df['annual_salary'].astype(float)))
    n_years = len(salary_df)

    rows = []
    for _, row in equiv_df.iterrows():
        ce = int(row['years_experience'])
        eyg = float(row['avg_equiv_years'])
        cum_current = sum(lookup.get(y, salary_max) for y in range(1, ce + 1))
        start_y = ce + 1
        end_y = int(ce + eyg) + 1
        cum_gained = sum(lookup.get(y, salary_max) for y in range(start_y, end_y))
        frac = eyg - int(eyg)
        if frac > 0:
            cum_gained += (lookup.get(end_y, salary_max) if end_y <= n_years
                           else salary_max) * frac
        rows.append({
            'radiologist':                row['radiologist'],
            'current_experience':         ce,
            'equiv_years_gained':         eyg,
            'cumulative_salary_to_date':  cum_current,
            'ai_leveraged_value':         cum_gained,
            'total_career_value':         cum_current + cum_gained,
            'value_increase_percentage': (cum_gained / cum_current * 100)
                                          if cum_current > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def compute_pair_count_weighted_kappa_aggregate(radiologist_df):
    """Pair-count-weighted Cohen's kappa aggregate across all reader-pairs
    (rad-rad + rad-model), separately for the without-AI and with-AI arms.

    Mirrors the convention in extended_data_fig_4.py — weights the rad-rad
    component (n=1289 case-pairs across the 55 unique rad-rad pairs) and the
    rad-model component (n=1100 case-pairs across the 11 reader-model pairs)
    by their underlying case-pair counts, giving the same aggregate cited in
    Table 1 row 11 and in the inter-rater agreement Results paragraph
    (0.324 without support; 0.484 with support).

    Differs from a simple unweighted mean of the 66 per-pair Cohen's kappas:
    the latter weights every reader-pair equally regardless of how many
    case-pairs back its estimate (and yields 0.338 / 0.484 on the bundled
    data), whereas the pair-count-weighted aggregate gives more weight to
    pairs with more common cases.

    Returns
    -------
    (agg_without, agg_with) : tuple of float
        Pair-count-weighted Cohen's kappa aggregate across all 2389 case-pairs
        in each arm.
    """
    rdf = radiologist_df
    out = []
    for with_seg in (False, True):
        sub = rdf[rdf['with_segmentation'] == with_seg]
        rr_p1, rr_p2 = [], []
        case_counts = sub.groupby('case_id').size()
        multi_cases = case_counts[case_counts > 1].index.tolist()
        for case_id in multi_cases:
            cd = sub[sub['case_id'] == case_id]
            rads = cd['radiologist'].unique()
            if len(rads) > 1:
                for r1, r2 in combinations(rads, 2):
                    rr_p1.append(int(cd[cd['radiologist'] == r1]['predicted_enhancement'].iloc[0]))
                    rr_p2.append(int(cd[cd['radiologist'] == r2]['predicted_enhancement'].iloc[0]))
        k_rr = cohen_kappa_score(rr_p1, rr_p2) if len(rr_p1) >= 2 else float('nan')
        n_rr = len(rr_p1)
        mr = sub[['predicted_enhancement', 'model_predicted_enhancement']].dropna()
        k_rm = (cohen_kappa_score(mr['predicted_enhancement'].astype(int),
                                  mr['model_predicted_enhancement'].astype(int))
                if len(mr) >= 2 else float('nan'))
        n_rm = len(mr)
        agg = (k_rr * n_rr + k_rm * n_rm) / (n_rr + n_rm)
        out.append(float(agg))
    return tuple(out)
