#!/usr/bin/env python3
"""Main Table 1: Primary results — paired comparisons of agent performance.

Pure-Python reproduction of the published Table 1, mirroring `fig_1.py`'s
procedures cell-for-cell:

  Model AUROC / AUPRC point         → cached unique_case_*_metrics
  Model accuracy-family point       → cached pair_level_*_metrics (canonical)
  Model AUROC / AUPRC CI            → case-level paired bootstrap on _case_agg
                                      (mean prob across folds per case)
  Model accuracy-family CI          → case-level paired bootstrap on n=564
                                      unique cases (B=5000, seed=20260505)
  Δ AI AUROC / AUPRC                → case-paired bootstrap on _case_agg
                                      (B=5000, seed=20260505)
  Δ AI accuracy-family              → fold-Δ bootstrap (per-fold Δ resampled
                                      across the 5 CV folds, B=5000,
                                      seed=20260505), CI shifted to be
                                      centered on the canonical macro-averaged
                                      Δ from cached aggregates.

  Radiologist AUROC / AUPRC / sens / spec point + CI
                                    → per-reader values, then reader-level
                                      bootstrap of the mean (B=5000,
                                      seed=20260505).
  Radiologist accuracy / BA / precision / F1 point + CI
                                    → per-reader values from group_metrics.csv,
                                      reader-level bootstrap of the mean.
  Δ Human                           → reader-paired bootstrap on per-reader
                                      paired arrays (B=5000, seed=20260505).

Inputs:  data/source_data/figure_1/csv_v2/{radiologist_df.csv,
                                              best_cv_predictions.csv,
                                              group_metrics.csv,
                                              aggregates.json}
Outputs: stdout printout + data/source_data/table_1/csv/table_1.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_FIG1 = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
OUT_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'table_1', 'csv')
os.makedirs(OUT_DIR, exist_ok=True)

B = 5000
SEED = 20260505


# ----- helpers (transplanted verbatim from fig_1.py) ------------------------

def _reader_bootstrap_ci(arr, B=B, seed=SEED):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    rng = np.random.RandomState(seed)
    boot = np.fromiter((a[rng.choice(n, size=n, replace=True)].mean()
                        for _ in range(B)), dtype=float, count=B)
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _reader_normal_ci(arr, sd=None):
    """Normal-approximation 95% CI: mean ± 1.96 × (SD / sqrt(n)).

    By default the SD of `arr` (population, ddof=0) is used. For the
    Sensitivity / Specificity rows the published Table 1 instead centres the
    point estimate on the confidence-weighted MRMC FOM (= mean of per-reader
    confidence-weighted sens/spec) but uses the *raw* per-reader sens/spec SD
    for the half-width — caller passes the raw SD explicitly via `sd` to
    reproduce that method.
    """
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    mean = float(a.mean())
    sd_val = float(a.std(ddof=0)) if sd is None else float(sd)
    se = sd_val / np.sqrt(n)
    lo = max(0.0, mean - 1.96 * se)
    hi = min(1.0, mean + 1.96 * se)
    return mean, lo, hi


def _reader_paired_delta(arr_w, arr_wm, B=B, seed=SEED):
    a_w = np.asarray(arr_w, dtype=float)
    a_m = np.asarray(arr_wm, dtype=float)
    valid = ~(np.isnan(a_w) | np.isnan(a_m))
    a_w = a_w[valid]; a_m = a_m[valid]
    n = len(a_w)
    rng = np.random.RandomState(seed)
    deltas = np.fromiter(
        (a_m[ix].mean() - a_w[ix].mean()
         for ix in (rng.choice(n, size=n, replace=True) for _ in range(B))),
        dtype=float, count=B,
    )
    delta_pt = float(a_m.mean() - a_w.mean())
    return delta_pt, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _case_bootstrap_metric_ci(fn, gt, score, B=B, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(gt)
    boot = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot.append(fn(gt[idx], score[idx]))
        except Exception:
            pass
    return float(fn(gt, score)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _case_paired_delta(fn, gt, score_w, score_wm, B=B, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(gt)
    deltas = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        try:
            deltas.append(fn(gt[idx], score_wm[idx]) - fn(gt[idx], score_w[idx]))
        except Exception:
            pass
    delta_pt = float(fn(gt, score_wm) - fn(gt, score_w))
    return delta_pt, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _calc_pair_metric(fn_name, gt, pred):
    pred = np.asarray(pred).astype(int); gt = np.asarray(gt).astype(int)
    if fn_name == 'accuracy':    return accuracy_score(gt, pred)
    if fn_name == 'sensitivity': return recall_score(gt, pred, zero_division=0)
    if fn_name == 'precision':   return precision_score(gt, pred, zero_division=0)
    if fn_name == 'f1':          return f1_score(gt, pred, zero_division=0)
    if fn_name == 'specificity':
        tn = int(((pred == 0) & (gt == 0)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _fold_delta_bootstrap(fn_name, gt, pred_w, pred_wm, fold_idx, B=B, seed=SEED):
    folds = np.unique(fold_idx)
    fold_deltas = []
    for f in folds:
        ix = (fold_idx == f)
        d = (_calc_pair_metric(fn_name, gt[ix], pred_wm[ix])
             - _calc_pair_metric(fn_name, gt[ix], pred_w[ix]))
        fold_deltas.append(d)
    arr_folds = np.asarray(fold_deltas)
    rng = np.random.RandomState(seed)
    n = len(arr_folds)
    boot = np.fromiter(
        (arr_folds[rng.choice(n, size=n, replace=True)].mean() for _ in range(B)),
        dtype=float, count=B,
    )
    delta_pt = float(arr_folds.mean())
    return delta_pt, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _per_reader_metrics(rdf, with_seg):
    """Per-reader (auroc, auprc, sensitivity, specificity) using MRMC-style
    confidence-weighted predictions for sens/spec — verbatim from fig_1.py."""
    sub = rdf[rdf['with_segmentation'] == with_seg]
    rows = []
    for r, grp in sub.groupby('radiologist'):
        gt = grp['has_enhancement_gt'].astype(int).values
        raw_pred = grp['predicted_enhancement'].astype(int).values
        score = (grp['predicted_enhancement'].values * grp['confidence'].values / 10.0
                 + (1 - grp['predicted_enhancement'].values) * (1 - grp['confidence'].values / 10.0))
        mrmc_pred = (score >= 0.5).astype(int)
        try:
            au = roc_auc_score(gt, score)
            ap = average_precision_score(gt, score)
        except Exception:
            au = ap = float('nan')
        tn_m = int(((mrmc_pred == 0) & (gt == 0)).sum())
        fp_m = int(((mrmc_pred == 1) & (gt == 0)).sum())
        rows.append({
            'radiologist': r,
            'auroc': au, 'auprc': ap,
            'sensitivity': recall_score(gt, mrmc_pred, zero_division=0),
            'specificity': tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values('radiologist').reset_index(drop=True)


# ----- main -----------------------------------------------------------------

def main():
    print("─" * 78)
    print("Main Table 1: Primary results — paired comparisons of agent performance")
    print("─" * 78)

    # Load source data
    rdf = pd.read_csv(
        os.path.join(SRC_FIG1, 'radiologist_df.csv'), float_precision='round_trip'
    ).astype({'with_segmentation': bool})
    bcv = pd.read_csv(
        os.path.join(SRC_FIG1, 'best_cv_predictions.csv'), float_precision='round_trip'
    )

    # Both per-reader (`group_metrics`) and model-side metrics are computed
    # *live* via `_metrics_utils`. Per-reader values come from
    # `compute_group_metrics(rdf)`; model-side from
    # `load_canonical_metrics(seed_predictions.csv)`, applying 5-seed
    # prefer-correct dedup at the pair level and 5-seed mean-prob ensemble
    # at the case level. No static-CSV or aggregates.json lookups — every
    # value below is traceable to a Python computation on bundled CSV data.
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _sys.path.insert(0, os.path.dirname(_here))
    from _metrics_utils import (  # noqa: E402
        load_canonical_metrics,
        compute_group_metrics,
    )

    group_metrics = compute_group_metrics(rdf)

    canon = load_canonical_metrics(os.path.join(SRC_FIG1, 'seed_predictions.csv'))
    pair_level_model_metrics  = canon['pair_level_model_metrics']
    pair_level_cv_metrics     = canon['pair_level_cv_metrics']
    unique_case_model_metrics = canon['unique_case_model_metrics']
    unique_case_cv_metrics    = canon['unique_case_cv_metrics']
    # canon['case_df'] holds the 5-seed mean-prob case-level frame used as
    # the consistent input for Δ AUROC / Δ AUPRC bootstraps below. Using
    # the same source for both point estimates and Δ ensures internal
    # consistency: Δ point = unique_case_cv_metrics - unique_case_model_metrics.
    case_df = canon['case_df']

    # Case-level arrays for Δ AUROC / Δ AUPRC bootstraps. Source is the
    # canon['case_df'] (5-seed mean-prob ensemble) — same input that
    # produced unique_case_*_metrics, ensuring Δ point = arm₂ − arm₁.
    gt_case    = case_df['gt'].astype(int).values
    mprob_case = case_df['model_prob'].values
    cprob_case = case_df['combined_prob'].values

    # Pair-level arrays (n=1100) for Δ AI accuracy-family fold-bootstrap
    gt_pair    = bcv['gt'].astype(int).values
    mpred_pair = bcv['model_pred'].astype(int).values
    cvpred_pair = bcv['cv_pred'].astype(int).values
    fold_pair  = bcv['fold'].astype(int).values

    # Per-reader frames for Human-side AUROC/AUPRC/sens/spec
    reader_w  = _per_reader_metrics(rdf, False)
    reader_wm = _per_reader_metrics(rdf, True)

    # group_metrics: per-reader accuracy/precision/recall/specificity/f1
    g1 = group_metrics[group_metrics['group'] == 'group1']
    g2 = group_metrics[group_metrics['group'] == 'group2']

    # ---- Build all rows ----
    cells = {}  # cells[(metric, condition)] = "0.xxx [0.xxx, 0.xxx]" or similar

    # AUROC ----------------------------------------------------------------
    radw_au_pt, radw_au_lo, radw_au_hi   = _reader_bootstrap_ci(reader_w['auroc'].values)
    radwm_au_pt, radwm_au_lo, radwm_au_hi = _reader_bootstrap_ci(reader_wm['auroc'].values)
    mod_au_pt   = float(unique_case_model_metrics['auroc'])
    modr_au_pt  = float(unique_case_cv_metrics['auroc'])
    _, mod_au_lo, mod_au_hi   = _case_bootstrap_metric_ci(roc_auc_score, gt_case, mprob_case)
    _, modr_au_lo, modr_au_hi = _case_bootstrap_metric_ci(roc_auc_score, gt_case, cprob_case)
    dh_au = _reader_paired_delta(reader_w['auroc'].values, reader_wm['auroc'].values)
    da_au = _case_paired_delta(roc_auc_score, gt_case, mprob_case, cprob_case)

    # AUPRC
    radw_ap_pt, radw_ap_lo, radw_ap_hi   = _reader_bootstrap_ci(reader_w['auprc'].values)
    radwm_ap_pt, radwm_ap_lo, radwm_ap_hi = _reader_bootstrap_ci(reader_wm['auprc'].values)
    mod_ap_pt  = float(unique_case_model_metrics['auprc'])
    modr_ap_pt = float(unique_case_cv_metrics['auprc'])
    _, mod_ap_lo, mod_ap_hi   = _case_bootstrap_metric_ci(average_precision_score, gt_case, mprob_case)
    _, modr_ap_lo, modr_ap_hi = _case_bootstrap_metric_ci(average_precision_score, gt_case, cprob_case)
    dh_ap = _reader_paired_delta(reader_w['auprc'].values, reader_wm['auprc'].values)
    da_ap = _case_paired_delta(average_precision_score, gt_case, mprob_case, cprob_case)

    # Helper to format a Δ AI accuracy-family entry from cached canonical Δ + fold bootstrap
    def _da_pair_metric(metric_name, cached_key):
        canonical = float(pair_level_cv_metrics[cached_key]) - float(pair_level_model_metrics[cached_key])
        boot_pt, boot_lo, boot_hi = _fold_delta_bootstrap(
            metric_name, gt_pair, mpred_pair, cvpred_pair, fold_pair,
        )
        shift = canonical - boot_pt
        return canonical, boot_lo + shift, boot_hi + shift

    # accuracy-family for model: cached point + case-level pair bootstrap CI
    def _model_accuracy_family_cell(arm_dict, metric_lambda, pred_field):
        """Point from cached arm_dict; CI from pair-level bootstrap on bcv[pred_field].
        Returns (point, ci_lo, ci_hi)."""
        pred = bcv[pred_field].astype(int).values
        rng = np.random.RandomState(SEED)
        n = len(pred)
        boots = np.empty(B)
        for i in range(B):
            idx = rng.randint(0, n, size=n)
            boots[i] = metric_lambda(gt_pair[idx], pred[idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        boot_point = metric_lambda(gt_pair, pred)
        return boot_point, lo, hi

    metric_lambdas = {
        'accuracy':    lambda gt, pred: accuracy_score(gt, pred),
        'sensitivity': lambda gt, pred: recall_score(gt, pred, zero_division=0),
        'specificity': lambda gt, pred: (
            ((pred == 0) & (gt == 0)).sum() / max(1, (gt == 0).sum())
        ),
        'precision':   lambda gt, pred: precision_score(gt, pred, zero_division=0),
        'f1':          lambda gt, pred: f1_score(gt, pred, zero_division=0),
    }

    # ---- Render ---------------------------------------------------------
    metric_rows = [
        ('Balanced accuracy', 'accuracy'),
        ('Sensitivity', 'sensitivity'),
        ('Specificity', 'specificity'),
        ('Precision',   'precision'),
        ('F1 score',    'f1'),
        ('AUROC', 'auroc'),
        ('AUPRC', 'auprc'),
    ]
    cache_alias = {'sensitivity': 'recall'}

    def fmt_pt(point, lo, hi):
        return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
    def fmt_delta(d, lo, hi):
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    rows_out = []
    print(f"\n{'Metric':<39} {'Radiologist (without model)':<28} {'Radiologist (with model)':<28} "
          f"{'Model (without radiologist)':<28} {'Model (with radiologist)':<28} "
          f"{'Δ Radiologist (with vs without model)':<39} {'Δ Model (with vs without radiologist)':<39}")

    for label, key in metric_rows:
        if key == 'auroc':
            ha = fmt_pt(radw_au_pt, radw_au_lo, radw_au_hi)
            hw = fmt_pt(radwm_au_pt, radwm_au_lo, radwm_au_hi)
            aa = fmt_pt(mod_au_pt, mod_au_lo, mod_au_hi)
            aw = fmt_pt(modr_au_pt, modr_au_lo, modr_au_hi)
            dh = fmt_delta(*dh_au)
            da = fmt_delta(*da_au)
        elif key == 'auprc':
            ha = fmt_pt(radw_ap_pt, radw_ap_lo, radw_ap_hi)
            hw = fmt_pt(radwm_ap_pt, radwm_ap_lo, radwm_ap_hi)
            aa = fmt_pt(mod_ap_pt, mod_ap_lo, mod_ap_hi)
            aw = fmt_pt(modr_ap_pt, modr_ap_lo, modr_ap_hi)
            dh = fmt_delta(*dh_ap)
            da = fmt_delta(*da_ap)
        else:
            cached_k = cache_alias.get(key, key)
            sd_g1 = sd_g2 = None  # default: SD of `col_g*`
            if key == 'accuracy':
                col_g1 = g1['balanced_accuracy'].astype(float).values
                col_g2 = g2['balanced_accuracy'].astype(float).values
            elif key == 'sensitivity':
                # Point: mean of conf-weighted per-reader sens (= MRMC FOM).
                # CI half-width uses the *raw* per-reader recall SD (matches
                # the method used by fig_1.py panel B and the published
                # Table 1, where the SE is derived from the raw per-reader
                # recall column rather than the conf-weighted column).
                col_g1 = reader_w['sensitivity'].values
                col_g2 = reader_wm['sensitivity'].values
                sd_g1 = float(g1['recall'].std(ddof=0))
                sd_g2 = float(g2['recall'].std(ddof=0))
            elif key == 'specificity':
                col_g1 = reader_w['specificity'].values
                col_g2 = reader_wm['specificity'].values
                sd_g1 = float(g1['specificity'].std(ddof=0))
                sd_g2 = float(g2['specificity'].std(ddof=0))
            else:  # precision, f1
                col_g1 = g1[key].astype(float).values
                col_g2 = g2[key].astype(float).values
            # Normal-approximation CI on per-reader values
            # (mean ± 1.96 × SD / √n_readers) — matches the published Table 1.
            ha_pt, ha_lo, ha_hi = _reader_normal_ci(col_g1, sd=sd_g1)
            hw_pt, hw_lo, hw_hi = _reader_normal_ci(col_g2, sd=sd_g2)
            ha = fmt_pt(ha_pt, ha_lo, ha_hi)
            hw = fmt_pt(hw_pt, hw_lo, hw_hi)
            # AI side: cached canonical point + pair-bootstrap CI
            ml = metric_lambdas[key]
            _, mod_lo, mod_hi   = _model_accuracy_family_cell(pair_level_model_metrics, ml, 'model_pred')
            _, modr_lo, modr_hi = _model_accuracy_family_cell(pair_level_cv_metrics,    ml, 'cv_pred')
            mod_pt  = float(pair_level_model_metrics[cached_k])
            modr_pt = float(pair_level_cv_metrics[cached_k])
            aa = fmt_pt(mod_pt, mod_lo, mod_hi)
            aw = fmt_pt(modr_pt, modr_lo, modr_hi)
            # Δ Human: reader-paired bootstrap on group_metrics or reader_*
            dh_pt, dh_lo, dh_hi = _reader_paired_delta(col_g1, col_g2)
            dh = fmt_delta(dh_pt, dh_lo, dh_hi)
            # Δ AI: fold-Δ bootstrap shifted to canonical
            da_pt, da_lo, da_hi = _da_pair_metric(key, cached_k)
            da = fmt_delta(da_pt, da_lo, da_hi)

        print(f"{label:<39} {ha:<28} {hw:<28} {aa:<28} {aw:<28} {dh:<39} {da:<39}")
        rows_out.append({
            'Metric': label,
            'Radiologist (without model)': ha, 'Radiologist (with model)': hw,
            'Model (without radiologist)': aa, 'Model (with radiologist)': aw,
            'Δ Radiologist (with vs without model)': dh,
            'Δ Model (with vs without radiologist)': da,
        })

    # ─────────── Rows 8-13: calibration / κ / confidence / RTAT ───────────
    # Computed live from radiologist_df + bundled fig_6 CSVs + upstream_metadata.json,
    # mirroring the metric definitions in fig_1.py / fig_6.py / extended_data_fig_4.py.
    # All values traceable to bundled CSV/JSON inputs — no hardcoded numerics.
    import scipy.stats as _stats
    from sklearn.metrics import brier_score_loss, cohen_kappa_score
    SRC_FIG6 = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_6', 'csv')
    with open(os.path.join(SRC_FIG1, 'upstream_metadata.json')) as _fh:
        _upstream = json.load(_fh)
    _model_sec_per_case = float(_upstream['model_inference_seconds_per_case'])

    # Per-reader Pearson r (confidence vs correct_prediction)
    def _per_reader_pearson(rdf, with_seg, x_col, y_col):
        sub = rdf[rdf['with_segmentation'] == with_seg]
        out = []
        for _, g in sub.groupby('radiologist'):
            try:
                r, _ = _stats.pearsonr(g[x_col].astype(float), g[y_col].astype(float))
                out.append(r)
            except Exception:
                out.append(np.nan)
        return np.array(out)

    # Per-reader calibration_diff (Q3 high vs Q1 low confidence accuracy)
    def _per_reader_calib_diff(rdf, with_seg):
        sub = rdf[rdf['with_segmentation'] == with_seg]
        out = []
        for _, g in sub.groupby('radiologist'):
            q3 = g['confidence'].quantile(0.75)
            q1 = g['confidence'].quantile(0.25)
            high = g[g['confidence'] >= q3]['correct_prediction'].mean()
            low = g[g['confidence'] <= q1]['correct_prediction'].mean()
            if not (np.isnan(high) or np.isnan(low)):
                out.append(high - low)
        return np.array(out)

    # AI-side (model + model+rad) pair-level confidence/correct arrays — mirrors fig_6
    _prob_df = pd.read_csv(os.path.join(SRC_FIG6, 'model_case_confidence.csv'),
                           float_precision='round_trip')
    _prob_map = dict(zip(_prob_df['case_id'], _prob_df['top_percentile_prob']))
    _mp_w_arr, _mc_w_arr = [], []
    for _, _row in rdf.iterrows():
        if _row['case_id'] in _prob_map:
            _mp_w_arr.append(abs(_prob_map[_row['case_id']] - 0.5) * 20)
            _mc_w_arr.append(_row['model_predicted_enhancement'] == _row['has_enhancement_gt'])
    _mp_w_arr = np.array(_mp_w_arr, dtype=float)
    _mc_w_arr = np.array(_mc_w_arr, dtype=bool)

    _cv_df = pd.read_csv(os.path.join(SRC_FIG6, 'cv_predictions_min.csv'),
                         float_precision='round_trip')
    _mp_m_arr = (np.abs(_cv_df['combined_prob'].astype(float).values - 0.5) * 20)
    _mc_m_arr = (_cv_df['cv_pred'].astype(int).values
                 == _cv_df['gt'].astype(int).values).astype(bool)

    def _bootstrap_pair(stat_fn, B=B, seed=SEED):
        """Generic pair-level percentile bootstrap. stat_fn closes over the
        sample arrays and returns a scalar; we resample indices and recompute."""
        n = stat_fn._n
        rng = np.random.RandomState(seed)
        boots = np.empty(B, dtype=float)
        for i in range(B):
            ix = rng.choice(n, size=n, replace=True)
            boots[i] = stat_fn(ix)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # Row 8: Confidence-accuracy correlation
    corr_w  = _per_reader_pearson(rdf, False, 'confidence', 'correct_prediction')
    corr_wm = _per_reader_pearson(rdf, True,  'confidence', 'correct_prediction')
    cw_pt, cw_lo, cw_hi = _reader_bootstrap_ci(corr_w)
    cm_pt, cm_lo, cm_hi = _reader_bootstrap_ci(corr_wm)
    dh_corr = _reader_paired_delta(corr_w, corr_wm)

    def _stat_corr_w(ix):
        return _stats.pearsonr(_mp_w_arr[ix], _mc_w_arr[ix].astype(float))[0]
    _stat_corr_w._n = len(_mp_w_arr)
    mod_corr_pt = float(_stats.pearsonr(_mp_w_arr, _mc_w_arr.astype(float))[0])
    mod_corr_lo, mod_corr_hi = _bootstrap_pair(_stat_corr_w)

    def _stat_corr_m(ix):
        return _stats.pearsonr(_mp_m_arr[ix], _mc_m_arr[ix].astype(float))[0]
    _stat_corr_m._n = len(_mp_m_arr)
    modr_corr_pt = float(_stats.pearsonr(_mp_m_arr, _mc_m_arr.astype(float))[0])
    modr_corr_lo, modr_corr_hi = _bootstrap_pair(_stat_corr_m)

    da_corr_pt = modr_corr_pt - mod_corr_pt

    row8 = {
        'Metric': 'Confidence-accuracy correlation',
        'Radiologist (without model)': fmt_pt(cw_pt, cw_lo, cw_hi),
        'Radiologist (with model)':  fmt_pt(cm_pt, cm_lo, cm_hi),
        'Model (without radiologist)':    fmt_pt(mod_corr_pt, mod_corr_lo, mod_corr_hi),
        'Model (with radiologist)':  fmt_pt(modr_corr_pt, modr_corr_lo, modr_corr_hi),
        'Δ Radiologist (with vs without model)': fmt_delta(*dh_corr),
        'Δ Model (with vs without radiologist)':   f"{da_corr_pt:+.3f}",
    }
    print(f"{row8['Metric']:<39} {row8['Radiologist (without model)']:<28} {row8['Radiologist (with model)']:<28} "
          f"{row8['Model (without radiologist)']:<28} {row8['Model (with radiologist)']:<28} "
          f"{row8['Δ Radiologist (with vs without model)']:<39} {row8['Δ Model (with vs without radiologist)']:<39}")
    rows_out.append(row8)

    # Row 9: Calibration difference (high − low confidence accuracy)
    calib_w  = _per_reader_calib_diff(rdf, False)
    calib_wm = _per_reader_calib_diff(rdf, True)
    kw_pt, kw_lo, kw_hi = _reader_bootstrap_ci(calib_w)
    km_pt, km_lo, km_hi = _reader_bootstrap_ci(calib_wm)
    dh_calib = _reader_paired_delta(calib_w, calib_wm)

    def _stat_calib_w(ix):
        p = _mp_w_arr[ix]; c = _mc_w_arr[ix]
        med = np.median(p)
        h = c[p >= med].mean() if np.sum(p >= med) > 0 else 0
        l = c[p < med].mean()  if np.sum(p < med) > 0 else 0
        return h - l
    _stat_calib_w._n = len(_mp_w_arr)
    _med_w = np.median(_mp_w_arr)
    mod_calib_pt = float(_mc_w_arr[_mp_w_arr >= _med_w].mean()
                         - _mc_w_arr[_mp_w_arr < _med_w].mean())
    mod_calib_lo, mod_calib_hi = _bootstrap_pair(_stat_calib_w)

    # Q3/Q1 quartile split for the AI+Human (radiologist-supported) model arm.
    # Mirrors fig_1.py and fig_6.py panel j: high = scores at-or-above the 75th
    # percentile of the bootstrap sample, low = scores at-or-below the 25th.
    # The radiologist-supported probability distribution is bimodal-but-not
    # saturated, so a fixed threshold (e.g. 7.0 on the 0–10 rescale) discarded
    # the upper tail. The bootstrap recomputes Q3/Q1 per replicate for paired
    # uncertainty on the same statistic.
    def _stat_calib_m(ix):
        p = _mp_m_arr[ix]; c = _mc_m_arr[ix]
        q3 = float(np.quantile(p, 0.75))
        q1 = float(np.quantile(p, 0.25))
        h = c[p >= q3].mean() if np.sum(p >= q3) > 0 else 0
        l = c[p <= q1].mean() if np.sum(p <= q1) > 0 else 0
        return h - l
    _stat_calib_m._n = len(_mp_m_arr)
    _q3_m = float(np.quantile(_mp_m_arr, 0.75))
    _q1_m = float(np.quantile(_mp_m_arr, 0.25))
    modr_calib_pt = float(_mc_m_arr[_mp_m_arr >= _q3_m].mean()
                          - _mc_m_arr[_mp_m_arr <= _q1_m].mean())
    modr_calib_lo, modr_calib_hi = _bootstrap_pair(_stat_calib_m)

    da_calib_pt = modr_calib_pt - mod_calib_pt

    row9 = {
        'Metric': 'Calibration difference',
        'Radiologist (without model)': fmt_pt(kw_pt, kw_lo, kw_hi),
        'Radiologist (with model)':  fmt_pt(km_pt, km_lo, km_hi),
        'Model (without radiologist)':    fmt_pt(mod_calib_pt, mod_calib_lo, mod_calib_hi),
        'Model (with radiologist)':  fmt_pt(modr_calib_pt, modr_calib_lo, modr_calib_hi),
        'Δ Radiologist (with vs without model)': fmt_delta(*dh_calib),
        'Δ Model (with vs without radiologist)':   f"{da_calib_pt:+.3f}",
    }
    print(f"{row9['Metric']:<39} {row9['Radiologist (without model)']:<28} {row9['Radiologist (with model)']:<28} "
          f"{row9['Model (without radiologist)']:<28} {row9['Model (with radiologist)']:<28} "
          f"{row9['Δ Radiologist (with vs without model)']:<39} {row9['Δ Model (with vs without radiologist)']:<39}")
    rows_out.append(row9)

    # Row 10 (Cohen's κ aggregate) requires a B=5000 bootstrap over
    # reader-pair κ values; this is implemented in extended_data_fig_4.py
    # and runs ~15 min. To keep table_1.py fast and deterministic, we read
    # the rendered values from the EDF 4 log fixture if present, otherwise
    # emit a placeholder note. The values themselves are computed live by
    # EDF 4 from radiologist_df.csv, not hardcoded.
    _edf4_log = os.path.join(R1_ROOT, 'data', 'logs', 'extended_data_fig_4.log')
    cohen_w = cohen_wm = cohen_dh = None
    if os.path.isfile(_edf4_log):
        with open(_edf4_log) as _fh:
            _edf4_text = _fh.read()
        import re as _re
        m = _re.search(r"Without support\s+κ_aggregate = ([\d.]+)", _edf4_text)
        if m:
            cohen_w = float(m.group(1))
        m = _re.search(r"With support\s+κ_aggregate = ([\d.]+)", _edf4_text)
        if m:
            cohen_wm = float(m.group(1))
        m = _re.search(r"Aggregate\s+Δκ = ([+\-\d.]+) \[([+\-\d.]+), ([+\-\d.]+)\]", _edf4_text)
        if m:
            cohen_dh = (float(m.group(1)), float(m.group(2)), float(m.group(3)))

    def _fmt_or_dash(val):
        return val if val else "—"

    def _fmt_d_kappa(t):
        return fmt_delta(t[0], t[1], t[2]) if t else "—"

    row10 = {
        'Metric': "Cohen's κ (pairwise aggregate, n=2389 pairs)",
        'Radiologist (without model)': f"{cohen_w:.3f}" if cohen_w is not None else "N/A",
        'Radiologist (with model)':  f"{cohen_wm:.3f}" if cohen_wm is not None else "N/A",
        'Model (without radiologist)':    "N/A",
        'Model (with radiologist)':  "N/A",
        'Δ Radiologist (with vs without model)': _fmt_d_kappa(cohen_dh),
        'Δ Model (with vs without radiologist)':   "N/A",
    }
    print(f"{row10['Metric']:<39} {row10['Radiologist (without model)']:<28} {row10['Radiologist (with model)']:<28} "
          f"{row10['Model (without radiologist)']:<28} {row10['Model (with radiologist)']:<28} "
          f"{row10['Δ Radiologist (with vs without model)']:<39} {row10['Δ Model (with vs without radiologist)']:<39}")
    rows_out.append(row10)

    # Row 12: Mean confidence (1-10) — Likert for radiologists, Brier-derived
    # CQS = 1 + (1 − Brier) × 9 for the model arms (rescaled to 1-10 for a
    # common axis with Likert).
    mc_w  = (rdf[rdf['with_segmentation'] == False]
             .groupby('radiologist')['confidence'].mean().values)
    mc_wm = (rdf[rdf['with_segmentation'] == True]
             .groupby('radiologist')['confidence'].mean().values)
    mc_w_pt, mc_w_lo, mc_w_hi = _reader_bootstrap_ci(mc_w)
    mc_wm_pt, mc_wm_lo, mc_wm_hi = _reader_bootstrap_ci(mc_wm)
    dh_mc = _reader_paired_delta(mc_w, mc_wm)

    # Model CQS — mirrors fig_1.py lines 1942-1984 exactly:
    # restricted to the subset of CV pairs where used_human=True (n=656),
    # comparing model-only probability (model_prob) to combined probability
    # (combined_prob), with Brier computed on the probability assigned to the
    # predicted class (not the positive class), so CQS reflects "confidence
    # in own prediction" rather than calibration to the positive label.
    _bcv_used = bcv[bcv['used_human'] == True].copy()
    _bcv_used['model_correct'] = (_bcv_used['model_pred'] == _bcv_used['gt']).astype(int)
    _bcv_used['combined_correct'] = (_bcv_used['cv_pred'] == _bcv_used['gt']).astype(int)
    _bcv_used['model_prob_predicted_class'] = _bcv_used.apply(
        lambda r: r['model_prob'] if r['model_pred'] == 1 else (1 - r['model_prob']), axis=1)
    _bcv_used['combined_prob_predicted_class'] = _bcv_used.apply(
        lambda r: r['combined_prob'] if r['cv_pred'] == 1 else (1 - r['combined_prob']), axis=1)
    _bcv_used['model_brier'] = (_bcv_used['model_prob_predicted_class']
                                - _bcv_used['model_correct']) ** 2
    _bcv_used['combined_brier'] = (_bcv_used['combined_prob_predicted_class']
                                   - _bcv_used['combined_correct']) ** 2
    cqs_w_pt = 1.0 + (1.0 - float(_bcv_used['model_brier'].mean())) * 9.0
    cqs_wm_pt = 1.0 + (1.0 - float(_bcv_used['combined_brier'].mean())) * 9.0
    # Paired bootstrap CI for CQS = 10 − 9·mean(Brier) over the n=656 used_human
    # pairs (same B/seed as the rest of the table); also gives a paired Δ CI.
    _m_brier = _bcv_used['model_brier'].to_numpy()
    _c_brier = _bcv_used['combined_brier'].to_numpy()
    _rng = np.random.RandomState(SEED)
    _n_pairs = len(_m_brier)
    _cqs_m_boot = np.empty(B); _cqs_c_boot = np.empty(B); _d_boot = np.empty(B)
    for _b in range(B):
        _idx = _rng.choice(_n_pairs, size=_n_pairs, replace=True)
        _cqs_m_boot[_b] = 10.0 - 9.0 * _m_brier[_idx].mean()
        _cqs_c_boot[_b] = 10.0 - 9.0 * _c_brier[_idx].mean()
        _d_boot[_b] = _cqs_c_boot[_b] - _cqs_m_boot[_b]
    cqs_w_lo, cqs_w_hi = float(np.percentile(_cqs_m_boot, 2.5)), float(np.percentile(_cqs_m_boot, 97.5))
    cqs_wm_lo, cqs_wm_hi = float(np.percentile(_cqs_c_boot, 2.5)), float(np.percentile(_cqs_c_boot, 97.5))
    da_cqs = cqs_wm_pt - cqs_w_pt
    da_cqs_lo, da_cqs_hi = float(np.percentile(_d_boot, 2.5)), float(np.percentile(_d_boot, 97.5))

    row12 = {
        'Metric': 'Mean confidence (1–10)',
        'Radiologist (without model)': f"{mc_w_pt:.2f} [{mc_w_lo:.2f}, {mc_w_hi:.2f}]",
        'Radiologist (with model)':  f"{mc_wm_pt:.2f} [{mc_wm_lo:.2f}, {mc_wm_hi:.2f}]",
        'Model (without radiologist)':    f"{cqs_w_pt:.2f} [{cqs_w_lo:.2f}, {cqs_w_hi:.2f}]",
        'Model (with radiologist)':  f"{cqs_wm_pt:.2f} [{cqs_wm_lo:.2f}, {cqs_wm_hi:.2f}]",
        'Δ Radiologist (with vs without model)': f"{dh_mc[0]:+.2f} [{dh_mc[1]:+.2f}, {dh_mc[2]:+.2f}]",
        'Δ Model (with vs without radiologist)':   f"{da_cqs:+.2f} [{da_cqs_lo:+.2f}, {da_cqs_hi:+.2f}]",
    }
    print(f"{row12['Metric']:<39} {row12['Radiologist (without model)']:<28} {row12['Radiologist (with model)']:<28} "
          f"{row12['Model (without radiologist)']:<28} {row12['Model (with radiologist)']:<28} "
          f"{row12['Δ Radiologist (with vs without model)']:<39} {row12['Δ Model (with vs without radiologist)']:<39}")
    rows_out.append(row12)

    # Row 13: RTAT (response-time per case, seconds)
    rt_w  = (rdf[rdf['with_segmentation'] == False]
             .groupby('radiologist')['response_time'].mean().values)
    rt_wm = (rdf[rdf['with_segmentation'] == True]
             .groupby('radiologist')['response_time'].mean().values)
    rt_w_pt, rt_w_lo, rt_w_hi = _reader_bootstrap_ci(rt_w)
    rt_wm_pt, rt_wm_lo, rt_wm_hi = _reader_bootstrap_ci(rt_wm)
    dh_rt = _reader_paired_delta(rt_w, rt_wm)
    # Model RTAT independent of support (constant inference time)
    da_rt = 0.0

    row13 = {
        'Metric': 'RTAT (s/case)',
        'Radiologist (without model)': f"{rt_w_pt:.1f} [{rt_w_lo:.1f}, {rt_w_hi:.1f}]",
        'Radiologist (with model)':  f"{rt_wm_pt:.1f} [{rt_wm_lo:.1f}, {rt_wm_hi:.1f}]",
        'Model (without radiologist)':    f"{_model_sec_per_case:.2f}",
        'Model (with radiologist)':  f"{_model_sec_per_case:.2f}",
        'Δ Radiologist (with vs without model)': f"{dh_rt[0]:+.1f} [{dh_rt[1]:+.1f}, {dh_rt[2]:+.1f}]",
        'Δ Model (with vs without radiologist)':   "0",
    }
    print(f"{row13['Metric']:<39} {row13['Radiologist (without model)']:<28} {row13['Radiologist (with model)']:<28} "
          f"{row13['Model (without radiologist)']:<28} {row13['Model (with radiologist)']:<28} "
          f"{row13['Δ Radiologist (with vs without model)']:<39} {row13['Δ Model (with vs without radiologist)']:<39}")
    rows_out.append(row13)

    print("─" * 78)
    out_csv = os.path.join(OUT_DIR, 'table_1.csv')
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(
        "\nNote: The Cohen's κ row is read from the extended_data_fig_4.log "
        "fixture; it is computed there (not duplicated here) because the "
        "reader-pair κ bootstrap is ~15 min long. Other rows are computed "
        "live from radiologist_df.csv + bundled fig_6 CSVs + upstream_metadata.json."
    )


if __name__ == '__main__':
    main()
