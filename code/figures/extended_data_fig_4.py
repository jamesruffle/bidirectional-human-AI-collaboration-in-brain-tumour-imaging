#!/usr/bin/env python3
"""Extended Data Figure 4 - Agreement comparisons (model vs radiologist, radiologist vs radiologist).

Self-contained reproduction script. Inputs:
  - data/source_data/extended_data_figure_4/csv/agreement_inputs.csv
    (2200 rows x 5 cols: case_id, radiologist, with_segmentation,
     predicted_enhancement, model_predicted_enhancement)

Output: data/figures/Extended_Data_Fig_4.png  (and .svg)

"""
from itertools import combinations
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sns.set_palette("husl")

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RDF_PATH = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2', 'radiologist_df.csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


def main():
    if not os.path.isfile(RDF_PATH):
        raise SystemExit(f"ERROR: radiologist_df.csv missing at {RDF_PATH}")
    os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

    # The 5 columns this figure uses (case_id / radiologist / with_segmentation /
    # predicted_enhancement / model_predicted_enhancement) are derived live
    # from radiologist_df.csv.
    print(f"Computing minimal agreement inputs live from {RDF_PATH}...")
    radiologist_df = pd.read_csv(RDF_PATH, float_precision='round_trip')[[
        'case_id', 'radiologist', 'with_segmentation',
        'predicted_enhancement', 'model_predicted_enhancement',
    ]]
    print(f"  Loaded {radiologist_df.shape[0]} rows x {radiologist_df.shape[1]} cols (live-derived)")

    # Generate Extended Data Figure 4 (2x2 layout)
    print("\nGenerating Extended Data Figure 4...")
    fig_s1 = plt.figure(figsize=(10, 10))

    # Add overall title
    fig_s1.suptitle('Agreement comparisons', y=0.96)

    gs = fig_s1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel a (row 0, col 0): Model-Radiologist Agreement WITHOUT support
    # Model predictions without radiologist input vs Radiologist predictions without model support
    ax_a = fig_s1.add_subplot(gs[0, 0])

    # Get data where NEITHER has support from the other
    without_support_data = radiologist_df[radiologist_df['with_segmentation'] == False].copy()
    without_support_data = without_support_data.dropna(subset=['model_predicted_enhancement'])

    model_pred = without_support_data['model_predicted_enhancement'].values
    rad_pred = without_support_data['predicted_enhancement'].values

    confusion_matrix_without = confusion_matrix(model_pred, rad_pred)
    kappa_without = cohen_kappa_score(model_pred, rad_pred)

    sns.heatmap(confusion_matrix_without, annot=True, fmt='d', cmap='Blues', ax=ax_a,
                xticklabels=['No enhancement', 'Enhancement'],
                yticklabels=['No enhancement', 'Enhancement'], cbar=False)

    ax_a.set_title('a) Model-radiologist agreement' + '\n' + 'Without support' + f' (κ = {kappa_without:.3f})')
    ax_a.set_xlabel('Radiologist prediction')
    ax_a.set_ylabel('Model prediction')

    # Panel b (row 0, col 1): Model-Radiologist Agreement WITH support
    ax_b = fig_s1.add_subplot(gs[0, 1])

    with_support_data = radiologist_df[radiologist_df['with_segmentation'] == True].copy()
    with_support_data = with_support_data.dropna(subset=['model_predicted_enhancement'])

    model_pred = with_support_data['model_predicted_enhancement'].values
    rad_pred = with_support_data['predicted_enhancement'].values

    confusion_matrix_with = confusion_matrix(model_pred, rad_pred)
    kappa_with = cohen_kappa_score(model_pred, rad_pred)

    sns.heatmap(confusion_matrix_with, annot=True, fmt='d', cmap='Blues', ax=ax_b,
                xticklabels=['No enhancement', 'Enhancement'],
                yticklabels=['No enhancement', 'Enhancement'], cbar=False)

    ax_b.set_title('b) Model-radiologist agreement' + '\n' + 'With support' + f' (κ = {kappa_with:.3f})')
    ax_b.set_xlabel('Radiologist prediction')
    ax_b.set_ylabel('Model prediction')

    # Panel c (row 1, col 0): Radiologist-Radiologist Agreement WITHOUT support
    ax_c = fig_s1.add_subplot(gs[1, 0])

    # Find cases reviewed by multiple radiologists
    case_radiologist_counts = radiologist_df.groupby(['case_id', 'with_segmentation']).size().reset_index(name='num_radiologists')
    multi_radiologist_cases = case_radiologist_counts[case_radiologist_counts['num_radiologists'] > 1]

    # Separate by segmentation condition - WITHOUT model support
    multi_without_seg = multi_radiologist_cases[multi_radiologist_cases['with_segmentation'] == 0]

    # Calculate pairwise agreements
    agreements_without = []
    for case_id in multi_without_seg['case_id'].tolist():
        case_data = radiologist_df[(radiologist_df['case_id'] == case_id) &
                                   (radiologist_df['with_segmentation'] == False)]
        radiologists = case_data['radiologist'].unique()

        if len(radiologists) > 1:
            for rad1, rad2 in combinations(radiologists, 2):
                rad1_pred = case_data[case_data['radiologist'] == rad1]['predicted_enhancement'].iloc[0]
                rad2_pred = case_data[case_data['radiologist'] == rad2]['predicted_enhancement'].iloc[0]
                agreements_without.append({
                    'rad1_pred': rad1_pred,
                    'rad2_pred': rad2_pred
                })

    all_rad1_preds = [agr['rad1_pred'] for agr in agreements_without]
    all_rad2_preds = [agr['rad2_pred'] for agr in agreements_without]
    kappa_rad_without = cohen_kappa_score(all_rad1_preds, all_rad2_preds)

    confusion_matrix_rad_without = np.zeros((2, 2), dtype=int)
    for agr in agreements_without:
        pred1 = int(agr['rad1_pred'])
        pred2 = int(agr['rad2_pred'])
        confusion_matrix_rad_without[pred1, pred2] += 1

    sns.heatmap(confusion_matrix_rad_without, annot=True, fmt='d', cmap='Blues', ax=ax_c,
               xticklabels=['No enhancement', 'Enhancement'],
               yticklabels=['No enhancement', 'Enhancement'], cbar=False)

    ax_c.set_title('c) Radiologist-radiologist agreement' + '\n' + 'Without support' + f' (κ = {kappa_rad_without:.3f})')
    ax_c.set_xlabel('Radiologist 2 prediction')
    ax_c.set_ylabel('Radiologist 1 prediction')

    # Panel d (row 1, col 1): Radiologist-Radiologist Agreement WITH support
    ax_d = fig_s1.add_subplot(gs[1, 1])

    multi_with_seg = multi_radiologist_cases[multi_radiologist_cases['with_segmentation'] == 1]

    agreements_with = []
    for case_id in multi_with_seg['case_id'].tolist():
        case_data = radiologist_df[(radiologist_df['case_id'] == case_id) &
                                   (radiologist_df['with_segmentation'] == True)]
        radiologists = case_data['radiologist'].unique()

        if len(radiologists) > 1:
            for rad1, rad2 in combinations(radiologists, 2):
                rad1_pred = case_data[case_data['radiologist'] == rad1]['predicted_enhancement'].iloc[0]
                rad2_pred = case_data[case_data['radiologist'] == rad2]['predicted_enhancement'].iloc[0]
                agreements_with.append({
                    'rad1_pred': rad1_pred,
                    'rad2_pred': rad2_pred
                })

    all_rad1_preds = [agr['rad1_pred'] for agr in agreements_with]
    all_rad2_preds = [agr['rad2_pred'] for agr in agreements_with]
    kappa_rad_with = cohen_kappa_score(all_rad1_preds, all_rad2_preds)

    confusion_matrix_rad_with = np.zeros((2, 2), dtype=int)
    for agr in agreements_with:
        pred1 = int(agr['rad1_pred'])
        pred2 = int(agr['rad2_pred'])
        confusion_matrix_rad_with[pred1, pred2] += 1

    sns.heatmap(confusion_matrix_rad_with, annot=True, fmt='d', cmap='Blues', ax=ax_d,
               xticklabels=['No enhancement', 'Enhancement'],
               yticklabels=['No enhancement', 'Enhancement'], cbar=False)

    ax_d.set_title('d) Radiologist-radiologist agreement' + '\n' + 'With support' + f' (κ = {kappa_rad_with:.3f})')
    ax_d.set_xlabel('Radiologist 2 prediction')
    ax_d.set_ylabel('Radiologist 1 prediction')

    edf4_path = os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_4.png')
    edf4_svg_path = os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_4.svg')
    plt.savefig(edf4_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(edf4_svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"\nExtended_Data_Fig_4 saved to: {edf4_path}")
    print(f"Extended_Data_Fig_4 saved to: {edf4_svg_path}")

    # ── Figure-displayed κ values ──
    print(f"\nFigure-displayed Cohen κ values (panel titles):")
    print(f"  a) Model-radiologist agreement, without support  κ = {kappa_without:.3f}")
    print(f"  b) Model-radiologist agreement, with support     κ = {kappa_with:.3f}")
    print(f"  c) Radiologist-radiologist agreement, without    κ = {kappa_rad_without:.3f}")
    print(f"  d) Radiologist-radiologist agreement, with       κ = {kappa_rad_with:.3f}")

    # Aggregate (rad-rad + rad-model) Cohen κ for manuscript paragraph 84
    n_rr_w = len(agreements_without); n_rr_m = len(agreements_with)
    n_mr_w = len(without_support_data); n_mr_m = len(with_support_data)
    agg_w  = (kappa_rad_without * n_rr_w + kappa_without * n_mr_w) / (n_rr_w + n_mr_w)
    agg_m  = (kappa_rad_with    * n_rr_m + kappa_with    * n_mr_m) / (n_rr_m + n_mr_m)
    print(f"\nAggregate pairwise Cohen κ across all comparisons (paragraph 84):")
    print(f"  Without support  κ_aggregate = {agg_w:.3f}  (rad-rad n={n_rr_w} + rad-model n={n_mr_w} = {n_rr_w+n_mr_w} pairs)")
    print(f"  With support     κ_aggregate = {agg_m:.3f}  (rad-rad n={n_rr_m} + rad-model n={n_mr_m} = {n_rr_m+n_mr_m} pairs)")

    # ── Bootstrap p-values for Cohen κ contrasts (paragraph 84) ──
    # Case-level paired bootstrap on the κ contrast for each comparison stream.
    # B=5000 with seed=20260505 to match other R1 bootstraps.
    print(f"\nBootstrap p-values for Cohen κ contrasts (paragraph 84, B=5000, seed=20260505):")
    rng = np.random.RandomState(20260505)
    B = 5000

    # 1) Rad-model contrast: bootstrap unique cases, recompute κ on resampled (model_pred, rad_pred) pairs
    mr_w = without_support_data[['case_id', 'predicted_enhancement', 'model_predicted_enhancement']].dropna().copy()
    mr_m = with_support_data[['case_id', 'predicted_enhancement', 'model_predicted_enhancement']].dropna().copy()
    mr_w_cases = mr_w['case_id'].unique(); mr_m_cases = mr_m['case_id'].unique()
    common_mr_cases = np.array(sorted(set(mr_w_cases) & set(mr_m_cases)))
    n_mr_cases = len(common_mr_cases)
    delta_mr_pt = kappa_with - kappa_without
    boot_mr = np.empty(B)
    mr_w_idx = {c: mr_w.index[mr_w['case_id'] == c].tolist() for c in common_mr_cases}
    mr_m_idx = {c: mr_m.index[mr_m['case_id'] == c].tolist() for c in common_mr_cases}
    for b in range(B):
        ix = rng.choice(n_mr_cases, size=n_mr_cases, replace=True)
        cases = common_mr_cases[ix]
        rows_w = []; rows_m = []
        for c in cases:
            rows_w.extend(mr_w_idx[c]); rows_m.extend(mr_m_idx[c])
        sub_w = mr_w.loc[rows_w]; sub_m = mr_m.loc[rows_m]
        kw = cohen_kappa_score(sub_w['model_predicted_enhancement'].values, sub_w['predicted_enhancement'].values)
        km = cohen_kappa_score(sub_m['model_predicted_enhancement'].values, sub_m['predicted_enhancement'].values)
        boot_mr[b] = km - kw
    p_mr = 2 * float(min((boot_mr <= 0).mean(), (boot_mr >= 0).mean()))
    lo_mr, hi_mr = float(np.percentile(boot_mr, 2.5)), float(np.percentile(boot_mr, 97.5))
    print(f"  Rad-model    Δκ = {delta_mr_pt:+.3f} [{lo_mr:+.3f}, {hi_mr:+.3f}]  Bootstrap p = {p_mr:.4f}  (n={n_mr_cases} common cases)")

    # 2) Rad-rad contrast: bootstrap unique cases, recompute κ on resampled rad-rad pair set
    rr_w_by_case = {}
    rr_m_by_case = {}
    for case_id in multi_without_seg['case_id'].tolist():
        case_data = radiologist_df[(radiologist_df['case_id'] == case_id) & (radiologist_df['with_segmentation'] == False)]
        rads = case_data['radiologist'].unique()
        if len(rads) > 1:
            for r1, r2 in combinations(rads, 2):
                p1 = int(case_data[case_data['radiologist'] == r1]['predicted_enhancement'].iloc[0])
                p2 = int(case_data[case_data['radiologist'] == r2]['predicted_enhancement'].iloc[0])
                rr_w_by_case.setdefault(case_id, []).append((p1, p2))
    for case_id in multi_with_seg['case_id'].tolist():
        case_data = radiologist_df[(radiologist_df['case_id'] == case_id) & (radiologist_df['with_segmentation'] == True)]
        rads = case_data['radiologist'].unique()
        if len(rads) > 1:
            for r1, r2 in combinations(rads, 2):
                p1 = int(case_data[case_data['radiologist'] == r1]['predicted_enhancement'].iloc[0])
                p2 = int(case_data[case_data['radiologist'] == r2]['predicted_enhancement'].iloc[0])
                rr_m_by_case.setdefault(case_id, []).append((p1, p2))
    common_rr_cases = np.array(sorted(set(rr_w_by_case.keys()) & set(rr_m_by_case.keys())))
    n_rr_cases = len(common_rr_cases)
    delta_rr_pt = kappa_rad_with - kappa_rad_without
    rng2 = np.random.RandomState(20260505)
    boot_rr = np.empty(B)
    for b in range(B):
        ix = rng2.choice(n_rr_cases, size=n_rr_cases, replace=True)
        cases = common_rr_cases[ix]
        pairs_w = []; pairs_m = []
        for c in cases:
            pairs_w.extend(rr_w_by_case[c]); pairs_m.extend(rr_m_by_case[c])
        a_w = np.array(pairs_w); a_m = np.array(pairs_m)
        kw = cohen_kappa_score(a_w[:, 0], a_w[:, 1])
        km = cohen_kappa_score(a_m[:, 0], a_m[:, 1])
        boot_rr[b] = km - kw
    p_rr = 2 * float(min((boot_rr <= 0).mean(), (boot_rr >= 0).mean()))
    lo_rr, hi_rr = float(np.percentile(boot_rr, 2.5)), float(np.percentile(boot_rr, 97.5))
    print(f"  Rad-rad      Δκ = {delta_rr_pt:+.3f} [{lo_rr:+.3f}, {hi_rr:+.3f}]  Bootstrap p = {p_rr:.4f}  (n={n_rr_cases} common cases)")

    # 3) Aggregate contrast
    delta_agg_pt = agg_m - agg_w
    rng3 = np.random.RandomState(20260505)
    boot_agg = np.empty(B)
    common_all = np.array(sorted(set(common_mr_cases) | set(common_rr_cases)))
    n_all = len(common_all)
    for b in range(B):
        ix = rng3.choice(n_all, size=n_all, replace=True)
        cases = common_all[ix]
        mr_pw = []; mr_pm = []; rr_pw = []; rr_pm = []
        for c in cases:
            if c in mr_w_idx:
                rows_w = mr_w_idx[c]; rows_m = mr_m_idx[c]
                mr_pw.extend(zip(mr_w.loc[rows_w, 'model_predicted_enhancement'].astype(int).tolist(),
                                 mr_w.loc[rows_w, 'predicted_enhancement'].astype(int).tolist()))
                mr_pm.extend(zip(mr_m.loc[rows_m, 'model_predicted_enhancement'].astype(int).tolist(),
                                 mr_m.loc[rows_m, 'predicted_enhancement'].astype(int).tolist()))
            if c in rr_w_by_case:
                rr_pw.extend(rr_w_by_case[c]); rr_pm.extend(rr_m_by_case[c])
        if not (mr_pw and mr_pm and rr_pw and rr_pm):
            boot_agg[b] = np.nan; continue
        a_mr_w = np.array(mr_pw); a_mr_m = np.array(mr_pm); a_rr_w = np.array(rr_pw); a_rr_m = np.array(rr_pm)
        kmr_w = cohen_kappa_score(a_mr_w[:, 0], a_mr_w[:, 1]); kmr_m = cohen_kappa_score(a_mr_m[:, 0], a_mr_m[:, 1])
        krr_w = cohen_kappa_score(a_rr_w[:, 0], a_rr_w[:, 1]); krr_m = cohen_kappa_score(a_rr_m[:, 0], a_rr_m[:, 1])
        n_mrw = len(a_mr_w); n_mrm = len(a_mr_m); n_rrw = len(a_rr_w); n_rrm = len(a_rr_m)
        agg_w_b = (krr_w * n_rrw + kmr_w * n_mrw) / (n_rrw + n_mrw)
        agg_m_b = (krr_m * n_rrm + kmr_m * n_mrm) / (n_rrm + n_mrm)
        boot_agg[b] = agg_m_b - agg_w_b
    boot_agg = boot_agg[~np.isnan(boot_agg)]
    p_agg = 2 * float(min((boot_agg <= 0).mean(), (boot_agg >= 0).mean()))
    lo_agg, hi_agg = float(np.percentile(boot_agg, 2.5)), float(np.percentile(boot_agg, 97.5))
    print(f"  Aggregate    Δκ = {delta_agg_pt:+.3f} [{lo_agg:+.3f}, {hi_agg:+.3f}]  Bootstrap p = {p_agg:.4f}  (n={len(boot_agg)} valid replicates)")

    # ── Per-pair κ direction split for paragraph 84 sentence ──
    # For each unique radiologist pair (rad-rad) and each radiologist (rad-model),
    # compute κ on shared cases with vs without support, and tally how many have
    # κ_with > κ_without. Manuscript claim: "77% higher with support, 23% without".
    print(f"\nPer-pair κ direction split (paragraph 84, '% higher with support'):")
    pair_results = []  # list of (pair_label, kappa_w, kappa_m)
    # Rad-rad pairs
    rad_ids = sorted(radiologist_df['radiologist'].unique())
    for r1, r2 in combinations(rad_ids, 2):
        # Build lists of (r1_pred, r2_pred) for each condition over shared cases
        cases_w = set(radiologist_df[(radiologist_df['radiologist'] == r1) & (radiologist_df['with_segmentation'] == False)]['case_id']) & \
                  set(radiologist_df[(radiologist_df['radiologist'] == r2) & (radiologist_df['with_segmentation'] == False)]['case_id'])
        cases_m = set(radiologist_df[(radiologist_df['radiologist'] == r1) & (radiologist_df['with_segmentation'] == True)]['case_id']) & \
                  set(radiologist_df[(radiologist_df['radiologist'] == r2) & (radiologist_df['with_segmentation'] == True)]['case_id'])
        if len(cases_w) < 2 or len(cases_m) < 2:
            continue
        rw1, rw2 = [], []; rm1, rm2 = [], []
        for c in cases_w:
            rw1.append(int(radiologist_df[(radiologist_df['radiologist'] == r1) & (radiologist_df['with_segmentation'] == False) & (radiologist_df['case_id'] == c)]['predicted_enhancement'].iloc[0]))
            rw2.append(int(radiologist_df[(radiologist_df['radiologist'] == r2) & (radiologist_df['with_segmentation'] == False) & (radiologist_df['case_id'] == c)]['predicted_enhancement'].iloc[0]))
        for c in cases_m:
            rm1.append(int(radiologist_df[(radiologist_df['radiologist'] == r1) & (radiologist_df['with_segmentation'] == True) & (radiologist_df['case_id'] == c)]['predicted_enhancement'].iloc[0]))
            rm2.append(int(radiologist_df[(radiologist_df['radiologist'] == r2) & (radiologist_df['with_segmentation'] == True) & (radiologist_df['case_id'] == c)]['predicted_enhancement'].iloc[0]))
        try:
            kw = cohen_kappa_score(rw1, rw2); km = cohen_kappa_score(rm1, rm2)
        except Exception:
            continue
        if np.isnan(kw) or np.isnan(km):
            continue
        pair_results.append((f"R{r1}-R{r2}", kw, km))
    # Rad-model pairs
    for r in rad_ids:
        sub_w = radiologist_df[(radiologist_df['radiologist'] == r) & (radiologist_df['with_segmentation'] == False)].dropna(subset=['model_predicted_enhancement'])
        sub_m = radiologist_df[(radiologist_df['radiologist'] == r) & (radiologist_df['with_segmentation'] == True)].dropna(subset=['model_predicted_enhancement'])
        if len(sub_w) < 2 or len(sub_m) < 2:
            continue
        try:
            kw = cohen_kappa_score(sub_w['model_predicted_enhancement'].values, sub_w['predicted_enhancement'].values)
            km = cohen_kappa_score(sub_m['model_predicted_enhancement'].values, sub_m['predicted_enhancement'].values)
        except Exception:
            continue
        if np.isnan(kw) or np.isnan(km):
            continue
        pair_results.append((f"R{r}-M", kw, km))
    n_pairs_total = len(pair_results)
    n_higher_with = sum(1 for _, kw, km in pair_results if km > kw)
    n_higher_without = sum(1 for _, kw, km in pair_results if kw > km)
    n_tied = n_pairs_total - n_higher_with - n_higher_without
    pct_with = 100.0 * n_higher_with / n_pairs_total
    pct_without = 100.0 * n_higher_without / n_pairs_total
    print(f"  Of {n_pairs_total} unique reader-pair κ comparisons:")
    print(f"    κ higher WITH support:    {n_higher_with} ({pct_with:.1f}%)")
    print(f"    κ higher WITHOUT support: {n_higher_without} ({pct_without:.1f}%)")
    if n_tied:
        print(f"    Tied:                     {n_tied} ({100.0 * n_tied / n_pairs_total:.1f}%)")


if __name__ == '__main__':
    main()
