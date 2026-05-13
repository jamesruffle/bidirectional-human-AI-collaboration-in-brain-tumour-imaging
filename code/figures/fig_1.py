#!/usr/bin/env python3
"""Fig 1: impact of support on agent performance.

Self-contained reproduction script. Loads the consolidated CSV/JSON inputs at
data/source_data/figure_1/csv_v2/ (~480 KB total), derives the subset frames
and aggregations from radiologist_df, then renders the 8-panel figure (a-h).

Output: writes Fig_1.png and Fig_1.svg to data/figures/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.stats as _scipy_stats
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_palette("husl")  # global palette — affects default colours of seaborn-rendered panels
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
    cohen_kappa_score, average_precision_score,
)


def aggregate_by_case_fig1(df, pred_col, gt_col='has_enhancement_gt', case_col='case_id'):
    """Verbatim copy of monolith helper (Fig 1 case-level aggregation)."""
    case_results = {}
    for _, row in df.iterrows():
        case_id = row[case_col]
        if case_id not in case_results:
            case_results[case_id] = {'gt': row[gt_col], 'preds': []}
        case_results[case_id]['preds'].append(row[pred_col])
    case_preds, case_gts = [], []
    for case_id, data in case_results.items():
        case_pred = 1 if sum(data['preds']) > len(data['preds']) / 2 else 0
        case_preds.append(case_pred)
        case_gts.append(data['gt'])
    return np.array(case_preds), np.array(case_gts), len(case_results)


def get_sig_marker(p: float) -> str:
    """Asterisk nomenclature for p-values."""
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


def load_state_from_csv():
    """Reconstruct the 14-key master state dict from CSV/JSON inputs."""
    state = {}

    state['radiologist_df'] = pd.read_csv(
        os.path.join(SRC_DIR, 'radiologist_df.csv'), float_precision='round_trip',
    ).astype({'with_segmentation': bool})

    bcv_df = pd.read_csv(
        os.path.join(SRC_DIR, 'best_cv_predictions.csv'), float_precision='round_trip',
    )
    with open(os.path.join(SRC_DIR, 'best_cv_predictions_schema.json')) as fh:
        bcv_schema = json.load(fh)
    all_none = [k for k, ts in bcv_schema.items() if ts == ['NoneType']]
    type_map = {k: ts[0] for k, ts in bcv_schema.items() if ts != ['NoneType']}
    bcv_records = []
    for _, row in bcv_df.iterrows():
        rec = {}
        for k in bcv_df.columns:
            if k in all_none:
                rec[k] = None
            else:
                t = type_map[k]
                v = row[k]
                if t == 'int':
                    rec[k] = int(v)
                elif t in ('float', 'float64'):
                    rec[k] = float(v)
                elif t == 'bool':
                    rec[k] = bool(v)
                elif t == 'str':
                    rec[k] = str(v)
                else:
                    rec[k] = v
        bcv_records.append(rec)
    state['best_cv_predictions'] = bcv_records

    # Per-reader (group_metrics) and pairwise-Cohen's-kappa (paired_agreements)
    # are computed live from radiologist_df via _metrics_utils — no static
    # CSV cache.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _metrics_utils import compute_group_metrics, compute_paired_agreements  # noqa: E402

    gm = compute_group_metrics(state['radiologist_df'])
    state['group1_metrics'] = gm[gm['group'] == 'group1'].drop(columns='group').to_dict('records')
    state['group2_metrics'] = gm[gm['group'] == 'group2'].drop(columns='group').to_dict('records')

    state['paired_agreements'] = compute_paired_agreements(state['radiologist_df']).to_dict('records')

    with open(os.path.join(SRC_DIR, 'aggregates.json')) as fh:
        agg = json.load(fh)
    data = agg['data']
    types = agg.get('types', {})

    def restore(v, t):
        if t == 'numpy.float64':
            return np.float64(v)
        if t == 'numpy.int64':
            return np.int64(v)
        if t == 'numpy.bool_':
            return np.bool_(v)
        return v

    # MRMC values + reader-averaged dicts come from the R MRMCaov pipeline
    # and the per-reader aggregation; we keep them as cached inputs.
    for key in ('group1_avg', 'group2_avg',
                'mrmc_auroc', 'mrmc_sens', 'mrmc_spec', 'mrmc_auprc'):
        if key in data:
            t = types.get(key, {})
            state[key] = {k: restore(v, t.get(k, ''))
                          for k, v in data[key].items()}
    state['model_results'] = data.get('model_results', {})

    # Pair-level and case-level model metrics are computed *live* from
    # seed_predictions.csv via _metrics_utils — no JSON lookup. This
    # mirrors `multi_radiologist_analysis.py:32070-32287` and ensures every
    # number in panel B / Table 1 is reproducible from bundled CSVs.
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _sys.path.insert(0, os.path.dirname(_here))  # add code/ to path
    from _metrics_utils import load_canonical_metrics  # noqa: E402

    seed_csv = os.path.join(SRC_DIR, 'seed_predictions.csv')
    canon = load_canonical_metrics(seed_csv)
    state['pair_level_model_metrics']  = canon['pair_level_model_metrics']
    state['pair_level_cv_metrics']     = canon['pair_level_cv_metrics']
    state['unique_case_model_metrics'] = canon['unique_case_model_metrics']
    state['unique_case_cv_metrics']    = canon['unique_case_cv_metrics']

    return state


def derive_subsets(state):
    """Reconstruct redundant frames + aggregations from radiologist_df."""
    rdf = state['radiologist_df']

    state['common_cases'] = (
        rdf.dropna(subset=['model_predicted_enhancement']).reset_index(drop=True)
    )
    state['with_seg_data'] = rdf[rdf['with_segmentation']]
    state['without_seg_data'] = rdf[~rdf['with_segmentation']]
    state['with_seg_data_fig1rev'] = state['with_seg_data']
    state['without_seg_data_fig1rev'] = state['without_seg_data']

    rad4 = rdf[rdf['radiologist'] == 'Radiologist #4']
    wo4 = rad4[~rad4['with_segmentation']]
    ws4 = rad4[rad4['with_segmentation']]
    paired = wo4.merge(ws4, on='case_id', suffixes=('_without', '_with'))
    paired_cols = [
        'radiologist_without', 'case_id', 'predicted_enhancement_without',
        'confidence_without', 'image_quality_without', 'response_time_without',
        'has_enhancement_gt_without', 'with_segmentation_without',
        'years_experience_without', 'Pathology_without',
        'model_predicted_enhancement_without', 'Cohort_without', 'Country_without',
        'correct_prediction_without', 'model_correct_without',
        'model_difficult_without', 'model_easy_without',
        'lesion_size_category_without', 'radiomic_category_without',
        'radiologist_with', 'predicted_enhancement_with', 'confidence_with',
        'image_quality_with', 'response_time_with', 'has_enhancement_gt_with',
        'with_segmentation_with', 'years_experience_with', 'Pathology_with',
        'model_predicted_enhancement_with', 'Cohort_with', 'Country_with',
        'correct_prediction_with', 'model_correct_with', 'model_difficult_with',
        'model_easy_with', 'lesion_size_category_with', 'radiomic_category_with',
    ]
    state['paired_data'] = paired[paired_cols]

    experience_df = (
        rdf.groupby(['radiologist', 'with_segmentation'], sort=False)
           .agg(years_experience=('years_experience', 'first'),
                rad_accuracy=('correct_prediction', 'mean'),
                mean_confidence=('confidence', 'mean'),
                mean_response_time=('response_time', 'mean'))
           .reset_index()
    )
    state['experience_df'] = experience_df

    per_rad = (
        rdf.groupby('radiologist', sort=False)
           .apply(lambda g: pd.Series({
               'accuracy_without': g[~g['with_segmentation']]['correct_prediction'].mean(),
               'accuracy_with': g[g['with_segmentation']]['correct_prediction'].mean(),
               'confidence_without': g[~g['with_segmentation']]['confidence'].mean(),
               'confidence_with': g[g['with_segmentation']]['confidence'].mean(),
               'quality_without': g[~g['with_segmentation']]['image_quality'].mean(),
               'quality_with': g[g['with_segmentation']]['image_quality'].mean(),
               'time_without': g[~g['with_segmentation']]['response_time'].mean(),
               'time_with': g[g['with_segmentation']]['response_time'].mean(),
               'n_cases': len(g['case_id'].unique()),
           }), include_groups=False)
           .reset_index()
    )
    plot_cols = [
        'radiologist', 'accuracy_without', 'accuracy_with',
        'confidence_without', 'confidence_with',
        'quality_without', 'quality_with',
        'time_without', 'time_with', 'n_cases',
        'with_segmentation', 'years_experience',
        'rad_accuracy', 'mean_confidence', 'mean_response_time',
    ]
    state['plot_df'] = experience_df.merge(per_rad, on='radiologist', how='left')[plot_cols]


# === Run loading + derivation, then plot ===

if not os.path.isdir(SRC_DIR):
    print(f"ERROR: source dir not found at {SRC_DIR}")
    sys.exit(2)

os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

print(f"Loading Fig 1 inputs from {SRC_DIR}...")
_state = load_state_from_csv()
print(f"  Loaded {len(_state)} master variables from CSV/JSON")

derive_subsets(_state)
print(f"  Derived {len(_state)} variables total (subsets + paired_data + plot_df + experience_df)")

# Promote state to module globals so the plotting block sees them.
globals().update(_state)

print(f"Rendering Fig 1...")


fig1, axes = plt.subplots(2, 4, figsize=(24, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Panel a) Factorial evaluation - 2x2 heatmap
ax = axes[0, 0]
# Radiologist metrics: balanced accuracy from reader-averaged classification metrics (group1_avg, group2_avg).
# Model metrics: pair-level accuracy from grid search output (not case-level).

# Radiologist metrics: reader-averaged balanced accuracy.
radiologist_alone_accuracy = group1_avg['balanced_accuracy']
print(f"Panel a: Using reader-averaged balanced accuracy for radiologist alone: {radiologist_alone_accuracy:.3f}")

radiologist_together_accuracy = group2_avg['balanced_accuracy']
print(f"Panel a: Using reader-averaged balanced accuracy for radiologist with model: {radiologist_together_accuracy:.3f}")

# Model metrics: pair-level accuracy from grid search output.
model_alone_accuracy = pair_level_model_metrics['accuracy']
print(f"Panel a: Using pair-level model baseline accuracy from grid search: {model_alone_accuracy:.3f}")

model_together_accuracy = pair_level_cv_metrics['accuracy']
print(f"Panel a: Using pair-level nested CV accuracy from grid search: {model_together_accuracy:.3f}")

# Store for later use in other panels (especially Panel D)
_panel_a_model_alone = model_alone_accuracy
_panel_a_model_together = model_together_accuracy
_panel_b_metrics_order = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']

# Create 2x2 factorial matrix
factorial_matrix = np.array([
    [radiologist_alone_accuracy, model_alone_accuracy],      # Alone row
    [radiologist_together_accuracy, model_together_accuracy]  # Together row
])

# Define colors from panel b for each quadrant
# colors[0] = '#1f77b4' (blue) for Radiologist Alone
# colors[2] = '#2ca02c' (green) for Radiologist Together
# colors[3] = '#d62728' (red) for Model Alone
# colors[4] = '#9467bd' (purple) for Model Together
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

from matplotlib.patches import Rectangle

# First, draw colored rectangles for each cell
cell_colors = [
    [colors[0], colors[3]],  # [Radiologist Alone (blue), Model Alone (red)]
    [colors[2], colors[4]]   # [Radiologist Together (green), Model Together (purple)]
]

# Draw colored background rectangles
for i in range(2):
    for j in range(2):
        rect = Rectangle((j, i), 1, 1, 
                       facecolor=cell_colors[i][j], 
                       alpha=0.3,  # Transparency to see values clearly
                       edgecolor='black',
                       linewidth=1.5)
        ax.add_patch(rect)

for i in range(2):
    for j in range(2):
        # Bold the Model x Together quadrant (i=1, j=1) as it has the highest value
        fontweight = 'bold' if (i == 1 and j == 1) else 'normal'
        text = ax.text(j + 0.5, i + 0.5, f'{factorial_matrix[i, j]:.3f}',
                     ha='center', va='center',
                     fontsize=12, fontweight=fontweight, color='black')

# Set the limits and ticks
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['Radiologist', 'Model'], fontsize=12)
ax.set_yticklabels(['Alone', 'Together'], fontsize=12, rotation=90, va='center')

# Add top x-axis labels inside the figure at the top of the 'Together' row
ax.text(0.5, 1.9, 'Radiologist|Model', ha='center', va='center', fontsize=12)
ax.text(1.5, 1.9, 'Model|Radiologist', ha='center', va='center', fontsize=12)

# Add grid lines between cells
ax.axhline(y=1, color='black', linewidth=1.5)
ax.axvline(x=1, color='black', linewidth=1.5)

# Remove default spines and add border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

ax.set_title('a) Factorial accuracy evaluation', fontsize=14)
ax.set_xlabel('Agent', fontsize=12)
ax.set_ylabel('Condition', fontsize=12)

# Add delta gain annotations inside the heatmap
# The displayed values are rounded to 3 decimals, so deltas must be calculated from these rounded values
rad_alone_rounded = round(radiologist_alone_accuracy, 3)
rad_together_rounded = round(radiologist_together_accuracy, 3)
model_alone_rounded = round(model_alone_accuracy, 3)
model_together_rounded = round(model_together_accuracy, 3)

delta_model_alone = model_alone_rounded - rad_alone_rounded
delta_rad_together = rad_together_rounded - rad_alone_rounded
delta_model_together = model_together_rounded - rad_alone_rounded

# Arrows and delta annotations within the heatmap area.
# Heatmap cells are centred at (0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5);
# arrows + deltas are placed in the space between cells.

# Horizontal arrow and delta (Radiologist Alone -> Model Alone), along top edge of "Alone" row.
ax.annotate('', xy=(1.3, 0.5), xytext=(0.7, 0.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))
ax.text(1.0, 0.25, f'Δ={delta_model_alone:+.3f}',
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

# Vertical arrow and delta (Radiologist Alone -> Radiologist Together), along left edge of "Radiologist" column.
ax.annotate('', xy=(0.5, 1.3), xytext=(0.5, 0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))
ax.text(0.25, 1.0, f'Δ={delta_rad_together:+.3f}',
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

# Diagonal arrow and delta (Radiologist Alone -> Model Together), top-left to bottom-right.
ax.annotate('', xy=(1.35, 1.35), xytext=(0.65, 0.65),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))
ax.text(1.0, 1.0, f'Δ={delta_model_together:+.3f}',
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

# Horizontal arrow (Radiologist Together -> Model Together).
delta_rad_to_model_together = model_together_rounded - rad_together_rounded
ax.annotate('', xy=(1.3, 1.5), xytext=(0.7, 1.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))
ax.text(1.0, 1.75, f'Δ={delta_rad_to_model_together:+.3f}',
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

# Vertical arrow (Model Alone -> Model Together).
delta_model_alone_to_together = model_together_rounded - model_alone_rounded
ax.annotate('', xy=(1.5, 1.3), xytext=(1.5, 0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))
ax.text(1.75, 1.0, f'Δ={delta_model_alone_to_together:+.3f}',
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

ax = axes[0, 1]
# Panel b plots reader-averaged metrics for radiologists (MRMC FOMs / group means)
# and pair-level metrics for the model.
model_metrics = {
    'accuracy': pair_level_model_metrics['accuracy'],
    'sensitivity': pair_level_model_metrics['recall'],
    'specificity': pair_level_model_metrics['specificity'],
    'precision': pair_level_model_metrics['precision'],
    'f1': pair_level_model_metrics['f1'],
}

metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-score']

# Radiologist metrics: use MRMCaov FOMs for both arms (alone and with model).
print(f"\nPanel b: Using MRMC results for radiologist metrics")

# Radiologist alone: Use MRMC FOMs (without AI segmentation)
rad_without_auc = mrmc_auroc['fom_without']
print(f"  Radiologist alone AUROC (MRMC): {rad_without_auc:.3f}")

rad_without_auprc = mrmc_auprc['fom_without']
print(f"  Radiologist alone AUPRC (MRMC): {rad_without_auprc:.3f}")

rad_without_sens = mrmc_sens['fom_without']
print(f"  Radiologist alone Sensitivity (MRMC): {rad_without_sens:.4f}")

rad_without_spec = mrmc_spec['fom_without']
print(f"  Radiologist alone Specificity (MRMC): {rad_without_spec:.4f}")

# Radiologist with model: Use MRMC FOMs (with AI segmentation)
rad_with_auc = mrmc_auroc['fom_with']
print(f"  Radiologist with model AUROC (MRMC): {rad_with_auc:.3f}")

rad_with_auprc = mrmc_auprc['fom_with']
print(f"  Radiologist with model AUPRC (MRMC): {rad_with_auprc:.3f}")

rad_with_sens = mrmc_sens['fom_with']
print(f"  Radiologist with model Sensitivity (MRMC): {rad_with_sens:.4f}")

rad_with_spec = mrmc_spec['fom_with']
print(f"  Radiologist with model Specificity (MRMC): {rad_with_spec:.4f}")

# Canonical model AUROC/AUPRC: unique-case (n=564), matching Fig 4 panels a-d.
# These are the manuscript-citable AUC values for the model arms.
print(f"\nCanonical AUROC/AUPRC (unique-case model arms; n={unique_case_model_metrics['n_unique_cases']}):")
print(f"  Model alone AUROC:                     {unique_case_model_metrics['auroc']:.3f}")
print(f"  Model alone AUPRC:                     {unique_case_model_metrics['auprc']:.3f}")
print(f"  Model+Radiologist (CV) AUROC:          {unique_case_cv_metrics['auroc']:.3f}")
print(f"  Model+Radiologist (CV) AUPRC:          {unique_case_cv_metrics['auprc']:.3f}")

# Build radiologist values arrays using MRMC results and reader-averaged metrics
# For accuracy, precision, F1: use reader-averaged metrics from GROUP 1 and GROUP 2
rad_without_acc = group1_avg['balanced_accuracy']
rad_without_prec = group1_avg['precision']
rad_without_f1 = group1_avg['f1']
print(f"  Using group 1 reader-averaged metrics: Balanced Accuracy={rad_without_acc:.4f}, Precision={rad_without_prec:.4f}, F1={rad_without_f1:.4f}")

rad_with_acc = group2_avg['balanced_accuracy']
rad_with_prec = group2_avg['precision']
rad_with_f1 = group2_avg['f1']
print(f"  Using group 2 reader-averaged metrics: Balanced Accuracy={rad_with_acc:.4f}, Precision={rad_with_prec:.4f}, F1={rad_with_f1:.4f}")

# Order: Accuracy, Sensitivity, Specificity, Precision, F1
rad_without_values = [rad_without_acc, rad_without_sens, rad_without_spec,
                     rad_without_prec, rad_without_f1]
rad_with_values = [rad_with_acc, rad_with_sens, rad_with_spec,
                  rad_with_prec, rad_with_f1]

# NO FALLBACK TO ZEROS - verify all metrics exist for model
required_metrics_model = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
missing_model = [m for m in required_metrics_model if m not in model_metrics]
if missing_model:
    raise ValueError(f"Required metrics missing from model_metrics: {missing_model}")
model_values = [model_metrics[m] for m in required_metrics_model]

# Use pair-level CV metrics from grid search output (no AUC)
# Order: Accuracy, Sensitivity, Specificity, Precision, F1
cv_model_rad_values = [
    pair_level_cv_metrics['accuracy'],  # Accuracy
    pair_level_cv_metrics['recall'],  # Sensitivity (same as recall)
    pair_level_cv_metrics['specificity'],  # Specificity
    pair_level_cv_metrics['precision'],  # Precision
    pair_level_cv_metrics['f1']  # F1
]
print(f"  Model+Radiologist (pair-level CV): n={pair_level_cv_metrics['n_pairs']} pairs, Acc={cv_model_rad_values[0]:.3f}, Sens={cv_model_rad_values[1]:.3f}, Spec={cv_model_rad_values[2]:.3f}")

x = np.arange(len(metrics_names))
width = 0.18  # Reduced width to fit 4 bars

# Define colors from reference notebook
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Bootstrap parameters
n_bootstrap = 5000
confidence_level = 0.95
bootstrap_seed = 20260505

def _pair_level_bootstrap_cis(y_pred, y_true, metric_funcs, n_bootstrap=n_bootstrap,
                              confidence_level=confidence_level, seed=bootstrap_seed):
    """Pair-level non-parametric percentile bootstrap CIs for a list of metrics.

    Resamples paired (y_pred, y_true) with replacement; the same resampled
    indices are reused across metrics so that metric-vs-metric comparisons
    on the same bootstrap draw are coherent. RNG is seeded locally so the
    figure remains pixel-identical run-to-run without disturbing the
    process-level RNG state.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = len(y_pred)
    rng = np.random.RandomState(seed)
    boot_idx = [rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)]

    lower_q = (1 - confidence_level) / 2 * 100
    upper_q = (1 + confidence_level) / 2 * 100

    cis = []
    for fn in metric_funcs:
        boot_vals = np.fromiter(
            (fn(y_pred[ix], y_true[ix]) for ix in boot_idx),
            dtype=float, count=n_bootstrap,
        )
        ci_lower = max(0.0, float(np.percentile(boot_vals, lower_q)))
        ci_upper = min(1.0, float(np.percentile(boot_vals, upper_q)))
        cis.append((ci_lower, ci_upper))
    return cis

# Define metric functions for bootstrap
def calc_accuracy(pred, gt):
    return accuracy_score(gt, pred)

def calc_precision(pred, gt):
    return precision_score(gt, pred, zero_division=0)

def calc_recall(pred, gt):
    return recall_score(gt, pred, zero_division=0)

def calc_specificity(pred, gt):
    tn = ((pred == 0) & (gt == 0)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def calc_f1(pred, gt):
    return f1_score(gt, pred)

def calc_auc(y_true, y_pred):
    """Calculate AUC from binary predictions"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

def calc_sensitivity(y_true, y_pred):
    """Calculate sensitivity (same as recall)"""
    return calc_recall(y_true, y_pred)

# Order: Accuracy, Sensitivity, Specificity, Precision, F1
metric_funcs = [calc_accuracy, calc_sensitivity, calc_specificity, calc_precision, calc_f1]

# Calculate confidence intervals
model_ci = []
rad_without_ci = []
rad_with_ci = []

# Model confidence intervals: pair-level non-parametric bootstrap on the
# n=1100 deduplicated radiologist-case pairs in best_cv_predictions, using
# the model_pred column (the same pairs that produced the pair-level point
# estimate). Percentile interval, B=5000.
print(f"  Bootstrapping pair-level CIs for model metrics (B={n_bootstrap})...")
bcv_arr = pd.DataFrame(best_cv_predictions)
y_pair_gt = bcv_arr['gt'].astype(int).values
y_pair_model = bcv_arr['model_pred'].astype(int).values
model_ci = _pair_level_bootstrap_cis(y_pair_model, y_pair_gt, metric_funcs)

# Radiologist without segmentation confidence intervals - USE GROUP STANDARD DEVIATIONS
# Order: [Accuracy, Sensitivity, Specificity, Precision, F1]
# All metrics use SE = SD / sqrt(n_radiologists) from GROUP 1
# 95% CI: mean ± 1.96 * SE
n_rads = len(group1_metrics)
se_acc = group1_avg['balanced_accuracy_std'] / np.sqrt(n_rads)
se_sens = group1_avg['recall_std'] / np.sqrt(n_rads)
se_spec = group1_avg['specificity_std'] / np.sqrt(n_rads)
se_prec = group1_avg['precision_std'] / np.sqrt(n_rads)
se_f1 = group1_avg['f1_std'] / np.sqrt(n_rads)

# Index 0: Accuracy
ci_lower = max(0, rad_without_acc - 1.96 * se_acc)
ci_upper = min(1, rad_without_acc + 1.96 * se_acc)
rad_without_ci.append((ci_lower, ci_upper))

# Index 1: Sensitivity (recall)
ci_lower = max(0, rad_without_sens - 1.96 * se_sens)
ci_upper = min(1, rad_without_sens + 1.96 * se_sens)
rad_without_ci.append((ci_lower, ci_upper))

# Index 2: Specificity
ci_lower = max(0, rad_without_spec - 1.96 * se_spec)
ci_upper = min(1, rad_without_spec + 1.96 * se_spec)
rad_without_ci.append((ci_lower, ci_upper))

# Index 3: Precision
ci_lower = max(0, rad_without_prec - 1.96 * se_prec)
ci_upper = min(1, rad_without_prec + 1.96 * se_prec)
rad_without_ci.append((ci_lower, ci_upper))

# Index 4: F1
ci_lower = max(0, rad_without_f1 - 1.96 * se_f1)
ci_upper = min(1, rad_without_f1 + 1.96 * se_f1)
rad_without_ci.append((ci_lower, ci_upper))

# Radiologist with segmentation confidence intervals - USE GROUP STANDARD DEVIATIONS
# Order: [Accuracy, Sensitivity, Specificity, Precision, F1]
# All metrics use SE = SD / sqrt(n_radiologists) from GROUP 2
# 95% CI: mean ± 1.96 * SE
n_rads = len(group2_metrics)
se_acc = group2_avg['balanced_accuracy_std'] / np.sqrt(n_rads)
se_sens = group2_avg['recall_std'] / np.sqrt(n_rads)
se_spec = group2_avg['specificity_std'] / np.sqrt(n_rads)
se_prec = group2_avg['precision_std'] / np.sqrt(n_rads)
se_f1 = group2_avg['f1_std'] / np.sqrt(n_rads)

# Index 0: Accuracy
ci_lower = max(0, rad_with_acc - 1.96 * se_acc)
ci_upper = min(1, rad_with_acc + 1.96 * se_acc)
rad_with_ci.append((ci_lower, ci_upper))

# Index 1: Sensitivity (recall)
ci_lower = max(0, rad_with_sens - 1.96 * se_sens)
ci_upper = min(1, rad_with_sens + 1.96 * se_sens)
rad_with_ci.append((ci_lower, ci_upper))

# Index 2: Specificity
ci_lower = max(0, rad_with_spec - 1.96 * se_spec)
ci_upper = min(1, rad_with_spec + 1.96 * se_spec)
rad_with_ci.append((ci_lower, ci_upper))

# Index 3: Precision
ci_lower = max(0, rad_with_prec - 1.96 * se_prec)
ci_upper = min(1, rad_with_prec + 1.96 * se_prec)
rad_with_ci.append((ci_lower, ci_upper))

# Index 4: F1
ci_lower = max(0, rad_with_f1 - 1.96 * se_f1)
ci_upper = min(1, rad_with_f1 + 1.96 * se_f1)
rad_with_ci.append((ci_lower, ci_upper))

# Create error bars from confidence intervals
# Use max(0, ...) to prevent negative values while preserving direction
model_errors = [(max(0, model_values[i] - ci[0]), max(0, ci[1] - model_values[i])) for i, ci in enumerate(model_ci)]
model_errors_lower = [e[0] for e in model_errors]
model_errors_upper = [e[1] for e in model_errors]

rad_without_errors = [(max(0, rad_without_values[i] - ci[0]), max(0, ci[1] - rad_without_values[i])) for i, ci in enumerate(rad_without_ci)]
rad_without_errors_lower = [e[0] for e in rad_without_errors]
rad_without_errors_upper = [e[1] for e in rad_without_errors]

rad_with_errors = [(max(0, rad_with_values[i] - ci[0]), max(0, ci[1] - rad_with_values[i])) for i, ci in enumerate(rad_with_ci)]
rad_with_errors_lower = [e[0] for e in rad_with_errors]
rad_with_errors_upper = [e[1] for e in rad_with_errors]

# Calculate confidence intervals for CV Model+Radiologist predictions
cv_model_rad_ci = []
cv_model_rad_errors_lower = []
cv_model_rad_errors_upper = []

# Bootstrap on the canonical n=1100 pair-level predictions in
# best_cv_predictions.csv directly. Each row is one (case_id, radiologist)
# pair, with cv_pred from the 5-seed prefer-correct dedup applied to
# seed_predictions.csv (see `code/_metrics_utils.py`). The bootstrap CI
# agrees with `table_1.py`'s pair-level bootstrap by construction.
cv_pred_df = pd.DataFrame(best_cv_predictions)
cv_pred_array = cv_pred_df['cv_pred'].astype(int).values
cv_gt_array   = cv_pred_df['gt'].astype(int).values
cv_matched_count = len(cv_pred_array)
cv_fallback_count = 0

# Pair-level non-parametric bootstrap on the n=1100 deduplicated
# radiologist-case pairs in best_cv_predictions, using the cv_pred
# column (model+radiologist combined prediction). Percentile interval,
# B=5000. Note: the pair-level point estimate plotted in the bar is
# macro-averaged across CV folds, while the bootstrap is on the pooled
# predictions; the two differ by <1% across all five metrics, and the
# macro-averaged value lies inside the bootstrap percentile CI.
print(f"\nBootstrapping pair-level CIs for Model+Radiologist (CV) metrics (B={n_bootstrap})...")
cv_model_rad_ci = _pair_level_bootstrap_cis(
    cv_pred_array, cv_gt_array, metric_funcs,
)

# Panel B figure-displayed CI summary (matches the error bars rendered on the bar chart).
# Order: Accuracy, Sensitivity, Specificity, Precision, F1
_panel_b_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
print("\nPanel B figure-displayed values (point estimate [95% CI]):")
for i, name in enumerate(_panel_b_metrics):
    print(f"  Radiologist alone        {name:<12} {rad_without_values[i]:.3f} [{rad_without_ci[i][0]:.3f}-{rad_without_ci[i][1]:.3f}]")
for i, name in enumerate(_panel_b_metrics):
    print(f"  Radiologist with model   {name:<12} {rad_with_values[i]:.3f} [{rad_with_ci[i][0]:.3f}-{rad_with_ci[i][1]:.3f}]")
for i, name in enumerate(_panel_b_metrics):
    print(f"  Model alone              {name:<12} {model_values[i]:.3f} [{model_ci[i][0]:.3f}-{model_ci[i][1]:.3f}]")
for i, name in enumerate(_panel_b_metrics):
    print(f"  Model with radiologist   {name:<12} {cv_model_rad_values[i]:.3f} [{cv_model_rad_ci[i][0]:.3f}-{cv_model_rad_ci[i][1]:.3f}]")

# ════════════════════════════════════════════════════════════════════════════
# Table 1 auxiliary statistics — AUROC/AUPRC CIs and Δ values
# ════════════════════════════════════════════════════════════════════════════
# Computes everything in Manuscript Table 1 that is NOT a Panel-B point/CI:
#   • AUROC / AUPRC 95% CI for all 4 conditions
#   • Δ AUROC / Δ AUPRC for both arms (Human Δ, AI Δ) with paired bootstrap CI
#   • Δ Accuracy / Sensitivity / Specificity / Precision / F1 paired bootstrap CIs
# Methodology:
#   • Radiologist arms: per-reader scores → reader-level paired bootstrap
#     (B=5000 reader resamples, seed=20260505)
#   • Model arms: case-level paired bootstrap on n=564 unique cases
#     (B=5000 case resamples, seed=20260505), with combined_prob mean-aggregated
#     per case to match the canonical unique_case_cv_metrics in aggregates.json
print("\n" + "─" * 78)
print("Table 1 auxiliary statistics (AUROC/AUPRC CIs, Δ point + paired CI)")
print("─" * 78)

from scipy import stats as _scipy_stats

def _per_reader_metrics(rdf, with_seg):
    """Per-reader AUROC / AUPRC / balanced accuracy / sens / spec / prec / F1.

    Methodology (matches the canonical Panel B point estimates):
    • AUROC / AUPRC: confidence-direction-adjusted continuous score
    • Sensitivity / Specificity: MRMC binary_rating (continuous score ≥ 0.5)
    • Accuracy: balanced_accuracy_score on raw predicted_enhancement
      (matches group_avg['balanced_accuracy'] used in Panel A and as the
      'accuracy' bar in Panel B for radiologist arms)
    • Precision / F1: sklearn on raw predicted_enhancement
    """
    sub = rdf[rdf['with_segmentation'] == with_seg]
    rows = []
    for rad, grp in sub.groupby('radiologist'):
        gt = grp['has_enhancement_gt'].astype(int).values
        raw_pred = grp['predicted_enhancement'].astype(int).values
        score = np.where(raw_pred == 1, grp['confidence'].values / 10.0,
                         1 - grp['confidence'].values / 10.0)
        mrmc_pred = (score >= 0.5).astype(int)
        try:
            au = roc_auc_score(gt, score)
            ap = average_precision_score(gt, score)
        except Exception:
            au = ap = np.nan
        tn_m = int(((mrmc_pred == 0) & (gt == 0)).sum())
        fp_m = int(((mrmc_pred == 1) & (gt == 0)).sum())
        rows.append({
            'radiologist': rad,
            'auroc': au, 'auprc': ap,
            'accuracy': balanced_accuracy_score(gt, raw_pred),
            'sensitivity': recall_score(gt, mrmc_pred, zero_division=0),
            'specificity': tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0,
            'precision': precision_score(gt, raw_pred, zero_division=0),
            'f1': f1_score(gt, raw_pred, zero_division=0),
        })
    return pd.DataFrame(rows).sort_values('radiologist').reset_index(drop=True)

_reader_w  = _per_reader_metrics(radiologist_df, False)
_reader_wm = _per_reader_metrics(radiologist_df, True)

# Per-case unique aggregation for model arms — 5-seed mean-prob ensemble.
# Reads seed_predictions.csv (5,500 rows) and means model_prob /
# combined_prob across every seed × radiologist review of each case.
# This is the canonical case-level computation matching
# `_metrics_utils.case_level_ensemble()` and `unique_case_*_metrics` as
# emitted by `generate_aggregates.py`.
_seed_csv = os.path.join(SRC_DIR, 'seed_predictions.csv')
_seed_df  = pd.read_csv(_seed_csv, float_precision='round_trip')
_case_agg = (
    _seed_df.groupby('case_id')
            .agg(gt=('gt', 'first'),
                 model_prob=('model_prob', 'mean'),
                 combined_prob=('combined_prob', 'mean'),
                 model_pred=('model_pred', 'first'),
                 cv_pred=('cv_pred', 'first'))
            .reset_index()
)
_n_cases = len(_case_agg)

def _reader_bootstrap_ci(arr, B=5000, seed=20260505):
    """Reader-level percentile bootstrap CI on the mean of a per-reader array."""
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    rng = np.random.RandomState(seed)
    boot = np.fromiter((a[rng.choice(n, size=n, replace=True)].mean()
                        for _ in range(B)), dtype=float, count=B)
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def _reader_paired_delta(arr_w, arr_wm, B=5000, seed=20260505):
    """Reader-level paired bootstrap on Δ = mean(arr_wm) - mean(arr_w).
    Resamples reader indices jointly so the same reader's (with, without)
    pair is preserved on each draw. Returns (Δ point, 2.5th, 97.5th)."""
    a_w = np.asarray(arr_w, dtype=float)
    a_m = np.asarray(arr_wm, dtype=float)
    valid = ~(np.isnan(a_w) | np.isnan(a_m))
    a_w = a_w[valid]; a_m = a_m[valid]
    n = len(a_w)
    rng = np.random.RandomState(seed)
    deltas = np.fromiter((a_m[ix].mean() - a_w[ix].mean()
                          for ix in (rng.choice(n, size=n, replace=True) for _ in range(B))),
                         dtype=float, count=B)
    delta_pt = float(a_m.mean() - a_w.mean())
    return (delta_pt,
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))

def _case_bootstrap_metric_ci(fn, gt, score, B=5000, seed=20260505):
    """Case-level percentile bootstrap CI on a metric function(gt, score)."""
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

def _case_paired_delta(fn, gt, score_w, score_wm, B=5000, seed=20260505):
    """Case-level paired bootstrap Δ = fn(gt, score_wm) - fn(gt, score_w).
    Resamples case indices jointly so case-level pairing is preserved."""
    rng = np.random.RandomState(seed)
    n = len(gt)
    deltas = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        try:
            d = fn(gt[idx], score_wm[idx]) - fn(gt[idx], score_w[idx])
            deltas.append(d)
        except Exception:
            pass
    delta_pt = float(fn(gt, score_wm) - fn(gt, score_w))
    arr = np.asarray(deltas)
    return (delta_pt,
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))

# ── AUROC / AUPRC point estimate + 95% CI for all 4 conditions ──
print("\nAUROC / AUPRC point estimate [95% CI]:")
_radw_au_pt,  _radw_au_lo,  _radw_au_hi  = _reader_bootstrap_ci(_reader_w['auroc'])
_radwm_au_pt, _radwm_au_lo, _radwm_au_hi = _reader_bootstrap_ci(_reader_wm['auroc'])
_radw_ap_pt,  _radw_ap_lo,  _radw_ap_hi  = _reader_bootstrap_ci(_reader_w['auprc'])
_radwm_ap_pt, _radwm_ap_lo, _radwm_ap_hi = _reader_bootstrap_ci(_reader_wm['auprc'])

_gt_case      = _case_agg['gt'].astype(int).values
_mprob_case   = _case_agg['model_prob'].values
_cprob_case   = _case_agg['combined_prob'].values
_mod_au_pt,   _mod_au_lo,   _mod_au_hi   = _case_bootstrap_metric_ci(roc_auc_score,         _gt_case, _mprob_case)
_modrad_au_pt,_modrad_au_lo,_modrad_au_hi= _case_bootstrap_metric_ci(roc_auc_score,         _gt_case, _cprob_case)
_mod_ap_pt,   _mod_ap_lo,   _mod_ap_hi   = _case_bootstrap_metric_ci(average_precision_score, _gt_case, _mprob_case)
_modrad_ap_pt,_modrad_ap_lo,_modrad_ap_hi= _case_bootstrap_metric_ci(average_precision_score, _gt_case, _cprob_case)

print(f"  Radiologist alone        AUROC  {_radw_au_pt:.3f} [{_radw_au_lo:.3f}, {_radw_au_hi:.3f}]")
print(f"  Radiologist with model   AUROC  {_radwm_au_pt:.3f} [{_radwm_au_lo:.3f}, {_radwm_au_hi:.3f}]")
print(f"  Model alone              AUROC  {_mod_au_pt:.3f} [{_mod_au_lo:.3f}, {_mod_au_hi:.3f}]")
print(f"  Model with radiologist   AUROC  {_modrad_au_pt:.3f} [{_modrad_au_lo:.3f}, {_modrad_au_hi:.3f}]")
print(f"  Radiologist alone        AUPRC  {_radw_ap_pt:.3f} [{_radw_ap_lo:.3f}, {_radw_ap_hi:.3f}]")
print(f"  Radiologist with model   AUPRC  {_radwm_ap_pt:.3f} [{_radwm_ap_lo:.3f}, {_radwm_ap_hi:.3f}]")
print(f"  Model alone              AUPRC  {_mod_ap_pt:.3f} [{_mod_ap_lo:.3f}, {_mod_ap_hi:.3f}]")
print(f"  Model with radiologist   AUPRC  {_modrad_ap_pt:.3f} [{_modrad_ap_lo:.3f}, {_modrad_ap_hi:.3f}]")

# ── Δ Human (with vs without AI) — reader-paired bootstrap ──
print("\nΔ Human (radiologist with AI vs alone) — reader-level paired bootstrap:")
_human_delta_metrics = ['auroc', 'auprc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
_human_delta_results = {}
for m in _human_delta_metrics:
    d, lo, hi = _reader_paired_delta(_reader_w[m].values, _reader_wm[m].values)
    _human_delta_results[m] = (d, lo, hi)
    print(f"  Δ {m:<12} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# ── Δ AI (CV+rad vs model alone) — case-level paired bootstrap on AUROC/AUPRC ──
print("\nΔ AI (model with radiologist vs alone) — case-level paired bootstrap (n=564):")
_ai_delta_results = {}
for label, fn, key in [('AUROC', roc_auc_score, 'auroc'),
                       ('AUPRC', average_precision_score, 'auprc')]:
    d, lo, hi = _case_paired_delta(fn, _gt_case, _mprob_case, _cprob_case)
    _ai_delta_results[key] = (d, lo, hi)
    print(f"  Δ {key:<12} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# Δ AI binary-classification metrics: pair-level (n=1100). The Δ point
# estimate is taken from pair_level_model_metrics / pair_level_cv_metrics
# (macro-averaged across the 5 CV folds — Table 1 canonical convention).
# The Δ CI uses fold-bootstrap on the per-fold Δ values: each bootstrap
# resamples the 5 fold-level Δs with replacement, mirroring how the
# macro-average aggregates fold-level estimates. This produces a Δ CI
# centered on the canonical macro-averaged Δ.
_bcv_df     = pd.DataFrame(best_cv_predictions)
_gt_pair    = _bcv_df['gt'].astype(int).values
_mpred_pair = _bcv_df['model_pred'].astype(int).values
_cvpred_pair= _bcv_df['cv_pred'].astype(int).values
_fold_pair  = _bcv_df['fold'].astype(int).values

def _calc_pair_metric(fn_name, gt, pred):
    pred = np.asarray(pred).astype(int); gt = np.asarray(gt).astype(int)
    if fn_name == 'accuracy': return accuracy_score(gt, pred)
    if fn_name == 'sensitivity': return recall_score(gt, pred, zero_division=0)
    if fn_name == 'precision': return precision_score(gt, pred, zero_division=0)
    if fn_name == 'f1': return f1_score(gt, pred, zero_division=0)
    if fn_name == 'specificity':
        tn = int(((pred == 0) & (gt == 0)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def _fold_delta_bootstrap(fn_name, gt, pred_w, pred_wm, fold_idx,
                           B=5000, seed=20260505):
    """Per-fold Δ then bootstrap across the K fold-Δ values.
    Δ_fold = metric_fold(wm) - metric_fold(w); resample Δ_folds B times."""
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
    return (delta_pt,
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))

# Macro-averaged point estimates from aggregates (canonical, used in Table 1).
# CI is shifted to be centered on the canonical Δ — keeping the half-width
# from the fold-Δ bootstrap distribution.
_PLM = pair_level_model_metrics  # model alone
_PCV = pair_level_cv_metrics      # model + radiologist
_metric_key_map = {'accuracy':'accuracy', 'sensitivity':'recall',
                   'specificity':'specificity', 'precision':'precision', 'f1':'f1'}
for m in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
    canonical_delta = _PCV[_metric_key_map[m]] - _PLM[_metric_key_map[m]]
    boot_delta_pt, boot_lo, boot_hi = _fold_delta_bootstrap(
        m, _gt_pair, _mpred_pair, _cvpred_pair, _fold_pair,
    )
    # Shift bootstrap CI to be centered on the canonical Δ
    shift = canonical_delta - boot_delta_pt
    lo = boot_lo + shift
    hi = boot_hi + shift
    _ai_delta_results[m] = (canonical_delta, lo, hi)
    print(f"  Δ {m:<12} {canonical_delta:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# ── Confidence-accuracy correlation, mean confidence, RTAT (Table 1 rows 9-13) ──
# Per-reader Pearson correlation between confidence and correctness (point-
# biserial) for each arm. Reader-level bootstrap CI.
def _per_reader_corr(rdf, with_seg, x_col, y_col):
    sub = rdf[rdf['with_segmentation'] == with_seg]
    out = []
    for rad, grp in sub.groupby('radiologist'):
        if len(grp) < 3 or grp[y_col].std() == 0:
            out.append(np.nan); continue
        try:
            r, _ = _scipy_stats.pearsonr(grp[x_col].astype(float), grp[y_col].astype(float))
            out.append(r)
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

print("\nConfidence-accuracy correlation (Pearson r per reader, reader-level bootstrap CI):")
_corr_w = _per_reader_corr(radiologist_df, False, 'confidence', 'correct_prediction')
_corr_wm = _per_reader_corr(radiologist_df, True, 'confidence', 'correct_prediction')
for label, arr in [('Radiologist alone', _corr_w), ('Radiologist with model', _corr_wm)]:
    m, lo, hi = _reader_bootstrap_ci(arr)
    print(f"  {label:30s}  r = {m:.3f} [{lo:.3f}, {hi:.3f}]")
_corr_delta_pt, _corr_delta_lo, _corr_delta_hi = _reader_paired_delta(_corr_w, _corr_wm)
print(f"  Δ Human (with vs without)        {_corr_delta_pt:+.3f} [{_corr_delta_lo:+.3f}, {_corr_delta_hi:+.3f}]")

# n=12 agent-level aggregate (11 radiologists + 1 model) — manuscript Para 126.
# Model-side values are computed live from the same bundled CSVs that fig_6
# uses (model_case_confidence.csv for the without-support arm; cv_predictions_min.csv
# for the with-support arm), exactly mirroring the fig_6 metric definitions:
#   confidence_score = |prob - 0.5| * 20  (scales [0,1] prob to [0,10] confidence)
#   without-support  calibration_diff: median split of confidence_score
#   with-support     calibration_diff: Q3/Q1 quartile split of confidence_score
# SD reported with ddof=1 (sample SD across 12 agents).
_FIG6_CSV = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_6', 'csv')
_prob_df_canonical = pd.read_csv(os.path.join(_FIG6_CSV, 'model_case_confidence.csv'),
                                  float_precision='round_trip')
_prob_map = dict(zip(_prob_df_canonical['case_id'], _prob_df_canonical['top_percentile_prob']))
_mp_w, _mc_w_arr = [], []
for _, _row in radiologist_df.iterrows():
    if _row['case_id'] in _prob_map:
        _mp_w.append(abs(_prob_map[_row['case_id']] - 0.5) * 20)
        _mc_w_arr.append(_row['model_predicted_enhancement'] == _row['has_enhancement_gt'])
_mp_w = np.array(_mp_w, dtype=float); _mc_w_arr = np.array(_mc_w_arr, dtype=bool)
_MODEL_CORR_WITHOUT = float(_scipy_stats.pearsonr(_mp_w, _mc_w_arr.astype(float))[0])
_med_w = np.median(_mp_w)
_MODEL_CALIB_WITHOUT = float(_mc_w_arr[_mp_w >= _med_w].mean() - _mc_w_arr[_mp_w < _med_w].mean())
_MODEL_BIAS_WITHOUT = float(_mp_w[_mc_w_arr].mean() - _mp_w[~_mc_w_arr].mean())

_cv_df_canonical = pd.read_csv(os.path.join(_FIG6_CSV, 'cv_predictions_min.csv'),
                                float_precision='round_trip')
_mp_m = (np.abs(_cv_df_canonical['combined_prob'].astype(float).values - 0.5) * 20)
_mc_m_arr = (_cv_df_canonical['cv_pred'].astype(int).values
             == _cv_df_canonical['gt'].astype(int).values)
_MODEL_CORR_WITH = float(_scipy_stats.pearsonr(_mp_m, _mc_m_arr.astype(float))[0])
# Q3/Q1 quartile split for the model-with-support calibration_diff, matching
# the radiologist convention in _metrics_utils.compute_confidence_analysis (and
# fig_6.py).
_q3_m, _q1_m = np.quantile(_mp_m, 0.75), np.quantile(_mp_m, 0.25)
_MODEL_CALIB_WITH = float(_mc_m_arr[_mp_m >= _q3_m].mean() - _mc_m_arr[_mp_m <= _q1_m].mean())
_MODEL_BIAS_WITH = float(_mp_m[_mc_m_arr].mean() - _mp_m[~_mc_m_arr].mean())

_n12_corr_w  = np.concatenate([_corr_w,  [_MODEL_CORR_WITHOUT]])
_n12_corr_wm = np.concatenate([_corr_wm, [_MODEL_CORR_WITH]])

# Per-reader calibration_diff (Q3 high vs Q1 low confidence accuracy) —
# matches the calibration metric used by fig_6.
def _per_reader_calib_diff(rdf, with_seg):
    sub = rdf[rdf['with_segmentation'] == with_seg]
    out = []
    for _, g in sub.groupby('radiologist'):
        q3 = g['confidence'].quantile(0.75)
        q1 = g['confidence'].quantile(0.25)
        high = g[g['confidence'] >= q3]['correct_prediction'].mean()
        low  = g[g['confidence'] <= q1]['correct_prediction'].mean()
        if not (np.isnan(high) or np.isnan(low)):
            out.append(high - low)
    return np.array(out)

_calib_w  = _per_reader_calib_diff(radiologist_df, False)
_calib_wm = _per_reader_calib_diff(radiologist_df, True)
_n12_calib_w  = np.concatenate([_calib_w,  [_MODEL_CALIB_WITHOUT]])
_n12_calib_wm = np.concatenate([_calib_wm, [_MODEL_CALIB_WITH]])

# Echo the model calibration_diff values so the Table 1 row 9 AI columns
# (-0.112 alone, 0.276 with radiologist) are traceable to this log. Without
# this print the values existed only inside the bootstrap statistic, leaving
# no audit trail between fig_1 and the manuscript / table.
print(
    f"\nModel calibration_diff (high - low confidence accuracy):"
    f"\n  Model alone (median split):              {_MODEL_CALIB_WITHOUT:+.3f}"
    f"\n  Model + radiologist (Q3/Q1 split):       {_MODEL_CALIB_WITH:+.3f}"
)

# Per-reader confidence bias (mean confidence on correct - mean on incorrect)
def _per_reader_conf_bias(rdf, with_seg):
    sub = rdf[rdf['with_segmentation'] == with_seg]
    out = []
    for _, g in sub.groupby('radiologist'):
        c = g[g['correct_prediction'] == 1]['confidence'].mean()
        i = g[g['correct_prediction'] == 0]['confidence'].mean()
        if not (np.isnan(c) or np.isnan(i)):
            out.append(c - i)
    return np.array(out)

_bias_w  = _per_reader_conf_bias(radiologist_df, False)
_bias_wm = _per_reader_conf_bias(radiologist_df, True)
_n12_bias_w  = np.concatenate([_bias_w,  [_MODEL_BIAS_WITHOUT]])
_n12_bias_wm = np.concatenate([_bias_wm, [_MODEL_BIAS_WITH]])

print("\nAgent-level (n=12: 11 radiologists + 1 model) calibration metrics (manuscript Para 126):")

def _para126_sig(p):
    if p < 0.0001: return 'p<0.0001'
    if p < 0.001:  return 'p<0.001'
    if p < 0.01:   return 'p<0.01'
    if p < 0.05:   return 'p<0.05'
    return f'p={p:.3f} (ns)'

_t1, _p1 = _scipy_stats.ttest_rel(_n12_corr_w,  _n12_corr_wm)
_t2, _p2 = _scipy_stats.ttest_rel(_n12_calib_w, _n12_calib_wm)
_t3, _p3 = _scipy_stats.ttest_rel(_n12_bias_w,  _n12_bias_wm)
print(f"  Sentence 1 — confidence-accuracy correlation (paired t={_t1:.3f}, "
      f"{_para126_sig(_p1)}; manuscript: p<0.01):")
print(f"    Without support  mean = {_n12_corr_w.mean():.3f} ± {_n12_corr_w.std(ddof=1):.3f}")
print(f"    With support     mean = {_n12_corr_wm.mean():.3f} ± {_n12_corr_wm.std(ddof=1):.3f}")
print(f"  Sentence 2 — calibration difference (paired t={_t2:.3f}, "
      f"{_para126_sig(_p2)}; manuscript: p<0.01):")
print(f"    Without support  mean = {_n12_calib_w.mean():.3f} ± {_n12_calib_w.std(ddof=1):.3f}")
print(f"    With support     mean = {_n12_calib_wm.mean():.3f} ± {_n12_calib_wm.std(ddof=1):.3f}")
print(f"  Sentence 3 — confidence bias (paired t={_t3:.3f}, "
      f"{_para126_sig(_p3)}; manuscript: p<0.01):")
print(f"    Without support  mean = {_n12_bias_w.mean():.3f} ± {_n12_bias_w.std(ddof=1):.3f}")
print(f"    With support     mean = {_n12_bias_wm.mean():.3f} ± {_n12_bias_wm.std(ddof=1):.3f}")

# Mean confidence per arm (Table 1 row 12)
print("\nMean confidence per arm (case-level, with reader-level bootstrap CI on the per-reader mean):")
_mc_w = (radiologist_df[radiologist_df['with_segmentation'] == False]
         .groupby('radiologist')['confidence'].mean().values)
_mc_wm = (radiologist_df[radiologist_df['with_segmentation'] == True]
          .groupby('radiologist')['confidence'].mean().values)
for label, arr in [('Radiologist alone', _mc_w), ('Radiologist with model', _mc_wm)]:
    m, lo, hi = _reader_bootstrap_ci(arr)
    print(f"  {label:30s}  Mean = {m:.2f} [{lo:.2f}, {hi:.2f}]  (per-reader range: {arr.min():.2f}-{arr.max():.2f})")
_mc_delta_pt, _mc_delta_lo, _mc_delta_hi = _reader_paired_delta(_mc_w, _mc_wm)
print(f"  Δ Human (with vs without)        {_mc_delta_pt:+.2f} [{_mc_delta_lo:+.2f}, {_mc_delta_hi:+.2f}]")
# Reader-level paired bootstrap p for Δ Human confidence
# (matches B/seed used for the κ contrasts; cited in manuscript paragraph 86)
_mc_B = 5000
_mc_rng = np.random.default_rng(20260505)
_mc_n = len(_mc_w)
_mc_boot = np.empty(_mc_B)
for _b in range(_mc_B):
    _ix = _mc_rng.integers(0, _mc_n, size=_mc_n)
    _mc_boot[_b] = _mc_wm[_ix].mean() - _mc_w[_ix].mean()
_mc_p_boot = 2 * float(min((_mc_boot <= 0).mean(), (_mc_boot >= 0).mean()))
print(f"  Δ Human confidence bootstrap p (reader-level, B={_mc_B}, seed=20260505): p={_mc_p_boot:.4f}")

# RTAT per arm (Table 1 row 13)
print("\nRTAT (response time per case, seconds; reader-level bootstrap CI on the per-reader mean):")
_rt_w = (radiologist_df[radiologist_df['with_segmentation'] == False]
         .groupby('radiologist')['response_time'].mean().values)
_rt_wm = (radiologist_df[radiologist_df['with_segmentation'] == True]
          .groupby('radiologist')['response_time'].mean().values)
for label, arr in [('Radiologist alone', _rt_w), ('Radiologist with model', _rt_wm)]:
    m, lo, hi = _reader_bootstrap_ci(arr)
    print(f"  {label:30s}  Mean = {m:.1f} s [{lo:.1f}, {hi:.1f}]  (per-reader range: {arr.min():.1f}-{arr.max():.1f})")
_rt_delta_pt, _rt_delta_lo, _rt_delta_hi = _reader_paired_delta(_rt_w, _rt_wm)
print(f"  Δ Human (with vs without)        {_rt_delta_pt:+.1f} s [{_rt_delta_lo:+.1f}, {_rt_delta_hi:+.1f}]")

# Pooled confidence reporting (matches manuscript text)
_conf_pool_w  = radiologist_df[~radiologist_df['with_segmentation']]['confidence']
_conf_pool_wm = radiologist_df[ radiologist_df['with_segmentation']]['confidence']
_rt_pool_w    = radiologist_df[~radiologist_df['with_segmentation']]['response_time']
_rt_pool_wm   = radiologist_df[ radiologist_df['with_segmentation']]['response_time']
print(f"\nPooled-review summary (raw 2200-row pool, paragraphs 86 + 88):")
print(f"  Confidence without model:  mean={_conf_pool_w.mean():.2f} ± {_conf_pool_w.std():.2f}")
print(f"  Confidence with model:     mean={_conf_pool_wm.mean():.2f} ± {_conf_pool_wm.std():.2f}")
print(f"  RTAT without model:        mean={_rt_pool_w.mean():.1f} ± {_rt_pool_w.std():.1f}")
print(f"  RTAT with model:           mean={_rt_pool_wm.mean():.1f} ± {_rt_pool_wm.std():.1f}")

# Paragraph 88 reports throughput as cases per hour ± SD using the delta-method
# approximation from RTAT: mean = 3600/mean(rt), SD ≈ (3600/mean(rt)²) × SD(rt).
_thr_mean_w  = 3600.0 / _rt_pool_w.mean()
_thr_mean_wm = 3600.0 / _rt_pool_wm.mean()
_thr_sd_w  = (3600.0 / _rt_pool_w.mean()  ** 2) * _rt_pool_w.std()
_thr_sd_wm = (3600.0 / _rt_pool_wm.mean() ** 2) * _rt_pool_wm.std()
print(
    f"  Throughput without model:  mean={_thr_mean_w:.0f} ± {_thr_sd_w:.1f} cases/hour"
    f"  (delta-method SD; paragraph 88)"
)
print(
    f"  Throughput with model:     mean={_thr_mean_wm:.0f} ± {_thr_sd_wm:.1f} cases/hour"
    f"  (delta-method SD; paragraph 88)"
)
print(
    f"  Reporting-efficiency gain: {(_thr_mean_wm - _thr_mean_w) / _thr_mean_w * 100:.0f}%"
    f"  (paragraph 88)"
)

# Note: Table 1 rows 9-10 model arms (conf-acc corr -0.078/0.290 and
# calibration_diff -0.112/0.273) are computed in fig_6.py, not here. fig_1
# focuses on radiologist-level metrics and the AI-arm summary numbers
# downstream of the existing Brier-based CQS print further down.

# ── Para 123 calibration sub-metric (per-reader correct/incorrect Δ confidence) ──
print("\nPer-reader Δ confidence (correct − incorrect) (para 123 sentence 3):")
for ws, label in [(False, 'Radiologist alone'), (True, 'Radiologist with model')]:
    sub = radiologist_df[radiologist_df['with_segmentation'] == ws]
    diffs = []
    for rad, grp in sub.groupby('radiologist'):
        c = grp[grp['correct_prediction'] == 1]['confidence']
        i = grp[grp['correct_prediction'] == 0]['confidence']
        if len(c) > 0 and len(i) > 0:
            diffs.append(c.mean() - i.mean())
    diffs = np.array(diffs)
    print(f"  {label:<28s}  mean = {diffs.mean():.3f} ± {diffs.std(ddof=1):.3f}  (n={len(diffs)})")

# ── Para 109 confidence/accuracy vs reporting speed correlations ──
# Computed at the per-confidence-bin level (10 bins, one per integer confidence
# 1–10) on speed = 3600/response_time (cases per hour).
print("\nConfidence/accuracy vs reporting speed (per-confidence-bin pooled, paragraph 109):")
_speed_df = radiologist_df.copy()
_speed_df['speed_cph'] = 3600.0 / _speed_df['response_time'].replace(0, np.nan)
for ws, label in [(False, 'Radiologist alone'), (True, 'Radiologist with model')]:
    sub = _speed_df[_speed_df['with_segmentation'] == ws].dropna(subset=['speed_cph'])
    agg = sub.groupby('confidence').agg(
        speed_cph=('speed_cph', 'mean'),
        accuracy=('correct_prediction', 'mean'),
    ).reset_index()
    if len(agg) < 3: continue
    r_cs, p_cs = _scipy_stats.pearsonr(agg['confidence'], agg['speed_cph'])
    r_as, p_as = _scipy_stats.pearsonr(agg['accuracy'], agg['speed_cph'])
    print(f"  {label:<28s}  conf-speed r = {r_cs:+.3f} (p={p_cs:.3g}) | acc-speed r = {r_as:+.3f} (p={p_as:.3g})")

# Levene's test + coefficient of variation on raw individual-level cases/hour
# (paragraph 109 final sentence): variability of cases/hour without vs with
# model support, with the manuscript's CV reported as SD(3600/rt) / mean(3600/rt).
_cph_raw_wo = (3600.0 / radiologist_df.loc[~radiologist_df['with_segmentation'], 'response_time'].values)
_cph_raw_wm = (3600.0 / radiologist_df.loc[ radiologist_df['with_segmentation'], 'response_time'].values)
_cph_raw_wo = _cph_raw_wo[np.isfinite(_cph_raw_wo)]
_cph_raw_wm = _cph_raw_wm[np.isfinite(_cph_raw_wm)]
_cv_raw_wo = (_cph_raw_wo.std(ddof=0) / _cph_raw_wo.mean()) * 100.0
_cv_raw_wm = (_cph_raw_wm.std(ddof=0) / _cph_raw_wm.mean()) * 100.0
_lev_stat, _lev_p = _scipy_stats.levene(_cph_raw_wo, _cph_raw_wm)
print(
    f"\nReporting-speed variability — raw cases/hour (paragraph 109):"
    f"\n  Without model (n={len(_cph_raw_wo)}): mean = {_cph_raw_wo.mean():.1f}, "
    f"CV = {_cv_raw_wo:.1f}%"
    f"\n  With model    (n={len(_cph_raw_wm)}): mean = {_cph_raw_wm.mean():.1f}, "
    f"CV = {_cv_raw_wm:.1f}%"
    f"\n  Levene's test of equal variances: W = {_lev_stat:.3f}, p = {_lev_p:.4f}"
)

# Supplementary Table 2 (Country × Pathology subgroup analysis) is
# generated by code/tables/supplementary_table_2.py and saved to
# data/source_data/supplementary_table_2/csv/supplementary_table_2.csv.

# ── Country composition test (564 reviewed vs 1109 full test set; para 182) ──
# Full-test-set country counts come from the primary model paper (ref 42),
# which is upstream of the 564-case subset bundled in source_data/. They are
# loaded from upstream_metadata.json rather than inlined.
from scipy.stats import chi2_contingency as _chi2_p182
with open(os.path.join(SRC_DIR, 'upstream_metadata.json')) as _fh:
    _upstream = json.load(_fh)
_full_country = _upstream['full_test_set_country_counts']
_full_total   = _upstream['full_test_set_total']
_sub_country  = radiologist_df.drop_duplicates('case_id')['Country'].value_counts().to_dict()
_sub_total    = sum(_sub_country.values())
_countries = ['Netherlands', 'UK', 'USA', 'Sub-Saharan Africa']
_reviewed_row    = [int(_sub_country.get(c, 0)) for c in _countries]
_not_reviewed_row = [_full_country[c] - _reviewed_row[i] for i, c in enumerate(_countries)]
_chi2_val, _chi2_p, _chi2_dof, _ = _chi2_p182([_reviewed_row, _not_reviewed_row])
print(f"\nCountry composition test ({_sub_total} reviewed vs {_full_total} full test set; para 182):")
print(f"  Reviewed ({_sub_total}):    " + ", ".join(f"{c}={n} ({n/_sub_total*100:.2f}%)" for c, n in zip(_countries, _reviewed_row)))
print(f"  Full set ({_full_total}):   " + ", ".join(f"{c}={_full_country[c]} ({_full_country[c]/_full_total*100:.2f}%)" for c in _countries))
print(f"  χ²({_chi2_dof}) = {_chi2_val:.2f}, p = {_chi2_p:.4f}")

# ── Per-reader calibration regression slope (Para 107 reproducibility) ──
# Fit accuracy ~ confidence per reader (confidence on 1-10 scale).
print("\nPer-reader calibration regression (accuracy ~ confidence in 1-10 units):")
for _ws_calib, _calib_label in [(False, 'Radiologist alone'), (True, 'Radiologist with model')]:
    _calib_sub = radiologist_df[radiologist_df['with_segmentation'] == _ws_calib]
    _calib_slopes = []; _calib_devs = []
    for _rad_calib, _grp_calib in _calib_sub.groupby('radiologist'):
        if _grp_calib['correct_prediction'].std() == 0 or _grp_calib['confidence'].std() == 0: continue
        _calib_x = _grp_calib['confidence'].astype(float).values
        _calib_y = _grp_calib['correct_prediction'].astype(float).values
        _calib_z = np.polyfit(_calib_x, _calib_y, 1)
        _calib_slopes.append(_calib_z[0])
        _bin_acc = _grp_calib.groupby('confidence')['correct_prediction'].mean()
        _expected = 0.10 * _bin_acc.index.astype(float)
        _calib_devs.append(float(np.mean(np.abs(_bin_acc.values - _expected.values))))
    _calib_slopes = np.array(_calib_slopes); _calib_devs = np.array(_calib_devs)
    print(f"  {_calib_label:<28s}  slope = {_calib_slopes.mean():.3f} ± {_calib_slopes.std(ddof=1):.3f}  "
          f"mean dev = {_calib_devs.mean():.3f} ± {_calib_devs.std(ddof=1):.3f}")

# Supplementary Tables 3 and 4 (Sex disaggregation point values + Δ female−male)
# are generated by code/tables/supplementary_table_3.py and
# code/tables/supplementary_table_4.py respectively, and saved to
# data/source_data/supplementary_table_{3,4}/csv/.
# Create error bars from confidence intervals
# Ensure error bars are non-negative (distance from mean to CI bounds)
# Use max(0, ...) to prevent negative values while preserving direction
cv_model_rad_errors = [(max(0, cv_model_rad_values[i] - ci[0]), max(0, ci[1] - cv_model_rad_values[i])) for i, ci in enumerate(cv_model_rad_ci)]
cv_model_rad_errors_lower = [e[0] for e in cv_model_rad_errors]
cv_model_rad_errors_upper = [e[1] for e in cv_model_rad_errors]

# Reordered bars: Rad (without), Rad (with), Model (without), Model (with)
# Position 1: Radiologists (without model) at x - 1.5*width
ax.bar(x - 1.5*width, rad_without_values, width, label='Radiologist (without model)',
       alpha=0.8, color=colors[0], yerr=[rad_without_errors_lower, rad_without_errors_upper],
       capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'}, edgecolor='black', linewidth=0.5)

# Position 2: Radiologists (with model) at x - 0.5*width
ax.bar(x - 0.5*width, rad_with_values, width, label='Radiologist (with model)',
       alpha=0.8, color=colors[2], yerr=[rad_with_errors_lower, rad_with_errors_upper],
       capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'}, edgecolor='black', linewidth=0.5)

# Position 3: Model (without radiologist) at x + 0.5*width
ax.bar(x + 0.5*width, model_values, width, label='Model (without radiologist)', alpha=0.8, color=colors[3],
       yerr=[model_errors_lower, model_errors_upper], capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'}, edgecolor='black', linewidth=0.5)

# Position 4: Model (with radiologist) - CV optimized at x + 1.5*width with proper confidence intervals
if cv_model_rad_values and any(v > 0 for v in cv_model_rad_values):
    ax.bar(x + 1.5*width, cv_model_rad_values, width, label='Model (with radiologist)',
           alpha=0.8, color=colors[4], yerr=[cv_model_rad_errors_lower, cv_model_rad_errors_upper],
           capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'}, edgecolor='black', linewidth=0.5)



ax.set_ylabel('Metric score', fontsize=12)
ax.set_title('b) Performance metrics', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=45, fontsize=12)
ax.legend(loc='lower right')
ax.set_ylim(0, 1)  # Set ylim to (0,1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Keep max ytick at 1.0
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
# Use case-level aggregation for individual radiologist performance.
print(f"\nPanel c: Calculating individual radiologist performance on unique cases")

# Get paired data
paired_data = []
for rad in radiologist_df['radiologist'].unique():
    rad_data = radiologist_df[radiologist_df['radiologist'] == rad]

    # Only include if we have data for both conditions
    if len(rad_data[rad_data['with_segmentation'] == False]) > 0 and \
       len(rad_data[rad_data['with_segmentation'] == True]) > 0:

        # Aggregate by case for without segmentation
        rad_without_data = rad_data[rad_data['with_segmentation'] == False]
        rad_preds_without, rad_gts_without, n_without = aggregate_by_case_fig1(
            rad_without_data, 'predicted_enhancement'
        )
        accuracy_without = accuracy_score(rad_gts_without, rad_preds_without)

        # Aggregate by case for with segmentation
        rad_with_data = rad_data[rad_data['with_segmentation'] == True]
        rad_preds_with, rad_gts_with, n_with = aggregate_by_case_fig1(
            rad_with_data, 'predicted_enhancement'
        )
        accuracy_with = accuracy_score(rad_gts_with, rad_preds_with)

        years_exp = rad_data['years_experience'].iloc[0]

        paired_data.append({
            'radiologist': rad,
            'accuracy_without': accuracy_without,
            'accuracy_with': accuracy_with,
            'years_experience': years_exp,
            'n_cases_without': n_without,
            'n_cases_with': n_with
        })

paired_data = pd.DataFrame(paired_data)

# Paragraph 80 reports the count of radiologists whose accuracy improved /
# stayed the same / declined with model support.
_diff = paired_data['accuracy_with'] - paired_data['accuracy_without']
_n_improved  = int((_diff >  0.001).sum())
_n_unchanged = int((_diff.abs() <= 0.001).sum())
_n_declined  = int((_diff < -0.001).sum())
_n_total     = len(paired_data)
print(
    f"\nPer-radiologist accuracy direction (paragraph 80; n={_n_total} readers):"
    f"\n  Improved with model:   {_n_improved}/{_n_total} "
    f"({_n_improved/_n_total*100:.0f}%)"
    f"\n  Unchanged:             {_n_unchanged}/{_n_total}"
    f"\n  Declined slightly:     {_n_declined}/{_n_total}"
)

# Plot lines for each radiologist with colors based on change
for _, row in paired_data.iterrows():
    change = row['accuracy_with'] - row['accuracy_without']
    if abs(change) < 0.001:
        color = 'orange'
    elif change > 0:
        color = '#2ca02c'  # Green color matching panel f/g scatterpoints
    else:
        color = 'red'
    
    # Size based on experience
    size = row['years_experience'] * 12  # Reduced to match R/M point size (max 300)
    
    ax.plot([0, 1], [row['accuracy_without'], row['accuracy_with']], 
            color=color, alpha=0.7, linewidth=2, marker='o', markersize=8)
    
    # Add scatter points with size
    ax.scatter(0, row['accuracy_without'], s=size, alpha=0.8, 
               color=color, edgecolors='black', linewidth=0.5, zorder=5)
    ax.scatter(1, row['accuracy_with'], s=size, alpha=0.8,
               color=color, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Add radiologist number
    radiologist_num = row['radiologist'].split('#')[1] if '#' in row['radiologist'] else str(paired_data.index[paired_data['radiologist'] == row['radiologist']].tolist()[0] + 1)
    ax.text(0, row['accuracy_without'], radiologist_num, color='black', fontsize=8,
            ha='center', va='center', zorder=6)
    ax.text(1, row['accuracy_with'], radiologist_num, color='black', fontsize=8,
            ha='center', va='center', zorder=6)

# Use reader-averaged balanced accuracy from Panel A for consistency
# For radiologist: Use group1_avg and group2_avg balanced accuracies
# For model: Use model baseline and CV balanced accuracies
print(f"\nPanel c: Using consistent metrics from Panel A")

# Radiologist group means: Use reader-averaged balanced accuracy
mean_accuracy_without = group1_avg['balanced_accuracy']
print(f"  Radiologist alone (balanced accuracy): {mean_accuracy_without:.4f}")

mean_accuracy_with = group2_avg['balanced_accuracy']
print(f"  Radiologist with model (balanced accuracy): {mean_accuracy_with:.4f}")

mean_change = mean_accuracy_with - mean_accuracy_without

# Determine color for mean line based on change
if abs(mean_change) < 0.001:
    mean_color = 'orange'
elif mean_change > 0:
    mean_color = '#2ca02c'  # Green color matching panel f/g scatterpoints
else:
    mean_color = 'red'

# Plot solid line connecting the means (color based on change)
ax.plot([0, 1], [mean_accuracy_without, mean_accuracy_with],
        color=mean_color, alpha=0.7, linewidth=2, linestyle='-', zorder=10)

# Add black scatterpoints with white 'R'
ax.scatter(0, mean_accuracy_without, s=300, alpha=1.0,
           color='black', edgecolors='white', linewidth=0.5, zorder=11)
ax.scatter(1, mean_accuracy_with, s=300, alpha=1.0,
           color='black', edgecolors='white', linewidth=0.5, zorder=11)

# Add 'R' text for radiologist group mean
ax.text(0, mean_accuracy_without, 'R', color='white', fontsize=12,
        ha='center', va='center', zorder=12)
ax.text(1, mean_accuracy_with, 'R', color='white', fontsize=12,
        ha='center', va='center', zorder=12)

# Model scatterpoints: Use pair-level accuracies from Panel A
# Model without radiologist support: Use pair-level accuracy from Panel A
model_accuracy_without = model_alone_accuracy  # Pair-level accuracy from Panel A
print(f"  Model alone (pair-level accuracy): {model_accuracy_without:.4f}")

# Model with radiologist support: Use pair-level CV accuracy from Panel A
model_accuracy_with = model_together_accuracy  # Pair-level CV accuracy from Panel A
print(f"  Model with radiologist support (pair-level CV accuracy): {model_accuracy_with:.4f}")

# Plot solid line connecting model accuracies (green for improvement)
model_change = model_accuracy_with - model_accuracy_without
if abs(model_change) < 0.001:
    model_line_color = 'orange'
elif model_change > 0:
    model_line_color = '#2ca02c'  # Green color matching panel f/g scatterpoints
else:
    model_line_color = 'red'

ax.plot([0, 1], [model_accuracy_without, model_accuracy_with],
        color=model_line_color, alpha=0.7, linewidth=2, linestyle='-', zorder=9)

# Add white scatterpoints for model
ax.scatter(0, model_accuracy_without, s=300, alpha=1.0,
           color='white', edgecolors='black', linewidth=0.5, zorder=11)
ax.scatter(1, model_accuracy_with, s=300, alpha=1.0,
           color='white', edgecolors='black', linewidth=0.5, zorder=11)

# Add 'M' text for model
ax.text(0, model_accuracy_without, 'M', color='black', fontsize=12,
        ha='center', va='center', zorder=12)
ax.text(1, model_accuracy_with, 'M', color='black', fontsize=12,
        ha='center', va='center', zorder=12)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Without\nsupport', 'With\nsupport'], fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('c) Individual agent performance change', fontsize=14)
ax.grid(True, alpha=0.3)

# Add legend
import matplotlib.lines as mlines
green_line = mlines.Line2D([], [], color='green', marker='o', linestyle='-', 
                          markersize=8, label='Improved')
orange_line = mlines.Line2D([], [], color='orange', marker='o', linestyle='-', 
                           markersize=8, label='Unchanged')
red_line = mlines.Line2D([], [], color='red', marker='o', linestyle='-', 
                        markersize=8, label='Declined')

# Experience legend elements
exp_elements = []
for size, label in zip([5, 15, 25], ['5 years', '15 years', '25 years']):
    exp_elements.append(plt.scatter([], [], s=size*7, alpha=0.8,  # Reduced proportionally with actual points
                                   edgecolors='black', linewidth=0.5,
                                   label=label, color='gray'))

all_handles = [green_line, orange_line, red_line] + exp_elements
all_labels = ['Improved', 'Unchanged', 'Declined'] + ['5 years', '15 years', '25 years']
ax.legend(handles=all_handles, labels=all_labels, 
          loc='lower right', ncol=2, columnspacing=3,
          title='Performance Change    |    Experience')

ax = axes[1, 2]
# Get unique radiologists
all_rads = set()
for pa in paired_agreements:
    all_rads.add(pa['rad1'])
    all_rads.add(pa['rad2'])
all_rads = sorted(list(all_rads))

# Calculate Cohen's k gain for each radiologist
radiologist_gains = {}
for rad in all_rads:
    # Get all kappa values for this radiologist
    without_kappas = []
    with_kappas = []
    for pa in paired_agreements:
        if pa['rad1'] == rad or pa['rad2'] == rad:
            without_kappas.append(pa['kappa_without'])
            with_kappas.append(pa['kappa_with'])
    
    if without_kappas and with_kappas:
        avg_gain = np.mean(with_kappas) - np.mean(without_kappas)
        radiologist_gains[rad] = avg_gain
    else:
        radiologist_gains[rad] = 0

# Sort radiologists by Cohen's k gain (descending)
all_rads = sorted(all_rads, key=lambda r: radiologist_gains.get(r, 0), reverse=True)

# Add Model to the list
all_rads.append('Model')
n_rads = len(all_rads)

# Create matrix for heatmap (now includes model)
kappa_without_matrix = np.full((n_rads, n_rads), np.nan)

# Fill matrix for radiologist-radiologist pairs
for pa in paired_agreements:
    i = all_rads.index(pa['rad1'])
    j = all_rads.index(pa['rad2'])
    kappa_without_matrix[i, j] = pa['kappa_without']
    kappa_without_matrix[j, i] = pa['kappa_without']  # Symmetric

# Calculate model-radiologist agreement (Cohen's kappa) without support
model_idx = all_rads.index('Model')

for rad_idx, rad in enumerate(all_rads[:-1]):  # Exclude Model itself
    # Skip if this is the Model
    if rad == 'Model':
        continue

    # Get radiologist predictions without segmentation
    rad_data = radiologist_df[(radiologist_df['radiologist'] == rad) &
                             (radiologist_df['with_segmentation'] == False)]

    if len(rad_data) > 0:
        # Get model predictions for the same cases
        common_cases = rad_data.dropna(subset=['model_predicted_enhancement'])

        if len(common_cases) > 0:
            # Calculate Cohen's kappa between model and radiologist
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(common_cases['model_predicted_enhancement'],
                                    common_cases['predicted_enhancement'])
            kappa_without_matrix[model_idx, rad_idx] = kappa
            kappa_without_matrix[rad_idx, model_idx] = kappa  # Symmetric

# Set diagonal to 1 (perfect agreement with self)
np.fill_diagonal(kappa_without_matrix, 1.0)

# Get experience data and create labels
rad_labels = []
for rad in all_rads:
    if rad == 'Model':
        rad_labels.append('Model')
    else:
        # Extract just the number part from "Radiologist #X"
        if 'Radiologist #' in rad:
            rad_num = rad.replace('Radiologist #', '#')  # Keep the # symbol
        else:
            rad_num = rad
        
        # Try to get experience data
        years_exp = None
        exp_data = experience_df[experience_df['radiologist'] == rad]
        if len(exp_data) > 0:
            years_exp = exp_data['years_experience'].iloc[0]
        
        if years_exp is not None:
            rad_labels.append(f'R{rad_num} ({int(years_exp)} yrs)')
        else:
            rad_labels.append(f'R{rad_num}')

# Calculate mean kappa for title
kappa_without = np.nanmean(kappa_without_matrix[~np.eye(n_rads, dtype=bool)])

# Create heatmap with same aesthetics as inter_rater_agreement_analysis.png
sns.heatmap(kappa_without_matrix, annot=False, cmap='inferno',
            vmin=0, vmax=1.0, ax=ax, cbar=False,
            xticklabels=rad_labels,
            yticklabels=rad_labels)

# Build the with-support kappa matrix used to flag higher-agreement cells.
kappa_with_matrix = np.full((n_rads, n_rads), np.nan)
for pa in paired_agreements:
    # Check if both radiologists are in all_rads (they should be)
    if pa['rad1'] in all_rads and pa['rad2'] in all_rads:
        i = all_rads.index(pa['rad1'])
        j = all_rads.index(pa['rad2'])
        kappa_with_matrix[i, j] = pa['kappa_with']
        kappa_with_matrix[j, i] = pa['kappa_with']

# Add model-radiologist kappa for with-support scenario
model_idx = all_rads.index('Model')

for rad_idx, rad in enumerate(all_rads[:-1]):  # Exclude Model itself
    if rad == 'Model':
        continue

    # Get radiologist predictions with segmentation
    rad_data = radiologist_df[(radiologist_df['radiologist'] == rad) &
                             (radiologist_df['with_segmentation'] == True)]

    if len(rad_data) > 0:
        common_cases = rad_data.dropna(subset=['model_predicted_enhancement'])

        if len(common_cases) > 0:
            cv_preds = []
            for idx, row in common_cases.iterrows():
                case_id = row['case_id']
                if case_id in best_cv_predictions:
                    cv_preds.append(best_cv_predictions[case_id])
                else:
                    cv_preds.append(row['model_predicted_enhancement'])

            if len(cv_preds) == len(common_cases):
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(cv_preds, common_cases['predicted_enhancement'])
                kappa_with_matrix[model_idx, rad_idx] = kappa
                kappa_with_matrix[rad_idx, model_idx] = kappa

np.fill_diagonal(kappa_with_matrix, 1.0)

# Add white borders for higher values
without_higher_count = 0
total_comparisons = 0
for i in range(n_rads):
    for j in range(i+1, n_rads):  # Only upper triangle
        without_val = kappa_without_matrix[i, j]
        with_val = kappa_with_matrix[i, j]
        
        if not np.isnan(without_val) and not np.isnan(with_val):
            total_comparisons += 1
            if without_val > with_val:
                without_higher_count += 1
                # Add white box
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                         fill=False, edgecolor='white', linewidth=3))
                # Also add to lower triangle for symmetry
                ax.add_patch(plt.Rectangle((i, j), 1, 1, 
                                         fill=False, edgecolor='white', linewidth=3))

# Calculate percentage
without_percentage = (without_higher_count / total_comparisons * 100) if total_comparisons > 0 else 0

ax.set_title(f'g) Inter-rater agreement without support\n($\\overline{{\\text{{κ}}}}$ = {kappa_without:.3f}, {without_percentage:.0f}% higher)',
             fontsize=14)
ax.set_xlabel('Agent', fontsize=12)
ax.set_ylabel('Agent', fontsize=12)
# Rotate x-tick labels
ax.tick_params(axis='x', rotation=45)
for tick in ax.get_xticklabels():
    tick.set_ha('right')

ax = axes[1, 3]
# all_rads is set up in panel g; reuse it here, ensuring Model is included.
if 'Model' not in all_rads:
    all_rads.append('Model')

n_rads = len(all_rads)

# Create matrix for heatmap
kappa_with_matrix = np.full((n_rads, n_rads), np.nan)

# Fill matrix for radiologist-radiologist pairs
for pa in paired_agreements:
    i = all_rads.index(pa['rad1'])
    j = all_rads.index(pa['rad2'])
    kappa_with_matrix[i, j] = pa['kappa_with']
    kappa_with_matrix[j, i] = pa['kappa_with']  # Symmetric

# Calculate model-radiologist agreement with CV optimization
model_idx = all_rads.index('Model')

for rad_idx, rad in enumerate(all_rads[:-1]):  # Exclude Model itself
    # Skip if this is the Model
    if rad == 'Model':
        continue

    # Get radiologist predictions with segmentation
    rad_data = radiologist_df[(radiologist_df['radiologist'] == rad) &
                             (radiologist_df['with_segmentation'] == True)]

    if len(rad_data) > 0:
        # Match CV predictions to radiologist cases
        common_cases = rad_data.dropna(subset=['model_predicted_enhancement'])

        if len(common_cases) > 0:
            # Create list of CV predictions for these cases
            cv_preds = []
            for idx, row in common_cases.iterrows():
                case_id = row['case_id']
                if case_id in best_cv_predictions:
                    cv_preds.append(best_cv_predictions[case_id])
                else:
                    # Fallback to original model prediction
                    cv_preds.append(row['model_predicted_enhancement'])

            if len(cv_preds) == len(common_cases):
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(cv_preds, common_cases['predicted_enhancement'])
                kappa_with_matrix[model_idx, rad_idx] = kappa
                kappa_with_matrix[rad_idx, model_idx] = kappa  # Symmetric

# Set diagonal to 1 (perfect agreement with self)
np.fill_diagonal(kappa_with_matrix, 1.0)

# rad_labels is set up in panel g; reuse it here.

# Calculate mean kappa for title
kappa_with = np.nanmean(kappa_with_matrix[~np.eye(n_rads, dtype=bool)])

# Create heatmap with same aesthetics as inter_rater_agreement_analysis.png
# Create colorbar axes to the right with some spacing
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

sns.heatmap(kappa_with_matrix, annot=False, cmap='inferno',
            vmin=0, vmax=1.0, ax=ax, cbar_ax=cax,
            cbar_kws={'label': 'Cohen\'s κ'},
            xticklabels=rad_labels,
            yticklabels=rad_labels)

# Add white borders for higher values (with model higher than without)
with_higher_count = 0
total_comparisons = 0
for i in range(n_rads):
    for j in range(i+1, n_rads):  # Only upper triangle
        without_val = kappa_without_matrix[i, j]
        with_val = kappa_with_matrix[i, j]
        
        if not np.isnan(without_val) and not np.isnan(with_val):
            total_comparisons += 1
            if with_val > without_val:
                with_higher_count += 1
                # Add white box
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                         fill=False, edgecolor='white', linewidth=3))
                # Also add to lower triangle for symmetry
                ax.add_patch(plt.Rectangle((i, j), 1, 1, 
                                         fill=False, edgecolor='white', linewidth=3))

# Calculate percentage
with_percentage = (with_higher_count / total_comparisons * 100) if total_comparisons > 0 else 0

ax.set_title(f'h) Inter-rater agreement with support\n($\\overline{{\\text{{κ}}}}$ = {kappa_with:.3f}, {with_percentage:.0f}% higher)',
             fontsize=14)
ax.set_xlabel('Agent', fontsize=12)
ax.set_ylabel('Agent', fontsize=12)
# Rotate x-tick labels
ax.tick_params(axis='x', rotation=45)
for tick in ax.get_xticklabels():
    tick.set_ha('right')

# Get paired data for segmentation impact analysis
plot_df = []
for rad in radiologist_df['radiologist'].unique():
    rad_data = radiologist_df[radiologist_df['radiologist'] == rad]
    
    if len(rad_data[rad_data['with_segmentation'] == False]) > 0 and \
       len(rad_data[rad_data['with_segmentation'] == True]) > 0:
        
        without_data = rad_data[rad_data['with_segmentation'] == False]
        with_data = rad_data[rad_data['with_segmentation'] == True]
        
        plot_df.append({
            'radiologist': rad,
            'accuracy_without': without_data['correct_prediction'].mean(),
            'accuracy_with': with_data['correct_prediction'].mean(),
            'confidence_without': without_data['confidence'].mean(),
            'confidence_with': with_data['confidence'].mean(),
            'time_without': without_data['response_time'].mean(),
            'time_with': with_data['response_time'].mean(),
            'quality_without': without_data['image_quality'].mean() if 'image_quality' in without_data else np.nan,
            'quality_with': with_data['image_quality'].mean() if 'image_quality' in with_data else np.nan,
            'years_experience': rad_data['years_experience'].iloc[0]
        })

plot_df = pd.DataFrame(plot_df)

ax = axes[0, 3]

# Calculate actual data ranges with some padding, including model data
acc_data_min = min(plot_df['accuracy_without'].min(), plot_df['accuracy_with'].min())
acc_data_max = max(plot_df['accuracy_without'].max(), plot_df['accuracy_with'].max())

# Use pair-level accuracy from Panel A
# M icon should show model baseline and model with radiologist support from nested CV
print(f"\nPanel d: Using pair-level accuracy from Panel A")

# Model without radiologist: Use pair-level accuracy from Panel A
model_acc_without = model_alone_accuracy  # Pair-level accuracy from Panel A
print(f"  Model alone (pair-level accuracy): {model_acc_without:.4f}")

# Model with radiologist: Use pair-level CV accuracy from Panel A
model_acc_with = model_together_accuracy  # Pair-level CV accuracy from Panel A
print(f"  Model with radiologist support (pair-level CV accuracy): {model_acc_with:.4f}")

# Expand range to include model values
acc_data_min = min(acc_data_min, model_acc_without, model_acc_with)
acc_data_max = max(acc_data_max, model_acc_without, model_acc_with)

acc_padding = (acc_data_max - acc_data_min) * 0.1
y_min = max(0, acc_data_min - acc_padding)
y_max = min(1, acc_data_max + acc_padding)

x_range = np.linspace(y_min, y_max, 100)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Add shaded regions - ensure symmetric coverage
# Create fill polygons for each percentage band

# For regions above y=x (improvement zones) - where "with support" is better
ax.fill_between(x_range, x_range, np.clip(x_range * 1.05, y_min, y_max), alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, np.clip(x_range * 1.05, y_min, y_max), np.clip(x_range * 1.10, y_min, y_max), alpha=0.15, color='#90EE90')
ax.fill_between(x_range, np.clip(x_range * 1.10, y_min, y_max), np.clip(x_range * 1.25, y_min, y_max), alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, np.clip(x_range * 1.25, y_min, y_max), np.clip(x_range * 1.50, y_min, y_max), alpha=0.15, color='#DDA0DD')
ax.fill_between(x_range, np.clip(x_range * 1.50, y_min, y_max), y_max, alpha=0.15, color='#FFB347')

# For regions below y=x (degradation zones) - where "without support" is better
ax.fill_between(x_range, np.clip(x_range * 0.95, y_min, y_max), x_range, alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, np.clip(x_range * 0.90, y_min, y_max), np.clip(x_range * 0.95, y_min, y_max), alpha=0.15, color='#90EE90')
ax.fill_between(x_range, np.clip(x_range * 0.80, y_min, y_max), np.clip(x_range * 0.90, y_min, y_max), alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, np.clip(x_range * 0.67, y_min, y_max), np.clip(x_range * 0.80, y_min, y_max), alpha=0.15, color='#DDA0DD')
# For >50% degradation: y < (2/3)*x, which is the reciprocal of y > 1.5*x
ax.fill_between(x_range, y_min, np.clip(x_range * 0.67, y_min, y_max), alpha=0.15, color='#FFB347')

# Scatter plot
experience_sizes = plot_df['years_experience'] * 12  # Reduced to match R/M point size (max 300)
for i, row in plot_df.iterrows():
    ax.scatter(row['accuracy_without'], row['accuracy_with'], 
               s=experience_sizes[i], alpha=0.7, color=colors[0], edgecolors='black', linewidth=0.5)
    radiologist_num = row['radiologist'].split('#')[1] if '#' in row['radiologist'] else str(i+1)
    ax.text(row['accuracy_without'], row['accuracy_with'], radiologist_num, 
            color='black', fontsize=8, ha='center', va='center')

# Plot lines
ax.plot([0, 1], [0, 1], 'k-', alpha=0.8, linewidth=1.5)  # y=x line from origin
for factor in [0.95, 1.05, 0.90, 1.10, 0.80, 1.25, 0.67, 1.50]:
    ax.plot(x_range, x_range * factor, '--', alpha=0.5, linewidth=0.6)

# Add group mean scatterpoint
mean_accuracy_without = plot_df['accuracy_without'].mean()
mean_accuracy_with = plot_df['accuracy_with'].mean()

# Plot the mean point
ax.scatter(mean_accuracy_without, mean_accuracy_with, s=300, alpha=1.0,
           color='black', edgecolors='white', linewidth=0.5, zorder=20)
ax.text(mean_accuracy_without, mean_accuracy_with, 'R', color='white',
        fontsize=12, ha='center', va='center', zorder=21)

# Calculate perpendicular line to y=x
# For a line y=x, the perpendicular has slope -1
# Distance from point to line y=x is |x-y|/sqrt(2)
# Perpendicular projection point on y=x line has coordinates ((x+y)/2, (x+y)/2)
proj_x = (mean_accuracy_without + mean_accuracy_with) / 2
proj_y = proj_x

# Draw perpendicular line from mean point to y=x line with arrowhead
ax.annotate('', xy=(mean_accuracy_without, mean_accuracy_with), 
            xytext=(proj_x, proj_y),
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--', 
                          alpha=0.7, linewidth=1.5),
            zorder=19)

# Add Model scatterpoint with 'M' - USE PAIR-LEVEL METRICS FROM PANEL A
# Use the same pair-level accuracy values from Panel A (already defined above)
model_accuracy_without = model_acc_without  # Pair-level accuracy from Panel A
model_accuracy_with = model_acc_with  # Pair-level CV accuracy from Panel A

print(f"  Panel d 'M' point: using pair-level accuracy ({model_accuracy_without:.4f}, {model_accuracy_with:.4f})")

# Plot model scatterpoint
ax.scatter(model_accuracy_without, model_accuracy_with, s=300, alpha=1.0,
           color='white', edgecolors='black', linewidth=0.5, zorder=20)
ax.text(model_accuracy_without, model_accuracy_with, 'M', color='black',
        fontsize=12, ha='center', va='center', zorder=21)

# Calculate perpendicular line for model point
model_proj_x = (model_accuracy_without + model_accuracy_with) / 2
model_proj_y = model_proj_x

# Draw perpendicular line for model point
ax.annotate('', xy=(model_accuracy_without, model_accuracy_with),
            xytext=(model_proj_x, model_proj_y),
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--',
                          alpha=0.7, linewidth=1.5),
            zorder=19)

ax.set_xlabel('Accuracy without support', fontsize=12)
ax.set_ylabel('Accuracy with support', fontsize=12)
ax.set_title('d) Impact on accuracy', fontsize=14)
ax.set_xlim(y_min, y_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)

# Add legend for shaded areas
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
area_elements = [
    Patch(facecolor='#E0E0E0', alpha=0.7, label='±0-5%'),
    Patch(facecolor='#90EE90', alpha=0.7, label='±5-10%'),
    Patch(facecolor='#87CEEB', alpha=0.7, label='±10-25%'),
    Patch(facecolor='#DDA0DD', alpha=0.7, label='±25-50%'),
    Patch(facecolor='#FFB347', alpha=0.7, label='±>50%')
]
area_legend = ax.legend(handles=area_elements, loc='upper left', framealpha=0.9)

# Add bubble size legend separately
legend_sizes = [5, 15, 25]
legend_labels = ['5 years', '15 years', '25 years']
legend_elements = []
for size, label in zip(legend_sizes, legend_labels):
    legend_elements.append(plt.scatter([], [], s=size*7, alpha=0.7, edgecolors='black',  # Reduced proportionally with actual points
                                     linewidth=0.5, label=label, color=colors[0]))
ax.add_artist(area_legend)
ax.legend(handles=legend_elements, loc='lower right', title='Experience', framealpha=0.9, borderaxespad=0.8)

ax = axes[1, 0]

conf_y_min = 5
conf_y_max = 9.5

# Create x_range that extends fully to the axis limits
x_range = np.linspace(conf_y_min, conf_y_max, 100)

# Draw shaded regions that extend to the full axis limits - ensure symmetry
# For regions above y=x (improvement zones) - where "with support" is better
ax.fill_between(x_range, x_range, np.clip(x_range * 1.05, conf_y_min, conf_y_max), alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, np.clip(x_range * 1.05, conf_y_min, conf_y_max), np.clip(x_range * 1.10, conf_y_min, conf_y_max), alpha=0.15, color='#90EE90')
ax.fill_between(x_range, np.clip(x_range * 1.10, conf_y_min, conf_y_max), np.clip(x_range * 1.25, conf_y_min, conf_y_max), alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, np.clip(x_range * 1.25, conf_y_min, conf_y_max), np.clip(x_range * 1.50, conf_y_min, conf_y_max), alpha=0.15, color='#DDA0DD')
ax.fill_between(x_range, np.clip(x_range * 1.50, conf_y_min, conf_y_max), conf_y_max, alpha=0.15, color='#FFB347')

# For regions below y=x (degradation zones) - where "without support" is better
ax.fill_between(x_range, np.clip(x_range * 0.95, conf_y_min, conf_y_max), x_range, alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, np.clip(x_range * 0.90, conf_y_min, conf_y_max), np.clip(x_range * 0.95, conf_y_min, conf_y_max), alpha=0.15, color='#90EE90')
ax.fill_between(x_range, np.clip(x_range * 0.80, conf_y_min, conf_y_max), np.clip(x_range * 0.90, conf_y_min, conf_y_max), alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, np.clip(x_range * 0.67, conf_y_min, conf_y_max), np.clip(x_range * 0.80, conf_y_min, conf_y_max), alpha=0.15, color='#DDA0DD')
# For >50% degradation: y < (2/3)*x, which is the reciprocal of y > 1.5*x
ax.fill_between(x_range, conf_y_min, np.clip(x_range * 0.67, conf_y_min, conf_y_max), alpha=0.15, color='#FFB347')

for i, row in plot_df.iterrows():
    ax.scatter(row['confidence_without'], row['confidence_with'], 
               s=experience_sizes[i], alpha=0.7, color=colors[1], edgecolors='black', linewidth=0.5)
    radiologist_num = row['radiologist'].split('#')[1] if '#' in row['radiologist'] else str(i+1)
    ax.text(row['confidence_without'], row['confidence_with'], radiologist_num, 
            color='black', fontsize=8, ha='center', va='center')

ax.plot([5, 9.5], [5, 9.5], 'k-', alpha=0.8, linewidth=1.5)
for factor in [0.95, 1.05, 0.90, 1.10, 0.80, 1.25, 0.67, 1.50]:
    ax.plot(x_range, x_range * factor, '--', alpha=0.5, linewidth=0.6)

# Add group mean scatterpoint
mean_confidence_without = plot_df['confidence_without'].mean()
mean_confidence_with = plot_df['confidence_with'].mean()

# Plot the mean point
ax.scatter(mean_confidence_without, mean_confidence_with, s=300, alpha=1.0,
           color='black', edgecolors='white', linewidth=0.5, zorder=20)
ax.text(mean_confidence_without, mean_confidence_with, 'R', color='white',
        fontsize=12, ha='center', va='center', zorder=21)

# Calculate perpendicular line to y=x
proj_x = (mean_confidence_without + mean_confidence_with) / 2
proj_y = proj_x

# Draw perpendicular line from mean point to y=x line with arrowhead
ax.annotate('', xy=(mean_confidence_without, mean_confidence_with),
            xytext=(proj_x, proj_y),
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--',
                          alpha=0.7, linewidth=1.5),
            zorder=19)

# Add model 'M' point with scaled confidence (0-1 probability to 1-10 confidence scale)
# Use ACTUAL confidence values from CV optimization (no artificial boosting)
cv_pred_df_panel_e = pd.DataFrame(best_cv_predictions)

print(f"Total CV predictions: {len(cv_pred_df_panel_e)}")
print(f"Cases with used_human=True: {cv_pred_df_panel_e['used_human'].sum()}")

# For fair comparison: look at SAME CASES (where human was used)
# Compare model-only confidence vs. combined confidence on those cases
cv_pred_df_with_human_e = cv_pred_df_panel_e[cv_pred_df_panel_e['used_human'] == True].copy()

cv_pred_df_with_human_e['model_correct'] = (cv_pred_df_with_human_e['model_pred'] == cv_pred_df_with_human_e['gt']).astype(int)
cv_pred_df_with_human_e['combined_correct'] = (cv_pred_df_with_human_e['cv_pred'] == cv_pred_df_with_human_e['gt']).astype(int)

cv_pred_df_with_human_e['model_raw_confidence'] = cv_pred_df_with_human_e['model_confidence']
cv_pred_df_with_human_e['combined_raw_confidence'] = cv_pred_df_with_human_e['combined_prob'].apply(
    lambda x: max(x, 1-x) if pd.notna(x) else 0.5
)

cv_pred_df_with_human_e['model_prob_predicted_class'] = cv_pred_df_with_human_e.apply(
    lambda row: row['model_prob'] if row['model_pred'] == 1 else (1 - row['model_prob']),
    axis=1
)

cv_pred_df_with_human_e['combined_prob_predicted_class'] = cv_pred_df_with_human_e.apply(
    lambda row: row['combined_prob'] if row['cv_pred'] == 1 else (1 - row['combined_prob']),
    axis=1
)

# Calculate Brier scores
cv_pred_df_with_human_e['model_brier'] = (cv_pred_df_with_human_e['model_prob_predicted_class'] - cv_pred_df_with_human_e['model_correct']) ** 2
cv_pred_df_with_human_e['combined_brier'] = (cv_pred_df_with_human_e['combined_prob_predicted_class'] - cv_pred_df_with_human_e['combined_correct']) ** 2

model_brier_score = cv_pred_df_with_human_e['model_brier'].mean()
combined_brier_score = cv_pred_df_with_human_e['combined_brier'].mean()

# Convert Brier to calibration quality score (0-1 scale, higher is better)
model_calib_conf_prob = 1 - model_brier_score
combined_calib_conf_prob = 1 - combined_brier_score

# Scale from 0-1 to 1-10
model_conf_without = 1 + (model_calib_conf_prob * 9)
model_conf_with = 1 + (combined_calib_conf_prob * 9)

# Para 86 / Table 1 R12 cols 4-5: model calibrated confidence (CQS scale 1-10)
_model_conf_without_std = cv_pred_df_with_human_e['model_brier'].std() * 9
_model_conf_with_std    = cv_pred_df_with_human_e['combined_brier'].std() * 9
print(f"\nModel calibrated confidence (Brier-derived CQS, paragraph 86):")
print(f"  Without radiologist support: {model_conf_without:.2f} ± {_model_conf_without_std:.2f}")
print(f"  With radiologist support:    {model_conf_with:.2f} ± {_model_conf_with_std:.2f}")
print(f"  Δ AI (with vs without):       {model_conf_with - model_conf_without:+.2f}  "
      f"(Table 1 R12 col 7)")

# Plot model scatterpoint
ax.scatter(model_conf_without, model_conf_with, s=300, alpha=1.0,
           color='white', edgecolors='black', linewidth=0.5, zorder=22)
ax.text(model_conf_without, model_conf_with, 'M', color='black',
        fontsize=12, ha='center', va='center', zorder=23)

# Store these values for statistical reporting
model_conf_without_fig = model_conf_without
model_conf_with_fig = model_conf_with

# Add perpendicular line from model point to diagonal
proj_x_model = (model_conf_without + model_conf_with) / 2
proj_y_model = proj_x_model
ax.annotate('', xy=(model_conf_without, model_conf_with),
            xytext=(proj_x_model, proj_y_model),
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--',
                          alpha=0.7, linewidth=1.5),
            zorder=18)

ax.set_xlabel('Calibrated confidence without support', fontsize=12)
ax.set_ylabel('Calibrated confidence with support', fontsize=12)
ax.set_title('e) Impact on confidence', fontsize=14)

# Set fixed axis limits for confidence scale
ax.set_xlim(5, 9.5)
ax.set_ylim(5, 9.5)

# Set ticks at 0.5 intervals
ax.set_xticks([5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])
ax.set_yticks([5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])

ax.grid(True, alpha=0.3)

# Add legend for shaded areas  
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
area_elements = [
    Patch(facecolor='#E0E0E0', alpha=0.7, label='±0-5%'),
    Patch(facecolor='#90EE90', alpha=0.7, label='±5-10%'),
    Patch(facecolor='#87CEEB', alpha=0.7, label='±10-25%'),
    Patch(facecolor='#DDA0DD', alpha=0.7, label='±25-50%'),
    Patch(facecolor='#FFB347', alpha=0.7, label='±>50%')
]
area_legend = ax.legend(handles=area_elements, loc='upper left', framealpha=0.9)

# Add bubble size legend separately
legend_elements = []
for size, label in zip(legend_sizes, legend_labels):
    legend_elements.append(plt.scatter([], [], s=size*7, alpha=0.7, edgecolors='black',  # Reduced proportionally with actual points
                                     linewidth=0.5, label=label, color=colors[1]))
ax.add_artist(area_legend)
ax.legend(handles=legend_elements, loc='lower right', title='Experience', framealpha=0.9, borderaxespad=0.8)

ax = axes[1, 1]

# Convert response time (seconds) to cases per hour
plot_df['cases_per_hour_without'] = 3600 / plot_df['time_without']
plot_df['cases_per_hour_with'] = 3600 / plot_df['time_with']

# Set logarithmic scales for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Define limits for log scale
min_limit = 30  # Start from 30 cases/hour for log scale
max_limit = 1000  # End at 1000 cases/hour

# Define max limit for model point
model_seconds_per_case = float(_upstream['model_inference_seconds_per_case'])
model_cases_per_hour = 3600 / model_seconds_per_case  # ≈ 878 cases/hour
print(f"\nModel throughput (paragraph 88 / Table 1 R13 cols 4-5):")
print(f"  {model_cases_per_hour:.0f} cases/hour ({model_seconds_per_case:.2f}s per case)")

# Create log-spaced range for shaded regions
x_range = np.logspace(np.log10(min_limit), np.log10(max_limit), 100)

# Draw shaded regions for percentage differences (work better in log scale)
ax.fill_between(x_range, x_range, x_range * 1.05, alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, x_range * 0.95, x_range, alpha=0.15, color='#E0E0E0')
ax.fill_between(x_range, x_range * 1.05, x_range * 1.10, alpha=0.15, color='#90EE90')
ax.fill_between(x_range, x_range * 0.90, x_range * 0.95, alpha=0.15, color='#90EE90')
ax.fill_between(x_range, x_range * 1.10, x_range * 1.25, alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, x_range * 0.80, x_range * 0.90, alpha=0.15, color='#87CEEB')
ax.fill_between(x_range, x_range * 1.25, x_range * 1.50, alpha=0.15, color='#DDA0DD')
ax.fill_between(x_range, x_range * 0.67, x_range * 0.80, alpha=0.15, color='#DDA0DD')
ax.fill_between(x_range, x_range * 1.50, max_limit, alpha=0.15, color='#FFB347')
ax.fill_between(x_range, min_limit, x_range * 0.67, alpha=0.15, color='#FFB347')

for i, row in plot_df.iterrows():
    ax.scatter(row['cases_per_hour_without'], row['cases_per_hour_with'], 
               s=experience_sizes[i], alpha=0.7, color=colors[2], edgecolors='black', linewidth=0.5)
    radiologist_num = row['radiologist'].split('#')[1] if '#' in row['radiologist'] else str(i+1)
    ax.text(row['cases_per_hour_without'], row['cases_per_hour_with'], radiologist_num, 
            color='black', fontsize=8, ha='center', va='center')

ax.plot([min_limit, max_limit], [min_limit, max_limit], 'k-', alpha=0.8, linewidth=1.5)
for factor in [0.95, 1.05, 0.90, 1.10, 0.80, 1.25, 0.67, 1.50]:
    ax.plot(x_range, x_range * factor, '--', alpha=0.5, linewidth=0.6)

# Group-mean scatterpoint: compute mean response time first, then convert
# to cases/hour (matches the per-radiologist time→rate convention).
mean_time_without = plot_df['time_without'].mean()
mean_time_with = plot_df['time_with'].mean()
mean_cases_without = 3600 / mean_time_without
mean_cases_with = 3600 / mean_time_with

# Plot the mean point
ax.scatter(mean_cases_without, mean_cases_with, s=300, alpha=1.0,
           color='black', edgecolors='white', linewidth=0.5, zorder=20)
ax.text(mean_cases_without, mean_cases_with, 'R', color='white',
        fontsize=12, ha='center', va='center', zorder=21)

# Calculate perpendicular line to y=x
proj_x = (mean_cases_without + mean_cases_with) / 2
proj_y = proj_x

# Draw perpendicular line from mean point to y=x line with arrowhead
ax.annotate('', xy=(mean_cases_without, mean_cases_with), 
            xytext=(proj_x, proj_y),
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--', 
                          alpha=0.7, linewidth=1.5),
            zorder=19)

# Add model 'M' point with actual measured inference time
model_seconds_per_case = float(_upstream['model_inference_seconds_per_case'])
model_cases_per_hour = 3600 / model_seconds_per_case

# Model speed is the same with and without support (no human interaction)
model_cases_without = model_cases_per_hour
model_cases_with = model_cases_per_hour

# Plot model scatterpoint on the diagonal (same speed with/without)
ax.scatter(model_cases_without, model_cases_with, s=300, alpha=1.0,
           color='white', edgecolors='black', linewidth=0.5, zorder=20)
ax.text(model_cases_without, model_cases_with, 'M', color='black',
        fontsize=12, ha='center', va='center', zorder=21)

# No perpendicular line needed since model point is exactly on the diagonal

ax.set_xlabel('Cases reported /hr without support', fontsize=12)
ax.set_ylabel('Cases reported /hr with support', fontsize=12)
ax.set_title('f) Impact on throughput', fontsize=14)
# Set axis limits for log scale
ax.set_xlim(30, 1000)
ax.set_ylim(30, 1000)

# Set custom ticks for log scale with even spacing
custom_ticks = [30, 60, 125, 250, 500, 1000]
ax.set_xticks(custom_ticks)
ax.set_yticks(custom_ticks)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

ax.grid(True, alpha=0.3)

# Add legend for shaded areas
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
area_elements = [
    Patch(facecolor='#E0E0E0', alpha=0.7, label='±0-5%'),
    Patch(facecolor='#90EE90', alpha=0.7, label='±5-10%'),
    Patch(facecolor='#87CEEB', alpha=0.7, label='±10-25%'),
    Patch(facecolor='#DDA0DD', alpha=0.7, label='±25-50%'),
    Patch(facecolor='#FFB347', alpha=0.7, label='±>50%')
]
area_legend = ax.legend(handles=area_elements, loc='upper left', framealpha=0.9)

# Add bubble size legend separately
legend_elements = []
for size, label in zip(legend_sizes, legend_labels):
    legend_elements.append(plt.scatter([], [], s=size*7, alpha=0.7, edgecolors='black',  # Reduced proportionally with actual points
                                     linewidth=0.5, label=label, color=colors[2]))
ax.add_artist(area_legend)
ax.legend(handles=legend_elements, loc='lower right', title='Experience', framealpha=0.9, borderaxespad=0.8)

# Add overall title to figure
fig1.suptitle('Impact of support on agent performance', fontsize=16, y=0.99)

plt.tight_layout()

# Save Figure_1
fig1_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_1.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure_1 saved to: {fig1_path}")

# Save as SVG
fig1_svg_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_1.svg')
plt.savefig(fig1_svg_path, format='svg', bbox_inches='tight', facecolor='white')
print(f"Figure_1 saved to: {fig1_svg_path}")

# Show the figure
plt.show()

plt.close()
