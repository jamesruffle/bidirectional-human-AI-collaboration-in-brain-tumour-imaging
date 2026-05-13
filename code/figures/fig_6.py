#!/usr/bin/env python3
"""Fig 6 - Enhancing healthcare value with artificial intelligence.

Self-contained reproduction script.

Output: data/figures/Fig_6.png (and .svg)
"""
from __future__ import annotations

import json
import os
import random as _pyrandom
import sys
import warnings
warnings.filterwarnings("ignore")

import dill
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import (
    pearsonr, ttest_rel, ttest_ind, fisher_exact,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_palette("husl")  # global palette for seaborn-rendered panels
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef,
)


stats = scipy.stats

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_6')
CSV_DIR = os.path.join(SRC_DIR, 'csv')
RNG_STATE_PATH = os.path.join(SRC_DIR, 'figure_6_rng_state.json')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')

if not os.path.isdir(CSV_DIR):
    print(f"ERROR: CSV inputs missing at {CSV_DIR}")
    sys.exit(2)
os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

print(f"Loading CSV/JSON inputs from {CSV_DIR}...")
_rc = dict(float_precision='round_trip')
# All per-radiologist intermediates (individual_perf, confidence calibration,
# equivalent-experience regression, NHS-salary financial valuation) are
# computed live from radiologist_df.csv + salary_progression.csv via shared
# helpers in _metrics_utils. No static aggregate caches.
sys.path.insert(0, os.path.dirname(HERE))
from _metrics_utils import (
    compute_individual_perf,
    compute_confidence_analysis,
    compute_equiv,
    compute_financial,
)

_RDF_PATH = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2', 'radiologist_df.csv')
_rdf_full = pd.read_csv(_RDF_PATH, **_rc)
salary_df = pd.read_csv(os.path.join(CSV_DIR, 'salary_progression.csv'), **_rc)

individual_perf_df = compute_individual_perf(_rdf_full)
confidence_analysis_df = compute_confidence_analysis(_rdf_full)
equiv_df = compute_equiv(individual_perf_df)
financial_df = compute_financial(equiv_df, salary_df)
# Paragraph 118 medians: additional radiologist experience leveraged by the
# model and the corresponding financial value, both with IQR.
_eyg_med = financial_df['equiv_years_gained'].median()
_eyg_q = financial_df['equiv_years_gained'].quantile([0.25, 0.75]).values
_alv_med = financial_df['ai_leveraged_value'].median()
_alv_q = financial_df['ai_leveraged_value'].quantile([0.25, 0.75]).values
print(
    f"\nMedian additional radiologist experience leveraged by model (paragraph 118):"
    f"\n  Equivalent years gained: {_eyg_med:.1f} years (IQR {_eyg_q[0]:.1f} - {_eyg_q[1]:.1f})"
    f"\n  AI-leveraged staff value: £{_alv_med:,.0f} (IQR £{_alv_q[0]:,.0f} - £{_alv_q[1]:,.0f})"
)
# Panel-h radiologist data is the same 8-column subset of radiologist_df.csv;
# read once and reuse instead of duplicating.
radiologist_df = _rdf_full[[
    'case_id', 'radiologist', 'with_segmentation', 'correct_prediction',
    'confidence', 'predicted_enhancement', 'has_enhancement_gt',
    'model_predicted_enhancement',
]].copy()
_cv_min = pd.read_csv(os.path.join(CSV_DIR, 'cv_predictions_min.csv'), **_rc)
_prob_df = pd.read_csv(os.path.join(CSV_DIR, 'model_case_confidence.csv'), **_rc)

with open(os.path.join(CSV_DIR, 'pair_level_metrics.json')) as _fh:
    _pair_metrics = json.load(_fh)

pair_level_model_metrics = _pair_metrics['model_alone']
pair_level_cv_metrics = _pair_metrics['model_with_human']

best_cv_predictions = _cv_min.to_dict(orient='records')
prob_data_pre = dict(zip(_prob_df['case_id'], _prob_df['top_percentile_prob']))

# Restore RNG state (panels a/b/i jitter depends on it). The state is
# stored as plain JSON (numpy + Python random states) so the bundle
# contains no opaque binary intermediates and no patient data.
if os.path.isfile(RNG_STATE_PATH) and os.path.getsize(RNG_STATE_PATH) > 0:
    import ast
    import base64
    import numpy as _np_local
    with open(RNG_STATE_PATH) as _fh:
        _rng_state = json.load(_fh)
    if _rng_state.get('numpy'):
        ns = _rng_state['numpy']
        arr = _np_local.frombuffer(
            base64.b64decode(ns['state_array_b64']),
            dtype=_np_local.dtype(ns['state_array_dtype']),
        ).copy()
        np.random.set_state((
            ns['bit_generator'], arr,
            ns['pos'], ns['has_gauss'], ns['cached_gauss'],
        ))
    if _rng_state.get('python'):
        ps = _rng_state['python']
        _state_tuple = ast.literal_eval(
            base64.b64decode(ps['state_b64']).decode()
        )
        _pyrandom.setstate((ps['version'], _state_tuple, ps['gauss']))
    print(f"  Restored RNG state from {os.path.basename(RNG_STATE_PATH)}")
else:
    print(f"  WARNING: RNG state file missing - output may not be pixel-identical")

fig6 = plt.figure(figsize=(20, 18))

# Add overall title with reduced y position
fig6.suptitle('Enhancing healthcare value with artificial intelligence', fontsize=16, y=0.92)

gs = fig6.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# Define reference color palette
reference_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Calculate model's equivalent years of experience based on accuracy
# First, establish the baseline relationship between experience and accuracy
model_equiv_years_without = None
model_equiv_years_with = None
model_equiv_years_gained = None
model_financial_value = None
model_cumulative_cost = None
model_ai_leveraged_value = None

# Get radiologist data without segmentation to establish baseline relationship
without_seg_data = individual_perf_df[individual_perf_df['with_segmentation'] == False].dropna(subset=['years_experience'])

# Fit linear regression: years_experience vs accuracy
from sklearn.linear_model import LinearRegression
X_exp = without_seg_data['years_experience'].values.reshape(-1, 1)
y_acc = without_seg_data['rad_accuracy'].values

acc_exp_model = LinearRegression().fit(X_exp, y_acc)
acc_intercept = acc_exp_model.intercept_
acc_slope = acc_exp_model.coef_[0]

print(f"\nModel Equivalent Experience Calculation:")
print(f"Baseline accuracy-experience relationship: accuracy = {acc_intercept:.4f} + {acc_slope:.4f} * years")

# Model accuracy without support - USE NESTED CV GRID SEARCH RESULTS
model_acc_without = pair_level_model_metrics.get('accuracy', None)
n_pairs = pair_level_model_metrics.get('n_pairs', 'unknown')
print(f"Model accuracy (without support, from nested CV grid search):")
print(f"  Accuracy: {model_acc_without:.4f} (n={n_pairs} radiology-case pairs)")
print(f"  (Using nested CV grid search results for consistency with Figure 1)")

# Model accuracy with support - USE NESTED CV GRID SEARCH RESULTS
model_acc_with = pair_level_cv_metrics.get('accuracy', None)
n_pairs = pair_level_cv_metrics.get('n_pairs', 'unknown')
print(f"Model accuracy (with radiologist support, from nested CV grid search):")
print(f"  Accuracy: {model_acc_with:.4f} (n={n_pairs} radiology-case pairs)")
print(f"  (Using nested CV grid search results for consistency with Figure 1)")
print(f"Accuracy change: {model_acc_without:.4f} -> {model_acc_with:.4f} (diff: {model_acc_with - model_acc_without:+.4f})")

# Calculate equivalent years of experience for model
if acc_slope > 0:
    model_equiv_years_without = (model_acc_without - acc_intercept) / acc_slope
    model_equiv_years_with = (model_acc_with - acc_intercept) / acc_slope
    model_equiv_years_gained = model_equiv_years_with - model_equiv_years_without

    # Ensure non-negative values
    model_equiv_years_without = max(0, model_equiv_years_without)
    model_equiv_years_with = max(0, model_equiv_years_with)
    model_equiv_years_gained = max(0, model_equiv_years_gained)

    print(f"Model accuracy without support: {model_acc_without:.4f} -> {model_equiv_years_without:.1f} equivalent years")
    print(f"Model accuracy with support: {model_acc_with:.4f} -> {model_equiv_years_with:.1f} equivalent years")
    print(f"Model equivalent years gained: {model_equiv_years_gained:.1f} years")

    # Calculate cumulative cost for model's equivalent experience
    model_cumulative_cost = 0
    for year in range(1, int(model_equiv_years_without) + 1):
        year_salary = salary_df[salary_df['experience_year'] == year]['annual_salary'].values[0] if year <= 49 else salary_df.iloc[-1]['annual_salary']
        model_cumulative_cost += year_salary

    # Calculate AI-leveraged value (additional years gained)
    model_ai_leveraged_value = 0
    for year in range(int(model_equiv_years_without) + 1, int(model_equiv_years_with) + 1):
        year_salary = salary_df[salary_df['experience_year'] == year]['annual_salary'].values[0] if year <= 49 else salary_df.iloc[-1]['annual_salary']
        model_ai_leveraged_value += year_salary
    _frac = model_equiv_years_gained - int(model_equiv_years_gained)
    if _frac > 0:
        _end_y = int(model_equiv_years_with) + 1
        _frac_salary = (salary_df[salary_df['experience_year'] == _end_y]['annual_salary'].values[0]
                        if _end_y <= 49 else salary_df.iloc[-1]['annual_salary'])
        model_ai_leveraged_value += _frac_salary * _frac

    print(f"Model cumulative cost: £{model_cumulative_cost:,.0f}")
    print(
        f"Model AI-leveraged value: £{model_ai_leveraged_value:,.0f}"
        f"  (rounded to nearest £1,000 for paragraph 120: £{round(model_ai_leveraged_value/1000)*1000:,.0f})"
    )
    # Paragraph 120: total model value = base + AI-leveraged.
    print(
        f"Model total value (paragraph 120):"
        f" £{model_cumulative_cost + model_ai_leveraged_value:,.0f} "
        f"(rounded to nearest £1,000: £{round((model_cumulative_cost + model_ai_leveraged_value)/1000)*1000:,.0f})"
    )

# Create nested grid for panel a and b in the first position
# Using same width and spacing as panels h and i
gs_nested = gs[0, 0].subgridspec(1, 2, wspace=0.5, width_ratios=[0.17, 0.17])
# Panel a (row 0, col 0, left): Actual experience
ax_0_0a = fig6.add_subplot(gs_nested[0, 0])
# Panel b (row 0, col 0, right): Actual cost
ax_0_0b = fig6.add_subplot(gs_nested[0, 1])

# GPU training metadata loaded from sidecar JSON so the measured training-time
# (aggregated from nnUNet cluster job logs) and the cloud-pricing assumption
# travel with the bundled source data rather than as inline literals.
with open(os.path.join(CSV_DIR, 'gpu_training_metadata.json')) as _fh:
    _gpu_meta = json.load(_fh)
gpu_hours = float(_gpu_meta['total_gpu_hours'])
gpu_cost_per_hour_gbp = float(_gpu_meta['cost_per_hour_gbp'])
_gpu_n_folds = int(_gpu_meta['n_folds'])
gpu_cost = gpu_hours * gpu_cost_per_hour_gbp

# Paragraph 118 reports total training time (5-fold) and per-fold breakdown,
# plus the rounded GPU training cost.
print(
    f"\nGPU training time (paragraph 118):"
    f"\n  Total training:   {gpu_hours:.1f} hours ({gpu_hours/24:.1f} days)"
    f"\n  Per-fold ({_gpu_n_folds}-fold): {gpu_hours/_gpu_n_folds:.1f} hours/fold ({gpu_hours/_gpu_n_folds/24:.1f} days/fold)"
    f"\n  Estimated cost:    £{gpu_cost:,.0f}  (rounded to nearest £1,000: £{round(gpu_cost/1000)*1000:,.0f})"
)

# Calculate average actual training hours and cumulative cost based on years of experience
# Use actual years of experience, not equivalent years gained
_WORKING_HOURS_PER_YEAR = float(_gpu_meta['working_hours_per_year'])
_CALENDAR_HOURS_PER_YEAR = float(_gpu_meta['calendar_hours_per_year'])
avg_actual_hours = equiv_df['years_experience'].mean() * _WORKING_HOURS_PER_YEAR

# Calculate cumulative cost based on NHS salary progression (salary_df injected by loader)

# Calculate average cumulative cost across all radiologists
total_cumulative_cost = 0
for years in equiv_df['years_experience'].values:
    radiologist_cost = 0
    for year in range(1, int(years) + 1):
        year_salary = salary_df[salary_df['experience_year'] == year]['annual_salary'].values[0] if year <= 49 else salary_df.iloc[-1]['annual_salary']
        radiologist_cost += year_salary
    total_cumulative_cost += radiologist_cost

avg_cumulative_cost = total_cumulative_cost / len(equiv_df)

# Panel a: Actual experience
x_labels = ['Model', 'Radiologist']
x_pos = np.arange(len(x_labels))
# GPU hours converted to years using actual calendar time (365 days * 24 hours = 8760 hours/year)
# Radiologist hours use working year (2000 hours/year)
years_data = [gpu_hours / _CALENDAR_HOURS_PER_YEAR, avg_actual_hours / _WORKING_HOURS_PER_YEAR]

# Plot years bars with texture for Model
bars_years = []
for i, (pos, val, label) in enumerate(zip(x_pos, years_data, x_labels)):
    is_model = label == 'Model'
    hatch_pattern = '//' if is_model else None
    bar = ax_0_0a.bar(pos, val, color=reference_colors[3] if is_model else reference_colors[1],
                      alpha=0.7, hatch=hatch_pattern, edgecolor='black', linewidth=0.5)
    bars_years.append(bar)

# Add individual radiologist scatter points
# Calculate actual training years based on years of experience
actual_training_years = equiv_df["years_experience"].values  # Already in years
x_scatter = np.full(len(actual_training_years), x_pos[1])
x_scatter = x_scatter + np.random.normal(0, 0.05, len(x_scatter))
ax_0_0a.scatter(x_scatter, actual_training_years, color=reference_colors[1], alpha=0.6, s=80, zorder=5)

for i, (x, y, rad_name, years_exp) in enumerate(zip(x_scatter, actual_training_years, equiv_df['radiologist'], equiv_df['years_experience'])):
    rad_num = rad_name.split('#')[1] if '#' in rad_name else str(i+1)
    ax_0_0a.text(x, y, rad_num, ha='center', va='center', fontsize=7, color='black', zorder=6)

ax_0_0a.set_xlabel('Agent', fontsize=10)
ax_0_0a.set_ylabel('Years', fontsize=10)
ax_0_0a.set_xticks(x_pos)
ax_0_0a.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax_0_0a.grid(True, alpha=0.3, axis='y')

# Set y-axis limits for years
ax_0_0a.set_ylim(0, 26)

# Add value annotations on top of bars with mean notation
for i, y_val in enumerate(years_data):
    if i == 1:  # Radiologist bar (which shows mean)
        ax_0_0a.text(x_pos[i], y_val + 0.2, f'x̄={y_val:.1f} yrs',
                    ha='center', va='bottom', fontsize=8)
    else:  # Model bar
        ax_0_0a.text(x_pos[i], y_val + 0.2, f'{y_val:.1f} yrs',
                    ha='center', va='bottom', fontsize=8)

ax_0_0a.set_title('a) Actual experience', fontsize=11)

# Panel b: Actual cost
cost_data = [gpu_cost, avg_cumulative_cost]

# Plot cost bars with texture for Model
bars_cost = []
for i, (pos, val, label) in enumerate(zip(x_pos, cost_data, x_labels)):
    is_model = label == 'Model'
    hatch_pattern = '//' if is_model else None
    bar = ax_0_0b.bar(pos, val, color=reference_colors[3] if is_model else reference_colors[1],
                      alpha=0.7, hatch=hatch_pattern, edgecolor='black', linewidth=0.5)
    bars_cost.append(bar)

# Add individual radiologist cost points
# Calculate cumulative cost for each radiologist
individual_costs = []
for years in equiv_df['years_experience'].values:
    radiologist_cost = 0
    for year in range(1, int(years) + 1):
        year_salary = salary_df[salary_df['experience_year'] == year]['annual_salary'].values[0] if year <= 49 else salary_df.iloc[-1]['annual_salary']
        radiologist_cost += year_salary
    individual_costs.append(radiologist_cost)

x_scatter_cost = np.full(len(individual_costs), x_pos[1])
x_scatter_cost = x_scatter_cost + np.random.normal(0, 0.05, len(x_scatter_cost))
ax_0_0b.scatter(x_scatter_cost, individual_costs, color=reference_colors[1], alpha=0.6, s=80, zorder=5)

for i, (x, y, rad_name) in enumerate(zip(x_scatter_cost, individual_costs, equiv_df['radiologist'])):
    rad_num = rad_name.split('#')[1] if '#' in rad_name else str(i+1)
    ax_0_0b.text(x, y, rad_num, ha='center', va='center', fontsize=7, color='black', zorder=6)

ax_0_0b.set_xlabel('Agent', fontsize=10)
ax_0_0b.set_ylabel('Cost (in £1000s)', fontsize=10)
ax_0_0b.set_xticks(x_pos)
ax_0_0b.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax_0_0b.grid(True, alpha=0.3, axis='y')

# Set y-axis limits for cost
ax_0_0b.set_ylim(0, 3000000)  # 3000 in thousands

# Format y-axis to show values in thousands
from matplotlib.ticker import FuncFormatter
ax_0_0b.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}"))

# Add value annotations on top of bars with mean notation
for i, c_val in enumerate(cost_data):
    if i == 1:  # Radiologist bar (which shows mean)
        # Convert to millions if over 1 million
        if c_val >= 1000000:
            ax_0_0b.text(x_pos[i], c_val + 60000, f'x̄=£{c_val/1000000:.3f}m',
                        ha='center', va='bottom', fontsize=8)
        else:
            ax_0_0b.text(x_pos[i], c_val + 60000, f'x̄=£{c_val:.0f}',
                        ha='center', va='bottom', fontsize=8)
    else:  # Model bar
        # Model cost is around £5000; round to nearest £1,000.
        if c_val >= 1000000:
            ax_0_0b.text(x_pos[i], c_val + 60000, f'£{c_val/1000000:.3f}m',
                        ha='center', va='bottom', fontsize=8)
        else:
            ax_0_0b.text(x_pos[i], c_val + 60000, f'£{round(c_val/1000)*1000:,.0f}',
                        ha='center', va='bottom', fontsize=8)

ax_0_0b.set_title('b) Actual cost', fontsize=11)

# Panel c (row 0, col 1): Equivalent years gained with AI-leveraged financial value
ax_0_1 = fig6.add_subplot(gs[0, 1])

if True:
    # Sort by equivalent years gained (descending)
    equiv_df_sorted = equiv_df.sort_values('avg_equiv_years', ascending=False).copy()
    merged_df = equiv_df_sorted.merge(financial_df[['radiologist', 'ai_leveraged_value']],
                                      on='radiologist', how='left')

    # Add model data if available
    if model_equiv_years_gained is not None and model_ai_leveraged_value is not None:
        model_row = pd.DataFrame({
            'radiologist': ['Model'],
            'avg_equiv_years': [model_equiv_years_gained],
            'years_experience': [model_equiv_years_without],  # Use equivalent baseline experience
            'ai_leveraged_value': [model_ai_leveraged_value]
        })
        # Concatenate model with radiologists
        merged_df = pd.concat([merged_df, model_row], ignore_index=True)
        # Re-sort by avg_equiv_years to intersperse model with radiologists
        merged_df = merged_df.sort_values('avg_equiv_years', ascending=False)

    color1 = reference_colors[0]
    ax_0_1.set_xlabel('Agent')
    ax_0_1.set_ylabel('Equivalent experience gained (years)', color='black')

    x_pos = np.arange(len(merged_df))

    # Create bar colors - same color scheme for all (model and radiologists)
    bar_colors = [color1 for i in range(len(merged_df))]

    # Plot bars with texture for Model
    for i in range(len(merged_df)):
        is_model = merged_df.iloc[i]['radiologist'] == 'Model'
        hatch_pattern = '//' if is_model else None
        ax_0_1.bar(x_pos[i], merged_df.iloc[i]['avg_equiv_years'],
                   color=bar_colors[i], alpha=0.7, hatch=hatch_pattern,
                   edgecolor='black', linewidth=0.5,
                   label='Equivalent experience gained (years)' if i == 0 else "")
    ax_0_1.tick_params(axis='y', labelcolor='black')
    ax_0_1.set_xticks(x_pos)

    # Create labels with years of experience (baseline->leveraged)
    labels_with_exp = []
    for i in range(len(merged_df)):
        row = merged_df.iloc[i]
        if row['radiologist'] == 'Model':
            # Format model label with its equivalent experience
            baseline_exp = row['years_experience']
            gained_exp = row.get('avg_equiv_years', 0)
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = baseline_exp + gained_exp
                labels_with_exp.append(f"Model ({baseline_exp:.0f}->{leveraged_exp:.0f}yrs)")
            else:
                labels_with_exp.append(f"Model ({baseline_exp:.0f}yrs)")
        elif 'Radiologist #' in row['radiologist']:
            rad_name = row['radiologist'].split('#')[1]
            years_exp = row.get('years_experience', None)
            gained_exp = row.get('avg_equiv_years', 0)
            if years_exp is not None and not pd.isna(years_exp):
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    labels_with_exp.append(f"R#{rad_name} ({years_exp:.0f}->{leveraged_exp:.0f}yrs)")
                else:
                    labels_with_exp.append(f"R#{rad_name} ({years_exp:.0f}yrs)")
            else:
                labels_with_exp.append(f"R#{rad_name}")
        else:
            rad_name = row['radiologist'][:3]
            years_exp = row.get('years_experience', None)
            gained_exp = row.get('avg_equiv_years', 0)
            if years_exp is not None and not pd.isna(years_exp):
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    labels_with_exp.append(f"R#{rad_name} ({years_exp:.0f}->{leveraged_exp:.0f}yrs)")
                else:
                    labels_with_exp.append(f"R#{rad_name} ({years_exp:.0f}yrs)")
            else:
                labels_with_exp.append(f"R#{rad_name}")
    # Match x-tick alignment to Panel E
    ax_0_1.set_xticklabels(labels_with_exp, rotation=45, ha='right')
    ax_0_1.grid(True, alpha=0.3, axis='y')

    # Create second y-axis for financial value
    ax2 = ax_0_1.twinx()
    color2 = reference_colors[1]
    ax2.set_ylabel('Value gained (in £1000s)', color='black')

    # Use same line plot colors for all (model and radiologists)
    line_colors = [color2 for i in range(len(merged_df))]

    # Plot each point with its color
    for i in range(len(merged_df)):
        ax2.plot(i, merged_df.iloc[i]['ai_leveraged_value'] / 1000,
                 color=line_colors[i], marker='o', markersize=8, alpha=0.8)

    # Connect all points with lines
    ax2.plot(range(len(merged_df)), merged_df['ai_leveraged_value'].values / 1000,
             color=color2, linewidth=2.5, alpha=0.8, label='Value gained (in £1000s)')

    ax2.tick_params(axis='y', labelcolor='black')

    ax_0_1.set_title('c) Experience and financial value gained from support')
    
    # Align y-axes so both start at 0
    ax_0_1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    # Add legends
    lines1, labels1 = ax_0_1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_0_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Panel c (row 0, col 2): Cumulative experience and value leveraged
ax_0_2 = fig6.add_subplot(gs[0, 2])

if True:
    financial_df_sorted = financial_df.copy()
    financial_df_sorted['total_value'] = financial_df_sorted['cumulative_salary_to_date'] + financial_df_sorted['ai_leveraged_value']

    # Add model data if available
    if model_cumulative_cost is not None and model_ai_leveraged_value is not None:
        model_financial_row = pd.DataFrame({
            'radiologist': ['Model'],
            'cumulative_salary_to_date': [model_cumulative_cost],
            'ai_leveraged_value': [model_ai_leveraged_value],
            'total_value': [model_cumulative_cost + model_ai_leveraged_value],
            'current_experience': [model_equiv_years_without] if model_equiv_years_without is not None else [0],
            'equiv_years_gained': [model_equiv_years_gained] if model_equiv_years_gained is not None else [0]
        })
        financial_df_sorted = pd.concat([financial_df_sorted, model_financial_row], ignore_index=True)

    financial_df_sorted = financial_df_sorted.sort_values('total_value', ascending=False)

    if 'years_experience' in equiv_df.columns:
        exp_data = equiv_df[['radiologist', 'years_experience']].drop_duplicates()
        # Only merge for radiologist rows, not model
        rad_mask = financial_df_sorted['radiologist'] != 'Model'
        rad_rows = financial_df_sorted[rad_mask].merge(exp_data, on='radiologist', how='left')
        model_rows = financial_df_sorted[~rad_mask]
        if model_equiv_years_without is not None:
            model_rows['years_experience'] = model_equiv_years_without
        financial_df_sorted = pd.concat([rad_rows, model_rows], ignore_index=True)
        financial_df_sorted = financial_df_sorted.sort_values('total_value', ascending=False)

    radiologists = financial_df_sorted['radiologist']
    x_pos_c = np.arange(len(radiologists))

    # Create bar colors - different colors for model
    cumulative_colors = [reference_colors[3] if financial_df_sorted.iloc[i]['radiologist'] == 'Model' else reference_colors[0]
                         for i in range(len(financial_df_sorted))]
    ai_colors = [reference_colors[4] if financial_df_sorted.iloc[i]['radiologist'] == 'Model' else reference_colors[2]
                 for i in range(len(financial_df_sorted))]

    # Set up left y-axis for experience (BARS)
    # Plot experience as stacked bars showing baseline + gained experience

    # Plot experience as stacked bars on left y-axis
    # Use unified colors: one for base, one for gained (regardless of agent type)
    # Use same colors as panels E, F, G for consistency
    base_exp_color = reference_colors[0]  # Blue - without support (base experience)
    gained_exp_color = reference_colors[1]  # Orange - with support (agent-leveraged experience)

    for i in range(len(financial_df_sorted)):
        row = financial_df_sorted.iloc[i]

        if 'years_experience' in row and not pd.isna(row.get('years_experience')):
            current_exp = row['years_experience']
            is_model = row['radiologist'] == 'Model'

            # Plot baseline experience (bottom bar) - same color for all agents
            # Add hatch pattern for model to match Panel C
            hatch_pattern = '//' if is_model else None
            ax_0_2.bar(i, current_exp, color=base_exp_color, alpha=0.7,
                      edgecolor='black', linewidth=0.5, hatch=hatch_pattern, zorder=2, width=0.7)

            # For agents with equiv_years_gained, add stacked bar for gained experience
            if 'equiv_years_gained' in row and not pd.isna(row.get('equiv_years_gained')) and row.get('equiv_years_gained', 0) > 0:
                gained_exp = row['equiv_years_gained']
                final_exp = current_exp + gained_exp

                # Plot gained experience (stacked on top of baseline) - same color for all agents
                # Add hatch pattern for model to match Panel C
                ax_0_2.bar(i, gained_exp, bottom=current_exp, color=gained_exp_color, alpha=0.8,
                          edgecolor='black', linewidth=0.5, hatch=hatch_pattern, zorder=2, width=0.7,
                          label='Experience gained' if i == 0 else '')

    print(f"\nMedian staff experience with vs without model support (paragraph 118):")

    # Filter for radiologists only (exclude model)
    radiologists_only = financial_df_sorted[financial_df_sorted['radiologist'] != 'Model'].copy()

    # Calculate total experience (baseline + gained) for each radiologist
    radiologists_only['total_experience'] = radiologists_only['years_experience'] + radiologists_only['equiv_years_gained'].fillna(0)
    radiologists_only['total_value'] = radiologists_only['cumulative_salary_to_date'] + radiologists_only['ai_leveraged_value'].fillna(0)

    # Calculate statistics for experience
    baseline_exp_values = radiologists_only['years_experience'].values
    total_exp_values = radiologists_only['total_experience'].values
    median_baseline_exp = np.median(baseline_exp_values)
    iqr_lower_baseline_exp = np.percentile(baseline_exp_values, 25)
    iqr_upper_baseline_exp = np.percentile(baseline_exp_values, 75)
    median_total_exp = np.median(total_exp_values)
    iqr_lower_total_exp = np.percentile(total_exp_values, 25)
    iqr_upper_total_exp = np.percentile(total_exp_values, 75)

    # Calculate statistics for financial value
    baseline_value_values = radiologists_only['cumulative_salary_to_date'].values
    total_value_values = radiologists_only['total_value'].values
    median_baseline_value = np.median(baseline_value_values)
    iqr_lower_baseline_value = np.percentile(baseline_value_values, 25)
    iqr_upper_baseline_value = np.percentile(baseline_value_values, 75)
    median_total_value = np.median(total_value_values)
    iqr_lower_total_value = np.percentile(total_value_values, 25)
    iqr_upper_total_value = np.percentile(total_value_values, 75)

    # Run paired t-tests
    from scipy.stats import ttest_rel
    exp_ttest = ttest_rel(baseline_exp_values, total_exp_values)
    value_ttest = ttest_rel(baseline_value_values, total_value_values)

    print(f"\nSTAFF EXPERIENCE:")
    print(f"  WITHOUT model support: {median_baseline_exp:.1f} years (IQR {iqr_lower_baseline_exp:.1f} - {iqr_upper_baseline_exp:.1f})")
    print(f"  WITH model support:    {median_total_exp:.1f} years (IQR {iqr_lower_total_exp:.1f} - {iqr_upper_total_exp:.1f})")
    # Paragraph 118 also expresses the experience as hours (× 2000 working hours/year).
    # Manuscript uses "~22,000 hours" with comma; print both formats for unambiguous matching.
    print(f"  WITHOUT model support, hours-equivalent: ~{median_baseline_exp*_WORKING_HOURS_PER_YEAR:,.0f} hours "
          f"(IQR {iqr_lower_baseline_exp*_WORKING_HOURS_PER_YEAR:.0f} - {iqr_upper_baseline_exp*_WORKING_HOURS_PER_YEAR:.0f})  "
          f"[paragraph 118: ~{median_baseline_exp*_WORKING_HOURS_PER_YEAR:.0f} hours]")
    print(f"  Paired t-test: t({len(radiologists_only)-1}) = {exp_ttest.statistic:.3f}, p = {exp_ttest.pvalue:.6f}")
    if exp_ttest.pvalue < 0.0001:
        print(f"  Significance: p < 0.0001 (****)")
    elif exp_ttest.pvalue < 0.001:
        print(f"  Significance: p < 0.001 (***)")
    elif exp_ttest.pvalue < 0.01:
        print(f"  Significance: p < 0.01 (**)")
    elif exp_ttest.pvalue < 0.05:
        print(f"  Significance: p < 0.05 (*)")
    else:
        print(f"  Significance: p ≥ 0.05 (ns)")

    print(f"\nSTAFF FINANCIAL VALUE:")
    print(f"  WITHOUT model support: £{median_baseline_value:,.0f} (IQR £{iqr_lower_baseline_value:,.0f} - £{iqr_upper_baseline_value:,.0f})")
    print(f"  WITH model support:    £{median_total_value:,.0f} (IQR £{iqr_lower_total_value:,.0f} - £{iqr_upper_total_value:,.0f})")
    print(f"  Paired t-test: t({len(radiologists_only)-1}) = {value_ttest.statistic:.3f}, p = {value_ttest.pvalue:.6f}")
    if value_ttest.pvalue < 0.0001:
        print(f"  Significance: p < 0.0001 (****)")
    elif value_ttest.pvalue < 0.001:
        print(f"  Significance: p < 0.001 (***)")
    elif value_ttest.pvalue < 0.01:
        print(f"  Significance: p < 0.01 (**)")
    elif value_ttest.pvalue < 0.05:
        print(f"  Significance: p < 0.05 (*)")
    else:
        print(f"  Significance: p ≥ 0.05 (ns)")


    # Create second y-axis for financial values (SCATTER POINTS)
    ax2_0_2 = ax_0_2.twinx()

    # Plot financial values as scatter points on right y-axis
    # Circles = base value, Diamonds = gained value
    for i in range(len(financial_df_sorted)):
        row = financial_df_sorted.iloc[i]
        is_model = row['radiologist'] == 'Model'

        base_value = row.get('cumulative_salary_to_date', 0) / 1000
        gained_value = row.get('ai_leveraged_value', 0) / 1000
        total_value = base_value + gained_value

        # Plot base value as CIRCLE - black with no border (match Panel C markersize=8)
        ax2_0_2.scatter(i, base_value, s=64, marker='o',
                       color='black', zorder=10, alpha=0.7)

        # Plot total value (base + gained) as DIAMOND - black with no border (match Panel C markersize=8)
        if gained_value > 0:
            ax2_0_2.scatter(i, total_value, s=64, marker='D',
                           color='black', zorder=11, alpha=0.9)

            # Draw a vertical line connecting base to total value
            ax2_0_2.plot([i, i], [base_value, total_value],
                        color='gray', linewidth=1.5, alpha=0.5, zorder=9)

    # Add legend manually (include both experience bars and financial scatter points)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        # Experience bars (left axis) - unified colors
        Patch(facecolor=base_exp_color, alpha=0.7, edgecolor='black', label='Base experience'),
        Patch(facecolor=gained_exp_color, alpha=0.8, edgecolor='black', label='Agent-leveraged experience'),
        # Financial scatter points (right axis) - shape indicates base vs total
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=8, label='Base value',
               linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=8, label='Agent-leveraged value',
               linestyle='None')
    ]

    ax_0_2.set_xlabel('Agent')
    # Match y-axis label styling to Panel C (no bold, black color)
    ax_0_2.set_ylabel('Equivalent experience (years) [bars]', color='black')
    ax2_0_2.set_ylabel('Value (in £1000s) [points]', color='black')
    ax_0_2.set_title('d) Cumulative experience and financial value leveraged')

    # Match y-axis tick styling to Panel C
    ax_0_2.tick_params(axis='y', labelcolor='black')
    ax2_0_2.tick_params(axis='y', labelcolor='black')

    # Ensure both axes have the same x-limits for proper alignment
    ax_0_2.set_xlim(-0.5, len(financial_df_sorted) - 0.5)
    ax2_0_2.set_xlim(-0.5, len(financial_df_sorted) - 0.5)

    ax_0_2.set_xticks(x_pos_c)

    labels_with_exp_c = []
    for i in range(len(financial_df_sorted)):
        row = financial_df_sorted.iloc[i]
        if row['radiologist'] == 'Model':
            years_exp = row.get('years_experience', model_equiv_years_without)
            gained_exp = row.get('equiv_years_gained', 0)
            if years_exp is not None:
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    labels_with_exp_c.append(f"Model ({years_exp:.0f}->{leveraged_exp:.0f}yrs)")
                else:
                    labels_with_exp_c.append(f"Model ({years_exp:.0f}yrs)")
            else:
                labels_with_exp_c.append("Model")
        elif 'Radiologist #' in row['radiologist']:
            rad_name = row['radiologist'].split('#')[1]
            years_exp = row.get('years_experience', None)
            gained_exp = row.get('equiv_years_gained', 0)
            if years_exp is not None and not pd.isna(years_exp):
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    labels_with_exp_c.append(f"R#{rad_name} ({years_exp:.0f}->{leveraged_exp:.0f}yrs)")
                else:
                    labels_with_exp_c.append(f"R#{rad_name} ({years_exp:.0f}yrs)")
            else:
                labels_with_exp_c.append(f"R#{rad_name}")
        else:
            rad_name = row['radiologist'][:3]
            years_exp = row.get('years_experience', None)
            gained_exp = row.get('equiv_years_gained', 0)
            if years_exp is not None and not pd.isna(years_exp):
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    labels_with_exp_c.append(f"R#{rad_name} ({years_exp:.0f}->{leveraged_exp:.0f}yrs)")
                else:
                    labels_with_exp_c.append(f"R#{rad_name} ({years_exp:.0f}yrs)")
            else:
                labels_with_exp_c.append(f"R#{rad_name}")
    # Match x-tick alignment to Panel E
    ax_0_2.set_xticklabels(labels_with_exp_c, rotation=45, ha='right')
    ax_0_2.legend(handles=legend_elements, loc='upper right')
    ax_0_2.grid(True, alpha=0.3, axis='y')

    if True:
        # Set left y-axis limits based on STACKED experience bars
        # Bars show baseline + gained, so max is the tallest stacked bar
        if 'years_experience' in financial_df_sorted.columns:
            max_stacked_exp = 0
            for i in range(len(financial_df_sorted)):
                row = financial_df_sorted.iloc[i]
                if 'years_experience' in row and not pd.isna(row.get('years_experience')):
                    current_exp = row['years_experience']
                    gained_exp = row.get('equiv_years_gained', 0) if not pd.isna(row.get('equiv_years_gained')) else 0
                    total_exp = current_exp + gained_exp
                    max_stacked_exp = max(max_stacked_exp, total_exp)

            # Set right y-axis to 7000 (in £1000s) as requested
            # Scale left y-axis proportionally to maintain visual correspondence
            max_total_value = financial_df_sorted['total_value'].max() / 1000  # Convert to thousands
            if max_total_value > 0:
                scale_factor = 7000 / (max_total_value * 1.15)
                ax_0_2.set_ylim(0, max_stacked_exp * 1.15 * scale_factor)
            else:
                ax_0_2.set_ylim(0, max_stacked_exp * 1.15)

        # Set right y-axis limits to 7000 (in £1000s)
        ax2_0_2.set_ylim(0, 7000)

# Calculate model performance metrics for panels e, f, g
# USE NESTED CV GRID SEARCH RESULTS for consistency with Figure 1 and panels c, d
# Use nested CV grid search results (model alone, without radiologist support)
model_metrics = {
    'accuracy': pair_level_model_metrics.get('accuracy', 0),
    'precision': pair_level_model_metrics.get('precision', 0),
    'recall': pair_level_model_metrics.get('recall', 0),
    'specificity': pair_level_model_metrics.get('specificity', 0),
    'f1': pair_level_model_metrics.get('f1', 0)
}
print(f"\nModel metrics for panels e, f, g (from nested CV grid search, n={pair_level_model_metrics.get('n_pairs', 'unknown')} pairs):")
print(f"  Accuracy: {model_metrics['accuracy']:.4f}")
print(f"  Precision: {model_metrics['precision']:.4f}")
print(f"  Recall: {model_metrics['recall']:.4f}")
print(f"  Specificity: {model_metrics['specificity']:.4f}")
print(f"  F1: {model_metrics['f1']:.4f}")

# Calculate radiologist performance metrics by condition
# Without segmentation
without_data = radiologist_df[radiologist_df['with_segmentation'] == False]
rad_metrics_without = {
    'accuracy': accuracy_score(without_data['has_enhancement_gt'],
                             without_data['predicted_enhancement']),
    'precision': precision_score(without_data['has_enhancement_gt'],
                               without_data['predicted_enhancement'], zero_division=0),
    'recall': recall_score(without_data['has_enhancement_gt'],
                         without_data['predicted_enhancement'], zero_division=0),
    'f1': f1_score(without_data['has_enhancement_gt'],
                 without_data['predicted_enhancement'])
}
tn = ((without_data['predicted_enhancement'] == 0) &
      (without_data['has_enhancement_gt'] == 0)).sum()
fp = ((without_data['predicted_enhancement'] == 1) &
      (without_data['has_enhancement_gt'] == 0)).sum()
rad_metrics_without['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

# With segmentation
with_data = radiologist_df[radiologist_df['with_segmentation'] == True]
rad_metrics_with = {
    'accuracy': accuracy_score(with_data['has_enhancement_gt'],
                             with_data['predicted_enhancement']),
    'precision': precision_score(with_data['has_enhancement_gt'],
                               with_data['predicted_enhancement'], zero_division=0),
    'recall': recall_score(with_data['has_enhancement_gt'],
                         with_data['predicted_enhancement'], zero_division=0),
    'f1': f1_score(with_data['has_enhancement_gt'],
                 with_data['predicted_enhancement'])
}
tn = ((with_data['predicted_enhancement'] == 0) &
      (with_data['has_enhancement_gt'] == 0)).sum()
fp = ((with_data['predicted_enhancement'] == 1) &
      (with_data['has_enhancement_gt'] == 0)).sum()
rad_metrics_with['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

# Calculate model confidence metrics before panels
model_conf_metrics = {'without': {}, 'with': {}}

if True:
    # Get common cases for model evaluation
    common_cases = radiologist_df.dropna(subset=['model_predicted_enhancement'])

    # Load probability maps for model confidence analysis
    print("\nCalculating model confidence metrics...")
    # prob_data is pre-computed and injected by the loader (see prob_data_pre); avoids
    # re-reading 564 *_et_probability.nii.gz from /media/jruffle/DATA/... at runtime.
    prob_data = dict(prob_data_pre)
    unique_cases = common_cases['case_id'].unique()
    print(f"Total unique cases to process: {len(unique_cases)}")
    print(f"Loaded probability maps: {len(prob_data)}/{len(unique_cases)} (from pre-computed CSV)")

    if prob_data:
        model_probs = []
        model_correct = []
        print(f"Processing confidence for {len(common_cases)} radiologist-case reviews")
        for _, row in common_cases.iterrows():
            if row['case_id'] in prob_data:
                model_prob = prob_data[row['case_id']]
                confidence_score = abs(model_prob - 0.5) * 2 * 10  # Scale to 1-10
                model_probs.append(confidence_score)
                model_correct.append(row['model_predicted_enhancement'] == row['has_enhancement_gt'])

        if len(model_probs) > 10:
            model_probs = np.array(model_probs)
            model_correct = np.array(model_correct)

            # Calculate confidence-accuracy correlation
            from scipy.stats import pearsonr
            model_conf_metrics['without']['conf_acc_corr'], _ = pearsonr(model_probs, model_correct.astype(float))

            # Median split for the model-alone calibration_diff. The model-alone
            # confidence distribution is concentrated at the maximum (10/10) for
            # most cases — Q1, median, and Q3 are all 10 — so a Q3/Q1 quartile
            # split (used for radiologists and the model-with-support arm) does
            # not separate cases meaningfully here. Median split separates "at
            # maximum" from "below maximum" which is the only meaningful contrast
            # available for this arm.
            median_conf = np.median(model_probs)
            high_conf_acc = model_correct[model_probs >= median_conf].mean() if np.sum(model_probs >= median_conf) > 0 else 0
            low_conf_acc = model_correct[model_probs < median_conf].mean() if np.sum(model_probs < median_conf) > 0 else 0
            model_conf_metrics['without']['calibration_diff'] = high_conf_acc - low_conf_acc

            # Calculate confidence bias (correct - incorrect confidence)
            correct_conf = model_probs[model_correct].mean() if np.sum(model_correct) > 0 else 0
            incorrect_conf = model_probs[~model_correct].mean() if np.sum(~model_correct) > 0 else 0
            model_conf_metrics['without']['confidence_bias'] = correct_conf - incorrect_conf

            print(f"Model without support - Conf-Acc Corr: {model_conf_metrics['without']['conf_acc_corr']:.3f}")
            print(f"Model without support - Calibration Diff: {model_conf_metrics['without']['calibration_diff']:.3f}")
            print(f"Model without support - Confidence Bias: {model_conf_metrics['without']['confidence_bias']:.3f}")
            print(f"Model without support - N reviews: {len(model_probs)}")

    # Model with support metrics (from CV optimization)
    if True:
        cv_predictions = best_cv_predictions

        # Extract probabilities and correctness from CV predictions
        cv_probs = []
        cv_correct = []
        missing_combined_count = 0
        for pred in cv_predictions:
            if 'combined_prob' not in pred:
                missing_combined_count += 1
                # If combined_prob is missing, skip this prediction
                print(f"WARNING: Missing combined_prob in CV prediction for case {pred.get('case_id', 'unknown')}")
                continue

            combined_prob = pred['combined_prob']
            confidence_score = abs(combined_prob - 0.5) * 2 * 10  # Scale to 1-10
            cv_probs.append(confidence_score)
            cv_correct.append(pred['cv_pred'] == pred['gt'])

        if missing_combined_count > 0:
            print(f"WARNING: {missing_combined_count} CV predictions missing combined_prob field")

        if len(cv_probs) > 10:
            cv_probs = np.array(cv_probs)
            cv_correct = np.array(cv_correct)

            # Calculate metrics for model with support
            model_conf_metrics['with']['conf_acc_corr'], _ = pearsonr(cv_probs, cv_correct.astype(float))

            # Q3/Q1 quartile split for the model-with-support calibration_diff,
            # matching the radiologist convention in
            # _metrics_utils.compute_confidence_analysis.
            q3_conf = np.quantile(cv_probs, 0.75)
            q1_conf = np.quantile(cv_probs, 0.25)
            high_conf_acc = cv_correct[cv_probs >= q3_conf].mean() if np.sum(cv_probs >= q3_conf) > 0 else 0
            low_conf_acc = cv_correct[cv_probs <= q1_conf].mean() if np.sum(cv_probs <= q1_conf) > 0 else 0
            model_conf_metrics['with']['calibration_diff'] = high_conf_acc - low_conf_acc

            correct_conf = cv_probs[cv_correct].mean() if np.sum(cv_correct) > 0 else 0
            incorrect_conf = cv_probs[~cv_correct].mean() if np.sum(~cv_correct) > 0 else 0
            model_conf_metrics['with']['confidence_bias'] = correct_conf - incorrect_conf

            print(f"Model with support - Conf-Acc Corr: {model_conf_metrics['with']['conf_acc_corr']:.3f}")
            print(f"Model with support - Calibration Diff: {model_conf_metrics['with']['calibration_diff']:.3f}")
            print(f"Model with support - Confidence Bias: {model_conf_metrics['with']['confidence_bias']:.3f}")

# ── Bootstrap 95% CIs for the Model conf-acc / calibration / confidence-bias
# metrics (Table 1 rows 8-9 cols 3-4 — AI arms). Pair-level percentile bootstrap.
# Per-arm calibration_diff conventions match the point-estimate logic above:
#   • without-support (model alone): median split (confidence distribution is
#     concentrated at the maximum so Q3/Q1 is degenerate)
#   • with-support (model+rad CV): Q3/Q1 quartile split (cv_probs is spread,
#     and this matches the radiologist Q3/Q1 convention in compute_confidence_analysis)
# B=5000, seed=20260505 — same convention as the existing fig_1 bootstrap.
print("\nBootstrap 95% CIs for Model AI-arm confidence metrics (B=5000, seed=20260505):")
def _ai_metric_bootstrap_cis(probs, correct, split, B=5000, seed=20260505):
    """Bootstrap CIs for (conf_acc_corr, calibration_diff, confidence_bias).
    `split`: 'median' (model-alone arm) or 'q3q1' (model-with arm)."""
    from scipy.stats import pearsonr
    probs = np.asarray(probs, dtype=float); correct = np.asarray(correct, dtype=bool)
    rng = np.random.RandomState(seed)
    n = len(probs)
    boot_corr = []; boot_calib = []; boot_bias = []
    for _ in range(B):
        ix = rng.choice(n, size=n, replace=True)
        p = probs[ix]; c = correct[ix]
        if p.std() == 0 or c.std() == 0: continue
        try:
            r, _ = pearsonr(p, c.astype(float))
            if split == 'median':
                thr = np.median(p)
                high_acc = c[p >= thr].mean() if (p >= thr).any() else 0.0
                low_acc  = c[p <  thr].mean() if (p <  thr).any() else 0.0
            else:  # q3q1
                q3 = np.quantile(p, 0.75); q1 = np.quantile(p, 0.25)
                high_acc = c[p >= q3].mean() if (p >= q3).any() else 0.0
                low_acc  = c[p <= q1].mean() if (p <= q1).any() else 0.0
            calib = high_acc - low_acc
            corr_conf = p[c].mean() if c.any() else 0.0
            wrong_conf = p[~c].mean() if (~c).any() else 0.0
            bias = corr_conf - wrong_conf
            boot_corr.append(r); boot_calib.append(calib); boot_bias.append(bias)
        except Exception:
            pass
    def _ci(a):
        a = np.asarray(a)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return _ci(boot_corr), _ci(boot_calib), _ci(boot_bias)

# Without-support arm (model alone) — median split
if 'model_probs' in dir() and len(model_probs) > 10:
    _ci_corr_w, _ci_calib_w, _ci_bias_w = _ai_metric_bootstrap_cis(model_probs, model_correct, split='median')
    print(f"  Model alone           Conf-Acc Corr: {model_conf_metrics['without']['conf_acc_corr']:+.3f} [{_ci_corr_w[0]:+.3f}, {_ci_corr_w[1]:+.3f}]")
    print(f"  Model alone           Calibration Diff: {model_conf_metrics['without']['calibration_diff']:+.3f} [{_ci_calib_w[0]:+.3f}, {_ci_calib_w[1]:+.3f}]")
    print(f"  Model alone           Confidence Bias: {model_conf_metrics['without']['confidence_bias']:+.3f} [{_ci_bias_w[0]:+.3f}, {_ci_bias_w[1]:+.3f}]")

# With-support arm (model+rad CV) — Q3/Q1 quartile split
if 'cv_probs' in dir() and len(cv_probs) > 10:
    _ci_corr_m, _ci_calib_m, _ci_bias_m = _ai_metric_bootstrap_cis(cv_probs, cv_correct, split='q3q1')
    print(f"  Model with rad        Conf-Acc Corr: {model_conf_metrics['with']['conf_acc_corr']:+.3f} [{_ci_corr_m[0]:+.3f}, {_ci_corr_m[1]:+.3f}]")
    print(f"  Model with rad        Calibration Diff: {model_conf_metrics['with']['calibration_diff']:+.3f} [{_ci_calib_m[0]:+.3f}, {_ci_calib_m[1]:+.3f}]")
    print(f"  Model with rad        Confidence Bias: {model_conf_metrics['with']['confidence_bias']:+.3f} [{_ci_bias_m[0]:+.3f}, {_ci_bias_m[1]:+.3f}]")

# ── Reader-level bootstrap CIs for radiologist-arm confidence metrics
# (Table 1 rows 9, 10 cols 1-2). Resamples the n=11 reader values from
# confidence_analysis_df B=5000 times, takes mean of each resample.
print("\nReader-level bootstrap 95% CIs for radiologist confidence metrics:")
def _reader_metric_ci(values, B=5000, seed=20260505):
    a = np.asarray(values, dtype=float); a = a[~np.isnan(a)]
    rng = np.random.RandomState(seed)
    n = len(a)
    boot = np.fromiter(
        (a[rng.choice(n, size=n, replace=True)].mean() for _ in range(B)),
        dtype=float, count=B,
    )
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

_reader_means = {}
for ws, label in [(False, 'Radiologist alone'), (True, 'Radiologist with model')]:
    _ws_data = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == ws]
    _reader_means[ws] = {}
    for metric in ['conf_acc_corr', 'calibration_diff', 'confidence_bias']:
        m, lo, hi = _reader_metric_ci(_ws_data[metric].values)
        _reader_means[ws][metric] = m
        sd = float(np.std(_ws_data[metric].values, ddof=1))
        print(f"  {label:<28s}  {metric:<18s}  {m:+.3f} ± {sd:.3f} [{lo:+.3f}, {hi:+.3f}]")

# ── Δ values (Table 1 rows 9, 10 cols 5 [Δ Human] and 7 [Δ AI]) ──
# Derived from the fig_6 point estimates above (with - without):
print("\nΔ values (Table 1 rows 9, 10 cols 5 [Δ Human] and 7 [Δ AI]):")
for metric, label in [('conf_acc_corr', 'Confidence-accuracy correlation'),
                       ('calibration_diff', 'Calibration difference')]:
    delta_human = _reader_means[True][metric] - _reader_means[False][metric]
    delta_ai = (model_conf_metrics['with'].get(metric, float('nan'))
                - model_conf_metrics['without'].get(metric, float('nan')))
    # Paired bootstrap CIs for Δ Human (reader-paired)
    arr_w = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == False].set_index('radiologist')[metric]
    arr_m = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == True].set_index('radiologist')[metric]
    common_rads = arr_w.index.intersection(arr_m.index)
    pa = arr_w.loc[common_rads].values
    pb = arr_m.loc[common_rads].values
    rng = np.random.RandomState(20260505)
    n = len(common_rads)
    boot_h = np.fromiter(((pb[ix] - pa[ix]).mean() for ix in (rng.choice(n, size=n, replace=True) for _ in range(5000))), dtype=float, count=5000)
    print(f"  Δ Human  {label:<35s} {delta_human:+.3f} [{np.percentile(boot_h, 2.5):+.3f}, {np.percentile(boot_h, 97.5):+.3f}]")
    print(f"  Δ AI     {label:<35s} {delta_ai:+.3f}")


# Panel a (row 1, col 0): Confidence-accuracy correlation by radiologist
ax_1_0 = fig6.add_subplot(gs[1, 0])

if True:
    radiologists = confidence_analysis_df['radiologist'].unique()

    # Extract experience data
    radiologist_exp_dict = {}
    exp_data = individual_perf_df[['radiologist', 'years_experience']].drop_duplicates()
    for rad in radiologists:
        rad_exp_data = exp_data[exp_data['radiologist'] == rad]
        if len(rad_exp_data) > 0 and not pd.isna(rad_exp_data['years_experience'].iloc[0]):
            years_exp = rad_exp_data['years_experience'].iloc[0]
        else:
            years_exp = 0
        radiologist_exp_dict[rad] = years_exp

    # Prepare data for panel a
    panel_data = []
    for rad in radiologists:
        without_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                            (confidence_analysis_df['with_segmentation'] == False)]
        with_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                         (confidence_analysis_df['with_segmentation'] == True)]

        # Get gained experience from equiv_df
        gained_exp = 0
        rad_equiv_data = equiv_df[equiv_df['radiologist'] == rad]
        if len(rad_equiv_data) > 0:
            gained_exp = rad_equiv_data['avg_equiv_years'].iloc[0]

        panel_data.append({
            'radiologist': rad,
            'years_exp': radiologist_exp_dict.get(rad, 0),
            'gained_exp': gained_exp,
            'without_corr': without_data['conf_acc_corr'].iloc[0] if len(without_data) > 0 else 0,
            'with_corr': with_data['conf_acc_corr'].iloc[0] if len(with_data) > 0 else 0,
        })

    # Add model to panel data if metrics are available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if 'conf_acc_corr' in model_conf_metrics['without'] and 'conf_acc_corr' in model_conf_metrics['with']:
            panel_data.append({
                'radiologist': 'Model',
                'years_exp': model_equiv_years_without if model_equiv_years_without is not None else 0,
                'gained_exp': model_equiv_years_gained if model_equiv_years_gained is not None else 0,
                'without_corr': model_conf_metrics['without']['conf_acc_corr'],
                'with_corr': model_conf_metrics['with']['conf_acc_corr'],
            })

    # Sort by with_corr (largest to smallest)
    panel_data_a = sorted(panel_data, key=lambda x: x['with_corr'], reverse=True)
    radiologists_a = [item['radiologist'] for item in panel_data_a]
    without_corr = [item['without_corr'] for item in panel_data_a]
    with_corr = [item['with_corr'] for item in panel_data_a]
    
    x_pos = np.arange(len(radiologists_a))
    width = 0.35
    
    colors_seg = [reference_colors[0], reference_colors[1]]

    # Plot bars with texture for Model
    for i, rad in enumerate(radiologists_a):
        is_model = rad == 'Model'
        hatch_pattern = '//' if is_model else None

        ax_1_0.bar(x_pos[i] - width/2, without_corr[i], width,
                  color=colors_seg[0], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
        ax_1_0.bar(x_pos[i] + width/2, with_corr[i], width,
                  color=colors_seg[1], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
    
    ax_1_0.set_xlabel('Agent')
    ax_1_0.set_ylabel('Confidence-accuracy correlation\n[Higher = better calibrated]')
    ax_1_0.set_title('e) Confidence-accuracy correlation')
    ax_1_0.set_xticks(x_pos)

    xtick_labels_a = []
    for item in panel_data_a:
        rad_name = item['radiologist']
        years_exp = item.get('years_exp', 0)
        gained_exp = item.get('gained_exp', 0)
        if rad_name == 'Model':
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_a.append(f'Model ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_a.append(f'Model ({int(years_exp)}yrs)')
        elif 'Radiologist #' in rad_name:
            rad_num = rad_name.split('#')[1]
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_a.append(f'R#{rad_num} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_a.append(f'R#{rad_num} ({int(years_exp)}yrs)')
        else:
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_a.append(f'R#{rad_name[:3]} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_a.append(f'R#{rad_name[:3]} ({int(years_exp)}yrs)')
    ax_1_0.set_xticklabels(xtick_labels_a, rotation=45, ha='right')

    # Create custom legend with generic Agent labels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_seg[0], alpha=0.7, label='Agent without support'),
        Patch(facecolor=colors_seg[1], alpha=0.7, label='Agent with support')
    ]
    ax_1_0.legend(handles=legend_elements, loc='upper right')
    ax_1_0.grid(True, alpha=0.3)
    ax_1_0.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Panel b (row 1, col 1): Calibration difference by radiologist
ax_1_1 = fig6.add_subplot(gs[1, 1])

if True:
    # Prepare data for panel b
    panel_data = []
    for rad in radiologists:
        without_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                            (confidence_analysis_df['with_segmentation'] == False)]
        with_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                         (confidence_analysis_df['with_segmentation'] == True)]

        # Get gained experience from equiv_df
        gained_exp = 0
        rad_equiv_data = equiv_df[equiv_df['radiologist'] == rad]
        if len(rad_equiv_data) > 0:
            gained_exp = rad_equiv_data['avg_equiv_years'].iloc[0]

        panel_data.append({
            'radiologist': rad,
            'years_exp': radiologist_exp_dict.get(rad, 0),
            'gained_exp': gained_exp,
            'without_calib': without_data['calibration_diff'].iloc[0] if len(without_data) > 0 else 0,
            'with_calib': with_data['calibration_diff'].iloc[0] if len(with_data) > 0 else 0,
        })

    # Add model to panel data if metrics are available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if 'calibration_diff' in model_conf_metrics['without'] and 'calibration_diff' in model_conf_metrics['with']:
            panel_data.append({
                'radiologist': 'Model',
                'years_exp': model_equiv_years_without if model_equiv_years_without is not None else 0,
                'gained_exp': model_equiv_years_gained if model_equiv_years_gained is not None else 0,
                'without_calib': model_conf_metrics['without']['calibration_diff'],
                'with_calib': model_conf_metrics['with']['calibration_diff'],
            })

    # Sort by with_calib (largest to smallest)
    panel_data_b = sorted(panel_data, key=lambda x: x['with_calib'], reverse=True)
    radiologists_b = [item['radiologist'] for item in panel_data_b]
    without_calib = [item['without_calib'] for item in panel_data_b]
    with_calib = [item['with_calib'] for item in panel_data_b]
    
    x_pos_b = np.arange(len(radiologists_b))

    # Plot bars with texture for Model
    for i, rad in enumerate(radiologists_b):
        is_model = rad == 'Model'
        hatch_pattern = '//' if is_model else None

        ax_1_1.bar(x_pos_b[i] - width/2, without_calib[i], width,
                  color=colors_seg[0], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
        ax_1_1.bar(x_pos_b[i] + width/2, with_calib[i], width,
                  color=colors_seg[1], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
    
    ax_1_1.set_xlabel('Agent')
    ax_1_1.set_ylabel('Calibration difference\n(High conf - low conf accuracy)\n[Higher = better calibrated]')
    ax_1_1.set_title('f) Confidence calibration')
    ax_1_1.set_xticks(x_pos_b)

    xtick_labels_b = []
    for item in panel_data_b:
        rad_name = item['radiologist']
        years_exp = item.get('years_exp', 0)
        gained_exp = item.get('gained_exp', 0)
        if rad_name == 'Model':
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_b.append(f'Model ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_b.append(f'Model ({int(years_exp)}yrs)')
        elif 'Radiologist #' in rad_name:
            rad_num = rad_name.split('#')[1]
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_b.append(f'R#{rad_num} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_b.append(f'R#{rad_num} ({int(years_exp)}yrs)')
        else:
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_b.append(f'R#{rad_name[:3]} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_b.append(f'R#{rad_name[:3]} ({int(years_exp)}yrs)')
    ax_1_1.set_xticklabels(xtick_labels_b, rotation=45, ha='right')

    # Create custom legend with generic Agent labels
    legend_elements = [
        Patch(facecolor=colors_seg[0], alpha=0.7, label='Agent without support'),
        Patch(facecolor=colors_seg[1], alpha=0.7, label='Agent with support')
    ]
    ax_1_1.legend(handles=legend_elements, loc='upper right')
    ax_1_1.grid(True, alpha=0.3)
    ax_1_1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Panel c (row 1, col 2): Confidence bias by radiologist
ax_1_2 = fig6.add_subplot(gs[1, 2])

if True:
    # Prepare data for panel c
    panel_data = []
    for rad in radiologists:
        without_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                            (confidence_analysis_df['with_segmentation'] == False)]
        with_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) &
                                         (confidence_analysis_df['with_segmentation'] == True)]

        # Get gained experience from equiv_df
        gained_exp = 0
        rad_equiv_data = equiv_df[equiv_df['radiologist'] == rad]
        if len(rad_equiv_data) > 0:
            gained_exp = rad_equiv_data['avg_equiv_years'].iloc[0]

        panel_data.append({
            'radiologist': rad,
            'years_exp': radiologist_exp_dict.get(rad, 0),
            'gained_exp': gained_exp,
            'without_bias': without_data['confidence_bias'].iloc[0] if len(without_data) > 0 else 0,
            'with_bias': with_data['confidence_bias'].iloc[0] if len(with_data) > 0 else 0,
        })

    # Add model to panel data if metrics are available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if 'confidence_bias' in model_conf_metrics['without'] and 'confidence_bias' in model_conf_metrics['with']:
            panel_data.append({
                'radiologist': 'Model',
                'years_exp': model_equiv_years_without if model_equiv_years_without is not None else 0,
                'gained_exp': model_equiv_years_gained if model_equiv_years_gained is not None else 0,
                'without_bias': model_conf_metrics['without']['confidence_bias'],
                'with_bias': model_conf_metrics['with']['confidence_bias'],
            })

    # Sort by with_bias (largest to smallest)
    panel_data_c = sorted(panel_data, key=lambda x: x['with_bias'], reverse=True)
    radiologists_c = [item['radiologist'] for item in panel_data_c]
    without_bias = [item['without_bias'] for item in panel_data_c]
    with_bias = [item['with_bias'] for item in panel_data_c]
    
    x_pos_c = np.arange(len(radiologists_c))

    # Plot bars with texture for Model
    for i, rad in enumerate(radiologists_c):
        is_model = rad == 'Model'
        hatch_pattern = '//' if is_model else None

        ax_1_2.bar(x_pos_c[i] - width/2, without_bias[i], width,
                  color=colors_seg[0], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
        ax_1_2.bar(x_pos_c[i] + width/2, with_bias[i], width,
                  color=colors_seg[1], alpha=0.7, hatch=hatch_pattern,
                  edgecolor='black', linewidth=0.5)
    
    ax_1_2.set_xlabel('Agent')
    ax_1_2.set_ylabel('Confidence bias\n(Correct - incorrect confidence)\n[Higher = better calibrated]')
    ax_1_2.set_title('g) Confidence bias')
    ax_1_2.set_xticks(x_pos_c)

    xtick_labels_c = []
    for item in panel_data_c:
        rad_name = item['radiologist']
        years_exp = item.get('years_exp', 0)
        gained_exp = item.get('gained_exp', 0)
        if rad_name == 'Model':
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_c.append(f'Model ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_c.append(f'Model ({int(years_exp)}yrs)')
        elif 'Radiologist #' in rad_name:
            rad_num = rad_name.split('#')[1]
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_c.append(f'R#{rad_num} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_c.append(f'R#{rad_num} ({int(years_exp)}yrs)')
        else:
            if gained_exp and not pd.isna(gained_exp):
                leveraged_exp = years_exp + gained_exp
                xtick_labels_c.append(f'R#{rad_name[:3]} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
            else:
                xtick_labels_c.append(f'R#{rad_name[:3]} ({int(years_exp)}yrs)')
    ax_1_2.set_xticklabels(xtick_labels_c, rotation=45, ha='right')

    # Create custom legend with generic Agent labels
    legend_elements = [
        Patch(facecolor=colors_seg[0], alpha=0.7, label='Agent without support'),
        Patch(facecolor=colors_seg[1], alpha=0.7, label='Agent with support')
    ]
    ax_1_2.legend(handles=legend_elements, loc='upper right')
    ax_1_2.grid(True, alpha=0.3)
    ax_1_2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


# Panel h split into two subplots (row 2, col 0): Radiologist and Model confidence density by accuracy
# Create nested grid for panel h split (similar to panels a and b)
gs_nested_h = gs[2, 0].subgridspec(1, 2, wspace=0.5, width_ratios=[0.17, 0.17])
# Left subplot: Radiologist confidence
ax_2_0_left = fig6.add_subplot(gs_nested_h[0, 0])
# Right subplot: Model confidence
ax_2_0_right = fig6.add_subplot(gs_nested_h[0, 1])

ax_2_0 = ax_2_0_left

# Recreate panel c from confidence_quality_relationships.png
combined_data = []
for i, with_seg in enumerate([False, True]):
    condition_data = radiologist_df[radiologist_df['with_segmentation'] == with_seg]

    for accuracy in [0, 1]:
        subset = condition_data[condition_data['correct_prediction'] == accuracy]
        for conf in subset['confidence']:
            combined_data.append({
                'confidence': conf,
                'accuracy': 'Correct' if accuracy == 1 else 'Incorrect',
                'condition': 'Without support' if i == 0 else 'With support'
            })

if combined_data:
    combined_df = pd.DataFrame(combined_data)

    # Create violin plot with split violins and transparency
    # Increased bw_adjust to make violins wider relative to internal boxes
    violin_parts = sns.violinplot(data=combined_df, x='accuracy', y='confidence', hue='condition',
                   ax=ax_2_0, palette=[reference_colors[0], reference_colors[1]], split=True, gap=.1,
                   inner="box", bw_adjust=2.5)

    # Set transparency for violin patches to match panels E, F, G
    for pc in ax_2_0.collections:
        pc.set_alpha(0.7)

    ax_2_0.set_xlabel('Prediction')
    ax_2_0.set_ylabel('Confidence')
    ax_2_0.set_title('h) Radiologist confidence')
    ax_2_0.set_ylim(0, 12.5)  # Increased ylim but keep yticks up to 10
    ax_2_0.set_yticks(range(0, 11))
    ax_2_0.get_legend().remove() if ax_2_0.get_legend() else None

    # Find common cases between prob_data and best_cv_predictions (used for the
    # WITH-support confidence arm)
    cv_case_ids = set([pred['case_id'] for pred in best_cv_predictions if 'case_id' in pred])
    prob_case_ids = set(prob_data.keys())
    common_model_cases = cv_case_ids.intersection(prob_case_ids)

    # Model without-support confidence (per radiologist-case review): used by Panel I violin
    model_without_correct_conf = []
    model_without_incorrect_conf = []
    if len(prob_data) > 0:
        for _, row in radiologist_df.iterrows():
            case_id = row['case_id']
            if case_id in prob_data:
                confidence_score = abs(prob_data[case_id] - 0.5) * 2 * 10
                if row['model_predicted_enhancement'] == row['has_enhancement_gt']:
                    model_without_correct_conf.append(confidence_score)
                else:
                    model_without_incorrect_conf.append(confidence_score)

    # Model with-support confidence (from CV predictions): used by Panel I violin
    model_with_correct_conf = []
    model_with_incorrect_conf = []
    if common_model_cases:
        for pred in best_cv_predictions:
            if 'combined_prob' in pred:
                confidence_score = abs(pred['combined_prob'] - 0.5) * 2 * 10
                if pred['cv_pred'] == pred['gt']:
                    model_with_correct_conf.append(confidence_score)
                else:
                    model_with_incorrect_conf.append(confidence_score)

    # Per-arm radiologist confidence by correctness (manuscript paragraph 125)
    from scipy.stats import ttest_ind as _ttest_ind_125
    _rad_corr_wo  = radiologist_df[(radiologist_df['with_segmentation'] == False) & (radiologist_df['correct_prediction'] == 1)]['confidence']
    _rad_corr_ws  = radiologist_df[(radiologist_df['with_segmentation'] == True)  & (radiologist_df['correct_prediction'] == 1)]['confidence']
    _rad_inc_wo   = radiologist_df[(radiologist_df['with_segmentation'] == False) & (radiologist_df['correct_prediction'] == 0)]['confidence']
    _rad_inc_ws   = radiologist_df[(radiologist_df['with_segmentation'] == True)  & (radiologist_df['correct_prediction'] == 0)]['confidence']
    _t_corr_p125,  _p_corr_p125  = _ttest_ind_125(_rad_corr_wo,  _rad_corr_ws)
    _t_inc_p125,   _p_inc_p125   = _ttest_ind_125(_rad_inc_wo,   _rad_inc_ws)
    print(f"\nRadiologist confidence by correctness, per arm (paragraph 125):")
    print(f"  CORRECT   without model: {_rad_corr_wo.mean():.2f} ± {_rad_corr_wo.std():.2f}  (n={len(_rad_corr_wo)})")
    print(f"  CORRECT   with model:    {_rad_corr_ws.mean():.2f} ± {_rad_corr_ws.std():.2f}  (n={len(_rad_corr_ws)})  ttest_ind p={_p_corr_p125:.6g}")
    print(f"  INCORRECT without model: {_rad_inc_wo.mean():.2f} ± {_rad_inc_wo.std():.2f}  (n={len(_rad_inc_wo)})")
    print(f"  INCORRECT with model:    {_rad_inc_ws.mean():.2f} ± {_rad_inc_ws.std():.2f}  (n={len(_rad_inc_ws)})  ttest_ind p={_p_inc_p125:.6g}  (paragraph 125: p={_p_inc_p125:.3f}, ns)")

    # Add unpaired t-tests for each accuracy group using all observations
    from scipy import stats
    
    bar_height = 11.5  # Positioned higher to avoid overlap with violin plots
    x_positions = [0, 1]  # Incorrect, Correct
    
    for i, accuracy in enumerate([0, 1]):
        acc_label = 'Correct' if accuracy == 1 else 'Incorrect'
        
        # Get all observations for this accuracy level
        without_model_data = radiologist_df[(radiologist_df['with_segmentation'] == False) & 
                                          (radiologist_df['correct_prediction'] == accuracy)]['confidence']
        with_model_data = radiologist_df[(radiologist_df['with_segmentation'] == True) & 
                                        (radiologist_df['correct_prediction'] == accuracy)]['confidence']
        
        # Perform PAIRED t-test on same case-radiologist combinations
        if len(without_model_data) > 0 and len(with_model_data) > 0:
            # Compute an independent-sample t-test for comparison
            t_stat_ind, p_value_ind = stats.ttest_ind(without_model_data, with_model_data)

            # Calculate PAIRED t-test on same case-radiologist combinations
            paired_confidence_without = []
            paired_confidence_with = []

            # Get all unique case-radiologist combinations
            for rad in radiologist_df['radiologist'].unique():
                for case in radiologist_df['case_id'].unique():
                    # Find this case-radiologist combination in both conditions
                    without_data = radiologist_df[(radiologist_df['radiologist'] == rad) &
                                                 (radiologist_df['case_id'] == case) &
                                                 (radiologist_df['with_segmentation'] == False) &
                                                 (radiologist_df['correct_prediction'] == i)]
                    with_data = radiologist_df[(radiologist_df['radiologist'] == rad) &
                                              (radiologist_df['case_id'] == case) &
                                              (radiologist_df['with_segmentation'] == True) &
                                              (radiologist_df['correct_prediction'] == i)]

                    # Only include if both conditions exist for this case-radiologist with this correctness
                    if len(without_data) > 0 and len(with_data) > 0:
                        paired_confidence_without.append(without_data['confidence'].values[0])
                        paired_confidence_with.append(with_data['confidence'].values[0])

            # Use paired t-test p-value for significance markers
            if len(paired_confidence_without) > 0:
                t_stat, p_value = stats.ttest_rel(paired_confidence_without, paired_confidence_with)
            else:
                # Fallback to independent t-test if no paired data available
                t_stat, p_value = t_stat_ind, p_value_ind

            # Echo computed p-values
            if len(paired_confidence_without) > 0:
                print(f"  Paired t-test (n={len(paired_confidence_without)} pairs): T={t_stat:.3f}, p={p_value:.6f}")
                print(f"  Using PAIRED t-test p-value for significance markers")

            # Add significance annotation bar for this accuracy group
            x_pos = x_positions[i]
            ax_2_0.plot([x_pos - 0.15, x_pos + 0.15], [bar_height, bar_height], 'k-', linewidth=1.5)

            # Add significance stars based on PAIRED t-test
            if p_value < 0.0001:
                sig_text = '****'
            elif p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            ax_2_0.text(x_pos, bar_height + 0.1, sig_text, ha='center', va='bottom', fontsize=12)
    ax_2_0.grid(True, alpha=0.3)
    
    # Create custom legend with matching transparency
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=reference_colors[0], alpha=0.5, label='Without support'),
        Patch(facecolor=reference_colors[1], alpha=0.5, label='With support')
    ]

    ax_2_0_right.set_title('i) Model confidence')
    ax_2_0_right.set_xlabel('Prediction')
    ax_2_0_right.set_ylabel('Confidence')
    ax_2_0_right.set_ylim(0, 12.5)  # Same as radiologist plot
    ax_2_0_right.set_yticks(range(0, 11))  # Same as radiologist plot
    ax_2_0_right.set_xticks([0, 1])
    ax_2_0_right.set_xticklabels(['Incorrect', 'Correct'])  # No rotation

    # Prepare model confidence data for strip plot
    model_strip_data = []

    # Add small jitter to zero-variance data for visualization
    # Note: Model without support consistently outputs confidence ~10, so we add
    # minimal jitter (σ=0.15) purely for visualization to make violin plots visible
    np.random.seed(42)  # For reproducibility
    jitter_amount = 0.15  # Small jitter for visualization only

    # Add model without support data
    if model_without_correct_conf:
        for conf in model_without_correct_conf:
            # Add small jitter to each value for visualization
            jittered_conf = conf + np.random.normal(0, jitter_amount)
            # Ensure jitter doesn't push values outside reasonable bounds
            jittered_conf = max(conf - 0.3, min(conf + 0.3, jittered_conf))
            model_strip_data.append({
                'confidence': jittered_conf,
                'accuracy': 'Correct',
                'condition': 'Without support'
            })

    if model_without_incorrect_conf:
        for conf in model_without_incorrect_conf:
            # Add small jitter to each value for visualization
            jittered_conf = conf + np.random.normal(0, jitter_amount)
            # Ensure jitter doesn't push values outside reasonable bounds
            jittered_conf = max(conf - 0.3, min(conf + 0.3, jittered_conf))
            model_strip_data.append({
                'confidence': jittered_conf,
                'accuracy': 'Incorrect',
                'condition': 'Without support'
            })

    # Add model with support data
    if model_with_correct_conf:
        for conf in model_with_correct_conf:
            model_strip_data.append({
                'confidence': conf,
                'accuracy': 'Correct',
                'condition': 'With support'
            })

    if model_with_incorrect_conf:
        for conf in model_with_incorrect_conf:
            model_strip_data.append({
                'confidence': conf,
                'accuracy': 'Incorrect',
                'condition': 'With support'
            })

    if model_strip_data:
        model_strip_df = pd.DataFrame(model_strip_data)

        # Create violin plot matching the radiologist plot style
        # Explicitly set the order to match what we expect
        violin_parts = sns.violinplot(data=model_strip_df, x='accuracy', y='confidence', hue='condition',
                       ax=ax_2_0_right, palette=[reference_colors[0], reference_colors[1]],
                       split=True, gap=.1,
                       inner="box",  # Box plot inside violin
                       bw_adjust=2.5,  # Increased to make violins wider relative to boxes
                       order=['Incorrect', 'Correct'],  # Explicit order
                       hue_order=['Without support', 'With support'])  # Explicit hue order

        # Set transparency for violin patches to match panels E, F, G
        for pc in ax_2_0_right.collections:
            pc.set_alpha(0.7)

        # Note: Jitter has been added to zero-variance data for visualization

        # Add grid for consistency
        ax_2_0_right.grid(True, alpha=0.3)

        print("\nModel confidence by correctness (paragraph 127, panel i):")

        # Perform t-tests between with/without support for each accuracy level
        bar_height = 11.5  # Same height as panel h
        x_positions = [0, 1]  # Incorrect, Correct

        for i, acc_label in enumerate(['Incorrect', 'Correct']):
            # Use ORIGINAL (non-jittered) confidence values for statistical testing
            if acc_label == 'Incorrect':
                without_support_conf = np.array(model_without_incorrect_conf)
                with_support_conf = np.array(model_with_incorrect_conf)
            else:  # Correct
                without_support_conf = np.array(model_without_correct_conf)
                with_support_conf = np.array(model_with_correct_conf)

            # Perform unpaired t-test if we have enough data
            if len(without_support_conf) > 0 and len(with_support_conf) > 0:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(without_support_conf, with_support_conf)

                # Echo computed p-values
                print(f"\nModel {acc_label} predictions:")
                print(f"  Number of cases without support: {len(without_support_conf)}")
                print(f"  Number of cases with support: {len(with_support_conf)}")
                # 3-dp for precision; 2-dp matches the manuscript paragraph 127 precision.
                print(f"  Without support mean: {without_support_conf.mean():.3f} ± {without_support_conf.std():.3f}  (paragraph 127: {without_support_conf.mean():.2f} ± {without_support_conf.std():.2f})")
                print(f"  With support mean: {with_support_conf.mean():.3f} ± {with_support_conf.std():.3f}  (paragraph 127: {with_support_conf.mean():.2f} ± {with_support_conf.std():.2f})")
                print(f"  T-statistic: {t_stat:.3f}")
                print(f"  P-value: {p_value:.6f}")

                # Add significance annotation bar for this accuracy group
                x_pos = x_positions[i]
                ax_2_0_right.plot([x_pos - 0.15, x_pos + 0.15], [bar_height, bar_height], 'k-', linewidth=1.5)

                # Add significance stars using same nomenclature as panel h
                if p_value < 0.0001:
                    sig_text = '****'
                elif p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                elif p_value < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'

                ax_2_0_right.text(x_pos, bar_height + 0.1, sig_text, ha='center', va='bottom', fontsize=12)

        # Reposition the legend from panel i to be centered between h and i
        legend = ax_2_0_right.get_legend()
        if legend:
            # Remove the auto-generated legend from right panel
            legend.remove()

        # Create a shared legend positioned between panels h and i
        from matplotlib.patches import Patch
        shared_legend_elements = [
            Patch(facecolor=reference_colors[0], alpha=0.5, label='Without support'),
            Patch(facecolor=reference_colors[1], alpha=0.5, label='With support')
        ]
        # Position centered between the two panels
        fig6.legend(handles=shared_legend_elements, title='Condition',
                   loc='lower center', bbox_to_anchor=(0.23, 0.05),
                   ncol=2, frameon=True)

# Panel e (row 2, col 1): Self-awareness vs calibration
ax_2_1 = fig6.add_subplot(gs[2, 1])

if True:
    for i, with_seg in enumerate([False, True]):
        condition_data = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == with_seg]
        
        # Uniform marker size for all points
        ax_2_1.scatter(condition_data['conf_acc_corr'], condition_data['calibration_diff'],
                      alpha=0.7, s=100, color=colors_seg[i], label=['Without support', 'With support'][i], zorder=2)

        # Remove radiologist labels (annotations) as requested

    # Calculate mean radiologist values for each condition
    mean_rad_without = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == False][['conf_acc_corr', 'calibration_diff']].mean()
    mean_rad_with = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == True][['conf_acc_corr', 'calibration_diff']].mean()

    # Add mean radiologist points
    ax_2_1.scatter(mean_rad_without['conf_acc_corr'], mean_rad_without['calibration_diff'],
                  marker='o', s=150, alpha=0.9, color=colors_seg[0],
                  edgecolors='black', linewidth=0.5, zorder=5)
    ax_2_1.text(mean_rad_without['conf_acc_corr'], mean_rad_without['calibration_diff'],
               'R', ha='center', va='center', fontsize=10, color='black', zorder=6)

    ax_2_1.scatter(mean_rad_with['conf_acc_corr'], mean_rad_with['calibration_diff'],
                  marker='o', s=150, alpha=0.9, color=colors_seg[1],
                  edgecolors='black', linewidth=0.5, zorder=5)
    ax_2_1.text(mean_rad_with['conf_acc_corr'], mean_rad_with['calibration_diff'],
               'R', ha='center', va='center', fontsize=10, color='black', zorder=6)

    # Add arrow connecting R points - offset endpoint to avoid obscuring arrowhead
    # Calculate direction vector and shorten arrow to point to edge of scatter point
    dx_r = mean_rad_with['conf_acc_corr'] - mean_rad_without['conf_acc_corr']
    dy_r = mean_rad_with['calibration_diff'] - mean_rad_without['calibration_diff']
    arrow_length_r = np.sqrt(dx_r**2 + dy_r**2)
    # Offset by roughly the radius of the scatter point (in data units)
    offset_factor = 0.02  # Adjust this value to fine-tune arrow endpoint
    end_x_r = mean_rad_with['conf_acc_corr'] - (dx_r/arrow_length_r) * offset_factor if arrow_length_r > 0 else mean_rad_with['conf_acc_corr']
    end_y_r = mean_rad_with['calibration_diff'] - (dy_r/arrow_length_r) * offset_factor if arrow_length_r > 0 else mean_rad_with['calibration_diff']

    ax_2_1.annotate('', xy=(end_x_r, end_y_r),
                   xytext=(mean_rad_without['conf_acc_corr'], mean_rad_without['calibration_diff']),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.7))

    # Add model as scatter points if metrics are available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if all(k in model_conf_metrics['without'] for k in ['conf_acc_corr', 'calibration_diff']) and \
           all(k in model_conf_metrics['with'] for k in ['conf_acc_corr', 'calibration_diff']):
            # Plot model without support
            ax_2_1.scatter(model_conf_metrics['without']['conf_acc_corr'],
                          model_conf_metrics['without']['calibration_diff'],
                          marker='o', s=150, alpha=0.9, color=colors_seg[0],
                          edgecolors='black', linewidth=0.5, zorder=5)
            ax_2_1.text(model_conf_metrics['without']['conf_acc_corr'],
                       model_conf_metrics['without']['calibration_diff'],
                       'M', ha='center', va='center', fontsize=10, color='black', zorder=6)

            # Plot model with support
            ax_2_1.scatter(model_conf_metrics['with']['conf_acc_corr'],
                          model_conf_metrics['with']['calibration_diff'],
                          marker='o', s=150, alpha=0.9, color=colors_seg[1],
                          edgecolors='black', linewidth=0.5, zorder=5)
            ax_2_1.text(model_conf_metrics['with']['conf_acc_corr'],
                       model_conf_metrics['with']['calibration_diff'],
                       'M', ha='center', va='center', fontsize=10, color='black', zorder=6)

            # Add arrow connecting M points - offset endpoint to avoid obscuring arrowhead
            dx_m = model_conf_metrics['with']['conf_acc_corr'] - model_conf_metrics['without']['conf_acc_corr']
            dy_m = model_conf_metrics['with']['calibration_diff'] - model_conf_metrics['without']['calibration_diff']
            arrow_length_m = np.sqrt(dx_m**2 + dy_m**2)
            # Same offset as R arrow
            end_x_m = model_conf_metrics['with']['conf_acc_corr'] - (dx_m/arrow_length_m) * offset_factor if arrow_length_m > 0 else model_conf_metrics['with']['conf_acc_corr']
            end_y_m = model_conf_metrics['with']['calibration_diff'] - (dy_m/arrow_length_m) * offset_factor if arrow_length_m > 0 else model_conf_metrics['with']['calibration_diff']

            ax_2_1.annotate('', xy=(end_x_m, end_y_m),
                           xytext=(model_conf_metrics['without']['conf_acc_corr'],
                                  model_conf_metrics['without']['calibration_diff']),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.7))

    # Quadrant boundaries: 50th percentile (median) split (used for Panel j)
    median_threshold_corr = confidence_analysis_df['conf_acc_corr'].quantile(0.5)
    median_threshold_calib = confidence_analysis_df['calibration_diff'].quantile(0.5)

    # Draw 50th percentile (median)-based quadrant lines (behind scatter points but above shading)
    ax_2_1.axhline(y=median_threshold_calib, color='black', linestyle='-', alpha=0.8, linewidth=1, zorder=1)
    ax_2_1.axvline(x=median_threshold_corr, color='black', linestyle='-', alpha=0.8, linewidth=1, zorder=1)
    
    ax_2_1.set_xlabel('Confidence-accuracy correlation (Self-awareness)')
    ax_2_1.set_ylabel('Calibration difference')
    ax_2_1.set_title('j) Agent self-awareness heuristic')
    ax_2_1.grid(True, alpha=0.3)
    
    # Calculate actual data ranges and set axis limits
    x_data_min = confidence_analysis_df['conf_acc_corr'].min()
    x_data_max = confidence_analysis_df['conf_acc_corr'].max()
    y_data_min = confidence_analysis_df['calibration_diff'].min()
    y_data_max = confidence_analysis_df['calibration_diff'].max()

    # Include model points in axis limits if available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if 'conf_acc_corr' in model_conf_metrics['without']:
            x_data_min = min(x_data_min, model_conf_metrics['without']['conf_acc_corr'])
            x_data_max = max(x_data_max, model_conf_metrics['without']['conf_acc_corr'])
        if 'conf_acc_corr' in model_conf_metrics['with']:
            x_data_min = min(x_data_min, model_conf_metrics['with']['conf_acc_corr'])
            x_data_max = max(x_data_max, model_conf_metrics['with']['conf_acc_corr'])
        if 'calibration_diff' in model_conf_metrics['without']:
            y_data_min = min(y_data_min, model_conf_metrics['without']['calibration_diff'])
            y_data_max = max(y_data_max, model_conf_metrics['without']['calibration_diff'])
        if 'calibration_diff' in model_conf_metrics['with']:
            y_data_min = min(y_data_min, model_conf_metrics['with']['calibration_diff'])
            y_data_max = max(y_data_max, model_conf_metrics['with']['calibration_diff'])
    
    x_padding = (x_data_max - x_data_min) * 0.1
    y_padding = (y_data_max - y_data_min) * 0.1
    
    ax_2_1.set_xlim(x_data_min - x_padding, x_data_max + x_padding)
    ax_2_1.set_ylim(y_data_min - y_padding, y_data_max + y_padding)
    
    # Shade quadrants with colors
    x_range = ax_2_1.get_xlim()
    y_range = ax_2_1.get_ylim()
    
    # Define quadrant colors based on the original text box colors
    quadrant_colors = {
        'top_left': reference_colors[4],     # Purple - High calibration, Low self-awareness
        'top_right': reference_colors[2],    # Green - High calibration, High self-awareness (IDEAL)
        'bottom_left': reference_colors[3],  # Red - Low calibration, Low self-awareness
        'bottom_right': reference_colors[0]  # Blue - Low calibration, High self-awareness
    }
    
    # Top left quadrant
    ax_2_1.fill_between([x_range[0], median_threshold_corr], [median_threshold_calib, median_threshold_calib], [y_range[1], y_range[1]],
                       color=quadrant_colors['top_left'], alpha=0.15, zorder=0)

    # Top right quadrant
    ax_2_1.fill_between([median_threshold_corr, x_range[1]], [median_threshold_calib, median_threshold_calib], [y_range[1], y_range[1]],
                       color=quadrant_colors['top_right'], alpha=0.15, zorder=0)

    # Bottom left quadrant
    ax_2_1.fill_between([x_range[0], median_threshold_corr], [y_range[0], y_range[0]], [median_threshold_calib, median_threshold_calib],
                       color=quadrant_colors['bottom_left'], alpha=0.15, zorder=0)

    # Bottom right quadrant
    ax_2_1.fill_between([median_threshold_corr, x_range[1]], [y_range[0], y_range[0]], [median_threshold_calib, median_threshold_calib],
                       color=quadrant_colors['bottom_right'], alpha=0.15, zorder=0)
    
    # Create legend for quadrant shading
    from matplotlib.patches import Patch
    quadrant_legend_elements = [
        Patch(facecolor=quadrant_colors['top_right'], alpha=0.3, label='↑ Calibration\n↑ Self-awareness (ideal)'),
        Patch(facecolor=quadrant_colors['top_left'], alpha=0.3, label='↑ Calibration\n↓ Self-awareness'),
        Patch(facecolor=quadrant_colors['bottom_right'], alpha=0.3, label='↓ Calibration\n↑ Self-awareness'),
        Patch(facecolor=quadrant_colors['bottom_left'], alpha=0.3, label='↓ Calibration\n↓ Self-awareness')
    ]
    
    ax_2_1.legend(handles=quadrant_legend_elements, loc='lower right', fontsize=9, frameon=True, title='Quadrant regions')
    
    # Calculate percentage optimally calibrated (in top right quadrant)
    # Include individual radiologists and model, but exclude 'R' mean point
    without_model_data = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == False]
    with_model_data = confidence_analysis_df[confidence_analysis_df['with_segmentation'] == True]

    # Count radiologist points in top right quadrant (high calibration, high self-awareness)
    # Use 50th percentile (median) thresholds to match the visual plot lines
    without_optimal = ((without_model_data['conf_acc_corr'] >= median_threshold_corr) &
                      (without_model_data['calibration_diff'] >= median_threshold_calib)).sum()
    with_optimal = ((with_model_data['conf_acc_corr'] >= median_threshold_corr) &
                   (with_model_data['calibration_diff'] >= median_threshold_calib)).sum()

    # Add model to the counts if available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        # Check if model WITHOUT support is in optimal quadrant
        if (model_conf_metrics['without']['conf_acc_corr'] >= median_threshold_corr and
            model_conf_metrics['without']['calibration_diff'] >= median_threshold_calib):
            without_optimal += 1

        # Check if model WITH support is in optimal quadrant
        if (model_conf_metrics['with']['conf_acc_corr'] >= median_threshold_corr and
            model_conf_metrics['with']['calibration_diff'] >= median_threshold_calib):
            with_optimal += 1

        # Adjust totals to include model (but not 'R' mean point)
        without_total = len(without_model_data) + 1  # +1 for model
        with_total = len(with_model_data) + 1  # +1 for model
    else:
        without_total = len(without_model_data)
        with_total = len(with_model_data)

    without_pct = (without_optimal / without_total * 100) if without_total > 0 else 0
    with_pct = (with_optimal / with_total * 100) if with_total > 0 else 0

    print(f"\nOptimal-quadrant thresholds (median split — paragraph 127):")
    print(f"  X-axis (conf_acc_corr):    {median_threshold_corr:.3f}")
    print(f"  Y-axis (calibration_diff): {median_threshold_calib:.3f}")
    print(f"\nOptimal-quadrant counts:")
    print(f"  Without support: {without_optimal}/{without_total} ({without_pct:.1f}%)")
    print(f"  With support:    {with_optimal}/{with_total} ({with_pct:.1f}%)")
    print(f"  Agents moving into optimal: {with_optimal - without_optimal}")

    # Additional quadrant-analysis output
    print(f"Quadrant placement (including Model, excluding R mean):")
    print(f"  Model WITHOUT in optimal quadrant: {model_conf_metrics['without']['conf_acc_corr'] >= median_threshold_corr and model_conf_metrics['without']['calibration_diff'] >= median_threshold_calib if model_conf_metrics.get('without') else 'N/A'}")
    print(f"  Model WITH in optimal quadrant: {model_conf_metrics['with']['conf_acc_corr'] >= median_threshold_corr and model_conf_metrics['with']['calibration_diff'] >= median_threshold_calib if model_conf_metrics.get('with') else 'N/A'}")

    # Analyze why model might not be in ideal quadrant
    if model_conf_metrics.get('with'):
        print(f"\n  Model WITH support quadrant analysis:")
        print(f"    X-axis (conf_acc_corr): {model_conf_metrics['with']['conf_acc_corr']:.3f} {'≥' if model_conf_metrics['with']['conf_acc_corr'] >= median_threshold_corr else '<'} 50th percentile/median ({median_threshold_corr:.3f})")
        print(f"    Y-axis (calibration_diff): {model_conf_metrics['with']['calibration_diff']:.3f} {'≥' if model_conf_metrics['with']['calibration_diff'] >= median_threshold_calib else '<'} 50th percentile/median ({median_threshold_calib:.3f})")

        if model_conf_metrics['with']['conf_acc_corr'] >= median_threshold_corr and model_conf_metrics['with']['calibration_diff'] < median_threshold_calib:
            print(f"    Model is in BOTTOM RIGHT quadrant (Higher self-awareness, Lower calibration)")
            print(f"    Model knows when it is right/wrong but confidence differences are small")
        elif model_conf_metrics['with']['conf_acc_corr'] < median_threshold_corr and model_conf_metrics['with']['calibration_diff'] >= median_threshold_calib:
            print(f"    Model is in TOP LEFT quadrant (Lower self-awareness, Higher calibration)")
        elif model_conf_metrics['with']['conf_acc_corr'] < median_threshold_corr and model_conf_metrics['with']['calibration_diff'] < median_threshold_calib:
            print(f"    Model is in BOTTOM LEFT quadrant (Lower self-awareness, Lower calibration)")
        else:
            print(f"    Model is in TOP RIGHT quadrant (IDEAL: Higher self-awareness, Higher calibration)")
    
    # Run Fisher's exact test
    from scipy.stats import fisher_exact
    # Create 2x2 contingency table
    # Rows: without model, with model
    # Columns: not in optimal quadrant, in optimal quadrant
    contingency_table = np.array([
        [without_total - without_optimal, without_optimal],
        [with_total - with_optimal, with_optimal]
    ])
    
    odds_ratio, fisher_p = fisher_exact(contingency_table)

    # 95% Wald CI on log-OR for the sample OR returned by fisher_exact
    # (paired with the Fisher exact p-value; common reporting convention).
    # SE(log OR) = sqrt(1/a + 1/b + 1/c + 1/d); requires every cell > 0.
    _cells = contingency_table.flatten()
    if (_cells > 0).all() and odds_ratio > 0:
        _se_log_or = float(np.sqrt(np.sum(1.0 / _cells)))
        _log_or = float(np.log(odds_ratio))
        _ci_lo = float(np.exp(_log_or - 1.96 * _se_log_or))
        _ci_hi = float(np.exp(_log_or + 1.96 * _se_log_or))
        _ci_str = f", 95% CI [{_ci_lo:.2f}, {_ci_hi:.2f}]"
    else:
        _ci_str = "  (CI undefined — zero cell in contingency table)"

    print(f"\nFisher's exact test — panel h optimal-calibration quadrant (paragraph 127):")
    print(f"  Without model: {without_optimal}/{without_total} ({without_pct:.1f}%) in optimal quadrant")
    print(f"  With model:    {with_optimal}/{with_total} ({with_pct:.1f}%) in optimal quadrant")
    print(f"  Odds ratio = {odds_ratio:.3f}{_ci_str}, p = {fisher_p:.4f}")
    
    # Add legend for scatter points with percentage optimally calibrated
    scatter_handles, scatter_labels = ax_2_1.get_legend_handles_labels()
    if scatter_handles:
        # Update labels to include percentage optimally calibrated
        updated_labels = [
            f'Without support\n({without_pct:.0f}% optimally calibrated)',
            f'With support\n({with_pct:.0f}% optimally calibrated)'
        ]
        # Filter only the scatter plot handles (first two)
        scatter_legend = ax_2_1.legend(handles=scatter_handles[:2], labels=updated_labels, 
                                      loc='upper left', frameon=True, title='Condition')
        # Add the quadrant legend back as a second legend
        ax_2_1.add_artist(scatter_legend)
        ax_2_1.legend(handles=quadrant_legend_elements, loc='lower right', fontsize=9, frameon=True, title='Quadrant regions')
    
# Panel f (row 2, col 2): Model impact on confidence calibration
ax_2_2 = fig6.add_subplot(gs[2, 2])

if True:
    improvement_data = []
    
    for rad in radiologists:
        without_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) & 
                                            (confidence_analysis_df['with_segmentation'] == False)]
        with_data = confidence_analysis_df[(confidence_analysis_df['radiologist'] == rad) & 
                                         (confidence_analysis_df['with_segmentation'] == True)]
        
        if len(without_data) > 0 and len(with_data) > 0:
            calib_improvement = with_data['calibration_diff'].iloc[0] - without_data['calibration_diff'].iloc[0]
            corr_improvement = with_data['conf_acc_corr'].iloc[0] - without_data['conf_acc_corr'].iloc[0]
            
            improvement_data.append({
                'radiologist': rad,
                'calibration_improvement': calib_improvement,
                'correlation_improvement': corr_improvement
            })
    
    # Add model improvement data if available
    if model_conf_metrics.get('without') and model_conf_metrics.get('with'):
        if 'calibration_diff' in model_conf_metrics['without'] and 'calibration_diff' in model_conf_metrics['with']:
            model_calib_improvement = model_conf_metrics['with']['calibration_diff'] - model_conf_metrics['without']['calibration_diff']
            model_corr_improvement = 0
            if 'conf_acc_corr' in model_conf_metrics['without'] and 'conf_acc_corr' in model_conf_metrics['with']:
                model_corr_improvement = model_conf_metrics['with']['conf_acc_corr'] - model_conf_metrics['without']['conf_acc_corr']

            improvement_data.append({
                'radiologist': 'Model',
                'calibration_improvement': model_calib_improvement,
                'correlation_improvement': model_corr_improvement
            })

    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)

        # Sort by calibration improvement (largest to smallest)
        improvement_df = improvement_df.sort_values('calibration_improvement', ascending=False)
        
        # Plot bars with texture for Model
        bars = []
        for i in range(len(improvement_df)):
            is_model = improvement_df.iloc[i]['radiologist'] == 'Model'
            hatch_pattern = '//' if is_model else None
            color = 'green' if improvement_df.iloc[i]['calibration_improvement'] > 0 else 'red'
            bar = ax_2_2.bar(i, improvement_df.iloc[i]['calibration_improvement'],
                            alpha=0.7, color=color, hatch=hatch_pattern,
                            edgecolor='black', linewidth=0.5)
            bars.append(bar[0])
        
        ax_2_2.set_xlabel('Agent')
        ax_2_2.set_ylabel('Calibration improvement\n(With support - without support)\n[>0 = better with support]')
        ax_2_2.set_title('k) Impact of support on calibration')
        ax_2_2.set_xticks(range(len(improvement_df)))

        # Add years of experience to x-tick labels (baseline->leveraged)
        xtick_labels_with_exp = []
        for rad in improvement_df['radiologist']:
            if rad == 'Model':
                years_exp = model_equiv_years_without if model_equiv_years_without is not None else 0
                gained_exp = model_equiv_years_gained if model_equiv_years_gained is not None else 0
                if gained_exp and not pd.isna(gained_exp):
                    leveraged_exp = years_exp + gained_exp
                    xtick_labels_with_exp.append(f'Model ({int(years_exp)}->{int(leveraged_exp)}yrs)')
                else:
                    xtick_labels_with_exp.append(f'Model ({int(years_exp)}yrs)')
            else:
                years_exp = radiologist_exp_dict.get(rad, 0)
                # Get gained experience from equiv_df
                gained_exp = 0
                rad_equiv_data = equiv_df[equiv_df['radiologist'] == rad]
                if len(rad_equiv_data) > 0:
                    gained_exp = rad_equiv_data['avg_equiv_years'].iloc[0]

                if 'Radiologist #' in rad:
                    rad_num = rad.split('#')[1]
                    if gained_exp and not pd.isna(gained_exp):
                        leveraged_exp = years_exp + gained_exp
                        xtick_labels_with_exp.append(f'R#{rad_num} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
                    else:
                        xtick_labels_with_exp.append(f'R#{rad_num} ({int(years_exp)}yrs)')
                else:
                    if gained_exp and not pd.isna(gained_exp):
                        leveraged_exp = years_exp + gained_exp
                        xtick_labels_with_exp.append(f'R#{rad[:3]} ({int(years_exp)}->{int(leveraged_exp)}yrs)')
                    else:
                        xtick_labels_with_exp.append(f'R#{rad[:3]} ({int(years_exp)}yrs)')

        ax_2_2.set_xticklabels(xtick_labels_with_exp, rotation=45, ha='right')
        ax_2_2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Set y-axis to start at 0 and set top to 0.75 as requested
        y_min, y_max = ax_2_2.get_ylim()
        ax_2_2.set_ylim(bottom=min(0, y_min), top=0.7)
        ax_2_2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            y_position = height + 0.01 if height > 0 else 0.01
            ax_2_2.text(bar.get_x() + bar.get_width()/2., y_position,
                       f'{height:.2f}', ha='center', va='bottom')

# Save Figure_6
fig6_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_6.png')
fig6_svg_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_6.svg')
plt.savefig(fig6_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig6_svg_path, format='svg', bbox_inches='tight', facecolor='white')
print(f"Fig_6 saved to: {fig6_path}")

print(f"\nFigure 6 saved to {FIGURES_OUTPUT_PATH}/Fig_6.png")
