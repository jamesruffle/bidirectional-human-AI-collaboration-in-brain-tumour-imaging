#!/usr/bin/env python3
"""Fig 5: strengthening the relationship between accuracy, experience, and confidence.

Self-contained reproduction script. Reads two minimal CSV inputs
(experience_df 22x6 + per-confidence-bin aggregate 20x5) and renders
Fig_5.png + Fig_5.svg.

Usage: python3 fig_5.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_palette("husl")

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RDF_PATH = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2', 'radiologist_df.csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')

if not os.path.isfile(RDF_PATH):
    print(f"ERROR: missing input {RDF_PATH}")
    sys.exit(2)

os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

# Both panel inputs are computed live from radiologist_df.csv.
print(f"Computing fig_5 panel inputs live from {RDF_PATH}...")
_rdf_full = pd.read_csv(RDF_PATH, float_precision='round_trip')
_rdf_full['with_segmentation'] = _rdf_full['with_segmentation'].astype(bool)

experience_df = (
    _rdf_full
    .groupby(['radiologist', 'with_segmentation'])
    .agg(years_experience=('years_experience', 'first'),
         rad_accuracy=('correct_prediction', 'mean'),
         mean_confidence=('confidence', 'mean'),
         mean_response_time=('response_time', 'mean'))
    .reset_index()
)

bins_df = (
    _rdf_full
    .groupby(['with_segmentation', 'confidence'])
    .agg(accuracy=('correct_prediction', 'mean'),
         mean_response_time=('response_time', 'mean'),
         n=('case_id', 'count'))
    .reset_index()
)
print(f"  experience_summary: {len(experience_df)} rows (live-derived)")
print(f"  calibration_bins:   {len(bins_df)} rows (live-derived)")


fig4 = plt.figure(figsize=(16, 18))

# Add overall title
fig4.suptitle('Strengthening the relationship between accuracy, experience, and confidence in radiologists', fontsize=16, y=0.92)

# Define the grid layout - 3 rows, 2 columns
gs = fig4.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig4.add_subplot(gs[0, 0])
ax2 = fig4.add_subplot(gs[0, 1])
ax3 = fig4.add_subplot(gs[1, 0])
ax4 = fig4.add_subplot(gs[1, 1])
ax5 = fig4.add_subplot(gs[2, 0])
ax6 = fig4.add_subplot(gs[2, 1])

# Define colors
reference_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors_seg = [reference_colors[0], reference_colors[1]]
condition_names = ['Without model', 'With model']

# Calculate standardized axis limits
all_rad_accuracy = experience_df['rad_accuracy']
all_confidence = experience_df['mean_confidence']

rad_acc_min = all_rad_accuracy.min() - 0.02
rad_acc_max = all_rad_accuracy.max() + 0.02
conf_min = all_confidence.min() - 0.2
conf_max = all_confidence.max() + 0.2

# Panel a) - Experience vs accuracy WITHOUT model
condition_data = experience_df[experience_df['with_segmentation'] == False]
cases_per_hour = 3600 / condition_data['mean_response_time'].fillna(30)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

ax1.scatter(condition_data['years_experience'], condition_data['rad_accuracy'],
           alpha=0.7, s=cases_per_hour_sizes, color=colors_seg[0])

# Add trend line
z = np.polyfit(condition_data['years_experience'], condition_data['rad_accuracy'], 1)
p = np.poly1d(z)
corr, p_val = stats.pearsonr(condition_data['years_experience'], condition_data['rad_accuracy'])

x_line = np.linspace(condition_data['years_experience'].min(),
                   condition_data['years_experience'].max(), 100)
p_str = 'p<0.001' if p_val <= 0.001 else f'p={p_val:.3f}'
ax1.plot(x_line, p(x_line), color=colors_seg[0], linestyle='-',
        alpha=0.8, linewidth=2, label=f'R²={corr**2:.3f}, {p_str}')

# Add radiologist labels
for _, row in condition_data.iterrows():
    rad_name = row['radiologist']
    if 'Radiologist #' in rad_name:
        rad_num = rad_name.split('#')[1]
        label = f'{rad_num}'
    else:
        label = rad_name[:3]
    ax1.text(row['years_experience'], row['rad_accuracy'], label,
            ha='center', va='center', fontsize=10, color='black')

ax1.set_ylim(rad_acc_min, rad_acc_max)
ax1.set_xlabel('Years of experience', fontsize=12)
ax1.set_ylabel('Radiologist accuracy', fontsize=12)
ax1.set_title('a) Experience vs accuracy - without model', fontsize=14)
ax1.grid(True, alpha=0.3)

# Add legends for cases per hour
cases_sizes = [120, 60, 30]  # cases per hour (3600/30, 3600/60, 3600/120)
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax1.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

handles, labels = ax1.get_legend_handles_labels()
reg_handles = [h for h, l in zip(handles, labels) if 'R²=' in l]
reg_labels = [l for l in labels if 'R²=' in l]
stats_legend = ax1.legend(reg_handles, reg_labels, loc='upper left')
ax1.add_artist(cases_legend)

# Panel b) - Experience vs accuracy WITH model
condition_data = experience_df[experience_df['with_segmentation'] == True]
cases_per_hour = 3600 / condition_data['mean_response_time'].fillna(30)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

ax2.scatter(condition_data['years_experience'], condition_data['rad_accuracy'],
           alpha=0.7, s=cases_per_hour_sizes, color=colors_seg[1])

# Add trend line
z = np.polyfit(condition_data['years_experience'], condition_data['rad_accuracy'], 1)
p = np.poly1d(z)
corr, p_val = stats.pearsonr(condition_data['years_experience'], condition_data['rad_accuracy'])

x_line = np.linspace(condition_data['years_experience'].min(),
                   condition_data['years_experience'].max(), 100)
p_str = 'p<0.001' if p_val <= 0.001 else f'p={p_val:.3f}'
ax2.plot(x_line, p(x_line), color=colors_seg[1], linestyle='-',
        alpha=0.8, linewidth=2, label=f'R²={corr**2:.3f}, {p_str}')

# Add radiologist labels
for _, row in condition_data.iterrows():
    rad_name = row['radiologist']
    if 'Radiologist #' in rad_name:
        rad_num = rad_name.split('#')[1]
        label = f'{rad_num}'
    else:
        label = rad_name[:3]
    ax2.text(row['years_experience'], row['rad_accuracy'], label,
            ha='center', va='center', fontsize=10, color='black')

ax2.set_ylim(rad_acc_min, rad_acc_max)
ax2.set_xlabel('Years of experience', fontsize=12)
ax2.set_ylabel('Radiologist accuracy', fontsize=12)
ax2.set_title('b) Experience vs accuracy - with model', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add legends for cases per hour
cases_sizes = [120, 60, 30]  # cases per hour
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax2.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

handles, labels = ax2.get_legend_handles_labels()
reg_handles = [h for h, l in zip(handles, labels) if 'R²=' in l]
reg_labels = [l for l in labels if 'R²=' in l]
stats_legend = ax2.legend(reg_handles, reg_labels, loc='upper left')
ax2.add_artist(cases_legend)

# Panel d) - Experience vs confidence WITHOUT model
condition_data = experience_df[experience_df['with_segmentation'] == False]
cases_per_hour = 3600 / condition_data['mean_response_time'].fillna(30)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

ax3.scatter(condition_data['years_experience'], condition_data['mean_confidence'],
           alpha=0.7, s=cases_per_hour_sizes, color=colors_seg[0])

# Add trend line
z = np.polyfit(condition_data['years_experience'], condition_data['mean_confidence'], 1)
p = np.poly1d(z)
corr, p_val = stats.pearsonr(condition_data['years_experience'], condition_data['mean_confidence'])

x_line = np.linspace(condition_data['years_experience'].min(),
                   condition_data['years_experience'].max(), 100)
p_str = 'p<0.001' if p_val <= 0.001 else f'p={p_val:.3f}'
ax3.plot(x_line, p(x_line), color=colors_seg[0], linestyle='-',
        alpha=0.8, linewidth=2, label=f'R²={corr**2:.3f}, {p_str}')

# Add radiologist labels
for _, row in condition_data.iterrows():
    rad_name = row['radiologist']
    if 'Radiologist #' in rad_name:
        rad_num = rad_name.split('#')[1]
        label = f'{rad_num}'
    else:
        label = rad_name[:3]
    ax3.text(row['years_experience'], row['mean_confidence'], label,
            ha='center', va='center', fontsize=10, color='black')

ax3.set_ylim(conf_min, conf_max)
ax3.set_xlabel('Years of experience', fontsize=12)
ax3.set_ylabel('Mean confidence', fontsize=12)
ax3.set_title('c) Experience vs confidence - without model', fontsize=14)
ax3.grid(True, alpha=0.3)

# Add legends for cases per hour
cases_sizes = [120, 60, 30]  # cases per hour
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax3.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

handles, labels = ax3.get_legend_handles_labels()
reg_handles = [h for h, l in zip(handles, labels) if 'R²=' in l]
reg_labels = [l for l in labels if 'R²=' in l]
stats_legend = ax3.legend(reg_handles, reg_labels, loc='upper left')
ax3.add_artist(cases_legend)

# Panel e) - Experience vs confidence WITH model
condition_data = experience_df[experience_df['with_segmentation'] == True]
cases_per_hour = 3600 / condition_data['mean_response_time'].fillna(30)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

ax4.scatter(condition_data['years_experience'], condition_data['mean_confidence'],
           alpha=0.7, s=cases_per_hour_sizes, color=colors_seg[1])

# Add trend line
z = np.polyfit(condition_data['years_experience'], condition_data['mean_confidence'], 1)
p = np.poly1d(z)
corr, p_val = stats.pearsonr(condition_data['years_experience'], condition_data['mean_confidence'])

x_line = np.linspace(condition_data['years_experience'].min(),
                   condition_data['years_experience'].max(), 100)
p_str = 'p<0.001' if p_val <= 0.001 else f'p={p_val:.3f}'
ax4.plot(x_line, p(x_line), color=colors_seg[1], linestyle='-',
        alpha=0.8, linewidth=2, label=f'R²={corr**2:.3f}, {p_str}')

# Add radiologist labels
for _, row in condition_data.iterrows():
    rad_name = row['radiologist']
    if 'Radiologist #' in rad_name:
        rad_num = rad_name.split('#')[1]
        label = f'{rad_num}'
    else:
        label = rad_name[:3]
    ax4.text(row['years_experience'], row['mean_confidence'], label,
            ha='center', va='center', fontsize=10, color='black')

ax4.set_ylim(conf_min, conf_max)
ax4.set_xlabel('Years of experience', fontsize=12)
ax4.set_ylabel('Mean confidence', fontsize=12)
ax4.set_title('d) Experience vs confidence - with model', fontsize=14)
ax4.grid(True, alpha=0.3)

# Add legends for cases per hour
cases_sizes = [120, 60, 30]  # cases per hour
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax4.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

handles, labels = ax4.get_legend_handles_labels()
reg_handles = [h for h, l in zip(handles, labels) if 'R²=' in l]
reg_labels = [l for l in labels if 'R²=' in l]
stats_legend = ax4.legend(reg_handles, reg_labels, loc='upper left')
ax4.add_artist(cases_legend)

# Panel f) - Radiologist calibration (confidence vs accuracy) - separated by segmentation
# WITHOUT model
seg_data = bins_df[bins_df['with_segmentation'] == False].sort_values('confidence')

# Calculate accuracy by integer confidence values (1, 2, 3, ..., 10)
confidence_values = np.arange(1, 11)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# bins_df pre-aggregates radiologist_df by (with_segmentation, confidence).
rad_acc_by_rad_conf = list(seg_data['accuracy'].values)
rad_time_by_rad_conf = list(seg_data['mean_response_time'].values)

# Add perfect calibration line from (0,0) to (10,1.0) - plot this first so it appears first in legend
x_perfect = np.array([0, 10])
y_perfect = np.array([0, 1.0])
ax5.plot(x_perfect, y_perfect, 'k--', linewidth=1.5, alpha=0.7,
        label='Perfect calibration', zorder=1)

# Plot scatter points with size based on response time
# Convert response times to cases per hour for sizing
cases_per_hour = 3600 / np.array(rad_time_by_rad_conf)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

# Calculate R² for perfect calibration (forced through origin: y = 0.1*x)
perfect_slope = 0.1

perfect_pred = perfect_slope * confidence_values
perfect_deviation = np.mean(np.abs(np.array(rad_acc_by_rad_conf) - perfect_pred))

ax5.scatter(confidence_values, np.array(rad_acc_by_rad_conf),
           s=cases_per_hour_sizes, alpha=0.7, color=colors_seg[0], zorder=2,
           label=f'Mean accuracy deviation={perfect_deviation:.3f}')

ax5.set_xlabel('Pooled radiologist confidence', fontsize=12)
ax5.set_ylabel('Pooled radiologist accuracy', fontsize=12)
ax5.set_title('e) Radiologist calibration - without model', fontsize=14)
ax5.set_xlim(-0.5, 10.5)
ax5.set_ylim(-0.05, 1.05)
ax5.grid(True, alpha=0.3)

# Add cases per hour legend
cases_sizes = [120, 60, 30]  # cases per hour
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax5.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

# Add main legend with Perfect calibration and statistics
handles, labels = ax5.get_legend_handles_labels()
# Get Perfect calibration line and scatter points with deviation statistics
main_handles = [h for h, l in zip(handles, labels) if 'Perfect calibration' in l or 'Mean accuracy deviation' in l]
main_labels = [l for l in labels if 'Perfect calibration' in l or 'Mean accuracy deviation' in l]
stats_legend = ax5.legend(main_handles, main_labels, loc='upper left')
ax5.add_artist(cases_legend)

# WITH model
seg_data = bins_df[bins_df['with_segmentation'] == True].sort_values('confidence')

# bins_df pre-aggregates radiologist_df by (with_segmentation, confidence).
rad_acc_by_rad_conf = list(seg_data['accuracy'].values)
rad_time_by_rad_conf = list(seg_data['mean_response_time'].values)

# Add perfect calibration line from (0,0) to (10,1.0) - plot this first so it appears first in legend
x_perfect = np.array([0, 10])
y_perfect = np.array([0, 1.0])
ax6.plot(x_perfect, y_perfect, 'k--', linewidth=1.5, alpha=0.7,
        label='Perfect calibration', zorder=1)

# Plot scatter points with size based on response time
# Convert response times to cases per hour for sizing
cases_per_hour = 3600 / np.array(rad_time_by_rad_conf)
cases_per_hour_sizes = cases_per_hour * 2  # Scale for visibility

# Calculate R² for perfect calibration line (forced through origin: y = 0.1*x)
perfect_slope = 0.1
perfect_pred = perfect_slope * confidence_values
perfect_deviation = np.mean(np.abs(np.array(rad_acc_by_rad_conf) - perfect_pred))

ax6.scatter(confidence_values, np.array(rad_acc_by_rad_conf),
           s=cases_per_hour_sizes, alpha=0.7, color=reference_colors[1], zorder=2,
           label=f'Mean accuracy deviation={perfect_deviation:.3f}')

ax6.set_xlabel('Pooled radiologist confidence', fontsize=12)
ax6.set_ylabel('Pooled radiologist accuracy', fontsize=12)
ax6.set_title('f) Radiologist calibration - with model', fontsize=14)
ax6.set_xlim(-0.5, 10.5)
ax6.set_ylim(-0.05, 1.05)
ax6.grid(True, alpha=0.3)

# Add cases per hour legend
cases_sizes = [120, 60, 30]  # cases per hour
cases_elements = []
for cases in cases_sizes:
    cases_elements.append(plt.scatter([], [], s=cases*2, alpha=0.7,
                                   color='gray', label=f'{cases}'))
cases_legend = ax6.legend(cases_elements, [f'{c}' for c in cases_sizes],
                       loc='lower right', title='Cases reported /hr', framealpha=0.9)

# Add main legend with Perfect calibration and statistics
handles, labels = ax6.get_legend_handles_labels()
# Get Perfect calibration line and scatter points with deviation statistics
main_handles = [h for h, l in zip(handles, labels) if 'Perfect calibration' in l or 'Mean accuracy deviation' in l]
main_labels = [l for l in labels if 'Perfect calibration' in l or 'Mean accuracy deviation' in l]
stats_legend = ax6.legend(main_handles, main_labels, loc='upper left')
ax6.add_artist(cases_legend)

plt.tight_layout()

fig_5_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_5.png')
plt.savefig(fig_5_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Fig_5 saved to: {fig_5_path}")

# Save as SVG
fig_5_svg_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_5.svg')
plt.savefig(fig_5_svg_path, format='svg', bbox_inches='tight', facecolor='white')
print(f"Fig_5 saved to: {fig_5_svg_path}")

# ── Figure-displayed regression statistics (printed for table/text consistency) ──
# Recompute the exact R²/p-value pairs that appear as regression labels in
# panels a-f. These must match the values cited in manuscript paragraphs 107
# and 109 (experience vs accuracy / confidence / response time).
print("\n" + "─" * 78)
print("Fig 5 regression statistics (matching figure panel labels):")
print("─" * 78)

def _reg_stats(x, y, label):
    valid = x.notna() & y.notna()
    x = x[valid].values; y = y[valid].values
    if len(x) < 3:
        return None
    r, p = stats.pearsonr(x, y)
    # AIC via statsmodels OLS (matches manuscript values; uses Gaussian
    # log-likelihood: -2·logL + 2·k where k=2 (slope + intercept))
    try:
        import statsmodels.api as sm
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        aic = float(ols.aic)
        slope = float(ols.params[1]); intercept = float(ols.params[0])
    except ImportError:
        z = np.polyfit(x, y, 1)
        yhat = np.polyval(z, x)
        rss = float(np.sum((y - yhat) ** 2))
        n = len(x)
        aic = n * np.log(rss / n) + 2 * 2
        slope = float(z[0]); intercept = float(z[1])
    # Print p with sub-millisecond precision when p<0.001 so the manuscript's
    # `p<0.001` claim cannot be misread as the rounded `p=0.001`.
    p_str = f"{p:.3g}" if p < 0.001 else f"{p:.3f}"
    print(f"  {label:55s}  r = {r:+.3f}  R² = {r**2:.3f}  p = {p_str}  AIC = {aic:.3f}")
    return {'r': r, 'r2': r**2, 'p': p, 'aic': aic, 'slope': slope, 'intercept': intercept}

# Panel a/b: Years vs accuracy
_w_acc = experience_df[experience_df['with_segmentation'] == False]
_m_acc = experience_df[experience_df['with_segmentation'] == True]
_pa = _reg_stats(_w_acc['years_experience'], _w_acc['rad_accuracy'],
                 'Years vs accuracy — without model')
_pb = _reg_stats(_m_acc['years_experience'], _m_acc['rad_accuracy'],
                 'Years vs accuracy — with model')

# Panel c/d: Years vs mean confidence
_pc = _reg_stats(_w_acc['years_experience'], _w_acc['mean_confidence'],
                 'Years vs mean confidence — without model')
_pd = _reg_stats(_m_acc['years_experience'], _m_acc['mean_confidence'],
                 'Years vs mean confidence — with model')

# Panel: Years vs response time (mean response time)
_pe = _reg_stats(_w_acc['years_experience'], _w_acc['mean_response_time'],
                 'Years vs mean response time — without model')
_pf = _reg_stats(_m_acc['years_experience'], _m_acc['mean_response_time'],
                 'Years vs mean response time — with model')

# Mean confidence vs mean response time (per reader, per arm)
_pg = _reg_stats(_w_acc['mean_confidence'], _w_acc['mean_response_time'],
                 'Confidence vs response time — without model')
_ph = _reg_stats(_m_acc['mean_confidence'], _m_acc['mean_response_time'],
                 'Confidence vs response time — with model')

# Per-reader-summary regression of mean confidence vs mean accuracy.
# NB: residual reported here is deviation from the FITTED line, not from
# perfect calibration; the panel-displayed deviation (computed below) is
# the pooled bin deviation from perfect calibration (y = 0.1·confidence).
print("\nPer-reader-summary regression (mean confidence/10 vs mean accuracy, fitted-line residual):")
def _calib_slope(df, label):
    if len(df) < 3: return None
    x = (df['mean_confidence'] / 10.0).values
    y = df['rad_accuracy'].values
    z = np.polyfit(x, y, 1)
    yhat = np.polyval(z, x)
    deviation = float(np.mean(np.abs(y - yhat)))
    print(f"  {label:30s}  slope = {z[0]:+.3f} per unit conf  intercept = {z[1]:+.3f}  fitted-line residual = {deviation:.3f}")
_calib_slope(_w_acc, 'Without model')
_calib_slope(_m_acc, 'With model')

# Per-reader calibration stats (matches manuscript Para 112).
# For each reader × support condition we compute two complementary stats:
#   * Slope: per-reader case-level regression of correct_prediction on
#     confidence (each case is one row, conf on the raw 1-10 scale).
#   * Bin deviation: per-reader bin-level deviation of accuracy from the
#     perfect calibration line (mean over confidence bins of
#     |acc_bin - 0.1·conf|). Bin-level aggregation matches what the panels
#     visualise; case-level slope matches how the manuscript reports it.
# Then aggregate (mean ± SD) across readers.
print("\nPer-reader calibration (case-level slope, bin-level deviation from perfect line):")
def _per_reader_calib(rdf_full, with_seg, label):
    sub = rdf_full[rdf_full['with_segmentation'] == with_seg]
    slopes, devs = [], []
    for rid, rsub in sub.groupby('radiologist'):
        conf_case = rsub['confidence'].values.astype(float)
        acc_case = rsub['correct_prediction'].values.astype(float)
        if len(np.unique(conf_case)) < 2:
            continue
        z = np.polyfit(conf_case, acc_case, 1)
        slopes.append(float(z[0]))
        bins = rsub.groupby('confidence')['correct_prediction'].mean()
        conf_bin = bins.index.values.astype(float)
        acc_bin = bins.values.astype(float)
        devs.append(float(np.mean(np.abs(acc_bin - 0.1 * conf_bin))))
    slopes = np.asarray(slopes); devs = np.asarray(devs)
    print(f"  {label:30s}  n_readers={len(slopes):>2}  "
          f"slope = {slopes.mean():+.3f} ± {slopes.std(ddof=1):.3f}  "
          f"bin deviation = {devs.mean():.3f} ± {devs.std(ddof=1):.3f}")
    return slopes, devs
_per_reader_calib(_rdf_full, False, 'Without model')
_per_reader_calib(_rdf_full, True,  'With model')

# Pooled-across-readers bin deviation actually shown on panels e and f.
# bins_df aggregates correct_prediction by (with_segmentation, confidence)
# across all readers, then computes |acc - 0.1·conf|.
print("\nPooled-across-readers bin deviation from perfect calibration (matches Fig 5 panels e/f legend):")
for with_seg, label in [(False, 'Without model'), (True, 'With model')]:
    sub = bins_df[bins_df['with_segmentation'] == with_seg].sort_values('confidence')
    conf = sub['confidence'].values.astype(float)
    acc = sub['accuracy'].values.astype(float)
    dev = float(np.mean(np.abs(acc - 0.1 * conf)))
    print(f"  {label:30s}  pooled bin deviation = {dev:.3f}")

print("─" * 78)

# Show the figure
plt.close()
