#!/usr/bin/env python3
"""Extended Data Figure 5: Human agent performance by pathology dataset.

Self-contained reproduction script. Reads two CSVs from
`data/source_data/extended_data_figure_5/csv/` and renders
`data/figures/Extended_Data_Fig_5.png` byte-identical to the canonical.

Inputs:
  - edf5_panel_data.csv         : 110 rows of per-(rad, pathology, condition) aggregates
  - edf5_radiologist_meta.csv   : 11 rows of radiologist experience metadata

"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
})

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'extended_data_figure_5', 'csv')
META_CSV = os.path.join(SRC_DIR, 'edf5_radiologist_meta.csv')
RDF_CSV  = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2', 'radiologist_df.csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


for f in (META_CSV, RDF_CSV):
    if not os.path.isfile(f):
        print(f"ERROR: required input missing: {f}")
        sys.exit(2)
os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

# Compute panel aggregate live from radiologist_df.csv: per-reader per-
# pathology per-condition mean of correct_prediction / confidence /
# response_time + n_cases.
print(f"Computing panel aggregate live from {RDF_CSV}...")
_rdf_full = pd.read_csv(RDF_CSV, float_precision='round_trip')
panel = (
    _rdf_full
    .groupby(['radiologist', 'Pathology', 'with_segmentation'])
    .agg(correct_prediction=('correct_prediction', 'mean'),
         confidence=('confidence', 'mean'),
         response_time=('response_time', 'mean'),
         n_cases=('case_id', 'count'))
    .reset_index()
)
print(f"  {panel.shape[0]} rows x {panel.shape[1]} cols (live-derived)")

print(f"Loading radiologist meta from {META_CSV}...")
meta = pd.read_csv(META_CSV, float_precision='round_trip')
print(f"  {meta.shape[0]} rows")

# Reconstruct a minimal radiologist_df-like frame with de-identified placeholder case_ids.
# De-identified case_ids preserve the heatmap groupby().mean() outputs byte-identically
# without exposing any patient-level data.
panel['case_id'] = [
    f'agg_{r}_{p}_{s}_{i}'
    for i, (r, p, s) in enumerate(zip(panel['radiologist'], panel['Pathology'], panel['with_segmentation']))
]
radiologist_df = panel[[
    'radiologist', 'Pathology', 'with_segmentation', 'case_id',
    'correct_prediction', 'confidence', 'response_time',
]].copy()

experience_df = meta.copy()


# Visualize performance by cohort and pathology (3x2 heatmap)
# User-adjustable width ratio for panels without colorbar
panel_width_ratio = 0.85  # Increase width of panels a, c, e

fig, axes = plt.subplots(3, 2, figsize=(16, 18),
                       gridspec_kw={'width_ratios': [panel_width_ratio, 1.05], 'wspace': 0.4, 'hspace': 0.5})

# Filter pathologies with at least MIN_CASES_PER_PATHOLOGY unique cases
pathology_counts = radiologist_df.groupby('Pathology')['case_id'].nunique()
MIN_CASES_PER_PATHOLOGY = 5
valid_pathologies = pathology_counts[pathology_counts >= MIN_CASES_PER_PATHOLOGY].index
print(f"Valid pathologies (>= {MIN_CASES_PER_PATHOLOGY} cases): {list(valid_pathologies)}")

pathology_data = radiologist_df[radiologist_df['Pathology'].isin(valid_pathologies)]

# Define metrics and their configurations - all use same color scheme
metrics = [
    ('correct_prediction', 'Accuracy', 'RdYlBu_r', 0, 1, 'higher'),
    ('confidence', 'Confidence (1-10)', 'RdYlBu_r', 1, 10, 'higher'),
    ('response_time', 'Response time (s)', 'RdYlBu_r', None, None, 'lower')
]

# Store all heatmap data for white border calculation
all_heatmap_data = {}

for row, (metric, title, cmap, vmin, vmax, better_direction) in enumerate(metrics):
    # Create heatmaps for without and with segmentation
    for col, with_seg in enumerate([False, True]):
        condition_data = pathology_data[pathology_data['with_segmentation'] == with_seg]
        condition_name = 'With Model' if with_seg else 'Without Model'

        # Create heatmap data
        heatmap_data = condition_data.groupby(['radiologist', 'Pathology'])[metric].mean().unstack(fill_value=np.nan)

        # Add mean row at the top with whitespace separation
        mean_row = heatmap_data.mean(axis=0)
        mean_row.name = 'Mean'

        # Create whitespace row
        whitespace_row = pd.Series(np.nan, index=heatmap_data.columns, name=' ')

        # Concatenate with whitespace row
        heatmap_data = pd.concat([pd.DataFrame([mean_row]),
                                pd.DataFrame([whitespace_row]),
                                heatmap_data])

        # Store for white border calculation
        all_heatmap_data[f'{metric}_{with_seg}'] = heatmap_data

        # Determine vmin/vmax for response time
        if vmin is None:
            vmin = heatmap_data.min().min()
        if vmax is None:
            vmax = heatmap_data.max().max()

        # Create the heatmap with common colorbar for each row (panels a&b, c&d, e&f)
        cbar = col == 1  # Show colorbar on right panel of each row

        # For consistency, make all heatmaps same width by adjusting colorbar
        if cbar:
            cbar_kws = {'label': title, 'shrink': 0.8}
        else:
            cbar_kws = None

        # Create mask to hide the whitespace row
        mask = pd.DataFrame(False, index=heatmap_data.index, columns=heatmap_data.columns)
        mask.loc[' ', :] = True  # Mask the whitespace row

        im = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap,
                   ax=axes[row, col], cbar=cbar,
                   vmin=vmin, vmax=vmax,
                   cbar_kws=cbar_kws,
                   mask=mask,
                   linewidths=0, linecolor='white')

        texts = axes[row, col].texts
        n_cols = len(heatmap_data.columns)
        for i, text in enumerate(texts):
            if text.get_text() == '-':
                row_idx = i // n_cols
                if row_idx < len(heatmap_data.index) and heatmap_data.index[row_idx] == ' ':
                    text.set_visible(False)

        title_formatted = title[0].upper() + title[1:].lower()
        condition_formatted = condition_name[0].lower() + condition_name[1:].lower()
        axes[row, col].set_title(f'{chr(97 + row*2 + col)}) {title_formatted} ({condition_formatted})')
        # Set empty xlabel for panels a-f
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('Radiologist')

        # X-axis tick rotation + size
        axes[row, col].tick_params(axis='x', rotation=45, labelsize=10)
        # Set horizontal alignment for better readability
        for tick in axes[row, col].get_xticklabels():
            tick.set_ha('right')
            tick.set_va('top')

        # Add newlines to multi-word x-tick labels
        current_labels = [label.get_text() for label in axes[row, col].get_xticklabels()]
        new_labels = []
        for label in current_labels:
            # Split on spaces and rejoin with newlines
            words = label.split()
            if len(words) > 1:
                # For labels with multiple words, put each word on a new line
                new_label = '\n'.join(words)
            else:
                new_label = label
            new_labels.append(new_label)
        axes[row, col].set_xticklabels(new_labels)

        # Add experience years to y-axis labels
        temp_experience_df = experience_df

        ADULT_NEURO_IDS = {'#1', '#4', '#5', '#6', '#7', '#9'}
        ADULT_PAEDIATRIC_NEURO_IDS = {'#8', '#11'}
        PAEDIATRIC_NEURO_IDS = {'#3'}
        HEAD_NECK_GENERAL_IDS = {'#10'}
        MUSCULO_GENERAL_IDS = {'#2'}

        def subspecialty_marker(num):
            if num in ADULT_NEURO_IDS:
                return '*'
            if num in ADULT_PAEDIATRIC_NEURO_IDS:
                return '†'
            if num in PAEDIATRIC_NEURO_IDS:
                return '‡'
            if num in HEAD_NECK_GENERAL_IDS:
                return '§'
            if num in MUSCULO_GENERAL_IDS:
                return '¶'
            return ''

        new_y_labels = []
        for label in heatmap_data.index:
            if label == 'Mean':
                new_y_labels.append('Mean')
            else:
                exp_data = temp_experience_df[temp_experience_df['radiologist'] == label]
                if len(exp_data) > 0:
                    years_exp = exp_data['years_experience'].iloc[0]
                    if 'Radiologist #' in label:
                        number_part = label.replace('Radiologist ', '')
                        marker = subspecialty_marker(number_part)
                        new_y_labels.append(f'{number_part}{marker} ({int(years_exp)} yrs)')
                    else:
                        new_y_labels.append(f'{label} ({int(years_exp)} yrs)')
                else:
                    new_y_labels.append(label)

        axes[row, col].set_yticklabels(new_y_labels)

        # Hide the y-axis tick mark for the whitespace row — the cell is
        # masked, the label is blank, but matplotlib still draws the small
        # tick line, which reads as a stray '-' in the gap between Mean
        # and the individual radiologists.
        for tick_idx, label in enumerate(heatmap_data.index):
            if label == ' ':
                tick = axes[row, col].yaxis.get_major_ticks()[tick_idx]
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)

white_border_counts = {}

for row, (metric, title, cmap, vmin, vmax, better_direction) in enumerate(metrics):
    without_data = all_heatmap_data[f'{metric}_False']
    with_data = all_heatmap_data[f'{metric}_True']

    # Add white borders for both conditions
    for col, with_seg in enumerate([False, True]):
        current_data = with_data if with_seg else without_data
        other_data = without_data if with_seg else with_data

        # Count white borders for this panel
        white_border_count = 0
        total_valid_cells = 0

        for rad_idx, radiologist in enumerate(current_data.index):
            # Skip whitespace row
            if radiologist == ' ':
                continue

            for path_idx, pathology in enumerate(current_data.columns):
                if radiologist in other_data.index and pathology in other_data.columns:
                    current_val = current_data.loc[radiologist, pathology]
                    other_val = other_data.loc[radiologist, pathology]

                    # Check if this is the better performing condition
                    if not pd.isna(current_val) and not pd.isna(other_val):
                        is_better = False
                        if better_direction == 'higher':
                            is_better = current_val > other_val
                        elif better_direction == 'lower':
                            is_better = current_val < other_val

                        # Only count non-Mean cells for percentage calculation
                        if radiologist != 'Mean':
                            total_valid_cells += 1

                        if is_better:
                            if radiologist != 'Mean':
                                white_border_count += 1
                            # Add white border (including for Mean row)
                            axes[row, col].add_patch(plt.Rectangle((path_idx, rad_idx), 1, 1,
                                                                 fill=False, edgecolor='white',
                                                                 linewidth=3))

        # Store percentage for this panel
        if total_valid_cells > 0:
            percentage = (white_border_count / total_valid_cells) * 100
            white_border_counts[(row, col)] = percentage

# Add overall title
fig.suptitle('Radiologist performance by pathology dataset (white borders indicate better performance)',
             fontsize=16, y=0.92)

legend_text = (
    'Subspecialty markers: * Adult Neuroradiologist;  '
    '† Adult and Paediatric Neuroradiologist;  '
    '‡ Paediatric Neuroradiologist\n'
    '§ Head and Neck and General Radiologist;  '
    '¶ Musculoskeletal and General Radiologist.'
)
fig.text(
    0.5, 0.015, legend_text,
    ha='center', va='bottom', fontsize=10, wrap=True,
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='0.8'),
)

plt.tight_layout()

# Save as both PNG and SVG with new name
plt.savefig(os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_5.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_5.svg'),
            format='svg', bbox_inches='tight')
print(f"Extended_Data_Fig_5 saved to: {os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_5.png')}")
print(f"Extended_Data_Fig_5 saved to: {os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_5.svg')}")
