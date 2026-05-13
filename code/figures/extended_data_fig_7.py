#!/usr/bin/env python3
"""Extended Data Figure 7: Pathology, size, and radiomic assessment.

Self-contained reproduction script (no runpy / sibling plotting block).
Reads a single 171 KB pre-merged CSV (`edf7_inputs.csv`) holding only the
8 columns the radar plotting actually consumes:
  radiologist, inferred_pathology, volume_category, radiomic_category,
  with_segmentation, correct_prediction, confidence, response_time

Output: data/figures/Extended_Data_Fig_7.png  (and .svg)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import seaborn as sns

sns.set_palette("husl")

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'extended_data_figure_7')
EDF7_INPUTS_CSV = os.path.join(SRC_DIR, 'edf7_inputs.csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


def main():
    if not os.path.isfile(EDF7_INPUTS_CSV):
        print(f"ERROR: required input missing: {EDF7_INPUTS_CSV}")
        sys.exit(2)
    os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

    print(f"Loading edf7_inputs from {EDF7_INPUTS_CSV}...")
    radiologist_df = pd.read_csv(EDF7_INPUTS_CSV, float_precision='round_trip')
    print(f"  {radiologist_df.shape[0]} rows x {radiologist_df.shape[1]} cols")
    print("Rendering Extended Data Figure 7...")
    _render(radiologist_df, FIGURES_OUTPUT_PATH)
    print(f"\nEDF 7 saved to {FIGURES_OUTPUT_PATH}/Extended_Data_Fig_7.png")


def _render(radiologist_df, FIGURES_OUTPUT_PATH):
    def add_radar_subplot(ax, categories, values_without, values_with, title, ylim=(0, 1.0)):
        """Add a radar subplot to the figure."""
        # Define reference color palette from nnunet_enhancement_prediction_article_figures.ipynb
        reference_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        N = len(categories)

        # Format categories with line breaks based on the panel type
        formatted_categories = []
        for cat in categories:
            if 'Pathology' in title or 'Radiomic' in title:
                # Split on spaces and join with newlines
                formatted_cat = '\n'.join(cat.split())
            else:
                # Keep other categories as is
                formatted_cat = cat
            formatted_categories.append(formatted_cat)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        values_without_plot = values_without + values_without[:1]
        values_with_plot = values_with + values_with[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Plot the lines for both conditions
        ax.plot(angles, values_without_plot, 'o-', linewidth=2.5,
                color=reference_colors[3], markersize=0, label='Without model')
        ax.plot(angles, values_with_plot, 'o-', linewidth=2.5,
                color=reference_colors[0], markersize=0, label='With model')

        # Create differential shading
        # For each segment between categories, determine which line is higher
        for i in range(len(angles) - 1):
            # Get the values for this segment
            angle_segment = [angles[i], angles[i+1]]
            without_segment = [values_without_plot[i], values_without_plot[i+1]]
            with_segment = [values_with_plot[i], values_with_plot[i+1]]

            # Determine if with_segmentation is better (higher for accuracy/confidence, lower for time)
            # For response time, lower is better, so we need to check the metric type
            if 'Response Time' in title or 'response time' in title.lower():
                # For response time, lower is better
                if with_segment[0] < without_segment[0] or with_segment[1] < without_segment[1]:
                    # With segmentation is better (lower time) - shade with "with" color
                    upper_vals = without_segment
                    lower_vals = with_segment
                    fill_color = reference_colors[0]  # With model color
                else:
                    # Without segmentation is better (lower time) - shade with "without" color
                    upper_vals = with_segment
                    lower_vals = without_segment
                    fill_color = reference_colors[3]  # Without model color
            else:
                # For accuracy and confidence, higher is better
                if with_segment[0] > without_segment[0] or with_segment[1] > without_segment[1]:
                    # With segmentation is better (higher value) - shade with "with" color
                    upper_vals = with_segment
                    lower_vals = without_segment
                    fill_color = reference_colors[0]  # With model color
                else:
                    # Without segmentation is better (higher value) - shade with "without" color
                    upper_vals = without_segment
                    lower_vals = with_segment
                    fill_color = reference_colors[3]  # Without model color

            # Create vertices for the filled area
            vertices = [(angle_segment[0], lower_vals[0]),
                        (angle_segment[1], lower_vals[1]),
                        (angle_segment[1], upper_vals[1]),
                        (angle_segment[0], upper_vals[0])]

            poly = Polygon(vertices, facecolor=fill_color, alpha=0.3, edgecolor='none')
            ax.add_patch(poly)

        # Set the category labels without degree annotations - moved out a bit like Figure_5.png panel b
        ax.set_thetagrids(np.degrees(angles[:-1]), formatted_categories)

        # Move labels out a bit from the edge (negative offset to move inward)
        for label in ax.get_xticklabels():
            label.set_position((label.get_position()[0], label.get_position()[1] - 0.15))

        # Remove any degree labels that might appear
        for label in ax.get_xticklabels():
            # Check if the label text contains degree symbols or numbers
            text = label.get_text()
            if '°' in text or text.replace('.', '').replace('-', '').isdigit():
                label.set_visible(False)

        ax.set_ylim(ylim)

        # Set radius tick labels to include the maximum value
        # Force the grid to extend to the maximum value
        if ylim[1] == 1.0:
            # For accuracy plots (0-1 range)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=0)
        elif ylim[1] == 10:
            # For confidence plots (0-10 range)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
            ax.set_rgrids([2, 4, 6, 8, 10], angle=0)
        elif ylim[1] == 120:
            # For response time plots (0-120 range)
            ax.set_yticks([20, 40, 60, 80, 100, 120])
            ax.set_yticklabels(['20', '40', '60', '80', '100', '120'], fontsize=10)
            ax.set_rgrids([20, 40, 60, 80, 100, 120], angle=0)
        elif ylim[1] == 140:
            # For cases per hour plots (0-140 range)
            ax.set_yticks([20, 40, 60, 80, 100, 120, 140])
            ax.set_yticklabels(['20', '40', '60', '80', '100', '120', '140'], fontsize=10)
            ax.set_rgrids([20, 40, 60, 80, 100, 120, 140], angle=0)
        elif ylim[1] == 160:
            # For cases per hour plots (0-160 range)
            ax.set_yticks([20, 40, 60, 80, 100, 120, 140, 160])
            ax.set_yticklabels(['20', '40', '60', '80', '100', '120', '140', '160'], fontsize=10)
            ax.set_rgrids([20, 40, 60, 80, 100, 120, 140, 160], angle=0)

        # Update title to have only first letter capitalized
        title_parts = title.split(')')
        if len(title_parts) > 1:
            panel_letter = title_parts[0] + ')'
            title_text = title_parts[1].strip()
            title_text = title_text[0].upper() + title_text[1:].lower() if title_text else ''
            formatted_title = f'{panel_letter} {title_text}'
        else:
            formatted_title = title[0].upper() + title[1:].lower() if title else ''

        ax.set_title(formatted_title, size=11, pad=15)
        ax.grid(True, alpha=0.3)

        # Set radial ticks based on ylim
        if ylim[1] == 1.0:
            ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        elif ylim[1] == 10:
            ax.set_rticks([2, 4, 6, 8, 10])
        elif ylim[1] == 60:
            ax.set_rticks([15, 30, 45, 60])
        elif ylim[1] == 120:
            ax.set_rticks([20, 40, 60, 80, 100, 120])
        elif ylim[1] == 140:
            ax.set_rticks([20, 40, 60, 80, 100, 120, 140])
        elif ylim[1] == 160:
            ax.set_rticks([20, 40, 60, 80, 100, 120, 140, 160])
        else:
            n_ticks = 5
            tick_interval = (ylim[1] - ylim[0]) / (n_ticks - 1)
            ax.set_rticks([ylim[0] + i * tick_interval for i in range(n_ticks)])

        ax.set_rlabel_position(45)  # Move radius labels to 45 degree position

        # Ensure the polar frame (outer circle) is visible
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_linewidth(1)
        ax.spines['polar'].set_color('black')

        return ax

    # Radar analysis

    combined_df_analysis = radiologist_df.copy()

    # Create new 3x3 figure for radar plots with increased height for row spacing
    fig = plt.figure(figsize=(18, 20))

    # Define reference color palette from nnunet_enhancement_prediction_article_figures.ipynb
    reference_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Define pathology categories
    pathology_categories = ['Meningioma', 'Metastases', 'Paediatric presurgical tumour',
                           'Postoperative glioma resection', 'Presurgical glioma']

    # Define size categories as in the requirement
    size_categories = ['Micro', 'Small', 'Medium', 'Large', 'Very Large']
    size_labels = ['Micro\n(<0.5cm³)', 'Small\n(0.5-1cm³)', 'Medium\n(1-5cm³)',
                   'Large\n(5-10cm³)', 'Very Large\n(>10cm³)']

    all_radiomic_categories = combined_df_analysis['radiomic_category'].dropna().unique().tolist()
    radiomic_categories = [
        cat for cat in all_radiomic_categories
        if not ('single lesion' in cat.lower() and 'unclassified' in cat.lower())
    ]


    # Process each metric and category combination
    print("\nCalculating metrics for radar plots...")

    ax_acc_path = fig.add_subplot(3, 3, 1, projection='polar')

    # Map the pathology names properly - use direct mapping (no changes)
    pathology_mapping = {
        'Meningioma': 'Meningioma',
        'Metastases': 'Metastases',
        'Paediatric presurgical tumour': 'Paediatric presurgical tumour',
        'Postoperative glioma resection': 'Postoperative glioma resection',
        'Presurgical glioma': 'Presurgical glioma'
    }

    acc_path_without = []
    acc_path_with = []
    valid_pathologies = []

    for display_path in pathology_categories:
        mapped_path = pathology_mapping.get(display_path, display_path)
        path_df = combined_df_analysis[combined_df_analysis['inferred_pathology'] == mapped_path]

        if len(path_df) > 0:
            without_seg = path_df[~path_df['with_segmentation']]
            with_seg = path_df[path_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                acc_without = without_seg['correct_prediction'].mean()
                acc_with = with_seg['correct_prediction'].mean()
                acc_path_without.append(acc_without)
                acc_path_with.append(acc_with)
                valid_pathologies.append(display_path)
                print(f"  {display_path}: without={acc_without:.3f}, with={acc_with:.3f}")

    if valid_pathologies:
        add_radar_subplot(ax_acc_path, valid_pathologies, acc_path_without, acc_path_with,
                         'a) Accuracy by Pathology', ylim=(0, 1.0))

    ax_acc_size = fig.add_subplot(3, 3, 2, projection='polar')

    acc_size_without = []
    acc_size_with = []
    valid_sizes = []

    for size_cat, size_label in zip(size_categories, size_labels):
        size_df = combined_df_analysis[combined_df_analysis['volume_category'] == size_cat]

        if len(size_df) > 0:
            without_seg = size_df[~size_df['with_segmentation']]
            with_seg = size_df[size_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                acc_without = without_seg['correct_prediction'].mean()
                acc_with = with_seg['correct_prediction'].mean()
                acc_size_without.append(acc_without)
                acc_size_with.append(acc_with)
                valid_sizes.append(size_label)
                print(f"  {size_cat}: without={acc_without:.3f}, with={acc_with:.3f}")

    if valid_sizes:
        add_radar_subplot(ax_acc_size, valid_sizes, acc_size_without, acc_size_with,
                         'b) Accuracy by Lesion Size', ylim=(0, 1.0))

    ax_acc_radiomic = fig.add_subplot(3, 3, 3, projection='polar')

    acc_radiomic_without = []
    acc_radiomic_with = []
    valid_radiomics = []

    # Create a filtered copy of the dataframe excluding any unclassified single lesions
    # Use case-insensitive filtering to catch variations
    mask = ~(combined_df_analysis['radiomic_category'].str.lower().str.contains('unclassified', na=False) &
             combined_df_analysis['radiomic_category'].str.lower().str.contains('single lesion', na=False))
    combined_df_radiomic_filtered = combined_df_analysis[mask].copy()

    for radiomic_cat in radiomic_categories:
        radiomic_df = combined_df_radiomic_filtered[combined_df_radiomic_filtered['radiomic_category'] == radiomic_cat]

        if len(radiomic_df) > 0:
            without_seg = radiomic_df[~radiomic_df['with_segmentation']]
            with_seg = radiomic_df[radiomic_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                acc_without = without_seg['correct_prediction'].mean()
                acc_with = with_seg['correct_prediction'].mean()
                acc_radiomic_without.append(acc_without)
                acc_radiomic_with.append(acc_with)
                valid_radiomics.append(radiomic_cat)
                print(f"  {radiomic_cat}: without={acc_without:.3f}, with={acc_with:.3f}")

    if valid_radiomics:
        add_radar_subplot(ax_acc_radiomic, valid_radiomics, acc_radiomic_without, acc_radiomic_with,
                         'c) Accuracy by Radiomic Category', ylim=(0, 1.0))

    ax_conf_path = fig.add_subplot(3, 3, 4, projection='polar')

    conf_path_without = []
    conf_path_with = []
    valid_pathologies_conf = []

    for display_path in pathology_categories:
        mapped_path = pathology_mapping.get(display_path, display_path)
        path_df = combined_df_analysis[combined_df_analysis['inferred_pathology'] == mapped_path]

        if len(path_df) > 0:
            without_seg = path_df[~path_df['with_segmentation']]
            with_seg = path_df[path_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                conf_without = without_seg['confidence'].mean()  # Already in 0-10 range
                conf_with = with_seg['confidence'].mean()
                conf_path_without.append(conf_without)
                conf_path_with.append(conf_with)
                valid_pathologies_conf.append(display_path)
                print(f"  {display_path}: without={conf_without:.3f}, with={conf_with:.3f}")

    if valid_pathologies_conf:
        add_radar_subplot(ax_conf_path, valid_pathologies_conf, conf_path_without, conf_path_with,
                         'd) Confidence by Pathology', ylim=(0, 10))

    ax_conf_size = fig.add_subplot(3, 3, 5, projection='polar')

    conf_size_without = []
    conf_size_with = []
    valid_sizes_conf = []

    for size_cat, size_label in zip(size_categories, size_labels):
        size_df = combined_df_analysis[combined_df_analysis['volume_category'] == size_cat]

        if len(size_df) > 0:
            without_seg = size_df[~size_df['with_segmentation']]
            with_seg = size_df[size_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                conf_without = without_seg['confidence'].mean()  # Already in 0-10 range
                conf_with = with_seg['confidence'].mean()
                conf_size_without.append(conf_without)
                conf_size_with.append(conf_with)
                valid_sizes_conf.append(size_label)
                print(f"  {size_cat}: without={conf_without:.3f}, with={conf_with:.3f}")

    if valid_sizes_conf:
        add_radar_subplot(ax_conf_size, valid_sizes_conf, conf_size_without, conf_size_with,
                         'e) Confidence by Lesion Size', ylim=(0, 10))

    ax_conf_radiomic = fig.add_subplot(3, 3, 6, projection='polar')

    conf_radiomic_without = []
    conf_radiomic_with = []
    valid_radiomics_conf = []

    # Create a filtered copy of the dataframe excluding any unclassified single lesions
    # Use case-insensitive filtering to catch variations
    mask = ~(combined_df_analysis['radiomic_category'].str.lower().str.contains('unclassified', na=False) &
             combined_df_analysis['radiomic_category'].str.lower().str.contains('single lesion', na=False))
    combined_df_radiomic_filtered = combined_df_analysis[mask].copy()

    for radiomic_cat in radiomic_categories:
        radiomic_df = combined_df_radiomic_filtered[combined_df_radiomic_filtered['radiomic_category'] == radiomic_cat]

        if len(radiomic_df) > 0:
            without_seg = radiomic_df[~radiomic_df['with_segmentation']]
            with_seg = radiomic_df[radiomic_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                conf_without = without_seg['confidence'].mean()  # Already in 0-10 range
                conf_with = with_seg['confidence'].mean()
                conf_radiomic_without.append(conf_without)
                conf_radiomic_with.append(conf_with)
                valid_radiomics_conf.append(radiomic_cat)
                print(f"  {radiomic_cat}: without={conf_without:.3f}, with={conf_with:.3f}")

    if valid_radiomics_conf:
        add_radar_subplot(ax_conf_radiomic, valid_radiomics_conf, conf_radiomic_without, conf_radiomic_with,
                         'f) Confidence by Radiomic Category', ylim=(0, 10))

    ax_time_path = fig.add_subplot(3, 3, 7, projection='polar')

    cases_path_without = []
    cases_path_with = []
    valid_pathologies_time = []

    for display_path in pathology_categories:
        mapped_path = pathology_mapping.get(display_path, display_path)
        path_df = combined_df_analysis[combined_df_analysis['inferred_pathology'] == mapped_path]

        if len(path_df) > 0:
            without_seg = path_df[~path_df['with_segmentation']]
            with_seg = path_df[path_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                # Convert response time to cases per hour
                time_without = without_seg['response_time'].mean()
                time_with = with_seg['response_time'].mean()
                cases_without = 3600 / time_without if time_without > 0 else 0
                cases_with = 3600 / time_with if time_with > 0 else 0
                cases_path_without.append(cases_without)
                cases_path_with.append(cases_with)
                valid_pathologies_time.append(display_path)
                print(f"  {display_path}: without={cases_without:.1f} cases/hr, with={cases_with:.1f} cases/hr")

    if valid_pathologies_time:
        add_radar_subplot(ax_time_path, valid_pathologies_time, cases_path_without, cases_path_with,
                         'g) Cases /hr by Pathology', ylim=(0, 160))

    ax_time_size = fig.add_subplot(3, 3, 8, projection='polar')

    cases_size_without = []
    cases_size_with = []
    valid_sizes_time = []

    for size_cat, size_label in zip(size_categories, size_labels):
        size_df = combined_df_analysis[combined_df_analysis['volume_category'] == size_cat]

        if len(size_df) > 0:
            without_seg = size_df[~size_df['with_segmentation']]
            with_seg = size_df[size_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                time_without = without_seg['response_time'].mean()
                time_with = with_seg['response_time'].mean()
                cases_without = 3600 / time_without if time_without > 0 else 0
                cases_with = 3600 / time_with if time_with > 0 else 0
                cases_size_without.append(cases_without)
                cases_size_with.append(cases_with)
                valid_sizes_time.append(size_label)
                print(f"  {size_cat}: without={cases_without:.1f} cases/hr, with={cases_with:.1f} cases/hr")

    if valid_sizes_time:
        add_radar_subplot(ax_time_size, valid_sizes_time, cases_size_without, cases_size_with,
                         'h) Cases /hr by Lesion Size', ylim=(0, 160))

    ax_time_radiomic = fig.add_subplot(3, 3, 9, projection='polar')

    cases_radiomic_without = []
    cases_radiomic_with = []
    valid_radiomics_time = []

    # Create a filtered copy of the dataframe excluding any unclassified single lesions
    # Use case-insensitive filtering to catch variations
    mask = ~(combined_df_analysis['radiomic_category'].str.lower().str.contains('unclassified', na=False) &
             combined_df_analysis['radiomic_category'].str.lower().str.contains('single lesion', na=False))
    combined_df_radiomic_filtered = combined_df_analysis[mask].copy()

    for radiomic_cat in radiomic_categories:
        radiomic_df = combined_df_radiomic_filtered[combined_df_radiomic_filtered['radiomic_category'] == radiomic_cat]

        if len(radiomic_df) > 0:
            without_seg = radiomic_df[~radiomic_df['with_segmentation']]
            with_seg = radiomic_df[radiomic_df['with_segmentation']]

            if len(without_seg) > 0 and len(with_seg) > 0:
                time_without = without_seg['response_time'].mean()
                time_with = with_seg['response_time'].mean()
                cases_without = 3600 / time_without if time_without > 0 else 0
                cases_with = 3600 / time_with if time_with > 0 else 0
                cases_radiomic_without.append(cases_without)
                cases_radiomic_with.append(cases_with)
                valid_radiomics_time.append(radiomic_cat)
                print(f"  {radiomic_cat}: without={cases_without:.1f} cases/hr, with={cases_with:.1f} cases/hr")

    if valid_radiomics_time:
        add_radar_subplot(ax_time_radiomic, valid_radiomics_time, cases_radiomic_without, cases_radiomic_with,
                         'i) Cases /hr by Radiomic Category', ylim=(0, 160))

    # Add overall legend with line and shaded area descriptions
    if valid_pathologies:  # Use any valid data to get legend handles
        handles, labels = ax_acc_path.get_legend_handles_labels()

        shaded_better_with = Patch(facecolor=reference_colors[0], alpha=0.3, label='Better with model')
        shaded_better_without = Patch(facecolor=reference_colors[3], alpha=0.3, label='Better without model')

        # Combine line handles and patch handles
        all_handles = handles + [shaded_better_with, shaded_better_without]
        all_labels = labels + ['Better with model', 'Better without model']

        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.96),
                  ncol=4, fontsize=12)

    # Add main title
    fig.suptitle('Pathology, size, and radiomic assessment of radiologist performance',
                 fontsize=16, y=0.98)

    # Increased top margin from 0.97 to 0.95 for more spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure as both PNG and SVG with new name
    output_path_png = os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_7.png')
    output_path_svg = os.path.join(FIGURES_OUTPUT_PATH, 'Extended_Data_Fig_7.svg')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')

    print(f"\nExtended_Data_Fig_7 saved to: {output_path_png}")
    print(f"Extended_Data_Fig_7 saved to: {output_path_svg}")

    # Print summary statistics
    print("\nRadar analysis summary:")
    print(f"Total cases analyzed: {len(combined_df_analysis) // 2}")
    print(f"Radiologists: {len(combined_df_analysis['radiologist'].unique())}")
    print(f"Pathology categories: {len(valid_pathologies)}")
    print(f"Size categories: {len(valid_sizes)}")
    print(f"Radiomic categories: {len(radiomic_categories)}")

    # Calculate overall improvements
    overall_acc_without = combined_df_analysis[~combined_df_analysis['with_segmentation']]['correct_prediction'].mean()
    overall_acc_with = combined_df_analysis[combined_df_analysis['with_segmentation']]['correct_prediction'].mean()
    overall_conf_without = combined_df_analysis[~combined_df_analysis['with_segmentation']]['confidence'].mean()
    overall_conf_with = combined_df_analysis[combined_df_analysis['with_segmentation']]['confidence'].mean()
    overall_time_without = combined_df_analysis[~combined_df_analysis['with_segmentation']]['response_time'].mean()
    overall_time_with = combined_df_analysis[combined_df_analysis['with_segmentation']]['response_time'].mean()

    # Convert response time to cases per hour for summary
    overall_cases_without = 3600 / overall_time_without if overall_time_without > 0 else 0
    overall_cases_with = 3600 / overall_time_with if overall_time_with > 0 else 0

    print(f"\nOverall improvements:")
    print(f"  Accuracy: {overall_acc_without:.3f} to {overall_acc_with:.3f} ({(overall_acc_with - overall_acc_without)*100:+.1f}%)")
    print(f"  Confidence: {overall_conf_without:.1f} to {overall_conf_with:.1f} ({(overall_conf_with - overall_conf_without):+.1f})")
    print(f"  Cases per hour: {overall_cases_without:.1f} to {overall_cases_with:.1f} ({(overall_cases_with - overall_cases_without):+.1f} cases/hr)")


if __name__ == '__main__':
    main()
