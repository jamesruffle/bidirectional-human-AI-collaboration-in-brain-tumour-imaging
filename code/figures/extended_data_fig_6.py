#!/usr/bin/env python3
"""Extended Data Figure 6: UMAP embedding (single-file self-contained).

Reads precomputed UMAP coordinates + per-case metric deltas from
`data/source_data/extended_data_figure_6/csv/umap_analysis_results.csv`
(~87 KB) and renders Extended_Data_Fig_6 directly.

N.B. this script uses pre-computed UMAP embeddings to preserve patient privacy and ensure byte-identical reproduction of the figure. The original UMAP was computed on a high-dimensional feature space derived from the imaging data, which cannot be shared directly due to privacy concerns. By using the pre-computed UMAP coordinates, we can reproduce the exact same figure without exposing any patient-level data.

Output: data/figures/Extended_Data_Fig_6.png  (and .svg)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sns.set_palette("husl")

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'extended_data_figure_6', 'csv')
UMAP_CSV = os.path.join(SRC_DIR, 'umap_analysis_results.csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


def main():
    if not os.path.isfile(UMAP_CSV):
        print(f"ERROR: UMAP CSV not found at {UMAP_CSV}")
        sys.exit(2)

    os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

    print(f"Loading UMAP results from {UMAP_CSV}...")
    # float_precision='round_trip' preserves bit-identical float64 round-trips
    # from the canonical CSV; the default parser drops the last bit on some
    # values, which causes a tiny anti-alias drift at the alpha-edge of points.
    umap_df = pd.read_csv(UMAP_CSV, float_precision='round_trip')
    print(f"  Loaded {len(umap_df)} rows ({os.path.getsize(UMAP_CSV)/1e3:.1f} KB)")

    output_dir = FIGURES_OUTPUT_PATH

    # Create visualization with space for colorbars below
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Effect of lesion morphology, pathology, size, and distribution on radiologist performance',
                 fontsize=16, fontweight='normal', y=0.82)

    # Create GridSpec with extra space at bottom for colorbars and legends
    gs = plt.GridSpec(3, 3, figure=fig, height_ratios=[1, 0.04, 0.11], hspace=0.07, wspace=0.3)

    # Create main axes for the three panels
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Normalize nonzero_voxels for point sizing
    sizes = 50 + 200 * (umap_df['nonzero_voxels'] - umap_df['nonzero_voxels'].min()) / \
            (umap_df['nonzero_voxels'].max() - umap_df['nonzero_voxels'].min())

    icefire_cmap = matplotlib.colormaps['icefire']
    n_bins = 256

    # Panel a: Accuracy difference
    ax = axes[0]

    # Use uniform marker shape (circle) for all points
    # Calculate alpha values proportional to absolute accuracy difference
    abs_values = np.abs(umap_df['accuracy_diff'])
    alpha_values = 0.2 + 0.8 * (abs_values / 0.5)  # Scale to max of 0.5
    alpha_values = np.clip(alpha_values, 0.2, 1.0)

    # Plot all points with uniform shape
    for idx, row in umap_df.iterrows():
        ax.scatter(row['UMAP1'], row['UMAP2'],
                  c=[row['accuracy_diff']], s=sizes[idx],
                  marker='o', cmap='icefire', vmin=-0.5, vmax=0.5,
                  alpha=alpha_values[idx], edgecolors='black', linewidth=0.5)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('a) Change in radiologist accuracy (Δ proportion)', fontweight='normal')
    ax.set_aspect('equal', adjustable='box')

    # Add horizontal colorbar below panel a with transparency gradient
    cax_a = fig.add_subplot(gs[1, 0])
    # Create a colormap with varying alpha
    colors = []
    for i in range(n_bins):
        val = (i / (n_bins - 1)) * 1.0 - 0.5  # Map to [-0.5, 0.5]
        abs_val = abs(val)
        # Calculate alpha based on distance from 0 (matching scatter plot logic)
        alpha = 0.2 + 0.8 * (abs_val / 0.5)  # Scale to [0.2, 1.0]
        rgba = list(icefire_cmap(i / (n_bins - 1)))
        rgba[3] = alpha
        colors.append(rgba)
    # Create custom colormap
    cmap_with_alpha = LinearSegmentedColormap.from_list('icefire_alpha', colors, N=n_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap_with_alpha, norm=plt.Normalize(vmin=-0.5, vmax=0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax_a, orientation='horizontal')
    cbar.set_label('Higher without model ← radiologist accuracy → Higher with model', fontsize=10)

    # Panel b: Confidence difference
    ax = axes[1]

    # Calculate alpha values proportional to absolute confidence difference
    abs_values = np.abs(umap_df['confidence_diff'])
    alpha_values = 0.2 + 0.8 * (abs_values / 10)  # Scale to max of 10
    alpha_values = np.clip(alpha_values, 0.2, 1.0)

    # Plot all points with uniform shape
    for idx, row in umap_df.iterrows():
        ax.scatter(row['UMAP1'], row['UMAP2'],
                  c=[row['confidence_diff']], s=sizes[idx],
                  marker='o', cmap='icefire', vmin=-10, vmax=10,
                  alpha=alpha_values[idx], edgecolors='black', linewidth=0.5)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('b) Change in radiologist confidence (Δ score)', fontweight='normal')
    ax.set_aspect('equal', adjustable='box')

    # Add horizontal colorbar below panel b with transparency gradient
    cax_b = fig.add_subplot(gs[1, 1])
    # Create custom colormap with transparency for confidence
    colors_conf = []
    for i in range(n_bins):
        val = (i / (n_bins - 1)) * 20.0 - 10  # Map to [-10, 10]
        abs_val = abs(val)
        # Calculate alpha based on distance from 0
        alpha = 0.2 + 0.8 * (abs_val / 10.0)  # Scale to [0.2, 1.0]
        rgba = list(icefire_cmap(i / (n_bins - 1)))
        rgba[3] = alpha
        colors_conf.append(rgba)
    cmap_conf_alpha = LinearSegmentedColormap.from_list('icefire_alpha_conf', colors_conf, N=n_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap_conf_alpha, norm=plt.Normalize(vmin=-10, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax_b, orientation='horizontal')
    cbar.set_label('Higher without model ← radiologist confidence → Higher with model', fontsize=10)

    # Panel c: Response time difference
    ax = axes[2]

    # Calculate alpha values proportional to absolute response time difference
    abs_values = np.abs(umap_df['response_time_diff'])
    alpha_values = 0.2 + 0.8 * (abs_values / 20)  # Scale to max of 20
    alpha_values = np.clip(alpha_values, 0.2, 1.0)

    # Plot all points with uniform shape
    for idx, row in umap_df.iterrows():
        ax.scatter(row['UMAP1'], row['UMAP2'],
                  c=[row['response_time_diff']], s=sizes[idx],
                  marker='o', cmap='icefire_r', vmin=-20, vmax=20,
                  alpha=alpha_values[idx], edgecolors='black', linewidth=0.5)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('c) Change in review speed (Δ seconds)', fontweight='normal')
    ax.set_aspect('equal', adjustable='box')

    # Add horizontal colorbar below panel c with transparency gradient
    cax_c = fig.add_subplot(gs[1, 2])
    # Create custom colormap with transparency for response time
    colors_time = []
    for i in range(n_bins):
        val = (i / (n_bins - 1)) * 40.0 - 20  # Map to [-20, 20]
        abs_val = abs(val)
        # Calculate alpha based on distance from 0
        alpha = 0.2 + 0.8 * (abs_val / 20.0)  # Scale to [0.2, 1.0]
        rgba = list(icefire_cmap(i / (n_bins - 1)))
        rgba[3] = alpha
        colors_time.append(rgba)
    cmap_time_alpha = LinearSegmentedColormap.from_list('icefire_alpha_time', colors_time, N=n_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap_time_alpha, norm=plt.Normalize(vmin=-20, vmax=20))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax_c, orientation='horizontal')
    cbar.set_label('Faster without model ← radiologist speed → Faster with model  ', fontsize=10)

    # Adjust colorbar positions
    upwards_jitter = 0.095
    pos_a = cax_a.get_position()
    cax_a.set_position([pos_a.x0, pos_a.y0 + upwards_jitter, pos_a.width, pos_a.height])

    pos_b = cax_b.get_position()
    cax_b.set_position([pos_b.x0, pos_b.y0 + upwards_jitter, pos_b.width, pos_b.height])

    pos_c = cax_c.get_position()
    cax_c.set_position([pos_c.x0, pos_c.y0 + upwards_jitter, pos_c.width, pos_c.height])

    size_legend_elements = []
    voxel_min = umap_df['nonzero_voxels'].min()
    voxel_max = umap_df['nonzero_voxels'].max()

    # Calculate percentiles to use actual data range
    voxel_percentiles = [
        (10, 'Micro (<0.5cm³)'),
        (25, 'Small (0.5-1cm³)'),
        (50, 'Medium (1-5cm³)'),
        (75, 'Large (5-10cm³)'),
        (90, 'Very Large (>10cm³)')
    ]

    for percentile, label in voxel_percentiles:
        # Use actual percentile values from the data
        voxel_count = np.percentile(umap_df['nonzero_voxels'], percentile)

        # Calculate the exact scatter plot size for this voxel count
        # Using the same formula as the scatter plots: sizes = 50 + 200 * (x - min) / (max - min)
        scatter_size = 50 + 200 * (voxel_count - voxel_min) / (voxel_max - voxel_min)

        # Convert scatter size to Line2D markersize
        # Scatter 's' parameter is area, Line2D markersize is diameter
        markersize = np.sqrt(scatter_size) * 1.0

        size_legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='gray', markeredgecolor='black',
                                         markersize=markersize, label=label))

    # Place size legend centered at the bottom with 5 columns as requested
    # Maintain Y position at 0.20 as in original
    fig.legend(handles=size_legend_elements, loc='center',
              bbox_to_anchor=(0.5, 0.20), title='Lesion size',
              fontsize=10, title_fontsize=10, ncol=5)

    # Adjust layout to accommodate colorbars and legends
    plt.tight_layout(rect=[0, 0.14, 0.5, 0.90])

    # Save figure
    output_path = os.path.join(output_dir, 'Extended_Data_Fig_6.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nExtended_Data_Fig_6 saved to: {output_path}")

    # Also save as SVG
    output_path_svg = os.path.join(output_dir, 'Extended_Data_Fig_6.svg')
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Extended_Data_Fig_6 saved to: {output_path_svg}")


if __name__ == '__main__':
    main()
