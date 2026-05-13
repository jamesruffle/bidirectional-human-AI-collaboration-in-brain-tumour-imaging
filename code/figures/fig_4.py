#!/usr/bin/env python3
"""Manuscript Figure 4 - Performance curves with/without support (self-contained).

Single self-contained reproduction script. Reads three small CSVs from
`data/source_data/figure_4/csv/` and produces `Fig_4.png` and `Fig_4.svg`
pixel-identical to the canonical reference.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score,
)

sns.set_palette("husl")

HERE = os.path.dirname(os.path.abspath(__file__))
R1_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(R1_ROOT, 'data', 'source_data', 'figure_4', 'csv')
FIGURES_OUTPUT_PATH = os.path.join(R1_ROOT, 'data', 'figures')


def main():
    for fname in ('model_per_case_scores.csv',
                  'cv_combined_predictions.csv',
                  'radiologist_reviews_minimal.csv'):
        p = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(p):
            print(f"ERROR: required CSV missing at {p}", file=sys.stderr)
            sys.exit(2)

    os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)

    model_df = pd.read_csv(os.path.join(SRC_DIR, 'model_per_case_scores.csv'),
                           float_precision='round_trip')
    cv_df = pd.read_csv(os.path.join(SRC_DIR, 'cv_combined_predictions.csv'),
                        float_precision='round_trip')
    radiologist_df = pd.read_csv(os.path.join(SRC_DIR, 'radiologist_reviews_minimal.csv'),
                                 float_precision='round_trip')

    prob_data_dict = dict(zip(model_df['case_id'], model_df['model_max_prob']))
    best_cv_predictions = cv_df.to_dict(orient='records')

    fig3 = plt.figure(figsize=(16, 24))

    # Add overall title
    fig3.suptitle('Performance curves with/without support', fontsize=16, y=0.92)  # Adjusted y position

    # Define the grid layout - 4 rows, 2 columns - reduced wspace for closer columns
    gs = fig3.add_gridspec(4, 2, hspace=0.25, wspace=0.2)

    ax1 = fig3.add_subplot(gs[0, 0])
    ax2 = fig3.add_subplot(gs[0, 1])
    ax3 = fig3.add_subplot(gs[1, 0])
    ax4 = fig3.add_subplot(gs[1, 1])
    ax5 = fig3.add_subplot(gs[2, 0])
    ax6 = fig3.add_subplot(gs[2, 1])
    ax7 = fig3.add_subplot(gs[3, 0])
    ax8 = fig3.add_subplot(gs[3, 1])

    # Define reference color palette
    reference_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Panel a) - Model ROC Curve
    common_cases = radiologist_df.dropna(subset=['model_predicted_enhancement'])

    # Per-case maximum ET probability is loaded from the bundled CSV — there
    # is no NIfTI-volume loader in this reproduction script.
    unique_cases = common_cases['case_id'].unique()
    print(f"Using pre-extracted prob_data_dict ({len(prob_data_dict)} cases)")

    # Create y_scores using UNIQUE CASES (aggregate multiple radiologist views)
    y_scores = []
    y_true_filtered = []

    # Aggregate by unique case
    for case_id in unique_cases:
        if case_id in prob_data_dict:
            y_scores.append(prob_data_dict[case_id])
            # Get ground truth for this case (same across all radiologist views)
            case_rows = common_cases[common_cases['case_id'] == case_id]
            if len(case_rows) > 0:
                y_true_filtered.append(case_rows.iloc[0]['has_enhancement_gt'])

    y_scores = np.array(y_scores)
    y_true = np.array(y_true_filtered)

    # Calculate ROC curve (unique-case scores; canonical model AUROC for the manuscript)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"  Model alone AUROC (unique-case, n={len(y_true)}): {roc_auc:.3f}")

    # Plot ROC curve
    ax1.plot(fpr, tpr, color='black', linewidth=3,
            label=f'Model (AUROC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('False positive rate', fontsize=12)
    ax1.set_ylabel('True positive rate', fontsize=12)
    ax1.set_title('a) Model alone - ROC', fontsize=14)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)


    # Calculate model accuracy for comparison
    model_accuracy = accuracy_score(common_cases['has_enhancement_gt'],
                                  common_cases['model_predicted_enhancement'])
    print(f"  Model accuracy: {model_accuracy:.3f}")

    # Panel b) - Model Precision-Recall Curve
    # Original PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Remove duplicate recall values and corresponding precision values
    # This helps prevent the jagged appearance in the curve
    unique_indices = np.where(np.diff(recall) != 0)[0]
    unique_indices = np.append(unique_indices, len(recall) - 1)
    recall_unique = recall[unique_indices]
    precision_unique = precision[unique_indices]

    # Calculate AUC using the original sklearn implementation
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"  Model alone AUPRC (unique-case, n={len(y_true)}): {pr_auc:.3f}")

    # Panel b) - CV-Optimized Model ROC Curve (Model with radiologist support)
    cv_predictions = best_cv_predictions

    # Extract ground truth and CV-optimized predictions with probabilities
    cv_y_true = []
    cv_y_scores = []

    # Create a mapping from case_id to mean probability for CV predictions
    # Using mean across radiologists is more representative than maximum
    case_to_cv_probs = {}
    for pred in cv_predictions:
        case_id = pred['case_id']
        gt = pred['gt']
        cv_prob = pred.get('combined_prob', pred.get('model_prob', 0.5))  # Use combined probability if available

        if case_id not in case_to_cv_probs:
            case_to_cv_probs[case_id] = {'gt': gt, 'probs': []}
        case_to_cv_probs[case_id]['probs'].append(cv_prob)

    # Extract unique cases for ROC curve using mean probability
    for case_id, data in case_to_cv_probs.items():
        cv_y_true.append(data['gt'])
        cv_y_scores.append(np.mean(data['probs']))  # Use mean instead of max

    cv_y_true = np.array(cv_y_true)
    cv_y_scores = np.array(cv_y_scores)

    # Calculate ROC curve for CV-optimized predictions (unique-case mean combined_prob;
    # canonical model+rad AUROC for the manuscript)
    cv_fpr, cv_tpr, _ = roc_curve(cv_y_true, cv_y_scores)
    cv_roc_auc = auc(cv_fpr, cv_tpr)
    print(f"  Model+Radiologist (CV) AUROC (unique-case, n={len(cv_y_true)}): {cv_roc_auc:.3f}")

    # Plot CV-optimized ROC curve (using black color to match panel a)
    ax2.plot(cv_fpr, cv_tpr, color='black', linewidth=3,
            label=f'Model with support (AUROC = {cv_roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('False positive rate', fontsize=12)
    ax2.set_ylabel('True positive rate', fontsize=12)
    ax2.set_title('b) Model with radiologist support - ROC', fontsize=14)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # If we have very few unique scores (likely binary predictions), plot step function
    if len(np.unique(y_scores)) <= 2:
        print("  WARNING: y_scores appear to be binary, not continuous probabilities")
        ax3.step(recall, precision, where='post', color='black', linewidth=3,
                label=f'Model (AUPRC = {pr_auc:.3f})')
    else:
        ax3.plot(recall_unique, precision_unique, color='black', linewidth=3,
                label=f'Model (AUPRC = {pr_auc:.3f})')
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('c) Model alone - PRC', fontsize=14)
    ax3.legend(fontsize=11, loc='lower left')
    ax3.grid(True, alpha=0.3)

    # The bimodal score distribution (concentrated near 0 and near 1) produces
    # the characteristic stepwise shape of the PR curve in this regime.

    # Panel d) - CV-Optimized Model PR Curve (Model with radiologist support)
    # Calculate precision-recall curve for CV-optimized predictions
    cv_precision, cv_recall, _ = precision_recall_curve(cv_y_true, cv_y_scores)

    # Remove duplicate recall values for smoother curve
    cv_unique_indices = np.where(np.diff(cv_recall) != 0)[0]
    cv_unique_indices = np.append(cv_unique_indices, len(cv_recall) - 1)
    cv_recall_unique = cv_recall[cv_unique_indices]
    cv_precision_unique = cv_precision[cv_unique_indices]

    # Calculate AUC
    cv_pr_auc = average_precision_score(cv_y_true, cv_y_scores)
    print(f"  Model+Radiologist (CV) AUPRC (unique-case, n={len(cv_y_true)}): {cv_pr_auc:.3f}")

    # Plot PR curve (using black color to match panel c)
    if len(np.unique(cv_y_scores)) <= 2:
        # Binary predictions - use step function
        ax4.step(cv_recall, cv_precision, where='post', color='black', linewidth=3,
                label=f'Model with support (AUPRC = {cv_pr_auc:.3f})')
    else:
        # Continuous predictions - use smooth line
        ax4.plot(cv_recall_unique, cv_precision_unique, color='black', linewidth=3,
                label=f'Model with support (AUPRC = {cv_pr_auc:.3f})')

    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('d) Model with radiologist support - PRC', fontsize=14)
    ax4.legend(fontsize=11, loc='lower left')
    ax4.grid(True, alpha=0.3)

    # Panel e) & f) - Without Model Performance
    without_seg_data = radiologist_df[radiologist_df['with_segmentation'] == False].copy()

    # Panel e) - ROC curves without model
    radiologist_aucs = []
    num_radiologists = without_seg_data['radiologist'].nunique()
    colors = [reference_colors[i % len(reference_colors)] for i in range(num_radiologists)]

    # Individual radiologist curves
    for i, radiologist in enumerate(without_seg_data['radiologist'].unique()):
        rad_data = without_seg_data[without_seg_data['radiologist'] == radiologist]
        if len(rad_data) > 5:
            y_true = rad_data['has_enhancement_gt']
            # Map confidence onto the predicted-class probability axis
            y_scores = []
            for _, row in rad_data.iterrows():
                if row['predicted_enhancement'] == 1:
                    y_scores.append(row['confidence'] / 10.0)
                else:
                    y_scores.append(1.0 - row['confidence'] / 10.0)

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            radiologist_aucs.append(roc_auc)

            ax5.plot(fpr, tpr, color=colors[i], linewidth=1, alpha=0.6, linestyle='--')

    # Overall cohort ROC
    overall_y_true = without_seg_data['has_enhancement_gt']
    # Map confidence onto the predicted-class probability axis
    overall_y_scores = []
    for _, row in without_seg_data.iterrows():
        if row['predicted_enhancement'] == 1:
            overall_y_scores.append(row['confidence'] / 10.0)
        else:
            overall_y_scores.append(1.0 - row['confidence'] / 10.0)
    overall_fpr, overall_tpr, _ = roc_curve(overall_y_true, overall_y_scores)

    # Reader-averaged AUROC (MRMC FOM) — matches fig_1 mrmc_auroc.fom_without and the
    # manuscript-cited value. The pooled "overall" trace is still drawn as the visual
    # mean curve, but the reported AUROC is the MRMC reader-average.
    rad_alone_auroc = float(np.mean(radiologist_aucs))
    print(f"  Radiologist alone AUROC (reader-averaged, n={len(radiologist_aucs)}): {rad_alone_auroc:.3f}")

    ax5.plot(overall_fpr, overall_tpr, color='black', linewidth=2.5, alpha=0.9,
            label=f'Mean radiologist performance (AUROC = {rad_alone_auroc:.3f})')

    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.8)
    ax5.set_xlabel('False positive rate', fontsize=12)
    ax5.set_ylabel('True positive rate', fontsize=12)
    ax5.set_title('e) Radiologist alone - ROC', fontsize=14)
    ax5.legend(fontsize=11, loc='lower right')
    ax5.grid(True, alpha=0.3)


    # Calculate and print accuracy for comparison
    accuracy_without = (without_seg_data['predicted_enhancement'] == without_seg_data['has_enhancement_gt']).mean()
    print(f"  Overall accuracy: {accuracy_without:.3f}")

    # Panel f) - Precision-Recall curves without model
    individual_auprcs_without = []
    for i, radiologist in enumerate(without_seg_data['radiologist'].unique()):
        rad_data = without_seg_data[without_seg_data['radiologist'] == radiologist]
        if len(rad_data) > 5:
            y_true = rad_data['has_enhancement_gt']
            # Map confidence onto the predicted-class probability axis
            y_scores = []
            for _, row in rad_data.iterrows():
                if row['predicted_enhancement'] == 1:
                    y_scores.append(row['confidence'] / 10.0)
                else:
                    y_scores.append(1.0 - row['confidence'] / 10.0)

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            individual_auprcs_without.append(pr_auc)

            ax7.plot(recall, precision, color=colors[i], linewidth=1, alpha=0.6, linestyle='--')

    # Overall cohort PR curve
    overall_y_true = without_seg_data['has_enhancement_gt']
    # Map confidence onto the predicted-class probability axis
    overall_y_scores = []
    for _, row in without_seg_data.iterrows():
        if row['predicted_enhancement'] == 1:
            overall_y_scores.append(row['confidence'] / 10.0)
        else:
            overall_y_scores.append(1.0 - row['confidence'] / 10.0)
    overall_precision, overall_recall, _ = precision_recall_curve(overall_y_true, overall_y_scores)

    rad_alone_auprc = float(np.mean(individual_auprcs_without))
    print(f"  Radiologist alone AUPRC (reader-averaged, n={len(individual_auprcs_without)}): {rad_alone_auprc:.3f}")

    ax7.plot(overall_recall, overall_precision, color='black', linewidth=2.5, alpha=0.9,
            label=f'Mean radiologist performance (AUPRC = {rad_alone_auprc:.3f})')

    ax7.set_xlabel('Recall', fontsize=12)
    ax7.set_ylabel('Precision', fontsize=12)
    ax7.set_title('g) Radiologist alone - PRC', fontsize=14)
    ax7.legend(fontsize=11, loc='lower left')
    ax7.grid(True, alpha=0.3)

    # Panel i) & j) - With Model Performance
    with_seg_data = radiologist_df[radiologist_df['with_segmentation'] == True].copy()

    # Panel i) - ROC curves with model
    radiologist_aucs_with = []

    # Individual radiologist curves
    for i, radiologist in enumerate(with_seg_data['radiologist'].unique()):
        rad_data = with_seg_data[with_seg_data['radiologist'] == radiologist]
        if len(rad_data) > 5:
            y_true = rad_data['has_enhancement_gt']
            # Map confidence onto the predicted-class probability axis
            y_scores = []
            for _, row in rad_data.iterrows():
                if row['predicted_enhancement'] == 1:
                    y_scores.append(row['confidence'] / 10.0)
                else:
                    y_scores.append(1.0 - row['confidence'] / 10.0)

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            radiologist_aucs_with.append(roc_auc)

            ax6.plot(fpr, tpr, color=colors[i], linewidth=1, alpha=0.6, linestyle='--')

    # Overall cohort ROC
    overall_y_true = with_seg_data['has_enhancement_gt']
    # Map confidence onto the predicted-class probability axis
    overall_y_scores = []
    for _, row in with_seg_data.iterrows():
        if row['predicted_enhancement'] == 1:
            overall_y_scores.append(row['confidence'] / 10.0)
        else:
            overall_y_scores.append(1.0 - row['confidence'] / 10.0)
    overall_fpr, overall_tpr, _ = roc_curve(overall_y_true, overall_y_scores)

    rad_with_auroc = float(np.mean(radiologist_aucs_with))
    print(f"  Radiologist+Model AUROC (reader-averaged, n={len(radiologist_aucs_with)}): {rad_with_auroc:.3f}")

    ax6.plot(overall_fpr, overall_tpr, color='black', linewidth=2.5, alpha=0.9,
            label=f'Mean radiologist performance (AUROC = {rad_with_auroc:.3f})')

    ax6.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.8)
    ax6.set_xlabel('False positive rate', fontsize=12)
    ax6.set_ylabel('True positive rate', fontsize=12)
    ax6.set_title('f) Radiologist with model support - ROC', fontsize=14)
    ax6.legend(fontsize=11, loc='lower right')
    ax6.grid(True, alpha=0.3)


    # Calculate and print accuracy for comparison
    accuracy_with = (with_seg_data['predicted_enhancement'] == with_seg_data['has_enhancement_gt']).mean()
    print(f"  Overall accuracy: {accuracy_with:.3f}")

    # Panel j) - Precision-Recall curves with model
    individual_auprcs_with = []
    for i, radiologist in enumerate(with_seg_data['radiologist'].unique()):
        rad_data = with_seg_data[with_seg_data['radiologist'] == radiologist]
        if len(rad_data) > 5:
            y_true = rad_data['has_enhancement_gt']
            # Map confidence onto the predicted-class probability axis
            y_scores = []
            for _, row in rad_data.iterrows():
                if row['predicted_enhancement'] == 1:
                    y_scores.append(row['confidence'] / 10.0)
                else:
                    y_scores.append(1.0 - row['confidence'] / 10.0)

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            individual_auprcs_with.append(pr_auc)

            ax8.plot(recall, precision, color=colors[i], linewidth=1, alpha=0.6, linestyle='--')

    # Overall cohort PR curve
    overall_y_true = with_seg_data['has_enhancement_gt']
    # Map confidence onto the predicted-class probability axis
    overall_y_scores = []
    for _, row in with_seg_data.iterrows():
        if row['predicted_enhancement'] == 1:
            overall_y_scores.append(row['confidence'] / 10.0)
        else:
            overall_y_scores.append(1.0 - row['confidence'] / 10.0)
    overall_precision, overall_recall, _ = precision_recall_curve(overall_y_true, overall_y_scores)

    rad_with_auprc = float(np.mean(individual_auprcs_with))
    print(f"  Radiologist+Model AUPRC (reader-averaged, n={len(individual_auprcs_with)}): {rad_with_auprc:.3f}")

    ax8.plot(overall_recall, overall_precision, color='black', linewidth=2.5, alpha=0.9,
            label=f'Mean radiologist performance (AUPRC = {rad_with_auprc:.3f})')

    ax8.set_xlabel('Recall', fontsize=12)
    ax8.set_ylabel('Precision', fontsize=12)
    ax8.set_title('h) Radiologist with model support - PRC', fontsize=14)
    ax8.legend(fontsize=11, loc='lower left')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save Figure 4
    fig_4_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_4.png')
    plt.savefig(fig_4_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fig_4 saved to: {fig_4_path}")

    # Save as SVG
    fig_4_svg_path = os.path.join(FIGURES_OUTPUT_PATH, 'Fig_4.svg')
    plt.savefig(fig_4_svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Fig_4 saved to: {fig_4_svg_path}")


if __name__ == '__main__':
    main()
