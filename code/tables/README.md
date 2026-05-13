# Per-table reproducibility package

Companion to `code/figures/`. Per Nature Communications source-data requirements, every numerical table in the manuscript and supplementary material has a corresponding standalone Python script in this directory. Each script is self-contained, reads only from CSVs under `data/source_data/`, and writes a CSV under `data/source_data/<table_dir>/csv/` plus a captured log under `data/logs/`.

All shared analysis logic — per-reader metrics, optimistic-dedup, case-level ensembles, paired bootstrap deltas — lives in `code/_metrics_utils.py`. Every table cell value derives from the same primary CSVs (`radiologist_df.csv`, `best_cv_predictions.csv`, `seed_predictions.csv`, `sex_metadata.csv`) by the same code path used by the figure scripts.

## Table inventory

| Table | Title | Script | CSV output | Log |
|---|---|---|---|---|
| **Main Table 1** | Primary results — paired comparisons of agent performance on the 564-case radiologist-reviewed cohort | `table_1.py` | `data/source_data/table_1/csv/table_1.csv` | `data/logs/table_1.log` |
| **Supp Table 1** | Reader subspecialty roster | (none — static reader→subspecialty lookup) | — | — |
| **Supp Table 2** | False-positive and false-negative geography and pathology distribution | `supplementary_table_2.py` | `data/source_data/supplementary_table_2/csv/supplementary_table_2.csv` | `data/logs/supplementary_table_2.log` |
| **Supp Table 3** | Patient-sex disaggregation of agent performance on the 82-case sex-metadata subset | `supplementary_table_3.py` | `data/source_data/supplementary_table_3/csv/supplementary_table_3.csv` | `data/logs/supplementary_table_3.log` |
| **Supp Table 4** | Female–male performance gaps (Δ = female − male) across the 82-case subset | `supplementary_table_4.py` | `data/source_data/supplementary_table_4/csv/supplementary_table_4.csv` | `data/logs/supplementary_table_4.log` |

## Reproducing all tables

The repo's parallel runner handles all four tables (and the eight figure scripts in `code/figures/`) concurrently:

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, scikit-learn, statsmodels
bash code/run_all.sh tables       # all 4 table scripts in parallel
```

Or run a single table directly:

```bash
python3 -W ignore code/tables/table_1.py 2>&1 | tee data/logs/table_1.log
```

Each log header records the start timestamp and host CPU count; the trailing `Wall clock: Ns` line records the run duration. See the bundled `data/logs/*.log` files.

## Reference runtimes (128-CPU host)

| Script | Wall clock | Why |
|---|---|---|
| `supplementary_table_2.py` | <1 s | Country × pathology groupby with ≥10-case threshold |
| `supplementary_table_3.py` | <1 s | Per-sex `compute_sex_metrics` aggregation on the 82-case sex-metadata subset |
| `supplementary_table_4.py` | <1 s | Reuses `supplementary_table_3.compute_sex_metrics` |
| `table_1.py` | ~40 s | Reader-level + case-level paired bootstrap deltas across all 7 metric rows (B=5,000) |

## Source-data dependencies

| Script | Reads from | Notes |
|---|---|---|
| `table_1.py` | `source_data/figure_1/csv_v2/` | `radiologist_df.csv`, `best_cv_predictions.csv`, `seed_predictions.csv`. Per-reader `group_metrics` derived live via `_metrics_utils.compute_group_metrics`; pair-level and case-level model metrics via `_metrics_utils.load_canonical_metrics(seed_predictions.csv)`. Same code path as `fig_1.py`, ensuring consistency: every Table 1 cell matches the corresponding Fig 1 panel. |
| `supplementary_table_2.py` | `source_data/figure_1/csv_v2/` (radiologist_df.csv, best_cv_predictions.csv) | Country × Pathology subgroup grouping with ≥10-case-per-arm threshold. |
| `supplementary_table_3.py` | `source_data/figure_1/csv_v2/` (+ `sex_metadata.csv`) | Per-sex point values for each agent × condition. Defines `compute_sex_metrics()` reused by `supplementary_table_4.py`. |
| `supplementary_table_4.py` | imports `supplementary_table_3.compute_sex_metrics` | Δ = female − male per metric × condition, computed from `supplementary_table_3.py`'s per-sex dict. |

## Bootstrap procedures used in `table_1.py`

| Quantity | Procedure |
|---|---|
| Δ AUROC / AUPRC (Human side) | Reader-level paired bootstrap on the n=11 readers; B=5,000; seed=20260505; percentile method |
| Δ AUROC / AUPRC (AI side) | Case-level paired bootstrap on n=564 unique cases; B=5,000; seed=20260505 |
| Δ accuracy-family (Human side) | Reader-level paired bootstrap on n=11 readers; B=5,000; seed=20260505 |
| Δ accuracy-family (AI side) | Fold-Δ bootstrap of per-fold Δ across the 5 CV folds; B=5,000; seed=20260505; CI centre-shifted to the canonical macro-averaged Δ |
| Cohen's κ contrasts | Pair-level bootstrap on the 2,389 pairwise comparisons (1,289 rad-rad + 1,100 rad-model); B=5,000; seed=20260505 |
| Point CIs (radiologist accuracy-family) | Normal approximation: mean ± 1.96 × (per-reader SD / √n_readers) |
| Point CIs (radiologist AUROC/AUPRC/sens/spec) | MRMCaov ORH variance estimate (via the figure-rendering pipeline; values cached in source-data CSVs) |
| Point CIs (model accuracy-family) | Reviewer-case-pair-level percentile bootstrap (n=1,100); B=5,000; seed=20260505 |
| Point CIs (model AUROC/AUPRC) | Case-level percentile bootstrap (n=564); B=5,000; seed=20260505 |

## Verification

- **Supp Tables 2–4**: every script-produced value matches the corresponding manuscript cell to within 3-decimal rounding. Supp Table 1 (reader subspecialty roster) is a static metadata table reproduced verbatim from the trial recruitment record.
- **Main Table 1** (`table_1.py`): every point estimate and 95% CI matches the published Table 1 cell-for-cell. The sampling unit and bootstrap procedure used to generate each cell are listed in the "Bootstrap procedures used in `table_1.py`" table above.
- **Calibration / Inter-rater agreement / Mean confidence / RTAT** rows of the published Table 1 are sourced from `fig_1.py`, `extended_data_fig_4.py`, and `fig_6.py` (each with its own bootstrap CIs at the appropriate sampling unit). `table_1.py` covers only the AUROC / AUPRC / accuracy-family rows; the calibration- and agreement-row values are produced by the figure scripts and persist there.
