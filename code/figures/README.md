# Per-figure reproducibility package

Per Nature Communications source-data requirements, every script-reproducible figure in the manuscript has a corresponding standalone Python script in this directory. Each script reads only from CSVs (plus, for Fig 6, a small JSON RNG-state file) under `data/source_data/`, applies the same plotting code that produced the published figure, and writes outputs to `data/figures/`. Numerical values reported in figure captions are printed to stdout.

All shared analysis logic — per-reader metrics, optimistic-dedup, case-level ensembles, calibration metrics, equivalent-experience regression, NHS salary integration, paired-agreement Cohen's κ — lives in `code/_metrics_utils.py`. Both `code/figures/` and `code/tables/` import from this module so every figure and table number derives from the same primary CSVs by the same code path.

## Figure inventory

| Manuscript figure | Title | Script | Output | Status |
|---|---|---|---|---|
| **Fig 1** | Impact of support on agent performance | `fig_1.py` | `Fig_1.{png,svg}` | reproducible from `radiologist_df.csv` + `best_cv_predictions.csv` + `seed_predictions.csv` |
| **Fig 2** | Enhancing patient brain images | (none — patient NIfTI imaging only) | — | not script-reproducible (controlled-access imaging) |
| **Fig 3** | Nonenhancing patient brain images | (none — patient NIfTI imaging only) | — | not script-reproducible (controlled-access imaging) |
| **Fig 4** | Performance curves with/without support | `fig_4.py` | `Fig_4.{png,svg}` | reproducible |
| **Fig 5** | Strengthening accuracy / experience / confidence | `fig_5.py` | `Fig_5.{png,svg}` | reproducible from `radiologist_df.csv` (live `groupby`) |
| **Fig 6** | Enhancing healthcare value | `fig_6.py` | `Fig_6.{png,svg}` | reproducible from `radiologist_df.csv` + `salary_progression.csv` (live regression) |
| **EDF 1** | Paradigms of evaluating AI value | (none — illustrator schematic) | — | not script-reproducible |
| **EDF 2** | Human-AI collaboration paradigms | (none — illustrator schematic) | — | not script-reproducible |
| **EDF 3** | Study schematic | (none — illustrator schematic) | — | not script-reproducible |
| **EDF 4** | Agreement comparisons | `extended_data_fig_4.py` | `Extended_Data_Fig_4.{png,svg}` | reproducible from `radiologist_df.csv` |
| **EDF 5** | Performance by pathology dataset | `extended_data_fig_5.py` | `Extended_Data_Fig_5.{png,svg}` | reproducible from `radiologist_df.csv` + `edf5_radiologist_meta.csv` |
| **EDF 6** | Effect of lesion morphology (UMAP) | `extended_data_fig_6.py` | `Extended_Data_Fig_6.{png,svg}` | reproducible from `umap_analysis_results.csv` (UMAP coordinates pre-computed; UMAP is non-deterministic across machines and is therefore bundled rather than re-run) |
| **EDF 7** | Pathology, size, radiomic assessment | `extended_data_fig_7.py` | `Extended_Data_Fig_7.{png,svg}` | reproducible |

## Reproducing all figures

The repo includes a parallel runner that runs every figure + table script concurrently and writes a timed log under `data/logs/<script>.log` for each one (header timestamp + final `Wall clock: Ns` line):

```bash
pip install -r requirements.txt   # numpy, matplotlib, pandas, Pillow, scikit-learn, scipy, seaborn, statsmodels
bash code/run_all.sh figures      # all 8 figure scripts in parallel
bash code/run_all.sh tables       # all 4 table scripts in parallel
bash code/run_all.sh              # everything in parallel (default)
```

To run a single script directly:

```bash
python3 -W ignore code/figures/fig_1.py 2>&1 | tee data/logs/fig_1.log
```

Each script is self-contained: data-loading and plotting live in the same file (no `runpy` indirection, no sibling plotting-block helpers).

## Reference runtimes (128-CPU host, parallel run)

The bundled `data/logs/<script>.log` files document the wall-clock runtime of each script measured on a 128-core host. Most scripts are I/O-bound (a few seconds); the bootstrap-heavy scripts dominate the longest path.

| Script | Wall clock | Why |
|---|---|---|
| `fig_4.py` | ~3 s | Pre-computed CV ensemble + radiomic CSVs; mostly plotting |
| `fig_5.py` | ~3 s | Live `groupby` over 2,200 reviews; small plotting overhead |
| `fig_1.py` | ~45 s | 6 reader-level + 6 case-level bootstrap CIs (B=5,000); pair-level model bootstrap |
| `fig_6.py` | ~15 s | Live confidence-calibration + equivalent-experience regression + bootstrap CIs |
| `extended_data_fig_4.py` | ~15 min | 5 bootstrap loops × 5,000 iters of Cohen's κ (the longest path) |
| `extended_data_fig_5.py` | ~5 s | Per-pathology stratified plotting |
| `extended_data_fig_6.py` | ~15 s | UMAP coordinates pre-computed; matplotlib only |
| `extended_data_fig_7.py` | ~3 s | Pre-binned aggregates |
| `table_1.py` | ~40 s | Reader-level + case-level paired bootstrap deltas across 7 metrics |
| `supplementary_table_*.py` | <1 s | Direct CSV groupby |
| **Parallel total** (`bash code/run_all.sh`) | **~15 min** (bounded by EDF 4) | All 12 scripts in parallel; CPU-bound with no contention on a multi-core host |

## Source-data dependencies

| Script | Reads from | Notes |
|---|---|---|
| `fig_1.py` | `source_data/figure_1/csv_v2/` | `radiologist_df.csv`, `best_cv_predictions.csv` (+ schema), `seed_predictions.csv`, `aggregates.json`. Per-reader `group_metrics` and pairwise Cohen's κ `paired_agreements` derived live via `_metrics_utils.compute_group_metrics` / `compute_paired_agreements`. Pair-level and case-level model metrics derived live via `_metrics_utils.load_canonical_metrics(seed_predictions.csv)`. `aggregates.json` retained only for the reader-averaged R MRMCaov outputs (`group1_avg`, `group2_avg`, `mrmc_*`) which come from the external R pipeline. |
| `fig_4.py` | `source_data/figure_4/csv/` | 3 inputs: `model_per_case_scores.csv`, `cv_combined_predictions.csv` (5-seed mean-prob ensemble), `radiologist_reviews_minimal.csv`. |
| `fig_5.py` | `source_data/figure_1/csv_v2/radiologist_df.csv` | Experience summary and confidence-calibration bins computed live via `groupby`. No cached intermediates. |
| `fig_6.py` | `source_data/figure_6/csv/` (+ `radiologist_df.csv`) | 4 inputs: `cv_predictions_min.csv`, `model_case_confidence.csv`, `pair_level_metrics.json`, `salary_progression.csv`, plus the canonical `radiologist_df.csv` from `figure_1/csv_v2/`. Individual radiologist performance, confidence calibration (Q3/Q1 quartile split), equivalent-experience regression, and NHS salary integration are all computed live via `_metrics_utils.compute_individual_perf` / `compute_confidence_analysis` / `compute_equiv` / `compute_financial`. The `figure_6_rng_state.json` file (~5 KB) restores the matplotlib jitter RNG state so scatter dot positions match the published figure. |
| `extended_data_fig_4.py` | `source_data/figure_1/csv_v2/radiologist_df.csv` | Reads the canonical primary frame and computes all agreement statistics live. No cached intermediates. |
| `extended_data_fig_5.py` | `source_data/extended_data_figure_5/csv/edf5_radiologist_meta.csv` (+ `radiologist_df.csv`) | Per-pathology arm × radiologist accuracy aggregate computed live; radiologist subspecialty mapping kept as the lone cached metadata file. |
| `extended_data_fig_6.py` | `source_data/extended_data_figure_6/csv/umap_analysis_results.csv` | UMAP is stochastic across machines/library versions; the 564-case 2D embedding is pre-computed once and bundled. |
| `extended_data_fig_7.py` | `source_data/extended_data_figure_7/edf7_inputs.csv` | Pathology / lesion-volume / radiomic-category bins (5 categories each). |

## Brain-image figures (Fig 2, Fig 3)

Composed from T1 / T2 / FLAIR / segmentation NIfTI volumes from the held-out test set. NIfTI imaging is held under controlled access (the institutional ethics framework precludes redistribution beyond the original consent scope) and is therefore not bundled. Reviewers needing to regenerate these figures can obtain source datasets from the public sources cited in the manuscript.

## Schematic figures (EDF 1, EDF 2, EDF 3)

Illustrator-rendered figures (paradigm / collaboration / study-design diagrams) — not reproducible from data. The published renditions are embedded in the manuscript Word document; no per-figure script ships for these.

## How each figure is reproduced

Each `code/figures/<script>.py` is self-contained:

1. Reads the bundled inputs (CSV / JSON) under `data/source_data/<dir>/`. Where required for byte-identical scatter-jitter reproduction (Fig 6 only), a small JSON RNG-state file is also bundled and restored at runtime.
2. Computes derived intermediates live via `_metrics_utils` from the upstream primary CSVs.
3. Renders the figure in-process using matplotlib.
4. Writes a PNG + SVG to `data/figures/` using the manuscript-aligned filenames (`Fig_<N>.{png,svg}`, `Extended_Data_Fig_<N>.{png,svg}`).
