# UCI + imodels Benchmark

This project contains a configurable benchmark script, `uciml.py`, for
classification on UCI-ML-Repo datasets with `imodels`.

## What the script does

- loads configurable UCI datasets (default IDs: `17` and `45`)
- uses original targets by default (for example, `heart_disease` stays multiclass)
- iterates sequentially over configurable `imodels` algorithms
- chooses train/test split based on dataset size
- computes F1 score and a best-effort model size
- supports repeated runs with different seeds and aggregates mean/std/95% CI
- optionally runs pairwise significance checks per dataset (over run seeds)
- saves raw, aggregated, and plot-ready results as CSV
- creates either one combined figure or two separate figures
- uses short plot labels by default for known datasets (for example, `BreastCancer`, `Heart`)
- auto-generates short plot labels for unknown datasets
- shows a clear error message if an algorithm does not support multiclass

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python uciml.py --no-show
```

Repeated runs with 95% CI error bars:

```bash
python uciml.py --n-runs 5 --error-bars ci95 --no-show
```

With optional significance check:

```bash
python uciml.py --n-runs 5 --error-bars ci95 --significance-check --alpha 0.05 --no-show
```

With short labels for datasets in plots:

```bash
python uciml.py --dataset-short-names "17:BreastCancer,45:Heart" --no-show
```

Mapping can also be done by dataset name:

```bash
python uciml.py --dataset-short-names "breast_cancer_wisconsin_diagnostic:BCWD,heart_disease:Heart" --no-show
```

Create one plot from multiple exported plot-data CSVs:

```bash
python plot_uciml_csvs.py --input-csvs "run_a/uci_imodels_plot_data.csv,run_b/uci_imodels_plot_data.csv" --plot-mode combined --plot-style dots --error-bars ci95 --no-show --output-dir merged_results
```

## Important options

- `--dataset-ids "17,45"`
- `--algorithms "SlipperClassifier,GreedyRuleListClassifier,C45TreeClassifier,GreedyTreeClassifier"`
- `--plot-mode combined` or `--plot-mode separate`
- `--plot-style dots|bars` (`dots` = Dot-Whisker, default)
- `--output-dir results`
- `--random-state 42`
- `--n-runs 5` (for example, 5 seeds: `random_state + i`)
- `--error-bars none|std|ci95` (default: `std`)
- `--dataset-short-names "17:BreastCancer,heart_disease:Heart"` (plot labels only)
  - Without this option, default short labels are used for known datasets.
  - Unknown datasets get an auto-generated short label.
  - You can override defaults by dataset ID or dataset name.
- `--significance-check`
- `--alpha 0.05`
- `--no-show` (save files only, no GUI window)

## Output

In the target directory (default `results`) the script creates:

- `uci_imodels_results.csv`
- `uci_imodels_results_agg.csv`
- `uci_imodels_plot_data.csv` (stable schema for downstream plotting)
- optional with `--significance-check`: `uci_imodels_significance.csv`
- for `combined`: `uci_imodels_combined.png`
- for `combined`: additionally `uci_imodels_combined.pdf`
- for `separate`: `uci_imodels_f1.png`, `uci_imodels_model_size.png`
- for `separate`: additionally `uci_imodels_f1.pdf` and `uci_imodels_model_size.pdf`

## Separate plotting script

Use `plot_uciml_csvs.py` to combine one or more `uci_imodels_plot_data.csv` files and create a merged plot.

- Dataset overlap across files is allowed.
- Duplicate `dataset_id + algorithm` rows across files are rejected.
- By workflow convention, each algorithm should appear in only one source CSV.

Main options:

- `--input-csvs "a.csv,b.csv"` (required)
- `--plot-mode combined|separate`
- `--plot-style dots|bars` (default: `dots`)
- `--error-bars none|std|ci95` (default: `std`)
- `--output-dir merged_results`
- `--no-show`

The default `dots` style renders a lighter dot-whisker plot (mean as point, uncertainty as horizontal error bars), which is usually easier to read than grouped bars for already-aggregated benchmark data.

In `combined` mode, the two metrics are arranged vertically and share a dedicated legend area on the right, which is usually better suited for paper figures than side-by-side panels with an in-plot legend.

Outputs of `plot_uciml_csvs.py`:

- `merged_plot_data.csv`
- `merged_ucimodels_combined.png` and `merged_ucimodels_combined.pdf`, or separate plot files in both formats

