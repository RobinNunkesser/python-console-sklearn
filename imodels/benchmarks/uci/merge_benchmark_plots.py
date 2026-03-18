"""Create benchmark plots from one or more exported plot-data CSV files.

Expected input schema is produced by `run_imodels_benchmark.py` as
`uci_imodels_plot_data.csv`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.shared_plotting import UCI_METRICS, plot_benchmark_results

REQUIRED_COLUMNS = {
    "dataset_id",
    "dataset",
    "algorithm",
    "f1_mean",
    "model_size_mean",
}


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]



def load_and_merge_plot_data(input_paths: list[Path]) -> pd.DataFrame:
    if not input_paths:
        raise ValueError("No input CSVs provided.")

    frames: list[pd.DataFrame] = []

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_txt = ", ".join(sorted(missing))
            raise ValueError(f"CSV '{path}' is missing required columns: {missing_txt}")

        if "plot_dataset" not in df.columns:
            df["plot_dataset"] = df["dataset"]

        df = df.copy()
        df["source_csv"] = str(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Optional strictness from your workflow: each algorithm should come from one source file only.
    algo_sources = merged.groupby("algorithm")["source_csv"].nunique()
    multi_source_algos = algo_sources[algo_sources > 1].index.tolist()
    if multi_source_algos:
        raise ValueError(
            "Algorithms found in multiple CSV files (not allowed for this workflow): "
            + ", ".join(sorted(multi_source_algos))
        )

    dup_mask = merged.duplicated(subset=["dataset_id", "algorithm"], keep=False)
    if dup_mask.any():
        dup_rows = merged.loc[dup_mask, ["dataset_id", "dataset", "algorithm", "source_csv"]]
        raise ValueError(
            "Duplicate dataset+algorithm rows found across inputs:\n"
            + dup_rows.to_string(index=False)
        )

    return merged


def plot_results(
    df: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    error_bars: str,
    plot_style: str,
    no_show: bool,
) -> None:
    plot_benchmark_results(
        df,
        dataset_label_col="plot_dataset",
        metrics=UCI_METRICS,
        output_dir=output_dir,
        output_basename_prefix="merged_ucimodels",
        plot_mode=plot_mode,
        error_bars=error_bars,
        plot_style=plot_style,
        no_show=no_show,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create merged plots from one or more benchmark plot-data CSVs")
    parser.add_argument(
        "--input-csvs",
        default=(
            "benchmarks/outputs/imodels/uci_imodels_plot_data.csv,"
            "benchmarks/outputs/exstracs/exstracs_plot_data.csv"
        ),
        help=(
            "Comma-separated list of input plot-data CSV files "
            "(default: benchmarks/outputs/imodels/uci_imodels_plot_data.csv,"
            "benchmarks/outputs/exstracs/exstracs_plot_data.csv)"
        ),
    )
    parser.add_argument("--plot-mode", default="combined", choices=["combined", "separate", "by_dataset"])
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="std", choices=["none", "std", "ci95"])
    parser.add_argument("--output-dir", default="benchmarks/outputs/merged")
    parser.add_argument("--no-show", action="store_true", help="Do not show matplotlib windows (save files only)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_paths = [Path(p) for p in parse_csv_list(args.input_csvs)]
    merged_df = load_and_merge_plot_data(input_paths)

    # Useful for debugging/traceability of merged sources.
    merged_csv_path = Path(args.output_dir) / "merged_plot_data.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"CSV saved (merged plot data): {merged_csv_path}")

    plot_results(
        merged_df,
        output_dir=Path(args.output_dir),
        plot_mode=args.plot_mode,
        error_bars=args.error_bars,
        plot_style=args.plot_style,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()

