"""Merge one or more multiplexer plot-data CSV files and render a combined plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.multiplexer.multiplexer_plotting import REQUIRED_PLOT_COLUMNS, plot_results


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
        missing = REQUIRED_PLOT_COLUMNS - set(df.columns)
        if missing:
            missing_txt = ", ".join(sorted(missing))
            raise ValueError(f"CSV '{path}' is missing required columns: {missing_txt}")

        df = df.copy()
        df["source_csv"] = str(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    dup_mask = merged.duplicated(subset=["dataset", "algorithm"], keep=False)
    if dup_mask.any():
        dup_rows = merged.loc[dup_mask, ["dataset", "algorithm", "source_csv"]]
        raise ValueError(
            "Duplicate dataset+algorithm rows found across inputs:\n"
            + dup_rows.to_string(index=False)
        )

    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a merged plot from one or more multiplexer plot-data CSV files"
    )
    parser.add_argument(
        "--input-csvs",
        default="benchmarks/outputs/multiplexer/multiplexer_plot_data.csv",
        help=(
            "Comma-separated list of multiplexer plot-data CSV files "
            "(default: benchmarks/outputs/multiplexer/multiplexer_plot_data.csv)"
        ),
    )
    parser.add_argument("--plot-mode", default="combined", choices=["combined", "separate", "by_dataset"])
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="std", choices=["none", "std", "ci95"])
    parser.add_argument("--output-dir", default="benchmarks/outputs/multiplexer/merged")
    parser.add_argument("--no-show", action="store_true", help="Do not show matplotlib windows")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    input_paths = [Path(p) for p in parse_csv_list(args.input_csvs)]
    merged_df = load_and_merge_plot_data(input_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_plot_data.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"CSV saved (merged plot data): {merged_csv_path}")

    plot_results(
        merged_df,
        output_dir=output_dir,
        plot_mode=args.plot_mode,
        error_bars=args.error_bars,
        plot_style=args.plot_style,
        no_show=args.no_show,
        output_basename_prefix="merged_multiplexer",
    )


if __name__ == "__main__":
    main()





