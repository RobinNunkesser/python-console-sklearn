"""Create benchmark plots from one or more exported plot-data CSV files.

Expected input schema is produced by `uciml.py` as `uci_imodels_plot_data.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "dataset_id",
    "dataset",
    "algorithm",
    "f1_mean",
    "model_size_mean",
}


def add_dataset_separators(ax: plt.Axes) -> None:
    ticks = sorted({float(tick) for tick in ax.get_yticks()})
    if len(ticks) < 2:
        return

    for left, right in zip(ticks, ticks[1:]):
        midpoint = (left + right) / 2
        ax.axhline(midpoint, color="0.88", linewidth=0.8, zorder=0)


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


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    title: str,
    ylabel: str,
    error_bars: str,
    plot_style: str,
    xscale: str = "linear",
) -> None:
    mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        raise ValueError(f"Missing required column for plotting: {mean_col}")

    pivot = df.pivot(index="plot_dataset", columns="algorithm", values=mean_col)

    err_data = None
    if error_bars != "none":
        err_col = f"{metric}_{error_bars}"
        if err_col in df.columns:
            err_data = df.pivot(index="plot_dataset", columns="algorithm", values=err_col).reindex_like(pivot)
            if xscale == "log":
                # On a log scale, mean - err must stay > 0.
                # Clip each error so it never reaches or exceeds the mean value.
                err_data = err_data.clip(upper=pivot * 0.9999)

    if plot_style == "bars":
        kwargs: dict[str, Any] = {"kind": "barh", "ax": ax}
        if err_data is not None:
            # barh: values are on the x-axis, so use xerr (not yerr).
            kwargs["xerr"] = err_data
            kwargs["capsize"] = 4

        pivot.plot(**kwargs)
    elif plot_style == "dots":
        datasets = list(pivot.index)
        algorithms = list(pivot.columns)
        y_base = np.arange(len(datasets), dtype=float)

        if len(algorithms) <= 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.3, 0.3, num=len(algorithms))

        colors = plt.rcParams.get("axes.prop_cycle", None)
        palette = colors.by_key().get("color", []) if colors is not None else []

        for idx, algorithm in enumerate(algorithms):
            color = palette[idx % len(palette)] if palette else None
            x = pivot[algorithm].to_numpy(dtype=float)
            y = y_base + offsets[idx]
            mask = ~np.isnan(x)

            if not mask.any():
                continue

            if err_data is not None and algorithm in err_data.columns:
                err_frame = cast(pd.DataFrame, err_data)
                err = err_frame[algorithm].to_numpy(dtype=float)
                err = np.nan_to_num(err, nan=0.0)
                ax.errorbar(
                    x[mask],
                    y[mask],
                    xerr=err[mask],
                    fmt="o",
                    capsize=4,
                    markersize=5,
                    elinewidth=1.2,
                    linewidth=1.2,
                    color=color,
                    label=algorithm,
                )
            else:
                ax.scatter(x[mask], y[mask], s=36, color=color, label=algorithm)

        ax.set_yticks(y_base, labels=datasets)
        ax.invert_yaxis()
    else:
        raise ValueError(f"Unknown plot_style: {plot_style}")

    ax.set_xscale(xscale)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Dataset")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)
    add_dataset_separators(ax)
    ax.legend(title="Algorithm")


def plot_results(
    df: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    error_bars: str,
    plot_style: str,
    no_show: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_mode == "combined":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        f1_ylabel = "F1" if error_bars == "none" else f"F1 (mean +/- {error_bars})"
        size_ylabel = "Model Size" if error_bars == "none" else f"Model Size (mean +/- {error_bars})"
        plot_metric(df, "f1", axes[0], "F1 by Dataset and Algorithm", f1_ylabel, error_bars, plot_style)
        plot_metric(
            df,
            "model_size",
            axes[1],
            "Model Size by Dataset and Algorithm",
            size_ylabel,
            error_bars,
            plot_style,
            xscale="log",
        )
        out = output_dir / "merged_ucimodels_combined.png"
        fig.savefig(out, dpi=150)
        print(f"Figure saved: {out}")
        if no_show:
            plt.close(fig)
        else:
            plt.show()
        return

    if plot_mode == "separate":
        fig_f1, ax_f1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
        f1_ylabel = "F1" if error_bars == "none" else f"F1 (mean +/- {error_bars})"
        plot_metric(df, "f1", ax_f1, "F1 by Dataset and Algorithm", f1_ylabel, error_bars, plot_style)
        out_f1 = output_dir / "merged_ucimodels_f1.png"
        fig_f1.savefig(out_f1, dpi=150)
        print(f"Figure saved: {out_f1}")

        fig_size, ax_size = plt.subplots(figsize=(8, 5), constrained_layout=True)
        size_ylabel = "Model Size" if error_bars == "none" else f"Model Size (mean +/- {error_bars})"
        plot_metric(df, "model_size", ax_size, "Model Size", size_ylabel, error_bars, plot_style, xscale="log")
        out_size = output_dir / "merged_ucimodels_model_size.png"
        fig_size.savefig(out_size, dpi=150)
        print(f"Figure saved: {out_size}")

        if no_show:
            plt.close(fig_f1)
            plt.close(fig_size)
        else:
            plt.show()
        return

    raise ValueError(f"Unknown plot_mode: {plot_mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create merged plots from one or more benchmark plot-data CSVs")
    parser.add_argument(
        "--input-csvs",
        default="results/uci_imodels_plot_data.csv,results_exstracs/exstracs_plot_data.csv",
        help=(
            "Comma-separated list of input plot-data CSV files "
            "(default: results/uci_imodels_plot_data.csv,results_exstracs/exstracs_plot_data.csv)"
        ),
    )
    parser.add_argument("--plot-mode", default="combined", choices=["combined", "separate"])
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="std", choices=["none", "std", "ci95"])
    parser.add_argument("--output-dir", default="results")
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

