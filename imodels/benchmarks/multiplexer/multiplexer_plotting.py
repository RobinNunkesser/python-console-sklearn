"""Thin adapter around the shared benchmark plotting module for multiplexer outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from benchmarks.shared_plotting import plot_benchmark_results

PLOT_COLUMNS = [
    "dataset",
    "algorithm",
    "runs_total",
    "accuracy_mean",
    "accuracy_std",
    "accuracy_ci95",
    "model_size_mean",
    "model_size_std",
    "model_size_ci95",
]

REQUIRED_PLOT_COLUMNS = {
    "dataset",
    "algorithm",
    "accuracy_mean",
    "model_size_mean",
}

MULTIPLEXER_METRICS: Final[list[dict[str, str]]] = [
    {
        "metric": "accuracy",
        "label": "Accuracy",
        "title": "Accuracy by Dataset and Algorithm",
        "separate_title": "Accuracy by Dataset and Algorithm",
        "xscale": "linear",
    },
    {
        "metric": "model_size",
        "label": "Model Size",
        "title": "Model Size by Dataset and Algorithm",
        "separate_title": "Model Size",
        "xscale": "log",
    },
]


def build_plot_export_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = agg_df.copy()
    for col in PLOT_COLUMNS:
        if col not in plot_df.columns:
            plot_df[col] = pd.NA
    return plot_df[PLOT_COLUMNS].copy()
def plot_results(
    agg_df: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    error_bars: str,
    plot_style: str,
    no_show: bool,
    output_basename_prefix: str = "multiplexer",
) -> None:
    plot_benchmark_results(
        agg_df,
        dataset_label_col="dataset",
        metrics=MULTIPLEXER_METRICS,
        output_dir=output_dir,
        output_basename_prefix=output_basename_prefix,
        plot_mode=plot_mode,
        error_bars=error_bars,
        plot_style=plot_style,
        no_show=no_show,
    )


