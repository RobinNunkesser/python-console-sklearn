"""Shared result aggregation/export helpers for UCI benchmark scripts."""

from __future__ import annotations

import pandas as pd

PLOT_EXPORT_COLUMNS = [
    "dataset_id",
    "dataset",
    "plot_dataset",
    "algorithm",
    "runs_total",
    "f1_mean",
    "f1_std",
    "f1_ci95",
    "model_size_mean",
    "model_size_std",
    "model_size_ci95",
]


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw runs to mean/std/95%-CI per dataset+algorithm."""
    if results_df.empty:
        return pd.DataFrame()

    group_cols = ["dataset_id", "dataset", "n_samples", "n_features", "algorithm"]
    grouped = results_df.groupby(group_cols, dropna=False)

    agg_df = grouped.agg(
        test_size=("test_size", "first"),
        f1_average=("f1_average", "first"),
        runs_total=("run_idx", "nunique"),
        f1_n=("f1", lambda s: int(s.notna().sum())),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        model_size_n=("model_size", lambda s: int(s.notna().sum())),
        model_size_mean=("model_size", "mean"),
        model_size_std=("model_size", "std"),
        error_n=("error", lambda s: int((s.fillna("").astype(str) != "").sum()))
        if "error" in results_df.columns
        else ("algorithm", "size"),
    ).reset_index()

    if "error" not in results_df.columns:
        agg_df["error_n"] = 0

    agg_df["f1_std"] = agg_df["f1_std"].fillna(0.0)
    agg_df["model_size_std"] = agg_df["model_size_std"].fillna(0.0)

    f1_den = agg_df["f1_n"].where(agg_df["f1_n"] > 0, 1) ** 0.5
    model_size_den = agg_df["model_size_n"].where(agg_df["model_size_n"] > 0, 1) ** 0.5

    agg_df["f1_ci95"] = 1.96 * agg_df["f1_std"] / f1_den
    agg_df["model_size_ci95"] = 1.96 * agg_df["model_size_std"] / model_size_den
    return agg_df



def build_plot_export_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable plot-data export schema for downstream plotting scripts."""
    if agg_df.empty:
        return pd.DataFrame(columns=PLOT_EXPORT_COLUMNS)

    export_df = agg_df.copy()
    if "plot_dataset" not in export_df.columns:
        export_df["plot_dataset"] = export_df["dataset"]

    for col in PLOT_EXPORT_COLUMNS:
        if col not in export_df.columns:
            export_df[col] = pd.NA

    return export_df[PLOT_EXPORT_COLUMNS].copy()

