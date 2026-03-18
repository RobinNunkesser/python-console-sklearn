"""Shared dataset and result helpers for multiplexer benchmarks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_multiplexer_csvs(real_dir: Path) -> list[Path]:
    files = sorted(real_dir.glob("Multiplexer*.csv"))
    if not files:
        raise FileNotFoundError(f"No Multiplexer*.csv files found in: {real_dir}")
    return files



def select_multiplexer_files(all_files: list[Path], raw_datasets: str, parse_csv_list) -> list[Path]:
    requested = set(parse_csv_list(raw_datasets)) if raw_datasets else None
    if requested is None:
        return all_files

    selected_files = [p for p in all_files if p.stem in requested]
    missing = sorted(requested - {p.stem for p in selected_files})
    if missing:
        raise ValueError(f"Unknown dataset names in --datasets: {', '.join(missing)}")
    return selected_files



def load_csv_dataset(path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y.astype(str))[0], index=df.index)

    return X, y



def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "algorithm"]
    grouped = results_df.groupby(group_cols, dropna=False)

    agg_df = grouped.agg(
        runs_total=("run_idx", "nunique"),
        accuracy_n=("accuracy", lambda s: int(s.notna().sum())),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        model_size_n=("model_size", lambda s: int(s.notna().sum())),
        model_size_mean=("model_size", "mean"),
        model_size_std=("model_size", "std"),
        error_n=("error", lambda s: int((s.fillna("").astype(str) != "").sum())),
    ).reset_index()

    agg_df["accuracy_std"] = agg_df["accuracy_std"].fillna(0.0)
    agg_df["model_size_std"] = agg_df["model_size_std"].fillna(0.0)

    acc_den = agg_df["accuracy_n"].where(agg_df["accuracy_n"] > 0, 1) ** 0.5
    size_den = agg_df["model_size_n"].where(agg_df["model_size_n"] > 0, 1) ** 0.5

    agg_df["accuracy_ci95"] = 1.96 * agg_df["accuracy_std"] / acc_den
    agg_df["model_size_ci95"] = 1.96 * agg_df["model_size_std"] / size_den
    return agg_df

