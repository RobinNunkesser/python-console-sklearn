"""Build LogicGP plot-data CSV from raw metrics CSV files.

Reads all *_Metrics.csv files from benchmarks/outputs/logicgp and writes
benchmarks/outputs/logicgp/logicgp_plot_data.csv with a schema compatible with
merged benchmark plot data.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.uci.shared_datasets import DEFAULT_DATASET_OPTIONS

OUTPUT_COLUMNS = [
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
    "source_csv",
]


def infer_dataset_from_filename(path: Path) -> str:
    """Infer dataset name from filenames like heart_disease_20260318211139_Metrics.csv."""
    stem = path.stem
    if stem.endswith("_Metrics"):
        stem = stem[: -len("_Metrics")]

    # Prefer removing a trailing timestamp segment if present.
    match = re.match(r"^(?P<dataset>.+)_\d{8,}$", stem)
    if match:
        return match.group("dataset")

    # Fallback: assume everything before first underscore is the dataset token.
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def compute_ci95(std: float, n: int) -> float:
    if n <= 0 or pd.isna(std):
        return float("nan")
    return 1.96 * float(std) / math.sqrt(n)


def auto_short_dataset_name(dataset_name: str) -> str:
    """Generate a short readable label for unknown datasets."""
    tokens = [tok for tok in str(dataset_name).replace("-", "_").split("_") if tok]
    if not tokens:
        return "Dataset"
    if len(tokens) == 1:
        return tokens[0][:14]
    if len(tokens) <= 3:
        return "".join(tok[:5].capitalize() for tok in tokens)[:18]
    acronym = "".join(tok[0].upper() for tok in tokens if tok)
    return acronym if acronym else "Dataset"


def build_short_name_mapping() -> dict[str, str]:
    """Build dataset-name -> short-label mapping from shared UCI defaults."""
    mapping: dict[str, str] = {}
    for cfg in DEFAULT_DATASET_OPTIONS.values():
        name = str(cfg.get("name", "")).strip().lower()
        short_name = cfg.get("short_name")
        if name and short_name:
            mapping[name] = str(short_name)
    return mapping


SHORT_NAME_BY_DATASET = build_short_name_mapping()


def load_raw_metrics(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*_Metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_Metrics.csv files found in: {input_dir}")

    rows: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        required = {"F1", "Size"}
        missing = required - set(df.columns)
        if missing:
            missing_txt = ", ".join(sorted(missing))
            raise ValueError(f"CSV '{csv_path.name}' is missing required columns: {missing_txt}")

        work = df[["F1", "Size"]].copy()
        work = work.rename(columns={"F1": "f1", "Size": "model_size"})
        work["dataset"] = infer_dataset_from_filename(csv_path)
        work["algorithm"] = "LogicGP"
        work["source_csv"] = str(csv_path)
        work["run_idx"] = range(len(work))
        rows.append(work)

    raw_df = pd.concat(rows, ignore_index=True)
    return raw_df


def aggregate_plot_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["dataset", "algorithm"], dropna=False)

    agg = grouped.agg(
        runs_total=("f1", lambda s: int(s.notna().sum())),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        model_size_mean=("model_size", "mean"),
        model_size_std=("model_size", "std"),
    ).reset_index()

    agg["f1_std"] = agg["f1_std"].fillna(0.0)
    agg["model_size_std"] = agg["model_size_std"].fillna(0.0)

    agg["f1_ci95"] = agg.apply(
        lambda row: compute_ci95(row["f1_std"], int(row["runs_total"])),
        axis=1,
    )
    agg["model_size_ci95"] = agg.apply(
        lambda row: compute_ci95(row["model_size_std"], int(row["runs_total"])),
        axis=1,
    )

    source_map = (
        raw_df.groupby(["dataset", "algorithm"], dropna=False)["source_csv"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .reset_index()
    )
    agg = agg.merge(source_map, on=["dataset", "algorithm"], how="left")

    dataset_order = sorted(agg["dataset"].astype(str).unique().tolist())
    dataset_id_map = {name: -(idx + 1) for idx, name in enumerate(dataset_order)}
    agg["dataset_id"] = agg["dataset"].astype(str).map(dataset_id_map)
    agg["plot_dataset"] = agg["dataset"].astype(str).map(
        lambda name: SHORT_NAME_BY_DATASET.get(name.strip().lower(), auto_short_dataset_name(name))
    )

    for col in OUTPUT_COLUMNS:
        if col not in agg.columns:
            agg[col] = pd.NA

    return agg[OUTPUT_COLUMNS].copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create logicgp_plot_data.csv from LogicGP metrics CSV files")
    parser.add_argument(
        "--input-dir",
        default="benchmarks/outputs/logicgp",
        help="Directory containing *_Metrics.csv files",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmarks/outputs/logicgp/logicgp_plot_data.csv",
        help="Target plot-data CSV path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    raw_df = load_raw_metrics(input_dir)
    plot_df = aggregate_plot_data(raw_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(output_csv, index=False)

    print(f"Loaded rows: {len(raw_df)}")
    print(f"Aggregated rows: {len(plot_df)}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()

