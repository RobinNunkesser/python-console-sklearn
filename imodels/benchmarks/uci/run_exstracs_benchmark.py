"""Benchmark ExSTraCS (scikit-ExSTraCS) on UCI-ML-Repo datasets.

Reuses dataset loading, aggregation, and CSV export from run_imodels_benchmark.py.
Results can be combined with imodels results via merge_benchmark_plots.py:

    python benchmarks/uci/run_exstracs_benchmark.py --no-show
    python benchmarks/uci/merge_benchmark_plots.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from skExSTraCS import ExSTraCS

# Reuse shared utilities from the imodels benchmark script — no duplication.
from run_imodels_benchmark import (
    DEFAULT_DATASET_OPTIONS,
    PLOT_EXPORT_COLUMNS,
    DatasetBundle,
    aggregate_results,
    build_dataset_configs,
    build_plot_export_df,
    choose_split_params,
    load_uci_dataset,
    parse_csv_list,
    resolve_plot_dataset_label,
)


# ---------------------------------------------------------------------------
# Algorithm registry for this script
# ---------------------------------------------------------------------------
# Variants: without compaction, with FU2 compaction, and with QRF compaction.
# Use "ExSTraCS_QRF" as the default.
EXSTRACS_ALGORITHMS = {
    "ExSTraCS":     {"rule_compaction": None},
    "ExSTraCS_FU1": {"rule_compaction": "Fu1"},
    "ExSTraCS_FU2": {"rule_compaction": "Fu2"},
    "ExSTraCS_QRF": {"rule_compaction": "QRF"},
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_numpy_arrays(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all features to numeric arrays suitable for ExSTraCS.

    Uses OrdinalEncoder (handles both categorical and numeric columns)
    combined with SimpleImputer for missing values.
    """
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    frames_train, frames_test = [], []

    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        frames_train.append(imputer.fit_transform(X_train[numeric_cols]))
        frames_test.append(imputer.transform(X_test[numeric_cols]))

    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        imputer = SimpleImputer(strategy="most_frequent")
        X_cat_train = imputer.fit_transform(X_train[categorical_cols].astype(str))
        X_cat_test = imputer.transform(X_test[categorical_cols].astype(str))
        frames_train.append(encoder.fit_transform(X_cat_train))
        frames_test.append(encoder.transform(X_cat_test))

    Xtr = np.hstack(frames_train) if frames_train else X_train.values
    Xte = np.hstack(frames_test) if frames_test else X_test.values
    return Xtr.astype(float), Xte.astype(float)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_exstracs(
    data: DatasetBundle,
    algorithm_name: str,
    rule_compaction: str | None,
    random_state: int,
    learning_iterations: int,
    population_size: int,
) -> dict[str, Any]:
    """Train ExSTraCS (optionally with rule compaction) and return result row."""
    split_params = choose_split_params(len(data.X))
    n_classes = len(pd.unique(data.y))

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        data.X,
        data.y,
        random_state=random_state,
        stratify=data.y,
        **split_params,
    )

    X_train, X_test = build_numpy_arrays(X_train_df, X_test_df)
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    model = ExSTraCS(
        learning_iterations=learning_iterations,
        N=population_size,
        rule_compaction=rule_compaction,
        random_state=random_state,
    )
    model.fit(X_train, y_train_np)
    y_pred = model.predict(X_test)

    avg_mode = "binary" if n_classes == 2 else "macro"
    f1 = float(f1_score(y_test_np, y_pred, average=avg_mode, zero_division=0))
    model_size_post_compaction = float(len(model.population.popSet))
    rc_removed = float(getattr(model.trackingObj, "RCCount", 0.0)) if hasattr(model, "trackingObj") else 0.0
    model_size_pre_compaction = model_size_post_compaction + rc_removed

    return {
        "dataset_id": data.dataset_id,
        "dataset": data.name,
        "n_samples": len(data.X),
        "n_features": data.X.shape[1],
        "algorithm": algorithm_name,
        "f1": f1,
        "f1_average": avg_mode,
        # Keep model_size as post-compaction for plotting compatibility.
        "model_size": model_size_post_compaction,
        "model_size_pre_compaction": model_size_pre_compaction,
        "model_size_post_compaction": model_size_post_compaction,
        "model_size_rc_removed": rc_removed,
        "test_size": split_params["test_size"],
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_ids: list[int],
    algorithm_names: list[str],
    random_state: int,
    n_runs: int,
    learning_iterations: int,
    population_size: int,
    dataset_short_names_by_id: dict[int, str],
    dataset_short_names_by_name: dict[str, str],
    output_dir: Path,
    no_show: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_configs = build_dataset_configs(dataset_ids)
    default_short_names_by_id = {
        cfg.dataset_id: cfg.short_name
        for cfg in dataset_configs
        if cfg.short_name
    }

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    invalid = [n for n in algorithm_names if n not in EXSTRACS_ALGORITHMS]
    if invalid:
        raise ValueError(
            f"Unknown algorithms: {invalid}. Available: {list(EXSTRACS_ALGORITHMS)}"
        )

    rows: list[dict[str, Any]] = []

    for ds_cfg in dataset_configs:
        print(f"\n--- Loading dataset {ds_cfg.dataset_id}: {ds_cfg.name} ---")
        try:
            data = load_uci_dataset(ds_cfg)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        n_classes = len(pd.unique(data.y))
        print(
            f"Samples={len(data.X)}, Features={data.X.shape[1]}, "
            f"Classes={n_classes}, target_mode={ds_cfg.target_mode}"
        )

        for algo_name in algorithm_names:
            algo_cfg = EXSTRACS_ALGORITHMS[algo_name]
            for run_idx in range(n_runs):
                seed = random_state + run_idx
                print(
                    f"  -> Training {algo_name} "
                    f"(run {run_idx + 1}/{n_runs}, seed={seed}, "
                    f"iterations={learning_iterations}, N={population_size}) ...",
                    end=" ",
                    flush=True,
                )
                try:
                    row = evaluate_exstracs(
                        data=data,
                        algorithm_name=algo_name,
                        rule_compaction=algo_cfg["rule_compaction"],
                        random_state=seed,
                        learning_iterations=learning_iterations,
                        population_size=population_size,
                    )
                    row["run_idx"] = run_idx
                    row["seed"] = seed
                    row["error"] = ""
                    rows.append(row)
                    print(
                        "ok | "
                        f"F1={row['f1']:.4f}, "
                        f"model_size_post={row['model_size_post_compaction']:.0f}, "
                        f"pre={row['model_size_pre_compaction']:.0f}, "
                        f"removed={row['model_size_rc_removed']:.0f}"
                    )
                except Exception as exc:
                    print(f"failed ({type(exc).__name__}: {exc})")
                    rows.append(
                        {
                            "dataset_id": data.dataset_id,
                            "dataset": data.name,
                            "n_samples": len(data.X),
                            "n_features": data.X.shape[1],
                            "algorithm": algo_name,
                            "f1": float("nan"),
                            "f1_average": "n/a",
                            "model_size": float("nan"),
                            "model_size_pre_compaction": float("nan"),
                            "model_size_post_compaction": float("nan"),
                            "model_size_rc_removed": float("nan"),
                            "test_size": choose_split_params(len(data.X))["test_size"],
                            "run_idx": run_idx,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    results_df = pd.DataFrame(rows)
    agg_df = aggregate_results(results_df)

    compaction_cols = [
        "model_size_pre_compaction",
        "model_size_post_compaction",
        "model_size_rc_removed",
    ]
    if not results_df.empty and set(compaction_cols).issubset(results_df.columns):
        comp_group_cols = ["dataset_id", "dataset", "algorithm"]
        comp_agg = (
            results_df.groupby(comp_group_cols, dropna=False)[compaction_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        comp_agg.columns = [
            "dataset_id", "dataset", "algorithm",
            "model_size_pre_compaction_mean", "model_size_pre_compaction_std",
            "model_size_post_compaction_mean", "model_size_post_compaction_std",
            "model_size_rc_removed_mean", "model_size_rc_removed_std",
        ]
        agg_df = agg_df.merge(comp_agg, on=comp_group_cols, how="left")

    if not agg_df.empty:
        agg_df = agg_df.copy()
        agg_df["plot_dataset"] = agg_df.apply(
            lambda row: resolve_plot_dataset_label(
                dataset_id=int(row["dataset_id"]),
                dataset_name=str(row["dataset"]),
                default_short_names_by_id=default_short_names_by_id,
                user_short_names_by_id=dataset_short_names_by_id,
                user_short_names_by_name=dataset_short_names_by_name,
            ),
            axis=1,
        )

    plot_export_df = build_plot_export_df(agg_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_raw = output_dir / "exstracs_results.csv"
    results_df.to_csv(csv_raw, index=False)
    print(f"\nCSV saved (raw): {csv_raw}")

    csv_agg = output_dir / "exstracs_results_agg.csv"
    agg_df.to_csv(csv_agg, index=False)
    print(f"CSV saved (aggregate): {csv_agg}")

    csv_plot = output_dir / "exstracs_plot_data.csv"
    plot_export_df.to_csv(csv_plot, index=False)
    print(f"CSV saved (plot data): {csv_plot}")

    return results_df, agg_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_dataset_short_names(raw: str) -> tuple[dict[int, str], dict[str, str]]:
    """Parse short labels from '17:BreastCancer,heart_disease:Heart'."""
    mapping_by_id: dict[int, str] = {}
    mapping_by_name: dict[str, str] = {}
    if not raw.strip():
        return mapping_by_id, mapping_by_name
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid short-label format '{item}'. Expected: id:label or name:label")
        key_raw, label = item.split(":", 1)
        key_raw, label = key_raw.strip(), label.strip()
        if not key_raw or not label:
            raise ValueError(f"Invalid short-label format '{item}'.")
        if key_raw.isdigit():
            mapping_by_id[int(key_raw)] = label
        else:
            mapping_by_name[key_raw.lower()] = label
    return mapping_by_id, mapping_by_name


def build_arg_parser() -> argparse.ArgumentParser:
    default_ids = ",".join(str(k) for k in sorted(DEFAULT_DATASET_OPTIONS))
    parser = argparse.ArgumentParser(
        description="ExSTraCS benchmark — connect results to imodels via merge_benchmark_plots.py"
    )
    parser.add_argument(
        "--dataset-ids", default=default_ids,
        help="Comma-separated UCI dataset IDs (default: all known datasets)",
    )
    parser.add_argument(
        "--algorithms", default="ExSTraCS_FU1",
        help=f"Comma-separated algorithm names. Available: {', '.join(EXSTRACS_ALGORITHMS)}",
    )
    parser.add_argument(
        "--learning-iterations", type=int, default=100000,
        help="ExSTraCS learning iterations (default: 100000)",
    )
    parser.add_argument(
        "--population-size", type=int, default=1000,
        help="ExSTraCS maximum population size N (default: 1000)",
    )
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Runs per dataset+algorithm (default: 3)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dataset-short-names", default="",
        help="Optional short plot labels, e.g. 17:BreastCancer,heart_disease:Heart",
    )
    parser.add_argument("--output-dir", default="benchmarks/outputs/exstracs")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not show matplotlib windows (save files only)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_ids = [int(s) for s in parse_csv_list(args.dataset_ids)]
    algorithm_names = parse_csv_list(args.algorithms)
    short_by_id, short_by_name = parse_dataset_short_names(args.dataset_short_names)

    results_df, agg_df = run_benchmark(
        dataset_ids=dataset_ids,
        algorithm_names=algorithm_names,
        random_state=args.random_state,
        n_runs=args.n_runs,
        learning_iterations=args.learning_iterations,
        population_size=args.population_size,
        dataset_short_names_by_id=short_by_id,
        dataset_short_names_by_name=short_by_name,
        output_dir=Path(args.output_dir),
        no_show=args.no_show,
    )

    if not results_df.empty:
        print("\nResults (raw, excerpt):")
        print(
            results_df[[
                "dataset", "algorithm", "seed", "f1",
                "model_size_pre_compaction",
                "model_size_post_compaction",
                "model_size_rc_removed",
            ]]
            .head(12).to_string(index=False)
        )

    if not agg_df.empty:
        print("\nResults (aggregate):")
        print(
            agg_df[[
                "dataset", "algorithm", "runs_total",
                "f1_mean", "f1_std", "model_size_mean", "model_size_std",
                "model_size_pre_compaction_mean",
                "model_size_post_compaction_mean",
                "model_size_rc_removed_mean",
            ]]
            .to_string(index=False)
        )

    print(
        "\nTo merge with imodels results:\n"
        f"  python benchmarks/uci/merge_benchmark_plots.py \\\n"
        f"    --input-csvs \"benchmarks/outputs/imodels/uci_imodels_plot_data.csv,"
        f"{args.output_dir}/exstracs_plot_data.csv\" \\\n"
        f"    --no-show --output-dir benchmarks/outputs/merged"
    )


if __name__ == "__main__":
    main()

