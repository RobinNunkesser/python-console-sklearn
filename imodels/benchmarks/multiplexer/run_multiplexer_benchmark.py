"""Benchmark imodels and ExSTraCS on local Multiplexer CSV datasets.

Differences to the UCI benchmark:
- no train/test split (fit + evaluate on the full dataset)
- metric is accuracy (instead of F1)
- dedicated plot output for multiplexer datasets
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from imodels import (
    C45TreeClassifier,
    DecisionTreeClassifier,
    FIGSClassifier,
    GreedyRuleListClassifier,
    GreedyTreeClassifier,
    HSTreeClassifier,
    OneRClassifier,
    SlipperClassifier,
    TaoTreeClassifier,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.multiplexer.multiplexer_plotting import build_plot_export_df, plot_results

ALGORITHM_REGISTRY: dict[str, Callable[..., Any]] = {
    "C45TreeClassifier": C45TreeClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "FIGSClassifier": FIGSClassifier,
    "GreedyRuleListClassifier": GreedyRuleListClassifier,
    "GreedyTreeClassifier": GreedyTreeClassifier,
    "HSTreeClassifier": HSTreeClassifier,
    "OneRClassifier": OneRClassifier,
    "SlipperClassifier": SlipperClassifier,
    "TaoTreeClassifier": TaoTreeClassifier,
}

EXSTRACS_ALGORITHMS: dict[str, dict[str, Any]] = {
    "ExSTraCS": {"rule_compaction": None},
    "ExSTraCS_FU1": {"rule_compaction": "Fu1"},
    "ExSTraCS_FU2": {"rule_compaction": "Fu2"},
    "ExSTraCS_QRF": {"rule_compaction": "QRF"},
}


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def estimate_model_size(model: Any) -> float:
    if hasattr(model, "complexity_"):
        try:
            return float(getattr(model, "complexity_"))
        except (TypeError, ValueError):
            pass

    if hasattr(model, "rules_"):
        rules = getattr(model, "rules_")
        if isinstance(rules, (list, tuple)):
            return float(len(rules))
        if hasattr(rules, "shape"):
            return float(rules.shape[0])

    if hasattr(model, "get_rules"):
        try:
            rules_df = model.get_rules()
            if isinstance(rules_df, pd.DataFrame):
                if {"type", "coef"}.issubset(rules_df.columns):
                    active = rules_df[(rules_df["type"] == "rule") & (rules_df["coef"].abs() > 1e-12)]
                    return float(len(active))
                return float(len(rules_df))
        except Exception:
            pass

    if hasattr(model, "tree_") and hasattr(model.tree_, "node_count"):
        try:
            return float(model.tree_.node_count)
        except Exception:
            pass

    if hasattr(model, "estimators_"):
        try:
            return float(len(model.estimators_))
        except Exception:
            pass

    return float("nan")

def list_multiplexer_csvs(real_dir: Path) -> list[Path]:
    files = sorted(real_dir.glob("Multiplexer*.csv"))
    if not files:
        raise FileNotFoundError(f"No Multiplexer*.csv files found in: {real_dir}")
    return files


def load_csv_dataset(path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Keep labels numeric where possible.
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y.astype(str))[0]
        y = pd.Series(y, index=df.index)

    return X, y


def instantiate_classifier(model_cls: Callable[..., Any], random_state: int) -> Any:
    try:
        sig = inspect.signature(model_cls)
        if "random_state" in sig.parameters:
            return model_cls(random_state=random_state)
    except (TypeError, ValueError):
        pass
    return model_cls()


def evaluate_imodels_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm_name: str,
    random_state: int,
) -> dict[str, Any]:
    model_cls = ALGORITHM_REGISTRY[algorithm_name]
    clf = instantiate_classifier(model_cls, random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", clf),
        ]
    )
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    acc = float(accuracy_score(y, y_pred))
    size = float(estimate_model_size(pipe.named_steps["model"]))
    return {
        "accuracy": acc,
        "model_size": size,
    }


def evaluate_exstracs_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    rule_compaction: str | None,
    random_state: int,
    learning_iterations: int,
    population_size: int,
) -> dict[str, Any]:
    try:
        from skExSTraCS import ExSTraCS
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "skExSTraCS is required for ExSTraCS algorithms. "
            "Install it or exclude ExSTraCS variants via --algorithms."
        ) from exc

    X_np = X.to_numpy(dtype=float, copy=True)
    y_np = pd.Series(y).to_numpy()

    model = ExSTraCS(
        learning_iterations=learning_iterations,
        N=population_size,
        rule_compaction=rule_compaction,
        random_state=random_state,
    )
    model.fit(X_np, y_np)
    y_pred = model.predict(X_np)

    acc = float(accuracy_score(y_np, y_pred))
    model_size_post = float(len(model.population.popSet))
    return {
        "accuracy": acc,
        "model_size": model_size_post,
    }


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

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run imodels + ExSTraCS on Multiplexer CSV datasets (no split, accuracy metric)."
    )
    parser.add_argument(
        "--real-dir",
        default="data/csv/Real",
        help="Directory containing Multiplexer*.csv files (relative paths are resolved from the repo root)",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset stems, e.g. Multiplexer6,Multiplexer11Modified",
    )
    default_algorithms = [*ALGORITHM_REGISTRY.keys(), "ExSTraCS_QRF"]
    parser.add_argument(
        "--algorithms",
        default=",".join(default_algorithms),
        help="Comma-separated algorithm names (imodels + ExSTraCS variants)",
    )
    parser.add_argument("--target-col", default="class")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--learning-iterations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--plot-mode", default="combined", choices=["combined", "separate", "by_dataset"])
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="std", choices=["none", "std", "ci95"])
    parser.add_argument(
        "--output-dir",
        default="benchmarks/outputs/multiplexer",
        help="Output directory (relative paths are resolved from the repo root)",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    real_dir = resolve_project_path(args.real_dir)
    all_files = list_multiplexer_csvs(real_dir)

    requested = set(parse_csv_list(args.datasets)) if args.datasets else None
    if requested is not None:
        selected_files = [p for p in all_files if p.stem in requested]
        missing = sorted(requested - {p.stem for p in selected_files})
        if missing:
            raise ValueError(f"Unknown dataset names in --datasets: {', '.join(missing)}")
    else:
        selected_files = all_files

    algorithm_names = parse_csv_list(args.algorithms)
    if not algorithm_names:
        raise ValueError("No algorithms selected")

    known_algorithms = set(ALGORITHM_REGISTRY) | set(EXSTRACS_ALGORITHMS)
    invalid = [name for name in algorithm_names if name not in known_algorithms]
    if invalid:
        raise ValueError(f"Unknown algorithms: {invalid}")

    rows: list[dict[str, Any]] = []

    for dataset_idx, csv_path in enumerate(selected_files, start=1):
        dataset_name = csv_path.stem
        X, y = load_csv_dataset(csv_path, target_col=args.target_col)
        print(f"\n--- [{dataset_idx}/{len(selected_files)}] {dataset_name} ---")
        print(f"Samples={len(X)}, Features={X.shape[1]}")

        for algo in algorithm_names:
            for run_idx in range(args.n_runs):
                seed = args.random_state + run_idx
                print(f"  -> {algo} (run {run_idx + 1}/{args.n_runs}, seed={seed}) ...", end=" ", flush=True)
                try:
                    if algo in ALGORITHM_REGISTRY:
                        metrics = evaluate_imodels_full_data(X, y, algo, random_state=seed)
                    else:
                        cfg = EXSTRACS_ALGORITHMS[algo]
                        metrics = evaluate_exstracs_full_data(
                            X,
                            y,
                            rule_compaction=cfg["rule_compaction"],
                            random_state=seed,
                            learning_iterations=args.learning_iterations,
                            population_size=args.population_size,
                        )

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_file": str(csv_path),
                            "algorithm": algo,
                            "accuracy": float(metrics["accuracy"]),
                            "model_size": float(metrics["model_size"]),
                            "run_idx": run_idx,
                            "seed": seed,
                            "error": "",
                        }
                    )
                    print(f"ok | acc={metrics['accuracy']:.4f}, model_size={metrics['model_size']:.1f}")
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_file": str(csv_path),
                            "algorithm": algo,
                            "accuracy": float("nan"),
                            "model_size": float("nan"),
                            "run_idx": run_idx,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"failed ({type(exc).__name__}: {exc})")

    results_df = pd.DataFrame(rows)
    agg_df = aggregate_results(results_df)
    plot_df = build_plot_export_df(agg_df)

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "multiplexer_results.csv"
    agg_csv = out_dir / "multiplexer_results_agg.csv"
    plot_csv = out_dir / "multiplexer_plot_data.csv"
    results_df.to_csv(raw_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    plot_df.to_csv(plot_csv, index=False)
    print(f"\nCSV saved (raw): {raw_csv}")
    print(f"CSV saved (aggregate): {agg_csv}")
    print(f"CSV saved (plot data): {plot_csv}")

    plot_results(
        agg_df,
        output_dir=out_dir,
        plot_mode=args.plot_mode,
        error_bars=args.error_bars,
        plot_style=args.plot_style,
        no_show=args.no_show,
    )

    if not agg_df.empty:
        print("\nAggregate results (excerpt):")
        print(
            agg_df[[
                "dataset",
                "algorithm",
                "runs_total",
                "accuracy_mean",
                "accuracy_std",
                "model_size_mean",
                "model_size_std",
            ]]
            .head(24)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

