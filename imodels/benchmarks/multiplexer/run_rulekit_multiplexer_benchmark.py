"""Benchmark RuleKit on local Multiplexer CSV datasets.

Follows the same conventions as run_multiplexer_benchmark.py:
- no train/test split (fit + evaluate on the full dataset)
- metric is accuracy
- uses the same Multiplexer*.csv files in data/csv/Real/
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rulekit.classification import RuleClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import os
os.environ['JAVA_HOME'] = os.popen('/usr/libexec/java_home').read().strip()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.multiplexer.multiplexer_plotting import build_plot_export_df, plot_results

ALGORITHM_REGISTRY: dict[str, Callable[..., Any]] = {
    "RuleClassifier": RuleClassifier,
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


class RuleKitPreprocessor:
    """Preprocessing pipeline that returns plain DataFrames (required by RuleClassifier)."""

    def __init__(self):
        self.numeric_imputer = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.encoder: OneHotEncoder | None = None

    def fit(self, X: pd.DataFrame) -> "RuleKitPreprocessor":
        self.numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        if self.numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy="median")
            self.numeric_imputer.fit(X[self.numeric_cols])
        if self.categorical_cols:
            self.encoder = make_one_hot_encoder()
            self.encoder.fit(X[self.categorical_cols].fillna("missing"))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result: dict[str, Any] = {}
        if self.numeric_cols:
            arr = self.numeric_imputer.transform(X[self.numeric_cols])
            for i, col in enumerate(self.numeric_cols):
                result[col] = arr[:, i]
        if self.categorical_cols:
            arr = self.encoder.transform(X[self.categorical_cols].fillna("missing"))
            for i, name in enumerate(self.encoder.get_feature_names_out(self.categorical_cols)):
                result[name] = arr[:, i]
        return pd.DataFrame(result)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


def estimate_model_size(model: Any) -> float:
    """Return number of induced rules for RuleKit, with generic fallbacks."""
    if hasattr(model, "model") and getattr(model, "model") is not None:
        ruleset = model.model
        try:
            if hasattr(ruleset, "rules"):
                return float(len(ruleset.rules))
        except Exception:
            pass
        try:
            if hasattr(ruleset, "_java_object"):
                return float(len(ruleset._java_object.getRules()))
        except Exception:
            pass

    if hasattr(model, "complexity_"):
        try:
            return float(model.complexity_)
        except (TypeError, ValueError):
            pass

    if hasattr(model, "rules_"):
        rules = model.rules_
        if isinstance(rules, (list, tuple)):
            return float(len(rules))
        if hasattr(rules, "shape"):
            return float(rules.shape[0])

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

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y.astype(str))[0], index=df.index)

    return X, y


def instantiate_classifier(model_cls: Callable[..., Any], random_state: int) -> Any:
    try:
        sig = inspect.signature(model_cls)
        if "random_state" in sig.parameters:
            return model_cls(random_state=random_state)
    except (TypeError, ValueError):
        pass
    return model_cls()


def evaluate_rulekit_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm_name: str,
    random_state: int,
) -> dict[str, Any]:
    model_cls = ALGORITHM_REGISTRY[algorithm_name]
    clf = instantiate_classifier(model_cls, random_state=random_state)

    preprocessor = RuleKitPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    # RuleClassifier requires a named pandas Series as labels
    if not isinstance(y, pd.Series):
        y_series = pd.Series(y, name="target")
    else:
        y_series = y.copy()
        if y_series.name is None:
            y_series.name = "target"

    clf.fit(X_processed, y_series)
    y_pred = clf.predict(X_processed)

    return {
        "accuracy": float(accuracy_score(y_series, y_pred)),
        "model_size": estimate_model_size(clf),
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
        description="Run RuleKit on Multiplexer CSV datasets (no split, accuracy metric)."
    )
    parser.add_argument(
        "--real-dir",
        default="data/csv/Real",
        help="Directory containing Multiplexer*.csv files (relative paths are resolved from the repo root)",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset stems, e.g. Multiplexer6,Multiplexer11",
    )
    parser.add_argument(
        "--algorithms",
        default="RuleClassifier",
        help="Comma-separated algorithm names (default: RuleClassifier)",
    )
    parser.add_argument("--target-col", default="class")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plot-mode", default="combined", choices=["combined", "separate", "by_dataset"])
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="std", choices=["none", "std", "ci95"])
    parser.add_argument(
        "--output-dir",
        default="benchmarks/outputs/multiplexer/rulekit",
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

    invalid = [n for n in algorithm_names if n not in ALGORITHM_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown algorithms: {invalid}. Known: {list(ALGORITHM_REGISTRY)}")

    rows: list[dict[str, Any]] = []

    for dataset_idx, csv_path in enumerate(selected_files, start=1):
        dataset_name = csv_path.stem
        X, y = load_csv_dataset(csv_path, target_col=args.target_col)
        print(f"\n--- [{dataset_idx}/{len(selected_files)}] {dataset_name} ---")
        print(f"  Samples={len(X)}, Features={X.shape[1]}")

        for algo in algorithm_names:
            for run_idx in range(args.n_runs):
                seed = args.random_state + run_idx
                print(f"  -> {algo} (run {run_idx + 1}/{args.n_runs}, seed={seed}) ...", end=" ", flush=True)
                try:
                    metrics = evaluate_rulekit_full_data(X, y, algo, random_state=seed)
                    rows.append({
                        "dataset": dataset_name,
                        "dataset_file": str(csv_path),
                        "algorithm": algo,
                        "accuracy": metrics["accuracy"],
                        "model_size": metrics["model_size"],
                        "run_idx": run_idx,
                        "seed": seed,
                        "error": "",
                    })
                    size_txt = f"{metrics['model_size']:.1f}" if pd.notna(metrics["model_size"]) else "n/a"
                    print(f"ok | acc={metrics['accuracy']:.4f}, model_size={size_txt}")
                except Exception as exc:
                    rows.append({
                        "dataset": dataset_name,
                        "dataset_file": str(csv_path),
                        "algorithm": algo,
                        "accuracy": float("nan"),
                        "model_size": float("nan"),
                        "run_idx": run_idx,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"failed ({type(exc).__name__}: {exc})")

    results_df = pd.DataFrame(rows)
    agg_df = aggregate_results(results_df)
    plot_df = build_plot_export_df(agg_df)

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "rulekit_multiplexer_results.csv"
    agg_csv = out_dir / "rulekit_multiplexer_results_agg.csv"
    plot_csv = out_dir / "rulekit_multiplexer_plot_data.csv"
    results_df.to_csv(raw_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    plot_df.to_csv(plot_csv, index=False)
    print(f"\nCSV saved (raw):       {raw_csv}")
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
        print("\nAggregate results:")
        print(
            agg_df[[
                "dataset", "algorithm", "runs_total",
                "accuracy_mean", "accuracy_std",
                "model_size_mean", "model_size_std",
            ]].to_string(index=False)
        )


if __name__ == "__main__":
    main()

