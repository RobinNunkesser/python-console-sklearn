"""Benchmark RuleKit classifiers on UCI-ML-Repo datasets.

RuleKit is a Java-based rule learning system. This benchmark evaluates
the RuleKit classifier on the same UCI-ML-Repo datasets used for other models.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rulekit.classification import RuleClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo
import os
os.environ['JAVA_HOME'] = os.popen('/usr/libexec/java_home').read().strip()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.shared_plotting import UCI_METRICS, plot_benchmark_results


ALGORITHM_REGISTRY: dict[str, Callable[..., Any]] = {
    "RuleClassifier": RuleClassifier,
}

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

DEFAULT_DATASET_OPTIONS: dict[int, dict[str, Any]] = {
    17: {
        "name": "breast_cancer_wisconsin_diagnostic",
        "short_name": "BreastCancer",
        "target_mode": "auto",
    },
    45: {
        "name": "heart_disease",
        "short_name": "Heart",
        "target_mode": "auto",
    },
}
DEFAULT_DATASET_OPTIONS.update({
    12: {
        "name": "balance_scale",
        "short_name": "Balance",
        "target_mode": "auto",
    },
    19: {
        "name": "car_evaluation",
        "short_name": "Car",
        "target_mode": "auto",
    },
    53: {
        "name": "iris",
        "short_name": "Iris",
        "target_mode": "auto",
    },
    78: {
        "name": "page_blocks_classification",
        "short_name": "PageBlocks",
        "target_mode": "auto",
    },
    109: {
        "name": "wine",
        "short_name": "Wine",
        "target_mode": "auto",
    },
    267: {
        "name": "banknote_authentication",
        "short_name": "Banknote",
        "target_mode": "auto",
    },
})


@dataclass
class DatasetConfig:
    dataset_id: int
    name: str
    short_name: str | None = None
    target_mode: str = "auto"


@dataclass
class DatasetBundle:
    dataset_id: int
    name: str
    X: pd.DataFrame
    y: pd.Series


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatible with older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def choose_split_params(n_samples: int) -> dict[str, Any]:
    """Typical train/test split choices by dataset size."""
    if n_samples < 500:
        return {"test_size": 0.30}
    if n_samples < 5_000:
        return {"test_size": 0.25}
    return {"test_size": 0.20}


class RuleKitPreprocessor:
    """Preprocessor that works well with RuleClassifier."""
    
    def __init__(self):
        self.numeric_imputer = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.encoder = None
    
    def fit(self, X: pd.DataFrame) -> "RuleKitPreprocessor":
        """Fit preprocessor on training data."""
        self.numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        
        if self.numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy="median")
            self.numeric_imputer.fit(X[self.numeric_cols])
        
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].fillna("missing")
            self.encoder = make_one_hot_encoder()
            self.encoder.fit(X_cat)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        result_data = {}
        
        if self.numeric_cols:
            X_numeric_processed = self.numeric_imputer.transform(X[self.numeric_cols])
            for i, col in enumerate(self.numeric_cols):
                result_data[col] = X_numeric_processed[:, i]
        
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].fillna("missing")
            X_cat_processed = self.encoder.transform(X_cat)
            cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
            for i, name in enumerate(cat_feature_names):
                result_data[name] = X_cat_processed[:, i]
        
        return pd.DataFrame(result_data)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


def instantiate_classifier(model_cls: Callable[..., Any], random_state: int) -> Any:
    """Set random_state only when the constructor supports it."""
    try:
        sig = inspect.signature(model_cls)
        if "random_state" in sig.parameters:
            return model_cls(random_state=random_state)
    except (TypeError, ValueError):
        pass
    return model_cls()


def estimate_model_size(model: Any) -> float:
    """Best-effort complexity metric with model-specific fallbacks."""
    # RuleKit stores induced rules in model.model (RuleSet).
    if hasattr(model, "model") and getattr(model, "model") is not None:
        ruleset = getattr(model, "model")
        try:
            if hasattr(ruleset, "rules"):
                return float(len(ruleset.rules))
        except Exception:
            pass
        try:
            if hasattr(ruleset, "_java_object"):
                java_rules = ruleset._java_object.getRules()  # pylint: disable=protected-access
                return float(len(java_rules))
        except Exception:
            pass

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


def normalize_target(y_raw: pd.Series, target_mode: str) -> pd.Series:
    y = y_raw.copy()

    if target_mode == "nonzero_is_positive":
        y_num = pd.to_numeric(y, errors="coerce")
        return (y_num.fillna(0) > 0).astype(int)

    if target_mode == "auto":
        if not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            return pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
        return y

    raise ValueError(f"Unknown target_mode: {target_mode}")


def load_uci_dataset(cfg: DatasetConfig) -> DatasetBundle:
    dataset = fetch_ucirepo(id=cfg.dataset_id)

    X = dataset.data.features.copy()
    targets = dataset.data.targets

    if isinstance(targets, pd.DataFrame):
        y = targets.iloc[:, 0].copy()
    else:
        y = targets.copy()

    y = normalize_target(y, target_mode=cfg.target_mode)

    valid_mask = ~y.isna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    name = cfg.name or dataset.metadata.get("name", f"uci_{cfg.dataset_id}")

    return DatasetBundle(dataset_id=cfg.dataset_id, name=name, X=X, y=y)


def evaluate_model(
    data: DatasetBundle,
    algorithm_name: str,
    algorithm_cls: Callable[..., Any],
    random_state: int,
) -> dict[str, Any]:
    def _is_probably_multiclass_unsupported(exc: Exception) -> bool:
        msg = str(exc).lower()
        patterns = [
            "binary",
            "multiclass",
            "only supports",
            "not support",
            "unsupported target",
            "label type",
        ]
        return any(p in msg for p in patterns)

    n_classes_total = len(pd.unique(data.y))
    split_params = choose_split_params(len(data.X))

    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        random_state=random_state,
        stratify=data.y,
        **split_params,
    )

    # Preprocess data using RuleKitPreprocessor
    preprocessor = RuleKitPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Ensure y_train is a pandas Series with a name (required by RuleClassifier)
    if not isinstance(y_train, pd.Series):
        y_train_series = pd.Series(y_train, name="target")
    else:
        if y_train.name is None:
            y_train_series = y_train.copy()
            y_train_series.name = "target"
        else:
            y_train_series = y_train

    clf = instantiate_classifier(algorithm_cls, random_state=random_state)

    try:
        clf.fit(X_train_processed, y_train_series)
        y_pred = clf.predict(X_test_processed)
    except Exception as exc:
        if n_classes_total > 2 and _is_probably_multiclass_unsupported(exc):
            raise ValueError(
                f"Algorithm '{algorithm_name}' does not support multiclass on dataset "
                f"'{data.name}' (classes={n_classes_total}). Original error: {exc}"
            ) from exc
        raise

    avg_mode = "binary" if n_classes_total == 2 else "macro"
    f1 = f1_score(y_test, y_pred, average=avg_mode)

    model_size = estimate_model_size(clf)

    return {
        "dataset_id": data.dataset_id,
        "dataset": data.name,
        "n_samples": len(data.X),
        "n_features": data.X.shape[1],
        "algorithm": algorithm_name,
        "f1": float(f1),
        "f1_average": avg_mode,
        "model_size": model_size,
        "test_size": split_params["test_size"],
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
            raise ValueError(
                f"Invalid short-label format '{item}'. Expected: dataset_id:label"
            )
        ds_id_raw, short_label = item.split(":", 1)
        key_raw = ds_id_raw.strip()
        short_label = short_label.strip()
        if not key_raw or not short_label:
            raise ValueError(
                f"Invalid short-label format '{item}'. Expected: dataset_id:label or dataset_name:label"
            )

        if key_raw.isdigit():
            mapping_by_id[int(key_raw)] = short_label
        else:
            mapping_by_name[key_raw.strip().lower()] = short_label

    return mapping_by_id, mapping_by_name


def auto_short_dataset_name(dataset_name: str, dataset_id: int) -> str:
    """Generate a short readable label for unknown datasets."""
    tokens = [tok for tok in str(dataset_name).replace("-", "_").split("_") if tok]
    if not tokens:
        return f"DS{dataset_id}"

    if len(tokens) == 1:
        label = tokens[0]
        return label[:14] if len(label) > 14 else label

    if len(tokens) <= 3:
        label = "".join(tok[:5].capitalize() for tok in tokens)
        return label[:18]

    acronym = "".join(tok[0].upper() for tok in tokens if tok)
    return acronym if acronym else f"DS{dataset_id}"


def resolve_plot_dataset_label(
    dataset_id: int,
    dataset_name: str,
    short_names_by_id: dict[int, str],
    short_names_by_name: dict[str, str],
) -> str:
    if dataset_id in short_names_by_id:
        return short_names_by_id[dataset_id]

    clean_name = dataset_name.strip().lower()
    if clean_name in short_names_by_name:
        return short_names_by_name[clean_name]

    return auto_short_dataset_name(dataset_name, dataset_id)


def run_benchmarks(
    dataset_ids: list[int],
    algorithm_names: list[str],
    num_runs: int,
    output_csv: Path,
) -> pd.DataFrame:
    """Run benchmarks and save results to CSV."""
    results = []

    for dataset_id in dataset_ids:
        if dataset_id not in DEFAULT_DATASET_OPTIONS:
            print(f"Warning: dataset {dataset_id} not in DEFAULT_DATASET_OPTIONS, skipping.")
            continue

        cfg = DatasetConfig(**DEFAULT_DATASET_OPTIONS[dataset_id], dataset_id=dataset_id)

        try:
            print(f"Loading dataset {dataset_id} ({cfg.name})...")
            data = load_uci_dataset(cfg)
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}: {e}")
            continue

        for algo_name in algorithm_names:
            if algo_name not in ALGORITHM_REGISTRY:
                print(f"Warning: algorithm '{algo_name}' not in registry, skipping.")
                continue

            algo_cls = ALGORITHM_REGISTRY[algo_name]
            print(f"  Benchmarking {algo_name} ({num_runs} runs)...")

            for run_idx in range(num_runs):
                try:
                    result = evaluate_model(
                        data=data,
                        algorithm_name=algo_name,
                        algorithm_cls=algo_cls,
                        random_state=42 + run_idx,
                    )
                    results.append(result)
                    print(f"    Run {run_idx + 1}/{num_runs}: f1={result['f1']:.4f}, size={result['model_size']:.1f}")
                except Exception as e:
                    print(f"    Run {run_idx + 1}/{num_runs}: ERROR - {e}")

    df = pd.DataFrame(results)

    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw results into summary statistics."""
    if df.empty:
        return df

    agg_dict = {
        "f1": ["mean", "std"],
        "model_size": ["mean", "std"],
        "n_samples": "first",
        "n_features": "first",
    }

    grouped = df.groupby(["dataset_id", "dataset", "algorithm"]).agg(agg_dict).reset_index()
    grouped.columns = ["dataset_id", "dataset", "algorithm", "f1_mean", "f1_std", "model_size_mean", "model_size_std", "n_samples", "n_features"]

    def ci95(series: pd.Series) -> float:
        n = len(series)
        if n < 2:
            return float("nan")
        se = series.std() / math.sqrt(n)
        return 1.96 * se

    for idx, row in grouped.iterrows():
        mask = (df["dataset_id"] == row["dataset_id"]) & (df["algorithm"] == row["algorithm"])
        grouped.at[idx, "f1_ci95"] = ci95(df.loc[mask, "f1"])
        grouped.at[idx, "model_size_ci95"] = ci95(df.loc[mask, "model_size"])

    grouped["runs_total"] = grouped.apply(
        lambda row: len(df[(df["dataset_id"] == row["dataset_id"]) & (df["algorithm"] == row["algorithm"])]),
        axis=1
    )

    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RuleKit on UCI-ML-Repo datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rulekit_benchmark.py --datasets 17,45
  python run_rulekit_benchmark.py --datasets 17 --runs 5
  python run_rulekit_benchmark.py --algorithms RuleKit --datasets 17,45,53
        """,
    )

    parser.add_argument(
        "--datasets",
        default="17,45",
        help="Comma-separated UCI dataset IDs (default: 17,45)",
    )
    parser.add_argument(
        "--algorithms",
        default="RuleClassifier",
        help="Comma-separated algorithm names (default: RuleClassifier)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per dataset/algorithm pair (default: 3)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "rulekit" / "rulekit_results.csv",
        help="Output CSV file for raw results",
    )
    parser.add_argument(
        "--output-agg-csv",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "rulekit" / "rulekit_results_agg.csv",
        help="Output CSV file for aggregated results",
    )
    parser.add_argument(
        "--plot-data-csv",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "rulekit" / "rulekit_plot_data.csv",
        help="Output CSV file for plotting data",
    )

    args = parser.parse_args()

    dataset_ids = [int(x.strip()) for x in args.datasets.split(",") if x.strip()]
    algorithm_names = [x.strip() for x in args.algorithms.split(",") if x.strip()]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_agg_csv.parent.mkdir(parents=True, exist_ok=True)
    args.plot_data_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RuleKit UCI-ML Benchmark")
    print("=" * 60)
    print(f"Datasets: {dataset_ids}")
    print(f"Algorithms: {algorithm_names}")
    print(f"Runs per configuration: {args.runs}")
    print()

    df_raw = run_benchmarks(dataset_ids, algorithm_names, args.runs, args.output_csv)

    if not df_raw.empty:
        df_agg = aggregate_results(df_raw)
        df_agg.to_csv(args.output_agg_csv, index=False)
        print(f"Aggregated results saved to {args.output_agg_csv}")

        print("\nAggregated Results:")
        print(df_agg.to_string())


if __name__ == "__main__":
    main()












