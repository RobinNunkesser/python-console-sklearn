"""Benchmark RuleKit classifiers on UCI-ML-Repo datasets.

RuleKit is a Java-based rule learning system. This benchmark evaluates
the RuleKit classifier on the same UCI-ML-Repo datasets used for other models.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import math
import os
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
    17: {"name": "breast_cancer_wisconsin_diagnostic", "short_name": "BreastCancer", "target_mode": "auto"},
    45: {"name": "heart_disease", "short_name": "Heart", "target_mode": "auto"},
    12: {"name": "balance_scale", "short_name": "Balance", "target_mode": "auto"},
    19: {"name": "car_evaluation", "short_name": "Car", "target_mode": "auto"},
    53: {"name": "iris", "short_name": "Iris", "target_mode": "auto"},
    78: {"name": "page_blocks_classification", "short_name": "PageBlocks", "target_mode": "auto"},
    109: {"name": "wine", "short_name": "Wine", "target_mode": "auto"},
    267: {"name": "banknote_authentication", "short_name": "Banknote", "target_mode": "auto"},
}


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
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def choose_split_params(n_samples: int) -> dict[str, Any]:
    if n_samples < 500:
        return {"test_size": 0.30}
    if n_samples < 5_000:
        return {"test_size": 0.25}
    return {"test_size": 0.20}


class RuleKitPreprocessor:
    """Preprocessing pipeline compatible with RuleClassifier (expects plain DataFrames)."""

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


def instantiate_classifier(model_cls: Callable[..., Any], random_state: int) -> Any:
    try:
        sig = inspect.signature(model_cls)
        if "random_state" in sig.parameters:
            return model_cls(random_state=random_state)
    except (TypeError, ValueError):
        pass
    return model_cls()


def estimate_model_size(model: Any) -> float:
    """Return number of induced rules for RuleKit, with generic fallbacks."""
    # RuleKit: model.model is a RuleSet with a .rules property
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
            enc = LabelEncoder()
            return pd.Series(enc.fit_transform(y.astype(str)), index=y.index)
        return y
    raise ValueError(f"Unknown target_mode: {target_mode}")


def load_uci_dataset(cfg: DatasetConfig) -> DatasetBundle:
    dataset = fetch_ucirepo(id=cfg.dataset_id)
    X = dataset.data.features.copy()
    targets = dataset.data.targets
    y = targets.iloc[:, 0].copy() if isinstance(targets, pd.DataFrame) else targets.copy()
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
        return any(p in msg for p in ["binary", "multiclass", "only supports", "not support", "unsupported target", "label type"])

    n_classes_total = len(pd.unique(data.y))
    split_params = choose_split_params(len(data.X))

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, random_state=random_state, stratify=data.y, **split_params,
    )

    preprocessor = RuleKitPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # RuleClassifier requires a named pandas Series as labels
    if not isinstance(y_train, pd.Series):
        y_train_series = pd.Series(y_train, name="target")
    else:
        y_train_series = y_train.copy()
        if y_train_series.name is None:
            y_train_series.name = "target"

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

    return {
        "dataset_id": data.dataset_id,
        "dataset": data.name,
        "n_samples": len(data.X),
        "n_features": data.X.shape[1],
        "algorithm": algorithm_name,
        "f1": float(f1),
        "f1_average": avg_mode,
        "model_size": estimate_model_size(clf),
        "test_size": split_params["test_size"],
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_dataset_short_names(raw: str) -> tuple[dict[int, str], dict[str, str]]:
    mapping_by_id: dict[int, str] = {}
    mapping_by_name: dict[str, str] = {}
    if not raw.strip():
        return mapping_by_id, mapping_by_name
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid short-label format '{item}'. Expected: dataset_id:label")
        ds_id_raw, short_label = item.split(":", 1)
        key_raw, short_label = ds_id_raw.strip(), short_label.strip()
        if key_raw.isdigit():
            mapping_by_id[int(key_raw)] = short_label
        else:
            mapping_by_name[key_raw.lower()] = short_label
    return mapping_by_id, mapping_by_name


def auto_short_dataset_name(dataset_name: str, dataset_id: int) -> str:
    tokens = [tok for tok in str(dataset_name).replace("-", "_").split("_") if tok]
    if not tokens:
        return f"DS{dataset_id}"
    if len(tokens) == 1:
        label = tokens[0]
        return label[:14] if len(label) > 14 else label
    if len(tokens) <= 3:
        return "".join(tok[:5].capitalize() for tok in tokens)[:18]
    return "".join(tok[0].upper() for tok in tokens) or f"DS{dataset_id}"


def resolve_plot_dataset_label(
    dataset_id: int,
    dataset_name: str,
    default_short_names_by_id: dict[int, str],
    user_short_names_by_id: dict[int, str],
    user_short_names_by_name: dict[str, str],
) -> str:
    if dataset_id in user_short_names_by_id:
        return user_short_names_by_id[dataset_id]
    normalized = dataset_name.strip().lower()
    if normalized in user_short_names_by_name:
        return user_short_names_by_name[normalized]
    if dataset_id in default_short_names_by_id:
        return default_short_names_by_id[dataset_id]
    return auto_short_dataset_name(dataset_name, dataset_id)


def build_dataset_configs(dataset_ids: list[int]) -> list[DatasetConfig]:
    configs: list[DatasetConfig] = []
    for dataset_id in dataset_ids:
        defaults = DEFAULT_DATASET_OPTIONS.get(dataset_id, {})
        configs.append(DatasetConfig(
            dataset_id=dataset_id,
            name=defaults.get("name", f"uci_{dataset_id}"),
            short_name=defaults.get("short_name"),
            target_mode=defaults.get("target_mode", "auto"),
        ))
    return configs


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
    ).reset_index()

    agg_df["f1_std"] = agg_df["f1_std"].fillna(0.0)
    agg_df["model_size_std"] = agg_df["model_size_std"].fillna(0.0)

    f1_den = agg_df["f1_n"].where(agg_df["f1_n"] > 0, 1) ** 0.5
    ms_den = agg_df["model_size_n"].where(agg_df["model_size_n"] > 0, 1) ** 0.5
    agg_df["f1_ci95"] = 1.96 * agg_df["f1_std"] / f1_den
    agg_df["model_size_ci95"] = 1.96 * agg_df["model_size_std"] / ms_den

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


def plot_results(
    agg_df: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    no_show: bool,
    error_bars: str,
    plot_style: str,
) -> None:
    if agg_df.empty:
        return
    plot_benchmark_results(
        agg_df,
        dataset_label_col="plot_dataset" if "plot_dataset" in agg_df.columns else "dataset",
        metrics=UCI_METRICS,
        output_dir=output_dir,
        output_basename_prefix="rulekit",
        plot_mode=plot_mode,
        error_bars=error_bars,
        plot_style=plot_style,
        no_show=no_show,
    )


def run_benchmark(
    dataset_ids: list[int],
    algorithm_names: list[str],
    n_runs: int,
    random_state: int,
    user_short_names_by_id: dict[int, str],
    user_short_names_by_name: dict[str, str],
    output_dir: Path,
    plot_mode: str,
    plot_style: str,
    no_show: bool,
    error_bars: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_configs = build_dataset_configs(dataset_ids)
    default_short_names_by_id = {
        cfg.dataset_id: cfg.short_name for cfg in dataset_configs if cfg.short_name
    }

    rows: list[dict[str, Any]] = []

    for ds_cfg in dataset_configs:
        print(f"\n--- Loading dataset {ds_cfg.dataset_id}: {ds_cfg.name} ---")
        try:
            data = load_uci_dataset(ds_cfg)
        except Exception as exc:
            print(f"  Failed to load: {exc}")
            continue
        class_count = len(pd.unique(data.y))
        print(f"  Samples={len(data.X)}, Features={data.X.shape[1]}, Classes={class_count}")

        for algo_name in algorithm_names:
            if algo_name not in ALGORITHM_REGISTRY:
                print(f"  Unknown algorithm '{algo_name}', skipping.")
                continue
            algo_cls = ALGORITHM_REGISTRY[algo_name]

            for run_idx in range(n_runs):
                seed = random_state + run_idx
                print(f"  -> {algo_name} (run {run_idx + 1}/{n_runs}, seed={seed}) ...", end=" ", flush=True)
                try:
                    row = evaluate_model(data, algo_name, algo_cls, random_state=seed)
                    row["run_idx"] = run_idx
                    row["seed"] = seed
                    row["error"] = ""
                    rows.append(row)
                    size_txt = f"{row['model_size']:.1f}" if pd.notna(row["model_size"]) else "n/a"
                    print(f"ok | F1={row['f1']:.4f}, model_size={size_txt}")
                except Exception as exc:
                    print(f"failed ({type(exc).__name__}: {exc})")
                    rows.append({
                        "dataset_id": data.dataset_id,
                        "dataset": data.name,
                        "n_samples": len(data.X),
                        "n_features": data.X.shape[1],
                        "algorithm": algo_name,
                        "f1": float("nan"),
                        "f1_average": "n/a",
                        "model_size": float("nan"),
                        "test_size": choose_split_params(len(data.X))["test_size"],
                        "run_idx": run_idx,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    results_df = pd.DataFrame(rows)
    agg_df = aggregate_results(results_df)

    # Add plot_dataset labels
    if not agg_df.empty:
        agg_df["plot_dataset"] = agg_df.apply(
            lambda row: resolve_plot_dataset_label(
                dataset_id=int(row["dataset_id"]),
                dataset_name=str(row["dataset"]),
                default_short_names_by_id=default_short_names_by_id,
                user_short_names_by_id=user_short_names_by_id,
                user_short_names_by_name=user_short_names_by_name,
            ),
            axis=1,
        )

    plot_export_df = build_plot_export_df(agg_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_raw = output_dir / "rulekit_results.csv"
    results_df.to_csv(csv_raw, index=False)
    print(f"\nCSV saved (raw):       {csv_raw}")

    csv_agg = output_dir / "rulekit_results_agg.csv"
    agg_df.to_csv(csv_agg, index=False)
    print(f"CSV saved (aggregate): {csv_agg}")

    csv_plot = output_dir / "rulekit_plot_data.csv"
    plot_export_df.to_csv(csv_plot, index=False)
    print(f"CSV saved (plot data): {csv_plot}")

    plot_results(agg_df, output_dir=output_dir, plot_mode=plot_mode,
                 no_show=no_show, error_bars=error_bars, plot_style=plot_style)

    return results_df, agg_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RuleKit UCI-ML benchmark")
    parser.add_argument(
        "--dataset-ids", default="12,17,19,45,53,78,109,267",
        help="Comma-separated UCI dataset IDs (default: 12,17,19,45,53,78,109,267)",
    )
    parser.add_argument(
        "--algorithms", default="RuleClassifier",
        help="Comma-separated algorithm names (default: RuleClassifier)",
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Runs per dataset/algorithm (default: 10)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--plot-mode", default="combined", choices=["combined", "separate", "by_dataset"],
    )
    parser.add_argument("--plot-style", default="dots", choices=["dots", "bars"])
    parser.add_argument("--error-bars", default="none", choices=["none", "std", "ci95"])
    parser.add_argument(
        "--dataset-short-names", default="",
        help="Optional short labels, e.g. 17:BreastCancer,heart_disease:Heart",
    )
    parser.add_argument("--output-dir", default="benchmarks/outputs/rulekit")
    parser.add_argument("--no-show", action="store_true", help="Save plots only, do not display them")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_ids = [int(s) for s in parse_csv_list(args.dataset_ids)]
    algorithm_names = parse_csv_list(args.algorithms)
    user_short_names_by_id, user_short_names_by_name = parse_dataset_short_names(args.dataset_short_names)

    print("=" * 60)
    print("RuleKit UCI-ML Benchmark")
    print("=" * 60)
    print(f"Datasets:   {dataset_ids}")
    print(f"Algorithms: {algorithm_names}")
    print(f"Runs:       {args.n_runs}  (random_state={args.random_state})")

    results_df, agg_df = run_benchmark(
        dataset_ids=dataset_ids,
        algorithm_names=algorithm_names,
        n_runs=args.n_runs,
        random_state=args.random_state,
        user_short_names_by_id=user_short_names_by_id,
        user_short_names_by_name=user_short_names_by_name,
        output_dir=Path(args.output_dir),
        plot_mode=args.plot_mode,
        plot_style=args.plot_style,
        no_show=args.no_show,
        error_bars=args.error_bars,
    )

    if not results_df.empty:
        print("\nResults (raw, excerpt):")
        print(results_df[["dataset", "algorithm", "seed", "f1", "model_size"]].head(12).to_string(index=False))

    if not agg_df.empty:
        print("\nResults (aggregate):")
        show_cols = ["dataset", "algorithm", "runs_total", "f1_mean", "f1_std", "f1_ci95",
                     "model_size_mean", "model_size_std", "model_size_ci95"]
        show_cols = [c for c in show_cols if c in agg_df.columns]
        print(agg_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

