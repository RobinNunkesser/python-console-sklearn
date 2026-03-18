"""Benchmark imodels classifiers on UCI-ML-Repo datasets.

By default, datasets 17 (Breast Cancer Wisconsin Diagnostic)
and 45 (Heart Disease) are loaded, train/test splits are chosen based on
dataset size, and multiple algorithms are evaluated sequentially.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
from imodels import (
    C45TreeClassifier,
    DecisionTreeClassifier,
    FIGSClassifier,
    GreedyRuleListClassifier,
    GreedyTreeClassifier,
    HSTreeClassifier,
    OneRClassifier,
    TaoTreeClassifier,
    SlipperClassifier,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo


ALGORITHM_REGISTRY: dict[str, Callable[..., Any]] = {
    # --- multiclass-capable classifiers (verified on 3-class iris) ---
    "C45TreeClassifier": C45TreeClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "FIGSClassifier": FIGSClassifier,
    "GreedyRuleListClassifier": GreedyRuleListClassifier,
    "GreedyTreeClassifier": GreedyTreeClassifier,
    "HSTreeClassifier": HSTreeClassifier,
    "OneRClassifier": OneRClassifier,
    "SlipperClassifier": SlipperClassifier,
    "TaoTreeClassifier": TaoTreeClassifier,
    # --- excluded (binary-only or broken) ---
    # BoostedRulesClassifier:    removed on user request
    # TreeGAMClassifier:         removed on user request
    # BayesianRuleListClassifier: binary only
    # BayesianRuleSetClassifier:  binary only / runtime error
    # DecisionTreeCCPClassifier:  requires pre-fitted estimator_
    # FPLassoClassifier:          multiclass not supported (RuleFit base)
    # FPSkopeClassifier:          requires boolean DataFrame input
    # HSOptimalTreeClassifier:    requires pre-fitted estimator_
    # RuleFitClassifier:          multiclass not supported
    # SkopeRulesClassifier:       collapses all non-zero labels to one class
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
    # Heart Disease remains multiclass by default (original classes).
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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


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
        # Encode string/categorical targets; keep numeric classes unchanged.
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

    clf = instantiate_classifier(algorithm_cls, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(data.X)),
            ("model", clf),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    except Exception as exc:
        if n_classes_total > 2 and _is_probably_multiclass_unsupported(exc):
            raise ValueError(
                f"Algorithm '{algorithm_name}' does not support multiclass on dataset "
                f"'{data.name}' (classes={n_classes_total}). Original error: {exc}"
            ) from exc
        raise

    avg_mode = "binary" if n_classes_total == 2 else "macro"
    f1 = f1_score(y_test, y_pred, average=avg_mode)

    model_size = estimate_model_size(pipeline.named_steps["model"])

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
    default_short_names_by_id: dict[int, str],
    user_short_names_by_id: dict[int, str],
    user_short_names_by_name: dict[str, str],
) -> str:
    """Apply label priority: user-id > user-name > default-id > auto."""
    if dataset_id in user_short_names_by_id:
        return user_short_names_by_id[dataset_id]

    normalized_name = str(dataset_name).strip().lower()
    if normalized_name in user_short_names_by_name:
        return user_short_names_by_name[normalized_name]

    if dataset_id in default_short_names_by_id:
        return default_short_names_by_id[dataset_id]

    return auto_short_dataset_name(dataset_name, dataset_id)


def build_dataset_configs(dataset_ids: list[int]) -> list[DatasetConfig]:
    configs: list[DatasetConfig] = []
    for dataset_id in dataset_ids:
        defaults = DEFAULT_DATASET_OPTIONS.get(dataset_id, {})
        configs.append(
            DatasetConfig(
                dataset_id=dataset_id,
                name=defaults.get("name", f"uci_{dataset_id}"),
                short_name=defaults.get("short_name"),
                target_mode=defaults.get("target_mode", "auto"),
            )
        )
    return configs


def plot_metric(
    results_df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    title: str,
    ylabel: str,
    error_bars: str = "none",
) -> None:
    mean_col = f"{metric}_mean" if f"{metric}_mean" in results_df.columns else metric
    dataset_label_col = "plot_dataset" if "plot_dataset" in results_df.columns else "dataset"
    # Show performance per dataset (x-axis), grouped by algorithm.
    pivot = results_df.pivot(index=dataset_label_col, columns="algorithm", values=mean_col)

    yerr = None
    if error_bars != "none":
        err_col = f"{metric}_{error_bars}"
        if err_col in results_df.columns:
            yerr = results_df.pivot(index=dataset_label_col, columns="algorithm", values=err_col).reindex_like(pivot)

    plot_kwargs: dict[str, Any] = {"kind": "bar", "ax": ax}
    if yerr is not None:
        plot_kwargs["yerr"] = yerr
        plot_kwargs["capsize"] = 4

    pivot.plot(**plot_kwargs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Algorithm")


def plot_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    no_show: bool,
    error_bars: str,
) -> None:
    if results_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_mode == "combined":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        f1_ylabel = "F1" if error_bars == "none" else f"F1 (mean +/- {error_bars})"
        size_ylabel = "Model Size" if error_bars == "none" else f"Model Size (mean +/- {error_bars})"
        plot_metric(results_df, "f1", axes[0], "F1 by Dataset and Algorithm", f1_ylabel, error_bars)
        plot_metric(results_df, "model_size", axes[1], "Model Size", size_ylabel, error_bars)
        combined_path = output_dir / "uci_imodels_combined.png"
        fig.savefig(combined_path, dpi=150)
        print(f"Figure saved: {combined_path}")
        if no_show:
            plt.close(fig)
        else:
            plt.show()
        return

    if plot_mode == "separate":
        fig_f1, ax_f1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
        f1_ylabel = "F1" if error_bars == "none" else f"F1 (mean +/- {error_bars})"
        plot_metric(results_df, "f1", ax_f1, "F1 by Dataset and Algorithm", f1_ylabel, error_bars)
        f1_path = output_dir / "uci_imodels_f1.png"
        fig_f1.savefig(f1_path, dpi=150)
        print(f"Figure saved: {f1_path}")

        fig_size, ax_size = plt.subplots(figsize=(8, 5), constrained_layout=True)
        size_ylabel = "Model Size" if error_bars == "none" else f"Model Size (mean +/- {error_bars})"
        plot_metric(results_df, "model_size", ax_size, "Model Size", size_ylabel, error_bars)
        size_path = output_dir / "uci_imodels_model_size.png"
        fig_size.savefig(size_path, dpi=150)
        print(f"Figure saved: {size_path}")

        if no_show:
            plt.close(fig_f1)
            plt.close(fig_size)
        else:
            plt.show()
        return

    raise ValueError(f"Unknown plot_mode: {plot_mode}")


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


def normal_cdf(x: float) -> float:
    """CDF of the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_ttest_normal_approx(diffs: pd.Series) -> tuple[float, float, float]:
    """Approximate paired t-test (two-sided) without external stats dependencies.

    Returns: (t_stat, p_value, mean_diff)
    """
    clean = diffs.dropna().astype(float)
    n = len(clean)
    if n < 2:
        return float("nan"), float("nan"), float("nan")

    mean_diff = float(clean.mean())
    std_diff = float(clean.std(ddof=1))

    if std_diff == 0.0:
        # No variance: all run-wise differences are identical.
        if mean_diff == 0.0:
            return 0.0, 1.0, mean_diff
        return float("inf") if mean_diff > 0 else float("-inf"), 0.0, mean_diff

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    # Normal approximation for p-value, robust without SciPy.
    p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return float(t_stat), float(p_value), mean_diff


def compute_significance_pairs(results_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Pairwise significance comparisons per dataset using shared seeds."""
    if results_df.empty:
        return pd.DataFrame()

    required_cols = {"dataset_id", "dataset", "algorithm", "seed", "f1"}
    if not required_cols.issubset(results_df.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    clean_df = results_df[results_df["f1"].notna()].copy()

    for (dataset_id, dataset_name), ds_df in clean_df.groupby(["dataset_id", "dataset"], dropna=False):
        algorithms = sorted(ds_df["algorithm"].dropna().unique().tolist())
        if len(algorithms) < 2:
            continue

        for algo_a, algo_b in itertools.combinations(algorithms, 2):
            a_df = ds_df[ds_df["algorithm"] == algo_a][["seed", "f1"]].rename(columns={"f1": "f1_a"})
            b_df = ds_df[ds_df["algorithm"] == algo_b][["seed", "f1"]].rename(columns={"f1": "f1_b"})
            paired = a_df.merge(b_df, on="seed", how="inner").dropna()

            if paired.empty:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset": dataset_name,
                        "algorithm_a": algo_a,
                        "algorithm_b": algo_b,
                        "n_pairs": 0,
                        "mean_f1_a": float("nan"),
                        "mean_f1_b": float("nan"),
                        "mean_diff_a_minus_b": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "alpha": alpha,
                        "significant": False,
                        "winner": "n/a",
                    }
                )
                continue

            diffs = paired["f1_a"] - paired["f1_b"]
            t_stat, p_value, mean_diff = paired_ttest_normal_approx(diffs)

            if math.isnan(mean_diff):
                winner = "n/a"
            elif mean_diff > 0:
                winner = algo_a
            elif mean_diff < 0:
                winner = algo_b
            else:
                winner = "tie"

            significant = bool(pd.notna(p_value) and p_value < alpha)

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": dataset_name,
                    "algorithm_a": algo_a,
                    "algorithm_b": algo_b,
                    "n_pairs": int(len(paired)),
                    "mean_f1_a": float(paired["f1_a"].mean()),
                    "mean_f1_b": float(paired["f1_b"].mean()),
                    "mean_diff_a_minus_b": float(mean_diff),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "alpha": alpha,
                    "significant": significant,
                    "winner": winner,
                }
            )

    return pd.DataFrame(rows)


def run_benchmark(
    dataset_ids: list[int],
    algorithm_names: list[str],
    random_state: int,
    n_runs: int,
    dataset_short_names_by_id: dict[int, str],
    dataset_short_names_by_name: dict[str, str],
    output_dir: Path,
    plot_mode: str,
    no_show: bool,
    error_bars: str,
    significance_check: bool,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_configs = build_dataset_configs(dataset_ids)
    default_short_names_by_id = {
        cfg.dataset_id: cfg.short_name
        for cfg in dataset_configs
        if cfg.short_name
    }

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    invalid_algorithms = [name for name in algorithm_names if name not in ALGORITHM_REGISTRY]
    if invalid_algorithms:
        known = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithms: {invalid_algorithms}. Available: {known}")

    rows: list[dict[str, Any]] = []

    for ds_cfg in dataset_configs:
        print(f"\n--- Loading dataset {ds_cfg.dataset_id}: {ds_cfg.name} ---")
        data = load_uci_dataset(ds_cfg)
        class_count = len(pd.unique(data.y))
        print(
            f"Samples={len(data.X)}, Features={data.X.shape[1]}, Classes={class_count}, "
            f"target_mode={ds_cfg.target_mode}"
        )

        for algo_name in algorithm_names:
            algo_cls = ALGORITHM_REGISTRY[algo_name]
            for run_idx in range(n_runs):
                seed = random_state + run_idx
                print(f"  -> Training {algo_name} (run {run_idx + 1}/{n_runs}, seed={seed}) ...", end=" ")
                try:
                    row = evaluate_model(data, algo_name, algo_cls, random_state=seed)
                    row["run_idx"] = run_idx
                    row["seed"] = seed
                    row["error"] = ""
                    rows.append(row)
                    size_txt = (
                        f"{row['model_size']:.1f}"
                        if pd.notna(row["model_size"])
                        else "n/a"
                    )
                    print(f"ok | F1={row['f1']:.4f}, model_size={size_txt}")
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
                            "test_size": choose_split_params(len(data.X))["test_size"],
                            "run_idx": run_idx,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    results_df = pd.DataFrame(rows)
    agg_df = aggregate_results(results_df)

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

    significance_df = compute_significance_pairs(results_df, alpha=alpha) if significance_check else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_raw_path = output_dir / "uci_imodels_results.csv"
    results_df.to_csv(csv_raw_path, index=False)
    print(f"\nCSV saved (raw): {csv_raw_path}")

    csv_agg_path = output_dir / "uci_imodels_results_agg.csv"
    agg_df.to_csv(csv_agg_path, index=False)
    print(f"CSV saved (aggregate): {csv_agg_path}")

    csv_plot_path = output_dir / "uci_imodels_plot_data.csv"
    plot_export_df.to_csv(csv_plot_path, index=False)
    print(f"CSV saved (plot data): {csv_plot_path}")

    if significance_check:
        csv_sig_path = output_dir / "uci_imodels_significance.csv"
        significance_df.to_csv(csv_sig_path, index=False)
        print(f"CSV saved (significance): {csv_sig_path}")

    plot_results(agg_df, output_dir=output_dir, plot_mode=plot_mode, no_show=no_show, error_bars=error_bars)
    return results_df, agg_df, significance_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UCI + imodels benchmark")
    parser.add_argument(
        "--dataset-ids",
        default="12,17,19,45,53,78,109,267",
        help="Comma-separated list of UCI dataset IDs (e.g., 17,45)",
    )
    parser.add_argument(
        "--algorithms",
        default="GreedyRuleListClassifier,GreedyTreeClassifier,OneRClassifier,SlipperClassifier,HSTreeClassifier",
        help="Comma-separated list of algorithm names",
    )
    parser.add_argument(
        "--plot-mode",
        default="combined",
        choices=["combined", "separate"],
        help="combined = one figure with two panels, separate = two figures",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of runs per dataset+algorithm with seeds random_state+i",
    )
    parser.add_argument(
        "--error-bars",
        default="none",
        choices=["none", "std", "ci95"],
        help="Error bars in plots: none, standard deviation, or 95%% confidence interval",
    )
    parser.add_argument(
        "--dataset-short-names",
        default="",
        help=(
            "Optional short plot labels as a list, e.g. "
            "17:BreastCancer,heart_disease:Heart"
        ),
    )
    parser.add_argument(
        "--significance-check",
        action="store_true",
        help="Optionally run pairwise significance tests per dataset",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the optional test (default 0.05)",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show matplotlib windows (save files only)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_ids = [int(s) for s in parse_csv_list(args.dataset_ids)]
    algorithm_names = parse_csv_list(args.algorithms)
    dataset_short_names_by_id, dataset_short_names_by_name = parse_dataset_short_names(args.dataset_short_names)

    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    results_df, agg_df, significance_df = run_benchmark(
        dataset_ids=dataset_ids,
        algorithm_names=algorithm_names,
        random_state=args.random_state,
        n_runs=args.n_runs,
        dataset_short_names_by_id=dataset_short_names_by_id,
        dataset_short_names_by_name=dataset_short_names_by_name,
        output_dir=Path(args.output_dir),
        plot_mode=args.plot_mode,
        no_show=args.no_show,
        error_bars=args.error_bars,
        significance_check=args.significance_check,
        alpha=args.alpha,
    )

    print("\nResults (raw, excerpt):")
    print(results_df[["dataset", "algorithm", "seed", "f1", "model_size"]].head(12).to_string(index=False))

    print("\nResults (aggregate):")
    print(
        agg_df[
            [
                "dataset",
                "algorithm",
                "runs_total",
                "f1_mean",
                "f1_std",
                "f1_ci95",
                "model_size_mean",
                "model_size_std",
                "model_size_ci95",
            ]
        ].to_string(index=False)
    )

    if args.significance_check:
        print("\nSignificance (pairwise, two-sided, normal approximation):")
        if significance_df.empty:
            print("No pairwise comparisons available.")
        else:
            print(
                significance_df[
                    [
                        "dataset",
                        "algorithm_a",
                        "algorithm_b",
                        "n_pairs",
                        "mean_diff_a_minus_b",
                        "p_value",
                        "alpha",
                        "significant",
                        "winner",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()


