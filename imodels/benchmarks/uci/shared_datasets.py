"""Shared UCI dataset helpers for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

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


def choose_split_params(n_samples: int) -> dict[str, Any]:
    if n_samples < 500:
        return {"test_size": 0.30}
    if n_samples < 5_000:
        return {"test_size": 0.25}
    return {"test_size": 0.20}


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
            raise ValueError(
                f"Invalid short-label format '{item}'. Expected: dataset_id:label or dataset_name:label"
            )
        key_raw, label = item.split(":", 1)
        key_raw = key_raw.strip()
        label = label.strip()
        if not key_raw or not label:
            raise ValueError(f"Invalid short-label format '{item}'.")
        if key_raw.isdigit():
            mapping_by_id[int(key_raw)] = label
        else:
            mapping_by_name[key_raw.lower()] = label
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
    acronym = "".join(tok[0].upper() for tok in tokens if tok)
    return acronym if acronym else f"DS{dataset_id}"


def resolve_plot_dataset_label(
    dataset_id: int,
    dataset_name: str,
    default_short_names_by_id: dict[int, str],
    user_short_names_by_id: dict[int, str],
    user_short_names_by_name: dict[str, str],
) -> str:
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

