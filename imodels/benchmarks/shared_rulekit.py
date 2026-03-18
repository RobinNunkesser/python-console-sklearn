"""Shared RuleKit setup and preprocessing helpers."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from benchmarks.shared_models import make_one_hot_encoder


def ensure_java_home() -> None:
    """Best-effort JAVA_HOME setup for local macOS usage without overriding existing config."""
    if os.environ.get("JAVA_HOME"):
        return
    java_home = os.popen("/usr/libexec/java_home 2>/dev/null").read().strip()
    if java_home:
        os.environ["JAVA_HOME"] = java_home


class RuleKitPreprocessor:
    """Preprocessing pipeline compatible with RuleClassifier (expects plain DataFrames)."""

    def __init__(self):
        self.numeric_imputer: SimpleImputer | None = None
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
        if self.numeric_cols and self.numeric_imputer is not None:
            arr = self.numeric_imputer.transform(X[self.numeric_cols])
            for i, col in enumerate(self.numeric_cols):
                result[col] = arr[:, i]
        if self.categorical_cols and self.encoder is not None:
            arr = self.encoder.transform(X[self.categorical_cols].fillna("missing"))
            for i, name in enumerate(self.encoder.get_feature_names_out(self.categorical_cols)):
                result[name] = arr[:, i]
        return pd.DataFrame(result)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


def ensure_named_target(y: pd.Series | Any, name: str = "target") -> pd.Series:
    if not isinstance(y, pd.Series):
        return pd.Series(y, name=name)
    y_named = y.copy()
    if y_named.name is None:
        y_named.name = name
    return y_named

