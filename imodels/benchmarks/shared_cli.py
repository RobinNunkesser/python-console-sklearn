"""Shared CLI/path helpers for benchmark scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

