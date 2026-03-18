"""Compatibility wrapper for the moved merged-plot entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "benchmarks" / "uci" / "merge_benchmark_plots.py"


if __name__ == "__main__":
    sys.path.insert(0, str(TARGET.parent))
    runpy.run_path(str(TARGET), run_name="__main__")
