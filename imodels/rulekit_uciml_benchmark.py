"""Compatibility wrapper for the moved RuleKit benchmark entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "benchmarks" / "uci" / "run_rulekit_benchmark.py"


if __name__ == "__main__":
    sys.path.insert(0, str(TARGET.parent))
    runpy.run_path(str(TARGET), run_name="__main__")
