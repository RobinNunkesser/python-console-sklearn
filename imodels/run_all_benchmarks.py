#!/usr/bin/env python3
"""Run all benchmark scripts sequentially.

Defaults are aligned with the existing benchmark scripts and run in this order:
1) UCI imodels
2) UCI ExSTraCS
3) UCI RuleKit
4) LogicGP plot-data generation
5) UCI merged plots
6) Multiplexer imodels+ExSTraCS
7) Multiplexer RuleKit
8) Multiplexer merged plots
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    script: str


STEPS: list[Step] = [
    Step("uci_imodels", "UCI: imodels", "benchmarks/uci/run_imodels_benchmark.py"),
    Step("uci_exstracs", "UCI: ExSTraCS", "benchmarks/uci/run_exstracs_benchmark.py"),
    Step("uci_rulekit", "UCI: RuleKit", "benchmarks/uci/run_rulekit_benchmark.py"),
    Step("logicgp_plot_data", "LogicGP: Build plot data", "logicgp_make_plot_data.py"),
    Step("uci_merge", "UCI: Merge plots", "benchmarks/uci/merge_benchmark_plots.py"),
    Step("mux_imodels", "Multiplexer: imodels+ExSTraCS", "benchmarks/multiplexer/run_multiplexer_benchmark.py"),
    Step("mux_rulekit", "Multiplexer: RuleKit", "benchmarks/multiplexer/run_rulekit_multiplexer_benchmark.py"),
    Step("mux_merge", "Multiplexer: Merge plots", "benchmarks/multiplexer/merge_benchmark_plots.py"),
]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all benchmark scripts sequentially")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run benchmark scripts (default: current interpreter)",
    )
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Optional comma-separated step keys to run. "
            "Available: " + ",".join(step.key for step in STEPS)
        ),
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip both merge plot scripts",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use minimal quick settings where possible (dataset/runs reduced)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Do not add --no-show (matplotlib windows may open)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining scripts even if one step fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser


def should_use_no_show(step: Step) -> bool:
    # Data-prep-only steps do not expose --no-show.
    return step.key not in {"logicgp_plot_data"}


def quick_args_for_step(step: Step) -> list[str]:
    if step.key == "uci_imodels":
        return ["--dataset-ids", "17", "--algorithms", "OneRClassifier", "--n-runs", "1"]
    if step.key == "uci_exstracs":
        return ["--dataset-ids", "17", "--algorithms", "ExSTraCS_QRF", "--n-runs", "1"]
    if step.key == "uci_rulekit":
        return ["--dataset-ids", "17", "--n-runs", "1"]
    if step.key == "mux_imodels":
        return ["--datasets", "Multiplexer6", "--algorithms", "OneRClassifier", "--n-runs", "1"]
    if step.key == "mux_rulekit":
        return ["--datasets", "Multiplexer6", "--n-runs", "1"]
    return []


def build_commands(args: argparse.Namespace) -> list[tuple[Step, list[str]]]:
    steps = STEPS

    if args.only:
        wanted = set(parse_csv_list(args.only))
        known = {step.key for step in STEPS}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown step key(s): {', '.join(unknown)}")
        steps = [step for step in steps if step.key in wanted]

    if args.skip_merge:
        steps = [step for step in steps if not step.key.endswith("merge")]

    commands: list[tuple[Step, list[str]]] = []
    for step in steps:
        cmd = [args.python, step.script]
        if args.quick:
            cmd.extend(quick_args_for_step(step))
        if not args.show and should_use_no_show(step):
            cmd.append("--no-show")
        commands.append((step, cmd))
    return commands


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent

    commands = build_commands(args)
    if not commands:
        print("No steps selected. Nothing to do.")
        return 0

    failures: list[str] = []
    started = time.time()

    print("Running benchmark pipeline in order:")
    for i, (step, cmd) in enumerate(commands, start=1):
        cmd_txt = " ".join(cmd)
        print(f"\n[{i}/{len(commands)}] {step.label}")
        print(f"$ {cmd_txt}")

        if args.dry_run:
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=repo_root)
        dt = time.time() - t0

        if result.returncode == 0:
            print(f"-> OK ({dt:.1f}s)")
            continue

        print(f"-> FAILED with exit code {result.returncode} ({dt:.1f}s)")
        failures.append(step.key)

        if not args.continue_on_error:
            break

    total = time.time() - started
    print(f"\nDone in {total:.1f}s")

    if failures:
        print("Failed steps: " + ", ".join(failures))
        return 1

    print("All selected steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

