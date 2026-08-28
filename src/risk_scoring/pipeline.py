"""Retrain entrypoint: raw CSVs to a registered, gated model in one command.

    python -m risk_scoring.pipeline retrain [--population baseline]
        [--report gate_report.md]

Training and gating are each runnable on their own; this module is the
seam that chains them so a retrain is a single unattended act. Judgment
calls it fixes:

- The gate runs against the version this command just registered, never
  merely the newest version in the registry.
- A failing gate exits non-zero, but the candidate stays registered and
  the report is still written. Suppressing a failed candidate would hide
  the evidence the gate exists to produce.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from risk_scoring.gate import GateRunOutcome, run_gate
from risk_scoring.train import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    TRAINING_CUTOFF,
    TrainingResult,
    train,
)


@dataclass(frozen=True)
class RetrainOutcome:
    """What one chained retrain produced."""

    training: TrainingResult
    gate: GateRunOutcome


def retrain(
    csv_dir: Path,
    repo_root: Path,
    *,
    cutoff: pd.Timestamp | None = None,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
    report_path: Path | None = None,
) -> RetrainOutcome:
    """Train, register, then gate the version just registered."""
    cutoff = pd.Timestamp(TRAINING_CUTOFF, tz="UTC") if cutoff is None else cutoff
    training = train(
        csv_dir, repo_root, cutoff=cutoff, seed=seed, holdout_fraction=holdout_fraction
    )
    gate_outcome = run_gate(
        csv_dir,
        repo_root,
        model_version=training.model_version,
        report_path=report_path,
    )
    return RetrainOutcome(training=training, gate=gate_outcome)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.pipeline",
        description="Retrain the readmission model and gate the result in one command.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("retrain", help="train, register, and gate from raw CSVs")
    run_parser.add_argument("--population", default="baseline")
    run_parser.add_argument("--cutoff", default=TRAINING_CUTOFF)
    run_parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    run_parser.add_argument("--holdout", type=float, default=HOLDOUT_FRACTION)
    run_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    csv_dir = repo_root / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")
    outcome = retrain(
        csv_dir,
        repo_root,
        cutoff=pd.Timestamp(args.cutoff, tz="UTC"),
        seed=args.seed,
        holdout_fraction=args.holdout,
        report_path=args.report,
    )
    print()
    print(outcome.gate.report)
    if outcome.gate.result.verdict != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
