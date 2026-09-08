"""Monitoring commands.

    python -m risk_scoring.monitoring reference [--population baseline]

``reference`` rebuilds the registered model's training window from the
frozen export and stores it as the thing every later window is compared
against. It is run once per registered model version, against the
database the replay will write into.

Judgment calls this module fixes:

- The model version defaults to the pin in ``configs/service.toml``, so a
  reference and the model actually serving cannot silently disagree.
  ``--model-version`` overrides it for building a reference ahead of a
  promotion.
- Re-running for a version that already has a reference prints what is
  stored and exits without writing. A reference is what a finished run's
  evaluations name, so overwriting one would change the meaning of rows
  already written. Building a reference for a *new* version is the
  supported move.
- The population is a command-line choice rather than a config value.
  The reference must be built from the export the model was trained on,
  which the training run itself records; the flag exists so an operator
  can point at a relocated data root, not so the choice can drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import psycopg

from risk_scoring.db import database_url
from risk_scoring.monitoring.reference import (
    build_reference,
    describe,
    read_reference,
    record_reference,
)
from risk_scoring.service.config import DEFAULT_CONFIG_RELPATH, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.monitoring",
        description="Build and store what monitoring compares a window against.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reference = sub.add_parser(
        "reference", help="store the registered model's training window as the reference"
    )
    reference.add_argument("--population", default="baseline")
    reference.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="the registered version to build for; defaults to the pin in configs/service.toml",
    )
    reference.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_RELPATH,
        help="the service config the model version is read from",
    )
    return parser


def _connect() -> psycopg.Connection[Any]:
    return psycopg.connect(database_url(), connect_timeout=5)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()

    if args.command == "reference":
        model_version = args.model_version
        if model_version is None:
            config_path = repo_root / args.config
            if not config_path.is_file():
                sys.exit(f"no service config at {config_path}; pass --model-version instead")
            model_version = load_config(config_path).model_version

        csv_dir = repo_root / "data" / args.population / "csv"
        if not csv_dir.is_dir():
            sys.exit(f"no CSV export at {csv_dir}; generate the population first")

        with _connect() as conn:
            stored = read_reference(conn, "readmission-risk", model_version)
            if stored is not None:
                print(describe(stored, stored.reference_id))
                print("\nalready stored; a promotion builds a reference for the new version")
                return
            try:
                built = build_reference(csv_dir, repo_root, model_version=model_version)
            except LookupError as exc:
                sys.exit(str(exc))
            reference_id = record_reference(conn, built)
        print(describe(built, reference_id))


if __name__ == "__main__":
    main()
