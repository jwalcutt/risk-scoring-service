"""MLflow tracking and registry configuration.

The tracking store and the model registry share one SQLite database at the
repo root (``mlflow.db``, gitignored); run artifacts live under ``mlruns/``.
Training code calls :func:`configure_tracking` once before any MLflow API.

Usage (from the repo root):
    python -m risk_scoring.tracking ui [--port PORT]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlflow

DB_FILENAME = "mlflow.db"
ARTIFACT_DIRNAME = "mlruns"
DEFAULT_EXPERIMENT = "readmission-risk"


def tracking_uri(repo_root: Path) -> str:
    """SQLAlchemy URI for the SQLite store backing both tracking and registry."""
    return f"sqlite:///{repo_root / DB_FILENAME}"


def configure_tracking(repo_root: Path, experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Point MLflow at the repo-local SQLite store and select the experiment.

    Creates the experiment on first use, with its artifact location pinned
    under the repo's ``mlruns/`` directory so artifacts never land in the
    caller's working directory. Returns the experiment id.
    """
    uri = tracking_uri(repo_root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    existing = mlflow.get_experiment_by_name(experiment)
    experiment_id: str
    if existing is not None:
        experiment_id = existing.experiment_id
    else:
        artifact_location = (repo_root / ARTIFACT_DIRNAME).as_uri()
        experiment_id = mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)
    return experiment_id


def ui_command(repo_root: Path, port: int = 5000) -> list[str]:
    """Argv that serves the MLflow UI against the repo's backing store."""
    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri(repo_root),
        "--port",
        str(port),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m risk_scoring.tracking")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui", help="serve the MLflow UI against the local SQLite store")
    ui.add_argument("--port", type=int, default=5000)

    args = parser.parse_args(argv)
    if args.command == "ui":
        subprocess.run(ui_command(Path.cwd(), port=args.port), check=True)


if __name__ == "__main__":
    main()
