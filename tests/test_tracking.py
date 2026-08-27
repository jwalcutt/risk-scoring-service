"""Tests for the MLflow tracking and registry configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from risk_scoring import tracking


def test_tracking_uri_is_sqlite_db_under_repo_root(tmp_path: Path) -> None:
    uri = tracking.tracking_uri(tmp_path)
    assert uri == f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_configure_creates_store_and_experiment(repo_root: Path) -> None:
    experiment_id = tracking.configure_tracking(repo_root, experiment="exp-a")

    assert (repo_root / "mlflow.db").exists()
    assert mlflow.get_tracking_uri() == tracking.tracking_uri(repo_root)
    assert mlflow.get_registry_uri() == tracking.tracking_uri(repo_root)

    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == "exp-a"
    assert experiment.artifact_location == (repo_root / "mlruns").as_uri()


def test_configure_is_idempotent(repo_root: Path) -> None:
    first = tracking.configure_tracking(repo_root, experiment="exp-b")
    second = tracking.configure_tracking(repo_root, experiment="exp-b")
    assert first == second


def test_ui_command_points_at_backing_store(tmp_path: Path) -> None:
    command = tracking.ui_command(tmp_path, port=5001)
    assert tracking.tracking_uri(tmp_path) in command
    assert "5001" in command


class _ConstantModel(mlflow.pyfunc.PythonModel):
    """Dummy artifact: predicts a constant score for every row."""

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[float]:
        return [self.score] * len(model_input)


def test_register_and_version_flow_end_to_end(repo_root: Path) -> None:
    """Log a dummy model twice, get two registry versions, load one back by alias."""
    tracking.configure_tracking(repo_root, experiment="exp-e2e")
    name = "dummy-model"

    for score in (0.1, 0.9):
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                name="model",
                python_model=_ConstantModel(score),
                registered_model_name=name,
            )

    client = MlflowClient()
    versions = {int(v.version) for v in client.search_model_versions(f"name = '{name}'")}
    assert versions == {1, 2}

    client.set_registered_model_alias(name, "champion", "1")
    loaded = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    scores = loaded.predict(pd.DataFrame({"x": [1, 2, 3]}))
    assert scores == [0.1, 0.1, 0.1]

    artifact_root = repo_root / "mlruns"
    assert artifact_root.is_dir() and any(artifact_root.iterdir())
