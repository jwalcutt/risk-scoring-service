"""Shared fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest


@pytest.fixture()
def repo_root(tmp_path: Path) -> Iterator[Path]:
    """A throwaway repo root; restores global MLflow URIs after the test."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    yield tmp_path
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)
