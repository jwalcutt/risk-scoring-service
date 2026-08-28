"""Provenance re-verification against a real service and a real registry.

The service writes an input hash and a score into every log row. These
tests take those rows back apart: rebuild the posted event from the source
export and recompute its digest, then load the model version the row
itself names and re-score the feature values the row stored. Both must
reproduce exactly.

The rules these tests pin:

- The digest covers the whole posted envelope, not the payload inside it.
- The stored feature values are the model's actual input, at full
  precision and in the column order the model expects, so re-scoring them
  returns the logged score to the last bit.
- The model loaded is the one the row names, never the one the service is
  currently pinned to.
- Comparison is exact. A one-ulp difference is a mismatch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from factories import write_gate_population, write_skew_population
from risk_scoring import predictions, train
from risk_scoring.populations import load_population
from risk_scoring.provenance import recompute_input_hash, rescore, verify_predictions
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.stream import build_stream
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def population(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    csv_dir = tmp_path_factory.mktemp("provenance-population") / "csv"
    write_skew_population(csv_dir)
    return load_population(csv_dir)


@pytest.fixture(scope="module")
def signal_repo(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, train.TrainingResult]]:
    """A registry holding a model whose score actually depends on its input.

    The shared ``trained_repo`` fixture trains on 46 rows and produces a
    constant predictor: every feature vector scores the base rate. The
    hash half of provenance would still be checkable against it, but the
    score half would pass no matter what ``rescore`` returned, so these
    tests train on the signal-bearing population instead.
    """
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("provenance-repo")
    write_gate_population(root / "data" / "baseline" / "csv")
    result = train.train(root / "data" / "baseline" / "csv", root)
    yield root, result
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


@pytest.fixture()
def logged(
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
    db_url: str,
) -> list[predictions.StoredPrediction]:
    """The whole population ingested through the service, as logged rows."""
    root, trained = signal_repo
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, db_url)
    with TestClient(app) as client:
        for event in build_stream(population):
            response = client.post("/events", json=event)
            assert response.status_code == 202, response.text
    with psycopg.connect(db_url) as conn:
        return predictions.all_predictions(conn)


def test_the_population_actually_produces_predictions(
    logged: list[predictions.StoredPrediction],
) -> None:
    """A vacuous log would make every check below pass while proving nothing."""
    assert len(logged) >= 3


def test_every_logged_prediction_reproduces_its_hash_and_score(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = signal_repo
    checks = verify_predictions(logged, population["encounters"], root)
    assert len(checks) == len(logged)
    broken = [check.describe() for check in checks if not check.ok]
    assert broken == []


def test_the_logged_model_version_is_what_gets_loaded(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    """Falling back to the service's pin would pass for a foreign row."""
    root, trained = signal_repo
    (first,) = logged[:1]
    checks = verify_predictions([first], population["encounters"], root)
    assert checks[0].model_uri == f"models:/{MODEL_NAME}/{trained.model_version}"


def test_a_prediction_naming_an_absent_model_version_fails_to_verify(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, trained = signal_repo
    foreign = predictions.StoredPrediction(
        **{**vars(logged[0]), "model_version": trained.model_version + 99}
    )
    with pytest.raises(Exception, match=str(trained.model_version + 99)):
        verify_predictions([foreign], population["encounters"], root)


def test_a_tampered_source_row_is_reported_as_a_hash_mismatch(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    """A break in the chain is a reported verdict, not an exception."""
    root, _ = signal_repo
    encounters = population["encounters"].copy()
    target = logged[0].encounter_id
    encounters.loc[encounters["Id"] == target, "START"] = "1999-01-01T00:00:00Z"

    (check,) = verify_predictions([logged[0]], encounters, root)
    assert not check.hash_matches
    assert check.score_matches
    assert not check.ok
    assert check.logged_hash in check.describe()


def test_a_prediction_with_no_source_row_raises(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = signal_repo
    encounters = population["encounters"]
    orphan = encounters.loc[encounters["Id"] != logged[0].encounter_id]
    with pytest.raises(KeyError, match=logged[0].encounter_id):
        verify_predictions([logged[0]], orphan, root)


def test_the_rescore_reads_the_stored_feature_values(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    """Altered stored values must change the score, or the check is vacuous."""
    root, _ = signal_repo
    assert all(check.ok for check in verify_predictions(logged, population["encounters"], root))

    altered = [
        predictions.StoredPrediction(
            **{
                **vars(prediction),
                "features": {
                    name: value * 2.0 + 1.0 for name, value in prediction.features.items()
                },
            }
        )
        for prediction in logged
    ]
    after = verify_predictions(altered, population["encounters"], root)
    assert any(not check.score_matches for check in after)
    assert all(check.hash_matches for check in after)


def test_the_logged_hash_is_over_the_envelope_the_stream_posts(
    logged: list[predictions.StoredPrediction],
    population: dict[str, pd.DataFrame],
) -> None:
    """Ties the stored digest to the exact source row, without a model."""
    encounters = population["encounters"]
    for prediction in logged:
        (row,) = [
            dict(candidate)
            for _, candidate in encounters.iterrows()
            if candidate["Id"] == prediction.encounter_id
        ]
        assert recompute_input_hash(row) == prediction.input_hash


def test_rescoring_reproduces_the_logged_score_to_the_last_bit(
    logged: list[predictions.StoredPrediction],
    signal_repo: tuple[Path, train.TrainingResult],
) -> None:
    """Stated without the verifier, so the claim does not rest on its own code."""
    import mlflow

    from risk_scoring.tracking import configure_tracking

    root, trained = signal_repo
    configure_tracking(root)
    model: Any = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{trained.model_version}")
    for prediction in logged:
        assert rescore(model, prediction.features) == prediction.score
