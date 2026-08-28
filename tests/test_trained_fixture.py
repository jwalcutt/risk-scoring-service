"""The shared model fixture must not be a constant predictor.

Most of the service suite compares scores: two runs agreeing row for row,
a stored feature vector re-scoring to its logged value, a restart changing
nothing. Every one of those assertions holds trivially against a model
that returns the same number for every input, so the fixture's usefulness
rests on a property nothing else checks.

It went unchecked once. The fixture trained on 46 rows against
``min_data_in_leaf`` 20, LightGBM took no split at all, and the booster
returned the base rate for every feature vector ever passed to it. The
suite stayed green throughout, because a constant model satisfies an
equality between two runs exactly as well as a real one does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pytest

from factories import write_skew_population
from risk_scoring import train
from risk_scoring.cohort import build_cohort
from risk_scoring.features import MODEL_INPUT_COLUMNS, build_features
from risk_scoring.populations import load_population
from risk_scoring.provenance import rescore
from risk_scoring.tracking import configure_tracking
from risk_scoring.train import MODEL_NAME

MINIMUM_DISTINCT_SCORES = 4


@pytest.fixture(scope="module")
def model(trained_repo: tuple[Path, train.TrainingResult]) -> Any:
    root, trained = trained_repo
    configure_tracking(root)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{trained.model_version}")


def test_the_fixture_model_is_not_a_constant_predictor(model: Any) -> None:
    """Three feature vectors this far apart cannot honestly share a score."""
    scores = {
        rescore(model, {name: 0.0 for name in MODEL_INPUT_COLUMNS}),
        rescore(model, {name: 1.0 for name in MODEL_INPUT_COLUMNS}),
        rescore(model, {name: 100.0 for name in MODEL_INPUT_COLUMNS}),
    }
    assert len(scores) > 1, f"the fixture model returns {scores.pop()!r} for every input"


def test_the_fixture_model_lands_in_the_signal_band(
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    """A model at chance is the symptom the constant predictor showed."""
    _, trained = trained_repo
    assert trained.in_band, f"fixture AUROC {trained.auroc} is outside the pre-registered band"


def test_the_fixture_model_spreads_scores_over_the_ingested_population(
    model: Any, tmp_path: Path
) -> None:
    """The population the service tests ingest must score more than one way."""
    csv_dir = tmp_path / "csv"
    write_skew_population(csv_dir)
    frames = load_population(csv_dir)
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    features = build_features(
        cohort, frames["encounters"], frames["medications"], frames["conditions"]
    )
    scores = {
        rescore(model, {name: float(features.iloc[row][name]) for name in MODEL_INPUT_COLUMNS})
        for row in range(len(features))
    }
    assert len(scores) >= MINIMUM_DISTINCT_SCORES, (
        f"{len(scores)} distinct scores across {len(features)} discharges leaves the "
        "score comparisons in the restart and ingest tests nearly vacuous"
    )
