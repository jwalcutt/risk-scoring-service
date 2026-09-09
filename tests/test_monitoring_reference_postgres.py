"""Building and storing the reference a monitoring window is compared against.

The rules these tests pin:

- The reference is what the registered model was actually fitted on. It
  is rebuilt from the frozen export using the cutoff, split seed, and
  holdout fraction read from that version's own training run, the way the
  gate rebuilds its holdout, so it cannot drift from what training did.
- Feature arrays span the whole training window; the score array is the
  patient-grouped holdout only. In-sample scores are optimistically
  shifted, and a reference built from them would make live score drift
  read low.
- The stored arrays equal the training pipeline's own frame for that
  version, column for column and value for value.
- One row per registered version. Re-running for a version that already
  has one is a no-op that says so, never a silent overwrite: a promotion
  writes a row for the new version, and a finished run's evaluations keep
  naming the reference they were measured against.
- A version that is not in the registry fails with a message naming it,
  not an MlflowException.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest

from risk_scoring import train
from risk_scoring.cohort import COHORT_VERSION, build_cohort, filter_training_window
from risk_scoring.features import FEATURE_VERSION, MODEL_INPUT_COLUMNS, build_features
from risk_scoring.monitoring import reference as reference_module
from risk_scoring.monitoring.reference import (
    ReferenceExistsError,
    build_reference,
    read_reference,
    record_reference,
)
from risk_scoring.populations import load_population
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db


@pytest.fixture()
def built(
    trained_repo: tuple[Path, train.TrainingResult],
) -> tuple[Path, train.TrainingResult, Any]:
    root, trained = trained_repo
    csv_dir = root / "data" / "baseline" / "csv"
    return root, trained, build_reference(csv_dir, root, model_version=trained.model_version)


# --- what the builder produces ---


def test_the_reference_names_the_version_and_the_code_it_was_fitted_under(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    _, trained, reference = built
    assert reference.model_name == MODEL_NAME
    assert reference.model_version == trained.model_version
    assert reference.feature_version == FEATURE_VERSION
    assert reference.cohort_version == COHORT_VERSION
    assert reference.split_seed == train.SPLIT_SEED


def test_the_feature_arrays_are_the_models_input_columns(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    """Identifiers are not features, and every model input must be present."""
    _, _, reference = built
    assert set(reference.features) == set(MODEL_INPUT_COLUMNS)


def test_the_feature_arrays_equal_the_training_pipelines_own_frame(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    root, _, reference = built
    frames = load_population(root / "data" / "baseline" / "csv")
    cutoff = train.TRAINING_CUTOFF
    cohort = filter_training_window(
        build_cohort(frames["encounters"], frames["patients"]).frame, cutoff
    )
    expected = build_features(
        cohort, frames["encounters"], frames["medications"], frames["conditions"]
    )
    for column in MODEL_INPUT_COLUMNS:
        assert reference.features[column] == pytest.approx(expected[column].tolist())


def test_the_feature_arrays_span_the_whole_training_window(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    _, trained, reference = built
    total = trained.n_train_rows + trained.n_holdout_rows
    assert reference.n_train_rows == total
    for column in MODEL_INPUT_COLUMNS:
        assert len(reference.features[column]) == total


def test_the_score_array_is_the_holdout_only(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    """In-sample scores would make live score drift read low."""
    _, trained, reference = built
    assert reference.n_holdout_rows == trained.n_holdout_rows
    assert len(reference.scores) == trained.n_holdout_rows
    assert reference.n_holdout_rows < reference.n_train_rows


def test_the_scores_are_real_probabilities(
    built: tuple[Path, train.TrainingResult, Any],
) -> None:
    _, _, reference = built
    scores = np.asarray(reference.scores, dtype=float)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert scores.min() < scores.max()


def test_a_version_that_is_not_registered_is_named_in_the_error(
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = trained_repo
    with pytest.raises(LookupError, match=rf"{MODEL_NAME}.*999"):
        build_reference(root / "data" / "baseline" / "csv", root, model_version=999)


# --- storing it ---


def test_a_stored_reference_reads_back_equal(
    db_conn: psycopg.Connection[Any], built: tuple[Path, train.TrainingResult, Any]
) -> None:
    _, trained, reference = built
    reference_id = record_reference(db_conn, reference)
    stored = read_reference(db_conn, MODEL_NAME, trained.model_version)
    assert stored is not None
    assert stored.reference_id == reference_id
    assert stored.features == reference.features
    assert stored.scores == reference.scores
    assert stored.model_version == trained.model_version


def test_the_arrays_survive_the_jsonb_round_trip_exactly(
    db_conn: psycopg.Connection[Any], built: tuple[Path, train.TrainingResult, Any]
) -> None:
    """Every statistic downstream reads these arrays; a lossy trip is a wrong answer."""
    _, trained, reference = built
    record_reference(db_conn, reference)
    stored = read_reference(db_conn, MODEL_NAME, trained.model_version)
    assert stored is not None
    for column in MODEL_INPUT_COLUMNS:
        assert stored.features[column] == reference.features[column]
    assert stored.scores == reference.scores


def test_recording_a_version_that_already_has_a_reference_is_refused(
    db_conn: psycopg.Connection[Any], built: tuple[Path, train.TrainingResult, Any]
) -> None:
    """A promotion writes a new version's row; it never edits an existing one."""
    _, trained, reference = built
    record_reference(db_conn, reference)
    with pytest.raises(ReferenceExistsError, match=str(trained.model_version)):
        record_reference(db_conn, reference)


def test_reading_a_reference_that_was_never_written_is_none(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert read_reference(db_conn, MODEL_NAME, 4) is None


def test_a_nan_in_the_arrays_raises_where_the_value_has_a_name(
    db_conn: psycopg.Connection[Any], built: tuple[Path, train.TrainingResult, Any]
) -> None:
    """Same rule as the prediction log: never a jsonb parse error at the INSERT."""
    _, _, reference = built
    broken = reference_module.Reference(
        model_name=reference.model_name,
        model_version=reference.model_version,
        feature_version=reference.feature_version,
        cohort_version=reference.cohort_version,
        training_cutoff=reference.training_cutoff,
        split_seed=reference.split_seed,
        n_train_rows=reference.n_train_rows,
        n_holdout_rows=reference.n_holdout_rows,
        features={**reference.features, "los_days": [float("nan")]},
        scores=reference.scores,
    )
    with pytest.raises(ValueError, match="not JSON compliant"):
        record_reference(db_conn, broken)

    count = db_conn.execute("SELECT count(*) FROM monitoring_reference").fetchone()
    assert count == (0,)


def test_the_stored_row_is_json_not_a_python_repr(
    db_conn: psycopg.Connection[Any], built: tuple[Path, train.TrainingResult, Any]
) -> None:
    _, _, reference = built
    record_reference(db_conn, reference)
    row = db_conn.execute("SELECT features FROM monitoring_reference").fetchone()
    assert row is not None
    assert isinstance(row[0], dict)
    assert json.dumps(row[0])
