"""Tests for the predictions log write path.

The rules these tests pin:

- One row per scored encounter. The encounter id is the log's
  idempotency key, so a second write for the same encounter adds
  nothing and says so by returning None rather than raising.
- Every provenance field a later phase needs to trace a score back to
  its exact model and input round-trips unchanged, including the feature
  values, which are stored so diagnosis never has to recompute them.
- Floating-point values survive the round trip exactly. A score or a
  feature value that shifts in storage would make the stored row a
  different input from the one the model actually saw.
- ``scored_at`` is set by the database at write time and lands at or
  after the wall clock the caller observed before writing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from risk_scoring import predictions

pytestmark = pytest.mark.db

FEATURES = {
    "age_at_discharge": 71.0,
    "los_days": 3.5,
    "prior_inpatient_180d": 2.0,
    "days_since_prev_discharge": 365.0,
    "prior_ed_180d": 0.0,
    "active_medication_count": 4.0,
    "active_disorder_count": 6.0,
    "flag_chf": 1.0,
    "flag_chronic_pulmonary": 0.0,
    "flag_dementia": 0.0,
    "flag_diabetes": 1.0,
    "flag_malignancy": 0.0,
    "flag_mi": 0.0,
    "flag_renal_disease": 1.0,
}


def _record(**overrides: Any) -> predictions.PredictionRecord:
    fields: dict[str, Any] = {
        "patient_id": "patient-1",
        "encounter_id": "encounter-1",
        "event_time": datetime(2025, 3, 4, 17, 30, 0, tzinfo=UTC),
        "input_hash": "a" * 64,
        "model_name": "readmission-risk",
        "model_version": 3,
        "feature_version": "1.0.0",
        "cohort_version": "1.0.0",
        "score": 0.1234567890123456,
        "features": dict(FEATURES),
    }
    fields.update(overrides)
    return predictions.PredictionRecord(**fields)


def _row_count(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute("SELECT count(*) FROM predictions").fetchone()
    assert row is not None
    return int(row[0])


def test_record_prediction_returns_an_id_and_round_trips_every_field(
    db_conn: psycopg.Connection[Any],
) -> None:
    record = _record()
    before = datetime.now(UTC)

    prediction_id = predictions.record_prediction(db_conn, record)

    assert prediction_id is not None
    stored = predictions.prediction_for_encounter(db_conn, "encounter-1")
    assert stored is not None
    assert stored.prediction_id == prediction_id
    assert stored.patient_id == record.patient_id
    assert stored.encounter_id == record.encounter_id
    assert stored.event_time == record.event_time
    assert stored.input_hash == record.input_hash
    assert stored.model_name == record.model_name
    assert stored.model_version == record.model_version
    assert stored.feature_version == record.feature_version
    assert stored.cohort_version == record.cohort_version
    assert stored.score == record.score
    assert stored.features == record.features
    assert stored.scored_at >= before - timedelta(seconds=1)


def test_score_and_features_survive_the_round_trip_exactly(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Exact equality, not tolerance: a shifted value is a different input."""
    record = _record(score=0.9999999999999999, features={**FEATURES, "los_days": 0.1 + 0.2})
    predictions.record_prediction(db_conn, record)

    stored = predictions.prediction_for_encounter(db_conn, "encounter-1")
    assert stored is not None
    assert stored.score == 0.9999999999999999
    assert stored.features["los_days"] == 0.1 + 0.2


def test_repost_of_a_scored_encounter_adds_no_second_row(
    db_conn: psycopg.Connection[Any],
) -> None:
    first = predictions.record_prediction(db_conn, _record())
    second = predictions.record_prediction(db_conn, _record())

    assert first is not None
    assert second is None
    assert _row_count(db_conn) == 1


def test_a_differing_rescore_of_the_same_encounter_never_overwrites(
    db_conn: psycopg.Connection[Any],
) -> None:
    """The first score for an encounter is the one that stands."""
    predictions.record_prediction(db_conn, _record(score=0.25))
    assert predictions.record_prediction(db_conn, _record(score=0.75, model_version=4)) is None

    stored = predictions.prediction_for_encounter(db_conn, "encounter-1")
    assert stored is not None
    assert stored.score == 0.25
    assert stored.model_version == 3


def test_each_encounter_gets_its_own_row(db_conn: psycopg.Connection[Any]) -> None:
    first = predictions.record_prediction(db_conn, _record(encounter_id="encounter-1"))
    second = predictions.record_prediction(db_conn, _record(encounter_id="encounter-2"))

    assert first is not None
    assert second is not None
    assert first != second
    assert _row_count(db_conn) == 2


def test_has_prediction_tracks_the_table(db_conn: psycopg.Connection[Any]) -> None:
    assert predictions.has_prediction(db_conn, "encounter-1") is False
    predictions.record_prediction(db_conn, _record())
    assert predictions.has_prediction(db_conn, "encounter-1") is True
    assert predictions.has_prediction(db_conn, "encounter-2") is False


def test_prediction_for_unscored_encounter_is_none(db_conn: psycopg.Connection[Any]) -> None:
    assert predictions.prediction_for_encounter(db_conn, "never-scored") is None


def test_recorded_prediction_survives_a_rollback_on_the_same_connection(
    db_conn: psycopg.Connection[Any],
) -> None:
    """The write commits on its own, so an acknowledged score is a durable score."""
    predictions.record_prediction(db_conn, _record())
    db_conn.rollback()

    assert predictions.has_prediction(db_conn, "encounter-1") is True


def test_all_predictions_is_empty_before_anything_is_scored(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert predictions.all_predictions(db_conn) == []


def test_all_predictions_returns_every_row_in_write_order(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Monitoring and the restart comparison both read the log in order."""
    for encounter in ("encounter-c", "encounter-a", "encounter-b"):
        predictions.record_prediction(db_conn, _record(encounter_id=encounter))

    stored = predictions.all_predictions(db_conn)

    assert [row.encounter_id for row in stored] == ["encounter-c", "encounter-a", "encounter-b"]
    assert [row.prediction_id for row in stored] == sorted(row.prediction_id for row in stored)


def test_all_predictions_round_trips_every_field(db_conn: psycopg.Connection[Any]) -> None:
    predictions.record_prediction(db_conn, _record())

    (stored,) = predictions.all_predictions(db_conn)

    assert stored == predictions.prediction_for_encounter(db_conn, "encounter-1")


def test_a_dropped_re_post_still_consumes_a_prediction_id(
    db_conn: psycopg.Connection[Any],
) -> None:
    """So ids gap after a resume, and two runs of one stream cannot be compared by id."""
    predictions.record_prediction(db_conn, _record(encounter_id="encounter-1"))
    predictions.record_prediction(db_conn, _record(encounter_id="encounter-1"))
    predictions.record_prediction(db_conn, _record(encounter_id="encounter-2"))

    assert [row.prediction_id for row in predictions.all_predictions(db_conn)] == [1, 3]
