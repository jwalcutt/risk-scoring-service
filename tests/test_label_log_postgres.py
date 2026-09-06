"""Tests for the labels table write path.

The rules these tests pin:

- A label attaches to a scored discharge by its encounter id, and the
  prediction id is looked up by the write, never supplied by the caller.
- One label per prediction. A second write for the same discharge adds
  nothing and says so by returning None rather than raising, because a
  harness resuming from a checkpoint re-releases a tick's labels.
- A label for a discharge that was never scored is a defect: the write
  raises and stores nothing.
- Every field round-trips unchanged, and the columns the database
  assigns land at or after the wall clock the caller observed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from risk_scoring import label_log, predictions

pytestmark = pytest.mark.db

DISCHARGED = datetime(2025, 3, 4, 17, 30, 0, tzinfo=UTC)
DUE = DISCHARGED + timedelta(days=30)


def _score(conn: psycopg.Connection[Any], encounter_id: str) -> int:
    scored = predictions.record_prediction(
        conn,
        predictions.PredictionRecord(
            patient_id="patient-1",
            encounter_id=encounter_id,
            event_time=DISCHARGED,
            input_hash="a" * 64,
            model_name="readmission-risk",
            model_version=3,
            feature_version="1.0.0",
            cohort_version="1.0.0",
            score=0.25,
            features={"age_at_discharge": 71.0},
        ),
    )
    assert scored is not None
    return scored


def _record(**overrides: Any) -> label_log.LabelRecord:
    fields: dict[str, Any] = {
        "encounter_id": "encounter-1",
        "label": 1,
        "label_version": "1.0.0",
        "due_at": DUE,
        "released_at": DUE,
    }
    fields.update(overrides)
    return label_log.LabelRecord(**fields)


def _row_count(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute("SELECT count(*) FROM labels").fetchone()
    assert row is not None
    return int(row[0])


def test_record_label_returns_an_id_and_round_trips_every_field(
    db_conn: psycopg.Connection[Any],
) -> None:
    prediction_id = _score(db_conn, "encounter-1")
    before = datetime.now(tz=UTC)
    released = DUE + timedelta(hours=1)
    label_id = label_log.record_label(db_conn, _record(released_at=released))
    assert label_id == 1

    (stored,) = label_log.all_labels(db_conn)
    assert stored.label_id == 1
    assert stored.prediction_id == prediction_id
    assert stored.encounter_id == "encounter-1"
    assert stored.label == 1
    assert stored.label_version == "1.0.0"
    assert stored.due_at == DUE
    assert stored.released_at == released
    assert stored.recorded_at >= before - timedelta(seconds=1)


def test_a_second_label_for_one_discharge_adds_no_row_and_returns_none(
    db_conn: psycopg.Connection[Any],
) -> None:
    _score(db_conn, "encounter-1")
    assert label_log.record_label(db_conn, _record()) == 1
    assert label_log.record_label(db_conn, _record(label=0)) is None
    (stored,) = label_log.all_labels(db_conn)
    assert stored.label == 1
    assert _row_count(db_conn) == 1


def test_a_label_for_an_unscored_discharge_raises_and_stores_nothing(
    db_conn: psycopg.Connection[Any],
) -> None:
    with pytest.raises(label_log.UnscoredDischargeError, match="encounter-1"):
        label_log.record_label(db_conn, _record())
    assert _row_count(db_conn) == 0
    # The connection is usable afterwards: the failed write was rolled back.
    _score(db_conn, "encounter-1")
    assert label_log.record_label(db_conn, _record()) == 1


def test_a_recorded_label_survives_a_rollback_on_the_same_connection(
    db_conn: psycopg.Connection[Any],
) -> None:
    _score(db_conn, "encounter-1")
    label_log.record_label(db_conn, _record())
    db_conn.rollback()
    assert _row_count(db_conn) == 1


def test_all_labels_is_empty_before_anything_is_released(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert label_log.all_labels(db_conn) == []


def test_all_labels_returns_every_row_in_write_order(db_conn: psycopg.Connection[Any]) -> None:
    for name in ("encounter-3", "encounter-1", "encounter-2"):
        _score(db_conn, name)
        label_log.record_label(db_conn, _record(encounter_id=name))
    assert [stored.encounter_id for stored in label_log.all_labels(db_conn)] == [
        "encounter-3",
        "encounter-1",
        "encounter-2",
    ]
    assert [stored.label_id for stored in label_log.all_labels(db_conn)] == [1, 2, 3]
