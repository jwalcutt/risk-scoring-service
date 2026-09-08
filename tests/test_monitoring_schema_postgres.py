"""The monitoring tables' schema, restated literally.

Every later part of the monitoring layer writes into these three tables,
and Grafana reads them directly, so a column renamed in a migration would
change the substrate under a dashboard and an offline audit at once.

These assertions are deliberately literal — they restate the schema
rather than deriving it from the code under test, so a change has to be
made in two places on purpose. The same rule the prediction log and the
labels table are held to.

Two invariants here are properties of the database rather than of any
writer, and both are proven by an insert that must fail:

- one evaluation per run per boundary, so a monitor killed between
  boundaries neither skips one nor writes a second row for one, while a
  database holding two finished runs can still hold both their grids;
- an alert's sim_at is its own evaluation's boundary, enforced through a
  composite foreign key, because detection time is measured in simulated
  days and must not depend on when a poll landed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg import errors

pytestmark = pytest.mark.db

# column name -> (data type as information_schema reports it, nullable)
REFERENCE_COLUMNS: dict[str, tuple[str, bool]] = {
    "reference_id": ("bigint", False),
    "model_name": ("text", False),
    "model_version": ("integer", False),
    "feature_version": ("text", False),
    "cohort_version": ("text", False),
    "training_cutoff": ("timestamp with time zone", False),
    "split_seed": ("integer", False),
    "n_train_rows": ("integer", False),
    "n_holdout_rows": ("integer", False),
    "features": ("jsonb", False),
    "scores": ("jsonb", False),
    "created_at": ("timestamp with time zone", False),
}

EVALUATION_COLUMNS: dict[str, tuple[str, bool]] = {
    "evaluation_id": ("bigint", False),
    "run_id": ("bigint", False),
    "reference_id": ("bigint", False),
    "window_start": ("timestamp with time zone", False),
    "boundary": ("timestamp with time zone", False),
    "thresholds_hash": ("text", False),
    "prediction_count": ("integer", False),
    "label_count": ("integer", False),
    "refusal_count": ("integer", False),
    "version_mismatch_count": ("integer", False),
    "statistics": ("jsonb", False),
    "evaluated_at": ("timestamp with time zone", False),
}

ALERT_COLUMNS: dict[str, tuple[str, bool]] = {
    "alert_id": ("bigint", False),
    "evaluation_id": ("bigint", False),
    "signal": ("text", False),
    "statistic": ("double precision", False),
    "threshold": ("double precision", False),
    "sim_at": ("timestamp with time zone", False),
    "raised_at": ("timestamp with time zone", False),
    "acknowledged_at": ("timestamp with time zone", True),
    "note": ("text", False),
}

START = datetime(2025, 1, 1, tzinfo=UTC)
BOUNDARY = START + timedelta(days=7)


# --- helpers, matching the prediction log's schema test ---


def _columns(conn: psycopg.Connection[Any], table: str) -> dict[str, tuple[str, bool]]:
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = %s",
        [table],
    ).fetchall()
    return {name: (data_type, nullable == "YES") for name, data_type, nullable in rows}


def _constraint_columns(
    conn: psycopg.Connection[Any], table: str, constraint_type: str
) -> set[tuple[str, ...]]:
    rows = conn.execute(
        "SELECT tc.constraint_name, kcu.column_name"
        " FROM information_schema.table_constraints AS tc"
        " JOIN information_schema.key_column_usage AS kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "  AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = 'public' AND tc.table_name = %s"
        "   AND tc.constraint_type = %s"
        " ORDER BY kcu.ordinal_position",
        [table, constraint_type],
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for constraint_name, column_name in rows:
        grouped.setdefault(constraint_name, []).append(column_name)
    return {tuple(columns) for columns in grouped.values()}


# --- row builders, each returning the id the next one needs ---


def _reference(conn: psycopg.Connection[Any], *, model_version: int = 4) -> int:
    row = conn.execute(
        "INSERT INTO monitoring_reference (model_name, model_version, feature_version,"
        " cohort_version, training_cutoff, split_seed, n_train_rows, n_holdout_rows,"
        " features, scores) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " RETURNING reference_id",
        [
            "readmission-risk",
            model_version,
            "1.1.0",
            "1.0.0",
            START,
            20260101,
            9049,
            2245,
            json.dumps({"los_days": [1.0, 2.0]}),
            json.dumps([0.1, 0.2]),
        ],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _run(conn: psycopg.Connection[Any], *, status: str = "finished") -> int:
    row = conn.execute(
        "INSERT INTO replay_runs (population, start_at, end_at, acceleration, sim_now, status)"
        " VALUES (%s, %s, %s, %s, %s, %s) RETURNING run_id",
        ["baseline", START, START + timedelta(days=365), 4.0, START, status],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _evaluation(
    conn: psycopg.Connection[Any],
    run_id: int,
    reference_id: int,
    *,
    window_start: datetime = START,
    boundary: datetime = BOUNDARY,
) -> int:
    row = conn.execute(
        "INSERT INTO monitoring_evaluations (run_id, reference_id, window_start, boundary,"
        " thresholds_hash, prediction_count, label_count, refusal_count,"
        " version_mismatch_count, statistics) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " RETURNING evaluation_id",
        [run_id, reference_id, window_start, boundary, "abc123", 50, 12, 0, 0, json.dumps({})],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _alert(
    conn: psycopg.Connection[Any],
    evaluation_id: int,
    *,
    signal: str = "los_days",
    sim_at: datetime = BOUNDARY,
) -> None:
    conn.execute(
        "INSERT INTO alerts (evaluation_id, signal, statistic, threshold, sim_at, note)"
        " VALUES (%s, %s, %s, %s, %s, %s)",
        [evaluation_id, signal, 0.001, 0.01, sim_at, ""],
    )


# --- columns ---


def test_reference_columns_names_types_and_nullability(db_conn: psycopg.Connection[Any]) -> None:
    assert _columns(db_conn, "monitoring_reference") == REFERENCE_COLUMNS


def test_evaluation_columns_names_types_and_nullability(db_conn: psycopg.Connection[Any]) -> None:
    assert _columns(db_conn, "monitoring_evaluations") == EVALUATION_COLUMNS


def test_alert_columns_names_types_and_nullability(db_conn: psycopg.Connection[Any]) -> None:
    """acknowledged_at is the only nullable column: an alert starts unacknowledged."""
    assert _columns(db_conn, "alerts") == ALERT_COLUMNS


# --- keys ---


def test_each_table_has_its_surrogate_primary_key(db_conn: psycopg.Connection[Any]) -> None:
    assert _constraint_columns(db_conn, "monitoring_reference", "PRIMARY KEY") == {
        ("reference_id",)
    }
    assert _constraint_columns(db_conn, "monitoring_evaluations", "PRIMARY KEY") == {
        ("evaluation_id",)
    }
    assert _constraint_columns(db_conn, "alerts", "PRIMARY KEY") == {("alert_id",)}


def test_one_reference_per_registered_model_version(db_conn: psycopg.Connection[Any]) -> None:
    """A promotion writes a row for the new version; it never edits this one."""
    assert ("model_name", "model_version") in _constraint_columns(
        db_conn, "monitoring_reference", "UNIQUE"
    )
    _reference(db_conn, model_version=4)
    with pytest.raises(errors.UniqueViolation):
        _reference(db_conn, model_version=4)


def test_a_reference_for_another_version_is_accepted(db_conn: psycopg.Connection[Any]) -> None:
    _reference(db_conn, model_version=4)
    _reference(db_conn, model_version=5)
    count = db_conn.execute("SELECT count(*) FROM monitoring_reference").fetchone()
    assert count == (2,)


def test_an_evaluation_names_a_run_and_a_reference(db_conn: psycopg.Connection[Any]) -> None:
    foreign_keys = _constraint_columns(db_conn, "monitoring_evaluations", "FOREIGN KEY")
    assert ("run_id",) in foreign_keys
    assert ("reference_id",) in foreign_keys


def test_an_evaluation_against_a_reference_that_does_not_exist_is_refused(
    db_conn: psycopg.Connection[Any],
) -> None:
    run_id = _run(db_conn)
    with pytest.raises(errors.ForeignKeyViolation):
        _evaluation(db_conn, run_id, 999)


# --- one evaluation per boundary ---


def test_a_second_evaluation_of_one_boundary_in_one_run_is_refused(
    db_conn: psycopg.Connection[Any],
) -> None:
    """A monitor killed between boundaries must not write a boundary twice."""
    reference_id = _reference(db_conn)
    run_id = _run(db_conn)
    _evaluation(db_conn, run_id, reference_id)
    with pytest.raises(errors.UniqueViolation):
        _evaluation(db_conn, run_id, reference_id)


def test_two_runs_may_each_evaluate_the_same_boundary(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Finished runs accumulate, and two over one span share every boundary."""
    reference_id = _reference(db_conn)
    first = _run(db_conn, status="finished")
    second = _run(db_conn, status="finished")
    _evaluation(db_conn, first, reference_id)
    _evaluation(db_conn, second, reference_id)
    count = db_conn.execute("SELECT count(*) FROM monitoring_evaluations").fetchone()
    assert count == (2,)


def test_a_boundary_at_or_before_its_window_start_is_refused(
    db_conn: psycopg.Connection[Any],
) -> None:
    reference_id = _reference(db_conn)
    run_id = _run(db_conn)
    with pytest.raises(errors.CheckViolation):
        _evaluation(db_conn, run_id, reference_id, window_start=BOUNDARY, boundary=BOUNDARY)


# --- an alert's instant is its evaluation's boundary ---


def test_an_alert_at_its_evaluations_boundary_is_accepted(
    db_conn: psycopg.Connection[Any],
) -> None:
    reference_id = _reference(db_conn)
    evaluation_id = _evaluation(db_conn, _run(db_conn), reference_id)
    _alert(db_conn, evaluation_id, sim_at=BOUNDARY)
    count = db_conn.execute("SELECT count(*) FROM alerts").fetchone()
    assert count == (1,)


@pytest.mark.parametrize("offset", [timedelta(seconds=-1), timedelta(seconds=1), timedelta(days=7)])
def test_an_alert_at_any_other_instant_is_unrepresentable(
    db_conn: psycopg.Connection[Any], offset: timedelta
) -> None:
    """Detection time is measured in simulated days, so sim_at cannot drift."""
    reference_id = _reference(db_conn)
    evaluation_id = _evaluation(db_conn, _run(db_conn), reference_id)
    with pytest.raises(errors.ForeignKeyViolation):
        _alert(db_conn, evaluation_id, sim_at=BOUNDARY + offset)


def test_one_alert_per_signal_per_evaluation(db_conn: psycopg.Connection[Any]) -> None:
    reference_id = _reference(db_conn)
    evaluation_id = _evaluation(db_conn, _run(db_conn), reference_id)
    _alert(db_conn, evaluation_id, signal="los_days")
    _alert(db_conn, evaluation_id, signal="score")
    with pytest.raises(errors.UniqueViolation):
        _alert(db_conn, evaluation_id, signal="los_days")


def test_an_alert_starts_unacknowledged(db_conn: psycopg.Connection[Any]) -> None:
    reference_id = _reference(db_conn)
    evaluation_id = _evaluation(db_conn, _run(db_conn), reference_id)
    _alert(db_conn, evaluation_id)
    row = db_conn.execute("SELECT acknowledged_at FROM alerts").fetchone()
    assert row == (None,)


# --- indexes the readers need ---


def test_evaluations_are_indexed_by_boundary(db_conn: psycopg.Connection[Any]) -> None:
    rows = db_conn.execute(
        "SELECT indexdef FROM pg_indexes"
        " WHERE schemaname = 'public' AND tablename = 'monitoring_evaluations'"
    ).fetchall()
    assert any("boundary" in definition for (definition,) in rows)


def test_unacknowledged_alerts_are_indexed(db_conn: psycopg.Connection[Any]) -> None:
    rows = db_conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'alerts'"
    ).fetchall()
    assert any("acknowledged_at IS NULL" in definition for (definition,) in rows)
