"""The labels table's schema, asserted column by column.

Realized performance is the join of this table with the predictions log,
and the never-early rule the replay harness lives by is a constraint on
it. As with the log and the run row, these assertions restate the schema
literally rather than deriving it from the code under test, so a
migration that changes the substrate has to change a test on purpose.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg import errors

pytestmark = pytest.mark.db

# column name -> (data type as information_schema reports it, nullable)
EXPECTED_COLUMNS: dict[str, tuple[str, bool]] = {
    "label_id": ("bigint", False),
    "prediction_id": ("bigint", False),
    "encounter_id": ("text", False),
    "label": ("integer", False),
    "label_version": ("text", False),
    "due_at": ("timestamp with time zone", False),
    "released_at": ("timestamp with time zone", False),
    "recorded_at": ("timestamp with time zone", False),
}

_INSERT = (
    "INSERT INTO labels (prediction_id, encounter_id, label, label_version, due_at, released_at)"
    " VALUES (%s, %s, %s, %s, %s, %s) RETURNING label_id"
)


def _score(conn: psycopg.Connection[Any], encounter_id: str) -> int:
    row = conn.execute(
        "INSERT INTO predictions (patient_id, encounter_id, event_time, input_hash,"
        " model_name, model_version, feature_version, cohort_version, score, features)"
        " VALUES ('p', %s, '2025-01-01T00:00:00Z', 'h', 'm', 1, '1.0.0', '1.0.0', 0.5, '{}')"
        " RETURNING prediction_id",
        [encounter_id],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _insert(
    conn: psycopg.Connection[Any],
    prediction_id: int,
    *,
    encounter_id: str = "e-1",
    label: int = 0,
    label_version: str = "1.0.0",
    due_at: str = "2025-01-31T00:00:00Z",
    released_at: str = "2025-01-31T00:00:00Z",
) -> int:
    row = conn.execute(
        _INSERT, [prediction_id, encounter_id, label, label_version, due_at, released_at]
    ).fetchone()
    assert row is not None
    return int(row[0])


def _columns(conn: psycopg.Connection[Any]) -> dict[str, tuple[str, bool]]:
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = 'labels'"
    ).fetchall()
    return {name: (data_type, nullable == "YES") for name, data_type, nullable in rows}


def _constraint_columns(
    conn: psycopg.Connection[Any], constraint_type: str
) -> set[tuple[str, ...]]:
    rows = conn.execute(
        "SELECT tc.constraint_name, kcu.column_name"
        " FROM information_schema.table_constraints AS tc"
        " JOIN information_schema.key_column_usage AS kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "  AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = 'public' AND tc.table_name = 'labels'"
        "   AND tc.constraint_type = %s"
        " ORDER BY kcu.ordinal_position",
        [constraint_type],
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for constraint_name, column_name in rows:
        grouped.setdefault(constraint_name, []).append(column_name)
    return {tuple(columns) for columns in grouped.values()}


def test_labels_columns_names_types_and_nullability(db_conn: psycopg.Connection[Any]) -> None:
    assert _columns(db_conn) == EXPECTED_COLUMNS


def test_label_id_is_the_primary_key(db_conn: psycopg.Connection[Any]) -> None:
    assert _constraint_columns(db_conn, "PRIMARY KEY") == {("label_id",)}


def test_a_label_attaches_to_exactly_one_scored_discharge(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Unique and a foreign key: one label per prediction, none without one."""
    assert ("prediction_id",) in _constraint_columns(db_conn, "UNIQUE")
    assert ("prediction_id",) in _constraint_columns(db_conn, "FOREIGN KEY")


def test_a_label_for_an_unscored_discharge_is_refused(db_conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(errors.ForeignKeyViolation):
        _insert(db_conn, 1)


def test_a_second_label_for_one_prediction_is_refused(db_conn: psycopg.Connection[Any]) -> None:
    scored = _score(db_conn, "e-1")
    _insert(db_conn, scored)
    with pytest.raises(errors.UniqueViolation):
        _insert(db_conn, scored, label=1)


@pytest.mark.parametrize("label", [-1, 2])
def test_a_label_is_zero_or_one(db_conn: psycopg.Connection[Any], label: int) -> None:
    scored = _score(db_conn, "e-1")
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, scored, label=label)


def test_a_label_released_before_it_is_due_is_refused(db_conn: psycopg.Connection[Any]) -> None:
    """The never-early rule is a property of the table, not only of the harness."""
    scored = _score(db_conn, "e-1")
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, scored, due_at="2025-01-31T00:00:00Z", released_at="2025-01-30T23:59:59Z")


def test_a_label_released_exactly_when_due_is_accepted(db_conn: psycopg.Connection[Any]) -> None:
    scored = _score(db_conn, "e-1")
    assert _insert(
        db_conn, scored, due_at="2025-01-31T00:00:00Z", released_at="2025-01-31T00:00:00Z"
    )


def test_label_id_and_recorded_at_are_assigned_by_the_database(
    db_conn: psycopg.Connection[Any],
) -> None:
    ids = [_insert(db_conn, _score(db_conn, f"e-{index}")) for index in range(3)]
    assert ids == [1, 2, 3]
    row = db_conn.execute(
        "SELECT recorded_at IS NOT NULL, recorded_at <= now() FROM labels WHERE label_id = 1"
    ).fetchone()
    assert row == (True, True)


def test_released_at_is_indexed_for_maturing_labels(db_conn: psycopg.Connection[Any]) -> None:
    """Monitoring reads labels as they arrive, by the instant they were released."""
    rows = db_conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'labels'"
    ).fetchall()
    assert any("released_at" in definition for (definition,) in rows)
