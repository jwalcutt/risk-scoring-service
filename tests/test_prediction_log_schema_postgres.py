"""The predictions log schema, asserted column by column.

Every later phase reads this table: drift windows, realized performance,
shadow comparison, and the ablation all query it, and the writeup traces
a score back through it. A migration that renames a column, widens a
type, or drops the uniqueness that keeps one score per discharge would
otherwise change that substrate silently. These assertions are
deliberately literal — they restate the schema rather than deriving it
from the code under test, so a change has to be made in two places on
purpose.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

pytestmark = pytest.mark.db

# column name -> (data type as information_schema reports it, nullable)
EXPECTED_COLUMNS: dict[str, tuple[str, bool]] = {
    "prediction_id": ("bigint", False),
    "patient_id": ("text", False),
    "encounter_id": ("text", False),
    "event_time": ("timestamp with time zone", False),
    "scored_at": ("timestamp with time zone", False),
    "input_hash": ("text", False),
    "model_name": ("text", False),
    "model_version": ("integer", False),
    "feature_version": ("text", False),
    "cohort_version": ("text", False),
    "score": ("double precision", False),
    "features": ("jsonb", False),
}


def _columns(conn: psycopg.Connection[Any]) -> dict[str, tuple[str, bool]]:
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = 'predictions'"
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
        " WHERE tc.table_schema = 'public' AND tc.table_name = 'predictions'"
        "   AND tc.constraint_type = %s"
        " ORDER BY kcu.ordinal_position",
        [constraint_type],
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for constraint_name, column_name in rows:
        grouped.setdefault(constraint_name, []).append(column_name)
    return {tuple(columns) for columns in grouped.values()}


def test_predictions_columns_names_types_and_nullability(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert _columns(db_conn) == EXPECTED_COLUMNS


def test_prediction_id_is_the_primary_key(db_conn: psycopg.Connection[Any]) -> None:
    assert _constraint_columns(db_conn, "PRIMARY KEY") == {("prediction_id",)}


def test_encounter_id_is_unique(db_conn: psycopg.Connection[Any]) -> None:
    """One score per discharge, enforced by the database, not by convention."""
    assert ("encounter_id",) in _constraint_columns(db_conn, "UNIQUE")


def test_prediction_id_is_generated_by_the_database(db_conn: psycopg.Connection[Any]) -> None:
    """Two identical replays of one stream produce the same ids, in order."""
    ids = [
        db_conn.execute(
            "INSERT INTO predictions (patient_id, encounter_id, event_time, input_hash,"
            " model_name, model_version, feature_version, cohort_version, score, features)"
            " VALUES ('p', %s, '2025-01-01T00:00:00Z', 'h', 'm', 1, '1.0.0', '1.0.0', 0.5, '{}')"
            " RETURNING prediction_id",
            [f"encounter-{index}"],
        ).fetchone()
        for index in range(3)
    ]
    assert [row[0] for row in ids if row is not None] == [1, 2, 3]


def test_event_time_is_indexed_for_rolling_windows(db_conn: psycopg.Connection[Any]) -> None:
    """Monitoring reads this table by time window on every evaluation."""
    rows = db_conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'predictions'"
    ).fetchall()
    assert any("event_time" in definition for (definition,) in rows)
