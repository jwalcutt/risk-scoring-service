"""The replay run row's schema, asserted column by column.

The run row is both the simulated clock and the checkpoint: the harness
resumes from it, monitoring reads ``sim_now`` from it, and whoever writes
``paused`` into it stops the clock. As with the predictions log, these
assertions restate the schema literally rather than deriving it from the
code under test, so a migration that changes the substrate has to change
a test on purpose.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg import errors

pytestmark = pytest.mark.db

# column name -> (data type as information_schema reports it, nullable)
EXPECTED_COLUMNS: dict[str, tuple[str, bool]] = {
    "run_id": ("bigint", False),
    "population": ("text", False),
    "start_at": ("timestamp with time zone", False),
    "end_at": ("timestamp with time zone", False),
    "acceleration": ("double precision", False),
    "sim_now": ("timestamp with time zone", False),
    "status": ("text", False),
    "cursor_at": ("text", True),
    "cursor_kind": ("integer", True),
    "cursor_row": ("text", True),
    "created_at": ("timestamp with time zone", False),
    "updated_at": ("timestamp with time zone", False),
}

_INSERT = (
    "INSERT INTO replay_runs (population, start_at, end_at, acceleration, sim_now, status,"
    " cursor_at, cursor_kind, cursor_row)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id"
)


def _insert(
    conn: psycopg.Connection[Any],
    *,
    population: str = "baseline",
    start_at: str = "2025-01-01T00:00:00Z",
    end_at: str = "2026-01-01T00:00:00Z",
    acceleration: float = 4.0,
    sim_now: str = "2025-01-01T00:00:00Z",
    status: str = "running",
    cursor: tuple[str | None, int | None, str | None] = (None, None, None),
) -> int:
    row = conn.execute(
        _INSERT,
        [population, start_at, end_at, acceleration, sim_now, status, *cursor],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _columns(conn: psycopg.Connection[Any]) -> dict[str, tuple[str, bool]]:
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = 'replay_runs'"
    ).fetchall()
    return {name: (data_type, nullable == "YES") for name, data_type, nullable in rows}


def _primary_key_columns(conn: psycopg.Connection[Any]) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT kcu.column_name"
        " FROM information_schema.table_constraints AS tc"
        " JOIN information_schema.key_column_usage AS kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "  AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = 'public' AND tc.table_name = 'replay_runs'"
        "   AND tc.constraint_type = 'PRIMARY KEY'"
        " ORDER BY kcu.ordinal_position"
    ).fetchall()
    return tuple(name for (name,) in rows)


def test_replay_runs_columns_names_types_and_nullability(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert _columns(db_conn) == EXPECTED_COLUMNS


def test_run_id_is_the_primary_key(db_conn: psycopg.Connection[Any]) -> None:
    assert _primary_key_columns(db_conn) == ("run_id",)


def test_run_id_is_generated_by_the_database(db_conn: psycopg.Connection[Any]) -> None:
    first = _insert(db_conn)
    db_conn.execute("UPDATE replay_runs SET status = 'finished'")
    second = _insert(db_conn)
    assert (first, second) == (1, 2)


def test_timestamps_default_to_the_wall_clock(db_conn: psycopg.Connection[Any]) -> None:
    run_id = _insert(db_conn)
    row = db_conn.execute(
        "SELECT created_at IS NOT NULL, updated_at IS NOT NULL, created_at <= now()"
        " FROM replay_runs WHERE run_id = %s",
        [run_id],
    ).fetchone()
    assert row == (True, True, True)


@pytest.mark.parametrize("status", ["stopped", "RUNNING", ""])
def test_status_outside_the_three_states_is_rejected(
    db_conn: psycopg.Connection[Any], status: str
) -> None:
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, status=status)


@pytest.mark.parametrize("open_status", ["running", "paused"])
def test_a_second_unfinished_run_is_rejected(
    db_conn: psycopg.Connection[Any], open_status: str
) -> None:
    """One database hosts one scoring run at a time; resume must find exactly one row."""
    _insert(db_conn, status=open_status)
    with pytest.raises(errors.UniqueViolation):
        _insert(db_conn, status="running")


def test_a_new_run_is_accepted_once_the_previous_one_finished(
    db_conn: psycopg.Connection[Any],
) -> None:
    _insert(db_conn, status="finished")
    _insert(db_conn, status="running")
    count = db_conn.execute("SELECT count(*) FROM replay_runs").fetchone()
    assert count == (2,)


@pytest.mark.parametrize(
    "cursor",
    [
        ("2025-01-02T00:00:00Z", None, None),
        (None, 2, None),
        (None, None, "[]"),
        ("2025-01-02T00:00:00Z", 2, None),
    ],
)
def test_cursor_columns_are_null_together(
    db_conn: psycopg.Connection[Any], cursor: tuple[str | None, int | None, str | None]
) -> None:
    """A half-written cursor is no checkpoint; the database refuses it."""
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, cursor=cursor)


def test_a_complete_cursor_is_accepted(db_conn: psycopg.Connection[Any]) -> None:
    _insert(db_conn, cursor=("2025-01-02T00:00:00Z", 2, "[('Id', 'e-1')]"))


@pytest.mark.parametrize("end_at", ["2025-01-01T00:00:00Z", "2024-12-31T00:00:00Z"])
def test_end_must_follow_start(db_conn: psycopg.Connection[Any], end_at: str) -> None:
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, end_at=end_at)


def test_sim_now_cannot_precede_start(db_conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, sim_now="2024-12-31T23:00:00Z")


@pytest.mark.parametrize("acceleration", [0.0, -4.0])
def test_acceleration_must_be_positive(
    db_conn: psycopg.Connection[Any], acceleration: float
) -> None:
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, acceleration=acceleration)


def test_population_must_be_named(db_conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(errors.CheckViolation):
        _insert(db_conn, population="")
