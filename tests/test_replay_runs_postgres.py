"""Reading and writing the replay run row.

The contract under test: a created run starts at its own start with no
cursor, every write commits on its own, the cursor round-trips as the
stream's three-part sort key, and a database can hold only one run that
is not finished.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from risk_scoring.replay import runs

pytestmark = pytest.mark.db

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)


def _create(conn: psycopg.Connection[Any], **overrides: Any) -> runs.ReplayRun:
    settings: dict[str, Any] = {
        "population": "baseline",
        "start_at": START,
        "end_at": END,
        "acceleration": 4.0,
    }
    settings.update(overrides)
    return runs.create_run(conn, **settings)


def test_a_created_run_starts_at_its_start_with_no_cursor(
    db_conn: psycopg.Connection[Any],
) -> None:
    run = _create(db_conn)

    assert run.population == "baseline"
    assert run.start_at == START
    assert run.end_at == END
    assert run.acceleration == 4.0
    assert run.sim_now == START
    assert run.status == "running"
    assert run.cursor is None
    assert run.created_at == run.updated_at


def test_a_created_run_reads_back_equal(db_conn: psycopg.Connection[Any]) -> None:
    run = _create(db_conn)
    assert runs.read_run(db_conn, run.run_id) == run


def test_a_created_run_survives_a_rollback(db_conn: psycopg.Connection[Any]) -> None:
    """Each write commits on its own; a later rollback cannot take it back."""
    run = _create(db_conn)
    db_conn.rollback()
    assert runs.read_run(db_conn, run.run_id) == run


def test_creating_a_run_while_one_is_open_fails_loudly(
    db_conn: psycopg.Connection[Any],
) -> None:
    first = _create(db_conn)
    with pytest.raises(runs.OpenRunError, match="already"):
        _create(db_conn)
    # The connection is still usable and the first run untouched.
    assert runs.read_run(db_conn, first.run_id) == first


@pytest.mark.parametrize("closing_status", ["finished"])
def test_a_new_run_can_start_once_the_previous_one_finished(
    db_conn: psycopg.Connection[Any], closing_status: str
) -> None:
    first = _create(db_conn)
    runs.set_status(db_conn, first.run_id, closing_status)
    second = _create(db_conn)
    assert second.run_id != first.run_id


def test_open_run_is_none_on_an_empty_database(db_conn: psycopg.Connection[Any]) -> None:
    assert runs.open_run(db_conn) is None


@pytest.mark.parametrize("status", ["running", "paused"])
def test_open_run_returns_the_one_unfinished_run(
    db_conn: psycopg.Connection[Any], status: str
) -> None:
    run = _create(db_conn)
    runs.set_status(db_conn, run.run_id, status)
    found = runs.open_run(db_conn)
    assert found is not None
    assert found.run_id == run.run_id
    assert found.status == status


def test_open_run_ignores_finished_runs(db_conn: psycopg.Connection[Any]) -> None:
    run = _create(db_conn)
    runs.set_status(db_conn, run.run_id, "finished")
    assert runs.open_run(db_conn) is None


def test_checkpoint_writes_sim_now_and_the_cursor(db_conn: psycopg.Connection[Any]) -> None:
    run = _create(db_conn)
    cursor = ("2025-01-02T13:00:00Z", 2, "[('Id', 'e-1'), ('STOP', '2025-01-02T13:00:00Z')]")
    later = datetime(2025, 1, 2, 14, tzinfo=UTC)

    runs.checkpoint(db_conn, run.run_id, sim_now=later, cursor=cursor)

    stored = runs.read_run(db_conn, run.run_id)
    assert stored.sim_now == later
    assert stored.cursor == cursor
    assert stored.status == "running"
    assert stored.updated_at >= run.updated_at


def test_checkpoint_can_clear_nothing_but_keeps_an_empty_tick(
    db_conn: psycopg.Connection[Any],
) -> None:
    """A tick that posted nothing still advances the clock; the cursor stays."""
    run = _create(db_conn)
    cursor = ("2025-01-02T13:00:00Z", 2, "row")
    two_pm = datetime(2025, 1, 2, 14, tzinfo=UTC)
    three_pm = datetime(2025, 1, 2, 15, tzinfo=UTC)
    runs.checkpoint(db_conn, run.run_id, sim_now=two_pm, cursor=cursor)
    runs.checkpoint(db_conn, run.run_id, sim_now=three_pm, cursor=cursor)

    stored = runs.read_run(db_conn, run.run_id)
    assert stored.sim_now == three_pm
    assert stored.cursor == cursor


def test_checkpoint_survives_a_rollback(db_conn: psycopg.Connection[Any]) -> None:
    run = _create(db_conn)
    later = datetime(2025, 1, 1, 1, tzinfo=UTC)
    runs.checkpoint(db_conn, run.run_id, sim_now=later, cursor=None)
    db_conn.rollback()
    assert runs.read_run(db_conn, run.run_id).sim_now == later


@pytest.mark.parametrize("status", ["paused", "running", "finished"])
def test_set_status_round_trips(db_conn: psycopg.Connection[Any], status: str) -> None:
    run = _create(db_conn)
    runs.set_status(db_conn, run.run_id, status)
    stored = runs.read_run(db_conn, run.run_id)
    assert stored.status == status
    assert stored.updated_at >= run.updated_at


def test_set_status_rejects_an_unknown_status_before_any_sql(
    db_conn: psycopg.Connection[Any],
) -> None:
    run = _create(db_conn)
    with pytest.raises(ValueError, match="stopped"):
        runs.set_status(db_conn, run.run_id, "stopped")
    assert runs.read_run(db_conn, run.run_id).status == "running"


def test_read_run_on_a_missing_id_raises(db_conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(LookupError, match="42"):
        runs.read_run(db_conn, 42)


def test_run_statuses_are_the_three_the_database_accepts() -> None:
    assert sorted(runs.RUN_STATUSES) == ["finished", "paused", "running"]
