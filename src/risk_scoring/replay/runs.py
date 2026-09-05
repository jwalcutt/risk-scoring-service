"""The replay run row: the simulated clock and the checkpoint in one place.

A run is one row in ``replay_runs``. ``sim_now`` is the run's simulated
instant; the cursor is the sort key of the last event posted, so a resumed
stream continues at the same event whatever list it was rebuilt from; the
status is how the clock is paused, by whoever writes it.

Judgment calls this module fixes:

- Each write commits on its own, matching the state layer and the log. A
  checkpoint that could be rolled back by a later failure would let the
  harness re-post from a point earlier than the one it reported.
- The database, not this module, enforces that a database holds at most
  one unfinished run. Creating a second one surfaces here as
  :class:`OpenRunError` rather than a driver exception, so the command
  layer can say what to do about it.
- Statuses are validated before any SQL so a typo fails the same way with
  or without a database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg import errors

RUN_STATUSES = frozenset({"running", "paused", "finished"})

StreamCursor = tuple[str, int, str]
"""Exactly ``stream.StreamEvent.sort_key``: arrival instant, kind order, row."""

_READ_COLUMNS = (
    "run_id",
    "population",
    "start_at",
    "end_at",
    "acceleration",
    "sim_now",
    "status",
    "cursor_at",
    "cursor_kind",
    "cursor_row",
    "created_at",
    "updated_at",
)


class OpenRunError(RuntimeError):
    """A run that is not finished already exists in this database."""


@dataclass(frozen=True)
class ReplayRun:
    """One replay run as stored, timestamps aware and in UTC."""

    run_id: int
    population: str
    start_at: datetime
    end_at: datetime
    acceleration: float
    sim_now: datetime
    status: str
    cursor: StreamCursor | None
    created_at: datetime
    updated_at: datetime


def create_run(
    conn: psycopg.Connection[Any],
    *,
    population: str,
    start_at: datetime,
    end_at: datetime,
    acceleration: float,
) -> ReplayRun:
    """Insert a running run whose clock stands at its start, and commit."""
    try:
        row = conn.execute(
            "INSERT INTO replay_runs (population, start_at, end_at, acceleration, sim_now, status)"
            " VALUES (%s, %s, %s, %s, %s, 'running')"
            f" RETURNING {', '.join(_READ_COLUMNS)}",
            [population, start_at, end_at, acceleration, start_at],
        ).fetchone()
        conn.commit()
    except errors.UniqueViolation as exc:
        conn.rollback()
        raise OpenRunError(
            "a replay run that is not finished already exists in this database;"
            " resume it or finish it before starting another"
        ) from exc
    except Exception:
        conn.rollback()
        raise
    if row is None:
        raise RuntimeError("replay_runs insert returned no row")
    return _stored(row)


def open_run(conn: psycopg.Connection[Any]) -> ReplayRun | None:
    """The one run that is not finished, or None; read-only."""
    row = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM replay_runs WHERE status <> 'finished'"
    ).fetchone()
    return None if row is None else _stored(row)


def read_run(conn: psycopg.Connection[Any], run_id: int) -> ReplayRun:
    """One run by id; read-only. Raises LookupError if there is none."""
    row = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM replay_runs WHERE run_id = %s", [run_id]
    ).fetchone()
    if row is None:
        raise LookupError(f"no replay run with id {run_id}")
    return _stored(row)


def read_status(conn: psycopg.Connection[Any], run_id: int) -> str:
    """One run's status; read-only. Raises LookupError if there is none.

    The tick loop asks this once per tick, so it reads the one column it
    needs rather than the whole row.
    """
    row = conn.execute("SELECT status FROM replay_runs WHERE run_id = %s", [run_id]).fetchone()
    if row is None:
        raise LookupError(f"no replay run with id {run_id}")
    status: str = row[0]
    return status


def checkpoint(
    conn: psycopg.Connection[Any],
    run_id: int,
    *,
    sim_now: datetime,
    cursor: StreamCursor | None,
) -> None:
    """Advance the clock and record the last posted event, and commit."""
    cursor_at, cursor_kind, cursor_row = (None, None, None) if cursor is None else cursor
    _write(
        conn,
        "UPDATE replay_runs SET sim_now = %s, cursor_at = %s, cursor_kind = %s,"
        " cursor_row = %s, updated_at = now() WHERE run_id = %s",
        [sim_now, cursor_at, cursor_kind, cursor_row, run_id],
    )


def set_status(conn: psycopg.Connection[Any], run_id: int, status: str) -> None:
    """Write the run's status, and commit."""
    if status not in RUN_STATUSES:
        raise ValueError(f"status must be one of {sorted(RUN_STATUSES)}; got {status!r}")
    _write(
        conn,
        "UPDATE replay_runs SET status = %s, updated_at = now() WHERE run_id = %s",
        [status, run_id],
    )


def _write(conn: psycopg.Connection[Any], statement: str, params: Sequence[Any]) -> None:
    try:
        updated = conn.execute(statement, params).rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if updated != 1:
        raise LookupError(f"no replay run with id {params[-1]}")


def _stored(row: Sequence[Any]) -> ReplayRun:
    values = dict(zip(_READ_COLUMNS, row, strict=True))
    cursor_parts = (values.pop("cursor_at"), values.pop("cursor_kind"), values.pop("cursor_row"))
    cursor: StreamCursor | None = None if cursor_parts[0] is None else cursor_parts
    return ReplayRun(cursor=cursor, **values)
