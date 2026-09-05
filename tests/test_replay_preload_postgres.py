"""Preloading history from before a replay's start straight into state.

The contract under test: after a preload, state holds every patient and
every event dated before the start, byte-identical to what per-event
ingestion would have stored, and nothing dated at or after it; nothing
has been scored; a second preload is a no-op; and a divergent row already
in state is refused as loudly as it is on the per-event path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from factories import write_skew_population
from risk_scoring import state
from risk_scoring.populations import load_population
from risk_scoring.replay import preload
from risk_scoring.stream import StreamEvent, ordered_events

pytestmark = pytest.mark.db

BEFORE = "2024-06-01T00:00:00Z"

# Cohort discharges in the skew population dated before BEFORE, counted by
# hand from the factory: e-edge-in, e-gap-ancient, e-gap-index, the two
# overlapping stays, both readmission stays, and e-fresh. The minor and the
# in-hospital death are excluded by the cohort rules; e-edge-index and
# e-full-index are discharged after BEFORE.
DISCHARGES_BEFORE = 8

_EVENT_TYPES: dict[str, type[state.AnyEvent]] = {
    "patient": state.PatientEvent,
    "encounter": state.EncounterEvent,
    "medication": state.MedicationEvent,
    "condition": state.ConditionEvent,
}


@pytest.fixture
def frames(tmp_path: Path) -> dict[str, pd.DataFrame]:
    csv_dir = tmp_path / "csv"
    write_skew_population(csv_dir)
    return load_population(csv_dir)


@pytest.fixture
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return ordered_events(frames["encounters"], frames["medications"], frames["conditions"])


def _expected_rows(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Row counts by kind from the source frames, by the arrival-instant rule."""
    return {
        "patient": len(frames["patients"]),
        "encounter": int((frames["encounters"]["STOP"] < BEFORE).sum()),
        "medication": int((frames["medications"]["START"] < BEFORE).sum()),
        "condition": int((frames["conditions"]["START"] + "T00:00:00Z" < BEFORE).sum()),
    }


def _table_counts(conn: psycopg.Connection[Any]) -> dict[str, int]:
    counts = {}
    for kind, table in (
        ("patient", "patients"),
        ("encounter", "encounters"),
        ("medication", "medications"),
        ("condition", "conditions"),
    ):
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert row is not None
        counts[kind] = int(row[0])
    return counts


def test_preload_loads_every_patient_and_every_earlier_event(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    summary = preload.preload_history(db_conn, frames, events, BEFORE)

    expected = _expected_rows(frames)
    assert summary.rows_loaded == expected
    assert _table_counts(db_conn) == expected
    assert summary.rows_already_present == 0
    assert summary.before == BEFORE


def test_preload_loads_nothing_dated_at_or_after_the_start(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    preload.preload_history(db_conn, frames, events, BEFORE)

    stored_encounters = db_conn.execute("SELECT id FROM encounters").fetchall()
    stored_ids = {row[0] for row in stored_encounters}
    for event in events:
        if event.kind == "encounter" and event.at >= BEFORE:
            assert event.row["Id"] not in stored_ids
    assert "e-edge-index" not in stored_ids
    assert "e-full-index" not in stored_ids


def test_preload_scores_nothing(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    summary = preload.preload_history(db_conn, frames, events, BEFORE)

    assert db_conn.execute("SELECT count(*) FROM predictions").fetchone() == (0,)
    assert summary.discharges_unscored == DISCHARGES_BEFORE


def test_preloaded_state_is_identical_to_per_event_ingestion(
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
) -> None:
    """Byte-identical read-back is what the skew check rides on; the bulk path keeps it."""
    with psycopg.connect(db_url_factory()) as bulk, psycopg.connect(db_url_factory()) as per_row:
        preload.preload_history(bulk, frames, events, BEFORE)
        for _, patient in frames["patients"].iterrows():
            state.record_event(per_row, state.PatientEvent.from_row(dict(patient)))
        for event in preload.history_before(events, BEFORE):
            state.record_event(per_row, _EVENT_TYPES[event.kind].from_row(event.row))

        for patient_id in frames["patients"]["Id"]:
            expected = state.patient_history(per_row, patient_id)
            actual = state.patient_history(bulk, patient_id)
            pd.testing.assert_frame_equal(actual.patients, expected.patients)
            pd.testing.assert_frame_equal(actual.encounters, expected.encounters)
            pd.testing.assert_frame_equal(actual.medications, expected.medications)
            pd.testing.assert_frame_equal(actual.conditions, expected.conditions)


def test_a_second_preload_is_a_noop(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    """A load that died halfway is resumed by loading again."""
    first = preload.preload_history(db_conn, frames, events, BEFORE)
    second = preload.preload_history(db_conn, frames, events, BEFORE)

    assert second.rows_loaded == dict.fromkeys(first.rows_loaded, 0)
    assert second.rows_already_present == sum(first.rows_loaded.values())
    assert _table_counts(db_conn) == first.rows_loaded


def test_a_small_batch_size_loads_everything(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    summary = preload.preload_history(db_conn, frames, events, BEFORE, batch_size=3)
    assert summary.rows_loaded == _expected_rows(frames)
    assert _table_counts(db_conn) == _expected_rows(frames)


def test_batch_size_must_be_positive(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        preload.preload_history(db_conn, frames, events, BEFORE, batch_size=0)


def test_a_divergent_row_already_in_state_is_refused(
    db_conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> None:
    stored = frames["encounters"].loc[frames["encounters"]["Id"] == "e-fresh"].iloc[0].to_dict()
    stored["ENCOUNTERCLASS"] = "emergency"
    state.record_event(db_conn, state.EncounterEvent.from_row(stored))

    with pytest.raises(state.EventConflictError, match="e-fresh"):
        preload.preload_history(db_conn, frames, events, BEFORE)
