"""Delayed labels against a real service, a real log, and the labels table.

The rules these tests pin, over the synthetic skew population and a
boundary population built for the purpose:

- Every label the harness releases equals the batch label for that
  discharge: ground truth comes from the export, never from state.
- No label is released before its discharge is 30 simulated days old,
  asserted by the same query the end-to-end run will use, and a
  readmission on day 29 has no label on day 29 and one on day 30.
- A discharge inside the final 30 days of the run gets no label, ever,
  and the run summary reports it pending.
- A label falling due for a discharge the log never scored stops the run
  before that tick is checkpointed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import psycopg
import pytest

from factories import make_encounter_row, make_patient_row, payload_frame
from replay_support import (
    END,
    MAX_SPEED,
    START,
    ClientPoster,
    Serve,
    prepare,
    read_labels,
    read_log,
    schedule_of,
    serving,
    skew_frames,
    stream_of,
)
from risk_scoring import label_log, state, train
from risk_scoring.labels import LABEL_VERSION, READMISSION_WINDOW_DAYS
from risk_scoring.replay import audit, clock, harness, runs
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent

pytestmark = pytest.mark.db

WINDOW = timedelta(days=READMISSION_WINDOW_DAYS)


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("labels-population") / "csv")


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return schedule_of(frames)


@pytest.fixture()
def serve(trained_repo: tuple[Any, train.TrainingResult]) -> Serve:
    return serving(trained_repo)


def _replay(
    serve: Serve,
    dsn: str,
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    *,
    pause_at: datetime | None = None,
) -> harness.RunSummary:
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        return harness.run_replay(
            conn,
            run,
            events,
            ClientPoster(client),
            labels=schedule,
            pacing=MAX_SPEED,
            pause_requested=lambda sim_now: pause_at is not None and sim_now >= pause_at,
        )


# The skew population


def test_every_released_label_is_the_batch_label_for_its_discharge(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    prepare(db_url, frames, events)
    _replay(serve, db_url, events, schedule)

    batch = audit.batch_labels(frames)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        released = label_log.all_labels(conn)
    assert released
    assert {row.encounter_id: row.label for row in released} == {
        name: batch[name] for name in (row.encounter_id for row in released)
    }
    assert {row.label_version for row in released} == {LABEL_VERSION}
    assert sum(row.label for row in released) == 1  # the one readmission in the span


def test_no_label_is_released_before_its_discharge_is_thirty_days_old(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    prepare(db_url, frames, events)
    _replay(serve, db_url, events, schedule)

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        assert audit.early_labels(conn) == 0
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        rows = conn.execute(
            "SELECT p.event_time, l.due_at, l.released_at"
            " FROM labels l JOIN predictions p USING (prediction_id)"
        ).fetchall()
    assert len(rows) == 5
    for event_time, due_at, released_at in rows:
        assert due_at == event_time + WINDOW
        assert released_at >= due_at
        # Released within the tick that made it due, never a tick late.
        assert released_at - due_at < clock.TICK


def test_a_discharge_inside_the_final_thirty_days_gets_no_label_and_is_pending(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    prepare(db_url, frames, events)
    summary = _replay(serve, db_url, events, schedule)

    assert summary.finished
    assert (summary.labels_released, summary.labels_pending) == (5, 1)
    scored = [row["encounter_id"] for row in read_log(db_url)]
    labelled = [row["encounter_id"] for row in read_labels(db_url)]
    assert len(scored) == 6
    assert set(scored) - set(labelled) == {"e-full-index"}
    # Discharged 2024-08-05, two days before the end: due in September.
    assert datetime(2024, 8, 5, 6, tzinfo=UTC) + WINDOW > END


def test_a_label_due_for_an_unscored_discharge_stops_the_run_before_its_checkpoint(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    """The schedule and the service disagree on the cohort: a defect, not a skip."""
    ghost = ScheduledLabel(
        due_at="2024-05-11T08:00:00Z",
        encounter_id="e-ghost",
        discharged_at="2024-04-11T08:00:00Z",
        label=0,
    )
    created = prepare(db_url, frames, events)
    with pytest.raises(label_log.UnscoredDischargeError, match="e-ghost"):
        _replay(serve, db_url, events, sorted([*schedule, ghost]))

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert clock.instant(run.sim_now) < ghost.due_at
    assert run.sim_now == datetime(2024, 5, 11, 7, tzinfo=UTC)
    # What was released before that tick stands; nothing from the tick does.
    assert [row["encounter_id"] for row in read_labels(db_url)] == ["e-gap-overlap-a"]


# The day-29 boundary


def _boundary_frames() -> dict[str, pd.DataFrame]:
    """Three index stays discharged together, readmitted on day 29, day 30, and a second past."""
    discharged = datetime(2024, 4, 5, 8, tzinfo=UTC)
    readmissions = {
        "p-29": discharged + WINDOW - timedelta(days=1),
        "p-30": discharged + WINDOW,
        "p-31": discharged + WINDOW + timedelta(seconds=1),
    }
    patients = [make_patient_row(Id=patient, BIRTHDATE="1960-01-01") for patient in readmissions]
    encounters = []
    for patient, readmitted in readmissions.items():
        encounters.append(
            make_encounter_row(
                Id=f"e-index-{patient}",
                PATIENT=patient,
                ENCOUNTERCLASS="inpatient",
                START=clock.instant(discharged - timedelta(days=3)),
                STOP=clock.instant(discharged),
            )
        )
        encounters.append(
            make_encounter_row(
                Id=f"e-readmit-{patient}",
                PATIENT=patient,
                ENCOUNTERCLASS="inpatient",
                START=clock.instant(readmitted),
                STOP=clock.instant(readmitted + timedelta(days=2)),
            )
        )
    return {
        "patients": pd.DataFrame(patients),
        "encounters": pd.DataFrame(encounters),
        "medications": payload_frame([], state.MEDICATION_COLUMNS),
        "conditions": payload_frame([], state.CONDITION_COLUMNS),
    }


def test_a_readmission_on_day_29_has_no_label_on_day_29_and_one_on_day_30(
    serve: Serve, db_url_factory: Callable[[], str]
) -> None:
    frames = _boundary_frames()
    events, schedule = stream_of(frames), schedule_of(frames)
    day_29 = datetime(2024, 5, 4, 8, tzinfo=UTC)
    day_30 = day_29 + timedelta(days=1)
    end = datetime(2024, 5, 12, tzinfo=UTC)
    dsn = db_url_factory()
    prepare(dsn, frames, events, start=START, end=end)

    paused = _replay(serve, dsn, events, schedule, pause_at=day_29)
    assert paused.paused and paused.sim_to == day_29
    assert read_labels(dsn) == []
    assert len(read_log(dsn)) == 3

    finished = _replay(serve, dsn, events, schedule)
    assert finished.finished
    assert {row["encounter_id"]: row["label"] for row in read_labels(dsn)} == {
        "e-index-p-29": 1,
        "e-index-p-30": 1,
        "e-index-p-31": 0,
    }
    assert {row["released_at"] for row in read_labels(dsn)} == {day_30}
    # The readmission stays were scored too; their own labels fall due after the end.
    assert len(read_log(dsn)) == 6
    assert (finished.labels_released, finished.labels_pending) == (3, 3)
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        assert audit.early_labels(conn) == 0
