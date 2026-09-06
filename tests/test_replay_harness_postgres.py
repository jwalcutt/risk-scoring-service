"""The tick loop against a real service, a real log, and the run row.

The service is an in-process test client over a throwaway database, the
same path an operator's harness takes to the containers. The rules these
tests pin:

- After a preload, a replay logs exactly what posting the same post-start
  events one at a time logs.
- The run row ends at the end instant with the last posted event as its
  cursor, and the loop never touches the status.
- Nothing dated before the start is posted or scored.
- A refusal from the service stops the run with the checkpoint where the
  last complete tick left it.
- The pause contract: when the run row's status becomes ``paused``,
  whoever wrote it, the loop finishes the tick it is in, checkpoints, and
  stops; resuming from that row completes the run to an identical log.

The byte-identity matrix across pauses and kills is in
``test_replay_resume_postgres.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from replay_support import (
    END,
    MAX_SPEED,
    START,
    ClientPoster,
    Serve,
    first_discharge_index,
    prepare,
    read_log,
    schedule_of,
    serving,
    skew_frames,
    stream_of,
    tick_containing,
)
from risk_scoring import predictions, train
from risk_scoring.replay import clock, harness, preload, runs
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent, envelope

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("replay-population") / "csv")


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return schedule_of(frames)


@pytest.fixture(scope="module")
def replayed(events: list[StreamEvent]) -> list[StreamEvent]:
    """The events the run owns: dated at or after the start."""
    return preload.replay_from(events, clock.instant(START))


@pytest.fixture()
def serve(trained_repo: tuple[Any, train.TrainingResult]) -> Serve:
    return serving(trained_repo)


def _replay(
    serve: Serve,
    dsn: str,
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    *,
    make_poster: Callable[[TestClient], ClientPoster] = ClientPoster,
) -> tuple[harness.RunSummary, ClientPoster]:
    """One invocation of the loop from wherever the run row stands."""
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        poster = make_poster(client)
        summary = harness.run_replay(conn, run, events, poster, labels=schedule, pacing=MAX_SPEED)
        return summary, poster


def _per_event_reference(
    serve: Serve,
    dsn: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> list[dict[str, Any]]:
    """The log from posting the post-start events one at a time, no clock at all."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        preload.preload_history(conn, frames, events, clock.instant(START))
    with serve(dsn) as client:
        poster = ClientPoster(client)
        for event in replayed:
            poster.post_event(envelope(event.kind, event.row))
    return read_log(dsn)


def test_the_fixture_replays_something_worth_comparing(replayed: list[StreamEvent]) -> None:
    """Every equality below would hold vacuously over an empty replay."""
    assert sum(event.kind == "encounter" for event in replayed) >= 6
    assert {event.kind for event in replayed} == {"encounter", "medication", "condition"}


def test_a_replay_logs_what_per_event_posting_logs(
    serve: Serve,
    db_url_factory: Any,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
) -> None:
    reference_dsn = db_url_factory()
    replay_dsn = db_url_factory()

    reference = _per_event_reference(serve, reference_dsn, frames, events, replayed)
    prepare(replay_dsn, frames, events)
    summary, poster = _replay(serve, replay_dsn, events, schedule)

    assert summary.finished
    assert read_log(replay_dsn) == reference
    assert len(reference) == 6
    assert summary.discharges_scored == 6
    assert len(poster.posted) == len(replayed)


def test_the_run_row_ends_at_the_end_with_the_last_event_as_its_cursor(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
) -> None:
    created = prepare(db_url, frames, events)
    _replay(serve, db_url, events, schedule)

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert run.sim_now == END
    assert run.cursor == replayed[-1].sort_key
    assert run.status == "running"


def test_nothing_before_the_start_is_posted_or_scored(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
) -> None:
    prepare(db_url, frames, events)
    _, poster = _replay(serve, db_url, events, schedule)

    # Exactly the post-start events, no demographics, nothing from before.
    assert poster.posted == [envelope(event.kind, event.row) for event in replayed]
    assert len(replayed) < len(events)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        scored_at = [row.event_time for row in predictions.all_predictions(conn)]
    assert scored_at and min(scored_at) >= START


def test_a_refusal_stops_the_run_with_the_checkpoint_before_that_tick(
    serve: Serve,
    db_url: str,
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
) -> None:
    """No preload, so the first discharge outruns its patient and the service says no."""
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        created = runs.create_run(
            conn, population="skew", start_at=START, end_at=END, acceleration=4
        )
    first_discharge = replayed[first_discharge_index(replayed) - 1]

    with pytest.raises(RuntimeError, match="refused"):
        _replay(serve, db_url, events, schedule)

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert clock.instant(run.sim_now) < first_discharge.at
    assert run.cursor is None or run.cursor < first_discharge.sort_key
    assert run.status == "running"


# The pause contract


class PauseWriter(ClientPoster):
    """Stands in for whoever writes ``paused`` to the run row mid-run.

    After ``pause_after`` posts it opens its own connection and writes the
    status, as an operator's ``pause`` command or a monitoring alert
    would, from outside the harness process.
    """

    def __init__(self, client: TestClient, *, dsn: str, pause_after: int) -> None:
        super().__init__(client)
        self.dsn = dsn
        self.pause_after = pause_after

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        ack = super().post_event(event)
        if len(self.posted) == self.pause_after:
            with psycopg.connect(self.dsn, connect_timeout=2) as conn:
                run = runs.open_run(conn)
                assert run is not None
                runs.set_status(conn, run.run_id, "paused")
        return ack


def test_the_harness_pauses_when_the_run_row_says_so_and_checkpoints_the_tick(
    serve: Serve,
    db_url_factory: Any,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
) -> None:
    """The field is written from outside mid-tick; the loop stops at that tick's end."""
    straight_dsn = db_url_factory()
    paused_dsn = db_url_factory()
    pause_after = len(replayed) // 2
    paused_tick = tick_containing(replayed[pause_after - 1].at)
    in_that_tick = [event for event in replayed if event.at <= clock.instant(paused_tick)]

    prepare(straight_dsn, frames, events)
    _replay(serve, straight_dsn, events, schedule)

    created = prepare(paused_dsn, frames, events)
    summary, poster = _replay(
        serve,
        paused_dsn,
        events,
        schedule,
        make_poster=lambda client: PauseWriter(client, dsn=paused_dsn, pause_after=pause_after),
    )

    # Stopped within the tick: every event due in it was posted, none after.
    assert summary.paused and not summary.finished
    assert summary.sim_to == paused_tick
    assert poster.posted == [envelope(event.kind, event.row) for event in in_that_tick]
    assert len(in_that_tick) < len(replayed)
    with psycopg.connect(paused_dsn, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert run.sim_now == paused_tick
    assert run.cursor == in_that_tick[-1].sort_key
    # The loop reads the status and never writes it.
    assert run.status == "paused"

    with psycopg.connect(paused_dsn, connect_timeout=2) as conn:
        runs.set_status(conn, created.run_id, "running")
    resumed, second = _replay(serve, paused_dsn, events, schedule)
    assert resumed.finished
    assert read_log(paused_dsn) == read_log(straight_dsn)
    assert poster.posted + second.posted == [envelope(event.kind, event.row) for event in replayed]


def test_a_run_that_starts_paused_posts_nothing(
    serve: Serve,
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    created = prepare(db_url, frames, events)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        runs.set_status(conn, created.run_id, "paused")

    summary, poster = _replay(serve, db_url, events, schedule)

    assert summary.paused
    assert summary.ticks == 0
    assert poster.posted == []
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert run.sim_now == START and run.cursor is None
