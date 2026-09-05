"""The tick loop against a real service, a real log, and the run row.

The service is an in-process test client over a throwaway database, the
same path an operator's harness takes to the containers. The rules these
tests pin:

- After a preload, a replay logs exactly what posting the same post-start
  events one at a time logs.
- A run killed between a post and its checkpoint re-posts from the last
  checkpoint on resume; the service answers the repeat as a no-op, and
  the log is identical to an uninterrupted run. Idempotent ingestion is
  what makes the per-tick checkpoint cheap, and this is the proof.
- The run row ends at the end instant with the last posted event as its
  cursor, and the loop never touches the status.
- Nothing dated before the start is posted or scored.
- A refusal from the service stops the run with the checkpoint where the
  last complete tick left it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from factories import write_skew_population
from risk_scoring import predictions, train
from risk_scoring.populations import load_population
from risk_scoring.replay import clock, harness, preload, runs
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.stream import StreamEvent, envelope, ordered_events
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

# Four cohort discharges fall before this start and six after it; the
# population's last event is dated the day before this end.
START = datetime(2024, 4, 1, tzinfo=UTC)
END = datetime(2024, 8, 7, tzinfo=UTC)
MAX_SPEED = clock.Pacing(acceleration=4, max_speed=True)

# Assigned by the database, so they differ between two runs of one stream
# by design and say nothing about whether the runs agree.
_VOLATILE_COLUMNS = ("prediction_id", "scored_at")


class KilledMidTick(RuntimeError):
    """The harness process died after a post and before its checkpoint."""


class ClientPoster:
    """The poster the loop needs, backed by the FastAPI test client."""

    def __init__(self, client: TestClient, *, die_after: int | None = None) -> None:
        self.client = client
        self.die_after = die_after
        self.posted: list[Mapping[str, Any]] = []
        self.acks: list[dict[str, Any]] = []

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.post("/events", json=event)
        if response.status_code != 202:
            raise RuntimeError(f"{event.get('event_type')} refused: {response.status_code}")
        self.posted.append(event)
        ack = dict(response.json())
        self.acks.append(ack)
        if self.die_after is not None and len(self.posted) == self.die_after:
            raise KilledMidTick(f"died after {self.die_after} posts")
        return ack


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    csv_dir = tmp_path_factory.mktemp("replay-population") / "csv"
    write_skew_population(csv_dir)
    return load_population(csv_dir)


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return ordered_events(frames["encounters"], frames["medications"], frames["conditions"])


@pytest.fixture(scope="module")
def replayed(events: list[StreamEvent]) -> list[StreamEvent]:
    """The events the run owns: dated at or after the start."""
    return preload.replay_from(events, clock.instant(START))


@pytest.fixture()
def serve(trained_repo: tuple[Path, train.TrainingResult]) -> Callable[[str], Any]:
    root, trained = trained_repo

    @contextmanager
    def instance(dsn: str) -> Iterator[TestClient]:
        app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, dsn)
        with TestClient(app) as client:
            yield client

    return instance


def _prepare(
    dsn: str, frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> runs.ReplayRun:
    """Preload history and open the run row, as the start command will."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        preload.preload_history(conn, frames, events, clock.instant(START))
        return runs.create_run(conn, population="skew", start_at=START, end_at=END, acceleration=4)


def _replay(
    serve: Callable[[str], Any],
    dsn: str,
    events: list[StreamEvent],
    *,
    die_after: int | None = None,
) -> tuple[harness.RunSummary | None, ClientPoster]:
    """One invocation of the loop from wherever the run row stands."""
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        poster = ClientPoster(client, die_after=die_after)
        try:
            summary = harness.run_replay(conn, run, events, poster, pacing=MAX_SPEED)
        except KilledMidTick:
            return None, poster
        return summary, poster


def _log(dsn: str) -> list[dict[str, Any]]:
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        rows = predictions.all_predictions(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in _VOLATILE_COLUMNS}
        for row in rows
    ]


def _per_event_reference(
    serve: Callable[[str], Any],
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
    return _log(dsn)


def test_the_fixture_replays_something_worth_comparing(replayed: list[StreamEvent]) -> None:
    """Every equality below would hold vacuously over an empty replay."""
    assert sum(event.kind == "encounter" for event in replayed) >= 6
    assert {event.kind for event in replayed} == {"encounter", "medication", "condition"}


def test_a_replay_logs_what_per_event_posting_logs(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    reference_dsn = db_url_factory()
    replay_dsn = db_url_factory()

    reference = _per_event_reference(serve, reference_dsn, frames, events, replayed)
    _prepare(replay_dsn, frames, events)
    summary, poster = _replay(serve, replay_dsn, events)

    assert summary is not None and summary.finished
    assert _log(replay_dsn) == reference
    assert len(reference) == 6
    assert summary.discharges_scored == 6
    assert len(poster.posted) == len(replayed)


def _discharge_index(replayed: list[StreamEvent]) -> int:
    for index, event in enumerate(replayed):
        if event.kind == "encounter" and event.row["ENCOUNTERCLASS"] == "inpatient":
            return index + 1
    raise AssertionError("no inpatient discharge after the start")


KILL_POINTS: list[Any] = [
    pytest.param(lambda replayed: 1, id="after-the-first-post"),
    pytest.param(_discharge_index, id="just-after-a-discharge"),
    pytest.param(lambda replayed: len(replayed) // 2, id="mid-stream"),
    pytest.param(lambda replayed: len(replayed), id="after-the-last-post"),
]


@pytest.mark.parametrize("choose_kill", KILL_POINTS)
def test_a_run_killed_before_its_checkpoint_resumes_to_an_identical_log(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
    choose_kill: Callable[[list[StreamEvent]], int],
) -> None:
    uninterrupted_dsn = db_url_factory()
    killed_dsn = db_url_factory()
    die_after = choose_kill(replayed)

    _prepare(uninterrupted_dsn, frames, events)
    _replay(serve, uninterrupted_dsn, events)

    _prepare(killed_dsn, frames, events)
    _, first = _replay(serve, killed_dsn, events, die_after=die_after)
    summary, second = _replay(serve, killed_dsn, events)

    assert summary is not None and summary.finished
    assert _log(killed_dsn) == _log(uninterrupted_dsn)
    # The resume re-posts from the last checkpoint: the killed post, and
    # whatever preceded it in the same tick. The service treats every
    # repeat as nothing new, and the two invocations together cover the
    # stream exactly once apart from that overlap.
    re_posted = [event for event in second.posted if event in first.posted]
    assert re_posted and first.posted[-1] in re_posted
    assert first.posted[-len(re_posted) :] == re_posted == second.posted[: len(re_posted)]
    assert all(ack["scored"] is False for ack in second.acks[: len(re_posted)])
    expected = [envelope(event.kind, event.row) for event in replayed]
    assert first.posted[: -len(re_posted)] + second.posted == expected


def test_the_run_row_ends_at_the_end_with_the_last_event_as_its_cursor(
    serve: Callable[[str], Any],
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    created = _prepare(db_url, frames, events)
    _replay(serve, db_url, events)

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert run.sim_now == END
    assert run.cursor == replayed[-1].sort_key
    assert run.status == "running"


def test_nothing_before_the_start_is_posted_or_scored(
    serve: Callable[[str], Any],
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    _prepare(db_url, frames, events)
    _, poster = _replay(serve, db_url, events)

    # Exactly the post-start events, no demographics, nothing from before.
    assert poster.posted == [envelope(event.kind, event.row) for event in replayed]
    assert len(replayed) < len(events)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        scored_at = [row.event_time for row in predictions.all_predictions(conn)]
    assert scored_at and min(scored_at) >= START


def test_a_refusal_stops_the_run_with_the_checkpoint_before_that_tick(
    serve: Callable[[str], Any],
    db_url: str,
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    """No preload, so the first discharge outruns its patient and the service says no."""
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        created = runs.create_run(
            conn, population="skew", start_at=START, end_at=END, acceleration=4
        )
    first_discharge = replayed[_discharge_index(replayed) - 1]

    with pytest.raises(RuntimeError, match="refused"):
        _replay(serve, db_url, events)

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert clock.instant(run.sim_now) < first_discharge.at
    assert run.cursor is None or run.cursor < first_discharge.sort_key
    assert run.status == "running"
