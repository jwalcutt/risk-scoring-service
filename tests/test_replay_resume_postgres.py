"""Byte identity across a pause and across a kill.

The exit criterion this file owns: a paused and resumed replay is
byte-identical in its outputs to an uninterrupted one. Three arms run the
same synthetic stream at max speed against an in-process service, each
over its own throwaway database:

- straight through;
- paused at a chosen simulated instant and resumed from the run row;
- killed after a chosen number of posts, before that tick's checkpoint,
  and restarted from the run row.

The pause points are the same kinds the restart test uses: before the
first event, after it, mid-stream, before and after a discharge, after
the last event, plus the instant a label would be released (a discharge
plus 30 simulated days), so the labels substep can add its table to this
comparison without moving the points.

``prediction_id`` and ``scored_at`` are excluded from every comparison.
The database assigns both, and a bigserial consumes a value even when the
log's conflict clause drops a re-post, so ids gap after a resume by
design and say nothing about whether the two runs agree.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg
import pytest

from replay_support import (
    END,
    MAX_SPEED,
    START,
    ClientPoster,
    KilledMidTick,
    Serve,
    first_discharge_index,
    prepare,
    read_labels,
    read_log,
    schedule_of,
    serving,
    skew_frames,
    stream_of,
    tick_containing,
)
from risk_scoring import train
from risk_scoring.replay import clock, harness, preload, runs
from risk_scoring.replay.release import LABEL_DELAY, ScheduledLabel, scheduled_within
from risk_scoring.stream import StreamEvent, envelope

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("resume-population") / "csv")


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return schedule_of(frames)


@pytest.fixture(scope="module")
def replayed(events: list[StreamEvent]) -> list[StreamEvent]:
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
    die_after: int | None = None,
    die_after_label: str | None = None,
    pause_at: datetime | None = None,
) -> tuple[harness.RunSummary | None, ClientPoster]:
    """One invocation from wherever the run row stands; None if it died."""
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        poster = ClientPoster(client, die_after=die_after)

        def pause_requested(sim_now: datetime) -> bool:
            return pause_at is not None and sim_now >= pause_at

        try:
            summary = harness.run_replay(
                conn,
                run,
                events,
                poster,
                labels=schedule,
                pacing=MAX_SPEED,
                pause_requested=pause_requested,
                release=_dying_release(conn, die_after_label),
            )
        except KilledMidTick:
            return None, poster
        return summary, poster


def _dying_release(conn: psycopg.Connection[Any], after: str | None) -> harness.Release:
    """The real labels-table release, dying after it writes the named label."""
    real = harness.database_release(conn)

    def release(label: ScheduledLabel, released_at: datetime) -> bool:
        written = real(label, released_at)
        if label.encounter_id == after:
            raise KilledMidTick(f"died after releasing {after}")
        return written

    return release


def _expected(replayed: list[StreamEvent]) -> list[dict[str, Any]]:
    return [envelope(event.kind, event.row) for event in replayed]


# Where to pause: a simulated instant, derived from the replayed stream.

Point = Callable[[list[StreamEvent]], datetime]


def _before_first(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[0].at) - clock.TICK


def _after_first(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[0].at)


def _mid_stream(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[len(replayed) // 2].at)


def _before_discharge(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[first_discharge_index(replayed) - 1].at) - clock.TICK


def _after_discharge(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[first_discharge_index(replayed) - 1].at)


def _after_last(replayed: list[StreamEvent]) -> datetime:
    return tick_containing(replayed[-1].at)


def _label_release(replayed: list[StreamEvent]) -> datetime:
    """When the first replayed discharge's label falls due: the tick that releases it is done."""
    discharge = replayed[first_discharge_index(replayed) - 1]
    return tick_containing(discharge.at) + LABEL_DELAY


def _before_label_release(replayed: list[StreamEvent]) -> datetime:
    """One tick earlier: the resumed invocation has to release it."""
    return _label_release(replayed) - clock.TICK


PAUSE_POINTS: list[Any] = [
    pytest.param(_before_first, id="before-the-first-event"),
    pytest.param(_after_first, id="after-the-first-event"),
    pytest.param(_mid_stream, id="mid-stream"),
    pytest.param(_before_discharge, id="before-a-discharge"),
    pytest.param(_after_discharge, id="after-a-discharge"),
    pytest.param(_after_last, id="after-the-last-event"),
    pytest.param(_before_label_release, id="before-a-label-release"),
    pytest.param(_label_release, id="at-a-label-release-instant"),
]

# Where to die: a count of posts made, the last of them left uncheckpointed.

KILL_POINTS: list[Any] = [
    pytest.param(lambda replayed: 0, id="before-the-first-post"),
    pytest.param(lambda replayed: 1, id="after-the-first-post"),
    pytest.param(lambda replayed: len(replayed) // 2, id="mid-stream"),
    pytest.param(lambda replayed: first_discharge_index(replayed) - 1, id="before-a-discharge"),
    pytest.param(first_discharge_index, id="just-after-a-discharge"),
    pytest.param(lambda replayed: len(replayed), id="after-the-last-post"),
]


def test_every_pause_point_lies_inside_the_run(replayed: list[StreamEvent]) -> None:
    """A point outside the span would make its arm a straight run and prove nothing."""
    points = {param.id: param.values[0](replayed) for param in PAUSE_POINTS}
    for name, instant in points.items():
        assert START <= instant < END, f"{name} pauses at {instant}, outside the run"
    assert points["before-the-first-event"] < points["after-the-first-event"]
    assert points["before-a-discharge"] < points["after-a-discharge"]
    assert points["before-a-label-release"] < points["at-a-label-release-instant"]
    assert points["after-the-last-event"] > points["mid-stream"]


@pytest.mark.parametrize("choose_pause", PAUSE_POINTS)
def test_a_paused_and_resumed_replay_logs_what_a_straight_one_logs(
    serve: Serve,
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
    choose_pause: Point,
) -> None:
    straight_dsn = db_url_factory()
    paused_dsn = db_url_factory()
    pause_at = choose_pause(replayed)

    prepare(straight_dsn, frames, events)
    straight, _ = _replay(serve, straight_dsn, events, schedule)

    prepare(paused_dsn, frames, events)
    first, before = _replay(serve, paused_dsn, events, schedule, pause_at=pause_at)
    second, after = _replay(serve, paused_dsn, events, schedule)

    assert straight is not None and straight.finished
    assert first is not None and first.paused and not first.finished
    assert first.sim_to == pause_at
    assert second is not None and second.finished
    assert second.sim_from == pause_at
    assert read_log(paused_dsn) == read_log(straight_dsn)
    assert read_labels(paused_dsn) == read_labels(straight_dsn)
    # A pause lands between ticks, so nothing is re-posted or re-released:
    # the two invocations cover the stream and the schedule exactly once.
    assert before.posted + after.posted == _expected(replayed)
    assert first.labels_released + second.labels_released == 5
    assert len(read_log(straight_dsn)) == 6
    assert len(read_labels(straight_dsn)) == 5


@pytest.mark.parametrize("choose_kill", KILL_POINTS)
def test_a_run_killed_before_its_checkpoint_resumes_to_an_identical_log(
    serve: Serve,
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    replayed: list[StreamEvent],
    choose_kill: Callable[[list[StreamEvent]], int],
) -> None:
    straight_dsn = db_url_factory()
    killed_dsn = db_url_factory()
    die_after = choose_kill(replayed)

    prepare(straight_dsn, frames, events)
    _replay(serve, straight_dsn, events, schedule)

    prepare(killed_dsn, frames, events)
    died, first = _replay(serve, killed_dsn, events, schedule, die_after=die_after)
    summary, second = _replay(serve, killed_dsn, events, schedule)

    assert died is None
    assert summary is not None and summary.finished
    assert read_log(killed_dsn) == read_log(straight_dsn)
    assert read_labels(killed_dsn) == read_labels(straight_dsn)
    # The resume re-posts from the last checkpoint: the killed post, and
    # whatever preceded it in the same tick. The service treats every
    # repeat as nothing new, and the two invocations together cover the
    # stream exactly once apart from that overlap.
    re_posted = [event for event in second.posted if event in first.posted]
    if first.posted:
        assert re_posted and first.posted[-1] in re_posted
        assert first.posted[-len(re_posted) :] == re_posted == second.posted[: len(re_posted)]
        assert all(ack["scored"] is False for ack in second.acks[: len(re_posted)])
        assert first.posted[: -len(re_posted)] + second.posted == _expected(replayed)
    else:
        assert re_posted == []
        assert second.posted == _expected(replayed)


def test_a_run_killed_after_a_label_write_resumes_to_identical_tables(
    serve: Serve,
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    """No post-counted kill lands in a label's tick, so the release itself dies here."""
    straight_dsn = db_url_factory()
    killed_dsn = db_url_factory()
    first_label = scheduled_within(schedule, clock.instant(START), clock.instant(END))[0]

    prepare(straight_dsn, frames, events)
    _replay(serve, straight_dsn, events, schedule)

    created = prepare(killed_dsn, frames, events)
    died, _ = _replay(serve, killed_dsn, events, schedule, die_after_label=first_label.encounter_id)
    assert died is None
    # The label is durable; the tick that wrote it is not checkpointed.
    assert [row["encounter_id"] for row in read_labels(killed_dsn)] == [first_label.encounter_id]
    with psycopg.connect(killed_dsn, connect_timeout=2) as conn:
        run = runs.read_run(conn, created.run_id)
    assert clock.instant(run.sim_now) < first_label.due_at

    summary, _ = _replay(serve, killed_dsn, events, schedule)
    assert summary is not None and summary.finished
    assert read_labels(killed_dsn) == read_labels(straight_dsn)
    assert read_log(killed_dsn) == read_log(straight_dsn)
    # The re-release was a no-op and is not counted.
    assert summary.labels_released == 4
