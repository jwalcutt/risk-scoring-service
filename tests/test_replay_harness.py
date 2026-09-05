"""The tick loop, driven with no database and no real clock.

The property under test is that what the loop posts is a function of the
stream and simulated time only. Tick size, pacing, and the wall clock may
change how long a run takes and when its checkpoints are written, never
what reaches the service or in what order. The rules these tests pin:

- One-hour ticks and seven-day ticks post the identical sequence.
- A wall clock that jumps forward a week mid-run posts the identical
  sequence; the loop catches up in a burst with no waiting.
- Pacing is visible only as time asked of ``sleep``.
- Each tick's checkpoint is written after that tick's posts, and a
  refusal from the service leaves the checkpoint where the last complete
  tick put it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from factories import make_condition_row, make_encounter_row, make_medication_row
from risk_scoring.replay import clock, harness
from risk_scoring.replay.runs import StreamCursor
from risk_scoring.stream import StreamEvent, envelope

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 15, tzinfo=UTC)
MAX_SPEED = clock.Pacing(acceleration=4, max_speed=True)
PACED = clock.Pacing(acceleration=4)


def _encounter(identity: str, stop: str) -> StreamEvent:
    row = make_encounter_row(Id=identity, PATIENT="p1", ENCOUNTERCLASS="inpatient", STOP=stop)
    return StreamEvent(at=stop, kind="encounter", row=row)


def _medication(code: str, start: str) -> StreamEvent:
    row = make_medication_row(PATIENT="p1", ENCOUNTER="e-any", CODE=code, START=start)
    return StreamEvent(at=start, kind="medication", row=row)


def _condition(code: str, day: str) -> StreamEvent:
    row = make_condition_row(PATIENT="p1", ENCOUNTER="e-any", CODE=code, START=day)
    return StreamEvent(at=f"{day}T00:00:00Z", kind="condition", row=row)


@pytest.fixture(scope="module")
def events() -> list[StreamEvent]:
    """Two weeks with instants on and off the hour, ties, and a quiet stretch."""
    unsorted = [
        _condition("1", "2025-01-01"),  # exactly at the start instant
        _medication("10", "2025-01-01T00:00:00Z"),  # ties with the condition above
        _encounter("e-1", "2025-01-01T00:00:00Z"),  # and with this discharge
        _medication("11", "2025-01-01T00:30:00Z"),
        _encounter("e-2", "2025-01-01T01:00:00Z"),  # exactly on a tick boundary
        _encounter("e-3", "2025-01-01T01:00:01Z"),  # one second past it
        _condition("2", "2025-01-03"),
        _medication("12", "2025-01-03T13:45:10Z"),
        _encounter("e-4", "2025-01-03T13:45:10Z"),
        # nothing for ten days
        _encounter("e-5", "2025-01-14T23:59:59Z"),
        _medication("13", "2025-01-15T00:00:00Z"),  # exactly at the end instant
    ]
    return sorted(unsorted, key=lambda event: event.sort_key)


@dataclass
class RecordingPoster:
    """Accepts everything; a discharge counts as scored."""

    posted: list[Mapping[str, Any]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        self.posted.append(event)
        self.calls.append(f"post:{event['payload'].get('Id') or event['payload']['CODE']}")
        return {"scored": event["event_type"] == "encounter"}


@dataclass
class RefusingPoster(RecordingPoster):
    """Refuses the nth post, as the service does with a 4xx."""

    refuse_at: int = 3

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if len(self.posted) + 1 == self.refuse_at:
            raise RuntimeError("encounter refused: 409")
        return super().post_event(event)


@dataclass
class FakeWall:
    """A wall clock that only moves when the loop sleeps, or when a test jumps it."""

    now: float = 1_000.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class Checkpoints:
    written: list[tuple[datetime, StreamCursor | None]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
        self.written.append((sim_now, cursor))
        self.calls.append(f"checkpoint:{sim_now.isoformat()}")


def _drive(
    events: list[StreamEvent],
    *,
    poster: RecordingPoster | None = None,
    checkpoint: Checkpoints | None = None,
    pacing: clock.Pacing = MAX_SPEED,
    wall: FakeWall | None = None,
    tick: timedelta = clock.TICK,
    sim_now: datetime = START,
    cursor: StreamCursor | None = None,
) -> tuple[harness.RunSummary, RecordingPoster, Checkpoints, FakeWall]:
    poster = RecordingPoster() if poster is None else poster
    checkpoint = Checkpoints() if checkpoint is None else checkpoint
    wall = FakeWall() if wall is None else wall
    summary = harness.drive(
        events,
        sim_now=sim_now,
        end_at=END,
        cursor=cursor,
        poster=poster,
        checkpoint=checkpoint,
        pacing=pacing,
        wall_clock=wall,
        sleep=wall.sleep,
        tick=tick,
    )
    return summary, poster, checkpoint, wall


# What is posted


def test_every_event_is_posted_exactly_once_in_stream_order(events: list[StreamEvent]) -> None:
    _, poster, _, _ = _drive(events)
    assert poster.posted == [envelope(event.kind, event.row) for event in events]


def test_one_hour_and_seven_day_ticks_post_the_identical_sequence(
    events: list[StreamEvent],
) -> None:
    _, hourly, _, _ = _drive(events, tick=timedelta(hours=1))
    _, weekly, _, _ = _drive(events, tick=timedelta(days=7))
    assert weekly.posted == hourly.posted
    assert len(hourly.posted) == len(events)


def test_a_wall_clock_that_jumps_a_week_posts_the_identical_sequence(
    events: list[StreamEvent],
) -> None:
    """The laptop slept: the loop wakes far behind and catches up in a burst."""
    _, reference, _, _ = _drive(events)

    wall = FakeWall()
    checkpoints = Checkpoints()

    class JumpingCheckpoints(Checkpoints):
        def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
            super().__call__(sim_now, cursor)
            if len(self.written) == 5:
                wall.now += 7 * 24 * 3600

    checkpoints = JumpingCheckpoints()
    _, jumped, _, _ = _drive(events, pacing=PACED, wall=wall, checkpoint=checkpoints)

    assert jumped.posted == reference.posted
    # Five paced ticks, then nothing but catching up.
    assert len(wall.sleeps) == 5


def test_the_clock_lands_on_the_end_and_the_cursor_on_the_last_event(
    events: list[StreamEvent],
) -> None:
    summary, _, checkpoints, _ = _drive(events)
    assert checkpoints.written[-1] == (END, events[-1].sort_key)
    assert summary.sim_to == END


def test_resuming_from_a_checkpoint_posts_only_what_follows_it(
    events: list[StreamEvent],
) -> None:
    """Both the cursor and sim_now come from the checkpoint; neither alone would do."""
    cursor_event = events[5]
    resumed_from = datetime(2025, 1, 1, 2, tzinfo=UTC)
    _, poster, _, _ = _drive(events, sim_now=resumed_from, cursor=cursor_event.sort_key)
    assert poster.posted == [envelope(event.kind, event.row) for event in events[6:]]


def test_a_run_already_at_its_end_does_nothing(events: list[StreamEvent]) -> None:
    summary, poster, checkpoints, _ = _drive(events, sim_now=END, cursor=events[-1].sort_key)
    assert poster.posted == []
    assert checkpoints.written == []
    assert summary.ticks == 0


# Checkpoints


def test_each_checkpoint_is_written_after_that_ticks_posts(events: list[StreamEvent]) -> None:
    calls: list[str] = []

    class SharedPoster(RecordingPoster):
        def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
            ack = super().post_event(event)
            calls.append(self.calls[-1])
            return ack

    class SharedCheckpoints(Checkpoints):
        def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
            super().__call__(sim_now, cursor)
            calls.append(self.calls[-1])

    _drive(events, poster=SharedPoster(), checkpoint=SharedCheckpoints(), tick=timedelta(days=7))
    first_week_end = (START + timedelta(days=7)).isoformat()
    assert calls[:10] == [
        "post:10",
        "post:1",
        "post:e-1",
        "post:11",
        "post:e-2",
        "post:e-3",
        "post:2",
        "post:12",
        "post:e-4",
        f"checkpoint:{first_week_end}",
    ]


def test_a_quiet_tick_still_checkpoints(events: list[StreamEvent]) -> None:
    """sim_now must advance through a stretch with nothing due."""
    _, _, checkpoints, _ = _drive(events, tick=timedelta(days=1))
    written = dict(checkpoints.written)
    quiet_day = datetime(2025, 1, 8, tzinfo=UTC)
    assert quiet_day in written
    assert written[quiet_day] == events[8].sort_key


def test_a_refusal_propagates_and_leaves_the_checkpoint_before_the_tick(
    events: list[StreamEvent],
) -> None:
    """The service said no; the run stops and the next resume re-posts the tick."""
    poster = RefusingPoster(refuse_at=8)
    checkpoints = Checkpoints()
    with pytest.raises(RuntimeError, match="refused"):
        _drive(events, poster=poster, checkpoint=checkpoints)

    # Seven posts landed: five in the first hour, one in the second, and
    # the condition at midnight on the third. The eighth is the first of
    # two events in the tick ending 14:00 that day; that tick was never
    # checkpointed, so the checkpoint still names the tick before it.
    assert len(poster.posted) == 7
    assert checkpoints.written[-1] == (datetime(2025, 1, 3, 13, tzinfo=UTC), events[6].sort_key)


# Pacing


def test_max_speed_never_sleeps(events: list[StreamEvent]) -> None:
    _, _, _, wall = _drive(events)
    assert wall.sleeps == []


def test_pacing_asks_for_exactly_the_wall_seconds_per_tick(events: list[StreamEvent]) -> None:
    """Pacing is observable, but only as time asked of sleep."""
    summary, poster, _, wall = _drive(events, pacing=PACED)
    assert len(wall.sleeps) == summary.ticks
    assert sum(wall.sleeps) == pytest.approx(summary.ticks * PACED.wall_seconds_per_tick())
    _, at_max_speed, _, _ = _drive(events)
    assert poster.posted == at_max_speed.posted


# The summary


def test_the_summary_counts_what_the_invocation_did(events: list[StreamEvent]) -> None:
    summary, _, _, _ = _drive(events, tick=timedelta(days=7))
    assert summary.sim_from == START
    assert summary.sim_to == END
    assert summary.end_at == END
    assert summary.events_posted == {"condition": 2, "medication": 4, "encounter": 5}
    assert summary.discharges_scored == 5
    assert summary.ticks == 2
    assert summary.finished


def test_the_summary_measures_wall_time_and_the_largest_gap_between_ticks(
    events: list[StreamEvent],
) -> None:
    wall = FakeWall()

    class SlowThirdTick(Checkpoints):
        def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
            super().__call__(sim_now, cursor)
            if len(self.written) == 3:
                wall.now += 90.0

    summary, _, _, _ = _drive(events, pacing=PACED, wall=wall, checkpoint=SlowThirdTick())
    assert summary.largest_tick_gap_seconds == pytest.approx(90.0 + PACED.wall_seconds_per_tick())
    assert summary.wall_seconds == pytest.approx(wall.now - 1_000.0)


def test_a_partial_invocation_reports_where_it_stopped(events: list[StreamEvent]) -> None:
    resumed_from = datetime(2025, 1, 10, tzinfo=UTC)
    summary, _, _, _ = _drive(events, sim_now=resumed_from, cursor=events[8].sort_key)
    assert summary.sim_from == resumed_from
    assert summary.events_posted == {"encounter": 1, "medication": 1}
    assert summary.finished


def test_the_report_names_the_span_and_the_counts(events: list[StreamEvent]) -> None:
    summary, _, _, _ = _drive(events, tick=timedelta(days=7))
    text = harness.report(summary)
    assert "2025-01-01T00:00:00Z to 2025-01-15T00:00:00Z" in text
    assert "5 encounter" in text
    assert "5 discharges scored" in text
    assert "2 ticks" in text
