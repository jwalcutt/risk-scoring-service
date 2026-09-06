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
- A pause request is honored before a tick's posts, and wall time spent
  paused never advances simulated time: the resumed loop continues from
  the checkpoint's ``sim_now`` whatever the wall clock did meanwhile.
- A label is released in the first tick whose ``sim_now`` is at or past
  its due instant, never before, with the tick's ``sim_now`` as its
  release instant; within a tick, labels and events are processed in
  simulated order with a label first at an equal instant. Tick size and
  the wall clock change neither which labels a tick releases relative to
  its posts nor the release instants.
- A label due after the invocation stopped is pending, and a resumed
  loop re-releases the labels of the tick that was never checkpointed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring.replay import clock, harness, release
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.replay.runs import StreamCursor
from risk_scoring.stream import StreamEvent, envelope

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 15, tzinfo=UTC)
MAX_SPEED = clock.Pacing(acceleration=4, max_speed=True)
PACED = clock.Pacing(acceleration=4)


def _label(encounter_id: str, due_at: str, label: int = 1) -> ScheduledLabel:
    """A label with a made-up discharge instant: only the due instant matters to the loop."""
    return ScheduledLabel(
        due_at=due_at, encounter_id=encounter_id, discharged_at="2024-12-01T00:00:00Z", label=label
    )


LABELS = sorted(
    [
        _label("l-tick", "2025-01-01T01:00:00Z"),  # a tick boundary shared with e-2
        _label("l-early", "2025-01-03T13:10:00Z", 0),  # before the two events in its tick
        _label("l-late", "2025-01-03T13:50:00Z"),  # after them, same tick
        _label("l-quiet", "2025-01-10T00:00:00Z"),  # a tick with no events at all
        _label("l-after", "2025-01-20T00:00:00Z"),  # past the end: pending forever
    ]
)


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
class RecordingReleaser:
    """Accepts every label once; a repeat is the table's no-op."""

    released: list[tuple[str, datetime, int]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def __call__(self, label: ScheduledLabel, released_at: datetime) -> bool:
        self.calls.append(f"release:{label.encounter_id}")
        if any(name == label.encounter_id for name, _, _ in self.released):
            return False
        self.released.append((label.encounter_id, released_at, label.label))
        return True


class KilledAfterRelease(RuntimeError):
    """The harness died after writing a label and before the tick's checkpoint."""


@dataclass
class DyingReleaser(RecordingReleaser):
    """Writes the named label, then dies as a process would mid-tick."""

    die_after: str = ""

    def __call__(self, label: ScheduledLabel, released_at: datetime) -> bool:
        written = super().__call__(label, released_at)
        if label.encounter_id == self.die_after:
            raise KilledAfterRelease(label.encounter_id)
        return written


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
    end_at: datetime = END,
    cursor: StreamCursor | None = None,
    pause_requested: Callable[[datetime], bool] | None = None,
    labels: list[ScheduledLabel] | None = None,
    release: RecordingReleaser | None = None,
) -> tuple[harness.RunSummary, RecordingPoster, Checkpoints, FakeWall]:
    poster = RecordingPoster() if poster is None else poster
    checkpoint = Checkpoints() if checkpoint is None else checkpoint
    wall = FakeWall() if wall is None else wall
    extra: dict[str, Any] = {}
    if pause_requested is not None:
        extra["pause_requested"] = pause_requested
    summary = harness.drive(
        events,
        sim_now=sim_now,
        end_at=end_at,
        cursor=cursor,
        poster=poster,
        checkpoint=checkpoint,
        pacing=pacing,
        wall_clock=wall,
        sleep=wall.sleep,
        tick=tick,
        labels=[] if labels is None else labels,
        release=RecordingReleaser() if release is None else release,
        **extra,
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


# Pausing


def test_a_pause_request_stops_the_loop_before_that_ticks_posts(
    events: list[StreamEvent],
) -> None:
    """Nothing is posted once a pause is observed; the last checkpoint is the tick before."""
    pause_at = datetime(2025, 1, 3, 13, tzinfo=UTC)  # the tick ending 14:00 holds two events
    summary, poster, checkpoints, _ = _drive(
        events, pause_requested=lambda sim_now: sim_now >= pause_at
    )

    assert summary.paused
    assert not summary.finished
    assert summary.sim_to == pause_at
    assert checkpoints.written[-1] == (pause_at, events[6].sort_key)
    assert poster.posted == [envelope(event.kind, event.row) for event in events[:7]]


def test_a_pause_request_is_read_after_the_pacing_sleep(events: list[StreamEvent]) -> None:
    """A pause written while the loop sleeps stops it without another tick's posts."""
    wall = FakeWall()
    asked: list[datetime] = []

    def pause_requested(sim_now: datetime) -> bool:
        asked.append(sim_now)
        return len(wall.sleeps) == 2

    summary, poster, checkpoints, _ = _drive(
        events, pacing=PACED, wall=wall, pause_requested=pause_requested
    )
    assert len(wall.sleeps) == 2
    assert summary.ticks == 1
    assert asked == [START, START + clock.TICK]
    assert checkpoints.written == [(START + clock.TICK, events[4].sort_key)]
    assert poster.posted == [envelope(event.kind, event.row) for event in events[:5]]


def test_wall_time_spent_paused_never_advances_simulated_time(
    events: list[StreamEvent],
) -> None:
    """Pause, let a wall day pass, resume from the checkpoint: the clock restarts from it."""
    _, reference, _, _ = _drive(events)
    pause_at = datetime(2025, 1, 2, 6, tzinfo=UTC)
    wall = FakeWall()

    _, before, paused_at, _ = _drive(
        events, pacing=PACED, wall=wall, pause_requested=lambda sim_now: sim_now >= pause_at
    )
    sim_now, cursor = paused_at.written[-1]
    assert sim_now == pause_at

    wall.now += 24 * 3600
    second, after, resumed, _ = _drive(
        events, pacing=PACED, wall=wall, sim_now=sim_now, cursor=cursor
    )

    assert resumed.written[0][0] == pause_at + clock.TICK
    assert second.sim_from == pause_at
    assert second.finished
    assert before.posted + after.posted == reference.posted
    # The paced sleeps resume at the full tick length: the anchor is reset,
    # so the wall day off is not treated as a burst to catch up on.
    assert resumed.written[0][0] - pause_at == clock.TICK
    assert wall.sleeps[-1] == pytest.approx(PACED.wall_seconds_per_tick())


def test_a_paused_summary_is_not_finished_and_a_finished_one_is_not_paused(
    events: list[StreamEvent],
) -> None:
    finished, _, _, _ = _drive(events)
    assert finished.finished and not finished.paused
    paused, _, _, _ = _drive(events, pause_requested=lambda sim_now: sim_now > START)
    assert paused.paused and not paused.finished


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
    assert "finished" in text


def test_the_report_names_a_pause(events: list[StreamEvent]) -> None:
    pause_at = datetime(2025, 1, 2, 6, tzinfo=UTC)
    summary, _, _, _ = _drive(events, pause_requested=lambda sim_now: sim_now >= pause_at)
    text = harness.report(summary)
    assert "paused at 2025-01-02T06:00:00Z" in text
    assert "stopped short" not in text


# Labels


def test_a_label_is_released_in_the_first_tick_at_or_past_its_due_instant(
    events: list[StreamEvent],
) -> None:
    """Never early: on the due instant when it is a tick boundary, else at the tick's end."""
    releaser = RecordingReleaser()
    summary, _, _, _ = _drive(events, labels=LABELS, release=releaser)
    assert releaser.released == [
        ("l-tick", datetime(2025, 1, 1, 1, tzinfo=UTC), 1),
        ("l-early", datetime(2025, 1, 3, 14, tzinfo=UTC), 0),
        ("l-late", datetime(2025, 1, 3, 14, tzinfo=UTC), 1),
        ("l-quiet", datetime(2025, 1, 10, tzinfo=UTC), 1),
    ]
    assert summary.labels_released == 4
    assert summary.labels_pending == 1


def test_within_a_tick_labels_and_events_are_processed_in_simulated_order(
    events: list[StreamEvent],
) -> None:
    """A label goes first at an equal instant; otherwise the instants decide."""
    calls: list[str] = []

    class SharedPoster(RecordingPoster):
        def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
            ack = super().post_event(event)
            calls.append(self.calls[-1])
            return ack

    class SharedReleaser(RecordingReleaser):
        def __call__(self, label: ScheduledLabel, released_at: datetime) -> bool:
            written = super().__call__(label, released_at)
            calls.append(self.calls[-1])
            return written

    class SharedCheckpoints(Checkpoints):
        def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
            super().__call__(sim_now, cursor)
            calls.append("checkpoint")

    _drive(
        events,
        poster=SharedPoster(),
        checkpoint=SharedCheckpoints(),
        labels=LABELS,
        release=SharedReleaser(),
    )
    assert calls[:7] == [
        "post:10",
        "post:1",
        "post:e-1",
        "post:11",
        "release:l-tick",  # due at 01:00, the same instant as e-2
        "post:e-2",
        "checkpoint",
    ]
    assert calls[calls.index("release:l-early") : calls.index("release:l-late") + 2] == [
        "release:l-early",  # due 13:10, before both events in the tick ending 14:00
        "post:12",
        "post:e-4",
        "release:l-late",  # due 13:50, after them
        "checkpoint",
    ]


def test_tick_size_changes_neither_the_labels_nor_their_order_relative_to_posts(
    events: list[StreamEvent],
) -> None:
    hourly = RecordingReleaser()
    weekly = RecordingReleaser()
    _drive(events, labels=LABELS, release=hourly, tick=timedelta(hours=1))
    _drive(events, labels=LABELS, release=weekly, tick=timedelta(days=7))
    assert [name for name, _, _ in weekly.released] == [name for name, _, _ in hourly.released]
    # The release instant is the tick's own: a coarser tick releases later, never earlier.
    assert all(
        coarse >= fine and coarse >= datetime.fromisoformat(LABELS[index].due_at)
        for index, ((_, fine, _), (_, coarse, _)) in enumerate(
            zip(hourly.released, weekly.released, strict=True)
        )
    )


def test_a_wall_clock_that_jumps_releases_the_identical_labels_at_the_identical_instants(
    events: list[StreamEvent],
) -> None:
    """A burst after the machine wakes still stamps each label with its own tick's sim_now."""
    reference = RecordingReleaser()
    _drive(events, labels=LABELS, release=reference)

    wall = FakeWall()

    class JumpingCheckpoints(Checkpoints):
        def __call__(self, sim_now: datetime, cursor: StreamCursor | None) -> None:
            super().__call__(sim_now, cursor)
            if len(self.written) == 5:
                wall.now += 7 * 24 * 3600

    jumped = RecordingReleaser()
    _drive(
        events,
        labels=LABELS,
        release=jumped,
        pacing=PACED,
        wall=wall,
        checkpoint=JumpingCheckpoints(),
    )
    assert jumped.released == reference.released


def test_a_readmission_on_day_29_has_no_label_on_day_29_and_one_on_day_30() -> None:
    """The exit criterion's own words, on the loop with a schedule built from the export."""
    discharged = "2025-01-05T08:00:00Z"
    frames = {
        "patients": pd.DataFrame([make_patient_row(Id="p-1", BIRTHDATE="1960-01-01")]),
        "encounters": pd.DataFrame(
            [
                make_encounter_row(
                    Id="e-index",
                    PATIENT="p-1",
                    ENCOUNTERCLASS="inpatient",
                    START="2025-01-01T08:00:00Z",
                    STOP=discharged,
                ),
                make_encounter_row(
                    Id="e-readmit",
                    PATIENT="p-1",
                    ENCOUNTERCLASS="inpatient",
                    START="2025-02-03T08:00:00Z",  # day 29
                    STOP="2025-02-06T08:00:00Z",
                ),
            ]
        ),
    }
    schedule = release.label_schedule(frames)
    stream = [
        StreamEvent(at=row["STOP"], kind="encounter", row=dict(row))
        for _, row in frames["encounters"].iterrows()
    ]
    day_29 = datetime(2025, 2, 3, 8, tzinfo=UTC)
    day_30 = datetime(2025, 2, 4, 8, tzinfo=UTC)
    end = datetime(2025, 2, 10, tzinfo=UTC)

    first = RecordingReleaser()
    paused, _, checkpoints, _ = _drive(
        stream,
        end_at=end,
        labels=schedule,
        release=first,
        pause_requested=lambda sim_now: sim_now >= day_29,
    )
    assert paused.sim_to == day_29
    assert first.released == []
    assert paused.labels_pending == 2

    second = RecordingReleaser()
    sim_now, cursor = checkpoints.written[-1]
    finished, _, _, _ = _drive(
        stream, sim_now=sim_now, end_at=end, cursor=cursor, labels=schedule, release=second
    )
    assert finished.finished
    assert second.released == [("e-index", day_30, 1)]
    assert finished.labels_pending == 1  # the readmission stay's own label, due in March


def test_resuming_re_releases_the_labels_of_the_tick_that_was_never_checkpointed(
    events: list[StreamEvent],
) -> None:
    """Died after writing l-early: the resume releases it again and the table drops the repeat."""
    dying = DyingReleaser(die_after="l-early")
    checkpoints = Checkpoints()
    with pytest.raises(KilledAfterRelease):
        _drive(events, labels=LABELS, release=dying, checkpoint=checkpoints)
    assert [name for name, _, _ in dying.released] == ["l-tick", "l-early"]
    sim_now, cursor = checkpoints.written[-1]
    assert sim_now == datetime(2025, 1, 3, 13, tzinfo=UTC)

    resumed = RecordingReleaser()
    summary, _, _, _ = _drive(
        events, sim_now=sim_now, cursor=cursor, labels=LABELS, release=resumed
    )
    assert [name for name, _, _ in resumed.released] == ["l-early", "l-late", "l-quiet"]
    assert summary.labels_released == 3


def test_a_repeat_release_is_not_counted(events: list[StreamEvent]) -> None:
    """The table's no-op answer keeps the count honest, as an unscored ack does for discharges."""
    releaser = RecordingReleaser()
    releaser.released.append(("l-tick", START, 1))
    summary, _, _, _ = _drive(events, labels=LABELS, release=releaser)
    assert summary.labels_released == 3
    assert "release:l-tick" in releaser.calls


def test_the_report_names_the_label_counts(events: list[StreamEvent]) -> None:
    summary, _, _, _ = _drive(events, labels=LABELS, release=RecordingReleaser())
    assert "4 labels released, 1 pending" in harness.report(summary)


def test_a_run_with_no_labels_reports_none(events: list[StreamEvent]) -> None:
    summary, _, _, _ = _drive(events)
    assert (summary.labels_released, summary.labels_pending) == (0, 0)
    assert "0 labels released, 0 pending" in harness.report(summary)
