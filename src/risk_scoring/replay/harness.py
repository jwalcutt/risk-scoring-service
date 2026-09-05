"""The tick loop: the stream posted to the service on the simulated clock.

Each tick advances ``sim_now`` by one step, posts every event due at or
before it in stream order, then checkpoints the clock and the cursor.
What gets posted is a function of the stream and simulated time only;
tick size, pacing, and the wall clock decide when a tick runs and never
what it posts. The tests in ``tests/test_replay_harness.py`` hold that
line, and it is what makes a paused-and-resumed run byte-identical to an
uninterrupted one.

Judgment calls this module fixes:

- Pacing is a schedule anchored at the moment the loop starts, not a
  fixed sleep per tick. Tick ``n`` is due ``n`` tick-lengths of wall time
  after the anchor; the loop sleeps only while it is ahead of that
  schedule. After the machine sleeps, ``time.time`` is far ahead of the
  schedule and the loop runs ticks back to back until it catches up,
  which is the burst the wall-clock decision expects.
- The checkpoint is written after the tick's posts, never before. A
  refusal from the service propagates unchanged, and because that tick
  was never checkpointed the resumed run re-posts it from the last
  checkpoint; the service answers what it has already stored as a no-op.
  A 4xx is a defect in the stream, not something to count and skip past.
- Only clinical events are posted. The preload loads every patient row,
  so demographics are already in state before the first tick; a splice
  has to keep that true for the population it brings in.
- A pause request is read once per iteration, after the pacing sleep and
  before the tick's posts. Nothing is posted once a pause is observed, a
  pause written while the loop sleeps costs at most that one tick, and
  the checkpoint the pause leaves behind is the last complete tick's.
  Whoever asked for the pause (an operator's Ctrl-C, the run row's
  status, a monitoring alert one day) is not this module's concern; the
  request is a value the loop takes.
- Wall time spent paused never advances simulated time, because the loop
  starts from the ``sim_now`` it is given and anchors its pacing schedule
  when it starts. A resumed run continues from the checkpoint whatever
  the wall clock did meanwhile.
- The loop takes its poster, its checkpoint, its clock, and its sleep as
  values, so the tick-size and jumping-clock tests need neither Postgres
  nor a real clock. :func:`run_replay` is the one place the loop is bound
  to a run row; it reads the row's status as the pause request and never
  writes it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import psycopg

from risk_scoring.replay import clock, runs
from risk_scoring.replay.emission import due_events
from risk_scoring.replay.preload import replay_from
from risk_scoring.replay.runs import ReplayRun, StreamCursor
from risk_scoring.stream import StreamEvent, envelope


class Poster(Protocol):
    """What the loop needs from a running service: one event in, its acknowledgement out."""

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]: ...


Checkpoint = Callable[[datetime, StreamCursor | None], None]
"""Record the clock and the last posted event after a tick's posts complete."""

Sleep = Callable[[float], None]

PauseRequested = Callable[[datetime], bool]
"""Asked with the current ``sim_now`` before each tick's posts; true stops the loop."""


def never(sim_now: datetime) -> bool:
    return False


@dataclass(frozen=True)
class RunSummary:
    """What one invocation of the loop did, for the operator's record."""

    sim_from: datetime
    sim_to: datetime
    end_at: datetime
    events_posted: dict[str, int]
    discharges_scored: int
    ticks: int
    wall_seconds: float
    largest_tick_gap_seconds: float
    paused: bool = False

    @property
    def finished(self) -> bool:
        return self.sim_to == self.end_at


def drive(
    events: Sequence[StreamEvent],
    *,
    sim_now: datetime,
    end_at: datetime,
    cursor: StreamCursor | None,
    poster: Poster,
    checkpoint: Checkpoint,
    pacing: clock.Pacing,
    wall_clock: clock.WallClock = clock.DEFAULT_WALL_CLOCK,
    sleep: Sleep = time.sleep,
    tick: timedelta = clock.TICK,
    pause_requested: PauseRequested = never,
) -> RunSummary:
    """Tick from ``sim_now`` to ``end_at``, posting what falls due along the way.

    ``events`` is the stream to post, in stream order, already restricted
    to what the replay owns. ``cursor`` is the sort key of the last event
    posted, or ``None`` at a fresh start. The loop returns early, with the
    summary marked paused, when ``pause_requested`` answers true.
    """
    sim_from = sim_now
    per_tick = pacing.wall_seconds_per_tick()
    wall_anchor = wall_clock()
    last_tick_end = wall_anchor
    ticks = 0
    largest_gap = 0.0
    events_posted: dict[str, int] = {}
    discharges_scored = 0
    paused = False

    while sim_now < end_at:
        due_wall = wall_anchor + (ticks + 1) * per_tick
        wait = due_wall - wall_clock()
        if wait > 0:
            sleep(wait)
        if pause_requested(sim_now):
            paused = True
            break

        target = clock.next_tick(sim_now, end_at, tick=tick)
        for event in due_events(events, cursor, clock.instant(target)):
            ack = poster.post_event(envelope(event.kind, event.row))
            cursor = event.sort_key
            events_posted[event.kind] = events_posted.get(event.kind, 0) + 1
            if ack.get("scored"):
                discharges_scored += 1
        sim_now = target
        checkpoint(sim_now, cursor)

        ticks += 1
        now = wall_clock()
        largest_gap = max(largest_gap, now - last_tick_end)
        last_tick_end = now

    return RunSummary(
        sim_from=sim_from,
        sim_to=sim_now,
        end_at=end_at,
        events_posted=events_posted,
        discharges_scored=discharges_scored,
        ticks=ticks,
        wall_seconds=wall_clock() - wall_anchor,
        largest_tick_gap_seconds=largest_gap,
        paused=paused,
    )


def run_replay(
    conn: psycopg.Connection[Any],
    run: ReplayRun,
    events: Sequence[StreamEvent],
    poster: Poster,
    *,
    pacing: clock.Pacing,
    wall_clock: clock.WallClock = clock.DEFAULT_WALL_CLOCK,
    sleep: Sleep = time.sleep,
    pause_requested: PauseRequested = never,
) -> RunSummary:
    """Drive a run from where its row says it stands, checkpointing to that row.

    ``events`` is the population's whole ordered stream; only the events
    dated at or after the run's start are posted, the preload having put
    the rest into state. The run pauses when its row's status reads
    ``paused``, whoever wrote it, or when ``pause_requested`` says so; the
    status is read here, never written.
    """

    def checkpoint(sim_now: datetime, cursor: StreamCursor | None) -> None:
        runs.checkpoint(conn, run.run_id, sim_now=sim_now, cursor=cursor)

    def pause(sim_now: datetime) -> bool:
        return pause_requested(sim_now) or runs.read_status(conn, run.run_id) == "paused"

    return drive(
        replay_from(events, clock.instant(run.start_at)),
        sim_now=run.sim_now,
        end_at=run.end_at,
        cursor=run.cursor,
        poster=poster,
        checkpoint=checkpoint,
        pacing=pacing,
        wall_clock=wall_clock,
        sleep=sleep,
        pause_requested=pause,
    )


def report(summary: RunSummary) -> str:
    """The invocation as plain text, simulated span first."""
    posted = (
        ", ".join(f"{count} {kind}" for kind, count in sorted(summary.events_posted.items()))
        or "none"
    )
    # The loop returns only at the end or on a pause; a refusal raises.
    standing = "finished" if summary.finished else f"paused at {_instant(summary.sim_to)}"
    span = f"{_instant(summary.sim_from)} to {_instant(summary.sim_to)}"
    return "\n".join(
        [
            f"simulated span: {span} ({standing})",
            f"events posted: {posted} ({sum(summary.events_posted.values())} total)",
            f"{summary.discharges_scored} discharges scored",
            f"{summary.ticks} ticks in {summary.wall_seconds:.1f} wall seconds, "
            f"largest gap between ticks {summary.largest_tick_gap_seconds:.1f} s",
        ]
    )


def _instant(moment: datetime) -> str:
    return clock.instant(moment)
