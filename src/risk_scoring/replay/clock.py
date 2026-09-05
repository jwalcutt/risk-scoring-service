"""Clock arithmetic for the replay tick.

The harness advances simulated time in discrete steps. At each step it
processes everything due at or before the new ``sim_now`` in one merged
simulated-time order: events by their arrival instant, label releases by
their due instant. When a label's due instant equals an event's arrival
instant, the label goes first: what was already determined lands before
what is new at that instant, the same reason medications and conditions
precede a discharge at one instant in the stream order.

Judgment calls this module fixes:

- One tick is one simulated hour. At four simulated days per wall minute
  that is 0.625 wall seconds, short enough that a pause request is
  honored within a second and long enough that the process is not
  spinning. Tick size must not change what a run writes; the tick loop's
  tests assert that, which is why the tick is a constant here and not a
  configuration value.
- The wall clock is ``time.time``. ``time.monotonic`` stops while the
  machine sleeps, so a sleeping laptop would freeze simulated time
  silently; ``time.time`` keeps counting, so on waking the harness finds
  simulated time far ahead and posts everything due in one burst. The
  burst is the stream outage the operating record expects to happen on
  its own, and it leaves a visible gap in ``scored_at`` while the
  simulated record stays correct. The clock is a value the loop takes,
  so a test can substitute one that jumps.
- A simulated instant is formatted with the stream's own timestamp
  format, so "due at or before ``sim_now``" is a string comparison
  against an event's ``at`` and no parsing happens in the loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from risk_scoring import state

TICK = timedelta(hours=1)

WallClock = Callable[[], float]
"""Seconds since the epoch, as ``time.time`` reports them."""

DEFAULT_WALL_CLOCK: WallClock = time.time

_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class Pacing:
    """How long the loop waits between ticks, in wall time.

    ``acceleration`` is simulated days per wall minute. ``max_speed``
    never waits, for tests and for the byte-identity checks, which care
    about what a run writes and not about how long it takes.
    """

    acceleration: float
    max_speed: bool = False

    def __post_init__(self) -> None:
        if self.acceleration <= 0:
            raise ValueError(f"acceleration must be positive; got {self.acceleration!r}")

    def wall_seconds_per_tick(self) -> float:
        if self.max_speed:
            return 0.0
        simulated_seconds_per_wall_second = self.acceleration * _SECONDS_PER_DAY / 60
        return TICK.total_seconds() / simulated_seconds_per_wall_second


def day_start(day: date) -> datetime:
    """Midnight UTC on a calendar date, as an aware datetime."""
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def instant(moment: datetime) -> str:
    """A simulated instant in the stream's timestamp format.

    Requires an aware datetime on a whole second: the stream carries whole
    seconds, so a truncated instant could misplace an event relative to
    ``sim_now``. The zone is normalized to UTC first, since the database
    driver hands timestamps back in the session zone.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"instant needs an aware datetime; got {moment!r}")
    if moment.microsecond != 0:
        raise ValueError(f"instant needs a whole second; got {moment!r}")
    return moment.astimezone(UTC).strftime(state.TIMESTAMP_FORMAT)


def next_tick(sim_now: datetime, end_at: datetime) -> datetime:
    """The simulated instant one tick on, clamped so the clock lands on the end."""
    return min(sim_now + TICK, end_at)
