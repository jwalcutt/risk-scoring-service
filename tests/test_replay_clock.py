"""Clock arithmetic for the replay tick.

The rules these tests pin:

- A tick is one simulated hour, and the clock lands exactly on the run's
  end rather than stepping past it.
- Pacing is wall seconds per tick derived from the acceleration, and max
  speed means no waiting at all.
- A simulated instant formats to the same string the stream carries, so
  "due at or before sim_now" is a plain string comparison against the
  event's own arrival instant.
- The wall clock the harness reads is ``time.time``, and it is a value
  the tick loop takes as an argument, never a fixed import.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from risk_scoring import state
from risk_scoring.replay import clock


def test_a_tick_is_one_simulated_hour() -> None:
    assert clock.TICK.total_seconds() == 3600


def test_default_pacing_waits_five_eighths_of_a_second_per_tick() -> None:
    """Four simulated days per wall minute: 96 ticks in 60 seconds."""
    assert clock.Pacing(acceleration=4).wall_seconds_per_tick() == 0.625


def test_doubling_the_acceleration_halves_the_wait() -> None:
    assert clock.Pacing(acceleration=8).wall_seconds_per_tick() == 0.3125


def test_max_speed_never_waits() -> None:
    assert clock.Pacing(acceleration=4, max_speed=True).wall_seconds_per_tick() == 0.0


@pytest.mark.parametrize("acceleration", [0, -4])
def test_pacing_rejects_a_non_positive_acceleration(acceleration: float) -> None:
    with pytest.raises(ValueError, match="acceleration"):
        clock.Pacing(acceleration=acceleration)


def test_day_start_is_midnight_utc() -> None:
    assert clock.day_start(date(2025, 1, 1)) == datetime(2025, 1, 1, tzinfo=UTC)


def test_instant_formats_like_the_stream() -> None:
    assert clock.instant(datetime(2025, 1, 1, tzinfo=UTC)) == "2025-01-01T00:00:00Z"


def test_instant_round_trips_through_the_state_parser() -> None:
    moment = datetime(2025, 3, 4, 13, 7, 9, tzinfo=UTC)
    assert state.parse_timestamp(clock.instant(moment)) == moment


def test_instant_normalizes_another_zone_to_utc() -> None:
    """psycopg hands back timestamps in the session zone; the string must not care."""
    eastern = timezone(timedelta(hours=-5))
    assert clock.instant(datetime(2024, 12, 31, 19, tzinfo=eastern)) == "2025-01-01T00:00:00Z"


def test_instant_refuses_a_naive_datetime() -> None:
    with pytest.raises(ValueError, match="aware"):
        clock.instant(datetime(2025, 1, 1))


def test_instant_refuses_sub_second_precision() -> None:
    """The stream carries whole seconds; a truncated instant would misplace an event."""
    with pytest.raises(ValueError, match="whole second"):
        clock.instant(datetime(2025, 1, 1, 0, 0, 0, 500, tzinfo=UTC))


def test_next_tick_advances_one_hour() -> None:
    now = datetime(2025, 1, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    assert clock.next_tick(now, end) == datetime(2025, 1, 1, 7, tzinfo=UTC)


def test_next_tick_lands_exactly_on_the_end() -> None:
    end = datetime(2025, 7, 1, tzinfo=UTC)
    assert clock.next_tick(end - timedelta(minutes=20), end) == end


def test_next_tick_at_the_end_stays_there() -> None:
    end = datetime(2025, 7, 1, tzinfo=UTC)
    assert clock.next_tick(end, end) == end


def test_the_wall_clock_is_time_dot_time() -> None:
    assert clock.DEFAULT_WALL_CLOCK is time.time
