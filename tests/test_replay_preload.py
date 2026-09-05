"""Which stream events count as history from before a replay's start.

Pure: no database and no clock. The rule is that an event dated strictly
before the start instant is history, an event at or after it belongs to
the replay, and the partition uses the same arrival instant the stream
posts by, so the two sides are exact complements.
"""

from __future__ import annotations

from risk_scoring.replay.preload import history_before, replay_from
from risk_scoring.stream import StreamEvent

BEFORE = "2025-01-01T00:00:00Z"


def _event(at: str, identity: str, kind: str = "encounter") -> StreamEvent:
    return StreamEvent(at=at, kind=kind, row={"Id": identity})


def test_keeps_only_events_dated_before_the_instant() -> None:
    earlier = _event("2024-12-31T23:59:59Z", "earlier")
    later = _event("2025-01-01T00:00:01Z", "later")
    assert history_before([earlier, later], BEFORE) == [earlier]


def test_an_event_at_exactly_the_instant_belongs_to_the_replay() -> None:
    at_start = _event(BEFORE, "at-start")
    assert history_before([at_start], BEFORE) == []


def test_order_is_preserved() -> None:
    events = [
        _event("2024-01-01T00:00:00Z", "a", "medication"),
        _event("2024-01-01T00:00:00Z", "b", "condition"),
        _event("2024-06-01T00:00:00Z", "c"),
        _event("2025-06-01T00:00:00Z", "d"),
    ]
    assert history_before(events, BEFORE) == events[:3]


def test_nothing_in_gives_nothing_out() -> None:
    assert history_before([], BEFORE) == []


def test_a_stream_entirely_after_the_start_has_no_history() -> None:
    assert history_before([_event("2025-03-01T00:00:00Z", "x")], BEFORE) == []


# The replay side of the partition


def test_replay_from_keeps_only_events_dated_at_or_after_the_instant() -> None:
    earlier = _event("2024-12-31T23:59:59Z", "earlier")
    at_start = _event(BEFORE, "at-start")
    later = _event("2025-01-01T00:00:01Z", "later")
    assert replay_from([earlier, at_start, later], BEFORE) == [at_start, later]


def test_the_two_sides_are_exact_complements_of_the_stream() -> None:
    events = [
        _event("2024-01-01T00:00:00Z", "a", "medication"),
        _event("2024-12-31T23:59:59Z", "b"),
        _event(BEFORE, "c", "condition"),
        _event("2025-06-01T00:00:00Z", "d"),
    ]
    assert history_before(events, BEFORE) + replay_from(events, BEFORE) == events


def test_replay_from_an_empty_stream_is_empty() -> None:
    assert replay_from([], BEFORE) == []
