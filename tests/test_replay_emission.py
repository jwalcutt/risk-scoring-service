"""What the tick loop owes the service at one simulated instant.

Pure: no database, no clock, no HTTP. The rules these tests pin:

- An event is due when its arrival instant is at or before ``sim_now``;
  the boundary is inclusive.
- Events come out in stream order and nothing at or before the cursor is
  ever yielded again.
- The cursor is a sort key rather than an index, so a stream rebuilt with
  different rows before the cursor resumes at the same next event.
"""

from __future__ import annotations

from risk_scoring.replay.emission import due_events
from risk_scoring.stream import StreamEvent


def _event(at: str, identity: str, kind: str = "encounter") -> StreamEvent:
    return StreamEvent(at=at, kind=kind, row={"Id": identity})


NOON = "2025-03-01T12:00:00Z"


def test_an_event_at_exactly_sim_now_is_due() -> None:
    at_noon = _event(NOON, "noon")
    assert due_events([at_noon], None, NOON) == [at_noon]


def test_an_event_one_second_after_sim_now_is_not_due() -> None:
    later = _event("2025-03-01T12:00:01Z", "later")
    assert due_events([later], None, NOON) == []


def test_due_events_keep_stream_order_across_kinds_at_one_instant() -> None:
    medication = _event(NOON, "m", "medication")
    condition = _event(NOON, "c", "condition")
    encounter = _event(NOON, "e", "encounter")
    stream = sorted([encounter, condition, medication], key=lambda event: event.sort_key)
    assert due_events(stream, None, NOON) == [medication, condition, encounter]


def test_nothing_at_or_before_the_cursor_is_yielded_again() -> None:
    first = _event("2025-03-01T08:00:00Z", "first")
    second = _event("2025-03-01T09:00:00Z", "second")
    third = _event("2025-03-01T10:00:00Z", "third")
    stream = [first, second, third]
    assert due_events(stream, second.sort_key, NOON) == [third]


def test_the_event_at_the_cursor_itself_is_excluded() -> None:
    only = _event(NOON, "only")
    assert due_events([only], only.sort_key, NOON) == []


def test_an_empty_tick_yields_nothing() -> None:
    later = _event("2025-03-02T00:00:00Z", "tomorrow")
    assert due_events([later], None, NOON) == []


def test_an_empty_stream_yields_nothing() -> None:
    assert due_events([], None, NOON) == []


def test_a_tick_yields_everything_due_since_the_cursor_and_nothing_beyond() -> None:
    events = [
        _event("2025-03-01T08:00:00Z", "posted-already"),
        _event("2025-03-01T09:00:00Z", "due-a", "medication"),
        _event("2025-03-01T11:59:59Z", "due-b"),
        _event(NOON, "due-c"),
        _event("2025-03-01T12:00:01Z", "not-yet"),
    ]
    assert due_events(events, events[0].sort_key, NOON) == events[1:4]


def test_the_cursor_is_a_sort_key_not_an_index() -> None:
    """A rebuilt or spliced stream resumes at the same event, whatever came before."""
    cursor_event = _event("2025-03-01T09:00:00Z", "cursor")
    next_event = _event("2025-03-01T10:00:00Z", "next")
    original = [_event("2025-03-01T08:00:00Z", "a"), cursor_event, next_event]
    rebuilt = [
        _event("2025-02-01T00:00:00Z", "spliced-in-earlier", "condition"),
        _event("2025-03-01T08:00:00Z", "a"),
        _event("2025-03-01T08:30:00Z", "spliced-in-later", "medication"),
        cursor_event,
        next_event,
    ]
    assert due_events(original, cursor_event.sort_key, NOON) == [next_event]
    assert due_events(rebuilt, cursor_event.sort_key, NOON) == [next_event]


def test_a_cursor_between_two_events_resumes_at_the_later_one() -> None:
    """The cursor's own event need not be in the list, as after a splice."""
    before = _event("2025-03-01T09:00:00Z", "before")
    after = _event("2025-03-01T10:00:00Z", "after")
    removed = _event("2025-03-01T09:30:00Z", "removed")
    assert due_events([before, after], removed.sort_key, NOON) == [after]
