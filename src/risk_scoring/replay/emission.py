"""What the tick loop owes the service at one simulated instant.

Pure: no database, no clock, no HTTP. Given the ordered stream, the sort
key of the last event posted, and the tick's simulated instant, this is
the list of events to post now, in stream order.

Judgment calls this module fixes:

- The boundary is inclusive: an event at exactly ``sim_now`` is due. The
  instant is the stream's own timestamp string, so "at or before" is a
  string comparison and no parsing happens per tick.
- The cursor is a sort key, not an index. The resume point is found by
  bisection over the stream's keys, so a stream rebuilt from the export,
  or spliced so that different rows precede the cursor, resumes at the
  same next event. Bisection also keeps a year of hourly ticks from
  scanning the stream from the top each time.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from risk_scoring.replay.runs import StreamCursor
from risk_scoring.stream import StreamEvent


def due_events(
    events: Sequence[StreamEvent], cursor: StreamCursor | None, sim_now: str
) -> list[StreamEvent]:
    """The events after ``cursor`` dated at or before ``sim_now``, in stream order.

    ``events`` must be in stream order, as :func:`stream.ordered_events`
    returns them. A cursor of ``None`` means nothing has been posted yet.
    """
    index = 0 if cursor is None else bisect_right(events, cursor, key=lambda e: e.sort_key)
    due = []
    while index < len(events) and events[index].at <= sim_now:
        due.append(events[index])
        index += 1
    return due
