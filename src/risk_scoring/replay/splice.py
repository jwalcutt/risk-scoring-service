"""Stream splicing: which population is the source of the stream, and from when.

Pure: no database, no clock, no HTTP. A replay's configuration names a
starting population and a list of splices, each a simulated date and the
population that takes over from it. This module turns that into segments,
cuts each population's stream and label schedule down to the segment that
owns it, and joins the pieces into the one stream and the one schedule the
tick loop runs over. The loop itself knows nothing about splices.

Judgment calls this module fixes:

- A segment runs from its instant to the next splice's, half-open: an
  event at exactly the splice instant belongs to the incoming population,
  and the outgoing population's event at that instant is dropped. The
  preload boundary for a segment is its own instant, so "history before
  the segment" and "the segment's events" are exact complements of one
  population's stream, the same partition the start uses.
- Labels follow the population of origin. A segment owns the labels of the
  discharges it owns, judged by the discharge instant, so a discharge
  posted from the outgoing population keeps that population's label even
  when its readmission stay falls after the splice and is never posted.
- Only a population other than the run's own is rekeyed. A splice back to
  the starting population is the same people, whose history is already in
  state under their own ids. A variant is rekeyed the same way in every
  segment that names it.
- The spliced schedule is re-sorted into due order. Each segment's share
  is already in due order and the segments are disjoint in time, so this
  is a no-op today; the sort guards the invariant the release bisects on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from risk_scoring.replay import clock
from risk_scoring.replay.config import ReplayConfig
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent


@dataclass(frozen=True)
class Segment:
    """One stretch of the replay and the population that sources it.

    ``from_at`` and ``until`` are simulated instants in the stream's
    timestamp format; ``until`` is ``None`` for the last segment.
    ``rekey`` says whether the population's ids are rewritten at load.
    """

    population: str
    from_at: str
    until: str | None
    rekey: bool

    def owns(self, at: str) -> bool:
        return self.from_at <= at and (self.until is None or at < self.until)


def segments(config: ReplayConfig) -> list[Segment]:
    """The run's segments in order: the start population, then one per splice."""
    names = [config.population, *(splice.population for splice in config.splices)]
    instants = [
        clock.instant(clock.day_start(config.start)),
        *(clock.instant(clock.day_start(splice.at)) for splice in config.splices),
    ]
    following = [*instants[1:], None]
    return [
        Segment(name, from_at, until, rekey=name != config.population)
        for name, from_at, until in zip(names, instants, following, strict=True)
    ]


def segment_events(segment: Segment, events: Sequence[StreamEvent]) -> list[StreamEvent]:
    """The events of one population's stream that the segment owns, in stream order."""
    return [event for event in events if segment.owns(event.at)]


def segment_labels(segment: Segment, schedule: Sequence[ScheduledLabel]) -> list[ScheduledLabel]:
    """The labels of one population's schedule whose discharges the segment owns."""
    return [item for item in schedule if segment.owns(item.discharged_at)]


def spliced_events(parts: Sequence[Sequence[StreamEvent]]) -> list[StreamEvent]:
    """The segments' events joined into one stream, in segment order."""
    return [event for part in parts for event in part]


def spliced_labels(parts: Sequence[Sequence[ScheduledLabel]]) -> list[ScheduledLabel]:
    """The segments' labels joined into one schedule, in due order."""
    return sorted(item for part in parts for item in part)
