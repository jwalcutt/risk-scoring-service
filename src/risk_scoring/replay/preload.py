"""History from before a replay's start, loaded into state without scoring.

A discharge in the first replayed month reads events from the months
before it: the 180-day counts and days-since-previous look back across the
start. State must therefore hold every event dated before the start
before the first replayed discharge arrives. Posting that history through
the service would score every pre-start discharge, roughly a decade of
in-sample ones, and at the measured posting rate take over an hour for the
full baseline. This module writes it straight into the state tables
instead, in batches, so nothing before the start is ever scored and the
replay proper posts only events dated at or after it.

Judgment calls this module fixes:

- Pre-start means the same thing on both sides of the partition. An event
  is history when its arrival instant, the one the stream posts it at, is
  strictly before the start instant; the replay posts everything at or
  after. The two sides are exact complements of one list.
- Every patient row is loaded, whatever its dates. Demographics lead the
  stream because a discharge that outran its patient is refused, and a
  patient whose first event is after the start still needs a birthdate
  by then.
- Rows are loaded one kind at a time. Final state is arrival-order
  independent by design, so nothing is lost by not interleaving, and a
  batch of one kind reports its new rows under one kind.
- The count of discharges left unscored comes from the shared cohort
  module and the cutoff split training uses, so it is the training
  cohort's own count by construction, never a second reading of the rule.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg

from risk_scoring import state
from risk_scoring.cohort import build_cohort, split_at_cutoff
from risk_scoring.stream import StreamEvent

DEFAULT_BATCH_SIZE = 5000

_EVENT_TYPES: dict[str, type[state.AnyEvent]] = {
    "encounter": state.EncounterEvent,
    "medication": state.MedicationEvent,
    "condition": state.ConditionEvent,
}


@dataclass(frozen=True)
class PreloadSummary:
    """What a preload put into state, and what it deliberately did not score."""

    before: str
    rows_loaded: dict[str, int]
    rows_already_present: int
    discharges_unscored: int


def history_before(events: Sequence[StreamEvent], before: str) -> list[StreamEvent]:
    """The events dated strictly before an instant, in stream order."""
    return [event for event in events if event.at < before]


def preload_history(
    conn: psycopg.Connection[Any],
    frames: Mapping[str, pd.DataFrame],
    events: Sequence[StreamEvent],
    before: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PreloadSummary:
    """Load every patient and every event dated before ``before`` into state.

    ``events`` is the population's ordered stream, passed in rather than
    rebuilt because the caller needs the same list for the replay itself.
    Loading is idempotent: a load that died partway is resumed by calling
    this again, and rows already present are counted, not rewritten.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")

    by_kind: dict[str, list[state.AnyEvent]] = {
        "patient": [
            state.PatientEvent.from_row(dict(row)) for _, row in frames["patients"].iterrows()
        ]
    }
    history = history_before(events, before)
    for kind, event_type in _EVENT_TYPES.items():
        by_kind[kind] = [event_type.from_row(event.row) for event in history if event.kind == kind]

    rows_loaded: dict[str, int] = {}
    rows_already_present = 0
    for kind, typed in by_kind.items():
        rows_loaded[kind] = 0
        for batch in _batches(typed, batch_size):
            inserted = state.record_batch(conn, batch)
            rows_loaded[kind] += inserted
            rows_already_present += len(batch) - inserted

    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    unscored = split_at_cutoff(cohort, pd.Timestamp(before)).before
    return PreloadSummary(
        before=before,
        rows_loaded=rows_loaded,
        rows_already_present=rows_already_present,
        discharges_unscored=len(unscored),
    )


def _batches(items: Sequence[state.AnyEvent], size: int) -> Iterator[Sequence[state.AnyEvent]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
