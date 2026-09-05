"""A population as one timestamp-ordered stream of ingestion events.

The service takes events one at a time, in timestamp order, and decides
what a discharge can see from what has already arrived. This module turns
four Synthea frames into exactly that stream: what arrives, when, and in
what shape.

Judgment calls it fixes:

- Each row arrives at the instant its own meaning implies. An encounter is
  a discharge notification, so it arrives at its STOP. Medications arrive
  at their START. Condition dates are date-only, so a condition arrives at
  midnight of its start date, which puts a condition recorded on a
  discharge date in front of that day's discharge, matching how the
  feature module judges condition activity against the date rather than
  the instant.
- Demographics lead the stream, unordered. The cohort rules need a
  birthdate, so a discharge that outran its patient is refused rather than
  scored.
- The tie-break is the whole row, so the order is total and independent of
  the order the frames happen to hold. Final state is arrival-order
  independent by design, but a deterministic stream keeps prediction logs
  comparable run to run.
- Payload field sets come from ``state``, which already owns them, rather
  than being restated here. A restated copy could drift from what the
  service stores with nothing to catch it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from risk_scoring import state

# Medications and conditions starting at the instant a stay ends were in
# effect during it, so they reach the service before that discharge.
_KIND_ORDER = {"medication": 0, "condition": 1, "encounter": 2}

EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "patient": state.PATIENT_COLUMNS,
    "encounter": state.ENCOUNTER_COLUMNS,
    "medication": state.MEDICATION_COLUMNS,
    "condition": state.CONDITION_COLUMNS,
}


@dataclass(frozen=True)
class StreamEvent:
    """One ingestion event and the simulated instant it arrives at."""

    at: str
    kind: str
    row: dict[str, str]

    @property
    def sort_key(self) -> tuple[str, int, str]:
        """Total order over the stream, with no tie left to input order."""
        return (self.at, _KIND_ORDER[self.kind], repr(sorted(self.row.items())))


def ordered_events(
    encounters: pd.DataFrame, medications: pd.DataFrame, conditions: pd.DataFrame
) -> list[StreamEvent]:
    """Interleave every clinical row into one timestamp-ordered stream."""
    events = [
        StreamEvent(at=row["STOP"], kind="encounter", row=dict(row))
        for _, row in encounters.iterrows()
    ]
    events += [
        StreamEvent(at=row["START"], kind="medication", row=dict(row))
        for _, row in medications.iterrows()
    ]
    events += [
        StreamEvent(at=f"{row['START']}T00:00:00Z", kind="condition", row=dict(row))
        for _, row in conditions.iterrows()
    ]
    return sorted(events, key=lambda event: event.sort_key)


def envelope(kind: str, row: Mapping[str, str]) -> dict[str, Any]:
    """One row as the event the service is posted, projected to the contract fields.

    The batch driver and the replay harness both post through this, so a
    row reaches the service as the same bytes whichever of them sends it.
    """
    return {"event_type": kind, "payload": {name: row[name] for name in EVENT_FIELDS[kind]}}


def build_stream(frames: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """The whole population as posted event envelopes, demographics first."""
    stream = [envelope("patient", dict(row)) for _, row in frames["patients"].iterrows()]
    stream += [
        envelope(item.kind, item.row)
        for item in ordered_events(
            frames["encounters"], frames["medications"], frames["conditions"]
        )
    ]
    return stream
