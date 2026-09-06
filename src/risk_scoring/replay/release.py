"""The label schedule: every cohort discharge's label, and the instant it falls due.

Pure: no database, no clock, no HTTP. The harness is the generator layer
and holds ground truth from the start; what it withholds is the release.
This module computes the whole schedule from the population export up
front, so that releasing a label is a lookup and never a computation over
state.

Judgment calls this module fixes:

- Labels come from the shared label module over the export, never from
  service state. Encounters reach the service at their STOP, so a
  readmission stay still open on day 30 has not reached state yet, and a
  label derived from state would disagree with the training label for
  the same discharge. The schedule's label is the batch label by
  construction.
- A label falls due exactly the readmission window after the discharge
  instant. Instants are the stream's own timestamp strings, so "due at
  or before ``sim_now``" is a string comparison, as it is for events.
- The order is total: by due instant, then encounter id. Two labels due
  at one instant are released in the same order in every run.
- A run's share of the schedule is the discharges inside its span,
  inclusive at both ends. Earlier discharges the preload left unscored
  and later ones are never posted, so neither can have a label; counting
  them as pending would misdescribe the run.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from risk_scoring.cohort import build_cohort
from risk_scoring.labels import READMISSION_WINDOW_DAYS, build_labels
from risk_scoring.replay import clock

LABEL_DELAY = timedelta(days=READMISSION_WINDOW_DAYS)


@dataclass(frozen=True, order=True)
class ScheduledLabel:
    """One discharge's label and the simulated instant it may be released."""

    due_at: str
    encounter_id: str
    discharged_at: str
    label: int


def label_schedule(frames: Mapping[str, pd.DataFrame]) -> list[ScheduledLabel]:
    """Every cohort discharge in the export, labelled, in due order."""
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    labels = build_labels(cohort, frames["encounters"])
    scheduled = [
        ScheduledLabel(
            due_at=clock.instant(stop + LABEL_DELAY),
            encounter_id=str(encounter_id),
            discharged_at=clock.instant(stop),
            label=int(label),
        )
        for encounter_id, stop, label in zip(
            cohort["encounter_id"], cohort["stop"], labels["label"], strict=True
        )
    ]
    return sorted(scheduled)


def scheduled_within(
    schedule: Sequence[ScheduledLabel], start: str, end: str
) -> list[ScheduledLabel]:
    """The labels of discharges dated inside ``[start, end]``, in due order."""
    return [item for item in schedule if start <= item.discharged_at <= end]


def due_labels(schedule: Sequence[ScheduledLabel], after: str, at: str) -> list[ScheduledLabel]:
    """The labels that fell due after ``after`` and at or before ``at``, in due order.

    ``schedule`` must be in due order, as :func:`label_schedule` returns
    it. ``after`` is the last checkpoint's instant: everything due at or
    before it was released by the tick that ended there.
    """
    first = bisect_right(schedule, after, key=lambda item: item.due_at)
    last = bisect_right(schedule, at, key=lambda item: item.due_at)
    return list(schedule[first:last])
