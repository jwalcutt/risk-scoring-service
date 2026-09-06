"""The label schedule: what the harness will release, and when.

Pure: no database, no clock. The rules these tests pin:

- A label falls due exactly 30 days after its discharge, and its value is
  the batch label for that discharge, so what the harness releases is
  what training would have learned from.
- Only cohort discharges are scheduled, in due order, with ties broken
  by encounter id so the order is total.
- The run's share of the schedule is the discharges inside its span,
  inclusive at both ends: the preload leaves earlier ones unscored, and
  later ones are never posted.
- What is due in a tick is what fell due after the last checkpoint and
  at or before the tick's instant.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factories import make_encounter_row, make_patient_row
from risk_scoring.cohort import build_cohort
from risk_scoring.labels import build_labels
from risk_scoring.replay import release
from risk_scoring.replay.release import ScheduledLabel


def _stay(
    encounter_id: str, patient: str, start: str, stop: str, kind: str = "inpatient"
) -> dict[str, str]:
    return make_encounter_row(
        Id=encounter_id, PATIENT=patient, ENCOUNTERCLASS=kind, START=start, STOP=stop
    )


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Two adults, a minor, a readmission, an emergency visit, and a tied due instant."""
    patients = [
        make_patient_row(Id="p-a", BIRTHDATE="1960-01-01"),
        make_patient_row(Id="p-b", BIRTHDATE="1970-01-01"),
        make_patient_row(Id="p-minor", BIRTHDATE="2015-01-01"),
    ]
    encounters = [
        _stay("e-index", "p-a", "2025-01-01T08:00:00Z", "2025-01-05T08:00:00Z"),
        # Readmitted 15 days later; that stay is itself a cohort discharge.
        _stay("e-readmit", "p-a", "2025-01-20T08:00:00Z", "2025-01-22T08:00:00Z"),
        # Discharged at the same instant as e-index: the same due instant.
        # Its next stay starts 36 days on, past the readmission window.
        _stay("e-tied", "p-b", "2025-01-03T00:00:00Z", "2025-01-05T08:00:00Z"),
        _stay("e-late", "p-b", "2025-02-10T00:00:00Z", "2025-02-10T12:34:56Z"),
        _stay("e-ed", "p-b", "2025-02-10T00:00:00Z", "2025-02-10T03:00:00Z", "emergency"),
        _stay("e-minor", "p-minor", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
    ]
    return {"patients": pd.DataFrame(patients), "encounters": pd.DataFrame(encounters)}


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return release.label_schedule(frames)


def test_every_cohort_discharge_is_scheduled_once_and_nothing_else_is(
    schedule: list[ScheduledLabel],
) -> None:
    assert sorted(item.encounter_id for item in schedule) == [
        "e-index",
        "e-late",
        "e-readmit",
        "e-tied",
    ]


def test_a_label_falls_due_thirty_days_after_its_discharge(
    schedule: list[ScheduledLabel],
) -> None:
    by_id = {item.encounter_id: item for item in schedule}
    assert by_id["e-index"].discharged_at == "2025-01-05T08:00:00Z"
    assert by_id["e-index"].due_at == "2025-02-04T08:00:00Z"
    assert by_id["e-late"].due_at == "2025-03-12T12:34:56Z"


def test_the_scheduled_label_is_the_batch_label(
    frames: dict[str, pd.DataFrame], schedule: list[ScheduledLabel]
) -> None:
    """The harness holds ground truth and withholds only the release."""
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    batch = build_labels(cohort, frames["encounters"]).set_index("encounter_id")["label"]
    assert {item.encounter_id: item.label for item in schedule} == batch.to_dict()
    assert batch.to_dict() == {"e-index": 1, "e-readmit": 0, "e-tied": 0, "e-late": 0}


def test_the_schedule_is_in_due_order_with_ties_broken_by_encounter_id(
    schedule: list[ScheduledLabel],
) -> None:
    assert [item.encounter_id for item in schedule] == ["e-index", "e-tied", "e-readmit", "e-late"]


def test_the_runs_share_is_inclusive_at_both_ends(schedule: list[ScheduledLabel]) -> None:
    """Earlier discharges are the preload's unscored ones; later ones are never posted."""
    within = release.scheduled_within(schedule, "2025-01-05T08:00:00Z", "2025-02-10T12:34:56Z")
    assert [item.encounter_id for item in within] == ["e-index", "e-tied", "e-readmit", "e-late"]
    narrower = release.scheduled_within(schedule, "2025-01-05T08:00:01Z", "2025-02-10T12:34:55Z")
    assert [item.encounter_id for item in narrower] == ["e-readmit"]


def test_due_labels_excludes_the_checkpoint_instant_and_includes_the_ticks(
    schedule: list[ScheduledLabel],
) -> None:
    due = "2025-02-04T08:00:00Z"
    assert release.due_labels(schedule, "2025-02-04T07:00:00Z", due) == schedule[:2]
    assert release.due_labels(schedule, due, "2025-02-04T09:00:00Z") == []


def test_due_labels_is_empty_on_a_quiet_tick(schedule: list[ScheduledLabel]) -> None:
    assert release.due_labels(schedule, "2025-01-10T00:00:00Z", "2025-01-10T01:00:00Z") == []


def test_due_labels_spans_a_burst(schedule: list[ScheduledLabel]) -> None:
    """A tick that covers weeks releases everything that fell due in them, in order."""
    assert (
        release.due_labels(schedule, "2025-01-01T00:00:00Z", "2025-03-01T00:00:00Z")
        == (schedule[:3])
    )
