"""Tests for narrowing a population down to a runnable set of patients.

Two properties carry the weight. Eligibility decides which patients are
worth posting at all, and an optional cutoff narrows that to patients the
model has never been trained through. Narrowing itself must keep every
row a selected patient owns: a patient's features read their whole
history, so a selection that dropped their older rows would make serving
features disagree with the training pipeline for a reason that is not a
defect in either.
"""

from __future__ import annotations

import pandas as pd

from factories import (
    CONDITION_DEFAULTS,
    ENCOUNTER_DEFAULTS,
    MEDICATION_DEFAULTS,
    PATIENT_DEFAULTS,
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring.sampling import eligible_patients, sample_patients, select_patients

CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")


def population(
    patients: list[dict[str, str]],
    encounters: list[dict[str, str]],
    medications: list[dict[str, str]] | None = None,
    conditions: list[dict[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "patients": pd.DataFrame(patients or None, columns=list(PATIENT_DEFAULTS)),
        "encounters": pd.DataFrame(encounters or None, columns=list(ENCOUNTER_DEFAULTS)),
        "medications": pd.DataFrame(medications or None, columns=list(MEDICATION_DEFAULTS)),
        "conditions": pd.DataFrame(conditions or None, columns=list(CONDITION_DEFAULTS)),
    }


def adult(patient_id: str) -> dict[str, str]:
    return make_patient_row(Id=patient_id, BIRTHDATE="1970-01-01")


def stay(encounter_id: str, patient_id: str, stop: str) -> dict[str, str]:
    return make_encounter_row(
        Id=encounter_id, PATIENT=patient_id, ENCOUNTERCLASS="inpatient", START=stop, STOP=stop
    )


# Eligibility


def test_only_patients_with_a_cohort_discharge_are_eligible() -> None:
    frames = population(
        [adult("p1"), adult("p2")],
        [
            stay("e1", "p1", "2024-03-03T00:00:00Z"),
            make_encounter_row(
                Id="e2",
                PATIENT="p2",
                ENCOUNTERCLASS="ambulatory",
                START="2024-03-03T00:00:00Z",
                STOP="2024-03-03T00:00:00Z",
            ),
        ],
    )
    assert list(eligible_patients(frames)) == ["p1"]


def test_a_cutoff_restricts_eligibility_to_a_post_cutoff_discharge() -> None:
    """A discharge at the cutoff instant counts; the model never saw it."""
    frames = population(
        [adult("p-before"), adult("p-at"), adult("p-after")],
        [
            stay("e-before", "p-before", "2024-12-31T23:59:59Z"),
            stay("e-at", "p-at", "2025-01-01T00:00:00Z"),
            stay("e-after", "p-after", "2025-06-06T00:00:00Z"),
        ],
    )
    assert set(eligible_patients(frames)) == {"p-before", "p-at", "p-after"}
    assert set(eligible_patients(frames, discharged_at_or_after=CUTOFF)) == {"p-at", "p-after"}


def test_an_under_18_discharge_does_not_make_its_patient_eligible() -> None:
    """The service would never score it, so posting the patient proves nothing."""
    frames = population(
        [make_patient_row(Id="p-child", BIRTHDATE="2020-01-01")],
        [stay("e1", "p-child", "2025-06-06T00:00:00Z")],
    )
    assert list(eligible_patients(frames, discharged_at_or_after=CUTOFF)) == []


# Sampling


def many_patients(count: int) -> dict[str, pd.DataFrame]:
    return population(
        [adult(f"p{index}") for index in range(count)],
        [stay(f"e{index}", f"p{index}", "2025-06-06T00:00:00Z") for index in range(count)],
    )


def test_the_same_seed_selects_the_same_patients() -> None:
    frames = many_patients(40)
    first = sample_patients(frames, count=10, seed=20260101)
    second = sample_patients(frames, count=10, seed=20260101)
    assert list(first["patients"]["Id"]) == list(second["patients"]["Id"])
    assert len(first["patients"]) == 10


def test_a_different_seed_selects_a_different_set() -> None:
    frames = many_patients(40)
    first = set(sample_patients(frames, count=10, seed=1)["patients"]["Id"])
    second = set(sample_patients(frames, count=10, seed=2)["patients"]["Id"])
    assert first != second


def test_a_count_above_the_pool_returns_the_whole_pool() -> None:
    frames = many_patients(5)
    assert len(sample_patients(frames, count=50, seed=20260101)["patients"]) == 5


def test_sampling_with_a_cutoff_draws_only_from_post_cutoff_patients() -> None:
    frames = population(
        [adult("p-old"), adult("p-new")],
        [
            stay("e-old", "p-old", "2020-01-01T00:00:00Z"),
            stay("e-new", "p-new", "2025-06-06T00:00:00Z"),
        ],
    )
    chosen = sample_patients(frames, count=10, seed=20260101, discharged_at_or_after=CUTOFF)
    assert list(chosen["patients"]["Id"]) == ["p-new"]


# Narrowing


def test_a_selected_patients_entire_history_is_kept() -> None:
    """Features read the whole history, so nothing older may be dropped."""
    frames = population(
        [adult("p1"), adult("p2")],
        [
            stay("e-old", "p1", "2019-04-04T00:00:00Z"),
            stay("e-held-out", "p1", "2025-06-06T00:00:00Z"),
            stay("e-other", "p2", "2020-07-07T00:00:00Z"),
        ],
        [make_medication_row(PATIENT="p1", ENCOUNTER="e-old", START="2020-02-02T00:00:00Z")],
        [make_condition_row(PATIENT="p1", ENCOUNTER="e-old", START="2018-01-01")],
    )
    chosen = sample_patients(frames, count=1, seed=20260101, discharged_at_or_after=CUTOFF)
    assert set(chosen["patients"]["Id"]) == {"p1"}
    assert sorted(chosen["encounters"]["Id"]) == ["e-held-out", "e-old"]
    assert list(chosen["medications"]["START"]) == ["2020-02-02T00:00:00Z"]
    assert list(chosen["conditions"]["START"]) == ["2018-01-01"]


def test_no_other_patients_rows_survive() -> None:
    frames = population(
        [adult("p1"), adult("p2")],
        [stay("e1", "p1", "2025-06-06T00:00:00Z"), stay("e2", "p2", "2025-06-06T00:00:00Z")],
        [make_medication_row(PATIENT="p2", ENCOUNTER="e2")],
        [make_condition_row(PATIENT="p2", ENCOUNTER="e2")],
    )
    chosen = select_patients(frames, {"p1"})
    assert list(chosen["patients"]["Id"]) == ["p1"]
    assert list(chosen["encounters"]["Id"]) == ["e1"]
    assert chosen["medications"].empty
    assert chosen["conditions"].empty


def test_narrowing_keeps_every_frame_and_its_columns() -> None:
    frames = population([adult("p1")], [stay("e1", "p1", "2025-06-06T00:00:00Z")])
    chosen = select_patients(frames, set())
    assert set(chosen) == set(frames)
    for name, frame in chosen.items():
        assert frame.empty
        assert list(frame.columns) == list(frames[name].columns)


def test_narrowing_keeps_the_source_row_order() -> None:
    """A deterministic stream needs a deterministic frame to build from."""
    frames = population(
        [adult("p1")],
        [
            stay("e-late", "p1", "2025-08-08T00:00:00Z"),
            stay("e-early", "p1", "2025-02-02T00:00:00Z"),
        ],
    )
    assert list(select_patients(frames, {"p1"})["encounters"]["Id"]) == ["e-late", "e-early"]
