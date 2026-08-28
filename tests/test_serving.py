"""Serving-time cohort admission and feature computation over event state.

These tests use hand-built ``PatientHistory`` frames rather than a
database: ``state.patient_history`` read-back equality is already pinned
in ``tests/test_state_postgres.py``, so the seam under test here is
``serving_features`` alone.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
    payload_frame,
)
from risk_scoring import serving, state

CHF_CODE = "88805009"


def _history(
    *,
    patients: list[dict[str, str]] | None = None,
    encounters: list[dict[str, str]] | None = None,
    medications: list[dict[str, str]] | None = None,
    conditions: list[dict[str, str]] | None = None,
) -> state.PatientHistory:
    return state.PatientHistory(
        patients=payload_frame(patients or [], state.PATIENT_COLUMNS),
        encounters=payload_frame(encounters or [], state.ENCOUNTER_COLUMNS),
        medications=payload_frame(medications or [], state.MEDICATION_COLUMNS),
        conditions=payload_frame(conditions or [], state.CONDITION_COLUMNS),
    )


def _worked_example() -> state.PatientHistory:
    """One patient whose expected feature row is worked out by hand below."""
    return _history(
        patients=[make_patient_row(BIRTHDATE="1960-01-01")],
        encounters=[
            make_encounter_row(
                Id="encounter-prior-inpatient",
                ENCOUNTERCLASS="inpatient",
                START="2023-11-01T08:00:00Z",
                STOP="2023-11-05T08:00:00Z",
            ),
            make_encounter_row(
                Id="encounter-prior-ed",
                ENCOUNTERCLASS="emergency",
                START="2023-12-01T10:00:00Z",
                STOP="2023-12-01T14:00:00Z",
            ),
            make_encounter_row(
                Id="encounter-index",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-03T08:00:00Z",
            ),
        ],
        medications=[
            make_medication_row(ENCOUNTER="encounter-index", START="2023-12-20T08:00:00Z", STOP="")
        ],
        conditions=[
            make_condition_row(ENCOUNTER="encounter-prior-inpatient", START="2023-06-01", STOP=""),
            make_condition_row(
                ENCOUNTER="encounter-prior-inpatient",
                START="2023-07-01",
                STOP="",
                CODE=CHF_CODE,
                DESCRIPTION="Chronic congestive heart failure (disorder)",
            ),
        ],
    )


def test_admitted_discharge_carries_the_hand_worked_feature_values() -> None:
    """Expected values are derived from the feature definitions, not the code.

    Born 1960-01-01 and discharged 2024-01-03, so age is 64 and the stay
    is 2.0 days. The 180-day window opens 2023-07-07, admitting the
    November inpatient stay and the December emergency visit. The gap runs
    from that stay's 2023-11-05T08:00 discharge to the 2024-01-01T08:00
    admission: 57 days. The open medication and both open disorders are
    active at discharge, and the heart-failure code sets one flag.
    """
    result = serving.serving_features(_worked_example(), "encounter-index")

    assert result is not None
    assert result.features.to_dict("records") == [
        {
            "encounter_id": "encounter-index",
            "patient_id": "patient-1",
            "age_at_discharge": 64,
            "los_days": 2.0,
            "prior_inpatient_180d": 1,
            "days_since_prev_discharge": 57.0,
            "prior_ed_180d": 1,
            "active_medication_count": 1,
            "active_disorder_count": 2,
            "flag_chf": 1,
            "flag_chronic_pulmonary": 0,
            "flag_dementia": 0,
            "flag_diabetes": 0,
            "flag_malignancy": 0,
            "flag_mi": 0,
            "flag_renal_disease": 0,
        }
    ]
    assert isinstance(result.features, pd.DataFrame)


def test_cohort_rules_decide_admission_at_serving() -> None:
    """Each exclusion is the cohort module's, reached through the same call."""
    assert serving.serving_features(_worked_example(), "encounter-prior-ed") is None


def test_in_hospital_death_is_not_scored() -> None:
    history = _history(
        patients=[make_patient_row(BIRTHDATE="1960-01-01", DEATHDATE="2024-01-03")],
        encounters=[make_encounter_row(ENCOUNTERCLASS="inpatient")],
    )

    assert serving.serving_features(history, "encounter-1") is None


def test_patient_under_18_at_discharge_is_not_scored() -> None:
    history = _history(
        patients=[make_patient_row(BIRTHDATE="2010-01-01")],
        encounters=[make_encounter_row(ENCOUNTERCLASS="inpatient")],
    )

    assert serving.serving_features(history, "encounter-1") is None


def test_open_encounter_is_not_a_scoring_event() -> None:
    history = _history(
        patients=[make_patient_row()],
        encounters=[make_encounter_row(ENCOUNTERCLASS="inpatient", STOP="")],
    )

    assert serving.serving_features(history, "encounter-1") is None


def test_encounter_absent_from_state_raises() -> None:
    with pytest.raises(serving.UnknownEncounterError):
        serving.serving_features(_worked_example(), "encounter-never-ingested")


def test_inpatient_discharge_without_demographics_raises() -> None:
    """Demographics arriving after the discharge is an ordering violation, not an exclusion."""
    history = _history(encounters=[make_encounter_row(ENCOUNTERCLASS="inpatient")])

    with pytest.raises(serving.UnknownPatientError, match="patient-1"):
        serving.serving_features(history, "encounter-1")


def test_open_encounter_without_demographics_is_not_a_scoring_event() -> None:
    """A stay still open is not scored, so its missing demographics are not yet a problem."""
    history = _history(encounters=[make_encounter_row(ENCOUNTERCLASS="inpatient", STOP="")])

    assert serving.serving_features(history, "encounter-1") is None


def test_features_ignore_events_recorded_after_the_scored_discharge() -> None:
    """State may already hold later events; none of them may reach the row."""
    history = _worked_example()
    later = payload_frame(
        [
            make_encounter_row(
                Id="encounter-readmission",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-13T08:00:00Z",
                STOP="2024-01-16T08:00:00Z",
            )
        ],
        state.ENCOUNTER_COLUMNS,
    )
    history = state.PatientHistory(
        patients=history.patients,
        encounters=pd.concat([history.encounters, later], ignore_index=True),
        medications=payload_frame(
            [make_medication_row(START="2024-01-13T08:00:00Z", STOP="")],
            state.MEDICATION_COLUMNS,
        ),
        conditions=payload_frame(
            [make_condition_row(START="2024-01-13", STOP="")], state.CONDITION_COLUMNS
        ),
    )

    result = serving.serving_features(history, "encounter-index")

    assert result is not None
    row = result.features.iloc[0]
    assert row["prior_inpatient_180d"] == 1
    assert row["active_medication_count"] == 0
    assert row["active_disorder_count"] == 0
