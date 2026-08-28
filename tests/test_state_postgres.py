"""Integration tests for event ingestion and per-patient state read-back.

The contract under test: recording events is idempotent under identical
re-posts, loud under divergent ones, and ``patient_history`` returns
frames whose values are byte-identical to the source rows — the property
serving-time feature recompute rides on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import psycopg
import pytest

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring import state

pytestmark = pytest.mark.db

RowFactory = Callable[..., dict[str, str]]

EVENT_TYPES: list[tuple[str, RowFactory, str, str]] = [
    ("patient", make_patient_row, "PatientEvent", "record_patient"),
    ("encounter", make_encounter_row, "EncounterEvent", "record_encounter"),
    ("medication", make_medication_row, "MedicationEvent", "record_medication"),
    ("condition", make_condition_row, "ConditionEvent", "record_condition"),
]

DIVERGENT_OVERRIDES: dict[str, dict[str, str]] = {
    "patient": {"DEATHDATE": "2024-06-01"},
    "encounter": {"ENCOUNTERCLASS": "inpatient"},
    "medication": {"STOP": "2024-02-01T08:00:00Z"},
    "condition": {"DESCRIPTION": "Something else (disorder)"},
}


def _record(conn: psycopg.Connection[Any], label: str, row: Mapping[str, str]) -> bool:
    _, _, event_name, record_name = next(t for t in EVENT_TYPES if t[0] == label)
    event = getattr(state, event_name).from_row(row)
    result: bool = getattr(state, record_name)(conn, event)
    return result


def _frame(rows: list[Mapping[str, str]], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{name: row[name] for name in columns} for row in rows], columns=columns)


def test_first_event_for_patient_creates_history(db_conn: psycopg.Connection[Any]) -> None:
    patient_row = make_patient_row()
    encounter_row = make_encounter_row()
    assert _record(db_conn, "patient", patient_row) is True
    assert _record(db_conn, "encounter", encounter_row) is True

    history = state.patient_history(db_conn, "patient-1")

    pd.testing.assert_frame_equal(history.patients, _frame([patient_row], state.PATIENT_COLUMNS))
    pd.testing.assert_frame_equal(
        history.encounters, _frame([encounter_row], state.ENCOUNTER_COLUMNS)
    )
    assert history.medications.empty
    assert history.conditions.empty


def test_state_after_sequence_matches_frames_built_from_rows(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Read-back equality: the property serving-time recompute depends on."""
    patient_row = make_patient_row()
    ed_row = make_encounter_row(
        Id="encounter-ed",
        ENCOUNTERCLASS="emergency",
        START="2023-12-01T10:00:00Z",
        STOP="2023-12-01T14:00:00Z",
    )
    inpatient_row = make_encounter_row(
        Id="encounter-inpatient",
        ENCOUNTERCLASS="inpatient",
        START="2024-01-01T08:00:00Z",
        STOP="2024-01-03T08:00:00Z",
    )
    medication_row = make_medication_row(ENCOUNTER="encounter-inpatient")
    condition_row = make_condition_row(ENCOUNTER="encounter-inpatient")

    _record(db_conn, "patient", patient_row)
    _record(db_conn, "encounter", ed_row)
    _record(db_conn, "encounter", inpatient_row)
    _record(db_conn, "medication", medication_row)
    _record(db_conn, "condition", condition_row)

    history = state.patient_history(db_conn, "patient-1")

    pd.testing.assert_frame_equal(history.patients, _frame([patient_row], state.PATIENT_COLUMNS))
    pd.testing.assert_frame_equal(
        history.encounters, _frame([ed_row, inpatient_row], state.ENCOUNTER_COLUMNS)
    )
    pd.testing.assert_frame_equal(
        history.medications, _frame([medication_row], state.MEDICATION_COLUMNS)
    )
    pd.testing.assert_frame_equal(
        history.conditions, _frame([condition_row], state.CONDITION_COLUMNS)
    )


def test_interleaved_patients_keep_separate_histories(
    db_conn: psycopg.Connection[Any],
) -> None:
    _record(db_conn, "patient", make_patient_row(Id="patient-a"))
    _record(db_conn, "patient", make_patient_row(Id="patient-b"))
    a_enc = make_encounter_row(Id="encounter-a", PATIENT="patient-a")
    b_enc = make_encounter_row(Id="encounter-b", PATIENT="patient-b")
    _record(db_conn, "encounter", a_enc)
    _record(db_conn, "medication", make_medication_row(PATIENT="patient-b"))
    _record(db_conn, "encounter", b_enc)
    _record(db_conn, "condition", make_condition_row(PATIENT="patient-a"))

    history_a = state.patient_history(db_conn, "patient-a")
    history_b = state.patient_history(db_conn, "patient-b")

    assert history_a.encounters["Id"].tolist() == ["encounter-a"]
    assert history_b.encounters["Id"].tolist() == ["encounter-b"]
    assert history_a.medications.empty
    assert history_b.medications["PATIENT"].tolist() == ["patient-b"]
    assert history_a.conditions["PATIENT"].tolist() == ["patient-a"]
    assert history_b.conditions.empty


@pytest.mark.parametrize(
    ("label", "make_row"), [(t[0], t[1]) for t in EVENT_TYPES], ids=[t[0] for t in EVENT_TYPES]
)
def test_repost_identical_event_is_noop(
    db_conn: psycopg.Connection[Any], label: str, make_row: RowFactory
) -> None:
    row = make_row()
    assert _record(db_conn, label, row) is True
    assert _record(db_conn, label, row) is False

    history = state.patient_history(db_conn, "patient-1")
    frames = [history.patients, history.encounters, history.medications, history.conditions]
    assert sum(len(frame) for frame in frames) == 1


@pytest.mark.parametrize(
    ("label", "make_row"), [(t[0], t[1]) for t in EVENT_TYPES], ids=[t[0] for t in EVENT_TYPES]
)
def test_repost_divergent_event_raises_conflict(
    db_conn: psycopg.Connection[Any], label: str, make_row: RowFactory
) -> None:
    row = make_row()
    _record(db_conn, label, row)

    divergent = make_row(**DIVERGENT_OVERRIDES[label])
    with pytest.raises(state.EventConflictError):
        _record(db_conn, label, divergent)

    history = state.patient_history(db_conn, "patient-1")
    stored = {
        "patient": history.patients,
        "encounter": history.encounters,
        "medication": history.medications,
        "condition": history.conditions,
    }[label]
    expected_columns = {
        "patient": state.PATIENT_COLUMNS,
        "encounter": state.ENCOUNTER_COLUMNS,
        "medication": state.MEDICATION_COLUMNS,
        "condition": state.CONDITION_COLUMNS,
    }[label]
    pd.testing.assert_frame_equal(stored, _frame([row], expected_columns))


def test_connection_usable_after_conflict(db_conn: psycopg.Connection[Any]) -> None:
    _record(db_conn, "encounter", make_encounter_row())
    with pytest.raises(state.EventConflictError):
        _record(db_conn, "encounter", make_encounter_row(ENCOUNTERCLASS="inpatient"))

    assert _record(db_conn, "encounter", make_encounter_row(Id="encounter-2")) is True


def test_history_ordered_by_start_regardless_of_arrival_order(
    db_conn: psycopg.Connection[Any],
) -> None:
    late = make_encounter_row(Id="encounter-late", START="2024-06-01T08:00:00Z")
    early = make_encounter_row(Id="encounter-early", START="2023-06-01T08:00:00Z")
    _record(db_conn, "encounter", late)
    _record(db_conn, "encounter", early)

    history = state.patient_history(db_conn, "patient-1")
    assert history.encounters["Id"].tolist() == ["encounter-early", "encounter-late"]


def test_unknown_patient_returns_empty_frames_with_columns(
    db_conn: psycopg.Connection[Any],
) -> None:
    history = state.patient_history(db_conn, "nobody")

    assert history.patients.empty
    assert tuple(history.patients.columns) == state.PATIENT_COLUMNS
    assert tuple(history.encounters.columns) == state.ENCOUNTER_COLUMNS
    assert tuple(history.medications.columns) == state.MEDICATION_COLUMNS
    assert tuple(history.conditions.columns) == state.CONDITION_COLUMNS


def test_empty_optional_fields_round_trip_as_empty_strings(
    db_conn: psycopg.Connection[Any],
) -> None:
    _record(db_conn, "patient", make_patient_row(DEATHDATE=""))
    _record(db_conn, "medication", make_medication_row(STOP=""))
    _record(db_conn, "condition", make_condition_row(STOP=""))

    history = state.patient_history(db_conn, "patient-1")

    assert history.patients.loc[0, "DEATHDATE"] == ""
    assert history.medications.loc[0, "STOP"] == ""
    assert history.conditions.loc[0, "STOP"] == ""
