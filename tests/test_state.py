"""Validation and construction tests for the event types in risk_scoring.state.

These are pure tests: no database. The contract under test is that a
constructed event is well-formed by construction (verbatim CSV strings,
format-checked, never converted) and that malformed payloads are rejected
loudly at construction time.
"""

from __future__ import annotations

import pytest

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring import state


def test_patient_event_builds_from_factory_row() -> None:
    event = state.PatientEvent.from_row(make_patient_row())
    assert event.id == "patient-1"
    assert event.birthdate == "1970-01-01"
    assert event.deathdate == ""


def test_encounter_event_builds_from_factory_row() -> None:
    event = state.EncounterEvent.from_row(make_encounter_row())
    assert event.id == "encounter-1"
    assert event.start == "2024-01-01T08:00:00Z"
    assert event.stop == "2024-01-03T08:00:00Z"
    assert event.patient == "patient-1"
    assert event.encounter_class == "wellness"


def test_medication_event_builds_from_factory_row() -> None:
    event = state.MedicationEvent.from_row(make_medication_row())
    assert event.start == "2024-01-01T08:00:00Z"
    assert event.stop == "2024-01-08T08:00:00Z"
    assert event.patient == "patient-1"
    assert event.encounter == "encounter-1"
    assert event.code == "308136"


def test_condition_event_builds_from_factory_row() -> None:
    event = state.ConditionEvent.from_row(make_condition_row())
    assert event.start == "2024-01-01"
    assert event.stop == "2024-01-08"
    assert event.patient == "patient-1"
    assert event.encounter == "encounter-1"
    assert event.system == "SNOMED-CT"
    assert event.code == "444814009"
    assert event.description == "Viral sinusitis (disorder)"


def test_from_row_ignores_columns_outside_the_payload() -> None:
    row = make_encounter_row()
    row["UNRELATED"] = "ignored"
    event = state.EncounterEvent.from_row(row)
    assert event.id == "encounter-1"


def test_from_row_missing_column_raises_key_error() -> None:
    row = make_encounter_row()
    del row["ENCOUNTERCLASS"]
    with pytest.raises(KeyError):
        state.EncounterEvent.from_row(row)


@pytest.mark.parametrize("column", ["Id", "PATIENT", "ENCOUNTERCLASS"])
def test_encounter_rejects_empty_identity_field(column: str) -> None:
    with pytest.raises(state.MalformedEventError):
        state.EncounterEvent.from_row(make_encounter_row(**{column: ""}))


def test_encounter_rejects_date_only_start() -> None:
    with pytest.raises(state.MalformedEventError):
        state.EncounterEvent.from_row(make_encounter_row(START="2024-01-01"))


def test_encounter_rejects_timestamp_without_zone_suffix() -> None:
    with pytest.raises(state.MalformedEventError):
        state.EncounterEvent.from_row(make_encounter_row(START="2024-01-01T08:00:00"))


def test_encounter_rejects_non_zero_padded_timestamp() -> None:
    with pytest.raises(state.MalformedEventError):
        state.EncounterEvent.from_row(make_encounter_row(START="2024-1-1T8:00:00Z"))


def test_encounter_allows_empty_stop() -> None:
    event = state.EncounterEvent.from_row(make_encounter_row(STOP=""))
    assert event.stop == ""


def test_encounter_rejects_malformed_stop() -> None:
    with pytest.raises(state.MalformedEventError):
        state.EncounterEvent.from_row(make_encounter_row(STOP="not-a-timestamp"))


@pytest.mark.parametrize("column", ["PATIENT", "ENCOUNTER", "CODE", "START"])
def test_medication_rejects_empty_key_component(column: str) -> None:
    with pytest.raises(state.MalformedEventError):
        state.MedicationEvent.from_row(make_medication_row(**{column: ""}))


def test_medication_rejects_date_only_start() -> None:
    with pytest.raises(state.MalformedEventError):
        state.MedicationEvent.from_row(make_medication_row(START="2024-01-01"))


def test_medication_allows_empty_stop() -> None:
    event = state.MedicationEvent.from_row(make_medication_row(STOP=""))
    assert event.stop == ""


def test_medication_rejects_malformed_stop() -> None:
    with pytest.raises(state.MalformedEventError):
        state.MedicationEvent.from_row(make_medication_row(STOP="2024-01-08"))


@pytest.mark.parametrize("column", ["PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "START"])
def test_condition_rejects_empty_key_component(column: str) -> None:
    with pytest.raises(state.MalformedEventError):
        state.ConditionEvent.from_row(make_condition_row(**{column: ""}))


def test_condition_rejects_full_timestamp_start() -> None:
    with pytest.raises(state.MalformedEventError):
        state.ConditionEvent.from_row(make_condition_row(START="2024-01-01T08:00:00Z"))


def test_condition_allows_empty_stop() -> None:
    event = state.ConditionEvent.from_row(make_condition_row(STOP=""))
    assert event.stop == ""


def test_condition_allows_empty_description() -> None:
    event = state.ConditionEvent.from_row(make_condition_row(DESCRIPTION=""))
    assert event.description == ""


def test_patient_rejects_empty_id() -> None:
    with pytest.raises(state.MalformedEventError):
        state.PatientEvent.from_row(make_patient_row(Id=""))


def test_patient_rejects_empty_birthdate() -> None:
    with pytest.raises(state.MalformedEventError):
        state.PatientEvent.from_row(make_patient_row(BIRTHDATE=""))


def test_patient_rejects_timestamp_birthdate() -> None:
    with pytest.raises(state.MalformedEventError):
        state.PatientEvent.from_row(make_patient_row(BIRTHDATE="1970-01-01T00:00:00Z"))


def test_patient_allows_empty_deathdate() -> None:
    event = state.PatientEvent.from_row(make_patient_row(DEATHDATE=""))
    assert event.deathdate == ""


def test_malformed_event_error_is_a_value_error() -> None:
    assert issubclass(state.MalformedEventError, ValueError)
