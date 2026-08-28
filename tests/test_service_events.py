"""Tests for the ingestion payload models.

The rules these tests pin:

- Payloads carry exactly the contract columns: the fields the shared
  cohort and feature modules read, plus the identity fields idempotent
  ingestion needs (ENCOUNTER and CODE for medications, ENCOUNTER for
  conditions, which have no Synthea row Id).
- Values are verbatim CSV strings: format-checked, never converted, so a
  validated payload round-trips to the exact posted dict.
- Extra fields, missing fields, malformed timestamps, empty identity
  fields, and unknown event types are all rejected loudly.
- Empty STOP is accepted everywhere the export can leave it empty (open
  stays, ongoing medications, unresolved conditions) and an empty
  DEATHDATE marks a living patient.
- The event envelope discriminates on event_type, including the patient
  event type added beyond the original three-type contract because the
  cohort cannot admit an encounter without demographics.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring.service.events import (
    ConditionPayload,
    EncounterPayload,
    Event,
    MedicationPayload,
    PatientPayload,
)

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

ENCOUNTER_FIELDS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
MEDICATION_FIELDS = ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE")
CONDITION_FIELDS = ("START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION")
PATIENT_FIELDS = ("Id", "BIRTHDATE", "DEATHDATE")


def _project(row: dict[str, str], fields: tuple[str, ...]) -> dict[str, str]:
    return {field: row[field] for field in fields}


# --- verbatim round-trips ---


def test_encounter_payload_preserves_factory_row_verbatim() -> None:
    projected = _project(make_encounter_row(ENCOUNTERCLASS="inpatient"), ENCOUNTER_FIELDS)
    assert EncounterPayload.model_validate(projected).model_dump() == projected


def test_medication_payload_preserves_factory_row_verbatim() -> None:
    projected = _project(make_medication_row(), MEDICATION_FIELDS)
    assert MedicationPayload.model_validate(projected).model_dump() == projected


def test_condition_payload_preserves_factory_row_verbatim() -> None:
    projected = _project(make_condition_row(), CONDITION_FIELDS)
    assert ConditionPayload.model_validate(projected).model_dump() == projected


def test_patient_payload_preserves_factory_row_verbatim() -> None:
    projected = _project(make_patient_row(), PATIENT_FIELDS)
    assert PatientPayload.model_validate(projected).model_dump() == projected


# --- rejections ---


def test_full_synthea_row_with_extra_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        EncounterPayload.model_validate(make_encounter_row())


def test_missing_required_field_rejected() -> None:
    projected = _project(make_encounter_row(), ENCOUNTER_FIELDS)
    del projected["PATIENT"]
    with pytest.raises(ValidationError):
        EncounterPayload.model_validate(projected)


@pytest.mark.parametrize(
    "start",
    ["2024-01-01", "2024-01-01 08:00:00", "01/01/2024T08:00:00Z", "not-a-time", ""],
)
def test_malformed_encounter_start_rejected(start: str) -> None:
    projected = _project(make_encounter_row(START=start), ENCOUNTER_FIELDS)
    with pytest.raises(ValidationError):
        EncounterPayload.model_validate(projected)


def test_datetime_where_condition_date_expected_rejected() -> None:
    projected = _project(make_condition_row(START="2024-01-01T08:00:00Z"), CONDITION_FIELDS)
    with pytest.raises(ValidationError):
        ConditionPayload.model_validate(projected)


@pytest.mark.parametrize("field", ["Id", "PATIENT"])
def test_empty_encounter_identity_field_rejected(field: str) -> None:
    projected = _project(make_encounter_row(**{field: ""}), ENCOUNTER_FIELDS)
    with pytest.raises(ValidationError):
        EncounterPayload.model_validate(projected)


def test_empty_patient_id_rejected() -> None:
    projected = _project(make_patient_row(Id=""), PATIENT_FIELDS)
    with pytest.raises(ValidationError):
        PatientPayload.model_validate(projected)


# --- empty STOP and DEATHDATE ---


def test_empty_stop_accepted_on_all_clinical_payloads() -> None:
    encounter = _project(make_encounter_row(STOP=""), ENCOUNTER_FIELDS)
    medication = _project(make_medication_row(STOP=""), MEDICATION_FIELDS)
    condition = _project(make_condition_row(STOP=""), CONDITION_FIELDS)
    assert EncounterPayload.model_validate(encounter).STOP == ""
    assert MedicationPayload.model_validate(medication).STOP == ""
    assert ConditionPayload.model_validate(condition).STOP == ""


def test_empty_deathdate_accepted_for_living_patient() -> None:
    projected = _project(make_patient_row(DEATHDATE=""), PATIENT_FIELDS)
    assert PatientPayload.model_validate(projected).DEATHDATE == ""


# --- envelope discrimination ---


def test_envelope_routes_each_event_type_to_its_payload_model() -> None:
    cases: list[tuple[str, dict[str, str], type[object]]] = [
        ("encounter", _project(make_encounter_row(), ENCOUNTER_FIELDS), EncounterPayload),
        ("medication", _project(make_medication_row(), MEDICATION_FIELDS), MedicationPayload),
        ("condition", _project(make_condition_row(), CONDITION_FIELDS), ConditionPayload),
        ("patient", _project(make_patient_row(), PATIENT_FIELDS), PatientPayload),
    ]
    for event_type, payload, payload_cls in cases:
        event = _EVENT_ADAPTER.validate_python({"event_type": event_type, "payload": payload})
        assert event.event_type == event_type
        assert isinstance(event.payload, payload_cls)


def test_unknown_event_type_rejected() -> None:
    payload = _project(make_encounter_row(), ENCOUNTER_FIELDS)
    with pytest.raises(ValidationError):
        _EVENT_ADAPTER.validate_python({"event_type": "observation", "payload": payload})


def test_envelope_rejects_extra_top_level_fields() -> None:
    payload = _project(make_encounter_row(), ENCOUNTER_FIELDS)
    with pytest.raises(ValidationError):
        _EVENT_ADAPTER.validate_python(
            {"event_type": "encounter", "payload": payload, "source": "replay"}
        )
