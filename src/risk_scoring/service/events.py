"""Ingestion payload contract: the event types the service accepts.

Payload schemas mirror the Synthea CSV columns the shared cohort and
feature modules read, so a validated payload maps to a CSV row by
identity. Track A's state layer adopts these models at the merge.

Judgment calls this module fixes:

- Every field is a verbatim CSV string: format-checked, never converted.
  Values must round-trip exactly (zero-padded timestamps), keeping
  "unmodified field values" literal for the input hash and letting state
  rebuild frames byte-identical to the batch export.
- Payloads carry only the columns the shared modules read, plus the
  identity fields idempotent ingestion needs: medications and conditions
  have no Synthea row Id, so they carry ENCOUNTER (and CODE for
  medications) toward a deterministic composite key.
- A ``patient`` event type extends the original three-type contract:
  the cohort module raises on an inpatient encounter whose patient is
  unknown, so demographics must arrive through the same stream, before
  the patient's first clinical event.
- ``extra="forbid"`` everywhere: an unknown field is a loud 422, never a
  silent drop, and the hashed bytes only ever contain contracted fields.
- Empty STOP stays allowed (open stays, ongoing medications, unresolved
  conditions), matching what the feature module already handles; an
  empty DEATHDATE marks a living patient.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_DATE_FORMAT = "%Y-%m-%d"


def _check_exact_format(value: str, fmt: str, label: str) -> str:
    """Require a value that round-trips through the format unchanged."""
    try:
        parsed = datetime.strptime(value, fmt)
    except ValueError as exc:
        raise ValueError(f"{label} must match {fmt!r}; got {value!r}") from exc
    if parsed.strftime(fmt) != value:
        raise ValueError(f"{label} must match {fmt!r} exactly; got {value!r}")
    return value


def _non_empty(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EncounterPayload(_StrictModel):
    Id: str
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTERCLASS: str

    @field_validator("Id", "PATIENT", "ENCOUNTERCLASS")
    @classmethod
    def _identity_non_empty(cls, value: str) -> str:
        return _non_empty(value, "encounter identity field")

    @field_validator("START")
    @classmethod
    def _start_is_timestamp(cls, value: str) -> str:
        return _check_exact_format(value, _TIMESTAMP_FORMAT, "encounter START")

    @field_validator("STOP")
    @classmethod
    def _stop_is_timestamp_or_empty(cls, value: str) -> str:
        if value == "":
            return value
        return _check_exact_format(value, _TIMESTAMP_FORMAT, "encounter STOP")


class MedicationPayload(_StrictModel):
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTER: str
    CODE: str

    @field_validator("PATIENT", "ENCOUNTER", "CODE")
    @classmethod
    def _identity_non_empty(cls, value: str) -> str:
        return _non_empty(value, "medication identity field")

    @field_validator("START")
    @classmethod
    def _start_is_timestamp(cls, value: str) -> str:
        return _check_exact_format(value, _TIMESTAMP_FORMAT, "medication START")

    @field_validator("STOP")
    @classmethod
    def _stop_is_timestamp_or_empty(cls, value: str) -> str:
        if value == "":
            return value
        return _check_exact_format(value, _TIMESTAMP_FORMAT, "medication STOP")


class ConditionPayload(_StrictModel):
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTER: str
    SYSTEM: str
    CODE: str
    DESCRIPTION: str

    @field_validator("PATIENT", "ENCOUNTER", "SYSTEM", "CODE")
    @classmethod
    def _identity_non_empty(cls, value: str) -> str:
        return _non_empty(value, "condition identity field")

    @field_validator("START")
    @classmethod
    def _start_is_date(cls, value: str) -> str:
        return _check_exact_format(value, _DATE_FORMAT, "condition START")

    @field_validator("STOP")
    @classmethod
    def _stop_is_date_or_empty(cls, value: str) -> str:
        if value == "":
            return value
        return _check_exact_format(value, _DATE_FORMAT, "condition STOP")


class PatientPayload(_StrictModel):
    Id: str
    BIRTHDATE: str
    DEATHDATE: str

    @field_validator("Id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        return _non_empty(value, "patient Id")

    @field_validator("BIRTHDATE")
    @classmethod
    def _birthdate_is_date(cls, value: str) -> str:
        return _check_exact_format(value, _DATE_FORMAT, "patient BIRTHDATE")

    @field_validator("DEATHDATE")
    @classmethod
    def _deathdate_is_date_or_empty(cls, value: str) -> str:
        if value == "":
            return value
        return _check_exact_format(value, _DATE_FORMAT, "patient DEATHDATE")


class EncounterEvent(_StrictModel):
    event_type: Literal["encounter"]
    payload: EncounterPayload


class MedicationEvent(_StrictModel):
    event_type: Literal["medication"]
    payload: MedicationPayload


class ConditionEvent(_StrictModel):
    event_type: Literal["condition"]
    payload: ConditionPayload


class PatientEvent(_StrictModel):
    event_type: Literal["patient"]
    payload: PatientPayload


Event = Annotated[
    EncounterEvent | MedicationEvent | ConditionEvent | PatientEvent,
    Field(discriminator="event_type"),
]
