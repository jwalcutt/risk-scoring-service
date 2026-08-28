"""Ingestion payload contract: the event types the service accepts.

Payload schemas mirror the Synthea CSV columns the shared cohort and
feature modules read, so a validated payload maps to a CSV row by
identity. :func:`to_state_event` converts one into the matching
:mod:`risk_scoring.state` event, which is what gets persisted.

Judgment calls this module fixes:

- The wire layer owns shape; ``state`` owns format. These models declare
  the field set and reject anything structurally wrong (missing field,
  unknown field, unknown event type) with FastAPI's 422. Every value
  rule — required-and-non-empty, the exact timestamp and date formats,
  which fields may be empty — lives once, in the ``state`` event
  dataclasses, and fires at conversion. Two copies of the format rules
  existed briefly while the ingestion boundary and the state layer were
  built in parallel; collapsing them is what keeps the wire schema and
  the stored schema from drifting apart.
- Every field is a verbatim CSV string, never converted, keeping
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
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from risk_scoring import state


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EncounterPayload(_StrictModel):
    Id: str
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTERCLASS: str


class MedicationPayload(_StrictModel):
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTER: str
    CODE: str


class ConditionPayload(_StrictModel):
    START: str
    STOP: str
    PATIENT: str
    ENCOUNTER: str
    SYSTEM: str
    CODE: str
    DESCRIPTION: str


class PatientPayload(_StrictModel):
    Id: str
    BIRTHDATE: str
    DEATHDATE: str


class EncounterEnvelope(_StrictModel):
    event_type: Literal["encounter"]
    payload: EncounterPayload


class MedicationEnvelope(_StrictModel):
    event_type: Literal["medication"]
    payload: MedicationPayload


class ConditionEnvelope(_StrictModel):
    event_type: Literal["condition"]
    payload: ConditionPayload


class PatientEnvelope(_StrictModel):
    event_type: Literal["patient"]
    payload: PatientPayload


Event = Annotated[
    EncounterEnvelope | MedicationEnvelope | ConditionEnvelope | PatientEnvelope,
    Field(discriminator="event_type"),
]

_STATE_EVENT_BY_TYPE: dict[str, type[state.AnyEvent]] = {
    "encounter": state.EncounterEvent,
    "medication": state.MedicationEvent,
    "condition": state.ConditionEvent,
    "patient": state.PatientEvent,
}


def to_state_event(event: Event) -> state.AnyEvent:
    """Convert a validated envelope into the state event it persists as.

    Raises :class:`risk_scoring.state.MalformedEventError` when a field
    value fails its format rule; the payload dict keys are the uppercase
    Synthea column names the ``from_row`` constructors already expect, so
    the conversion is a lookup, never a re-listing of the columns.
    """
    return _STATE_EVENT_BY_TYPE[event.event_type].from_row(event.payload.model_dump())
