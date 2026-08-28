"""Per-patient event state: typed events over the raw-history tables.

The service persists each patient's raw event history — the encounter,
medication, condition, and patient fields the shared cohort and feature
modules read — and recomputes features through those modules at scoring
time. This module owns the typed side of that contract: event values are
verbatim CSV strings, format-checked but never converted, so state can
hand back frames byte-identical to the batch CSV path.

Judgment calls this module fixes:

- Validation happens at construction (``__post_init__``), so a constructed
  event is well-formed by definition and rejection is loud and testable
  without a database. Malformed values raise :class:`MalformedEventError`.
- Empty string, never ``None``, is the missing-value representation,
  mirroring how the training pipeline reads CSVs
  (``dtype=str, keep_default_na=False``).
- Optional fields (encounter/medication/condition ``stop``, patient
  ``deathdate``) accept ``""`` or an exactly formatted value. Required
  timestamps and dates must round-trip through their format unchanged,
  so non-zero-padded near-misses are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT = "%Y-%m-%d"


class MalformedEventError(ValueError):
    """A payload field is empty where required or fails its exact format."""


def _check_exact_format(value: str, fmt: str, label: str) -> None:
    """Require a value that round-trips through the format unchanged."""
    try:
        parsed = datetime.strptime(value, fmt)
    except ValueError as exc:
        raise MalformedEventError(f"{label} must match {fmt!r}; got {value!r}") from exc
    if parsed.strftime(fmt) != value:
        raise MalformedEventError(f"{label} must match {fmt!r} exactly; got {value!r}")


def _check_optional_format(value: str, fmt: str, label: str) -> None:
    if value != "":
        _check_exact_format(value, fmt, label)


def _check_non_empty(value: str, label: str) -> None:
    if not value:
        raise MalformedEventError(f"{label} must not be empty")


@dataclass(frozen=True)
class PatientEvent:
    """Demographics row: the patient columns the cohort module reads."""

    id: str
    birthdate: str
    deathdate: str

    def __post_init__(self) -> None:
        _check_non_empty(self.id, "patient Id")
        _check_exact_format(self.birthdate, DATE_FORMAT, "patient BIRTHDATE")
        _check_optional_format(self.deathdate, DATE_FORMAT, "patient DEATHDATE")

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> PatientEvent:
        return cls(id=row["Id"], birthdate=row["BIRTHDATE"], deathdate=row["DEATHDATE"])


@dataclass(frozen=True)
class EncounterEvent:
    """Encounter row: the columns the cohort and feature modules read."""

    id: str
    start: str
    stop: str
    patient: str
    encounter_class: str

    def __post_init__(self) -> None:
        _check_non_empty(self.id, "encounter Id")
        _check_non_empty(self.patient, "encounter PATIENT")
        _check_non_empty(self.encounter_class, "encounter ENCOUNTERCLASS")
        _check_exact_format(self.start, TIMESTAMP_FORMAT, "encounter START")
        _check_optional_format(self.stop, TIMESTAMP_FORMAT, "encounter STOP")

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> EncounterEvent:
        return cls(
            id=row["Id"],
            start=row["START"],
            stop=row["STOP"],
            patient=row["PATIENT"],
            encounter_class=row["ENCOUNTERCLASS"],
        )


@dataclass(frozen=True)
class MedicationEvent:
    """Medication row; has no Synthea row Id, so every key field is required."""

    start: str
    stop: str
    patient: str
    encounter: str
    code: str

    def __post_init__(self) -> None:
        _check_non_empty(self.patient, "medication PATIENT")
        _check_non_empty(self.encounter, "medication ENCOUNTER")
        _check_non_empty(self.code, "medication CODE")
        _check_exact_format(self.start, TIMESTAMP_FORMAT, "medication START")
        _check_optional_format(self.stop, TIMESTAMP_FORMAT, "medication STOP")

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> MedicationEvent:
        return cls(
            start=row["START"],
            stop=row["STOP"],
            patient=row["PATIENT"],
            encounter=row["ENCOUNTER"],
            code=row["CODE"],
        )


@dataclass(frozen=True)
class ConditionEvent:
    """Condition row; date-only timestamps, matching the Synthea export."""

    start: str
    stop: str
    patient: str
    encounter: str
    system: str
    code: str
    description: str

    def __post_init__(self) -> None:
        _check_non_empty(self.patient, "condition PATIENT")
        _check_non_empty(self.encounter, "condition ENCOUNTER")
        _check_non_empty(self.system, "condition SYSTEM")
        _check_non_empty(self.code, "condition CODE")
        _check_exact_format(self.start, DATE_FORMAT, "condition START")
        _check_optional_format(self.stop, DATE_FORMAT, "condition STOP")

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> ConditionEvent:
        return cls(
            start=row["START"],
            stop=row["STOP"],
            patient=row["PATIENT"],
            encounter=row["ENCOUNTER"],
            system=row["SYSTEM"],
            code=row["CODE"],
            description=row["DESCRIPTION"],
        )
