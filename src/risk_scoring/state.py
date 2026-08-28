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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT = "%Y-%m-%d"

PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE")
ENCOUNTER_COLUMNS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
MEDICATION_COLUMNS = ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE")
CONDITION_COLUMNS = ("START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION")


class MalformedEventError(ValueError):
    """A payload field is empty where required or fails its exact format."""


class EventConflictError(RuntimeError):
    """A re-posted event matched an existing key with different field values."""


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


@dataclass(frozen=True)
class _TableSpec:
    """One state table: database columns (event field names) and its natural key."""

    table: str
    db_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    frame_columns: tuple[str, ...]
    patient_column: str


_PATIENT_SPEC = _TableSpec(
    table="patients",
    db_columns=("id", "birthdate", "deathdate"),
    key_columns=("id",),
    frame_columns=PATIENT_COLUMNS,
    patient_column="id",
)
_ENCOUNTER_SPEC = _TableSpec(
    table="encounters",
    db_columns=("id", "start", "stop", "patient", "encounter_class"),
    key_columns=("id",),
    frame_columns=ENCOUNTER_COLUMNS,
    patient_column="patient",
)
_MEDICATION_SPEC = _TableSpec(
    table="medications",
    db_columns=("start", "stop", "patient", "encounter", "code"),
    key_columns=("patient", "encounter", "code", "start"),
    frame_columns=MEDICATION_COLUMNS,
    patient_column="patient",
)
_CONDITION_SPEC = _TableSpec(
    table="conditions",
    db_columns=("start", "stop", "patient", "encounter", "system", "code", "description"),
    key_columns=("patient", "encounter", "code", "start"),
    frame_columns=CONDITION_COLUMNS,
    patient_column="patient",
)


def _record(conn: psycopg.Connection[Any], spec: _TableSpec, values: dict[str, str]) -> bool:
    """Insert one event row, commit, and report whether it was new.

    Identical re-posts are a silent no-op; a re-post whose key exists with
    different field values rolls back and raises :class:`EventConflictError`.
    Each call commits its own row: an acknowledged event must be a persisted
    event for crash retries and replay resumes to hold, so callers must not
    wrap record calls in a larger transaction they intend to roll back.
    """
    column_list = ", ".join(spec.db_columns)
    placeholders = ", ".join(["%s"] * len(spec.db_columns))
    key_filter = " AND ".join(f"{name} = %s" for name in spec.key_columns)
    try:
        cursor = conn.execute(
            f"INSERT INTO {spec.table} ({column_list}) VALUES ({placeholders})"
            f" ON CONFLICT ({', '.join(spec.key_columns)}) DO NOTHING",
            [values[name] for name in spec.db_columns],
        )
        if cursor.rowcount == 1:
            conn.commit()
            return True
        stored_row = conn.execute(
            f"SELECT {column_list} FROM {spec.table} WHERE {key_filter}",
            [values[name] for name in spec.key_columns],
        ).fetchone()
        if stored_row is None:
            raise RuntimeError(f"{spec.table} row vanished between insert and read-back")
        stored = dict(zip(spec.db_columns, stored_row, strict=True))
        if stored == values:
            conn.commit()
            return False
        key = {name: values[name] for name in spec.key_columns}
        diffs = ", ".join(
            f"{name}: stored {stored[name]!r} != posted {values[name]!r}"
            for name in spec.db_columns
            if stored[name] != values[name]
        )
        raise EventConflictError(f"{spec.table} key {key} already ingested with {diffs}")
    except Exception:
        conn.rollback()
        raise


def record_patient(conn: psycopg.Connection[Any], event: PatientEvent) -> bool:
    """Persist a patient event; True if new, False on an identical re-post."""
    return _record(conn, _PATIENT_SPEC, asdict(event))


def record_encounter(conn: psycopg.Connection[Any], event: EncounterEvent) -> bool:
    """Persist an encounter event; True if new, False on an identical re-post."""
    return _record(conn, _ENCOUNTER_SPEC, asdict(event))


def record_medication(conn: psycopg.Connection[Any], event: MedicationEvent) -> bool:
    """Persist a medication event; True if new, False on an identical re-post."""
    return _record(conn, _MEDICATION_SPEC, asdict(event))


def record_condition(conn: psycopg.Connection[Any], event: ConditionEvent) -> bool:
    """Persist a condition event; True if new, False on an identical re-post."""
    return _record(conn, _CONDITION_SPEC, asdict(event))


@dataclass(frozen=True)
class PatientHistory:
    """One patient's raw history, shaped like the batch CSV frames.

    Columns carry the uppercase Synthea names in export order (restricted to
    the payload subset), values are verbatim strings with ``""`` for missing,
    and rows are ordered by start time, so the shared cohort and feature
    modules can consume these frames exactly as they consume CSV loads.
    """

    patients: pd.DataFrame
    encounters: pd.DataFrame
    medications: pd.DataFrame
    conditions: pd.DataFrame


def _history_frame(
    conn: psycopg.Connection[Any], spec: _TableSpec, patient_id: str, order_by: str
) -> pd.DataFrame:
    rows = conn.execute(
        f"SELECT {', '.join(spec.db_columns)} FROM {spec.table}"
        f" WHERE {spec.patient_column} = %s ORDER BY {order_by}",
        [patient_id],
    ).fetchall()
    return pd.DataFrame(rows, columns=list(spec.frame_columns))


def patient_history(conn: psycopg.Connection[Any], patient_id: str) -> PatientHistory:
    """Read one patient's full event history; read-only, never commits."""
    return PatientHistory(
        patients=_history_frame(conn, _PATIENT_SPEC, patient_id, "id"),
        encounters=_history_frame(conn, _ENCOUNTER_SPEC, patient_id, "start, id"),
        medications=_history_frame(conn, _MEDICATION_SPEC, patient_id, "start, encounter, code"),
        conditions=_history_frame(conn, _CONDITION_SPEC, patient_id, "start, encounter, code"),
    )
