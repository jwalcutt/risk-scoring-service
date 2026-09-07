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
- A non-empty encounter or condition ``stop`` must not precede its
  ``start``. Equality is allowed, since the export records zero-length
  encounters. A reversed encounter would otherwise reach the feature
  module as a negative length of stay and be scored and logged as if it
  were valid. Medications are exempt: the generator emits a small share
  of prescriptions with ``STOP`` one to six days before ``START`` (714 of
  553,590 in the frozen baseline), the feature module reads such a row as
  never active, and refusing it would stop a replay on generator output.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import psycopg

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT = "%Y-%m-%d"

PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE")
ENCOUNTER_COLUMNS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
MEDICATION_COLUMNS = ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE")
CONDITION_COLUMNS = ("START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION")


class MalformedEventError(ValueError):
    """A payload field is empty where required or fails its exact format."""


class EventConflictError(RuntimeError):
    """A re-posted event matched an existing key with different field values.

    Carries the table, the key the poster supplied, and the payload names
    of the columns that differ. It never carries the stored or posted
    values: the message travels back to the poster, and echoing the
    stored row would let anyone who can guess a key read the record
    behind it. The values go to the server log at the point of detection.
    """

    def __init__(self, table: str, key: Mapping[str, str], columns: Sequence[str]) -> None:
        self.table = table
        self.key = dict(key)
        self.columns = tuple(columns)
        super().__init__(
            f"{table} key {self.key} already ingested with different values for "
            f"{', '.join(self.columns)}"
        )


class PatientEventLimitError(RuntimeError):
    """Storing this event would take its patient past the per-patient event cap."""


def _check_exact_format(value: str, fmt: str, label: str) -> None:
    """Require a value that round-trips through the format unchanged."""
    try:
        parsed = datetime.strptime(value, fmt)
    except ValueError as exc:
        raise MalformedEventError(f"{label} must match {fmt!r}; got {value!r}") from exc
    if parsed.strftime(fmt) != value:
        raise MalformedEventError(f"{label} must match {fmt!r} exactly; got {value!r}")


def parse_timestamp(value: str) -> datetime:
    """Parse a stored event timestamp into an aware UTC datetime.

    Event values stay verbatim strings everywhere state is concerned, but
    the prediction log stores real timestamps, so the conversion lives
    here, beside the format that guarantees it is lossless: every stored
    value has already round-tripped through TIMESTAMP_FORMAT, whose
    literal Z fixes the zone.
    """
    _check_exact_format(value, TIMESTAMP_FORMAT, "timestamp")
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def _check_optional_format(value: str, fmt: str, label: str) -> None:
    if value != "":
        _check_exact_format(value, fmt, label)


def _check_non_empty(value: str, label: str) -> None:
    if not value:
        raise MalformedEventError(f"{label} must not be empty")


def _check_interval_order(start: str, stop: str, fmt: str, label: str) -> None:
    """Require a non-empty stop at or after start; both already format-checked."""
    if stop == "":
        return
    if datetime.strptime(stop, fmt) < datetime.strptime(start, fmt):
        raise MalformedEventError(f"{label} STOP must not precede START; got {stop!r} < {start!r}")


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
        _check_interval_order(self.start, self.stop, TIMESTAMP_FORMAT, "encounter")

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
        _check_interval_order(self.start, self.stop, DATE_FORMAT, "condition")

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


AnyEvent = PatientEvent | EncounterEvent | MedicationEvent | ConditionEvent
"""Any recordable event. The service's wire models convert into this union."""


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
    key_columns=("patient", "encounter", "code", "start", "stop"),
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


_RECORD_ATTEMPTS = 3
"""How many times an insert that conflicted may fail to read its conflicting row back."""

_CAPPED_SPECS = (_ENCOUNTER_SPEC, _MEDICATION_SPEC, _CONDITION_SPEC)

# One round trip over the three patient-leading btrees; demographics are one
# row per patient by primary key and never count.
_PATIENT_ROW_COUNT_SQL = " + ".join(
    f"(SELECT count(*) FROM {spec.table} WHERE {spec.patient_column} = %s)"
    for spec in _CAPPED_SPECS
)


def _patient_row_count(conn: psycopg.Connection[Any], patient_id: str) -> int:
    row = conn.execute(f"SELECT {_PATIENT_ROW_COUNT_SQL}", [patient_id] * len(_CAPPED_SPECS))
    count = row.fetchone()
    if count is None:
        raise RuntimeError("row count query returned nothing")
    return int(count[0])


def _record(
    conn: psycopg.Connection[Any],
    spec: _TableSpec,
    values: dict[str, str],
    max_patient_rows: int | None = None,
) -> bool:
    """Insert one event row, commit, and report whether it was new.

    Identical re-posts are a silent no-op; a re-post whose key exists with
    different field values rolls back and raises :class:`EventConflictError`.
    Each call commits its own row: an acknowledged event must be a persisted
    event for crash retries and replay resumes to hold, so callers must not
    wrap record calls in a larger transaction they intend to roll back.

    An insert that conflicted and then read back no row is the trace of
    another writer, not a broken table: the conflicting row was there for
    the insert and gone for the read. The insert and read-back run again,
    a bounded number of times, before that becomes an error.

    With ``max_patient_rows``, a new encounter, medication, or condition
    that would take its patient's stored total past the cap is rolled back
    and raises :class:`PatientEventLimitError`. The count runs inside the
    insert's own transaction, after the insert and before the commit, so it
    is exact for this connection and costs one indexed round trip. A
    re-post never reaches it: a row that already exists is not new.
    """
    column_list = ", ".join(spec.db_columns)
    placeholders = ", ".join(["%s"] * len(spec.db_columns))
    key_filter = " AND ".join(f"{name} = %s" for name in spec.key_columns)
    insert = (
        f"INSERT INTO {spec.table} ({column_list}) VALUES ({placeholders})"
        f" ON CONFLICT ({', '.join(spec.key_columns)}) DO NOTHING"
    )
    read_back = f"SELECT {column_list} FROM {spec.table} WHERE {key_filter}"
    try:
        for _ in range(_RECORD_ATTEMPTS):
            inserted = conn.execute(insert, [values[name] for name in spec.db_columns])
            if inserted.rowcount == 1:
                if max_patient_rows is not None and spec in _CAPPED_SPECS:
                    patient_id = values[spec.patient_column]
                    total = _patient_row_count(conn, patient_id)
                    if total > max_patient_rows:
                        raise PatientEventLimitError(
                            f"patient {patient_id!r} already holds {total - 1} events, the "
                            f"configured limit of {max_patient_rows} per patient; this "
                            f"{spec.table[:-1]} was not stored"
                        )
                conn.commit()
                return True
            stored_row = conn.execute(
                read_back, [values[name] for name in spec.key_columns]
            ).fetchone()
            if stored_row is not None:
                break
        else:
            raise RuntimeError(
                f"{spec.table} row conflicted on insert but was absent on read-back "
                f"{_RECORD_ATTEMPTS} times in a row"
            )
        stored = dict(zip(spec.db_columns, stored_row, strict=True))
        if stored == values:
            conn.commit()
            return False
        payload_name = dict(zip(spec.db_columns, spec.frame_columns, strict=True))
        key = {payload_name[name]: values[name] for name in spec.key_columns}
        differing = [name for name in spec.db_columns if stored[name] != values[name]]
        logger.warning(
            "%s key %s already ingested with %s",
            spec.table,
            key,
            ", ".join(
                f"{payload_name[name]}: stored {stored[name]!r} != posted {values[name]!r}"
                for name in differing
            ),
        )
        raise EventConflictError(spec.table, key, [payload_name[name] for name in differing])
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


_SPEC_BY_EVENT: dict[type[AnyEvent], _TableSpec] = {
    PatientEvent: _PATIENT_SPEC,
    EncounterEvent: _ENCOUNTER_SPEC,
    MedicationEvent: _MEDICATION_SPEC,
    ConditionEvent: _CONDITION_SPEC,
}


def record_event(
    conn: psycopg.Connection[Any], event: AnyEvent, *, max_patient_rows: int | None = None
) -> bool:
    """Persist any event, dispatching on its type.

    The entry point for callers that hold an event without caring which
    kind it is, such as the ingestion endpoint. Same contract as the
    per-type functions: True if new, False on an identical re-post. With
    ``max_patient_rows``, a new clinical event past the patient's cap
    raises :class:`PatientEventLimitError` and stores nothing.
    """
    return _record(conn, _SPEC_BY_EVENT[type(event)], asdict(event), max_patient_rows)


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


@dataclass(frozen=True)
class HistoryWindow:
    """The rows one discharge's features can read, as bounds on the stored strings.

    Every bound is a verbatim timestamp or date in the export's format, so
    the comparison is the same lexicographic order the tables are already
    indexed and sorted by. Which rows a feature can read is the feature
    module's business; the caller derives these values from it (see
    ``risk_scoring.serving.history_window``), and this module only applies
    them.
    """

    encounter_stop_from: str
    """Encounters are read when their STOP is at or after this instant."""

    discharge: str
    """The discharge instant: encounter STOP and medication START at or
    before it, medication STOP after it or empty."""

    discharge_date: str
    """The discharge date: condition START on or before it, resolved or not."""


def _history_frame(
    conn: psycopg.Connection[Any],
    spec: _TableSpec,
    patient_id: str,
    order_by: str,
    bounds: str = "",
    params: Sequence[str] = (),
) -> pd.DataFrame:
    rows = conn.execute(
        f"SELECT {', '.join(spec.db_columns)} FROM {spec.table}"
        f" WHERE {spec.patient_column} = %s{bounds} ORDER BY {order_by}",
        [patient_id, *params],
    ).fetchall()
    return pd.DataFrame(rows, columns=list(spec.frame_columns))


def has_patient(conn: psycopg.Connection[Any], patient_id: str) -> bool:
    """Whether a patient's demographics are recorded; read-only, never commits."""
    return conn.execute("SELECT 1 FROM patients WHERE id = %s", [patient_id]).fetchone() is not None


def patient_history(
    conn: psycopg.Connection[Any], patient_id: str, window: HistoryWindow | None = None
) -> PatientHistory:
    """Read one patient's event history; read-only, never commits.

    Without a window this is the whole record. With one, each table is
    narrowed to the rows inside it, which is what scoring one discharge
    reads: the frames are exactly what the feature module would have
    read from the full record, minus rows it would have ignored.
    """
    if window is None:
        return PatientHistory(
            patients=_history_frame(conn, _PATIENT_SPEC, patient_id, "id"),
            encounters=_history_frame(conn, _ENCOUNTER_SPEC, patient_id, "start, id"),
            medications=_history_frame(
                conn, _MEDICATION_SPEC, patient_id, "start, encounter, code"
            ),
            conditions=_history_frame(conn, _CONDITION_SPEC, patient_id, "start, encounter, code"),
        )
    return PatientHistory(
        patients=_history_frame(conn, _PATIENT_SPEC, patient_id, "id"),
        encounters=_history_frame(
            conn,
            _ENCOUNTER_SPEC,
            patient_id,
            "start, id",
            " AND stop >= %s AND stop <= %s",
            [window.encounter_stop_from, window.discharge],
        ),
        medications=_history_frame(
            conn,
            _MEDICATION_SPEC,
            patient_id,
            "start, encounter, code",
            " AND start <= %s AND (stop = '' OR stop > %s)",
            [window.discharge, window.discharge],
        ),
        conditions=_history_frame(
            conn,
            _CONDITION_SPEC,
            patient_id,
            "start, encounter, code",
            " AND start <= %s",
            [window.discharge_date],
        ),
    )


def record_batch(conn: psycopg.Connection[Any], events: Sequence[AnyEvent]) -> int:
    """Persist many events in one transaction and commit; the number that were new.

    For loading history in bulk, where the per-row commit of
    :func:`record_event` costs more than the write itself. The batch is
    the unit of acknowledgement here: one commit covers it, and a caller
    that dies mid-load resumes by loading again, since identical re-posts
    are dropped by the conflict clause exactly as they are per row.

    Divergence is still refused loudly. When fewer rows landed than the
    batch holds, every event in it is re-run through :func:`record_event`,
    which is a no-op for an identical re-post and raises
    :class:`EventConflictError` for a divergent one. The batch has
    committed by then, so what was new in it stays: nothing acknowledged
    is ever rolled back.
    """
    if not events:
        return 0
    grouped: dict[type[AnyEvent], list[dict[str, str]]] = {}
    for event in events:
        grouped.setdefault(type(event), []).append(asdict(event))
    inserted = 0
    try:
        for event_type, rows in grouped.items():
            spec = _SPEC_BY_EVENT[event_type]
            column_list = ", ".join(spec.db_columns)
            placeholders = ", ".join(["%s"] * len(spec.db_columns))
            with conn.cursor() as cursor:
                cursor.executemany(
                    f"INSERT INTO {spec.table} ({column_list}) VALUES ({placeholders})"
                    f" ON CONFLICT ({', '.join(spec.key_columns)}) DO NOTHING",
                    [[row[name] for name in spec.db_columns] for row in rows],
                )
                inserted += cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if inserted < len(events):
        for event in events:
            record_event(conn, event)
    return inserted
