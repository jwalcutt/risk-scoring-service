"""Integration tests for event ingestion and per-patient state read-back.

The contract under test: recording events is idempotent under identical
re-posts, loud under divergent ones, and ``patient_history`` returns
frames whose values are byte-identical to the source rows — the property
serving-time feature recompute rides on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from types import SimpleNamespace
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

# Medications are absent: their natural key is their whole payload, so a
# medication event cannot diverge. Every other type carries fields outside
# its key that a buggy producer could contradict.
DIVERGENT_OVERRIDES: dict[str, dict[str, str]] = {
    "patient": {"DEATHDATE": "2024-06-01"},
    "encounter": {"ENCOUNTERCLASS": "inpatient"},
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
    ("label", "make_row"),
    [(t[0], t[1]) for t in EVENT_TYPES if t[0] in DIVERGENT_OVERRIDES],
    ids=[t[0] for t in EVENT_TYPES if t[0] in DIVERGENT_OVERRIDES],
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
        "condition": state.CONDITION_COLUMNS,
    }[label]
    pd.testing.assert_frame_equal(stored, _frame([row], expected_columns))


def test_conflict_error_names_the_column_but_not_the_stored_value(
    db_conn: psycopg.Connection[Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The exception is what the poster hears; the values it contradicts stay server-side."""
    _record(db_conn, "encounter", make_encounter_row(ENCOUNTERCLASS="emergency"))

    with (
        caplog.at_level(logging.WARNING, logger="risk_scoring.state"),
        pytest.raises(state.EventConflictError) as excinfo,
    ):
        _record(db_conn, "encounter", make_encounter_row(ENCOUNTERCLASS="inpatient"))

    error = excinfo.value
    assert error.table == "encounters"
    assert error.key == {"Id": "encounter-1"}
    assert error.columns == ("ENCOUNTERCLASS",)
    message = str(error)
    assert "encounters" in message
    assert "encounter-1" in message
    assert "ENCOUNTERCLASS" in message
    assert "emergency" not in message
    assert "inpatient" not in message

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    logged = warnings[0].getMessage()
    assert "encounter-1" in logged
    assert "ENCOUNTERCLASS" in logged
    assert "emergency" in logged
    assert "inpatient" in logged


def test_connection_usable_after_conflict(db_conn: psycopg.Connection[Any]) -> None:
    _record(db_conn, "encounter", make_encounter_row())
    with pytest.raises(state.EventConflictError):
        _record(db_conn, "encounter", make_encounter_row(ENCOUNTERCLASS="inpatient"))

    assert _record(db_conn, "encounter", make_encounter_row(Id="encounter-2")) is True


def test_history_ordered_by_start_regardless_of_arrival_order(
    db_conn: psycopg.Connection[Any],
) -> None:
    late = make_encounter_row(
        Id="encounter-late", START="2024-06-01T08:00:00Z", STOP="2024-06-03T08:00:00Z"
    )
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


def test_same_drug_same_encounter_same_start_but_different_stop_are_distinct(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Synthea emits a single dispense and a continuing course as two rows.

    They share patient, encounter, code, and start, differing only in stop
    and the cost columns the payload drops, so the medication key must carry
    stop or the second row is lost and the active-medication count undercounts.
    """
    single_dispense = make_medication_row(STOP="2016-11-23T03:09:52Z", START="2016-11-23T03:09:52Z")
    year_long = make_medication_row(STOP="2017-11-29T03:09:52Z", START="2016-11-23T03:09:52Z")

    assert _record(db_conn, "medication", single_dispense) is True
    assert _record(db_conn, "medication", year_long) is True
    assert _record(db_conn, "medication", year_long) is False

    history = state.patient_history(db_conn, "patient-1")
    assert history.medications["STOP"].tolist() == [
        "2016-11-23T03:09:52Z",
        "2017-11-29T03:09:52Z",
    ]


@pytest.mark.parametrize(("label", "factory", "event_name", "_record_name"), EVENT_TYPES)
def test_record_event_dispatches_on_the_event_type(
    db_conn: psycopg.Connection[Any],
    label: str,
    factory: RowFactory,
    event_name: str,
    _record_name: str,
) -> None:
    """One entry point for callers that hold an event without knowing its kind."""
    row = factory()
    event = getattr(state, event_name).from_row(row)

    assert state.record_event(db_conn, event) is True
    assert state.record_event(db_conn, event) is False

    patient = row["Id"] if label == "patient" else row["PATIENT"]
    history = state.patient_history(db_conn, patient)
    stored = {
        "patient": history.patients,
        "encounter": history.encounters,
        "medication": history.medications,
        "condition": history.conditions,
    }[label]
    assert len(stored) == 1


# Batched writes, for loading history that predates a replay.


def test_record_batch_reads_back_identical_to_per_row_recording(
    db_conn: psycopg.Connection[Any],
) -> None:
    patient_row = make_patient_row()
    early = make_encounter_row(Id="encounter-early", START="2023-06-01T08:00:00Z")
    late = make_encounter_row(
        Id="encounter-late", START="2024-06-01T08:00:00Z", STOP="2024-06-03T08:00:00Z"
    )
    medication_row = make_medication_row(ENCOUNTER="encounter-late")
    condition_row = make_condition_row(ENCOUNTER="encounter-late")
    events: list[state.AnyEvent] = [
        state.PatientEvent.from_row(patient_row),
        state.EncounterEvent.from_row(late),
        state.MedicationEvent.from_row(medication_row),
        state.EncounterEvent.from_row(early),
        state.ConditionEvent.from_row(condition_row),
    ]

    assert state.record_batch(db_conn, events) == 5

    history = state.patient_history(db_conn, "patient-1")
    pd.testing.assert_frame_equal(history.patients, _frame([patient_row], state.PATIENT_COLUMNS))
    pd.testing.assert_frame_equal(
        history.encounters, _frame([early, late], state.ENCOUNTER_COLUMNS)
    )
    pd.testing.assert_frame_equal(
        history.medications, _frame([medication_row], state.MEDICATION_COLUMNS)
    )
    pd.testing.assert_frame_equal(
        history.conditions, _frame([condition_row], state.CONDITION_COLUMNS)
    )


def test_record_batch_survives_a_rollback(db_conn: psycopg.Connection[Any]) -> None:
    """The batch commits as a whole; a later rollback cannot take it back."""
    state.record_batch(db_conn, [state.EncounterEvent.from_row(make_encounter_row())])
    db_conn.rollback()
    assert len(state.patient_history(db_conn, "patient-1").encounters) == 1


def test_record_batch_of_identical_reposts_is_a_noop(
    db_conn: psycopg.Connection[Any],
) -> None:
    events: list[state.AnyEvent] = [
        state.PatientEvent.from_row(make_patient_row()),
        state.EncounterEvent.from_row(make_encounter_row()),
    ]
    state.record_batch(db_conn, events)

    assert state.record_batch(db_conn, events) == 0
    history = state.patient_history(db_conn, "patient-1")
    assert (len(history.patients), len(history.encounters)) == (1, 1)


def test_record_batch_counts_only_the_rows_that_were_new(
    db_conn: psycopg.Connection[Any],
) -> None:
    """A batch that resumes over rows already loaded reports the new ones alone."""
    first = state.EncounterEvent.from_row(make_encounter_row(Id="encounter-1"))
    second = state.EncounterEvent.from_row(make_encounter_row(Id="encounter-2"))
    state.record_batch(db_conn, [first])

    assert state.record_batch(db_conn, [first, second]) == 1


def test_record_batch_with_a_divergent_row_raises_and_keeps_the_new_rows(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Divergence is still refused loudly; what was new in the batch stays committed."""
    stored = make_encounter_row(Id="encounter-1", ENCOUNTERCLASS="emergency")
    _record(db_conn, "encounter", stored)
    divergent = make_encounter_row(Id="encounter-1", ENCOUNTERCLASS="inpatient")
    fresh = make_encounter_row(Id="encounter-2")

    with pytest.raises(state.EventConflictError):
        state.record_batch(
            db_conn,
            [state.EncounterEvent.from_row(divergent), state.EncounterEvent.from_row(fresh)],
        )

    history = state.patient_history(db_conn, "patient-1")
    pd.testing.assert_frame_equal(
        history.encounters, _frame([stored, fresh], state.ENCOUNTER_COLUMNS)
    )
    # The connection is still usable.
    assert _record(db_conn, "encounter", make_encounter_row(Id="encounter-3")) is True


def test_record_batch_of_nothing_records_nothing(db_conn: psycopg.Connection[Any]) -> None:
    assert state.record_batch(db_conn, []) == 0


# The read-back race: an insert that hit a conflict, then a read-back that
# finds no row. With more than one writer this is a moment to retry, not a
# reason to fail the request.

_READ_BACK = "FROM encounters WHERE id = %s"


class _TamperedReadBack(psycopg.Connection[Any]):
    """A real connection whose encounter read-backs pass through a hook first."""

    read_backs: int
    before_read_back: Callable[[], bool]
    """Runs before each read-back; returning False makes that read-back find nothing."""

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        if isinstance(query, str) and query.startswith("SELECT") and _READ_BACK in query:
            self.read_backs += 1
            if not self.before_read_back():
                return SimpleNamespace(fetchone=lambda: None)
        return super().execute(query, params, **kwargs)


@pytest.fixture()
def tampered(db_url: str) -> Iterator[_TamperedReadBack]:
    conn = _TamperedReadBack.connect(db_url, connect_timeout=2)
    conn.read_backs = 0
    conn.before_read_back = lambda: True
    try:
        yield conn
    finally:
        conn.close()


def test_a_row_that_vanished_before_the_read_back_is_inserted_on_retry(
    db_url: str, tampered: _TamperedReadBack
) -> None:
    row = make_encounter_row()
    with psycopg.connect(db_url, connect_timeout=2, autocommit=True) as other:
        state.record_encounter(other, state.EncounterEvent.from_row(row))

        def delete_it_once() -> bool:
            if tampered.read_backs == 1:
                other.execute("DELETE FROM encounters WHERE id = %s", [row["Id"]])
            return True

        tampered.before_read_back = delete_it_once

        assert state.record_encounter(tampered, state.EncounterEvent.from_row(row)) is True

    history = state.patient_history(tampered, "patient-1")
    pd.testing.assert_frame_equal(history.encounters, _frame([row], state.ENCOUNTER_COLUMNS))


def test_a_read_back_that_never_finds_the_row_gives_up_after_a_bounded_number_of_tries(
    db_url: str, tampered: _TamperedReadBack
) -> None:
    row = make_encounter_row()
    with psycopg.connect(db_url, connect_timeout=2, autocommit=True) as other:
        state.record_encounter(other, state.EncounterEvent.from_row(row))
    tampered.before_read_back = lambda: False

    with pytest.raises(RuntimeError, match="encounters"):
        state.record_encounter(tampered, state.EncounterEvent.from_row(row))

    assert 1 < tampered.read_backs <= 5
    # The failed attempt rolled back, so the connection is still usable.
    assert _record(tampered, "encounter", make_encounter_row(Id="encounter-2")) is True


# The per-patient event cap.


def _events(patient: str, count: int) -> list[state.AnyEvent]:
    return [
        state.MedicationEvent.from_row(make_medication_row(PATIENT=patient, CODE=f"drug-{i}"))
        for i in range(count)
    ]


def test_an_event_past_the_patient_cap_is_refused_and_not_stored(
    db_conn: psycopg.Connection[Any],
) -> None:
    first, second, third = _events("patient-1", 3)
    assert state.record_event(db_conn, first, max_patient_rows=2) is True
    assert state.record_event(db_conn, second, max_patient_rows=2) is True

    with pytest.raises(state.PatientEventLimitError, match=r"patient-1.*\b2\b"):
        state.record_event(db_conn, third, max_patient_rows=2)

    assert state.patient_history(db_conn, "patient-1").medications["CODE"].tolist() == [
        "drug-0",
        "drug-1",
    ]
    # The connection is still usable.
    assert state.record_event(db_conn, _events("patient-2", 1)[0], max_patient_rows=2) is True


def test_a_repost_at_the_cap_is_still_a_noop(db_conn: psycopg.Connection[Any]) -> None:
    """A resumed replay re-posts what it already sent; the cap must not turn that into a refusal."""
    first, second = _events("patient-1", 2)
    state.record_event(db_conn, first, max_patient_rows=2)
    state.record_event(db_conn, second, max_patient_rows=2)

    assert state.record_event(db_conn, second, max_patient_rows=2) is False


def test_the_cap_counts_every_clinical_event_kind_but_not_demographics(
    db_conn: psycopg.Connection[Any],
) -> None:
    encounter = state.EncounterEvent.from_row(make_encounter_row(ENCOUNTERCLASS="inpatient"))
    condition = state.ConditionEvent.from_row(make_condition_row())
    (medication,) = _events("patient-1", 1)
    demographics = state.PatientEvent.from_row(make_patient_row())
    state.record_event(db_conn, encounter, max_patient_rows=2)
    state.record_event(db_conn, condition, max_patient_rows=2)

    with pytest.raises(state.PatientEventLimitError):
        state.record_event(db_conn, medication, max_patient_rows=2)
    assert state.record_event(db_conn, demographics, max_patient_rows=2) is True


def test_no_cap_means_no_limit(db_conn: psycopg.Connection[Any]) -> None:
    for event in _events("patient-1", 5):
        assert state.record_event(db_conn, event) is True


# Bounded read-back, for scoring one discharge without loading a lifetime.


DISCHARGE_WINDOW = state.HistoryWindow(
    encounter_stop_from="2023-06-02T08:00:00Z",
    discharge="2024-06-05T08:00:00Z",
    discharge_date="2024-06-05",
)


def _stay(encounter_id: str, stop: str, start: str = "2023-01-01T08:00:00Z") -> dict[str, str]:
    return make_encounter_row(Id=encounter_id, ENCOUNTERCLASS="inpatient", START=start, STOP=stop)


def test_windowed_history_keeps_the_rows_a_discharge_can_read_and_no_others(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Each bound is probed a second, or a day, to either side.

    Encounters are read by STOP between the lookback floor and the
    discharge instant, both inclusive; an open stay has no STOP and is
    invisible to every feature. Medications are read when started at or
    before the discharge and stopped after it or never. Conditions are
    read when started on or before the discharge date, however long ago
    and resolved or not, because the comorbidity flags read resolved
    history.
    """
    _record(db_conn, "patient", make_patient_row())
    for encounter_id, stop in [
        ("e-before-floor", "2023-06-02T07:59:59Z"),
        ("e-on-floor", "2023-06-02T08:00:00Z"),
        ("e-scored", "2024-06-05T08:00:00Z"),
        ("e-after-discharge", "2024-06-05T08:00:01Z"),
        ("e-open", ""),
    ]:
        _record(db_conn, "encounter", _stay(encounter_id, stop))
    for code, start, stop in [
        ("m-stopped-at-discharge", "2024-01-01T08:00:00Z", "2024-06-05T08:00:00Z"),
        ("m-stopped-after", "2024-01-01T08:00:00Z", "2024-06-05T08:00:01Z"),
        ("m-open", "2024-01-01T08:00:00Z", ""),
        ("m-started-at-discharge", "2024-06-05T08:00:00Z", ""),
        ("m-started-after", "2024-06-05T08:00:01Z", ""),
    ]:
        _record(db_conn, "medication", make_medication_row(CODE=code, START=start, STOP=stop))
    for code, start, stop in [
        ("c-on-discharge-date", "2024-06-05", ""),
        ("c-day-after", "2024-06-06", ""),
        ("c-resolved-years-ago", "2015-01-01", "2015-02-01"),
    ]:
        _record(db_conn, "condition", make_condition_row(CODE=code, START=start, STOP=stop))

    history = state.patient_history(db_conn, "patient-1", DISCHARGE_WINDOW)

    assert history.patients["Id"].tolist() == ["patient-1"]
    assert history.encounters["Id"].tolist() == ["e-on-floor", "e-scored"]
    assert history.medications["CODE"].tolist() == [
        "m-open",
        "m-stopped-after",
        "m-started-at-discharge",
    ]
    assert history.conditions["CODE"].tolist() == ["c-resolved-years-ago", "c-on-discharge-date"]


def test_unwindowed_history_is_the_whole_record(db_conn: psycopg.Connection[Any]) -> None:
    _record(
        db_conn, "encounter", _stay("e-ancient", "2001-01-01T08:00:00Z", "2000-12-30T08:00:00Z")
    )
    _record(db_conn, "encounter", _stay("e-open", ""))
    _record(db_conn, "condition", make_condition_row(START="2030-01-01", STOP=""))

    history = state.patient_history(db_conn, "patient-1")

    assert history.encounters["Id"].tolist() == ["e-ancient", "e-open"]
    assert len(history.conditions) == 1
