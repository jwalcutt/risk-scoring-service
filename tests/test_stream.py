"""Tests for the ordered event stream and its payload projection.

Two properties matter here. Arrival ordering decides what a discharge can
see when it is scored: a medication or condition that was in effect during
a stay must reach the service before the discharge that ends it. Payload
projection decides what the service is willing to accept at all, so every
event this module builds is validated against the wire contract rather
than against a second copy of the column sets.
"""

from __future__ import annotations

import pandas as pd
from pydantic import TypeAdapter

from factories import (
    CONDITION_DEFAULTS,
    ENCOUNTER_DEFAULTS,
    MEDICATION_DEFAULTS,
    PATIENT_DEFAULTS,
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring import state
from risk_scoring.service.events import Event
from risk_scoring.stream import EVENT_FIELDS, build_stream, envelope, ordered_events

_EVENT = TypeAdapter(Event)


def _frames(
    encounters: list[dict[str, str]],
    medications: list[dict[str, str]],
    conditions: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(encounters or None, columns=list(ENCOUNTER_DEFAULTS)),
        pd.DataFrame(medications or None, columns=list(MEDICATION_DEFAULTS)),
        pd.DataFrame(conditions or None, columns=list(CONDITION_DEFAULTS)),
    )


def _population(
    patients: list[dict[str, str]],
    encounters: list[dict[str, str]],
    medications: list[dict[str, str]],
    conditions: list[dict[str, str]],
) -> dict[str, pd.DataFrame]:
    encounter_frame, medication_frame, condition_frame = _frames(
        encounters, medications, conditions
    )
    return {
        "patients": pd.DataFrame(patients or None, columns=list(PATIENT_DEFAULTS)),
        "encounters": encounter_frame,
        "medications": medication_frame,
        "conditions": condition_frame,
    }


# Arrival ordering


def test_encounter_arrives_at_its_stop() -> None:
    row = make_encounter_row(Id="e1", START="2024-01-01T08:00:00Z", STOP="2024-01-05T09:00:00Z")
    (event,) = ordered_events(*_frames([row], [], []))
    assert (event.kind, event.at) == ("encounter", "2024-01-05T09:00:00Z")
    assert event.row["Id"] == "e1"


def test_medication_arrives_at_its_start() -> None:
    row = make_medication_row(START="2024-01-02T10:00:00Z", STOP="2024-03-02T10:00:00Z")
    (event,) = ordered_events(*_frames([], [row], []))
    assert (event.kind, event.at) == ("medication", "2024-01-02T10:00:00Z")


def test_condition_arrives_at_midnight_of_its_date_only_start() -> None:
    row = make_condition_row(START="2024-01-02")
    (event,) = ordered_events(*_frames([], [], [row]))
    assert (event.kind, event.at) == ("condition", "2024-01-02T00:00:00Z")


def test_stream_is_ordered_by_arrival_instant() -> None:
    early = make_encounter_row(Id="early", STOP="2024-01-01T00:00:00Z")
    late = make_encounter_row(Id="late", STOP="2024-06-01T00:00:00Z")
    events = ordered_events(*_frames([late, early], [], []))
    assert [event.row["Id"] for event in events] == ["early", "late"]


def test_medications_and_conditions_precede_a_discharge_at_the_same_instant() -> None:
    """They were in effect during the stay, so the discharge must see them."""
    discharge = make_encounter_row(Id="e1", STOP="2024-01-02T00:00:00Z")
    medication = make_medication_row(START="2024-01-02T00:00:00Z")
    condition = make_condition_row(START="2024-01-02")
    events = ordered_events(*_frames([discharge], [medication], [condition]))
    assert [event.kind for event in events] == ["medication", "condition", "encounter"]


def test_rows_arriving_at_one_instant_are_ordered_deterministically() -> None:
    """Two prescriptions of one drug share an instant; the order must not vary."""
    first = make_medication_row(START="2024-01-02T00:00:00Z", STOP="2024-01-09T00:00:00Z")
    second = make_medication_row(START="2024-01-02T00:00:00Z", STOP="2024-02-09T00:00:00Z")
    forward = ordered_events(*_frames([], [first, second], []))
    reversed_input = ordered_events(*_frames([], [second, first], []))
    assert [event.row["STOP"] for event in forward] == [
        event.row["STOP"] for event in reversed_input
    ]


def test_empty_population_yields_no_events() -> None:
    assert ordered_events(*_frames([], [], [])) == []


def test_an_event_carries_every_source_column_as_a_string() -> None:
    """The tie-break sorts the whole row, so a projected row would reorder ties silently."""
    encounter = make_encounter_row(Id="e1")
    medication = make_medication_row(ENCOUNTER="e1")
    condition = make_condition_row(ENCOUNTER="e1")
    frames = _frames([encounter], [medication], [condition])
    by_kind = {event.kind: event.row for event in ordered_events(*frames)}
    assert list(by_kind["encounter"]) == list(frames[0].columns)
    assert list(by_kind["medication"]) == list(frames[1].columns)
    assert list(by_kind["condition"]) == list(frames[2].columns)
    assert by_kind["encounter"] == encounter
    assert all(isinstance(value, str) for row in by_kind.values() for value in row.values())


# Payload projection


def test_the_projected_field_sets_are_the_stored_column_sets() -> None:
    assert EVENT_FIELDS == {
        "patient": state.PATIENT_COLUMNS,
        "encounter": state.ENCOUNTER_COLUMNS,
        "medication": state.MEDICATION_COLUMNS,
        "condition": state.CONDITION_COLUMNS,
    }


def test_every_projected_payload_validates_against_the_wire_contract() -> None:
    """The service must accept every event this module builds."""
    stream = build_stream(
        _population(
            [make_patient_row(Id="p1")],
            [make_encounter_row(Id="e1", PATIENT="p1")],
            [make_medication_row(PATIENT="p1", ENCOUNTER="e1")],
            [make_condition_row(PATIENT="p1", ENCOUNTER="e1")],
        )
    )
    assert {event["event_type"] for event in stream} == set(EVENT_FIELDS)
    for event in stream:
        validated = _EVENT.validate_python(event)
        assert validated.event_type == event["event_type"]


def test_the_stream_leads_with_demographics() -> None:
    """A discharge that outran its patient is refused, not scored."""
    stream = build_stream(
        _population(
            [make_patient_row(Id="p1"), make_patient_row(Id="p2")],
            [make_encounter_row(Id="e1", PATIENT="p1"), make_encounter_row(Id="e2", PATIENT="p2")],
            [make_medication_row(PATIENT="p1", ENCOUNTER="e1")],
            [make_condition_row(PATIENT="p2", ENCOUNTER="e2")],
        )
    )
    kinds = [event["event_type"] for event in stream]
    assert kinds[:2] == ["patient", "patient"]
    assert "patient" not in kinds[2:]


def test_projected_payloads_carry_only_the_contract_fields() -> None:
    stream = build_stream(
        _population(
            [make_patient_row(Id="p1")],
            [make_encounter_row(Id="e1", PATIENT="p1")],
            [],
            [],
        )
    )
    for event in stream:
        assert tuple(event["payload"]) == EVENT_FIELDS[event["event_type"]]


def test_payload_values_are_verbatim_source_strings() -> None:
    """An open stay carries an empty STOP, not a filled-in one."""
    stream = build_stream(
        _population([make_patient_row(Id="p1")], [make_encounter_row(Id="e1", STOP="")], [], [])
    )
    (encounter,) = [event for event in stream if event["event_type"] == "encounter"]
    assert encounter["payload"]["STOP"] == ""
    assert all(isinstance(value, str) for value in encounter["payload"].values())


def test_the_stream_covers_every_row_exactly_once() -> None:
    frames = _population(
        [make_patient_row(Id="p1"), make_patient_row(Id="p2")],
        [make_encounter_row(Id="e1", PATIENT="p1"), make_encounter_row(Id="e2", PATIENT="p2")],
        [make_medication_row(PATIENT="p1", ENCOUNTER="e1")],
        [make_condition_row(PATIENT="p2", ENCOUNTER="e2")],
    )
    stream = build_stream(frames)
    assert len(stream) == sum(len(frame) for frame in frames.values())
    encounter_ids = [
        event["payload"]["Id"] for event in stream if event["event_type"] == "encounter"
    ]
    assert sorted(encounter_ids) == ["e1", "e2"]


def test_an_empty_population_yields_an_empty_stream() -> None:
    assert build_stream(_population([], [], [], [])) == []


# Envelopes


def test_an_envelope_projects_exactly_the_contract_fields_in_order() -> None:
    row = make_encounter_row(Id="e1", PATIENT="p1", STOP="")
    posted = envelope("encounter", row)
    assert posted["event_type"] == "encounter"
    assert tuple(posted["payload"]) == EVENT_FIELDS["encounter"]
    assert posted["payload"]["STOP"] == ""
    assert _EVENT.validate_python(posted).event_type == "encounter"


def test_build_stream_posts_the_same_envelope_the_harness_would() -> None:
    """One projection, so the batch driver and the replay post identical bytes."""
    encounter = make_encounter_row(Id="e1", PATIENT="p1")
    stream = build_stream(_population([make_patient_row(Id="p1")], [encounter], [], []))
    assert stream[1] == envelope("encounter", encounter)
