"""Stream splicing, pure: segments from a config, and what each one owns.

From a splice instant the named population is the source of every event
and every label, and nothing before it changes. The rules these tests
pin:

- A config with no splices is one segment from the start, not rekeyed. Each
  splice opens a segment at midnight of its date; the previous segment ends
  there. Only populations other than the run's own are rekeyed, so a splice
  back to the starting population is the same people.
- A segment owns the events dated at or after its start and before the
  next splice (the last segment to the end of the stream), in stream order.
  An event at exactly the splice instant belongs to the incoming population;
  the outgoing population's event at that instant is dropped. The preload
  boundary for a segment is its start, so preload and segment are exact
  complements per population.
- Labels follow the population of origin: a segment owns the labels of the
  discharges it owns, judged by the discharge instant, whatever the label's
  readmission did after the splice.
- The spliced stream resumes across the splice from a cursor: the last event
  of the outgoing population leads to the first of the incoming.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from factories import make_encounter_row, make_patient_row, payload_frame
from risk_scoring import state
from risk_scoring.replay.config import ReplayConfig, Splice
from risk_scoring.replay.emission import due_events
from risk_scoring.replay.release import ScheduledLabel, label_schedule
from risk_scoring.replay.splice import (
    Segment,
    segment_events,
    segment_labels,
    segments,
    spliced_events,
    spliced_labels,
)
from risk_scoring.stream import StreamEvent

START = "2025-01-01T00:00:00Z"
SPLICE = "2025-04-01T00:00:00Z"
SECOND = "2025-09-01T00:00:00Z"


def _config(*splices: Splice) -> ReplayConfig:
    return ReplayConfig(
        population="baseline",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        acceleration=4,
        splices=splices,
    )


def _event(at: str, name: str, kind: str = "encounter") -> StreamEvent:
    return StreamEvent(at=at, kind=kind, row={"Id": name, "STOP": at})


def _label(discharged_at: str, name: str, label: int = 0) -> ScheduledLabel:
    due = f"{discharged_at[:4]}-{int(discharged_at[5:7]) + 1:02d}{discharged_at[7:]}"
    return ScheduledLabel(due_at=due, encounter_id=name, discharged_at=discharged_at, label=label)


# Segments


def test_no_splices_is_one_segment_from_the_start() -> None:
    assert segments(_config()) == [Segment("baseline", START, None, rekey=False)]


def test_each_splice_opens_a_segment_and_closes_the_previous_one() -> None:
    config = _config(
        Splice(date(2025, 4, 1), "care_protocol"), Splice(date(2025, 9, 1), "demographic_shift")
    )
    assert segments(config) == [
        Segment("baseline", START, SPLICE, rekey=False),
        Segment("care_protocol", SPLICE, SECOND, rekey=True),
        Segment("demographic_shift", SECOND, None, rekey=True),
    ]


def test_a_splice_back_to_the_starting_population_is_not_rekeyed() -> None:
    config = _config(
        Splice(date(2025, 4, 1), "care_protocol"), Splice(date(2025, 9, 1), "baseline")
    )
    assert [segment.rekey for segment in segments(config)] == [False, True, False]


def test_a_variant_is_rekeyed_in_every_segment_naming_it() -> None:
    config = _config(
        Splice(date(2025, 4, 1), "care_protocol"),
        Splice(date(2025, 6, 1), "baseline"),
        Splice(date(2025, 9, 1), "care_protocol"),
    )
    assert [segment.rekey for segment in segments(config)] == [False, True, False, True]


# What a segment owns


def test_events_before_the_splice_belong_to_the_outgoing_segment_only() -> None:
    outgoing = Segment("a", START, SPLICE, rekey=False)
    events = [
        _event("2024-12-31T23:59:59Z", "before-start"),
        _event(START, "at-start"),
        _event("2025-02-01T08:00:00Z", "mid"),
        _event("2025-03-31T23:59:59Z", "last-before"),
        _event(SPLICE, "at-splice"),
        _event("2025-05-01T08:00:00Z", "after"),
    ]
    assert [event.row["Id"] for event in segment_events(outgoing, events)] == [
        "at-start",
        "mid",
        "last-before",
    ]


def test_events_at_or_after_the_splice_belong_to_the_incoming_segment() -> None:
    incoming = Segment("b", SPLICE, None, rekey=True)
    events = [
        _event("2025-03-31T23:59:59Z", "before"),
        _event(SPLICE, "at-splice"),
        _event("2025-05-01T08:00:00Z", "after"),
        _event("2026-06-01T08:00:00Z", "past-the-end"),
    ]
    assert [event.row["Id"] for event in segment_events(incoming, events)] == [
        "at-splice",
        "after",
        "past-the-end",
    ]


def test_a_middle_segment_is_bounded_on_both_sides() -> None:
    middle = Segment("b", SPLICE, SECOND, rekey=True)
    events = [
        _event("2025-03-01T00:00:00Z", "before"),
        _event(SPLICE, "first"),
        _event("2025-08-31T23:59:59Z", "last"),
        _event(SECOND, "next-segment"),
    ]
    assert [event.row["Id"] for event in segment_events(middle, events)] == ["first", "last"]


def test_labels_follow_the_discharge_instant_with_the_same_boundary() -> None:
    outgoing = Segment("a", START, SPLICE, rekey=False)
    incoming = Segment("b", SPLICE, None, rekey=True)
    schedule = [
        _label("2024-12-01T08:00:00Z", "before-start"),
        _label("2025-03-31T23:59:59Z", "last-before"),
        _label(SPLICE, "at-splice"),
        _label("2025-05-01T08:00:00Z", "after"),
    ]
    assert [item.encounter_id for item in segment_labels(outgoing, schedule)] == ["last-before"]
    assert [item.encounter_id for item in segment_labels(incoming, schedule)] == [
        "at-splice",
        "after",
    ]


def test_a_label_keeps_its_origin_when_the_readmission_lies_past_the_splice() -> None:
    """The batch label over the outgoing export, though its readmission is never posted."""
    patients = payload_frame(
        [make_patient_row(Id="p", BIRTHDATE="1960-01-01")], state.PATIENT_COLUMNS
    )
    encounters = payload_frame(
        [
            make_encounter_row(
                Id="e-index",
                PATIENT="p",
                ENCOUNTERCLASS="inpatient",
                START="2025-03-20T08:00:00Z",
                STOP="2025-03-25T08:00:00Z",
            ),
            make_encounter_row(
                Id="e-readmit",
                PATIENT="p",
                ENCOUNTERCLASS="inpatient",
                START="2025-04-05T08:00:00Z",
                STOP="2025-04-08T08:00:00Z",
            ),
        ],
        state.ENCOUNTER_COLUMNS,
    )
    frames: dict[str, pd.DataFrame] = {"patients": patients, "encounters": encounters}
    outgoing = Segment("a", START, SPLICE, rekey=False)

    owned = segment_labels(outgoing, label_schedule(frames))

    assert [(item.encounter_id, item.label) for item in owned] == [("e-index", 1)]


# The spliced whole


def test_spliced_events_are_the_segments_in_order() -> None:
    a = [_event("2025-01-05T00:00:00Z", "a1"), _event("2025-03-05T00:00:00Z", "a2")]
    b = [_event(SPLICE, "b1"), _event("2025-05-05T00:00:00Z", "b2")]
    whole = spliced_events([a, b])
    assert [event.row["Id"] for event in whole] == ["a1", "a2", "b1", "b2"]
    assert [event.sort_key for event in whole] == sorted(event.sort_key for event in whole)


def test_spliced_labels_are_in_due_order() -> None:
    a = [_label("2025-01-05T00:00:00Z", "a1"), _label("2025-03-05T00:00:00Z", "a2")]
    b = [_label(SPLICE, "b1"), _label("2025-05-05T00:00:00Z", "b2")]
    whole = spliced_labels([a, b])
    assert [item.encounter_id for item in whole] == ["a1", "a2", "b1", "b2"]
    assert whole == sorted(whole)


def test_the_cursor_crosses_the_splice() -> None:
    a = [_event("2025-01-05T00:00:00Z", "a1"), _event("2025-03-05T00:00:00Z", "a2")]
    b = [_event(SPLICE, "b1"), _event("2025-05-05T00:00:00Z", "b2")]
    whole = spliced_events([a, b])

    due = due_events(whole, a[-1].sort_key, "2025-04-01T01:00:00Z")

    assert [event.row["Id"] for event in due] == ["b1"]
