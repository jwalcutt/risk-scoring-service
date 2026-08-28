"""Sanity tests for the Synthea CSV row builders.

The expected headers below are transcribed directly from the frozen
Synthea v4.0.0 CSV export, so a builder that drifts from the real schema
fails here before any cohort test can pass against a wrong shape.
"""

from pathlib import Path

import pandas as pd
import pytest

from factories import (
    CONDITION_DEFAULTS,
    ENCOUNTER_DEFAULTS,
    MEDICATION_DEFAULTS,
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
    ordered_events,
    write_rows_csv,
)

REAL_PATIENT_HEADER = (
    "Id,BIRTHDATE,DEATHDATE,SSN,DRIVERS,PASSPORT,PREFIX,FIRST,MIDDLE,LAST,SUFFIX,"
    "MAIDEN,MARITAL,RACE,ETHNICITY,GENDER,BIRTHPLACE,ADDRESS,CITY,STATE,COUNTY,"
    "FIPS,ZIP,LAT,LON,HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME"
)
REAL_ENCOUNTER_HEADER = (
    "Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,"
    "DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,"
    "REASONDESCRIPTION"
)
REAL_MEDICATION_HEADER = (
    "START,STOP,PATIENT,PAYER,ENCOUNTER,CODE,DESCRIPTION,BASE_COST,PAYER_COVERAGE,"
    "DISPENSES,TOTALCOST,REASONCODE,REASONDESCRIPTION"
)
REAL_CONDITION_HEADER = "START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION"


def test_patient_row_keys_match_real_header_in_order() -> None:
    row = make_patient_row()
    assert list(row.keys()) == REAL_PATIENT_HEADER.split(",")


def test_encounter_row_keys_match_real_header_in_order() -> None:
    row = make_encounter_row()
    assert list(row.keys()) == REAL_ENCOUNTER_HEADER.split(",")


def test_medication_row_keys_match_real_header_in_order() -> None:
    row = make_medication_row()
    assert list(row.keys()) == REAL_MEDICATION_HEADER.split(",")


def test_condition_row_keys_match_real_header_in_order() -> None:
    row = make_condition_row()
    assert list(row.keys()) == REAL_CONDITION_HEADER.split(",")


def test_overrides_are_applied() -> None:
    row = make_encounter_row(Id="e42", ENCOUNTERCLASS="inpatient")
    assert row["Id"] == "e42"
    assert row["ENCOUNTERCLASS"] == "inpatient"


def test_encounter_default_class_satisfies_no_inclusion_rule() -> None:
    assert make_encounter_row()["ENCOUNTERCLASS"] == "wellness"


def test_patient_default_is_living() -> None:
    assert make_patient_row()["DEATHDATE"] == ""


def test_unknown_override_key_raises() -> None:
    with pytest.raises(ValueError, match="ENCOUNTER_CLASS"):
        make_encounter_row(ENCOUNTER_CLASS="inpatient")


def test_write_rows_csv_round_trips_header_and_values(tmp_path: Path) -> None:
    path = tmp_path / "encounters.csv"
    write_rows_csv(path, [make_encounter_row(Id="e1"), make_encounter_row(Id="e2")])
    lines = path.read_text().splitlines()
    assert lines[0] == REAL_ENCOUNTER_HEADER
    assert lines[1].startswith("e1,")
    assert lines[2].startswith("e2,")


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
