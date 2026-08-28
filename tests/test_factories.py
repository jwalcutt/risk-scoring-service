"""Sanity tests for the Synthea CSV row builders.

The expected headers below are transcribed directly from the frozen
Synthea v4.0.0 CSV export, so a builder that drifts from the real schema
fails here before any cohort test can pass against a wrong shape.
"""

from pathlib import Path

import pytest

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
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
