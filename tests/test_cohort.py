"""Tests for the cohort definition.

The scoring event is an inpatient discharge. Excluded: non-inpatient
encounters, encounters where the patient died in hospital (date-only
DEATHDATE within the stay, or the anomalous case of a death date before
admission), and patients under 18 at discharge. Age is computed at the
discharge date. Each rule gets fixture encounters exercising both sides
of its boundary.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from factories import make_encounter_row, make_patient_row, write_rows_csv
from risk_scoring.cohort import (
    COHORT_VERSION,
    build_cohort,
    filter_training_window,
    load_cohort,
    split_at_cutoff,
)


def frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


ADULT = make_patient_row(Id="p-adult", BIRTHDATE="1970-01-01")


def test_inpatient_discharge_is_included() -> None:
    encounters = frame([make_encounter_row(Id="e1", PATIENT="p-adult", ENCOUNTERCLASS="inpatient")])
    result = build_cohort(encounters, frame([ADULT]))
    assert list(result.frame["encounter_id"]) == ["e1"]


@pytest.mark.parametrize(
    "encounter_class",
    [
        "ambulatory",
        "emergency",
        "home",
        "hospice",
        "outpatient",
        "snf",
        "urgentcare",
        "virtual",
        "wellness",
    ],
)
def test_non_inpatient_classes_are_excluded(encounter_class: str) -> None:
    encounters = frame(
        [make_encounter_row(Id="e1", PATIENT="p-adult", ENCOUNTERCLASS=encounter_class)]
    )
    result = build_cohort(encounters, frame([ADULT]))
    assert result.frame.empty
    assert result.exclusions.non_inpatient == 1


def test_cohort_row_carries_ids_timestamps_and_age() -> None:
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START="2024-03-01T10:30:00Z",
                STOP="2024-03-05T16:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([ADULT]))
    row = result.frame.iloc[0]
    assert row["encounter_id"] == "e1"
    assert row["patient_id"] == "p-adult"
    assert row["start"] == pd.Timestamp("2024-03-01T10:30:00Z")
    assert row["stop"] == pd.Timestamp("2024-03-05T16:00:00Z")
    assert row["age_at_discharge"] == 54


def test_death_during_stay_is_excluded() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="1950-01-01", DEATHDATE="2024-01-02")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-04T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert result.frame.empty
    assert result.exclusions.in_hospital_death == 1


def test_death_on_discharge_date_is_excluded() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="1950-01-01", DEATHDATE="2024-01-04")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-04T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert result.frame.empty
    assert result.exclusions.in_hospital_death == 1


def test_earlier_encounters_of_later_deceased_patient_are_included() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="1950-01-01", DEATHDATE="2024-06-01")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-04T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert list(result.frame["encounter_id"]) == ["e1"]
    assert result.exclusions.in_hospital_death == 0


def test_death_before_admission_is_excluded_and_counted_as_anomaly() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="1950-01-01", DEATHDATE="2023-12-01")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-04T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert result.frame.empty
    assert result.exclusions.death_before_admission == 1
    assert result.exclusions.in_hospital_death == 0


def test_under_18_at_discharge_is_excluded() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="2006-06-15")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-06-10T08:00:00Z",
                STOP="2024-06-14T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert result.frame.empty
    assert result.exclusions.under_18 == 1


def test_18th_birthday_on_discharge_date_is_included() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="2006-06-15")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-06-10T08:00:00Z",
                STOP="2024-06-15T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert list(result.frame["encounter_id"]) == ["e1"]
    assert result.frame.iloc[0]["age_at_discharge"] == 18


def test_age_is_computed_at_discharge_not_admission() -> None:
    patient = make_patient_row(Id="p1", BIRTHDATE="1990-06-15")
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p1",
                ENCOUNTERCLASS="inpatient",
                START="2024-06-13T08:00:00Z",
                STOP="2024-06-16T08:00:00Z",
            )
        ]
    )
    result = build_cohort(encounters, frame([patient]))
    assert result.frame.iloc[0]["age_at_discharge"] == 34


def test_each_qualifying_encounter_yields_one_row() -> None:
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START="2024-01-01T08:00:00Z",
                STOP="2024-01-04T08:00:00Z",
            ),
            make_encounter_row(
                Id="e2",
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START="2024-02-01T08:00:00Z",
                STOP="2024-02-03T08:00:00Z",
            ),
            make_encounter_row(Id="e3", PATIENT="p-adult", ENCOUNTERCLASS="outpatient"),
        ]
    )
    result = build_cohort(encounters, frame([ADULT]))
    assert list(result.frame["encounter_id"]) == ["e1", "e2"]


def test_inpatient_encounter_with_unknown_patient_raises() -> None:
    encounters = frame(
        [make_encounter_row(Id="e-orphan", PATIENT="p-missing", ENCOUNTERCLASS="inpatient")]
    )
    with pytest.raises(ValueError, match="e-orphan"):
        build_cohort(encounters, frame([ADULT]))


def test_load_cohort_reads_csv_directory(tmp_path: Path) -> None:
    write_rows_csv(
        tmp_path / "patients.csv", [ADULT, make_patient_row(Id="p-child", BIRTHDATE="2020-01-01")]
    )
    write_rows_csv(
        tmp_path / "encounters.csv",
        [
            make_encounter_row(Id="e1", PATIENT="p-adult", ENCOUNTERCLASS="inpatient"),
            make_encounter_row(Id="e2", PATIENT="p-child", ENCOUNTERCLASS="inpatient"),
            make_encounter_row(Id="e3", PATIENT="p-adult", ENCOUNTERCLASS="emergency"),
        ],
    )
    result = load_cohort(tmp_path)
    assert list(result.frame["encounter_id"]) == ["e1"]
    assert result.exclusions.under_18 == 1
    assert result.exclusions.non_inpatient == 1


def test_cohort_version_is_declared() -> None:
    assert isinstance(COHORT_VERSION, str)
    assert COHORT_VERSION


# --- timestamp parsing ---

# Encounter START/STOP are "%Y-%m-%dT%H:%M:%SZ" and patient
# BIRTHDATE/DEATHDATE are "%Y-%m-%d" in the Synthea export. Parsing pins
# those formats rather than letting pandas infer them, so a column that
# stops conforming raises here instead of being reinterpreted by
# dateutil under a different reading of the same digits.

PARSE_FAILURE = r"doesn't match format|unconverted data remains"

AMBIGUOUS_MINUTE_START = "2007-02-05T20:07:18Z"
AMBIGUOUS_MINUTE_STOP = "2007-02-09T20:07:18Z"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("START", "2024-03-01 10:30:00"),
        ("START", "03/01/2024"),
        ("STOP", "2024-03-05 16:00:00"),
        ("STOP", "05/03/2024"),
    ],
)
def test_non_conforming_encounter_timestamp_raises(column: str, value: str) -> None:
    encounters = frame(
        [
            make_encounter_row(
                Id="e1", PATIENT="p-adult", ENCOUNTERCLASS="inpatient", **{column: value}
            )
        ]
    )
    with pytest.raises(ValueError, match=PARSE_FAILURE):
        build_cohort(encounters, frame([ADULT]))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("BIRTHDATE", "1970-01-01T00:00:00Z"),
        ("BIRTHDATE", "01/01/1970"),
        ("DEATHDATE", "2024-03-03T00:00:00Z"),
        ("DEATHDATE", "03/03/2024"),
    ],
)
def test_non_conforming_patient_date_raises(column: str, value: str) -> None:
    encounters = frame([make_encounter_row(Id="e1", PATIENT="p-adult", ENCOUNTERCLASS="inpatient")])
    patient = make_patient_row(Id="p-adult", **{"BIRTHDATE": "1970-01-01", column: value})
    with pytest.raises(ValueError, match=PARSE_FAILURE):
        build_cohort(encounters, frame([patient]))


def test_conforming_values_parse_to_the_declared_instants() -> None:
    """A timestamp whose format pandas cannot infer still parses exactly."""
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START=AMBIGUOUS_MINUTE_START,
                STOP=AMBIGUOUS_MINUTE_STOP,
            )
        ]
    )
    row = build_cohort(encounters, frame([ADULT])).frame.iloc[0]
    assert row["start"] == pd.Timestamp(AMBIGUOUS_MINUTE_START)
    assert row["stop"] == pd.Timestamp(AMBIGUOUS_MINUTE_STOP)
    assert row["age_at_discharge"] == 37


def test_parsing_never_falls_back_to_dateutil() -> None:
    """No column is parsed element-by-element under a guessed format."""
    encounters = frame(
        [
            make_encounter_row(
                Id="e1",
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START=AMBIGUOUS_MINUTE_START,
                STOP=AMBIGUOUS_MINUTE_STOP,
            ),
            make_encounter_row(
                Id="e2",
                PATIENT="p-deceased",
                ENCOUNTERCLASS="inpatient",
                START="2007-03-05T20:07:18Z",
                STOP="2007-03-09T20:07:18Z",
            ),
        ]
    )
    deceased = make_patient_row(Id="p-deceased", BIRTHDATE="1960-05-04", DEATHDATE="2007-03-07")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = build_cohort(encounters, frame([ADULT, deceased]))
    assert list(result.frame["encounter_id"]) == ["e1"]
    assert result.exclusions.in_hospital_death == 1


def test_empty_death_date_still_reads_as_missing() -> None:
    encounters = frame([make_encounter_row(Id="e1", PATIENT="p-adult", ENCOUNTERCLASS="inpatient")])
    result = build_cohort(encounters, frame([make_patient_row(Id="p-adult", DEATHDATE="")]))
    assert list(result.frame["encounter_id"]) == ["e1"]
    assert result.exclusions.in_hospital_death == 0


CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")


def cohort_at(stops: dict[str, str]) -> pd.DataFrame:
    """A cohort frame of adult inpatient discharges at the given STOPs."""
    encounters = frame(
        [
            make_encounter_row(
                Id=encounter_id,
                PATIENT="p-adult",
                ENCOUNTERCLASS="inpatient",
                START=stop,
                STOP=stop,
            )
            for encounter_id, stop in stops.items()
        ]
    )
    return build_cohort(encounters, frame([ADULT])).frame


def test_a_discharge_at_the_cutoff_instant_is_held_out() -> None:
    cohort = cohort_at(
        {
            "before": "2024-12-31T23:59:59Z",
            "at": "2025-01-01T00:00:00Z",
            "after": "2025-01-01T00:00:01Z",
        }
    )
    split = split_at_cutoff(cohort, CUTOFF)
    assert list(split.before["encounter_id"]) == ["before"]
    assert list(split.at_or_after["encounter_id"]) == ["at", "after"]


def test_the_two_halves_partition_the_cohort() -> None:
    cohort = cohort_at(
        {
            "e1": "2023-05-05T00:00:00Z",
            "e2": "2024-12-31T23:59:59Z",
            "e3": "2025-01-01T00:00:00Z",
            "e4": "2026-02-02T00:00:00Z",
        }
    )
    split = split_at_cutoff(cohort, CUTOFF)
    before = set(split.before["encounter_id"])
    at_or_after = set(split.at_or_after["encounter_id"])
    assert before.isdisjoint(at_or_after)
    assert before | at_or_after == set(cohort["encounter_id"])
    assert len(split.before) + len(split.at_or_after) == len(cohort)


def test_the_training_half_is_what_the_training_window_keeps() -> None:
    cohort = cohort_at(
        {
            "e1": "2023-05-05T00:00:00Z",
            "e2": "2025-01-01T00:00:00Z",
            "e3": "2026-02-02T00:00:00Z",
        }
    )
    pd.testing.assert_frame_equal(
        split_at_cutoff(cohort, CUTOFF).before, filter_training_window(cohort, CUTOFF)
    )


def test_both_halves_are_reindexed_from_zero() -> None:
    cohort = cohort_at(
        {
            "e1": "2023-05-05T00:00:00Z",
            "e2": "2025-06-06T00:00:00Z",
            "e3": "2024-05-05T00:00:00Z",
        }
    )
    split = split_at_cutoff(cohort, CUTOFF)
    assert list(split.before.index) == [0, 1]
    assert list(split.at_or_after.index) == [0]


def test_splitting_an_empty_cohort_gives_two_empty_halves() -> None:
    cohort = cohort_at({"e1": "2023-05-05T00:00:00Z"}).iloc[:0]
    split = split_at_cutoff(cohort, CUTOFF)
    assert split.before.empty
    assert split.at_or_after.empty
    assert list(split.before.columns) == list(cohort.columns)
    assert list(split.at_or_after.columns) == list(cohort.columns)
