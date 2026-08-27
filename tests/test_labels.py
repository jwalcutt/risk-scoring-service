"""Tests for the 30-day readmission label module.

Labels are computed per cohort row (an adult inpatient discharge) from the
raw encounters frame. The rules these tests pin:

- A readmission is any inpatient encounter for the same patient whose
  START is strictly after the index discharge's STOP and at or before
  STOP plus 30 days; equality at exactly 30 days counts.
- An inpatient encounter whose START is at or before the index STOP is a
  continuation or transfer of the same episode, never a readmission.
- Candidate readmissions are judged on the raw encounters frame, not the
  cohort, so a readmission stay that ends in death still labels the index
  discharge 1.
- Death within the window without a qualifying readmission labels 0: the
  label is readmission, not death, and the module never reads patient
  records.
- Output is one row per cohort row, in cohort order, with integer labels.
"""

import pandas as pd

from factories import ENCOUNTER_DEFAULTS, make_encounter_row
from risk_scoring.labels import LABEL_COLUMNS, LABEL_VERSION, build_labels

SCORE_START = "2024-06-01T08:00:00Z"
SCORE_STOP = "2024-06-05T08:00:00Z"


def make_cohort_row(
    encounter_id: str = "e-score",
    patient_id: str = "p1",
    start: str = SCORE_START,
    stop: str = SCORE_STOP,
    age: int = 54,
) -> dict[str, object]:
    return {
        "encounter_id": encounter_id,
        "patient_id": patient_id,
        "start": pd.Timestamp(start),
        "stop": pd.Timestamp(stop),
        "age_at_discharge": age,
    }


def encounter_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(ENCOUNTER_DEFAULTS))
    return pd.DataFrame(rows)


def labels_for(
    cohort_rows: list[dict[str, object]], encounters: list[dict[str, str]]
) -> pd.DataFrame:
    return build_labels(cohort=pd.DataFrame(cohort_rows), encounters=encounter_frame(encounters))


def single_label(encounters: list[dict[str, str]]) -> int:
    index_encounter = make_encounter_row(
        Id="e-score",
        PATIENT="p1",
        ENCOUNTERCLASS="inpatient",
        START=SCORE_START,
        STOP=SCORE_STOP,
    )
    result = labels_for([make_cohort_row()], [index_encounter, *encounters])
    assert len(result) == 1
    return int(result["label"].iloc[0])


def later_stay(
    start: str,
    stop: str,
    encounter_id: str = "e-later",
    patient: str = "p1",
    encounter_class: str = "inpatient",
) -> dict[str, str]:
    return make_encounter_row(
        Id=encounter_id,
        PATIENT=patient,
        ENCOUNTERCLASS=encounter_class,
        START=start,
        STOP=stop,
    )


# --- output contract ---


def test_label_version_is_declared() -> None:
    assert isinstance(LABEL_VERSION, str)
    assert LABEL_VERSION


def test_output_columns_match_declared_label_columns() -> None:
    result = labels_for([make_cohort_row()], [])
    assert tuple(result.columns) == LABEL_COLUMNS
    assert result["label"].dtype == "int64"


# --- window boundaries ---


def test_inpatient_admission_shortly_after_discharge_labels_one() -> None:
    assert single_label([later_stay("2024-06-05T08:00:01Z", "2024-06-08T08:00:00Z")]) == 1


def test_inpatient_admission_at_exactly_thirty_days_after_discharge_labels_one() -> None:
    assert single_label([later_stay("2024-07-05T08:00:00Z", "2024-07-08T08:00:00Z")]) == 1


def test_inpatient_admission_one_second_past_thirty_days_labels_zero() -> None:
    assert single_label([later_stay("2024-07-05T08:00:01Z", "2024-07-08T08:00:00Z")]) == 0


# --- continuation rule ---


def test_admission_starting_exactly_at_discharge_stop_is_a_continuation_not_a_readmission() -> None:
    assert single_label([later_stay(SCORE_STOP, "2024-06-08T08:00:00Z")]) == 0


def test_admission_starting_before_discharge_stop_is_a_continuation_not_a_readmission() -> None:
    assert single_label([later_stay("2024-06-04T08:00:00Z", "2024-06-08T08:00:00Z")]) == 0


# --- candidate eligibility ---


def test_discharge_with_no_subsequent_encounter_labels_zero() -> None:
    assert single_label([]) == 0


def test_readmission_of_a_different_patient_does_not_count() -> None:
    assert (
        single_label([later_stay("2024-06-10T08:00:00Z", "2024-06-12T08:00:00Z", patient="p2")])
        == 0
    )


def test_non_inpatient_encounter_within_thirty_days_does_not_count() -> None:
    assert (
        single_label(
            [
                later_stay(
                    "2024-06-10T08:00:00Z",
                    "2024-06-10T09:00:00Z",
                    encounter_class="emergency",
                )
            ]
        )
        == 0
    )


def test_death_within_thirty_days_without_readmission_labels_zero() -> None:
    death_certification = later_stay(
        "2024-06-20T08:00:00Z", "2024-06-20T09:00:00Z", encounter_class="ambulatory"
    )
    assert single_label([death_certification]) == 0


def test_readmission_stay_ending_in_death_still_labels_the_index_discharge_one() -> None:
    # The candidate stay never becomes a cohort row (in-hospital death), but the
    # index discharge was still followed by an inpatient admission within 30 days.
    assert single_label([later_stay("2024-06-10T08:00:00Z", "2024-06-12T08:00:00Z")]) == 1


# --- alignment ---


def test_labels_align_one_row_per_cohort_row_in_cohort_order() -> None:
    stay_a = make_encounter_row(
        Id="e-a", PATIENT="p1", ENCOUNTERCLASS="inpatient", START=SCORE_START, STOP=SCORE_STOP
    )
    stay_b = make_encounter_row(
        Id="e-b",
        PATIENT="p1",
        ENCOUNTERCLASS="inpatient",
        START="2024-06-20T08:00:00Z",
        STOP="2024-06-25T08:00:00Z",
    )
    other_patient_stay = make_encounter_row(
        Id="e-c",
        PATIENT="p2",
        ENCOUNTERCLASS="inpatient",
        START="2024-06-02T08:00:00Z",
        STOP="2024-06-03T08:00:00Z",
    )
    cohort_rows = [
        make_cohort_row(encounter_id="e-a"),
        make_cohort_row(
            encounter_id="e-b", start="2024-06-20T08:00:00Z", stop="2024-06-25T08:00:00Z"
        ),
        make_cohort_row(
            encounter_id="e-c",
            patient_id="p2",
            start="2024-06-02T08:00:00Z",
            stop="2024-06-03T08:00:00Z",
        ),
    ]
    result = labels_for(cohort_rows, [stay_a, stay_b, other_patient_stay])

    assert list(result["encounter_id"]) == ["e-a", "e-b", "e-c"]
    # Stay B is both A's readmission and its own index row with no later stay.
    assert list(result["label"]) == [1, 0, 0]
