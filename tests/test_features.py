"""Tests for the shared feature pipeline.

Features are computed per cohort row (an adult inpatient discharge) as of
that row's discharge timestamp, from hand-built patient histories. No
feature may see data recorded after the scoring discharge. The rules
these tests pin:

- Prior-event windows span 180 days ending at discharge, inclusive at
  the far edge: an event exactly 180 days old still counts. Prior
  encounters are dated by their STOP, so a stay still open at scoring
  time is invisible.
- Days since previous discharge runs from the previous inpatient STOP to
  the current admission START, floored at 0 for overlapping stays and
  capped at 365; a patient with no prior discharge gets the cap value as
  sentinel.
- A medication is active at discharge when its START is at or before the
  discharge instant and its STOP is empty or strictly after it, so a
  prescription stopping exactly at discharge is inactive.
- Condition dates are date-only, so activity is judged against the
  discharge date: START on the discharge date counts as active, STOP on
  the discharge date counts as ended.
- The active-condition count includes only SNOMED "(disorder)" entries;
  findings and situations (employment, stress, medication review) are
  administrative noise, not comorbidity.
- Comorbidity flags cover seven categories (CHF, chronic pulmonary,
  dementia, diabetes, malignancy, MI, renal disease) from curated code
  lists. Flags are history-based: a resolved qualifying condition still
  flags. The history-of-MI situation code 399211009 flags MI; suspected
  cancer situation codes do not flag malignancy.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from factories import (
    CONDITION_DEFAULTS,
    ENCOUNTER_DEFAULTS,
    MEDICATION_DEFAULTS,
    make_condition_row,
    make_encounter_row,
    make_medication_row,
)
from risk_scoring.features import FEATURE_COLUMNS, FEATURE_VERSION, build_features

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


def frame(rows: list[dict[str, str]], defaults: dict[str, str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(defaults))
    return pd.DataFrame(rows)


def features_for(
    cohort_rows: list[dict[str, object]],
    encounters: list[dict[str, str]] | None = None,
    medications: list[dict[str, str]] | None = None,
    conditions: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    return build_features(
        cohort=pd.DataFrame(cohort_rows),
        encounters=frame(encounters or [], ENCOUNTER_DEFAULTS),
        medications=frame(medications or [], MEDICATION_DEFAULTS),
        conditions=frame(conditions or [], CONDITION_DEFAULTS),
    )


def single(
    encounters: list[dict[str, str]] | None = None,
    medications: list[dict[str, str]] | None = None,
    conditions: list[dict[str, str]] | None = None,
    *,
    start: str = SCORE_START,
    stop: str = SCORE_STOP,
    age: int = 54,
) -> "pd.Series[object]":
    cohort_rows = [make_cohort_row(start=start, stop=stop, age=age)]
    result = features_for(cohort_rows, encounters, medications, conditions)
    assert len(result) == 1
    return result.iloc[0]


def iso_before(timestamp: str, delta: timedelta) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (parsed - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def prior_stay(
    stop: str, start: str | None = None, encounter_id: str = "e-prior"
) -> dict[str, str]:
    if start is None:
        start = iso_before(stop, timedelta(days=2))
    return make_encounter_row(
        Id=encounter_id, PATIENT="p1", ENCOUNTERCLASS="inpatient", START=start, STOP=stop
    )


FLAG_COLUMNS = (
    "flag_chf",
    "flag_chronic_pulmonary",
    "flag_dementia",
    "flag_diabetes",
    "flag_malignancy",
    "flag_mi",
    "flag_renal_disease",
)


# --- output contract ---


def test_feature_version_is_declared() -> None:
    assert isinstance(FEATURE_VERSION, str)
    assert FEATURE_VERSION


def test_output_columns_match_declared_feature_columns() -> None:
    result = features_for([make_cohort_row()])
    assert tuple(result.columns) == FEATURE_COLUMNS


def test_one_row_per_cohort_row_in_cohort_order() -> None:
    rows = [
        make_cohort_row(encounter_id="e1", patient_id="p1"),
        make_cohort_row(encounter_id="e2", patient_id="p2"),
    ]
    result = features_for(rows)
    assert list(result["encounter_id"]) == ["e1", "e2"]
    assert list(result["patient_id"]) == ["p1", "p2"]


def test_age_at_discharge_passes_through_from_cohort() -> None:
    assert single(age=71)["age_at_discharge"] == 71


def test_los_days_from_admission_to_discharge() -> None:
    row = single(start="2024-06-01T08:00:00Z", stop="2024-06-03T20:00:00Z")
    assert row["los_days"] == 2.5


# --- prior 180-day inpatient count ---


def test_prior_inpatient_stays_within_180_days_count() -> None:
    row = single(
        encounters=[
            prior_stay("2024-05-01T08:00:00Z", encounter_id="e-a"),
            prior_stay("2024-02-01T08:00:00Z", encounter_id="e-b"),
            prior_stay("2023-06-01T08:00:00Z", encounter_id="e-old"),
        ]
    )
    assert row["prior_inpatient_180d"] == 2


def test_prior_stay_exactly_180_days_before_discharge_counts() -> None:
    row = single(encounters=[prior_stay("2023-12-08T08:00:00Z")])
    assert row["prior_inpatient_180d"] == 1


def test_prior_stay_older_than_180_days_does_not_count() -> None:
    row = single(encounters=[prior_stay("2023-12-08T07:59:59Z")])
    assert row["prior_inpatient_180d"] == 0


def test_scoring_encounter_does_not_count_itself() -> None:
    scoring = make_encounter_row(
        Id="e-score",
        PATIENT="p1",
        ENCOUNTERCLASS="inpatient",
        START=SCORE_START,
        STOP=SCORE_STOP,
    )
    row = single(encounters=[scoring])
    assert row["prior_inpatient_180d"] == 0


def test_stay_still_open_at_discharge_is_invisible() -> None:
    overlapping = prior_stay("2024-06-20T08:00:00Z", start="2024-05-25T08:00:00Z")
    row = single(encounters=[overlapping])
    assert row["prior_inpatient_180d"] == 0
    assert row["days_since_prev_discharge"] == 365.0


def test_prior_non_inpatient_encounters_do_not_count_as_inpatient() -> None:
    ambulatory = make_encounter_row(
        Id="e-amb",
        PATIENT="p1",
        ENCOUNTERCLASS="ambulatory",
        START="2024-05-01T08:00:00Z",
        STOP="2024-05-01T09:00:00Z",
    )
    row = single(encounters=[ambulatory])
    assert row["prior_inpatient_180d"] == 0


# --- days since previous discharge ---


def test_days_since_prev_discharge_runs_from_prior_stop_to_admission() -> None:
    row = single(encounters=[prior_stay("2024-05-22T08:00:00Z")])
    assert row["days_since_prev_discharge"] == 10.0


def test_days_since_prev_discharge_uses_most_recent_prior_stop() -> None:
    row = single(
        encounters=[
            prior_stay("2024-04-01T08:00:00Z", encounter_id="e-a"),
            prior_stay("2024-05-22T08:00:00Z", encounter_id="e-b"),
        ]
    )
    assert row["days_since_prev_discharge"] == 10.0


def test_days_since_prev_discharge_caps_at_365() -> None:
    row = single(encounters=[prior_stay("2023-04-28T08:00:00Z")])
    assert row["days_since_prev_discharge"] == 365.0


def test_no_prior_discharge_gets_cap_value_as_sentinel() -> None:
    assert single()["days_since_prev_discharge"] == 365.0


def test_prior_stop_after_admission_floors_at_zero() -> None:
    overlapping = prior_stay("2024-06-03T08:00:00Z", start="2024-05-20T08:00:00Z")
    row = single(encounters=[overlapping])
    assert row["days_since_prev_discharge"] == 0.0


# --- prior 180-day ED visits ---


def ed_visit(stop: str, encounter_class: str = "emergency") -> dict[str, str]:
    start = iso_before(stop, timedelta(hours=4))
    return make_encounter_row(
        Id=f"e-ed-{stop}", PATIENT="p1", ENCOUNTERCLASS=encounter_class, START=start, STOP=stop
    )


def test_ed_visits_within_180_days_count() -> None:
    row = single(
        encounters=[
            ed_visit("2024-05-15T12:00:00Z"),
            ed_visit("2024-01-15T12:00:00Z"),
            ed_visit("2023-06-15T12:00:00Z"),
        ]
    )
    assert row["prior_ed_180d"] == 2


def test_ed_visit_exactly_180_days_before_discharge_counts() -> None:
    row = single(encounters=[ed_visit("2023-12-08T08:00:00Z")])
    assert row["prior_ed_180d"] == 1


def test_urgentcare_is_not_an_ed_visit() -> None:
    row = single(encounters=[ed_visit("2024-05-15T12:00:00Z", encounter_class="urgentcare")])
    assert row["prior_ed_180d"] == 0


def test_ed_visit_after_discharge_is_invisible() -> None:
    row = single(encounters=[ed_visit("2024-06-10T12:00:00Z")])
    assert row["prior_ed_180d"] == 0


# --- active medication count ---


def test_medications_active_at_discharge_count() -> None:
    row = single(
        medications=[
            make_medication_row(PATIENT="p1", START="2024-05-01T08:00:00Z", STOP=""),
            make_medication_row(
                PATIENT="p1", START="2024-01-15T08:00:00Z", STOP="2024-08-01T08:00:00Z"
            ),
            make_medication_row(
                PATIENT="p1", START="2024-01-01T08:00:00Z", STOP="2024-02-01T08:00:00Z"
            ),
        ]
    )
    assert row["active_medication_count"] == 2


def test_medication_started_after_discharge_is_invisible() -> None:
    row = single(
        medications=[make_medication_row(PATIENT="p1", START="2024-06-10T08:00:00Z", STOP="")]
    )
    assert row["active_medication_count"] == 0


def test_medication_stopping_exactly_at_discharge_is_not_active() -> None:
    row = single(
        medications=[
            make_medication_row(PATIENT="p1", START="2024-05-01T08:00:00Z", STOP=SCORE_STOP)
        ]
    )
    assert row["active_medication_count"] == 0


def test_medication_starting_exactly_at_discharge_is_active() -> None:
    row = single(medications=[make_medication_row(PATIENT="p1", START=SCORE_STOP, STOP="")])
    assert row["active_medication_count"] == 1


# --- active disorder count ---


def test_active_disorder_conditions_count() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2024-01-01",
                STOP="",
                CODE="59621000",
                DESCRIPTION="Essential hypertension (disorder)",
            ),
            make_condition_row(
                PATIENT="p1",
                START="2024-01-01",
                STOP="2024-03-01",
                CODE="444814009",
                DESCRIPTION="Viral sinusitis (disorder)",
            ),
        ]
    )
    assert row["active_disorder_count"] == 1


@pytest.mark.parametrize(
    ("code", "description"),
    [
        ("160903007", "Full-time employment (finding)"),
        ("314529007", "Medication review due (situation)"),
        ("73595000", "Stress (finding)"),
    ],
)
def test_findings_and_situations_do_not_count_as_disorders(code: str, description: str) -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1", START="2024-01-01", STOP="", CODE=code, DESCRIPTION=description
            )
        ]
    )
    assert row["active_disorder_count"] == 0


def test_condition_starting_on_discharge_date_counts() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2024-06-05",
                STOP="",
                CODE="59621000",
                DESCRIPTION="Essential hypertension (disorder)",
            )
        ]
    )
    assert row["active_disorder_count"] == 1


def test_condition_stopping_on_discharge_date_is_not_active() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2024-01-01",
                STOP="2024-06-05",
                CODE="59621000",
                DESCRIPTION="Essential hypertension (disorder)",
            )
        ]
    )
    assert row["active_disorder_count"] == 0


# --- comorbidity flags ---


@pytest.mark.parametrize(
    ("code", "description", "flag"),
    [
        ("88805009", "Chronic congestive heart failure (disorder)", "flag_chf"),
        ("87433001", "Pulmonary emphysema (disorder)", "flag_chronic_pulmonary"),
        ("26929004", "Alzheimer's disease (disorder)", "flag_dementia"),
        ("44054006", "Diabetes mellitus type 2 (disorder)", "flag_diabetes"),
        ("254837009", "Malignant neoplasm of breast (disorder)", "flag_malignancy"),
        ("22298006", "Myocardial infarction (disorder)", "flag_mi"),
        ("431855005", "Chronic kidney disease stage 1 (disorder)", "flag_renal_disease"),
    ],
)
def test_each_flag_set_by_representative_code(code: str, description: str, flag: str) -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1", START="2020-01-01", STOP="", CODE=code, DESCRIPTION=description
            )
        ]
    )
    assert row[flag] == 1
    assert sum(int(row[column]) for column in FLAG_COLUMNS) == 1


def test_resolved_qualifying_condition_still_flags() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2010-01-01",
                STOP="2010-02-01",
                CODE="22298006",
                DESCRIPTION="Myocardial infarction (disorder)",
            )
        ]
    )
    assert row["flag_mi"] == 1
    assert row["active_disorder_count"] == 0


def test_history_of_mi_situation_code_flags_mi() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2010-01-01",
                STOP="",
                CODE="399211009",
                DESCRIPTION="History of myocardial infarction (situation)",
            )
        ]
    )
    assert row["flag_mi"] == 1
    assert row["active_disorder_count"] == 0


def test_suspected_cancer_code_does_not_flag_malignancy() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2020-01-01",
                STOP="",
                CODE="315268008",
                DESCRIPTION="Suspected prostate cancer (situation)",
            )
        ]
    )
    assert row["flag_malignancy"] == 0


def test_qualifying_condition_recorded_after_discharge_does_not_flag() -> None:
    row = single(
        conditions=[
            make_condition_row(
                PATIENT="p1",
                START="2024-07-01",
                STOP="",
                CODE="44054006",
                DESCRIPTION="Diabetes mellitus type 2 (disorder)",
            )
        ]
    )
    assert row["flag_diabetes"] == 0


# --- patients with no history, and cross-patient isolation ---


def test_no_history_patient_gets_zeros_sentinel_and_no_missing_values() -> None:
    row = single()
    assert row["prior_inpatient_180d"] == 0
    assert row["prior_ed_180d"] == 0
    assert row["active_medication_count"] == 0
    assert row["active_disorder_count"] == 0
    assert row["days_since_prev_discharge"] == 365.0
    assert all(row[column] == 0 for column in FLAG_COLUMNS)
    assert not row.isna().any()


def test_history_of_other_patients_is_ignored() -> None:
    other_stay = make_encounter_row(
        Id="e-other",
        PATIENT="p2",
        ENCOUNTERCLASS="inpatient",
        START="2024-05-20T08:00:00Z",
        STOP="2024-05-22T08:00:00Z",
    )
    row = single(
        encounters=[other_stay],
        medications=[make_medication_row(PATIENT="p2", START="2024-05-01T08:00:00Z", STOP="")],
        conditions=[
            make_condition_row(
                PATIENT="p2",
                START="2020-01-01",
                STOP="",
                CODE="44054006",
                DESCRIPTION="Diabetes mellitus type 2 (disorder)",
            )
        ],
    )
    assert row["prior_inpatient_180d"] == 0
    assert row["active_medication_count"] == 0
    assert row["active_disorder_count"] == 0
    assert row["flag_diabetes"] == 0
    assert row["days_since_prev_discharge"] == 365.0
