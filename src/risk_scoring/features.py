"""Shared feature pipeline for 30-day readmission scoring.

Features are computed per cohort row (an adult inpatient discharge) as
of that row's discharge timestamp, from raw Synthea CSV frames. The same
module serves training and scoring, so no feature may read anything
recorded after the scoring discharge. ``FEATURE_VERSION`` is logged with
every prediction so any scored row can be traced to the exact feature
definitions that produced it.

Judgment calls this module fixes:

- Prior-event windows span 180 days ending at the discharge instant,
  inclusive at the far edge. Prior encounters are dated by their STOP,
  so a stay still open at scoring time is invisible; the scoring
  encounter never counts itself.
- Days since previous discharge runs from the most recent prior
  inpatient STOP (at or before the scoring discharge) to the current
  admission START, floored at 0 for overlapping stays and capped at
  ``DAYS_SINCE_PREV_DISCHARGE_CAP``. A patient with no prior discharge
  gets the cap value as sentinel.
- ED visits are ``ENCOUNTERCLASS == "emergency"`` only; urgent care is
  a distinct class and is not counted.
- A medication is active at discharge when its START is at or before
  the discharge instant and its STOP is empty or strictly after it.
  DISPENSES and TOTALCOST are deliberately unused: they accumulate over
  the prescription's full lifespan and would leak the future.
- Condition dates are date-only, so condition activity is judged
  against the discharge date: START on the discharge date counts as
  active, STOP on the discharge date counts as ended.
- The active-condition count includes only entries whose description
  carries the SNOMED "(disorder)" suffix. Findings and situations
  (employment status, stress, medication review) are administrative
  noise, not comorbidity.
- Comorbidity flags cover seven categories: congestive heart failure,
  chronic pulmonary disease, dementia, diabetes, malignancy, myocardial
  infarction, and chronic renal disease. Flags are history-based: any
  qualifying condition recorded on or before the discharge date sets
  the flag, resolved or not. The code lists were curated by a keyword
  screen over the full condition-code inventory of the three frozen
  generated populations, with these choices: the history-of-MI
  situation code 399211009 qualifies; suspected-cancer situation codes
  do not; malignancy means malignant neoplasm (including leukemia and
  lymphoma) and excludes in-situ and uncertain-behavior neoplasms; an
  ICD10 code starting with "C" is malignant by that system's structure;
  renal disease is chronic kidney disease only, excluding acute
  postoperative renal failure.
- Timestamp columns are parsed under the exact formats the Synthea
  export writes: encounter and medication timestamps as
  ``%Y-%m-%dT%H:%M:%SZ``, condition dates as ``%Y-%m-%d``. Pandas would
  otherwise guess a format from the first element and fall back to
  per-element dateutil parsing when the guess fails, which is how the
  same digits get read two ways. A non-conforming value raises here
  instead, matching the formats ``risk_scoring.state`` already enforces
  when an event is constructed.
- Patient-level lifetime aggregates (HEALTHCARE_EXPENSES,
  HEALTHCARE_COVERAGE, INCOME) and DEATHDATE are never features: all
  four encode information from beyond the scoring instant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "1.0.0"

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

DATE_FORMAT = "%Y-%m-%d"

DAYS_SINCE_PREV_DISCHARGE_CAP = 365.0

WINDOW_DAYS = 180

FEATURE_COLUMNS = (
    "encounter_id",
    "patient_id",
    "age_at_discharge",
    "los_days",
    "prior_inpatient_180d",
    "days_since_prev_discharge",
    "prior_ed_180d",
    "active_medication_count",
    "active_disorder_count",
    "flag_chf",
    "flag_chronic_pulmonary",
    "flag_dementia",
    "flag_diabetes",
    "flag_malignancy",
    "flag_mi",
    "flag_renal_disease",
)

# The two identifier columns are not features. Training, the gate, the
# serving path, and provenance re-verification all need the model's actual
# input column set, so it is named once here rather than re-sliced at each
# call site where the slice could silently drift.
MODEL_INPUT_COLUMNS = FEATURE_COLUMNS[2:]

FLAG_CODES: dict[str, frozenset[str]] = {
    "flag_chf": frozenset({"88805009", "84114007"}),
    "flag_chronic_pulmonary": frozenset({"233678006", "87433001", "185086009", "195967001"}),
    "flag_dementia": frozenset({"26929004", "230265002"}),
    "flag_diabetes": frozenset(
        {
            "44054006",
            "127013003",
            "90781000119102",
            "157141000119108",
            "368581000119106",
            "1551000119108",
            "97331000119101",
            "1501000119109",
            "427089005",
            "60951000119105",
        }
    ),
    "flag_malignancy": frozenset(
        {
            "91861009",
            "93143009",
            "93761005",
            "94260004",
            "94503003",
            "109838007",
            "126906006",
            "254632001",
            "254637007",
            "254837009",
            "363406005",
            "424132000",
            "67811000119102",
        }
    ),
    "flag_mi": frozenset({"22298006", "399211009", "401314000", "401303003"}),
    "flag_renal_disease": frozenset({"431855005", "431856006", "433144002", "431857002"}),
}


def _parse_optional_timestamps(raw: pd.Series, fmt: str, utc: bool) -> pd.Series:
    cleaned = raw.fillna("").astype(str)
    return pd.to_datetime(cleaned.where(cleaned != "", None), format=fmt, utc=utc)


def _counts_for(mask: pd.Series, encounter_ids: pd.Series, index: pd.Index) -> pd.Series:
    return mask.groupby(encounter_ids).sum().reindex(index, fill_value=0).astype(int)


def build_features(
    cohort: pd.DataFrame,
    encounters: pd.DataFrame,
    medications: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute one feature row per cohort row, as of that row's discharge."""
    out = cohort.loc[:, ["encounter_id", "patient_id", "age_at_discharge"]].reset_index(drop=True)
    score_start = pd.to_datetime(cohort["start"], format=TIMESTAMP_FORMAT, utc=True).reset_index(
        drop=True
    )
    score_stop = pd.to_datetime(cohort["stop"], format=TIMESTAMP_FORMAT, utc=True).reset_index(
        drop=True
    )
    score_date = score_stop.dt.tz_localize(None).dt.normalize()
    out["los_days"] = (score_stop - score_start).dt.total_seconds() / 86400.0

    anchor = pd.DataFrame(
        {
            "encounter_id": out["encounter_id"],
            "patient_id": out["patient_id"],
            "score_start": score_start,
            "score_stop": score_stop,
            "score_date": score_date,
        }
    )
    index = pd.Index(out["encounter_id"])

    enc = pd.DataFrame(
        {
            "Id": encounters["Id"],
            "patient_id": encounters["PATIENT"],
            "encounter_class": encounters["ENCOUNTERCLASS"],
            "prior_stop": _parse_optional_timestamps(
                encounters["STOP"], TIMESTAMP_FORMAT, utc=True
            ),
        }
    )
    joined = anchor.merge(enc, on="patient_id", how="inner")
    known = (joined["prior_stop"] <= joined["score_stop"]) & (
        joined["Id"] != joined["encounter_id"]
    )
    in_window = known & (
        joined["prior_stop"]
        >= joined["score_stop"] - pd.Timedelta(np.timedelta64(WINDOW_DAYS, "D"))
    )
    inpatient = joined["encounter_class"] == "inpatient"
    out["prior_inpatient_180d"] = _counts_for(
        (in_window & inpatient), joined["encounter_id"], index
    ).to_numpy()
    prev_stop = (
        joined.loc[known & inpatient]
        .groupby("encounter_id")["prior_stop"]
        .max()
        .reindex(index)
        .to_numpy()
    )
    gap_days = (score_start.to_numpy() - prev_stop) / np.timedelta64(1, "D")
    out["days_since_prev_discharge"] = (
        pd.Series(gap_days, dtype="float64")
        .clip(lower=0.0, upper=DAYS_SINCE_PREV_DISCHARGE_CAP)
        .fillna(DAYS_SINCE_PREV_DISCHARGE_CAP)
        .to_numpy()
    )
    emergency = joined["encounter_class"] == "emergency"
    out["prior_ed_180d"] = _counts_for(
        (in_window & emergency), joined["encounter_id"], index
    ).to_numpy()

    med = pd.DataFrame(
        {
            "patient_id": medications["PATIENT"],
            "med_start": pd.to_datetime(medications["START"], format=TIMESTAMP_FORMAT, utc=True),
            "med_stop": _parse_optional_timestamps(medications["STOP"], TIMESTAMP_FORMAT, utc=True),
        }
    )
    joined_med = anchor.merge(med, on="patient_id", how="inner")
    med_active = (joined_med["med_start"] <= joined_med["score_stop"]) & (
        joined_med["med_stop"].isna() | (joined_med["med_stop"] > joined_med["score_stop"])
    )
    out["active_medication_count"] = _counts_for(
        med_active, joined_med["encounter_id"], index
    ).to_numpy()

    cond = pd.DataFrame(
        {
            "patient_id": conditions["PATIENT"],
            "cond_start": pd.to_datetime(conditions["START"], format=DATE_FORMAT),
            "cond_stop": _parse_optional_timestamps(conditions["STOP"], DATE_FORMAT, utc=False),
            "system": conditions["SYSTEM"],
            "code": conditions["CODE"].astype(str),
            "description": conditions["DESCRIPTION"].astype(str),
        }
    )
    joined_cond = anchor.merge(cond, on="patient_id", how="inner")
    recorded = joined_cond["cond_start"] <= joined_cond["score_date"]
    cond_active = recorded & (
        joined_cond["cond_stop"].isna() | (joined_cond["cond_stop"] > joined_cond["score_date"])
    )
    is_disorder = joined_cond["description"].str.endswith("(disorder)")
    out["active_disorder_count"] = _counts_for(
        (cond_active & is_disorder), joined_cond["encounter_id"], index
    ).to_numpy()

    icd10_malignant = (joined_cond["system"] == "ICD10") & joined_cond["code"].str.startswith("C")
    for flag, codes in FLAG_CODES.items():
        qualifies = joined_cond["code"].isin(codes)
        if flag == "flag_malignancy":
            qualifies = qualifies | icd10_malignant
        out[flag] = (
            (_counts_for((recorded & qualifies), joined_cond["encounter_id"], index) > 0)
            .astype(int)
            .to_numpy()
        )

    return out.loc[:, list(FEATURE_COLUMNS)]
