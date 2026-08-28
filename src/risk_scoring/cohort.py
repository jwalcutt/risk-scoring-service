"""Cohort definition for 30-day readmission scoring.

The scoring event is an inpatient discharge. An encounter enters the
cohort when its ENCOUNTERCLASS is ``inpatient``, the patient did not die
in hospital, and the patient is at least 18 years old on the discharge
date. Each qualifying encounter yields one cohort row keyed by encounter
id, scored at the discharge timestamp.

Judgment calls this module fixes:

- Synthea's DEATHDATE is date-only while encounter START/STOP are full
  datetimes. A death counts as in-hospital when the death date falls
  within [date(START), date(STOP)] inclusive, so a death recorded on the
  discharge date excludes the encounter.
- A death date earlier than the admission date is a data anomaly. The
  encounter is excluded and counted separately rather than silently kept.
- Age is calendar age on the discharge date; a patient whose 18th
  birthday is the discharge date is included (exclusion is strictly
  under 18).
- Excluded encounters are attributed to the first rule they fail, in the
  order: encounter class, death before admission, in-hospital death,
  under 18.
- Encounter START/STOP are parsed under ``%Y-%m-%dT%H:%M:%SZ`` and
  patient BIRTHDATE/DEATHDATE under ``%Y-%m-%d``, the exact formats the
  Synthea export writes. Pandas would otherwise guess a format from the
  first element and fall back to per-element dateutil parsing when the
  guess fails, which is how the same digits get read two ways. A
  non-conforming value raises here instead.
- An inpatient encounter whose patient id has no patients.csv row raises
  rather than being dropped, because a silent drop would mask joined-data
  corruption in later replay phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from risk_scoring.populations import load_population

COHORT_VERSION = "1.0.0"

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

DATE_FORMAT = "%Y-%m-%d"

COHORT_COLUMNS = ("encounter_id", "patient_id", "start", "stop", "age_at_discharge")


@dataclass(frozen=True)
class ExclusionCounts:
    non_inpatient: int
    death_before_admission: int
    in_hospital_death: int
    under_18: int


@dataclass(frozen=True)
class CohortResult:
    frame: pd.DataFrame
    exclusions: ExclusionCounts


@dataclass(frozen=True)
class CohortSplit:
    """A cohort cut in two at a timestamp, by discharge."""

    before: pd.DataFrame
    """Discharges strictly before the cutoff: what a model may train on."""

    at_or_after: pd.DataFrame
    """Discharges at or after the cutoff: what the model has never seen."""


def split_at_cutoff(cohort: pd.DataFrame, cutoff: pd.Timestamp) -> CohortSplit:
    """Partition cohort rows by whether the discharge precedes the cutoff.

    Returning both halves from one call is what makes "held out" the exact
    complement of "trained on". Stated as two independent filters in two
    modules, a boundary that drifted on one side would leak discharges
    into training and silently vanish from any held-out evaluation.

    A discharge landing exactly on the cutoff instant is held out.
    """
    stop = cohort["stop"]
    return CohortSplit(
        before=cohort.loc[stop < cutoff].reset_index(drop=True),
        at_or_after=cohort.loc[stop >= cutoff].reset_index(drop=True),
    )


def filter_training_window(cohort: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep cohort rows whose discharge STOP is strictly before the cutoff."""
    return split_at_cutoff(cohort, cutoff).before


def _parse_death_dates(raw: pd.Series) -> pd.Series:
    cleaned = raw.fillna("").astype(str)
    return pd.to_datetime(cleaned.where(cleaned != "", None), format=DATE_FORMAT, utc=True)


def build_cohort(encounters: pd.DataFrame, patients: pd.DataFrame) -> CohortResult:
    """Apply the cohort rules to raw Synthea encounters and patients frames."""
    inpatient = encounters.loc[encounters["ENCOUNTERCLASS"] == "inpatient"]
    non_inpatient = len(encounters) - len(inpatient)

    demographics = patients.loc[:, ["Id", "BIRTHDATE", "DEATHDATE"]].rename(
        columns={"Id": "PATIENT"}
    )
    merged = inpatient.merge(demographics, on="PATIENT", how="left", validate="many_to_one")

    orphaned = merged["BIRTHDATE"].isna()
    if bool(orphaned.any()):
        orphan_ids = ", ".join(merged.loc[orphaned, "Id"])
        raise ValueError(f"inpatient encounters reference unknown patients: {orphan_ids}")

    start = pd.to_datetime(merged["START"], format=TIMESTAMP_FORMAT, utc=True)
    stop = pd.to_datetime(merged["STOP"], format=TIMESTAMP_FORMAT, utc=True)
    birth = pd.to_datetime(merged["BIRTHDATE"], format=DATE_FORMAT, utc=True)
    death = _parse_death_dates(merged["DEATHDATE"])

    before_birthday = (stop.dt.month < birth.dt.month) | (
        (stop.dt.month == birth.dt.month) & (stop.dt.day < birth.dt.day)
    )
    age = stop.dt.year - birth.dt.year - before_birthday.astype(int)

    death_before_admission = death.notna() & (death < start.dt.normalize())
    in_hospital_death = death.notna() & ~death_before_admission & (death <= stop.dt.normalize())
    under_18 = ~death_before_admission & ~in_hospital_death & (age < 18)
    included = ~death_before_admission & ~in_hospital_death & ~under_18

    cohort = pd.DataFrame(
        {
            "encounter_id": merged["Id"],
            "patient_id": merged["PATIENT"],
            "start": start,
            "stop": stop,
            "age_at_discharge": age,
        }
    ).loc[included]

    return CohortResult(
        frame=cohort.reset_index(drop=True),
        exclusions=ExclusionCounts(
            non_inpatient=non_inpatient,
            death_before_admission=int(death_before_admission.sum()),
            in_hospital_death=int(in_hospital_death.sum()),
            under_18=int(under_18.sum()),
        ),
    )


def load_cohort(csv_dir: Path) -> CohortResult:
    """Build the cohort from a Synthea CSV export directory."""
    frames = load_population(csv_dir, frames=("encounters", "patients"))
    return build_cohort(frames["encounters"], frames["patients"])
