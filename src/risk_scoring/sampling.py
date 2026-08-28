"""Narrowing a population down to a runnable set of patients.

The frozen populations hold millions of rows, and posting one whole is
neither necessary nor informative. Narrowing is exact rather than
approximate: every feature reads only the scored patient's own rows, so a
patient's features do not depend on which other patients share the frame.

Judgment calls this module fixes:

- Eligibility means holding a discharge the service would actually score.
  Sampling from all patients would mostly draw people the service never
  scores, and a cohort discharge is the only kind that counts, so an
  in-hospital death or an under-18 stay makes a patient no more eligible
  than an outpatient visit does.
- ``select_patients`` takes patient ids and no dates. Narrowing to a time
  window would silently break the training-serving comparison, because a
  patient's features read their whole history; taking no dates at all
  makes that structural instead of a rule a caller has to remember.
- A cutoff narrows *eligibility*, never the rows kept. A patient chosen
  for a post-cutoff discharge is posted with every row they own, back to
  their earliest.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import pandas as pd

from risk_scoring.cohort import build_cohort, split_at_cutoff

_PATIENT_KEY = {
    "patients": "Id",
    "encounters": "PATIENT",
    "medications": "PATIENT",
    "conditions": "PATIENT",
}


def select_patients(
    frames: Mapping[str, pd.DataFrame], patient_ids: Collection[str]
) -> dict[str, pd.DataFrame]:
    """Restrict every frame to the given patients, keeping all their rows."""
    chosen = set(patient_ids)
    return {
        name: frame.loc[frame[_PATIENT_KEY[name]].isin(chosen)] for name, frame in frames.items()
    }


def eligible_patients(
    frames: Mapping[str, pd.DataFrame], *, discharged_at_or_after: pd.Timestamp | None = None
) -> pd.Index:
    """Patients holding a cohort discharge, optionally one past a cutoff."""
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    if discharged_at_or_after is not None:
        cohort = split_at_cutoff(cohort, discharged_at_or_after).at_or_after
    unique: list[str] = cohort["patient_id"].unique().tolist()
    return pd.Index(unique)


def sample_patients(
    frames: Mapping[str, pd.DataFrame],
    *,
    count: int,
    seed: int,
    discharged_at_or_after: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Every row of a seeded sample of eligible patients."""
    eligible = eligible_patients(frames, discharged_at_or_after=discharged_at_or_after)
    chosen = eligible.to_series().sample(n=min(count, len(eligible)), random_state=seed)
    return select_patients(frames, set(chosen.tolist()))
