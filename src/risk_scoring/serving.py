"""Serving-time scoring inputs computed from one patient's event state.

The service scores a discharge by reading that patient's recorded event
history and running it through the same two functions the training
pipeline runs: ``cohort.build_cohort`` decides admission and
``features.build_features`` computes the feature row. Neither rule is
re-expressed here. This module is the glue that narrows the batch call to
a single encounter, so "one cohort module and one feature module, shared
verbatim" stays a structural property rather than a claim a test has to
chase.

Judgment calls this module fixes:

- The cohort check runs over the single encounter being scored, not the
  patient's whole history. Every cohort rule is per-encounter (class,
  in-hospital death, age at discharge), so the narrowed call is exactly
  equivalent and does no redundant work. Feature computation, by
  contrast, receives the full history, because prior encounters,
  medications, and conditions are what the features read.
- Scoring triggers on a discharge. An encounter still open at ingestion
  (empty ``STOP``) is not a scoring event and yields no scoring input;
  the cohort module never sees one, because a completed CSV export has
  no such row and admitting one would produce a feature row anchored to
  a missing timestamp.
- Asking to score an encounter absent from state is a caller error, not
  an exclusion, and raises rather than returning nothing. A silent
  ``None`` there would make a lost ingestion look like a routine cohort
  exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk_scoring.cohort import build_cohort
from risk_scoring.features import build_features
from risk_scoring.state import PatientHistory


class UnknownEncounterError(LookupError):
    """The encounter to score has no row in the patient's recorded history."""


@dataclass(frozen=True)
class ScoringInput:
    """One admitted discharge: its single-row cohort frame and feature frame."""

    cohort: pd.DataFrame
    features: pd.DataFrame


def serving_features(history: PatientHistory, encounter_id: str) -> ScoringInput | None:
    """Compute the scoring input for one recorded encounter.

    Returns ``None`` when the encounter is not a scoring event: still open,
    or excluded by the cohort rules. Raises :class:`UnknownEncounterError`
    if the encounter is not in ``history``.
    """
    encounters = history.encounters
    scored = encounters.loc[encounters["Id"] == encounter_id]
    if scored.empty:
        raise UnknownEncounterError(f"encounter {encounter_id!r} is not in the patient's history")
    if scored["STOP"].iloc[0] == "":
        return None

    cohort = build_cohort(scored, history.patients).frame
    if cohort.empty:
        return None

    features = build_features(cohort, encounters, history.medications, history.conditions)
    return ScoringInput(cohort=cohort, features=features)
