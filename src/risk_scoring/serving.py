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
- A discharge whose patient has no demographics in state raises too. The
  cohort module already refuses to admit an encounter with an unknown
  patient; raising a named error here turns that into something the
  caller can answer for, rather than a bare ValueError from a shared
  module. It means the event stream delivered a discharge before the
  demographics it depends on, which is an ordering violation worth
  reporting, never a reason to skip the score. The check is a public
  function so the ingestion path can run it before the discharge is
  written, and refuse without storing anything.
- ``history_window`` narrows what state reads for one discharge to the
  rows its features can see, so a patient's lifetime is not reloaded on
  every event. The bounds come from the feature module's own rules:
  encounters read by STOP, back ``ENCOUNTER_LOOKBACK_DAYS`` from the
  admission and forward to the discharge instant; medications active
  at that instant; conditions recorded on or before the discharge date,
  with no lower bound because the comorbidity flags read resolved
  history. Nothing is re-expressed: rows outside the window are rows
  ``build_features`` would have ignored, and the skew test proves the
  bounded and unbounded reads score identically on rows a second to
  either side of each bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from risk_scoring.cohort import build_cohort
from risk_scoring.features import ENCOUNTER_LOOKBACK_DAYS, build_features
from risk_scoring.state import (
    DATE_FORMAT,
    TIMESTAMP_FORMAT,
    EncounterEvent,
    HistoryWindow,
    PatientHistory,
    parse_timestamp,
)


class UnknownEncounterError(LookupError):
    """The encounter to score has no row in the patient's recorded history."""


class UnknownPatientError(LookupError):
    """The encounter to score belongs to a patient with no recorded demographics."""


def require_demographics(recorded: bool, encounter_id: str, patient_id: str) -> None:
    """Raise :class:`UnknownPatientError` unless the patient's demographics are recorded.

    ``recorded`` is whatever the caller knows about the patient's presence in
    state; the error it raises names both the encounter and the patient.
    """
    if not recorded:
        raise UnknownPatientError(
            f"encounter {encounter_id!r} belongs to patient {patient_id!r}, whose "
            "demographics have not been recorded; the cohort rules need a birthdate"
        )


@dataclass(frozen=True)
class ScoringInput:
    """One admitted discharge: its single-row cohort frame and feature frame."""

    cohort: pd.DataFrame
    features: pd.DataFrame


def history_window(encounter: EncounterEvent) -> HistoryWindow | None:
    """The rows scoring this encounter can read, or ``None`` for a stay still open.

    An open stay has no discharge instant to bound against and is not a
    scoring event, so there is nothing to read for it.
    """
    if encounter.stop == "":
        return None
    admitted = parse_timestamp(encounter.start)
    floor = admitted - timedelta(days=ENCOUNTER_LOOKBACK_DAYS)
    discharged = parse_timestamp(encounter.stop)
    return HistoryWindow(
        encounter_stop_from=floor.strftime(TIMESTAMP_FORMAT),
        discharge=encounter.stop,
        discharge_date=discharged.strftime(DATE_FORMAT),
    )


def serving_features(history: PatientHistory, encounter_id: str) -> ScoringInput | None:
    """Compute the scoring input for one recorded encounter.

    Returns ``None`` when the encounter is not a scoring event: still open,
    or excluded by the cohort rules. Raises :class:`UnknownEncounterError`
    if the encounter is not in ``history``, and :class:`UnknownPatientError`
    if the patient's demographics have not been recorded.
    """
    encounters = history.encounters
    scored = encounters.loc[encounters["Id"] == encounter_id]
    if scored.empty:
        raise UnknownEncounterError(f"encounter {encounter_id!r} is not in the patient's history")
    if scored["STOP"].iloc[0] == "":
        return None
    require_demographics(not history.patients.empty, encounter_id, scored["PATIENT"].iloc[0])

    cohort = build_cohort(scored, history.patients).frame
    if cohort.empty:
        return None

    features = build_features(cohort, encounters, history.medications, history.conditions)
    return ScoringInput(cohort=cohort, features=features)
