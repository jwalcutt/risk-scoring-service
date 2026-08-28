"""The training-serving skew check.

The batch pipeline reads a whole population's CSVs at once; the service
sees the same rows one event at a time and recomputes each discharge's
features from persisted state at the moment that discharge arrives. Both
paths must produce byte-identical feature values. The assertion is exact
equality, never a tolerance: a feature that differs by a rounding step
between training and serving is still a skew bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from factories import ordered_events, write_skew_population
from risk_scoring import serving, state
from risk_scoring.cohort import build_cohort
from risk_scoring.features import FEATURE_COLUMNS, build_features

pytestmark = pytest.mark.db


def _replay(conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ingest the population as an event stream, scoring discharges on arrival."""
    for _, row in frames["patients"].iterrows():
        state.record_patient(conn, state.PatientEvent.from_row(dict(row)))

    scored: list[pd.DataFrame] = []
    for event in ordered_events(frames["encounters"], frames["medications"], frames["conditions"]):
        if event.kind == "medication":
            state.record_medication(conn, state.MedicationEvent.from_row(event.row))
            continue
        if event.kind == "condition":
            state.record_condition(conn, state.ConditionEvent.from_row(event.row))
            continue
        state.record_encounter(conn, state.EncounterEvent.from_row(event.row))
        history = state.patient_history(conn, event.row["PATIENT"])
        result = serving.serving_features(history, event.row["Id"])
        if result is not None:
            scored.append(result.features)

    if not scored:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    return pd.concat(scored, ignore_index=True)


def _sorted_by_encounter(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("encounter_id").reset_index(drop=True)


@pytest.fixture()
def skew_frames(tmp_path: Path) -> dict[str, pd.DataFrame]:
    """The boundary population, loaded exactly as the training pipeline loads it."""
    csv_dir = tmp_path / "csv"
    write_skew_population(csv_dir)
    return {
        name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in ("patients", "encounters", "medications", "conditions")
    }


def _batch_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    return build_features(cohort, frames["encounters"], frames["medications"], frames["conditions"])


def test_population_exercises_the_boundaries_it_claims(
    skew_frames: dict[str, pd.DataFrame],
) -> None:
    """Guard the fixture: a population that collapses would make the skew test vacuous."""
    batch = _batch_features(skew_frames).set_index("encounter_id")

    assert len(batch) == 10
    assert batch.loc["e-fresh", "days_since_prev_discharge"] == 365.0
    assert batch.loc["e-fresh", "prior_inpatient_180d"] == 0
    assert batch.loc["e-edge-index", "prior_inpatient_180d"] == 1
    assert batch.loc["e-edge-index", "prior_ed_180d"] == 1
    assert batch.loc["e-gap-index", "days_since_prev_discharge"] == 365.0
    assert batch.loc["e-gap-overlap-b", "days_since_prev_discharge"] == 0.0
    assert batch.loc["e-readmit-2", "days_since_prev_discharge"] == 10.0
    assert batch.loc["e-full-index", "active_medication_count"] == 3
    assert batch.loc["e-full-index", "active_disorder_count"] == 4
    assert batch.loc["e-full-index", "flag_chf"] == 1
    assert batch.loc["e-full-index", "flag_mi"] == 1
    assert batch.loc["e-full-index", "flag_malignancy"] == 1


def test_serving_features_equal_training_features_exactly(
    db_conn: psycopg.Connection[Any], skew_frames: dict[str, pd.DataFrame]
) -> None:
    batch = _sorted_by_encounter(_batch_features(skew_frames))
    served = _sorted_by_encounter(_replay(db_conn, skew_frames))

    assert served["encounter_id"].tolist() == batch["encounter_id"].tolist()
    assert len(served) == 10
    pd.testing.assert_frame_equal(served, batch, check_exact=True)


def test_excluded_encounters_update_state_without_scoring(
    db_conn: psycopg.Connection[Any], skew_frames: dict[str, pd.DataFrame]
) -> None:
    """Every encounter the cohort rules reject still lands in state."""
    served = _replay(db_conn, skew_frames)
    rejected = {"e-edge-out", "e-edge-ed", "e-minor", "e-died", "e-out-wellness", "e-out-ed"}
    encounters = skew_frames["encounters"]

    assert rejected.isdisjoint(served["encounter_id"])
    for patient in ("p-edge", "p-minor", "p-died", "p-outpatient"):
        stored = set(state.patient_history(db_conn, patient).encounters["Id"])
        theirs = set(encounters.loc[encounters["PATIENT"] == patient, "Id"])
        assert rejected & theirs <= stored
