"""Tests for the training entrypoint.

The rules these tests pin:

- Only cohort discharges with STOP strictly before the cutoff enter
  training and holdout; a discharge stopping exactly at the cutoff is
  excluded.
- The holdout split is grouped by patient: no patient contributes rows
  to both sides, and the split is deterministic for a fixed seed.
- One call trains from a raw Synthea CSV directory to a registered
  MLflow model version carrying the cohort, feature, and label versions,
  the cutoff and seed, the holdout metrics, and the signal-band verdict.
- The registered model, loaded back through pyfunc, returns probability
  scores in [0, 1].
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
    write_rows_csv,
)
from risk_scoring import train
from risk_scoring.cohort import build_cohort
from risk_scoring.features import FEATURE_COLUMNS, build_features

CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")


@pytest.fixture()
def repo_root(tmp_path: Path) -> Iterator[Path]:
    """A throwaway repo root; restores global MLflow URIs after the test."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    yield tmp_path
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


def cohort_row(encounter_id: str, patient_id: str, stop: str) -> dict[str, object]:
    stop_ts = pd.Timestamp(stop)
    return {
        "encounter_id": encounter_id,
        "patient_id": patient_id,
        "start": stop_ts - pd.Timedelta(np.timedelta64(3, "D")),
        "stop": stop_ts,
        "age_at_discharge": 54,
    }


# --- training window ---


def test_discharge_stopping_just_before_cutoff_is_kept() -> None:
    cohort = pd.DataFrame([cohort_row("e1", "p1", "2024-12-31T23:59:59Z")])
    kept = train.filter_training_window(cohort, CUTOFF)
    assert list(kept["encounter_id"]) == ["e1"]


def test_discharge_stopping_exactly_at_cutoff_is_excluded() -> None:
    cohort = pd.DataFrame([cohort_row("e1", "p1", "2025-01-01T00:00:00Z")])
    assert train.filter_training_window(cohort, CUTOFF).empty


def test_discharge_stopping_after_cutoff_is_excluded() -> None:
    cohort = pd.DataFrame(
        [
            cohort_row("e1", "p1", "2024-06-01T08:00:00Z"),
            cohort_row("e2", "p2", "2025-06-01T08:00:00Z"),
        ]
    )
    kept = train.filter_training_window(cohort, CUTOFF)
    assert list(kept["encounter_id"]) == ["e1"]


# --- grouped split ---


def _many_rows_per_patient() -> pd.Series:
    return pd.Series([f"p{i}" for i in range(30) for _ in range(3)])


def test_grouped_split_puts_no_patient_on_both_sides() -> None:
    patient_ids = _many_rows_per_patient()
    train_idx, holdout_idx = train.grouped_split(patient_ids, 0.2, seed=20260101)

    train_patients = set(patient_ids.iloc[train_idx])
    holdout_patients = set(patient_ids.iloc[holdout_idx])
    assert train_patients
    assert holdout_patients
    assert not train_patients & holdout_patients
    assert len(train_idx) + len(holdout_idx) == len(patient_ids)


def test_grouped_split_is_deterministic_for_a_fixed_seed() -> None:
    patient_ids = _many_rows_per_patient()
    first = train.grouped_split(patient_ids, 0.2, seed=20260101)
    second = train.grouped_split(patient_ids, 0.2, seed=20260101)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_grouped_split_holdout_fraction_is_approximately_requested() -> None:
    patient_ids = pd.Series([f"p{i}" for i in range(100)])
    _, holdout_idx = train.grouped_split(patient_ids, 0.2, seed=20260101)
    assert 10 <= len(holdout_idx) <= 35


# --- end to end ---


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_training_csvs(csv_dir: Path) -> int:
    """Synthetic population with a learnable signal; returns the cohort row count.

    Forty adult patients each have one index inpatient stay well before the
    cutoff. Even-numbered patients carry a prior emergency visit (the
    signal) and an inpatient readmission 10 days after the index discharge;
    the readmission stays are cohort rows themselves, so the pre-cutoff
    cohort holds 60 rows. One extra patient discharges after the cutoff and
    must be excluded.
    """
    patients = []
    encounters = []
    for i in range(40):
        pid = f"p{i:02d}"
        patients.append(make_patient_row(Id=pid, BIRTHDATE="1960-01-01"))
        index_start = datetime(2024, 3, 1, 8, 0, tzinfo=UTC) + timedelta(days=i)
        index_stop = index_start + timedelta(days=3)
        encounters.append(
            make_encounter_row(
                Id=f"e-index-{pid}",
                PATIENT=pid,
                ENCOUNTERCLASS="inpatient",
                START=_iso(index_start),
                STOP=_iso(index_stop),
            )
        )
        if i % 2 == 0:
            ed_visit = index_start - timedelta(days=30)
            encounters.append(
                make_encounter_row(
                    Id=f"e-ed-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="emergency",
                    START=_iso(ed_visit),
                    STOP=_iso(ed_visit + timedelta(hours=4)),
                )
            )
            readmit_start = index_stop + timedelta(days=10)
            encounters.append(
                make_encounter_row(
                    Id=f"e-readmit-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="inpatient",
                    START=_iso(readmit_start),
                    STOP=_iso(readmit_start + timedelta(days=2)),
                )
            )

    patients.append(make_patient_row(Id="p-late", BIRTHDATE="1960-01-01"))
    encounters.append(
        make_encounter_row(
            Id="e-late",
            PATIENT="p-late",
            ENCOUNTERCLASS="inpatient",
            START="2025-01-30T08:00:00Z",
            STOP="2025-02-02T08:00:00Z",
        )
    )

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(
        csv_dir / "medications.csv",
        [make_medication_row(PATIENT="p00", ENCOUNTER="e-index-p00")],
    )
    write_rows_csv(
        csv_dir / "conditions.csv",
        [make_condition_row(PATIENT="p00", ENCOUNTER="e-index-p00")],
    )
    return 60


def test_train_end_to_end_registers_model_with_versions_metrics_and_band_tag(
    repo_root: Path,
) -> None:
    csv_dir = repo_root / "csv"
    expected_rows = write_training_csvs(csv_dir)

    result = train.train(csv_dir, repo_root)

    assert result.n_train_rows + result.n_holdout_rows == expected_rows
    # The post-cutoff patient is excluded by the training window.
    assert result.n_train_patients + result.n_holdout_patients == 40
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 < result.holdout_prevalence < 1.0
    assert result.in_band == (0.65 <= result.auroc <= 0.92)

    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{train.MODEL_NAME}'")
    assert [int(v.version) for v in versions] == [result.model_version] == [1]

    run = client.get_run(result.run_id)
    for param in (
        "cohort_version",
        "feature_version",
        "label_version",
        "training_cutoff",
        "split_seed",
        "holdout_fraction",
        "num_boost_round",
    ):
        assert param in run.data.params
    assert run.data.metrics["holdout_auroc"] == pytest.approx(result.auroc)
    assert run.data.tags["auroc_in_band"] in {"true", "false"}
    assert run.data.tags["signal_band"] == "0.65-0.92"

    loaded = mlflow.pyfunc.load_model(f"models:/{train.MODEL_NAME}/1")
    encounters = pd.read_csv(csv_dir / "encounters.csv", dtype=str, keep_default_na=False)
    patients = pd.read_csv(csv_dir / "patients.csv", dtype=str, keep_default_na=False)
    medications = pd.read_csv(csv_dir / "medications.csv", dtype=str, keep_default_na=False)
    conditions = pd.read_csv(csv_dir / "conditions.csv", dtype=str, keep_default_na=False)
    cohort = build_cohort(encounters, patients).frame
    features = build_features(cohort, encounters, medications, conditions)
    model_input = features[list(FEATURE_COLUMNS[2:])].astype("float64")
    scores = np.asarray(loaded.predict(model_input), dtype=float)
    assert scores.shape == (len(features),)
    assert bool(((scores >= 0.0) & (scores <= 1.0)).all())


def test_cli_run_trains_from_population_directory(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_training_csvs(repo_root / "data" / "baseline" / "csv")
    monkeypatch.chdir(repo_root)

    train.main(["run"])

    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{train.MODEL_NAME}'")
    assert [int(v.version) for v in versions] == [1]
