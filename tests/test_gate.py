"""Tests for the evaluation gate.

The rules these tests pin:

- The gate verdict is pass only when every check passes: AUROC inside
  the pre-registered signal band on both sides, ECE within threshold,
  and (when an expected value is supplied) the recomputed holdout AUROC
  reproducing the training run's logged value.
- An above-ceiling AUROC fails loudly with a SUSPECTED LEAKAGE message
  naming the ceiling; a below-floor AUROC cites the fallback ladder.
- Subgroups are report-only: small or single-class subgroups have their
  metrics suppressed with a note, and never move the verdict.
- Subgroup membership covers four age bands, both sexes (joined from the
  patients frame), and the seven comorbidity flags.
- The rendered report carries the verdict, every check, the headline
  metrics with confidence intervals, and the subgroup table.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mlflow import MlflowClient

from factories import write_gate_population, write_leak_population, write_training_csvs
from risk_scoring import gate, train
from risk_scoring.cohort import build_cohort
from risk_scoring.features import FEATURE_COLUMNS, build_features
from risk_scoring.labels import build_labels

# --- deterministic fixtures ---

_SCORE_LEVELS = tuple((2 * i + 1) / 20 for i in range(10))  # 0.05, 0.15, ... 0.95
_ROWS_PER_LEVEL = 20


def _calibrated_sample() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], pd.Series[str]]:
    """200 rows, one patient each, exactly calibrated by construction.

    Each score level s carries s * 20 positives out of 20 rows, so every
    equal-count bin's observed rate equals its mean score exactly (ECE 0)
    while the ranking is imperfect (AUROC lands mid-band).
    """
    scores = []
    y = []
    for level in _SCORE_LEVELS:
        positives = round(level * _ROWS_PER_LEVEL)
        scores.extend([level] * _ROWS_PER_LEVEL)
        y.extend([1.0] * positives + [0.0] * (_ROWS_PER_LEVEL - positives))
    patient_ids = pd.Series([f"p{i:03d}" for i in range(len(scores))])
    return np.array(y), np.array(scores), patient_ids


def _evaluate(**kwargs: object) -> gate.GateResult:
    y, scores, patient_ids = _calibrated_sample()
    return gate.evaluate(scores, y, patient_ids, n_replicates=100, **kwargs)  # type: ignore[arg-type]


# --- verdict checks ---


def test_evaluate_passes_in_band_well_calibrated_scores() -> None:
    result = _evaluate()
    assert result.verdict == "pass"
    assert all(check.passed for check in result.checks)
    assert 0.65 <= result.auroc.value <= 0.92
    assert result.ece.value <= 0.05
    assert result.n_rows == 200
    assert result.n_patients == 200
    assert result.prevalence == 0.5


def test_evaluate_fails_above_ceiling_with_loud_leakage_message() -> None:
    y, _, patient_ids = _calibrated_sample()
    scores = y * 0.9 + 0.05  # perfectly separated: AUROC 1.0
    result = gate.evaluate(scores, y, patient_ids, n_replicates=100)

    assert result.verdict == "fail"
    ceiling = next(c for c in result.checks if c.name == "auroc_below_band_ceiling")
    assert not ceiling.passed
    assert "SUSPECTED LEAKAGE" in ceiling.detail
    assert "0.92" in ceiling.detail


def test_evaluate_fails_below_floor_citing_fallback_ladder() -> None:
    y, scores, patient_ids = _calibrated_sample()
    shuffled_y = np.random.default_rng(0).permutation(y)  # break the score-label association
    result = gate.evaluate(scores, shuffled_y, patient_ids, n_replicates=100)

    assert result.verdict == "fail"
    floor = next(c for c in result.checks if c.name == "auroc_above_band_floor")
    assert not floor.passed
    assert "fallback ladder" in floor.detail
    assert "0.65" in floor.detail


def test_evaluate_fails_when_ece_exceeds_threshold() -> None:
    y, scores, patient_ids = _calibrated_sample()
    inflated = 0.7 + scores * 0.29  # monotone: AUROC unchanged, calibration wrecked
    result = gate.evaluate(inflated, y, patient_ids, n_replicates=100)

    assert result.verdict == "fail"
    ece = next(c for c in result.checks if c.name == "ece_within_threshold")
    assert not ece.passed
    band_checks = [c for c in result.checks if c.name.startswith("auroc_")]
    assert all(c.passed for c in band_checks)


def test_evaluate_fails_on_holdout_reproduction_mismatch() -> None:
    result = _evaluate(expected_auroc=0.999)
    assert result.verdict == "fail"
    repro = next(c for c in result.checks if c.name == "holdout_reproduced")
    assert not repro.passed


def test_evaluate_passes_reproduction_check_when_auroc_matches() -> None:
    baseline = _evaluate()
    result = _evaluate(expected_auroc=baseline.auroc.value)
    repro = next(c for c in result.checks if c.name == "holdout_reproduced")
    assert repro.passed
    assert result.verdict == "pass"


def test_evaluate_omits_reproduction_check_without_expected_value() -> None:
    result = _evaluate()
    assert all(c.name != "holdout_reproduced" for c in result.checks)


# --- subgroups ---


def test_small_subgroup_is_reported_but_excluded_from_metrics() -> None:
    y, scores, patient_ids = _calibrated_sample()
    subgroups = pd.DataFrame(
        {
            "big": [True] * 200,
            "tiny": [True] * 5 + [False] * 195,
        }
    )
    result = gate.evaluate(scores, y, patient_ids, subgroups=subgroups, n_replicates=100)

    by_name = {s.name: s for s in result.subgroups}
    assert by_name["big"].auroc is not None
    assert by_name["big"].n_rows == 200
    assert by_name["tiny"].auroc is None
    assert by_name["tiny"].n_rows == 5
    assert "insufficient" in by_name["tiny"].note
    assert result.verdict == "pass"  # subgroups never move the verdict


def test_single_class_subgroup_is_suppressed() -> None:
    y, scores, patient_ids = _calibrated_sample()
    all_positive = y == 1.0
    subgroups = pd.DataFrame({"positives_only": all_positive})
    result = gate.evaluate(scores, y, patient_ids, subgroups=subgroups, n_replicates=100)

    (subgroup,) = result.subgroups
    assert subgroup.auroc is None
    assert "single" in subgroup.note


def test_build_subgroups_produces_age_sex_and_flag_columns() -> None:
    features = pd.DataFrame(
        {
            "patient_id": ["pa", "pb", "pc", "pd"],
            "age_at_discharge": [49, 50, 79, 80],
            "flag_chf": [1, 0, 0, 0],
            "flag_chronic_pulmonary": [0, 0, 0, 0],
            "flag_dementia": [0, 0, 0, 0],
            "flag_diabetes": [0, 1, 0, 0],
            "flag_malignancy": [0, 0, 0, 0],
            "flag_mi": [0, 0, 0, 0],
            "flag_renal_disease": [0, 0, 0, 1],
        }
    )
    patients = pd.DataFrame(
        {
            "Id": ["pa", "pb", "pc", "pd"],
            "GENDER": ["M", "F", "F", "M"],
        }
    )

    subgroups = gate.build_subgroups(features, patients)

    assert list(subgroups.columns) == [
        "age_18_49",
        "age_50_64",
        "age_65_79",
        "age_80_plus",
        "sex_male",
        "sex_female",
        "flag_chf",
        "flag_chronic_pulmonary",
        "flag_dementia",
        "flag_diabetes",
        "flag_malignancy",
        "flag_mi",
        "flag_renal_disease",
    ]
    assert len(subgroups) == 4
    assert subgroups.dtypes.eq(bool).all()
    assert list(subgroups["age_18_49"]) == [True, False, False, False]
    assert list(subgroups["age_50_64"]) == [False, True, False, False]
    assert list(subgroups["age_65_79"]) == [False, False, True, False]
    assert list(subgroups["age_80_plus"]) == [False, False, False, True]
    assert list(subgroups["sex_male"]) == [True, False, False, True]
    assert list(subgroups["sex_female"]) == [False, True, True, False]
    assert list(subgroups["flag_chf"]) == [True, False, False, False]
    assert list(subgroups["flag_renal_disease"]) == [False, False, False, True]


# --- report rendering ---


def test_render_report_contains_verdict_checks_metrics_and_subgroups() -> None:
    y, scores, patient_ids = _calibrated_sample()
    subgroups = pd.DataFrame({"big": [True] * 200, "tiny": [True] * 5 + [False] * 195})
    result = gate.evaluate(scores, y, patient_ids, subgroups=subgroups, n_replicates=100)

    report = gate.render_report(
        result,
        model_version=3,
        candidate_run_id="run-abc",
        data_dir="data/baseline/csv",
        cutoff="2025-01-01",
        seed=20260101,
    )

    assert "PASS" in report
    assert "run-abc" in report
    assert "version 3" in report
    for check in result.checks:
        assert check.name in report
    assert f"{result.auroc.value:.4f}" in report
    assert f"{result.auroc.ci_low:.4f}" in report
    assert f"{result.auroc.ci_high:.4f}" in report
    assert "big" in report and "tiny" in report
    assert "insufficient" in report


def test_render_report_headlines_leakage_on_ceiling_breach() -> None:
    y, _, patient_ids = _calibrated_sample()
    scores = y * 0.9 + 0.05
    result = gate.evaluate(scores, y, patient_ids, n_replicates=100)

    report = gate.render_report(
        result,
        model_version=1,
        candidate_run_id="run-leak",
        data_dir="data/baseline/csv",
        cutoff="2025-01-01",
        seed=20260101,
    )

    assert "FAIL" in report
    assert "SUSPECTED LEAKAGE" in report


# --- end to end against a trained, registered model ---


@pytest.fixture(scope="module")
def trained_repo(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, Path, train.TrainingResult]]:
    """One gate population trained and registered once for this module."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("gate-repo")
    csv_dir = root / "data" / "baseline" / "csv"
    write_gate_population(csv_dir)
    result = train.train(csv_dir, root)
    yield root, csv_dir, result
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


def test_gate_end_to_end_passes_and_logs_mlflow_run(
    trained_repo: tuple[Path, Path, train.TrainingResult],
) -> None:
    root, csv_dir, trained = trained_repo
    report_path = root / "gate_report.md"

    outcome = gate.run_gate(csv_dir, root, report_path=report_path)
    result = outcome.result

    assert outcome.model_version == trained.model_version
    assert result.verdict == "pass"
    repro = next(c for c in result.checks if c.name == "holdout_reproduced")
    assert repro.passed
    assert len(result.subgroups) == 13

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("readmission-risk")
    assert experiment is not None
    gate_runs = client.search_runs([experiment.experiment_id], "tags.run_type = 'gate'")
    assert len(gate_runs) >= 1
    gate_run = gate_runs[0]
    assert gate_run.data.tags["candidate_run_id"] == trained.run_id
    assert gate_run.data.tags["candidate_model_version"] == str(trained.model_version)
    assert gate_run.data.tags["gate_verdict"] == "pass"
    assert gate_run.data.metrics["gate_auroc"] == pytest.approx(result.auroc.value)
    assert gate_run.data.metrics["gate_ece"] == pytest.approx(result.ece.value)
    artifact_paths = [a.path for a in client.list_artifacts(gate_run.info.run_id)]
    assert "gate_report.md" in artifact_paths

    version = client.get_model_version(train.MODEL_NAME, str(trained.model_version))
    assert version.tags["gate_verdict"] == "pass"
    assert version.tags["gate_run_id"] == gate_run.info.run_id

    report = report_path.read_text()
    assert "PASS" in report
    assert trained.run_id in report


def test_cli_run_gates_registered_model_and_prints_report(
    trained_repo: tuple[Path, Path, train.TrainingResult],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _, _ = trained_repo
    monkeypatch.chdir(root)

    gate.main(["run"])

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "gate_auroc" not in out  # report is markdown prose, not raw metric keys


def test_cli_exits_nonzero_on_failing_gate(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The deterministic training population separates perfectly, so the
    # honest model lands above the band ceiling and the gate must fail.
    csv_dir = repo_root / "data" / "baseline" / "csv"
    write_training_csvs(csv_dir)
    train.train(csv_dir, repo_root)
    monkeypatch.chdir(repo_root)

    with pytest.raises(SystemExit) as excinfo:
        gate.main(["run"])

    assert excinfo.value.code == 1
    assert "FAIL" in capsys.readouterr().out


# --- the required negative test: leaked patient splits ---


def test_leaked_split_candidate_is_flagged_loudly(tmp_path: Path) -> None:
    """A candidate trained on a row-level (patient-leaking) split must fail.

    The leak population carries no cross-patient signal at all: outcomes
    are a per-patient coin. Any discrimination a row-split candidate
    shows on its holdout is memorized through patients it already saw in
    training, and the gate must flag the inflated score loudly.
    """
    csv_dir = tmp_path / "csv"
    write_leak_population(csv_dir)
    encounters = pd.read_csv(csv_dir / "encounters.csv", dtype=str, keep_default_na=False)
    patients = pd.read_csv(csv_dir / "patients.csv", dtype=str, keep_default_na=False)
    medications = pd.read_csv(csv_dir / "medications.csv", dtype=str, keep_default_na=False)
    conditions = pd.read_csv(csv_dir / "conditions.csv", dtype=str, keep_default_na=False)
    cohort = build_cohort(encounters, patients).frame
    labels = build_labels(cohort, encounters)
    features = build_features(cohort, encounters, medications, conditions)
    x = features.loc[:, list(FEATURE_COLUMNS[2:])].astype("float64")
    y = labels["label"].to_numpy(dtype=float)

    # Deliberately broken: rows are shuffled and split with no patient
    # grouping, so most evaluation patients also appear in training.
    positions = np.random.default_rng(20260101).permutation(len(x))
    split_at = int(len(positions) * 0.8)
    train_idx, holdout_idx = positions[:split_at], positions[split_at:]
    leaked = set(features["patient_id"].iloc[train_idx]) & set(
        features["patient_id"].iloc[holdout_idx]
    )
    assert leaked  # the split really does leak patients

    booster = lgb.train(
        train.LGBM_PARAMS,
        lgb.Dataset(x.iloc[train_idx], label=y[train_idx]),
        num_boost_round=train.NUM_BOOST_ROUND,
    )
    scores = np.asarray(booster.predict(x.iloc[holdout_idx]), dtype=float)

    result = gate.evaluate(
        scores, y[holdout_idx], features["patient_id"].iloc[holdout_idx], n_replicates=100
    )

    assert result.verdict == "fail"
    ceiling = next(c for c in result.checks if c.name == "auroc_below_band_ceiling")
    assert not ceiling.passed
    assert result.auroc.value > 0.92
    assert "SUSPECTED LEAKAGE" in ceiling.detail

    report = gate.render_report(
        result,
        model_version=99,
        candidate_run_id="run-leaked",
        data_dir=str(csv_dir),
        cutoff="2025-01-01",
        seed=20260101,
    )
    assert "FAIL" in report
    assert "SUSPECTED LEAKAGE" in report
