"""Tests for the pure evaluation primitives.

The rules these tests pin:

- Calibration bins are equal-count over score rank (stable sort), so no
  bin is empty at skewed prevalence; the bins partition every row.
- Expected calibration error is the count-weighted mean absolute gap
  between each bin's mean score and its observed event rate.
- The bootstrap resamples unique patients with replacement, keeping every
  row of a drawn patient with multiplicity: patients are never split.
- Bootstrap draws are deterministic for a fixed seed and differ across
  seeds; confidence intervals are percentile-based and ordered.
- Replicates whose resampled labels collapse to a single class are
  skipped, and the number of replicates actually used is reported; if no
  replicate is usable the computation refuses rather than guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from risk_scoring import evaluation

# --- calibration bins and ECE ---


def test_ece_matches_hand_computed_value() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    # Two equal-count bins: (0.1, 0.2) mean 0.15 vs rate 0.5, and
    # (0.8, 0.9) mean 0.85 vs rate 0.5; each gap is 0.35.
    ece = evaluation.expected_calibration_error(y, scores, n_bins=2)
    assert ece == pytest.approx(0.35)


def test_ece_is_zero_when_bin_means_match_observed_rates() -> None:
    scores = np.array([0.4] * 5 + [0.8] * 5)
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    ece = evaluation.expected_calibration_error(y, scores, n_bins=2)
    assert ece == pytest.approx(0.0)


def test_calibration_bins_partition_all_rows() -> None:
    rng = np.random.default_rng(7)
    scores = rng.uniform(size=23)
    y = (rng.uniform(size=23) < scores).astype(float)

    bins = evaluation.calibration_bins(y, scores, n_bins=10)

    assert sum(b.count for b in bins) == 23
    assert all(b.count > 0 for b in bins)
    assert all(b.lower <= b.upper for b in bins)
    lowers = [b.lower for b in bins]
    uppers = [b.upper for b in bins]
    assert lowers == sorted(lowers)
    assert uppers == sorted(uppers)


def test_calibration_bin_stats_match_their_rows() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.2, 0.3, 0.6, 0.7])
    bins = evaluation.calibration_bins(y, scores, n_bins=2)

    assert bins[0].mean_score == pytest.approx(0.25)
    assert bins[0].observed_rate == pytest.approx(0.0)
    assert bins[1].mean_score == pytest.approx(0.65)
    assert bins[1].observed_rate == pytest.approx(1.0)


# --- patient-level bootstrap ---


def _grouped_rows() -> pd.Series:
    return pd.Series(["pA", "pA", "pA", "pB", "pB", "pC"])


def test_bootstrap_indices_keep_patients_intact() -> None:
    patient_ids = _grouped_rows()
    row_positions = {
        patient: set(np.flatnonzero((patient_ids == patient).to_numpy()))
        for patient in ("pA", "pB", "pC")
    }

    for indices in evaluation.patient_bootstrap_indices(patient_ids, n_replicates=20, seed=1):
        drawn = pd.Series(indices)
        drawn_patients = patient_ids.iloc[indices].reset_index(drop=True)
        for patient in drawn_patients.unique():
            patient_rows = drawn[(drawn_patients == patient).to_numpy()]
            expected = row_positions[patient]
            counts = patient_rows.value_counts()
            # Every row of the drawn patient appears, all with the same
            # multiplicity: whole patients, never fragments.
            assert set(counts.index) == expected
            assert counts.nunique() == 1


def test_bootstrap_indices_are_deterministic_for_a_fixed_seed() -> None:
    patient_ids = _grouped_rows()
    first = list(evaluation.patient_bootstrap_indices(patient_ids, n_replicates=5, seed=3))
    second = list(evaluation.patient_bootstrap_indices(patient_ids, n_replicates=5, seed=3))
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_bootstrap_indices_differ_across_seeds() -> None:
    patient_ids = _grouped_rows()
    first = list(evaluation.patient_bootstrap_indices(patient_ids, n_replicates=5, seed=3))
    second = list(evaluation.patient_bootstrap_indices(patient_ids, n_replicates=5, seed=4))
    assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_bootstrap_ci_brackets_point_estimate_and_is_ordered() -> None:
    rng = np.random.default_rng(11)
    patient_ids = pd.Series([f"p{i}" for i in range(40) for _ in range(2)])
    scores = rng.uniform(size=80)
    y = (rng.uniform(size=80) < scores).astype(float)

    result = evaluation.bootstrap_ci(
        lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
        y,
        scores,
        patient_ids,
        n_replicates=200,
        seed=5,
    )

    assert result.value == pytest.approx(float(roc_auc_score(y, scores)))
    assert result.ci_low <= result.value <= result.ci_high
    assert result.n_replicates_used == 200


def test_bootstrap_skips_single_class_replicates() -> None:
    # Two patients, one all-positive and one all-negative: any replicate
    # drawing the same patient twice collapses to a single class.
    patient_ids = pd.Series(["pA", "pA", "pB", "pB"])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    result = evaluation.bootstrap_ci(
        lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
        y,
        scores,
        patient_ids,
        n_replicates=50,
        seed=2,
    )

    assert 0 < result.n_replicates_used < 50


def test_bootstrap_refuses_when_no_replicate_is_usable() -> None:
    patient_ids = pd.Series(["pA", "pA"])
    y = np.array([1.0, 1.0])
    scores = np.array([0.9, 0.8])

    with pytest.raises(ValueError, match="single-class"):
        evaluation.bootstrap_ci(
            lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
            y,
            scores,
            patient_ids,
            n_replicates=10,
            seed=2,
        )
