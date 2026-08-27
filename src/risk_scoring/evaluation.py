"""Pure evaluation primitives: calibration, ECE, patient-level bootstrap.

These functions are model-agnostic and policy-free: they compute numbers
from labels, scores, and patient identifiers, and nothing else. Gate
thresholds and verdicts live elsewhere, so later monitoring work can
reuse the same primitives without dragging gate policy along.

Judgment calls this module fixes:

- Calibration bins are equal-count over score rank (quantile-style), not
  equal-width. At the ~12% prevalence this project runs at, scores skew
  low and equal-width bins run empty; equal-count bins keep every bin
  populated. Rank order uses a stable sort so ties bin deterministically.
- Expected calibration error is the count-weighted mean absolute gap
  between each bin's mean score and its observed event rate.
- The bootstrap resampling unit is the patient, not the row: unique
  patients are drawn with replacement and every row of a drawn patient
  is kept with multiplicity. Rows of one patient are correlated, so
  row-level resampling would understate the intervals.
- Confidence intervals are percentile intervals over the replicate
  metric values, 2.5th to 97.5th.
- A replicate whose resampled labels collapse to a single class is
  skipped (rank metrics are undefined there) and the count of replicates
  actually used is reported alongside the interval. If every replicate
  collapses, the computation raises rather than returning a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

ECE_BINS = 10

BOOTSTRAP_REPLICATES = 1000

BOOTSTRAP_SEED = 20260101

CI_QUANTILES = (0.025, 0.975)

Metric = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-count score bin: score range, mean score, observed rate."""

    lower: float
    upper: float
    mean_score: float
    observed_rate: float
    count: int


@dataclass(frozen=True)
class MetricCI:
    """A point estimate with its percentile bootstrap interval."""

    value: float
    ci_low: float
    ci_high: float
    n_replicates_used: int


def calibration_bins(
    y: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    n_bins: int = ECE_BINS,
) -> tuple[CalibrationBin, ...]:
    """Split rows into equal-count bins by score rank and summarize each."""
    if len(y) != len(scores):
        raise ValueError("y and scores must have the same length")
    if len(y) == 0:
        raise ValueError("calibration needs at least one row")
    order = np.argsort(scores, kind="stable")
    bins = []
    for chunk in np.array_split(order, min(n_bins, len(order))):
        chunk_scores = scores[chunk]
        chunk_y = y[chunk]
        bins.append(
            CalibrationBin(
                lower=float(chunk_scores.min()),
                upper=float(chunk_scores.max()),
                mean_score=float(chunk_scores.mean()),
                observed_rate=float(chunk_y.mean()),
                count=len(chunk),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    y: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    n_bins: int = ECE_BINS,
) -> float:
    """Count-weighted mean absolute gap between bin mean score and rate."""
    bins = calibration_bins(y, scores, n_bins)
    total = sum(b.count for b in bins)
    return float(sum(b.count * abs(b.mean_score - b.observed_rate) for b in bins) / total)


def patient_bootstrap_indices(
    patient_ids: pd.Series[str],
    n_replicates: int,
    seed: int,
) -> Iterator[npt.NDArray[np.intp]]:
    """Yield row-position arrays resampling unique patients with replacement."""
    ids = patient_ids.reset_index(drop=True)
    unique_patients = ids.unique()
    rows_by_patient = {
        patient: np.flatnonzero((ids == patient).to_numpy()).astype(np.intp)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    for _ in range(n_replicates):
        drawn = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        yield np.concatenate([rows_by_patient[patient] for patient in drawn])


def bootstrap_ci(
    metric: Metric,
    y: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    patient_ids: pd.Series[str],
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> MetricCI:
    """Point estimate plus a percentile CI from patient-level resampling."""
    if np.unique(y).size < 2:
        raise ValueError("labels are single-class; rank metrics are undefined")
    value = metric(y, scores)
    replicate_values = []
    for indices in patient_bootstrap_indices(patient_ids, n_replicates, seed):
        y_rep = y[indices]
        if np.unique(y_rep).size < 2:
            continue
        replicate_values.append(metric(y_rep, scores[indices]))
    if not replicate_values:
        raise ValueError("every bootstrap replicate was single-class; too few patients per class")
    ci_low, ci_high = np.quantile(replicate_values, CI_QUANTILES)
    return MetricCI(
        value=value,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_replicates_used=len(replicate_values),
    )
