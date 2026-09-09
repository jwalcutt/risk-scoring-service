"""One boundary's evaluation: what the window held, and what it says.

An evaluation is a pure function of the tables and the boundary. Every
query here is bounded by the boundary and never by "now", so a boundary
evaluated while the harness is still posting and the same boundary
evaluated over the finished tables produce equal rows. That is the
property the whole monitoring layer rests on, and it is why the label
join takes a ``released_by`` bound rather than reading whatever has
matured by the time the query runs.

Judgment calls this module fixes:

- The grid is a pure function of the run's span and the cadence, and a
  boundary handed to :func:`evaluate` must lie on it. An arbitrary instant
  would produce a perfectly valid-looking row that no second evaluation
  could reproduce, and the unique index on ``(run_id, boundary)`` would
  then be guarding a set nobody defined.
- A run whose length is not a whole number of cadences leaves its last few
  days unevaluated. That is what a fixed grid does; stretching the final
  window to reach the end would make one window a different size from the
  other fifty-one and silently change what its statistics mean.
- ``prediction_count`` is the window's, while ``label_count`` counts the
  labels released inside the window. Two different questions: how much the
  service scored lately, and how much ground truth arrived lately. A label
  pipeline that stalls shows in the second and not the first.
- Realized performance is cumulative over the run to the boundary, not
  over the drift window. Labels mature 30 simulated days after discharge
  and the drift window is the last 30 days, so a realized metric over that
  window would be empty at every boundary by construction. Cumulative with
  a ``released_by`` bound is both non-empty and reproducible, and it is the
  series a reader wants as labels mature.
- Below the minimum prediction count every drift statistic is ``None``,
  the score's PSI with them, and the signal keys stay present so a sparse
  window leaves a gap in a series rather than removing the series. The
  count signals are unaffected, which is the point: an outage is exactly
  the window too sparse to compute drift on.
- A feature column in the log that the reference does not carry, or the
  reverse, raises. Both mean the log and the registered model disagree
  about what a feature row is, which is a version skew the version
  mismatch count cannot express because it compares version strings.
- Feature versions are compared by major and minor, reusing
  ``service.compatibility.major_minor``, because the feature module
  defines a patch bump as moving no value. A version string this code
  cannot parse counts as a mismatch, matching how the startup guard treats
  the same case.
- Refusals are counted by a function that returns zero, because no table
  records them yet. The count is a column and a threshold now so that
  adding the table changes this one function and no schema.
- ``judge`` refuses a config whose digest is not the one the evaluation was
  computed under. An evaluation row names its rules; judging it against
  different rules would produce alerts that the row cannot account for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import psycopg

from risk_scoring.monitoring import signals
from risk_scoring.monitoring.config import MonitoringConfig
from risk_scoring.monitoring.reference import Reference
from risk_scoring.monitoring.statistics import (
    KSResult,
    ProportionResult,
    PSIResult,
    ks_two_sample,
    population_stability_index,
    two_proportion_test,
)
from risk_scoring.replay.realized import RealizedPerformance, realized_performance
from risk_scoring.replay.runs import ReplayRun
from risk_scoring.service.compatibility import major_minor

REALIZED_KEY = "realized"
"""The one key in the statistics payload that is not a signal."""


@dataclass(frozen=True)
class WindowStatistics:
    """Every statistic a window produced, or the absence of them all."""

    minimum_met: bool
    ks: dict[str, KSResult]
    proportions: dict[str, ProportionResult]
    score_psi: PSIResult


@dataclass(frozen=True)
class Evaluation:
    """One boundary's row, before it is written."""

    run_id: int
    reference_id: int
    window_start: datetime
    boundary: datetime
    thresholds_hash: str
    prediction_count: int
    label_count: int
    refusal_count: int
    version_mismatch_count: int
    expected_predictions: float
    statistics: WindowStatistics
    realized: RealizedPerformance

    def p_value_for(self, signal: str) -> float | None:
        """The p-value for one drift signal, whichever test produced it."""
        if signal in self.statistics.ks:
            return self.statistics.ks[signal].p_value
        if signal in self.statistics.proportions:
            return self.statistics.proportions[signal].p_value
        raise ValueError(f"{signal} is not a drift signal of this evaluation")

    def statistics_payload(self) -> dict[str, Any]:
        """The ``statistics`` jsonb: every reported signal, keyed by name."""
        payload: dict[str, Any] = {}
        for signal, ks in self.statistics.ks.items():
            payload[signal] = {"test": "ks", **asdict(ks)}
        for signal, proportion in self.statistics.proportions.items():
            payload[signal] = {"test": "two_proportion", **asdict(proportion)}
        payload[signals.SCORE_PSI_SIGNAL] = {
            "test": "psi",
            **asdict(self.statistics.score_psi),
        }
        payload[signals.VOLUME_SIGNAL] = {
            "test": "count",
            "count": self.prediction_count,
            "expected": self.expected_predictions,
        }
        payload[signals.REFUSAL_SIGNAL] = {"test": "count", "count": self.refusal_count}
        payload[signals.VERSION_MISMATCH_SIGNAL] = {
            "test": "count",
            "count": self.version_mismatch_count,
        }
        payload[REALIZED_KEY] = {"test": "realized", **asdict(self.realized)}
        return payload


@dataclass(frozen=True)
class Alert:
    """One signal that crossed its threshold at one boundary."""

    signal: str
    statistic: float
    threshold: float
    sim_at: datetime
    note: str


def boundaries(
    start_at: datetime, end_at: datetime, config: MonitoringConfig
) -> tuple[datetime, ...]:
    """Every boundary of a run: its start plus whole cadences, never past its end."""
    if end_at <= start_at:
        raise ValueError(f"a run's start must be before its end; got {start_at} and {end_at}")
    grid: list[datetime] = []
    step = 1
    while (boundary := start_at + config.cadence * step) <= end_at:
        grid.append(boundary)
        step += 1
    return tuple(grid)


def window_for(
    boundary: datetime, start_at: datetime, config: MonitoringConfig
) -> tuple[datetime, datetime]:
    """The half-open window a boundary evaluates, clipped at the run's start."""
    if boundary <= start_at:
        raise ValueError(
            f"a window must end after the run starts; got boundary {boundary} and start {start_at}"
        )
    return max(start_at, boundary - config.window), boundary


def compare(
    reference: Reference,
    window_features: Mapping[str, Sequence[float]],
    window_scores: Sequence[float],
    *,
    minimum_predictions: int,
) -> WindowStatistics:
    """Every drift statistic for one window, or every one suppressed."""
    unknown = set(window_features) - set(reference.features)
    if unknown:
        raise ValueError(
            f"the window carries feature columns the reference does not:"
            f" {', '.join(sorted(unknown))}"
        )
    missing = set(signals.CONTINUOUS_SIGNALS + signals.FLAG_SIGNALS) - set(reference.features)
    if missing:
        raise ValueError(f"the reference carries no {', '.join(sorted(missing))}")
    n_window = len(window_scores)
    minimum_met = n_window >= minimum_predictions
    scores = _array(window_scores)
    reference_scores = _array(reference.scores)

    ks: dict[str, KSResult] = {}
    for signal in signals.CONTINUOUS_SIGNALS:
        ks[signal] = _ks_or_suppressed(
            _array(reference.features[signal]),
            _array(window_features.get(signal, ())),
            suppressed=not minimum_met,
        )
    ks[signals.SCORE_SIGNAL] = _ks_or_suppressed(
        reference_scores, scores, suppressed=not minimum_met
    )

    proportions: dict[str, ProportionResult] = {}
    for signal in signals.FLAG_SIGNALS:
        proportions[signal] = _proportion_or_suppressed(
            _array(reference.features[signal]),
            _array(window_features.get(signal, ())),
            suppressed=not minimum_met,
        )

    if minimum_met:
        score_psi = population_stability_index(reference_scores, scores)
    else:
        score_psi = PSIResult(
            value=None,
            n_bins=0,
            n_reference=int(reference_scores.size),
            n_window=n_window,
            bins=(),
        )
    return WindowStatistics(
        minimum_met=minimum_met, ks=ks, proportions=proportions, score_psi=score_psi
    )


def evaluate(
    conn: psycopg.Connection[Any],
    run: ReplayRun,
    reference: Reference,
    reference_id: int,
    boundary: datetime,
    config: MonitoringConfig,
) -> Evaluation:
    """One boundary's evaluation, read entirely from below the boundary."""
    if boundary not in boundaries(run.start_at, run.end_at, config):
        raise ValueError(
            f"{boundary} is not a boundary of run {run.run_id};"
            f" boundaries sit at {run.start_at} plus whole multiples of {config.cadence}"
        )
    window_start, _ = window_for(boundary, run.start_at, config)
    window_features, window_scores, versions = _read_window(conn, window_start, boundary)
    return Evaluation(
        run_id=run.run_id,
        reference_id=reference_id,
        window_start=window_start,
        boundary=boundary,
        thresholds_hash=config.thresholds_hash,
        prediction_count=len(window_scores),
        label_count=_label_count(conn, window_start, boundary),
        refusal_count=refusal_count(conn, window_start, boundary),
        version_mismatch_count=_version_mismatch_count(versions, reference),
        expected_predictions=config.expected_predictions(boundary - window_start),
        statistics=compare(
            reference,
            window_features,
            window_scores,
            minimum_predictions=config.minimum_predictions,
        ),
        realized=realized_performance(conn, run.start_at, boundary, released_by=boundary),
    )


def judge(evaluation: Evaluation, config: MonitoringConfig) -> tuple[Alert, ...]:
    """Every signal of one evaluation that crossed its threshold."""
    if evaluation.thresholds_hash != config.thresholds_hash:
        raise ValueError(
            f"this evaluation was computed under thresholds_hash"
            f" {evaluation.thresholds_hash}, not {config.thresholds_hash}"
        )
    thresholds = config.thresholds
    alerts: list[Alert] = []
    for signal in signals.DRIFT_SIGNALS:
        p_value = evaluation.p_value_for(signal)
        floor = thresholds.p_floor_for(signal)
        if p_value is not None and p_value < floor:
            alerts.append(
                _alert(
                    signal,
                    p_value,
                    floor,
                    evaluation.boundary,
                    f"{signal} differs from the reference: p {p_value:.3g} below {floor:.3g}",
                )
            )
    volume_floor = thresholds.volume_floor_fraction * evaluation.expected_predictions
    if evaluation.prediction_count < volume_floor:
        alerts.append(
            _alert(
                signals.VOLUME_SIGNAL,
                float(evaluation.prediction_count),
                volume_floor,
                evaluation.boundary,
                f"{evaluation.prediction_count} predictions against"
                f" {evaluation.expected_predictions:.1f} expected,"
                f" below the floor of {volume_floor:.1f}",
            )
        )
    if evaluation.refusal_count > thresholds.refusal_ceiling:
        alerts.append(
            _alert(
                signals.REFUSAL_SIGNAL,
                float(evaluation.refusal_count),
                float(thresholds.refusal_ceiling),
                evaluation.boundary,
                f"{evaluation.refusal_count} refused events, above the ceiling of"
                f" {thresholds.refusal_ceiling}",
            )
        )
    if evaluation.version_mismatch_count > thresholds.version_mismatch_ceiling:
        alerts.append(
            _alert(
                signals.VERSION_MISMATCH_SIGNAL,
                float(evaluation.version_mismatch_count),
                float(thresholds.version_mismatch_ceiling),
                evaluation.boundary,
                f"{evaluation.version_mismatch_count} predictions are not the reference's"
                f" model or feature version, above the ceiling of"
                f" {thresholds.version_mismatch_ceiling}",
            )
        )
    return tuple(alerts)


def refusal_count(conn: psycopg.Connection[Any], window_start: datetime, boundary: datetime) -> int:
    """Events the service refused inside the window, by their stream instant.

    Nothing records a refusal yet: the harness stops on one. The signal has
    a column, a threshold, and this seam so that recording refusals changes
    this function and nothing else.
    """
    return 0


def _read_window(
    conn: psycopg.Connection[Any], window_start: datetime, boundary: datetime
) -> tuple[dict[str, list[float]], list[float], list[tuple[int, str]]]:
    rows = conn.execute(
        "SELECT features, score, model_version, feature_version FROM predictions"
        " WHERE event_time >= %s AND event_time < %s ORDER BY prediction_id",
        [window_start, boundary],
    ).fetchall()
    features: dict[str, list[float]] = {}
    scores: list[float] = []
    versions: list[tuple[int, str]] = []
    for stored, score, model_version, feature_version in rows:
        for column, value in stored.items():
            features.setdefault(column, []).append(float(value))
        scores.append(float(score))
        versions.append((int(model_version), str(feature_version)))
    return features, scores, versions


def _label_count(conn: psycopg.Connection[Any], window_start: datetime, boundary: datetime) -> int:
    """Labels released inside the window: how much ground truth arrived lately."""
    row = conn.execute(
        "SELECT count(*) FROM labels WHERE released_at >= %s AND released_at < %s",
        [window_start, boundary],
    ).fetchone()
    return 0 if row is None else int(row[0])


def _version_mismatch_count(versions: Sequence[tuple[int, str]], reference: Reference) -> int:
    return sum(
        1
        for model_version, feature_version in versions
        if model_version != reference.model_version
        or not _same_series(feature_version, reference.feature_version)
    )


def _same_series(one: str, other: str) -> bool:
    try:
        return major_minor(one) == major_minor(other)
    except ValueError:
        # A version this code cannot read is never treated as a match.
        return False


def _alert(signal: str, statistic: float, threshold: float, sim_at: datetime, note: str) -> Alert:
    return Alert(
        signal=signal,
        statistic=float(statistic),
        threshold=float(threshold),
        sim_at=sim_at,
        note=note,
    )


def _ks_or_suppressed(
    reference: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
    *,
    suppressed: bool,
) -> KSResult:
    if not suppressed:
        return ks_two_sample(reference, window)
    return KSResult(
        statistic=None,
        p_value=None,
        n_reference=int(reference.size),
        n_window=int(window.size),
        n_effective=None,
        n_distinct=0,
    )


def _proportion_or_suppressed(
    reference: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
    *,
    suppressed: bool,
) -> ProportionResult:
    if not suppressed:
        return two_proportion_test(reference, window)
    return ProportionResult(
        reference_rate=None,
        window_rate=None,
        z=None,
        p_value=None,
        n_reference=int(reference.size),
        n_window=int(window.size),
        k_reference=int(np.count_nonzero(reference)),
        k_window=int(np.count_nonzero(window)),
    )


def _array(values: Sequence[float]) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)
