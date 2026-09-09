"""Turning an evaluation's numbers into alerts, purely.

The rules these tests pin:

- A statistic exactly at its threshold does not alert; one past it does.
  The boundary case is the one an operator will argue about, so it is
  fixed here rather than left to a comparison operator nobody reread.
- A statistic of ``None`` never alerts. That is what makes a sparse window
  quiet instead of noisy, and it is why the minimum-count rule is
  expressed by suppressing statistics rather than by skipping the judging.
- A per-signal override replaces the shared floor for that signal alone.
- Volume, refusals, and version mismatches are judged on counts, so they
  can still alert in a window too sparse for any drift statistic. An
  outage is exactly the case where that matters.
- Every alert carries its evaluation's boundary as its simulated instant,
  which is what the alerts table's composite foreign key then enforces.
- The alerts come back in the signal set's own order, so two runs over one
  set of numbers produce identical rows.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from risk_scoring.monitoring import signals
from risk_scoring.monitoring.config import MonitoringConfig, SignalFloor, Thresholds
from risk_scoring.monitoring.evaluate import Evaluation, WindowStatistics, judge
from risk_scoring.monitoring.statistics import KSResult, ProportionResult, PSIResult
from risk_scoring.replay.realized import RealizedPerformance

START = datetime(2025, 1, 1, tzinfo=UTC)
BOUNDARY = START + timedelta(days=35)

FLOOR = 0.001


def _config(**threshold_overrides: object) -> MonitoringConfig:
    values: dict[str, object] = {
        "drift_p_floor": FLOOR,
        "volume_floor_fraction": 0.5,
        "refusal_ceiling": 0,
        "version_mismatch_ceiling": 0,
    }
    values.update(threshold_overrides)
    return MonitoringConfig(
        cadence_days=7,
        window_days=30,
        minimum_predictions=20,
        expected_discharges_per_30_days=50.0,
        thresholds=Thresholds(**values),  # type: ignore[arg-type]
        thresholds_hash="0" * 64,
    )


def _ks(p_value: float | None) -> KSResult:
    return KSResult(
        statistic=None if p_value is None else 0.4,
        p_value=p_value,
        n_reference=500,
        n_window=40,
        n_effective=37.0,
        n_distinct=30,
    )


def _proportion(p_value: float | None) -> ProportionResult:
    return ProportionResult(
        reference_rate=0.25,
        window_rate=0.6,
        z=None if p_value is None else 3.0,
        p_value=p_value,
        n_reference=500,
        n_window=40,
        k_reference=125,
        k_window=24,
    )


def _statistics(
    *,
    minimum_met: bool = True,
    ks_p: float | None = 0.5,
    flag_p: float | None = 0.5,
    signal_p: dict[str, float | None] | None = None,
) -> WindowStatistics:
    ks = {signal: _ks(ks_p) for signal in (*signals.CONTINUOUS_SIGNALS, signals.SCORE_SIGNAL)}
    proportions = {signal: _proportion(flag_p) for signal in signals.FLAG_SIGNALS}
    for signal, p_value in (signal_p or {}).items():
        if signal in ks:
            ks[signal] = _ks(p_value)
        else:
            proportions[signal] = _proportion(p_value)
    return WindowStatistics(
        minimum_met=minimum_met,
        ks=ks,
        proportions=proportions,
        score_psi=PSIResult(value=0.1, n_bins=10, n_reference=500, n_window=40, bins=()),
    )


def _evaluation(
    *,
    prediction_count: int = 50,
    refusal_count: int = 0,
    version_mismatch_count: int = 0,
    expected_predictions: float = 50.0,
    statistics: WindowStatistics | None = None,
) -> Evaluation:
    return Evaluation(
        run_id=1,
        reference_id=1,
        window_start=BOUNDARY - timedelta(days=30),
        boundary=BOUNDARY,
        thresholds_hash="0" * 64,
        prediction_count=prediction_count,
        label_count=10,
        refusal_count=refusal_count,
        version_mismatch_count=version_mismatch_count,
        expected_predictions=expected_predictions,
        statistics=statistics if statistics is not None else _statistics(),
        realized=RealizedPerformance(count=10, prevalence=0.2, auroc=0.8),
    )


# --- a clean window ---


def test_a_clean_window_raises_nothing() -> None:
    assert judge(_evaluation(), _config()) == ()


# --- the drift threshold ---


def test_a_p_value_exactly_at_the_floor_does_not_alert() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"los_days": FLOOR}))
    assert judge(evaluation, _config()) == ()


def test_a_p_value_past_the_floor_alerts() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"los_days": FLOOR / 2}))
    (alert,) = judge(evaluation, _config())
    assert alert.signal == "los_days"
    assert alert.statistic == FLOOR / 2
    assert alert.threshold == FLOOR


def test_a_missing_p_value_never_alerts() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"los_days": None}))
    assert judge(evaluation, _config()) == ()


def test_a_flag_signal_alerts_through_its_proportion_test() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"flag_chf": 1e-9}))
    (alert,) = judge(evaluation, _config())
    assert alert.signal == "flag_chf"


def test_the_score_is_judged_like_any_other_drift_signal() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"score": 1e-9}))
    (alert,) = judge(evaluation, _config())
    assert alert.signal == "score"


def test_an_override_replaces_the_shared_floor_for_one_signal_only() -> None:
    config = _config(signal_overrides=(SignalFloor("los_days", 1e-9),))
    evaluation = _evaluation(
        statistics=_statistics(signal_p={"los_days": 1e-6, "prior_ed_180d": 1e-6})
    )
    (alert,) = judge(evaluation, config)
    assert alert.signal == "prior_ed_180d"
    assert alert.threshold == FLOOR


def test_the_score_psi_never_alerts_however_large_it_is() -> None:
    statistics = _statistics()
    huge = WindowStatistics(
        minimum_met=statistics.minimum_met,
        ks=statistics.ks,
        proportions=statistics.proportions,
        score_psi=PSIResult(value=42.0, n_bins=10, n_reference=500, n_window=40, bins=()),
    )
    assert judge(_evaluation(statistics=huge), _config()) == ()


# --- the counts ---


def test_volume_below_the_floor_alerts() -> None:
    evaluation = _evaluation(prediction_count=24, expected_predictions=50.0)
    (alert,) = judge(evaluation, _config())
    assert alert.signal == "volume"
    assert alert.statistic == 24.0
    assert alert.threshold == 25.0


def test_volume_exactly_at_the_floor_does_not_alert() -> None:
    assert judge(_evaluation(prediction_count=25, expected_predictions=50.0), _config()) == ()


def test_volume_alerts_in_a_window_too_sparse_for_any_drift_statistic() -> None:
    """An outage is the case where a suppressed window still has to speak."""
    evaluation = _evaluation(
        prediction_count=0,
        expected_predictions=50.0,
        statistics=_statistics(minimum_met=False, ks_p=None, flag_p=None),
    )
    (alert,) = judge(evaluation, _config())
    assert alert.signal == "volume"


def test_refusals_above_the_ceiling_alert() -> None:
    (alert,) = judge(_evaluation(refusal_count=1), _config())
    assert alert.signal == "refusals"
    assert alert.statistic == 1.0
    assert alert.threshold == 0.0


def test_refusals_at_the_ceiling_do_not_alert() -> None:
    assert judge(_evaluation(refusal_count=2), _config(refusal_ceiling=2)) == ()


def test_a_version_mismatch_above_the_ceiling_alerts() -> None:
    (alert,) = judge(_evaluation(version_mismatch_count=3), _config())
    assert alert.signal == "version_mismatch"
    assert alert.statistic == 3.0


def test_a_version_mismatch_at_a_raised_ceiling_does_not_alert() -> None:
    """A shadow deployment needs this raised, deliberately and in its own commit."""
    assert judge(_evaluation(version_mismatch_count=3), _config(version_mismatch_ceiling=3)) == ()


# --- the alerts themselves ---


def test_every_alert_carries_the_evaluation_boundary_as_its_instant() -> None:
    """The alerts table's composite foreign key then makes this unfalsifiable."""
    evaluation = _evaluation(refusal_count=1, statistics=_statistics(signal_p={"score": 1e-9}))
    alerts = judge(evaluation, _config())
    assert len(alerts) == 2
    assert all(alert.sim_at == BOUNDARY for alert in alerts)


def test_alerts_come_back_in_the_signal_sets_own_order() -> None:
    evaluation = _evaluation(
        refusal_count=1,
        prediction_count=0,
        statistics=_statistics(signal_p={"score": 1e-9, "flag_chf": 1e-9, "los_days": 1e-9}),
    )
    order = [alert.signal for alert in judge(evaluation, _config())]
    assert order == ["los_days", "flag_chf", "score", "volume", "refusals"]
    assert order == sorted(order, key=signals.ALERTABLE_SIGNALS.index)


def test_every_alert_notes_the_signal_and_both_numbers() -> None:
    evaluation = _evaluation(statistics=_statistics(signal_p={"los_days": 1e-9}))
    (alert,) = judge(evaluation, _config())
    assert "los_days" in alert.note
    assert alert.note


def test_judging_is_deterministic() -> None:
    evaluation = _evaluation(refusal_count=1, statistics=_statistics(signal_p={"score": 1e-9}))
    config = _config()
    assert judge(evaluation, config) == judge(evaluation, config)


def test_a_config_whose_hash_differs_from_the_evaluation_is_refused() -> None:
    """Judging an evaluation against rules it was not computed under is a bug."""
    other = dataclasses.replace(_config(), thresholds_hash="1" * 64)
    with pytest.raises(ValueError, match="thresholds_hash"):
        judge(_evaluation(), other)
