"""The evaluation grid, the window, and the comparison, with no database.

The rules these tests pin:

- Boundaries sit at the run's start plus whole multiples of the cadence,
  starting one cadence in, and never past the run's end. A run's last few
  days go unevaluated when its length is not a multiple of the cadence,
  which is a property of a fixed grid and is stated rather than patched.
- A window is half-open at the boundary and clipped at the run's start, so
  the first boundaries of a run see short windows.
- Below the minimum prediction count every drift statistic is ``None``,
  the score's PSI with them, while the counts on the result stay real. The
  signal keys are all still present, so a sparse window leaves a gap in a
  dashboard series rather than removing the series.
- The comparison is a pure function of the two samples and reads the
  reference's arrays without reordering them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from risk_scoring.monitoring import signals
from risk_scoring.monitoring.config import MonitoringConfig, Thresholds
from risk_scoring.monitoring.evaluate import boundaries, compare, window_for
from risk_scoring.monitoring.reference import Reference

START = datetime(2025, 1, 1, tzinfo=UTC)


def _config(**overrides: object) -> MonitoringConfig:
    values: dict[str, object] = {
        "cadence_days": 7,
        "window_days": 30,
        "minimum_predictions": 20,
        "expected_discharges_per_30_days": 50.0,
        "thresholds": Thresholds(
            drift_p_floor=0.001,
            volume_floor_fraction=0.5,
            refusal_ceiling=0,
            version_mismatch_ceiling=0,
        ),
        "thresholds_hash": "0" * 64,
    }
    values.update(overrides)
    return MonitoringConfig(**values)  # type: ignore[arg-type]


def _reference_features(n_rows: int) -> dict[str, list[float]]:
    continuous = {
        column: [float(row % 50) for row in range(n_rows)] for column in signals.CONTINUOUS_SIGNALS
    }
    flags = {
        column: [float(row % 4 == 0) for row in range(n_rows)] for column in signals.FLAG_SIGNALS
    }
    return continuous | flags


def _reference(n_rows: int = 500) -> Reference:
    """A reference whose columns are easy to shift a window away from."""
    return Reference(
        model_name="readmission-risk",
        model_version=4,
        feature_version="1.1.0",
        cohort_version="1.0.0",
        training_cutoff=START,
        split_seed=20260101,
        n_train_rows=n_rows,
        n_holdout_rows=n_rows,
        features=_reference_features(n_rows),
        scores=[0.02 + 0.001 * (row % 200) for row in range(n_rows)],
    )


def _window(n_rows: int, *, shift: float = 0.0) -> tuple[dict[str, list[float]], list[float]]:
    features = {
        column: [float(row % 50) + shift for row in range(n_rows)]
        for column in signals.CONTINUOUS_SIGNALS
    }
    features |= {
        column: [float(row % 4 == 0) for row in range(n_rows)] for column in signals.FLAG_SIGNALS
    }
    scores = [0.02 + 0.001 * (row % 200) + shift for row in range(n_rows)]
    return features, scores


# --- the grid ---


def test_the_first_boundary_is_one_cadence_into_the_run() -> None:
    """Nothing is evaluated at the run's start, where no window exists yet."""
    grid = boundaries(START, START + timedelta(days=90), _config())
    assert grid[0] == START + timedelta(days=7)


def test_boundaries_step_by_the_cadence() -> None:
    grid = boundaries(START, START + timedelta(days=90), _config())
    assert grid == tuple(START + timedelta(days=7 * k) for k in range(1, 13))


def test_a_boundary_exactly_at_the_run_end_is_evaluated() -> None:
    """Its window is complete data, so there is no reason to drop it."""
    grid = boundaries(START, START + timedelta(days=84), _config())
    assert grid[-1] == START + timedelta(days=84)


def test_a_run_length_that_is_not_a_multiple_of_the_cadence_leaves_a_tail() -> None:
    """A fixed grid cannot cover the last six days. Stated, not patched."""
    end = START + timedelta(days=90)
    grid = boundaries(START, end, _config())
    assert grid[-1] == START + timedelta(days=84)
    assert end - grid[-1] == timedelta(days=6)


def test_a_run_shorter_than_one_cadence_has_no_boundaries() -> None:
    assert boundaries(START, START + timedelta(days=6), _config()) == ()


def test_the_grid_follows_the_configured_cadence() -> None:
    grid = boundaries(START, START + timedelta(days=30), _config(cadence_days=10))
    assert grid == tuple(START + timedelta(days=10 * k) for k in (1, 2, 3))


def test_the_grid_requires_an_ordered_span() -> None:
    with pytest.raises(ValueError, match="before"):
        boundaries(START, START, _config())


# --- the window ---


def test_a_full_window_reaches_back_the_configured_length() -> None:
    boundary = START + timedelta(days=35)
    assert window_for(boundary, START, _config()) == (boundary - timedelta(days=30), boundary)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_an_early_window_is_clipped_at_the_run_start(k: int) -> None:
    """The first four boundaries of a 7-day, 30-day grid see short windows."""
    boundary = START + timedelta(days=7 * k)
    assert window_for(boundary, START, _config()) == (START, boundary)


def test_the_fifth_window_is_the_first_full_one() -> None:
    boundary = START + timedelta(days=35)
    window_start, _ = window_for(boundary, START, _config())
    assert window_start > START


def test_a_window_must_end_after_the_run_starts() -> None:
    with pytest.raises(ValueError, match="after"):
        window_for(START, START, _config())


# --- the minimum count ---


def test_a_window_at_the_minimum_reports_real_statistics() -> None:
    """At the threshold, not past it: twenty predictions is enough."""
    features, scores = _window(20)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert result.minimum_met
    assert all(value.p_value is not None for value in result.ks.values())
    assert result.score_psi.value is not None


def test_a_window_below_the_minimum_reports_no_statistic_at_all() -> None:
    features, scores = _window(19)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert not result.minimum_met
    assert all(value.p_value is None for value in result.ks.values())
    assert all(value.statistic is None for value in result.ks.values())
    assert all(value.p_value is None for value in result.proportions.values())
    assert result.score_psi.value is None


def test_a_suppressed_window_still_names_every_signal() -> None:
    """A sparse window leaves a gap in a series rather than removing the series."""
    features, scores = _window(5)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert set(result.ks) == {*signals.CONTINUOUS_SIGNALS, signals.SCORE_SIGNAL}
    assert set(result.proportions) == set(signals.FLAG_SIGNALS)


def test_a_suppressed_window_still_reports_its_counts() -> None:
    """The row has to say how sparse it was, or the None is unreadable."""
    features, scores = _window(5)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert result.ks["los_days"].n_window == 5
    assert result.ks["los_days"].n_reference == 500
    assert result.proportions["flag_chf"].n_window == 5


def test_an_empty_window_is_suppressed_and_does_not_raise() -> None:
    result = compare(_reference(), {}, [], minimum_predictions=20)
    assert not result.minimum_met
    assert result.ks["score"].n_window == 0


def test_a_minimum_of_zero_lets_an_empty_window_through_to_the_none_rules() -> None:
    """With no minimum the statistics themselves refuse an empty sample."""
    result = compare(_reference(), {}, [], minimum_predictions=0)
    assert result.minimum_met
    assert result.ks["score"].p_value is None


# --- the comparison ---


def test_an_identical_window_is_not_flagged() -> None:
    reference = _reference()
    features = {column: list(values) for column, values in reference.features.items()}
    result = compare(reference, features, list(reference.scores), minimum_predictions=20)
    assert result.ks["los_days"].statistic == 0.0
    assert result.score_psi.value == 0.0


def test_a_shifted_window_is_flagged_on_the_shifted_columns() -> None:
    features, scores = _window(60, shift=200.0)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert result.ks["los_days"].p_value == pytest.approx(0.0, abs=1e-12)
    assert result.ks["score"].p_value == pytest.approx(0.0, abs=1e-12)


def test_the_flags_are_compared_by_proportion_and_the_others_by_ks() -> None:
    features, scores = _window(60)
    result = compare(_reference(), features, scores, minimum_predictions=20)
    assert result.proportions["flag_chf"].reference_rate == pytest.approx(0.25)
    assert result.ks["age_at_discharge"].n_distinct > 0


def test_the_comparison_is_deterministic_and_leaves_the_reference_alone() -> None:
    reference = _reference()
    before = {column: list(values) for column, values in reference.features.items()}
    features, scores = _window(60)
    first = compare(reference, features, scores, minimum_predictions=20)
    second = compare(reference, features, scores, minimum_predictions=20)
    assert first == second
    assert {column: list(values) for column, values in reference.features.items()} == before


def test_a_window_column_the_reference_does_not_carry_is_refused() -> None:
    """A feature column that reached the log but not the reference is a version skew."""
    features, scores = _window(60)
    features["invented_column"] = [0.0] * 60
    with pytest.raises(ValueError, match="invented_column"):
        compare(_reference(), features, scores, minimum_predictions=20)


def test_a_reference_missing_a_signal_is_refused() -> None:
    reference = _reference()
    trimmed = dict(reference.features)
    del trimmed["los_days"]
    features, scores = _window(60)
    with pytest.raises(ValueError, match="los_days"):
        compare(
            Reference(**{**reference.__dict__, "features": trimmed}),
            features,
            scores,
            minimum_predictions=20,
        )
