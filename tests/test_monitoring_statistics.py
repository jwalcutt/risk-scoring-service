"""The three drift statistics, over synthetic arrays and nothing else.

The rules these tests pin:

- The KS statistic is the largest gap between two right-continuous ECDFs
  over the merged sample's distinct values, so a tied or capped column
  gives the true value rather than an artifact of the sort order.
- Its p-value is the asymptotic Kolmogorov series, clamped to the unit
  interval and saturating at 1 below the argument where the series stops
  converging.
- The flag test is a pooled two-proportion z with a two-sided tail through
  ``erfc``, and zero pooled variance is no evidence rather than a division
  by zero.
- PSI bins come from the reference's quantiles alone, collapse
  deterministically when the reference ties, and substitute an epsilon on
  both sides so an empty bin cannot make the sum infinite.
- A statistic that cannot be computed from the arrays given is ``None``. A
  non-finite value in an input array is a different thing and raises.
- Every returned number is a builtin, every result serializes as jsonb
  with no NaN, and no call mutates or reorders its inputs.

Expected values were computed independently and cross-checked against
``scipy.stats`` in a throwaway script; scipy is not a dependency of this
repository and nothing here imports it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import numpy.typing as npt
import pytest

from risk_scoring.monitoring import statistics as st


def _floats(*values: float) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


# --- the Kolmogorov series ---


@pytest.mark.parametrize(
    ("argument", "survival"),
    [(1.22385, 0.10), (1.35810, 0.05), (1.62762, 0.01), (1.94947, 0.001)],
)
def test_kolmogorov_sf_matches_published_critical_values(argument: float, survival: float) -> None:
    assert st.kolmogorov_sf(argument) == pytest.approx(survival, abs=1e-4)


@pytest.mark.parametrize("argument", [0.0, -1.0, 0.01])
def test_kolmogorov_sf_is_exactly_one_at_and_below_the_convergence_floor(
    argument: float,
) -> None:
    """The series does not converge below about 0.043, where the true value is 1."""
    assert st.kolmogorov_sf(argument) == 1.0


def test_kolmogorov_sf_never_leaves_the_unit_interval() -> None:
    """Without the clamp the alternating sum exceeds 1 by about 1.6e-15 near 0.05."""
    values = np.array([st.kolmogorov_sf(float(t)) for t in np.linspace(0.0, 12.0, 12001)])
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_kolmogorov_sf_is_monotone_above_the_convergence_floor() -> None:
    """Catches a sign error in the alternation. Below 0.07 it is float noise against 1."""
    values = np.array([st.kolmogorov_sf(float(t)) for t in np.linspace(0.2, 12.0, 5000)])
    assert np.all(np.diff(values) <= 0.0)


def test_kolmogorov_sf_matches_its_leading_term_far_in_the_tail() -> None:
    assert st.kolmogorov_sf(4.0) == 2 * math.exp(-32)


def test_kolmogorov_sf_refuses_a_non_finite_argument() -> None:
    with pytest.raises(ValueError, match="finite"):
        st.kolmogorov_sf(float("nan"))


# --- the normal tail ---


def test_normal_cdf_matches_known_values() -> None:
    assert st.normal_cdf(0.0) == 0.5
    assert st.normal_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert st.normal_cdf(-1.0) == pytest.approx(0.158655253931, abs=1e-12)


def test_two_sided_normal_p_is_exactly_one_at_zero() -> None:
    assert st.two_sided_normal_p(0.0) == 1.0


def test_two_sided_normal_p_stays_positive_where_one_minus_erf_collapses() -> None:
    """This is why the tail is erfc and not 1 - cdf: the naive form returns 0.0 here."""
    z = math.sqrt(120.0)
    assert st.two_sided_normal_p(z) == pytest.approx(6.326068263677e-28, rel=1e-9)
    assert 2 * (1 - st.normal_cdf(z)) == 0.0


def test_two_sided_normal_p_is_symmetric_in_the_sign_of_z() -> None:
    assert st.two_sided_normal_p(-2.5) == st.two_sided_normal_p(2.5)


# --- KS under ties ---


def test_ks_statistic_is_zero_when_tied_samples_have_the_same_shape() -> None:
    """A merged-cumsum shortcut reports 0.5 here, and which 0.5 depends on the sort."""
    result = st.ks_two_sample(_floats(0.0, 0.0, 1.0, 1.0), _floats(0.0, 1.0))
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.n_distinct == 2


def test_ks_statistic_matches_a_hand_computed_mass_point_case() -> None:
    # F_ref(1) = 3/4 against F_win(1) = 1/4, so D = 1/2 and n_effective = 2.
    # sqrt(2) * 1/2 gives 2t^2 = 1, and the series is 2e^-1 - 2e^-4 + 2e^-9 - ...
    result = st.ks_two_sample(_floats(1.0, 1.0, 1.0, 2.0), _floats(1.0, 2.0, 2.0, 2.0))
    assert result.statistic == 0.5
    assert result.n_effective == 2.0
    assert result.p_value == pytest.approx(0.6993741991310155, rel=1e-12)


def test_ks_statistic_finds_the_supremum_at_a_value_only_the_window_holds() -> None:
    """Evaluating the ECDFs on the reference's values alone would report 0.25."""
    result = st.ks_two_sample(_floats(0.0, 1.0, 2.0, 3.0), _floats(1.5, 1.5, 1.5))
    assert result.statistic == 0.5


def test_ks_is_symmetric_in_its_arguments() -> None:
    """Catches a mismatched side= that would manufacture a gap at every tie."""
    reference, window = _floats(1.0, 1.0, 1.0, 2.0), _floats(1.0, 2.0, 2.0, 2.0)
    forward = st.ks_two_sample(reference, window)
    backward = st.ks_two_sample(window, reference)
    assert forward.statistic == backward.statistic
    assert forward.p_value == backward.p_value
    assert (forward.n_reference, forward.n_window) == (backward.n_window, backward.n_reference)


def test_ks_is_unaffected_by_the_order_of_the_inputs() -> None:
    reference, window = np.arange(50.0), np.arange(20.0, 40.0)
    shuffled = st.ks_two_sample(reference[::-1], window[::-1])
    assert shuffled == st.ks_two_sample(reference, window)


def test_ks_sees_a_shift_in_the_discharge_gap_cap_mass() -> None:
    """The 365-day cap costs no power: the gap is attained just below it."""
    reference = np.concatenate((np.full(5000, 10.0), np.full(5000, 365.0)))
    result = st.ks_two_sample(reference, np.full(50, 365.0))
    assert result.statistic == 0.5
    assert result.n_distinct == 2
    assert result.p_value is not None and result.p_value < 1e-9


def test_ks_over_five_rows_against_twelve_thousand_claims_nothing() -> None:
    """The sparse case that matters: a real statistic, no significance, no exception."""
    reference = np.arange(12000.0)
    result = st.ks_two_sample(reference, reference[::2400])
    assert result.n_window == 5
    # The difference of two divisions lands one ulp off the exact 2399/12000.
    assert result.statistic == pytest.approx(2399 / 12000, rel=1e-12)
    assert result.p_value == pytest.approx(0.9883440232, abs=1e-9)
    assert result.n_effective == pytest.approx(4.997917534360683, rel=1e-12)


def test_ks_flags_a_disjoint_window() -> None:
    result = st.ks_two_sample(np.arange(100.0), np.arange(200.0, 220.0))
    assert result.statistic == 1.0
    assert result.p_value == pytest.approx(6.6764755907e-15, rel=1e-6)


def test_ks_p_value_does_not_underflow_at_the_real_sample_sizes() -> None:
    """A log-scale panel needs the extreme case to stay above zero."""
    result = st.ks_two_sample(np.arange(11294.0), np.arange(20000.0, 20060.0))
    assert result.statistic == 1.0
    assert result.p_value is not None and result.p_value > 0.0
    assert result.p_value == pytest.approx(2.891310e-52, rel=1e-5)


def test_ks_effective_size_is_set_by_the_window_not_the_reference() -> None:
    """Growing the reference from 11,294 to a million buys half a percent."""
    small = st.ks_two_sample(np.arange(11294.0), np.arange(60.0))
    large = st.ks_two_sample(np.arange(1_000_000.0), np.arange(60.0))
    assert small.n_effective == pytest.approx(59.68293112559451, rel=1e-12)
    assert large.n_effective is not None and large.n_effective < 60.0


@pytest.mark.parametrize("empty_side", ["reference", "window"])
def test_ks_returns_none_for_an_empty_sample(empty_side: str) -> None:
    full, empty = np.arange(10.0), np.asarray([], dtype=np.float64)
    reference, window = (empty, full) if empty_side == "reference" else (full, empty)
    result = st.ks_two_sample(reference, window)
    assert result.statistic is None
    assert result.p_value is None
    assert result.n_effective is None
    assert (result.n_reference, result.n_window) == (reference.size, window.size)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_ks_refuses_a_non_finite_value(bad: float) -> None:
    """A NaN is a defect in a stored array, not a fact about the window."""
    with pytest.raises(ValueError, match="finite"):
        st.ks_two_sample(_floats(1.0, bad), _floats(1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        st.ks_two_sample(_floats(1.0, 2.0), _floats(1.0, bad))


def test_ks_refuses_an_array_that_is_not_one_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        st.ks_two_sample(np.zeros((3, 2)), _floats(1.0))


# --- the two-proportion test ---


def test_two_proportion_matches_a_hand_computed_z_and_p() -> None:
    # 20 of 100 against 10 of 20: pooled 30/120 = 0.25, variance
    # 0.25 * 0.75 * (1/100 + 1/20) = 0.01125, and 0.3^2 / 0.01125 = 8.
    result = st.two_proportion_test(_binary(100, 20), _binary(20, 10))
    assert result.reference_rate == 0.2
    assert result.window_rate == 0.5
    assert result.z == pytest.approx(2 * math.sqrt(2), rel=1e-12)
    assert result.p_value == pytest.approx(math.erfc(2.0), rel=1e-12)
    assert (result.k_reference, result.k_window) == (20, 10)


def test_two_proportion_is_exactly_one_when_the_rates_agree() -> None:
    result = st.two_proportion_test(_binary(100, 50), _binary(20, 10))
    assert result.z == 0.0
    assert result.p_value == 1.0


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_two_proportion_treats_two_constant_samples_as_no_evidence(value: float) -> None:
    """Pooled variance is exactly zero. The samples agree in all this test can see."""
    result = st.two_proportion_test(np.full(100, value), np.full(20, value))
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_two_proportion_flags_a_flag_that_appears_from_nowhere() -> None:
    result = st.two_proportion_test(np.zeros(100), np.ones(20))
    assert result.z == pytest.approx(math.sqrt(120.0), rel=1e-12)
    assert result.p_value is not None and 0.0 < result.p_value < 1e-25


def test_two_proportion_p_value_can_underflow_at_the_real_reference_size() -> None:
    """Unlike the KS tail this one does reach exactly zero, and that is recorded."""
    result = st.two_proportion_test(np.zeros(11294), np.ones(20))
    assert result.z == pytest.approx(106.36728820459793, rel=1e-9)
    assert result.p_value == 0.0


def test_two_proportion_is_computable_when_only_the_window_is_all_zero() -> None:
    """The common rare-flag case, and not the degenerate one above."""
    result = st.two_proportion_test(_binary(100, 20), np.zeros(20))
    assert result.z == pytest.approx(-2.1908902300206643, rel=1e-12)
    assert result.p_value == pytest.approx(0.028459736916310596, rel=1e-12)


def test_two_proportion_z_follows_the_window() -> None:
    above = st.two_proportion_test(_binary(100, 20), _binary(20, 10))
    below = st.two_proportion_test(_binary(100, 50), _binary(20, 2))
    assert above.z is not None and above.z > 0.0
    assert below.z is not None and below.z < 0.0


@pytest.mark.parametrize("empty_side", ["reference", "window"])
def test_two_proportion_returns_none_for_an_empty_sample(empty_side: str) -> None:
    full, empty = _binary(20, 5), np.asarray([], dtype=np.float64)
    reference, window = (empty, full) if empty_side == "reference" else (full, empty)
    result = st.two_proportion_test(reference, window)
    assert result.z is None
    assert result.p_value is None
    if empty_side == "reference":
        assert result.reference_rate is None
        assert result.window_rate == 0.25
    else:
        assert result.reference_rate == 0.25
        assert result.window_rate is None


@pytest.mark.parametrize("bad", [2.0, 0.5])
def test_two_proportion_refuses_a_column_that_is_not_binary(bad: float) -> None:
    """Without this a misrouted count column gives a negative variance."""
    with pytest.raises(ValueError, match="0 or 1"):
        st.two_proportion_test(_floats(0.0, 1.0, bad), _floats(0.0, 1.0))


def test_two_proportion_names_a_non_finite_value_before_it_names_the_binary_rule() -> None:
    """A NaN is a defect in a stored array, which is the more specific complaint."""
    with pytest.raises(ValueError, match="finite"):
        st.two_proportion_test(_floats(0.0, 1.0, float("nan")), _floats(0.0, 1.0))


# --- population stability index ---


def test_psi_is_exactly_zero_for_identical_samples() -> None:
    reference = np.arange(100.0)
    result = st.population_stability_index(reference, reference.copy())
    assert result.value == 0.0
    assert result.n_bins == 10
    assert all(b.contribution == 0.0 for b in result.bins)


def test_psi_matches_a_hand_computed_two_bin_split() -> None:
    # Two bins each holding half the reference; the window puts 3/4 in the
    # first and 1/4 in the second. 0.25*ln(1.5) + 0.25*ln(2) = 0.25*ln(3).
    result = st.population_stability_index(np.arange(100.0), _floats(0.0, 0.0, 0.0, 99.0), n_bins=2)
    assert result.value == pytest.approx(0.25 * math.log(3), rel=1e-12)
    assert result.n_bins == 2


def test_psi_matches_a_hand_computed_case_with_empty_bins() -> None:
    # Five of ten deciles hold no window row. 3.45 of the 3.80 below is the
    # epsilon substitution rather than anything measured.
    expected = 0.5 * math.log(2) + 5 * (0.1 - st.PSI_EPSILON) * math.log(0.1 / st.PSI_EPSILON)
    result = st.population_stability_index(np.arange(100.0), np.arange(50.0))
    assert result.value == pytest.approx(expected, rel=1e-12)
    assert sum(1 for b in result.bins if b.window_fraction == 0.0) == 5


def test_psi_bins_are_the_reference_deciles_with_finite_edges() -> None:
    """An infinite outer edge would be unserializable, so the index is clipped instead."""
    result = st.population_stability_index(np.arange(1000.0), np.arange(500.0))
    assert all(b.reference_fraction == pytest.approx(0.1) for b in result.bins)
    assert all(math.isfinite(b.lower) and math.isfinite(b.upper) for b in result.bins)
    assert all(
        earlier.upper == later.lower
        for earlier, later in zip(result.bins, result.bins[1:], strict=False)
    )


def test_psi_places_a_window_value_outside_the_reference_range_in_an_end_bin() -> None:
    result = st.population_stability_index(np.arange(100.0), _floats(-50.0, 500.0), n_bins=2)
    assert result.bins[0].window_fraction == 0.5
    assert result.bins[-1].window_fraction == 0.5


@pytest.mark.parametrize("n_window", [5, 20, 60])
def test_psi_bin_count_does_not_depend_on_the_window(n_window: int) -> None:
    """Deriving it from the window would make consecutive dashboard points incomparable."""
    result = st.population_stability_index(np.arange(1000.0), np.arange(float(n_window)))
    assert result.n_bins == 10


def test_psi_collapses_tied_reference_edges_and_reports_the_realized_count() -> None:
    """A tree ensemble's leaf values tie, so this is the ordinary case, not a corner."""
    reference = np.asarray([0.03] * 90 + [0.5] * 10, dtype=np.float64)
    result = st.population_stability_index(reference, np.asarray([0.03] * 50 + [0.5] * 10))
    assert result.n_bins == 2
    assert tuple(b.reference_fraction for b in result.bins) == (0.9, 0.1)


def test_psi_stays_finite_when_a_reference_bin_is_empty() -> None:
    """An interpolated edge can bracket no reference row, which is why epsilon is symmetric."""
    reference = _floats(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0)
    result = st.population_stability_index(reference, _floats(0.7))
    assert result.n_bins == 4
    assert result.value is not None and math.isfinite(result.value)
    assert result.value == pytest.approx(17.474039417775078, rel=1e-9)
    assert any(b.reference_fraction == 0.0 for b in result.bins)


def test_psi_is_none_for_an_empty_window() -> None:
    """Letting the epsilon rule fire here would fabricate a PSI near 6.9 from no data."""
    result = st.population_stability_index(np.arange(100.0), np.asarray([], dtype=np.float64))
    assert result.value is None
    assert result.bins == ()


def test_psi_is_none_when_every_reference_score_is_identical() -> None:
    result = st.population_stability_index(np.full(100, 0.4), _floats(0.4, 0.5))
    assert result.value is None
    assert result.n_bins == 0


PSI_BATTERY: tuple[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], ...] = (
    (np.arange(100.0), np.arange(100.0).copy()),
    (np.arange(100.0), np.arange(50.0)),
    (np.arange(1000.0), np.arange(1000.0)[::20] + 300.0),
    (np.asarray([0.03] * 90 + [0.5] * 10), np.asarray([0.5] * 20)),
    (_floats(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0), _floats(0.7)),
)


@pytest.mark.parametrize(("reference", "window"), PSI_BATTERY)
def test_psi_value_is_the_sum_of_its_stored_contributions(
    reference: npt.NDArray[np.float64], window: npt.NDArray[np.float64]
) -> None:
    """The stored row checks itself."""
    result = st.population_stability_index(reference, window)
    assert result.value == pytest.approx(sum(b.contribution for b in result.bins), rel=1e-12)


@pytest.mark.parametrize(("reference", "window"), PSI_BATTERY)
def test_psi_is_never_negative(
    reference: npt.NDArray[np.float64], window: npt.NDArray[np.float64]
) -> None:
    """Both factors of every term share a sign, so each term is non-negative."""
    result = st.population_stability_index(reference, window)
    assert result.value is not None and result.value >= 0.0
    assert all(b.contribution >= 0.0 for b in result.bins)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    reference = np.arange(1000.0)
    values = [
        st.population_stability_index(reference, reference[::20] + shift).value
        for shift in (0.0, 100.0, 300.0, 900.0)
    ]
    assert all(value is not None for value in values)
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_psi_epsilon_stays_below_the_smallest_mass_a_window_can_express() -> None:
    """One real observation in a 60-row window must never look like an empty bin."""
    assert st.PSI_EPSILON < 1.0 / 60.0


def test_psi_refuses_fewer_than_two_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        st.population_stability_index(np.arange(10.0), np.arange(5.0), n_bins=1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_psi_refuses_a_non_finite_score(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        st.population_stability_index(_floats(0.1, bad), _floats(0.1, 0.2))


# --- cross-cutting ---


def _every_result() -> list[object]:
    empty = np.asarray([], dtype=np.float64)
    return [
        st.ks_two_sample(np.arange(100.0), np.arange(20.0)),
        st.ks_two_sample(np.arange(100.0), empty),
        st.ks_two_sample(_floats(0.0, 0.0, 1.0, 1.0), _floats(0.0, 1.0)),
        st.two_proportion_test(_binary(100, 20), _binary(20, 10)),
        st.two_proportion_test(np.zeros(100), np.zeros(20)),
        st.two_proportion_test(_binary(20, 5), empty),
        st.population_stability_index(np.arange(100.0), np.arange(50.0)),
        st.population_stability_index(np.arange(100.0), empty),
        st.population_stability_index(np.full(100, 0.4), _floats(0.4)),
        st.population_stability_index(np.asarray([0.03] * 90 + [0.5] * 10), np.asarray([0.5] * 20)),
    ]


def test_every_result_serializes_as_jsonb_with_no_nan() -> None:
    for result in _every_result():
        json.dumps(asdict(result), allow_nan=False)  # type: ignore[call-overload]


def test_every_numeric_field_is_a_builtin_and_not_a_numpy_scalar() -> None:
    """np.float64 subclasses float and serializes, so isinstance would miss the leak."""

    def check(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list | tuple):
            for item in value:
                check(item)
        else:
            assert type(value) in (float, int, str, bool, type(None)), (
                f"{value!r} is a {type(value)!r}, not a builtin"
            )

    for result in _every_result():
        check(asdict(result))  # type: ignore[call-overload]


def test_results_are_identical_across_repeated_calls_and_leave_the_inputs_alone() -> None:
    """The reference arrays are reused at every boundary, so an in-place sort would corrupt them."""
    reference, window = np.arange(100.0), np.arange(20.0, 40.0)
    reference_before, window_before = reference.copy(), window.copy()
    for call in (st.ks_two_sample, st.population_stability_index):
        assert call(reference, window) == call(reference, window)
    flags_reference, flags_window = _binary(100, 20), _binary(20, 10)
    assert st.two_proportion_test(flags_reference, flags_window) == st.two_proportion_test(
        flags_reference, flags_window
    )
    assert np.array_equal(reference, reference_before)
    assert np.array_equal(window, window_before)


def _binary(n: int, k: int) -> npt.NDArray[np.float64]:
    """An array of ``n`` values with ``k`` ones, for the proportion test."""
    out = np.zeros(n, dtype=np.float64)
    out[:k] = 1.0
    return out
