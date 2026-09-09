"""Hand-rolled drift statistics comparing a window's sample to the reference's.

This module is ``evaluation.py``'s sibling: arithmetic over float arrays,
with no knowledge of the database, the reference row, boundaries, or
thresholds. Which column gets which test lives in ``signals.py`` and the
policy about what a sparse window may claim lives in ``evaluate.py``.

Judgment calls this module fixes:

- The KS statistic is the largest gap between two right-continuous
  empirical distribution functions, evaluated at the distinct values of
  the merged sample. Both samples are heavily tied. The counts are small
  integers, and ``days_since_prev_discharge`` has a mass point at 365.0
  that means both "capped" and "no prior discharge". The familiar
  shortcut, a running sum of steps over the concatenated sort, is exact
  only without cross-sample ties: for a reference of ``[0, 0, 1, 1]``
  against a window of ``[0, 1]`` the true gap is zero and the running sum
  wanders to a half inside the tied block, so the answer is not even a
  function of the two samples. Both ECDFs take their full jump before
  being differenced, which is why both searches use ``side="right"``.
- The cap costs no power, which is worth stating because it looks as
  though it should. Both ECDFs reach 1 at 365.0, so a change in the
  no-prior-discharge rate is detected at the largest value below the cap,
  where the gap equals the difference in cap mass exactly.
- The p-value is the asymptotic Kolmogorov series, truncated when a term
  stops mattering and clamped to the unit interval. The clamp is not
  decorative: alternating-series cancellation pushes the unclamped sum
  above 1 by about 1.6e-15 for arguments near 0.05, and a p-value above 1
  would reach jsonb. Below about 0.043 the series does not converge and
  the true value is 1 to within float64, so the term cap saturates there.
- Read the KS p-value as a monotone drift score, not as a false-alarm
  rate. Because 1/n_effective is 1/n + 1/m, the effective size never
  exceeds the window's own: 11,294 reference rows against a 60-row window
  give 59.68, half a percent better than an infinite reference. The window
  alone sets the resolution, and at 60 rows the smallest gap detectable at
  the asymptotic five percent point is 0.176 of the distribution. Two
  biases then push the same way, toward under-alerting: the asymptotic
  form is roughly ten to fifteen percent too generous near p = 0.05 at
  these sizes, and ties shrink the gap achievable under the null, which
  also gives the sparse count columns a p-value with a handful of atoms
  rather than a smooth null. The alternative that fixes the tie problem
  exactly is a fixed-seed permutation p-value, deterministic and
  dependency-free; it is not built because the thresholds are calibrated
  from the measured distribution over a clean run either way.
- The flag test is a pooled two-proportion z, and its two-sided tail goes
  through ``math.erfc`` rather than ``1 - normal_cdf``. The naive form
  collapses to exactly zero past about eight standard errors, which is
  reachable here: a flag going from absent to universal in a 20-row window
  against a 100-row reference gives z near 11. The tail does still
  underflow to zero against a real 11,000-row reference, where z is
  near 106. That is recorded rather than clamped, because a floor invented
  to keep a log axis happy would be a number nobody measured.
- Zero pooled variance is no evidence, not a division by zero. Two samples
  that are all zero, or all one, agree in the only respect this statistic
  can see, so z is 0 and p is 1. The counts on the result make the
  degeneracy plain without a flag field.
- The two-proportion normal approximation wants the pooled count at five
  or more, which at a 60-row window needs a rate above eight percent.
  Several comorbidity flags are rarer than that, so this test sits outside
  its validity range exactly where a case-mix shift would show. The
  statistic is computed anyway and the four counts are exposed, so a
  reader can apply the rule; suppressing rare flags is policy and does not
  belong here. Fisher's exact test would remove the weakness through
  ``math.lgamma`` with no new dependency, and is the named alternative.
- PSI bins are the reference's own quantiles. Ties in the reference
  collapse edges, exactly, through ``np.unique``: a tree ensemble's leaf
  values do repeat, so this is ordinary rather than a corner case, and the
  realized bin count is reported because a sum over seven bins is not
  comparable to one over ten. Collapsing with a tolerance instead would
  add a parameter and make the count depend on the dtype.
- Bin assignment is unbounded at both ends, since a window pushing past
  the reference's range is drift, but the stored edges stay finite. An
  infinite outer edge is the obvious way to say "unbounded" and it poisons
  the row, because ``allow_nan=False`` rejects an infinity as hard as a
  NaN. Clipping the index rather than infinitizing the edge is what keeps
  both properties.
- The epsilon substituted for an empty bin is applied to both sides. A
  reference bin can be empty even after exact deduplication, because an
  interpolated quantile edge need not bracket any reference row, and a
  one-sided epsilon then takes the logarithm of a ratio with zero
  underneath. With both sides floored, every ratio lies between epsilon
  and its reciprocal, so the sum is finite as a property of the rule
  rather than by luck. The value 1e-4 is the largest conventional choice
  that still sits far below the smallest mass a 60-row window can express,
  1/60, so a bin holding one real observation can never read as empty, and
  a fixed value keeps the series comparable to itself across boundaries
  where a rule like 0.5/n would not.
- An empty window short-circuits to ``None`` before the epsilon rule runs.
  Letting it fire over ten bins fabricates a PSI near 6.9 out of no data,
  which reads as catastrophic drift.
- ``evaluation.calibration_bins`` is deliberately not reused for PSI. It
  bins by rank position, which cannot place a window value in a reference
  bin, and under ties adjacent rank bins share a value so the assignment
  would be ambiguous. PSI needs value edges.
- A statistic that cannot be computed from the arrays given is ``None``,
  following ``replay.realized_performance`` rather than
  ``evaluation.bootstrap_ci``, which raises: a monitor reads sparse
  windows on every evaluation and an uncomputable metric there is a fact.
  A non-finite value in an input array is a different category. It is a
  defect in a stored array, so it raises here, where the value still has a
  name, rather than surfacing as a jsonb parse error at the insert.
- Every numeric field is converted to a builtin. ``np.int64`` fails
  ``json.dumps`` loudly, but ``np.float64`` subclasses ``float`` and
  serializes, which makes a leak latent instead. Nothing sorts in place,
  because the reference arrays are read once and reused at every boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

KS_SERIES_TERMS = 100
"""Terms before the Kolmogorov series gives up and saturates at 1."""

KS_TERM_RATIO_EPSILON = 1e-6
KS_TERM_SUM_EPSILON = 1e-16

PSI_BINS = 10
"""Reference quantile bins, matching ``evaluation.ECE_BINS`` and the decile habit."""

PSI_EPSILON = 1e-4
"""The mass substituted for an empty bin, on both sides. See the module docstring."""

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class KSResult:
    """A two-sample KS comparison of one window column against the reference."""

    statistic: float | None
    p_value: float | None
    n_reference: int
    n_window: int
    n_effective: float | None
    n_distinct: int


@dataclass(frozen=True)
class ProportionResult:
    """A pooled two-proportion z test of one window flag against the reference."""

    reference_rate: float | None
    window_rate: float | None
    z: float | None
    p_value: float | None
    n_reference: int
    n_window: int
    k_reference: int
    k_window: int


@dataclass(frozen=True)
class PSIBin:
    """One reference-quantile bin: its edges, both masses, and its term of the sum."""

    lower: float
    upper: float
    reference_fraction: float
    window_fraction: float
    contribution: float


@dataclass(frozen=True)
class PSIResult:
    """The score's population stability index over the reference's own bins."""

    value: float | None
    n_bins: int
    n_reference: int
    n_window: int
    bins: tuple[PSIBin, ...]


def kolmogorov_sf(t: float) -> float:
    """The Kolmogorov limiting survival function, clamped to the unit interval."""
    if not math.isfinite(t):
        raise ValueError(f"the Kolmogorov series needs a finite argument; got {t!r}")
    if t <= 0.0:
        return 1.0
    exponent = -2.0 * t * t
    total = 0.0
    sign = 2.0
    previous = 0.0
    for k in range(1, KS_SERIES_TERMS + 1):
        term = sign * math.exp(exponent * k * k)
        total += term
        if abs(term) <= KS_TERM_RATIO_EPSILON * previous or abs(term) <= KS_TERM_SUM_EPSILON * abs(
            total
        ):
            return min(1.0, max(0.0, total))
        sign = -sign
        previous = abs(term)
    # Only reachable below about t = 0.043, where the true value is 1.0.
    return 1.0


def normal_cdf(x: float) -> float:
    """The standard normal distribution function, through ``math.erf``."""
    if not math.isfinite(x):
        raise ValueError(f"the normal CDF needs a finite argument; got {x!r}")
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def two_sided_normal_p(z: float) -> float:
    """The two-sided normal tail for ``z``, through ``erfc`` so it cannot reach zero early."""
    if not math.isfinite(z):
        raise ValueError(f"the normal tail needs a finite argument; got {z!r}")
    return math.erfc(abs(z) / _SQRT2)


def ks_two_sample(
    reference: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
) -> KSResult:
    """Tie-correct two-sample KS statistic and its asymptotic p-value."""
    _require_sample(reference, "reference")
    _require_sample(window, "window")
    n_reference, n_window = int(reference.size), int(window.size)
    if n_reference == 0 or n_window == 0:
        return KSResult(
            statistic=None,
            p_value=None,
            n_reference=n_reference,
            n_window=n_window,
            n_effective=None,
            n_distinct=0,
        )
    reference_sorted = np.sort(reference)
    window_sorted = np.sort(window)
    grid = np.unique(np.concatenate((reference_sorted, window_sorted)))
    # side="right" on both, so a tied value has taken its full jump in each
    # ECDF before the difference is taken.
    f_reference = np.searchsorted(reference_sorted, grid, side="right") / n_reference
    f_window = np.searchsorted(window_sorted, grid, side="right") / n_window
    statistic = float(np.max(np.abs(f_reference - f_window)))
    n_effective = (n_reference * n_window) / (n_reference + n_window)
    return KSResult(
        statistic=statistic,
        p_value=kolmogorov_sf(math.sqrt(n_effective) * statistic),
        n_reference=n_reference,
        n_window=n_window,
        n_effective=float(n_effective),
        n_distinct=int(grid.size),
    )


def two_proportion_test(
    reference: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
) -> ProportionResult:
    """Pooled two-proportion z test over two arrays of zeros and ones."""
    _require_binary(reference, "reference")
    _require_binary(window, "window")
    n_reference, n_window = int(reference.size), int(window.size)
    k_reference = int(np.count_nonzero(reference))
    k_window = int(np.count_nonzero(window))
    reference_rate = k_reference / n_reference if n_reference else None
    window_rate = k_window / n_window if n_window else None
    if reference_rate is None or window_rate is None:
        return ProportionResult(
            reference_rate=reference_rate,
            window_rate=window_rate,
            z=None,
            p_value=None,
            n_reference=n_reference,
            n_window=n_window,
            k_reference=k_reference,
            k_window=k_window,
        )
    pooled = (k_reference + k_window) / (n_reference + n_window)
    variance = pooled * (1.0 - pooled) * (1.0 / n_reference + 1.0 / n_window)
    if variance <= 0.0:
        # Both samples constant and equal: no evidence of a difference.
        z, p_value = 0.0, 1.0
    else:
        z = (window_rate - reference_rate) / math.sqrt(variance)
        p_value = two_sided_normal_p(z)
    return ProportionResult(
        reference_rate=reference_rate,
        window_rate=window_rate,
        z=float(z),
        p_value=float(p_value),
        n_reference=n_reference,
        n_window=n_window,
        k_reference=k_reference,
        k_window=k_window,
    )


def population_stability_index(
    reference: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
    *,
    n_bins: int = PSI_BINS,
    epsilon: float = PSI_EPSILON,
) -> PSIResult:
    """PSI of the window against the reference, over the reference's quantile bins."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2; got {n_bins!r}")
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be strictly between 0 and 1; got {epsilon!r}")
    _require_sample(reference, "reference")
    _require_sample(window, "window")
    n_reference, n_window = int(reference.size), int(window.size)
    empty = PSIResult(value=None, n_bins=0, n_reference=n_reference, n_window=n_window, bins=())
    if n_reference == 0 or n_window == 0:
        return empty
    probabilities = np.linspace(0.0, 1.0, n_bins + 1)
    # method="linear" named explicitly: a future numpy default must not
    # silently redefine a statistic already stored on an evaluation row.
    edges = np.unique(
        np.asarray(np.quantile(reference, probabilities, method="linear"), dtype=np.float64)
    )
    if edges.size < 2:
        return empty
    realized_bins = int(edges.size - 1)
    # Searching the interior edges only returns a bin index directly, and
    # leaves both ends unbounded while the stored edges stay finite.
    interior = edges[1:-1]
    reference_counts = np.bincount(
        np.searchsorted(interior, reference, side="right"), minlength=realized_bins
    )
    window_counts = np.bincount(
        np.searchsorted(interior, window, side="right"), minlength=realized_bins
    )
    bins: list[PSIBin] = []
    total = 0.0
    for index in range(realized_bins):
        reference_fraction = float(reference_counts[index]) / n_reference
        window_fraction = float(window_counts[index]) / n_window
        floored_reference = max(reference_fraction, epsilon)
        floored_window = max(window_fraction, epsilon)
        contribution = (floored_window - floored_reference) * math.log(
            floored_window / floored_reference
        )
        total += contribution
        bins.append(
            PSIBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                reference_fraction=reference_fraction,
                window_fraction=window_fraction,
                contribution=contribution,
            )
        )
    return PSIResult(
        value=float(total),
        n_bins=realized_bins,
        n_reference=n_reference,
        n_window=n_window,
        bins=tuple(bins),
    )


def _require_sample(values: npt.NDArray[np.float64], label: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"the {label} sample must be one-dimensional; got shape {values.shape}")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError(
            f"the {label} sample holds a value that is not finite;"
            f" a NaN or an infinity in a stored array is a defect, not a window's shape"
        )


def _require_binary(values: npt.NDArray[np.float64], label: str) -> None:
    _require_sample(values, label)
    if values.size and not bool(((values == 0.0) | (values == 1.0)).all()):
        raise ValueError(
            f"the {label} sample must hold only 0 or 1; a flag test over other values"
            f" gives a pooled rate above 1 and a negative variance"
        )
