"""Alert rules as data: the evaluation grid, the minimum count, the thresholds.

Judgment calls this module fixes:

- Nothing has a default. Every value is indexed out of the file, so a
  threshold nobody wrote down is a ``KeyError`` at load rather than a
  number the code chose. The service config gives its pool size a default
  because an operator has no opinion about it; a threshold is the opposite
  kind of value.
- A threshold override naming a signal this code does not know is refused,
  which no other config in this repository does. The values here become
  the evidence that the thresholds were fixed before any failure was
  injected, and a misspelled key that quietly takes the shared floor would
  leave that evidence saying something untrue. Only signals carrying a
  p-value may be overridden: volume and the two counts are judged on
  counts, so a p-value floor for them would parse and mean nothing.
- Range and relationship rules live on the dataclasses, so a config built
  in code and one read from a file go through the same validation. This
  module has no command-line overrides on purpose: a run that could
  override a threshold from the shell would make the committed file
  decorative.
- ``thresholds_hash`` is the digest of the whole file as read, not of the
  thresholds table alone. Every value in the file changes what an
  evaluation decides, and the point of the hash is that an evaluation row
  names the rules that produced it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from risk_scoring.datagen.manifest import sha256_file
from risk_scoring.monitoring import signals

DEFAULT_CONFIG_RELPATH = Path("configs/monitoring.toml")

REFERENCE_WINDOW = timedelta(days=30)
"""The window length ``expected_discharges_per_30_days`` is quoted over."""


@dataclass(frozen=True)
class SignalFloor:
    """A p-value floor for one drift signal, raised or lowered on its own."""

    signal: str
    p_floor: float

    def __post_init__(self) -> None:
        if self.signal not in signals.DRIFT_SIGNALS:
            raise ValueError(
                f"thresholds.signal_overrides.{self.signal} does not name a signal carrying a"
                f" p-value; the drift signals are {', '.join(signals.DRIFT_SIGNALS)}"
            )
        _require_probability(self.p_floor, f"thresholds.signal_overrides.{self.signal}")


@dataclass(frozen=True)
class Thresholds:
    """What each signal is judged against."""

    drift_p_floor: float
    volume_floor_fraction: float
    refusal_ceiling: int
    version_mismatch_ceiling: int
    signal_overrides: tuple[SignalFloor, ...] = ()

    def __post_init__(self) -> None:
        _require_probability(self.drift_p_floor, "thresholds.drift_p_floor")
        _require_fraction(self.volume_floor_fraction, "thresholds.volume_floor_fraction")
        _require_non_negative_int(self.refusal_ceiling, "thresholds.refusal_ceiling")
        _require_non_negative_int(
            self.version_mismatch_ceiling, "thresholds.version_mismatch_ceiling"
        )
        seen: set[str] = set()
        for override in self.signal_overrides:
            if override.signal in seen:
                raise ValueError(
                    f"thresholds.signal_overrides names {override.signal} twice;"
                    f" a signal has one floor"
                )
            seen.add(override.signal)

    def p_floor_for(self, signal: str) -> float:
        """The floor in force for one drift signal, override first."""
        if signal not in signals.DRIFT_SIGNALS:
            raise ValueError(
                f"{signal} carries no p-value, so it has no p-value floor;"
                f" the drift signals are {', '.join(signals.DRIFT_SIGNALS)}"
            )
        for override in self.signal_overrides:
            if override.signal == signal:
                return override.p_floor
        return self.drift_p_floor


@dataclass(frozen=True)
class MonitoringConfig:
    cadence_days: int
    window_days: int
    minimum_predictions: int
    expected_discharges_per_30_days: float
    thresholds: Thresholds
    thresholds_hash: str

    def __post_init__(self) -> None:
        _require_positive_int(self.cadence_days, "monitoring.cadence_days")
        _require_positive_int(self.window_days, "monitoring.window_days")
        if self.window_days < self.cadence_days:
            raise ValueError(
                f"monitoring.window_days must be at least monitoring.cadence_days, or simulated"
                f" time falls between windows and is never read; got window {self.window_days}"
                f" and cadence {self.cadence_days}"
            )
        _require_non_negative_int(self.minimum_predictions, "monitoring.minimum_predictions")
        rate = self.expected_discharges_per_30_days
        if isinstance(rate, bool) or not isinstance(rate, int | float) or rate <= 0:
            raise ValueError(
                f"monitoring.expected_discharges_per_30_days must be a positive number;"
                f" got {rate!r}"
            )
        if not self.thresholds_hash:
            raise ValueError("thresholds_hash must be the digest of the config in force")

    @property
    def cadence(self) -> timedelta:
        """The step between boundaries."""
        return timedelta(days=self.cadence_days)

    @property
    def window(self) -> timedelta:
        """How far back a boundary looks, before clipping at the run's start."""
        return timedelta(days=self.window_days)

    def expected_predictions(self, window: timedelta) -> float:
        """The discharge count a window of this length implies.

        Scaled by length rather than assumed to be a full window, because
        the first boundaries of a run see windows clipped at its start and
        comparing those against a 30-day expectation would read as an
        outage every time.
        """
        if window <= timedelta(0):
            raise ValueError(f"a window must have positive length; got {window}")
        return self.expected_discharges_per_30_days * (
            window.total_seconds() / REFERENCE_WINDOW.total_seconds()
        )


def load_config(path: Path) -> MonitoringConfig:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    monitoring = raw["monitoring"]
    thresholds = raw["thresholds"]
    overrides = tuple(
        SignalFloor(signal=signal, p_floor=_require_number(floor, f"thresholds.{signal}"))
        for signal, floor in thresholds.get("signal_overrides", {}).items()
    )
    return MonitoringConfig(
        cadence_days=monitoring["cadence_days"],
        window_days=monitoring["window_days"],
        minimum_predictions=monitoring["minimum_predictions"],
        expected_discharges_per_30_days=monitoring["expected_discharges_per_30_days"],
        thresholds=Thresholds(
            drift_p_floor=_require_number(thresholds["drift_p_floor"], "thresholds.drift_p_floor"),
            volume_floor_fraction=_require_number(
                thresholds["volume_floor_fraction"], "thresholds.volume_floor_fraction"
            ),
            refusal_ceiling=thresholds["refusal_ceiling"],
            version_mismatch_ceiling=thresholds["version_mismatch_ceiling"],
            signal_overrides=overrides,
        ),
        thresholds_hash=sha256_file(path),
    )


def _require_number(value: object, label: str) -> float:
    # bool is an int subclass, so check it explicitly.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number; got {value!r}")
    return float(value)


def _require_probability(value: object, label: str) -> None:
    number = _require_number(value, label)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{label} must be strictly between 0 and 1; got {number!r}")


def _require_fraction(value: object, label: str) -> None:
    number = _require_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a fraction between 0 and 1; got {number!r}")


def _require_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive whole number of days; got {value!r}")


def _require_non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a whole number of zero or more; got {value!r}")
