"""The monitoring config: the grid, the counts, and the alert thresholds.

The rules these tests pin:

- The committed values, so a change to any of them is a visible diff and a
  test failure rather than a quiet retune.
- Every number in the committed file is still a placeholder. The clean
  twelve-month replay measures the false-alarm rate and sets these; until
  then the file must say so, and this test is what will fail when it does.
- Range and relationship rules live on the dataclasses, so a config built
  in code is validated exactly like one read from a file.
- A threshold override naming a signal the code does not know is refused.
  Every other config in this repository ignores a key it does not
  recognize; this one cannot, because its values become the evidence that
  the thresholds were fixed before any incident fired, and a misspelled
  key that silently takes the shared default defeats that.
- The hash on every evaluation row is the file's own digest, so a changed
  threshold is visible in the data and not only in a commit.
"""

from __future__ import annotations

import tomllib
from datetime import timedelta
from pathlib import Path

import pytest

from risk_scoring.datagen.manifest import sha256_file
from risk_scoring.monitoring.config import (
    DEFAULT_CONFIG_RELPATH,
    MonitoringConfig,
    SignalFloor,
    Thresholds,
    load_config,
)

COMMITTED = Path(DEFAULT_CONFIG_RELPATH)

VALID = """
[monitoring]
cadence_days = 7
window_days = 30
minimum_predictions = 20
expected_discharges_per_30_days = 50

[thresholds]
drift_p_floor = 0.001
volume_floor_fraction = 0.5
refusal_ceiling = 0
version_mismatch_ceiling = 0
"""


def _write(tmp_path: Path, text: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "monitoring.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _thresholds(**overrides: object) -> Thresholds:
    values: dict[str, object] = {
        "drift_p_floor": 0.001,
        "volume_floor_fraction": 0.5,
        "refusal_ceiling": 0,
        "version_mismatch_ceiling": 0,
    }
    values.update(overrides)
    return Thresholds(**values)  # type: ignore[arg-type]  # keyword soup for the invalid cases


def _config(**overrides: object) -> MonitoringConfig:
    values: dict[str, object] = {
        "cadence_days": 7,
        "window_days": 30,
        "minimum_predictions": 20,
        "expected_discharges_per_30_days": 50.0,
        "thresholds": _thresholds(),
        "thresholds_hash": "0" * 64,
    }
    values.update(overrides)
    return MonitoringConfig(**values)  # type: ignore[arg-type]  # keyword soup for the invalid cases


# --- the committed file ---


def test_committed_config_loads() -> None:
    config = load_config(COMMITTED)
    assert config.cadence_days == 7
    assert config.window_days == 30
    assert config.minimum_predictions == 20
    assert config.expected_discharges_per_30_days == 50.0


def test_committed_thresholds_are_the_recorded_values() -> None:
    thresholds = load_config(COMMITTED).thresholds
    assert thresholds.drift_p_floor == 0.001
    assert thresholds.volume_floor_fraction == 0.5
    assert thresholds.refusal_ceiling == 0
    assert thresholds.version_mismatch_ceiling == 0
    assert thresholds.signal_overrides == ()


def test_committed_values_are_still_marked_as_placeholders() -> None:
    """This test is what fails when the clean replay sets the real thresholds."""
    assert "placeholder" in COMMITTED.read_text(encoding="utf-8").lower()


def test_committed_cadence_and_window_match_the_recorded_grid() -> None:
    config = load_config(COMMITTED)
    assert config.cadence == timedelta(days=7)
    assert config.window == timedelta(days=30)


# --- the hash ---


def test_thresholds_hash_is_the_file_digest() -> None:
    assert load_config(COMMITTED).thresholds_hash == sha256_file(COMMITTED)


def test_a_changed_threshold_changes_the_hash(tmp_path: Path) -> None:
    first = load_config(_write(tmp_path / "a", VALID)).thresholds_hash
    second = load_config(
        _write(tmp_path / "b", VALID.replace("drift_p_floor = 0.001", "drift_p_floor = 0.01"))
    ).thresholds_hash
    assert first != second


# --- missing and mistyped values ---


def test_missing_monitoring_table_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, "[thresholds]\ndrift_p_floor = 0.001\n"))


@pytest.mark.parametrize(
    "key",
    [
        "cadence_days",
        "window_days",
        "minimum_predictions",
        "expected_discharges_per_30_days",
        "drift_p_floor",
        "volume_floor_fraction",
        "refusal_ceiling",
        "version_mismatch_ceiling",
    ],
)
def test_every_value_is_required(tmp_path: Path, key: str) -> None:
    """Nothing here has a default: a threshold nobody wrote down is not a threshold."""
    lines = [line for line in VALID.splitlines() if not line.startswith(f"{key} =")]
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, "\n".join(lines)))


@pytest.mark.parametrize("key", ["cadence_days", "window_days", "minimum_predictions"])
def test_a_bool_is_not_an_integer(tmp_path: Path, key: str) -> None:
    """bool is an int subclass, so it has to be refused by name."""
    text = "\n".join(
        f"{key} = true" if line.startswith(f"{key} = ") else line for line in VALID.splitlines()
    )
    with pytest.raises(ValueError, match=key):
        load_config(_write(tmp_path, text))


def test_a_string_threshold_is_refused(tmp_path: Path) -> None:
    text = VALID.replace("drift_p_floor = 0.001", 'drift_p_floor = "0.001"')
    with pytest.raises(ValueError, match="drift_p_floor"):
        load_config(_write(tmp_path, text))


# --- range and relationship rules ---


def test_cadence_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cadence_days"):
        _config(cadence_days=0)


def test_window_must_not_be_shorter_than_the_cadence() -> None:
    """A window shorter than the step leaves simulated time nobody ever reads."""
    with pytest.raises(ValueError, match="window_days"):
        _config(cadence_days=7, window_days=6)


def test_minimum_predictions_may_be_zero_but_not_negative() -> None:
    assert _config(minimum_predictions=0).minimum_predictions == 0
    with pytest.raises(ValueError, match="minimum_predictions"):
        _config(minimum_predictions=-1)


def test_expected_rate_must_be_positive() -> None:
    """It is a divisor, so zero is not a lenient setting but an error."""
    with pytest.raises(ValueError, match="expected_discharges_per_30_days"):
        _config(expected_discharges_per_30_days=0.0)


@pytest.mark.parametrize("floor", [0.0, 1.0, -0.1, 1.5])
def test_p_floor_must_be_strictly_inside_zero_and_one(floor: float) -> None:
    with pytest.raises(ValueError, match="drift_p_floor"):
        _thresholds(drift_p_floor=floor)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_volume_fraction_must_be_a_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="volume_floor_fraction"):
        _thresholds(volume_floor_fraction=fraction)


def test_a_volume_fraction_of_one_is_allowed() -> None:
    """Alerting on any shortfall at all is a strict setting, not an invalid one."""
    assert _thresholds(volume_floor_fraction=1.0).volume_floor_fraction == 1.0


@pytest.mark.parametrize("key", ["refusal_ceiling", "version_mismatch_ceiling"])
def test_ceilings_may_be_zero_but_not_negative(key: str) -> None:
    assert getattr(_thresholds(**{key: 0}), key) == 0
    with pytest.raises(ValueError, match=key):
        _thresholds(**{key: -1})


# --- per-signal overrides ---


def test_an_override_names_a_known_drift_signal() -> None:
    override = SignalFloor(signal="los_days", p_floor=0.0001)
    assert _thresholds(signal_overrides=(override,)).p_floor_for("los_days") == 0.0001


def test_a_signal_without_an_override_takes_the_shared_floor() -> None:
    thresholds = _thresholds(signal_overrides=(SignalFloor("los_days", 0.0001),))
    assert thresholds.p_floor_for("age_at_discharge") == 0.001


def test_an_unknown_override_signal_is_refused() -> None:
    """Every other config ignores a key it does not know. This one may not."""
    with pytest.raises(ValueError, match="los_dayz"):
        SignalFloor(signal="los_dayz", p_floor=0.0001)


def test_a_count_signal_cannot_take_a_p_value_override() -> None:
    """Volume and refusals are judged on counts; a p-value floor means nothing."""
    with pytest.raises(ValueError, match="volume"):
        SignalFloor(signal="volume", p_floor=0.0001)


def test_two_overrides_for_one_signal_are_refused() -> None:
    with pytest.raises(ValueError, match="los_days"):
        _thresholds(
            signal_overrides=(SignalFloor("los_days", 0.0001), SignalFloor("los_days", 0.001))
        )


def test_an_override_is_read_from_the_file(tmp_path: Path) -> None:
    text = VALID + "\n[thresholds.signal_overrides]\nlos_days = 0.0001\n"
    thresholds = load_config(_write(tmp_path, text)).thresholds
    assert thresholds.signal_overrides == (SignalFloor("los_days", 0.0001),)


def test_an_unknown_override_in_the_file_is_refused(tmp_path: Path) -> None:
    text = VALID + "\n[thresholds.signal_overrides]\nlos_dayz = 0.0001\n"
    with pytest.raises(ValueError, match="los_dayz"):
        load_config(_write(tmp_path, text))


def test_p_floor_for_refuses_a_signal_that_carries_no_p_value() -> None:
    with pytest.raises(ValueError, match="volume"):
        _thresholds().p_floor_for("volume")


# --- the expected-volume scaling ---


def test_expected_predictions_scales_with_the_window() -> None:
    """The first boundaries see short windows and must be compared fairly."""
    config = _config(expected_discharges_per_30_days=60.0)
    assert config.expected_predictions(timedelta(days=30)) == 60.0
    assert config.expected_predictions(timedelta(days=7)) == 14.0


def test_expected_predictions_refuses_an_empty_window() -> None:
    with pytest.raises(ValueError, match="window"):
        _config().expected_predictions(timedelta(0))


# --- the committed file's own shape ---


def test_committed_file_carries_no_unexpected_tables() -> None:
    """A stray table would be silently ignored, so the shape is pinned here."""
    with COMMITTED.open("rb") as fh:
        raw = tomllib.load(fh)
    assert set(raw) == {"monitoring", "thresholds"}
