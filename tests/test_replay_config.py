"""Replay config: which population, over which simulated span, how fast.

The rules these tests pin:

- The committed defaults equal the training cutoff and the generation
  reference date, read from their own sources, so the three cannot drift.
- Dates are unquoted TOML dates. A quoted string, a datetime, or a
  missing key is refused loudly rather than guessed at.
- Range rules (start before end, positive acceleration, splices strictly
  inside the span and strictly increasing) hold for a loaded file and for
  a file with command-line overrides applied, through the same code.
- Max speed is a pacing choice, not configuration: the config has no
  such field.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import date, datetime
from pathlib import Path

import pytest

from risk_scoring.datagen import config as generation_config
from risk_scoring.replay.config import (
    DEFAULT_CONFIG_RELPATH,
    ReplayConfig,
    Splice,
    add_config_arguments,
    apply_overrides,
    load_config,
)
from risk_scoring.train import TRAINING_CUTOFF

_REPO_ROOT = Path(__file__).resolve().parent.parent

VALID = """
[replay]
population = "baseline"
start = 2025-01-01
end = 2026-01-01
acceleration = 4
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "replay.toml"
    path.write_text(body)
    return path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    return parser.parse_args(argv)


# The committed file


def test_committed_defaults_match_the_cutoff_and_the_reference_date() -> None:
    config = load_config(_REPO_ROOT / "configs" / "replay.toml")
    generation = generation_config.load_config(_REPO_ROOT / "configs" / "generation.toml")
    reference = datetime.strptime(generation.generation.reference_date, "%Y%m%d").date()

    assert config.start.isoformat() == TRAINING_CUTOFF
    assert config.end == reference
    assert config.population == "baseline"
    assert config.acceleration == 4
    assert config.splices == ()


def test_default_config_relpath_is_the_committed_file() -> None:
    assert DEFAULT_CONFIG_RELPATH.as_posix() == "configs/replay.toml"
    assert (_REPO_ROOT / DEFAULT_CONFIG_RELPATH).is_file()


# Loading


def test_valid_file_loads(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID))
    assert config == ReplayConfig(
        population="baseline",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        acceleration=4.0,
    )
    assert isinstance(config.acceleration, float)


def test_splices_load_in_file_order(tmp_path: Path) -> None:
    body = (
        VALID
        + """
[[splice]]
at = 2025-04-01
population = "care_protocol"

[[splice]]
at = 2025-09-01
population = "demographic_shift"
"""
    )
    config = load_config(_write(tmp_path, body))
    assert config.splices == (
        Splice(at=date(2025, 4, 1), population="care_protocol"),
        Splice(at=date(2025, 9, 1), population="demographic_shift"),
    )


def test_missing_replay_table_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, '[other]\npopulation = "baseline"\n'))


@pytest.mark.parametrize("key", ["population", "start", "end", "acceleration"])
def test_missing_key_raises(tmp_path: Path, key: str) -> None:
    body = "\n".join(line for line in VALID.splitlines() if not line.startswith(f"{key} "))
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, body))


def test_splice_missing_population_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, VALID + "\n[[splice]]\nat = 2025-04-01\n"))


@pytest.mark.parametrize("key", ["start", "end"])
def test_quoted_date_is_rejected(tmp_path: Path, key: str) -> None:
    body = "\n".join(
        line if not line.startswith(f"{key} ") else f'{key} = "{line.split(" = ")[1]}"'
        for line in VALID.splitlines()
    )
    with pytest.raises(ValueError, match=rf"replay\.{key} must be an unquoted TOML date"):
        load_config(_write(tmp_path, body))


def test_datetime_valued_date_is_rejected(tmp_path: Path) -> None:
    body = VALID.replace("start = 2025-01-01", "start = 2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match=r"replay\.start must be an unquoted TOML date"):
        load_config(_write(tmp_path, body))


def test_quoted_splice_date_is_rejected(tmp_path: Path) -> None:
    body = VALID + '\n[[splice]]\nat = "2025-04-01"\npopulation = "care_protocol"\n'
    with pytest.raises(ValueError, match=r"splice\.at must be an unquoted TOML date"):
        load_config(_write(tmp_path, body))


@pytest.mark.parametrize("value", ["true", "0", "-4", '"4"'])
def test_bad_acceleration_is_rejected(tmp_path: Path, value: str) -> None:
    body = VALID.replace("acceleration = 4", f"acceleration = {value}")
    with pytest.raises(ValueError, match="acceleration"):
        load_config(_write(tmp_path, body))


def test_fractional_acceleration_is_accepted(tmp_path: Path) -> None:
    body = VALID.replace("acceleration = 4", "acceleration = 0.5")
    assert load_config(_write(tmp_path, body)).acceleration == 0.5


def test_empty_population_is_rejected(tmp_path: Path) -> None:
    body = VALID.replace('population = "baseline"', 'population = ""')
    with pytest.raises(ValueError, match="population"):
        load_config(_write(tmp_path, body))


@pytest.mark.parametrize("end", ["2025-01-01", "2024-12-31"])
def test_end_must_follow_start(tmp_path: Path, end: str) -> None:
    body = VALID.replace("end = 2026-01-01", f"end = {end}")
    with pytest.raises(ValueError, match="end must be after"):
        load_config(_write(tmp_path, body))


@pytest.mark.parametrize("at", ["2024-12-01", "2025-01-01", "2026-01-01", "2026-02-01"])
def test_splice_outside_the_span_is_rejected(tmp_path: Path, at: str) -> None:
    body = VALID + f'\n[[splice]]\nat = {at}\npopulation = "care_protocol"\n'
    with pytest.raises(ValueError, match="strictly inside"):
        load_config(_write(tmp_path, body))


@pytest.mark.parametrize("second", ["2025-04-01", "2025-03-01"])
def test_splices_must_strictly_increase(tmp_path: Path, second: str) -> None:
    body = (
        VALID
        + '\n[[splice]]\nat = 2025-04-01\npopulation = "care_protocol"\n'
        + f'\n[[splice]]\nat = {second}\npopulation = "demographic_shift"\n'
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_config(_write(tmp_path, body))


def test_empty_splice_population_is_rejected(tmp_path: Path) -> None:
    body = VALID + '\n[[splice]]\nat = 2025-04-01\npopulation = ""\n'
    with pytest.raises(ValueError, match="population"):
        load_config(_write(tmp_path, body))


# Command-line overrides


def test_config_argument_defaults_to_the_committed_relpath() -> None:
    assert _parse([]).config == DEFAULT_CONFIG_RELPATH


def test_no_overrides_leaves_the_config_equal(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID))
    assert apply_overrides(config, _parse([])) == config


def test_each_override_applies_alone(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID))

    assert apply_overrides(config, _parse(["--population", "care_protocol"])) == ReplayConfig(
        population="care_protocol", start=config.start, end=config.end, acceleration=4.0
    )
    assert apply_overrides(config, _parse(["--start", "2025-03-01"])).start == date(2025, 3, 1)
    assert apply_overrides(config, _parse(["--end", "2025-07-01"])).end == date(2025, 7, 1)
    assert apply_overrides(config, _parse(["--acceleration", "16"])).acceleration == 16.0


def test_overrides_keep_the_splices(tmp_path: Path) -> None:
    body = VALID + '\n[[splice]]\nat = 2025-04-01\npopulation = "care_protocol"\n'
    config = load_config(_write(tmp_path, body))
    assert apply_overrides(config, _parse(["--end", "2025-07-01"])).splices == config.splices


def test_an_override_that_breaks_the_span_is_rejected(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID))
    with pytest.raises(ValueError, match="end must be after"):
        apply_overrides(config, _parse(["--end", "2024-06-01"]))


def test_an_override_that_strands_a_splice_is_rejected(tmp_path: Path) -> None:
    body = VALID + '\n[[splice]]\nat = 2025-09-01\npopulation = "care_protocol"\n'
    config = load_config(_write(tmp_path, body))
    with pytest.raises(ValueError, match="strictly inside"):
        apply_overrides(config, _parse(["--end", "2025-07-01"]))


def test_a_malformed_date_override_is_an_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse(["--start", "March 1st"])
    assert "--start" in capsys.readouterr().err


def test_max_speed_is_not_configuration() -> None:
    """Pacing must not change outputs, so it never lives beside the stream definition."""
    assert "max_speed" not in {field.name for field in fields(ReplayConfig)}
