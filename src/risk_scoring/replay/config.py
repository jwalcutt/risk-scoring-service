"""Replay config: which population is streamed, over which span, how fast.

Judgment calls this module fixes:

- Dates are unquoted TOML dates. The parser types them, so a malformed
  date is refused before this module sees it; a quoted string or a
  datetime is refused here by name, matching how the service config
  treats its model version: an explicit type or a loud failure, never a
  guess.
- Range rules live on the dataclass, so a config assembled from a file
  and command-line overrides is validated by the same code as one read
  from a file alone.
- Max speed is not configuration. Pacing must not change what a run
  writes, so it never sits beside the values that define the stream.
- Splices are plain data read from the file: a simulated date and the
  population that takes over from it. Nothing here schedules anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_RELPATH = Path("configs/replay.toml")


@dataclass(frozen=True)
class Splice:
    """From ``at``, ``population`` replaces the current source of every event."""

    at: date
    population: str

    def __post_init__(self) -> None:
        _require_population(self.population, "splice.population")


@dataclass(frozen=True)
class ReplayConfig:
    population: str
    start: date
    end: date
    acceleration: float
    splices: tuple[Splice, ...] = ()

    def __post_init__(self) -> None:
        _require_population(self.population, "replay.population")
        if self.end <= self.start:
            raise ValueError(
                f"replay.end must be after replay.start; got start {self.start} and end {self.end}"
            )
        if self.acceleration <= 0:
            raise ValueError(
                f"replay.acceleration must be positive (simulated days per wall minute);"
                f" got {self.acceleration!r}"
            )
        previous: date | None = None
        for splice in self.splices:
            if not self.start < splice.at < self.end:
                raise ValueError(
                    f"splice.at must lie strictly inside the run ({self.start} to {self.end});"
                    f" got {splice.at}"
                )
            if previous is not None and splice.at <= previous:
                raise ValueError(
                    f"splice dates must be strictly increasing; got {splice.at} after {previous}"
                )
            previous = splice.at


def load_config(path: Path) -> ReplayConfig:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    replay = raw["replay"]
    acceleration = replay["acceleration"]
    # bool is an int subclass, so check it explicitly.
    if isinstance(acceleration, bool) or not isinstance(acceleration, int | float):
        raise ValueError(
            f"replay.acceleration must be a number (simulated days per wall minute);"
            f" got {acceleration!r}"
        )
    splices = tuple(
        Splice(at=_require_date(entry["at"], "splice.at"), population=entry["population"])
        for entry in raw.get("splice", [])
    )
    return ReplayConfig(
        population=replay["population"],
        start=_require_date(replay["start"], "replay.start"),
        end=_require_date(replay["end"], "replay.end"),
        acceleration=float(acceleration),
        splices=splices,
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """The config file and the per-value overrides a replay command accepts."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_RELPATH)
    parser.add_argument("--population", default=None, help="frozen population to stream")
    parser.add_argument(
        "--start", type=date.fromisoformat, default=None, help="first simulated day, YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None, help="simulated day the run ends on"
    )
    parser.add_argument(
        "--acceleration", type=float, default=None, help="simulated days per wall minute"
    )


def apply_overrides(config: ReplayConfig, args: argparse.Namespace) -> ReplayConfig:
    """The config with every override that was given, re-validated as a whole."""
    overrides: dict[str, Any] = {
        name: value
        for name in ("population", "start", "end", "acceleration")
        if (value := getattr(args, name)) is not None
    }
    return dataclasses.replace(config, **overrides)


def _require_date(value: object, label: str) -> date:
    # datetime is a date subclass, so refuse it explicitly.
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{label} must be an unquoted TOML date (YYYY-MM-DD); got {value!r}")
    return value


def _require_population(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must name a frozen population; got {value!r}")
