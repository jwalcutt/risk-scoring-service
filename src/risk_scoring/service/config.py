"""Service config: the committed record of which registered model serves.

Judgment calls this module fixes:

- The pinned version must be an explicit positive integer in the TOML.
  Strings (including "latest" and numeric strings), booleans, zero, and
  negatives are rejected loudly, and no code path resolves "newest", so
  serving an unpinned model is structurally impossible rather than a
  convention a review has to catch.
- A missing model table or key raises instead of defaulting: a service
  with no pin must refuse to start, not guess.
- The connection pool size is the one key with a default. It caps how many
  Postgres connections the service opens, sized to the server's
  ``max_connections`` budget rather than to what serves; leaving it out
  means the default of 10, and a value that is not a positive integer is
  rejected the same way a bad pin is.
- The per-patient event cap has a default for the same reason. It guards
  against one patient id absorbing unbounded events, and a service
  without a stated cap is safer with the default than with none, so
  ``[limits]`` is optional. A cap that is present must be a positive
  integer; a bool, a string, or a float is rejected.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_RELPATH = Path("configs/service.toml")
DEFAULT_POOL_SIZE = 10

DEFAULT_MAX_EVENTS_PER_PATIENT = 20_000
"""Encounters, medications, and conditions one patient id may accumulate.

Sized against the frozen populations: they average 145 to 338 such rows
per patient, and the largest single patient posted in a recorded run
carried 826, so the default sits roughly twenty-five times above the
heaviest generated patient while still bounding what a hostile client can
attach to one id.
"""


@dataclass(frozen=True)
class ServiceConfig:
    model_name: str
    model_version: int
    pool_size: int = DEFAULT_POOL_SIZE
    """The most Postgres connections the service holds open at once."""

    max_events_per_patient: int = DEFAULT_MAX_EVENTS_PER_PATIENT
    """Clinical rows one patient id may accumulate before events are refused."""


def _positive_int(value: object, key: str) -> int:
    # bool is an int subclass, so check it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer; got {value!r}")
    return value


def load_config(path: Path) -> ServiceConfig:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    model = raw["model"]
    version = model["version"]
    try:
        _positive_int(version, "model.version")
    except ValueError as exc:
        raise ValueError(
            f"model.version must be an explicit registered version number (a positive "
            f'integer); got {version!r}. "latest" and aliases are not accepted.'
        ) from exc
    pool_size = _positive_int(
        raw.get("database", {}).get("pool_size", DEFAULT_POOL_SIZE), "database.pool_size"
    )
    limits = raw.get("limits", {})
    cap = DEFAULT_MAX_EVENTS_PER_PATIENT
    if "max_events_per_patient" in limits:
        cap = _positive_int(limits["max_events_per_patient"], "limits.max_events_per_patient")
    return ServiceConfig(
        model_name=model["name"],
        model_version=version,
        pool_size=pool_size,
        max_events_per_patient=cap,
    )
