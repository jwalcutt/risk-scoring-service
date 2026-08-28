"""Service config: the committed record of which registered model serves.

Judgment calls this module fixes:

- The pinned version must be an explicit positive integer in the TOML.
  Strings (including "latest" and numeric strings), booleans, zero, and
  negatives are rejected loudly, and no code path resolves "newest", so
  serving an unpinned model is structurally impossible rather than a
  convention a review has to catch.
- A missing table or key raises instead of defaulting: a service with no
  pin must refuse to start, not guess.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_RELPATH = Path("configs/service.toml")


@dataclass(frozen=True)
class ServiceConfig:
    model_name: str
    model_version: int


def load_config(path: Path) -> ServiceConfig:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    model = raw["model"]
    version = model["version"]
    # bool is an int subclass, so check it explicitly.
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(
            f"model.version must be an explicit registered version number (a positive "
            f'integer); got {version!r}. "latest" and aliases are not accepted.'
        )
    return ServiceConfig(model_name=model["name"], model_version=version)
