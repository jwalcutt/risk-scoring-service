"""Tests for the service configuration loader.

The rules these tests pin:

- The pinned model version must be an explicit positive integer in the
  committed TOML; strings (including "latest" and numeric strings),
  booleans, zero, and negatives are all rejected loudly. There is no
  code path that resolves "newest".
- A missing [model] table or missing key fails loudly, never defaults.
- The committed configs/service.toml parses, and its model name matches
  the registry name training uses, so the two cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from risk_scoring.service.config import ServiceConfig, load_config
from risk_scoring.train import MODEL_NAME

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "service.toml"
    path.write_text(body)
    return path


def test_valid_file_loads_into_dataclass(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\nversion = 3\n')
    config = load_config(path)
    assert config == ServiceConfig(model_name="readmission-risk", model_version=3)
    assert isinstance(config.model_version, int)


def test_version_latest_rejected_with_explicit_rule(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\nversion = "latest"\n')
    with pytest.raises(ValueError, match="explicit registered version"):
        load_config(path)


def test_version_numeric_string_rejected_without_coercion(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\nversion = "3"\n')
    with pytest.raises(ValueError, match="explicit registered version"):
        load_config(path)


@pytest.mark.parametrize("version", ["0", "-1"])
def test_non_positive_version_rejected(tmp_path: Path, version: str) -> None:
    path = _write(tmp_path, f'[model]\nname = "readmission-risk"\nversion = {version}\n')
    with pytest.raises(ValueError, match="explicit registered version"):
        load_config(path)


def test_boolean_version_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\nversion = true\n')
    with pytest.raises(ValueError, match="explicit registered version"):
        load_config(path)


def test_missing_model_table_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '[other]\nname = "readmission-risk"\n')
    with pytest.raises(KeyError):
        load_config(path)


def test_missing_version_key_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\n')
    with pytest.raises(KeyError):
        load_config(path)


def test_committed_config_parses_and_matches_registry_name() -> None:
    config = load_config(_REPO_ROOT / "configs" / "service.toml")
    assert config.model_name == MODEL_NAME
    assert config.model_version >= 1
