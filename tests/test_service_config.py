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


# --- the connection pool size ---


def test_pool_size_defaults_to_ten_when_the_database_table_is_absent(tmp_path: Path) -> None:
    path = _write(tmp_path, '[model]\nname = "readmission-risk"\nversion = 3\n')
    assert load_config(path).pool_size == 10


def test_pool_size_loads_from_the_database_table(tmp_path: Path) -> None:
    path = _write(
        tmp_path, '[model]\nname = "readmission-risk"\nversion = 3\n\n[database]\npool_size = 4\n'
    )
    assert load_config(path).pool_size == 4


@pytest.mark.parametrize("pool_size", ["0", "-2", '"10"', "true"])
def test_pool_size_must_be_a_positive_integer(tmp_path: Path, pool_size: str) -> None:
    path = _write(
        tmp_path,
        f'[model]\nname = "readmission-risk"\nversion = 3\n\n[database]\npool_size = {pool_size}\n',
    )
    with pytest.raises(ValueError, match="pool_size"):
        load_config(path)


def test_committed_config_pins_the_pool_size() -> None:
    config = load_config(_REPO_ROOT / "configs" / "service.toml")
    assert config.pool_size == 10
