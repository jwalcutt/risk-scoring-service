"""Tests for the generation runner's overwrite guard."""

from pathlib import Path
from unittest.mock import patch

import pytest

from risk_scoring.datagen.config import load_config
from risk_scoring.datagen.generate import FrozenOutputError, run_generation

CONFIG_TOML = """
[synthea]
version = "v4.0.0"
jar_url = "https://example.com/synthea-with-dependencies.jar"
jar_sha256 = "abc123"
jar_path = "tools/synthea/synthea-with-dependencies.jar"

[generation]
seed = 20260101
clinician_seed = 20260101
reference_date = "20260101"
population_size = 10000
state = "Massachusetts"

[exporter]
"exporter.csv.export" = "true"

[populations.baseline]
"""


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / "generation.toml").write_text(CONFIG_TOML)
    return tmp_path


def test_refuses_to_overwrite_existing_output(repo_root: Path) -> None:
    existing = repo_root / "data" / "baseline"
    existing.mkdir(parents=True)
    (existing / "csv").mkdir()
    config = load_config(repo_root / "generation.toml")

    with (
        patch("risk_scoring.datagen.generate.subprocess.run") as mock_run,
        pytest.raises(FrozenOutputError),
    ):
        run_generation(config, "baseline", repo_root)
    mock_run.assert_not_called()


def test_force_allows_regeneration_over_existing_output(repo_root: Path) -> None:
    existing = repo_root / "data" / "baseline"
    existing.mkdir(parents=True)
    (existing / "old.csv").write_text("x\n")
    config = load_config(repo_root / "generation.toml")

    with patch("risk_scoring.datagen.generate.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_generation(config, "baseline", repo_root, force=True)

    mock_run.assert_called_once()
    argv = mock_run.call_args.args[0]
    assert argv[0] == "java"
    assert argv[-1] == "Massachusetts"


def test_runs_synthea_into_population_output_dir(repo_root: Path) -> None:
    config = load_config(repo_root / "generation.toml")

    with patch("risk_scoring.datagen.generate.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_generation(config, "baseline", repo_root)

    argv = mock_run.call_args.args[0]
    assert f"--exporter.baseDirectory={repo_root / 'data' / 'baseline'}" in argv
