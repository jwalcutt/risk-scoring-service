"""Tests for the generation runner's overwrite guard and the jar integrity checks."""

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from risk_scoring.datagen.__main__ import main
from risk_scoring.datagen.config import InsecureJarUrlError, load_config
from risk_scoring.datagen.download import UnpinnedJarError, ensure_jar
from risk_scoring.datagen.generate import FrozenOutputError, run_generation

JAR_BYTES = b"synthea jar bytes\n"
# sha256 of JAR_BYTES, computed once outside the code under test.
JAR_SHA256 = "f38be504b5a040d04f7f5c4d135444f1e17761fbc60ed53e766776f6eaef64e8"

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


@pytest.fixture
def fake_download() -> Iterator[MagicMock]:
    """Stand in for the network fetch: write JAR_BYTES where the jar was asked for."""

    def fetch(url: str, dest: Path) -> None:
        Path(dest).write_bytes(JAR_BYTES)

    with patch("risk_scoring.datagen.download.urllib.request.urlretrieve") as mock_fetch:
        mock_fetch.side_effect = fetch
        yield mock_fetch


def _write_config(repo_root: Path, jar_sha256: str) -> Path:
    path = repo_root / "generation.toml"
    path.write_text(CONFIG_TOML.replace('jar_sha256 = "abc123"', f'jar_sha256 = "{jar_sha256}"'))
    return path


def test_ensure_jar_refuses_downloaded_jar_when_checksum_unpinned(
    tmp_path: Path, fake_download: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(_write_config(tmp_path, jar_sha256=""))

    with pytest.raises(UnpinnedJarError) as excinfo:
        ensure_jar(config, tmp_path)

    fake_download.assert_called_once()
    assert JAR_SHA256 in str(excinfo.value)
    assert JAR_SHA256 in capsys.readouterr().out


def test_ensure_jar_refuses_local_jar_when_checksum_unpinned(
    tmp_path: Path, fake_download: MagicMock
) -> None:
    config = load_config(_write_config(tmp_path, jar_sha256=""))
    jar = tmp_path / config.synthea.jar_path
    jar.parent.mkdir(parents=True)
    jar.write_bytes(JAR_BYTES)

    with pytest.raises(UnpinnedJarError):
        ensure_jar(config, tmp_path)

    fake_download.assert_not_called()


def test_ensure_jar_returns_jar_that_matches_pinned_checksum(
    tmp_path: Path, fake_download: MagicMock
) -> None:
    config = load_config(_write_config(tmp_path, jar_sha256=JAR_SHA256))

    jar = ensure_jar(config, tmp_path)

    assert jar == tmp_path / config.synthea.jar_path
    assert jar.read_bytes() == JAR_BYTES


def test_ensure_jar_refuses_to_fetch_over_http(tmp_path: Path, fake_download: MagicMock) -> None:
    config = load_config(_write_config(tmp_path, jar_sha256=JAR_SHA256))
    insecure = dataclasses.replace(
        config,
        synthea=dataclasses.replace(
            config.synthea, jar_url="http://example.com/synthea-with-dependencies.jar"
        ),
    )

    with pytest.raises(InsecureJarUrlError):
        ensure_jar(insecure, tmp_path)

    fake_download.assert_not_called()


def test_generate_command_exits_with_digest_instead_of_running_unpinned_jar(
    tmp_path: Path, fake_download: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs").mkdir()
    _write_config(tmp_path, jar_sha256="").rename(tmp_path / "configs" / "generation.toml")
    monkeypatch.chdir(tmp_path)

    with (
        patch("risk_scoring.datagen.generate.subprocess.run") as mock_run,
        pytest.raises(SystemExit) as excinfo,
    ):
        main(["generate", "baseline"])

    mock_run.assert_not_called()
    # A string exit code is how the CLI reports a clean message with status 1.
    assert isinstance(excinfo.value.code, str)
    assert JAR_SHA256 in excinfo.value.code
