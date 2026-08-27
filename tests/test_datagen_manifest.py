"""Tests for the frozen-population checksum manifest."""

import json
from pathlib import Path

import pytest

from risk_scoring.datagen.manifest import build_manifest, verify_manifest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "patients.csv").write_text("Id,BIRTHDATE\np1,1950-01-01\np2,1980-06-15\n")
    (csv_dir / "encounters.csv").write_text("Id,START,STOP,PATIENT\ne1,a,b,p1\n")
    return tmp_path


def test_build_manifest_records_every_csv_with_hash_size_and_rows(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={"seed": 20260101})

    files = manifest["files"]
    assert set(files) == {"csv/patients.csv", "csv/encounters.csv"}
    patients = files["csv/patients.csv"]
    assert len(patients["sha256"]) == 64
    assert patients["bytes"] == (data_dir / "csv" / "patients.csv").stat().st_size
    assert patients["rows"] == 2  # header excluded
    assert manifest["config"] == {"seed": 20260101}


def test_verify_manifest_passes_on_untouched_data(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={})
    assert verify_manifest(data_dir, manifest) == []


def test_verify_manifest_detects_modified_file(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={})
    (data_dir / "csv" / "patients.csv").write_text("Id,BIRTHDATE\np1,1950-01-02\n")

    problems = verify_manifest(data_dir, manifest)
    assert any("csv/patients.csv" in p and "mismatch" in p for p in problems)


def test_verify_manifest_detects_missing_file(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={})
    (data_dir / "csv" / "encounters.csv").unlink()

    problems = verify_manifest(data_dir, manifest)
    assert any("csv/encounters.csv" in p and "missing" in p for p in problems)


def test_verify_manifest_detects_extra_file(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={})
    (data_dir / "csv" / "conditions.csv").write_text("Id\nx\n")

    problems = verify_manifest(data_dir, manifest)
    assert any("csv/conditions.csv" in p and "extra" in p for p in problems)


def test_manifest_is_json_serializable(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, config_echo={"seed": 1})
    round_tripped = json.loads(json.dumps(manifest))
    assert verify_manifest(data_dir, round_tripped) == []
