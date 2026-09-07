"""The datagen commands, driven through ``main()`` against a throwaway repo root.

The rule these tests pin: every subcommand that takes a population resolves
it against the configured names, so a typo exits with the same message
everywhere and ``all`` loops over every population.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from risk_scoring.datagen.__main__ import main

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

[populations.baseline]

[populations.care_protocol]
modules_dir = "synthea_modules/care_protocol"
"""

PATIENTS_CSV = """Id,BIRTHDATE,DEATHDATE,GENDER
p1,1950-01-01,,M
p2,2015-06-15,,F
"""

ENCOUNTERS_CSV = """Id,START,STOP,PATIENT,ENCOUNTERCLASS
e1,2025-01-01T08:00:00Z,2025-01-05T10:00:00Z,p1,inpatient
e2,2025-02-10T10:00:00Z,2025-02-10T11:00:00Z,p2,ambulatory
"""


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "generation.toml").write_text(CONFIG_TOML)
    for population in ("baseline", "care_protocol"):
        csv_dir = tmp_path / "data" / population / "csv"
        csv_dir.mkdir(parents=True)
        (csv_dir / "patients.csv").write_text(PATIENTS_CSV)
        (csv_dir / "encounters.csv").write_text(ENCOUNTERS_CSV)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_sanity_rejects_an_unknown_population_before_reading_anything(repo_root: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["sanity", "basline"])
    assert "unknown population 'basline'" in str(exc_info.value)


def test_sanity_all_prints_a_block_per_configured_population(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["sanity", "all"])

    lines = capsys.readouterr().out.splitlines()
    assert lines.index("baseline:") < lines.index("care_protocol:")
    assert lines.count("  total_encounters: 2") == 2


def test_sanity_defaults_to_the_baseline_population(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["sanity"])

    out = capsys.readouterr().out
    assert out.startswith("baseline:\n")
    assert "care_protocol" not in out
