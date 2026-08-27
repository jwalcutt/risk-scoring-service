"""Tests for the generation config loader and Synthea argv builder."""

from pathlib import Path

import pytest

from risk_scoring.datagen.config import GenerationConfig, build_synthea_argv, load_config

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
"exporter.fhir.export" = "false"

[populations.baseline]

[populations.care_protocol]
modules_dir = "synthea_modules/care_protocol"

[populations.demographic_shift]
age_range = "55-100"
"""


@pytest.fixture
def config(tmp_path: Path) -> GenerationConfig:
    path = tmp_path / "generation.toml"
    path.write_text(CONFIG_TOML)
    return load_config(path)


def test_load_config_reads_pinned_synthea_release(config: GenerationConfig) -> None:
    assert config.synthea.version == "v4.0.0"
    assert config.synthea.jar_sha256 == "abc123"
    assert config.synthea.jar_path == Path("tools/synthea/synthea-with-dependencies.jar")


def test_load_config_reads_generation_parameters(config: GenerationConfig) -> None:
    assert config.generation.seed == 20260101
    assert config.generation.clinician_seed == 20260101
    assert config.generation.reference_date == "20260101"
    assert config.generation.population_size == 10000
    assert config.generation.state == "Massachusetts"


def test_load_config_reads_all_populations(config: GenerationConfig) -> None:
    assert set(config.populations) == {"baseline", "care_protocol", "demographic_shift"}
    assert config.populations["care_protocol"].modules_dir == Path("synthea_modules/care_protocol")
    assert config.populations["demographic_shift"].age_range == "55-100"
    assert config.populations["baseline"].modules_dir is None
    assert config.populations["baseline"].age_range is None


def test_baseline_argv_contains_seed_date_size_and_state(config: GenerationConfig) -> None:
    repo_root = Path("/repo")
    argv = build_synthea_argv(config, "baseline", repo_root)

    assert argv[:3] == ["java", "-jar", "/repo/tools/synthea/synthea-with-dependencies.jar"]
    for flag, value in [
        ("-s", "20260101"),
        ("-cs", "20260101"),
        ("-r", "20260101"),
        ("-p", "10000"),
    ]:
        idx = argv.index(flag)
        assert argv[idx + 1] == value
    assert "--exporter.csv.export=true" in argv
    assert "--exporter.fhir.export=false" in argv
    assert "--exporter.baseDirectory=/repo/data/baseline" in argv
    # State is Synthea's positional argument and must come last.
    assert argv[-1] == "Massachusetts"


def test_baseline_argv_has_no_variant_flags(config: GenerationConfig) -> None:
    argv = build_synthea_argv(config, "baseline", Path("/repo"))
    assert "-d" not in argv
    assert "-a" not in argv


def test_care_protocol_argv_adds_local_modules_dir(config: GenerationConfig) -> None:
    argv = build_synthea_argv(config, "care_protocol", Path("/repo"))
    idx = argv.index("-d")
    assert argv[idx + 1] == "/repo/synthea_modules/care_protocol"
    assert "--exporter.baseDirectory=/repo/data/care_protocol" in argv


def test_demographic_shift_argv_adds_age_range(config: GenerationConfig) -> None:
    argv = build_synthea_argv(config, "demographic_shift", Path("/repo"))
    idx = argv.index("-a")
    assert argv[idx + 1] == "55-100"


def test_unknown_population_raises_key_error(config: GenerationConfig) -> None:
    with pytest.raises(KeyError):
        build_synthea_argv(config, "nonexistent", Path("/repo"))
