"""Generation config: the committed record of how the frozen populations are produced."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SyntheaConfig:
    version: str
    jar_url: str
    jar_sha256: str
    jar_path: Path


@dataclass(frozen=True)
class GenerationParams:
    seed: int
    clinician_seed: int
    reference_date: str
    population_size: int
    state: str


@dataclass(frozen=True)
class PopulationSpec:
    name: str
    modules_dir: Path | None = None
    age_range: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    synthea: SyntheaConfig
    generation: GenerationParams
    exporter: dict[str, str] = field(default_factory=dict)
    populations: dict[str, PopulationSpec] = field(default_factory=dict)


def load_config(path: Path) -> GenerationConfig:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    synthea = SyntheaConfig(
        version=raw["synthea"]["version"],
        jar_url=raw["synthea"]["jar_url"],
        jar_sha256=raw["synthea"]["jar_sha256"],
        jar_path=Path(raw["synthea"]["jar_path"]),
    )
    gen = raw["generation"]
    generation = GenerationParams(
        seed=gen["seed"],
        clinician_seed=gen["clinician_seed"],
        reference_date=gen["reference_date"],
        population_size=gen["population_size"],
        state=gen["state"],
    )
    populations = {
        name: PopulationSpec(
            name=name,
            modules_dir=Path(spec["modules_dir"]) if "modules_dir" in spec else None,
            age_range=spec.get("age_range"),
        )
        for name, spec in raw.get("populations", {}).items()
    }
    return GenerationConfig(
        synthea=synthea,
        generation=generation,
        exporter=dict(raw.get("exporter", {})),
        populations=populations,
    )


def output_dir(repo_root: Path, population: str) -> Path:
    return repo_root / "data" / population


def build_synthea_argv(config: GenerationConfig, population: str, repo_root: Path) -> list[str]:
    """Build the exact Synthea invocation for one population.

    Pure function so the committed config can be asserted against the real
    command line without running Java.
    """
    spec = config.populations[population]
    gen = config.generation

    argv = [
        "java",
        "-jar",
        str(repo_root / config.synthea.jar_path),
        "-s",
        str(gen.seed),
        "-cs",
        str(gen.clinician_seed),
        "-r",
        gen.reference_date,
        "-p",
        str(gen.population_size),
    ]
    if spec.modules_dir is not None:
        argv += ["-d", str(repo_root / spec.modules_dir)]
    if spec.age_range is not None:
        argv += ["-a", spec.age_range]
    argv.append(f"--exporter.baseDirectory={output_dir(repo_root, population)}")
    argv += [f"--{key}={value}" for key, value in config.exporter.items()]
    # Synthea takes the state as a positional argument after all options.
    argv.append(gen.state)
    return argv
