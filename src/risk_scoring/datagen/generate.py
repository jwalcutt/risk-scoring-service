"""Run Synthea for one population, guarding already-generated (frozen) output."""

from __future__ import annotations

import subprocess
from pathlib import Path

from risk_scoring.datagen.config import GenerationConfig, build_synthea_argv, output_dir


class FrozenOutputError(RuntimeError):
    """Raised when generation would overwrite an existing population."""


def run_generation(
    config: GenerationConfig,
    population: str,
    repo_root: Path,
    force: bool = False,
) -> None:
    dest = output_dir(repo_root, population)
    if dest.exists() and any(dest.iterdir()) and not force:
        raise FrozenOutputError(
            f"{dest} already contains generated data; pass force=True to regenerate"
        )

    argv = build_synthea_argv(config, population, repo_root)
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Synthea exited with status {result.returncode} for '{population}'")
