"""Command-line entrypoint for the data generation tooling.

Usage (from the repo root):
    python -m risk_scoring.datagen download
    python -m risk_scoring.datagen generate [population|all] [--force]
    python -m risk_scoring.datagen manifest [population|all]
    python -m risk_scoring.datagen verify [population|all]
    python -m risk_scoring.datagen sanity [population]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from risk_scoring.datagen.config import GenerationConfig, load_config, output_dir
from risk_scoring.datagen.download import ensure_jar
from risk_scoring.datagen.generate import run_generation
from risk_scoring.datagen.manifest import build_manifest, verify_manifest
from risk_scoring.datagen.sanity import compute_sanity_stats

CONFIG_PATH = Path("configs/generation.toml")
MANIFEST_DIR = Path("data_manifests")


def _config_echo(config: GenerationConfig, population: str) -> dict[str, Any]:
    spec = config.populations[population]
    return {
        "synthea_version": config.synthea.version,
        "jar_sha256": config.synthea.jar_sha256,
        "seed": config.generation.seed,
        "clinician_seed": config.generation.clinician_seed,
        "reference_date": config.generation.reference_date,
        "population_size": config.generation.population_size,
        "state": config.generation.state,
        "population": population,
        "modules_dir": str(spec.modules_dir) if spec.modules_dir else None,
        "age_range": spec.age_range,
        "exporter": config.exporter,
    }


def _populations(config: GenerationConfig, selector: str) -> list[str]:
    if selector == "all":
        return list(config.populations)
    if selector not in config.populations:
        sys.exit(f"unknown population '{selector}'; expected one of {list(config.populations)}")
    return [selector]


def _csv_dir(repo_root: Path, population: str) -> Path:
    return output_dir(repo_root, population) / "csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m risk_scoring.datagen")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch and verify the pinned Synthea jar")

    for name, help_text in [
        ("generate", "run Synthea for a population"),
        ("manifest", "write the checksum manifest for a population"),
        ("verify", "check generated data against its committed manifest"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("population", nargs="?", default="all")
        if name == "generate":
            p.add_argument("--force", action="store_true")

    p = sub.add_parser("sanity", help="print rough sanity stats for a population")
    p.add_argument("population", nargs="?", default="baseline")

    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    config = load_config(repo_root / CONFIG_PATH)

    if args.command == "download":
        ensure_jar(config, repo_root)
    elif args.command == "generate":
        ensure_jar(config, repo_root)
        for population in _populations(config, args.population):
            print(f"generating '{population}'...")
            run_generation(config, population, repo_root, force=args.force)
    elif args.command == "manifest":
        for population in _populations(config, args.population):
            echo = _config_echo(config, population)
            manifest = build_manifest(_csv_dir(repo_root, population), echo)
            dest = repo_root / MANIFEST_DIR / f"{population}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"wrote {dest} ({len(manifest['files'])} files)")
    elif args.command == "verify":
        failed = False
        for population in _populations(config, args.population):
            manifest = json.loads((repo_root / MANIFEST_DIR / f"{population}.json").read_text())
            problems = verify_manifest(_csv_dir(repo_root, population), manifest)
            if problems:
                failed = True
                print(f"{population}: FAILED")
                for problem in problems:
                    print(f"  {problem}")
            else:
                print(f"{population}: ok ({len(manifest['files'])} files match)")
        if failed:
            sys.exit(1)
    elif args.command == "sanity":
        as_of = datetime.strptime(config.generation.reference_date, "%Y%m%d").date()
        stats = compute_sanity_stats(_csv_dir(repo_root, args.population), as_of=as_of)
        for key, value in dataclasses.asdict(stats).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
