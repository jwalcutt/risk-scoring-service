"""Checksum manifests that make the frozen populations verifiable without committing them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    """Data rows in a CSV file, excluding the header line."""
    with path.open("rb") as fh:
        lines = sum(1 for _ in fh)
    return max(lines - 1, 0)


def _csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.csv"))


def build_manifest(data_dir: Path, config_echo: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in _csv_files(data_dir):
        rel = path.relative_to(data_dir).as_posix()
        files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": _count_rows(path),
        }
    return {"config": config_echo, "files": files}


def verify_manifest(data_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return a list of discrepancies; an empty list means the data matches the manifest."""
    problems: list[str] = []
    recorded: dict[str, dict[str, Any]] = manifest["files"]
    present = {p.relative_to(data_dir).as_posix(): p for p in _csv_files(data_dir)}

    for rel, entry in recorded.items():
        path = present.get(rel)
        if path is None:
            problems.append(f"missing file: {rel}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"checksum mismatch: {rel}")

    problems.extend(f"extra file: {rel}" for rel in sorted(present.keys() - recorded.keys()))
    return problems
