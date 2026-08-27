"""Download the pinned Synthea release jar and verify it against the committed checksum."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from risk_scoring.datagen.config import GenerationConfig
from risk_scoring.datagen.manifest import sha256_file


class ChecksumMismatchError(RuntimeError):
    """Raised when the downloaded jar does not match the pinned SHA-256."""


def ensure_jar(config: GenerationConfig, repo_root: Path) -> Path:
    """Download the jar if absent, then verify it against the pinned checksum.

    An empty pinned checksum means the jar has not been fetched before; the
    computed hash is printed so it can be recorded in the generation config.
    """
    jar = repo_root / config.synthea.jar_path
    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {config.synthea.jar_url} -> {jar}")
        urllib.request.urlretrieve(config.synthea.jar_url, jar)

    digest = sha256_file(jar)
    if not config.synthea.jar_sha256:
        print(f"jar sha256 (record this in configs/generation.toml): {digest}")
    elif digest != config.synthea.jar_sha256:
        raise ChecksumMismatchError(
            f"jar checksum {digest} does not match pinned {config.synthea.jar_sha256}"
        )
    return jar
