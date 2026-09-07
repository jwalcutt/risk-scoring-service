"""Download the pinned Synthea release jar and verify it against the committed checksum."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from risk_scoring.datagen.config import GenerationConfig, require_https_url
from risk_scoring.datagen.manifest import sha256_file


class ChecksumMismatchError(RuntimeError):
    """Raised when the downloaded jar does not match the pinned SHA-256."""


class UnpinnedJarError(RuntimeError):
    """Raised when no SHA-256 is pinned, so the jar on disk cannot be verified.

    Carries the digest of the jar that was fetched so the caller can record it.
    """

    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__(
            f"jar_sha256 is empty, so the jar was fetched but not run; record {digest} "
            "as jar_sha256 in configs/generation.toml and run again"
        )


def ensure_jar(config: GenerationConfig, repo_root: Path) -> Path:
    """Download the jar if absent, verify it against the pinned checksum, and return it.

    A jar is only returned once its digest matches the pin. With an empty
    pin the digest is printed and ``UnpinnedJarError`` is raised, so the
    bootstrap is two steps: fetch and record the hash, then generate.
    """
    jar = repo_root / config.synthea.jar_path
    if not jar.exists():
        require_https_url(config.synthea.jar_url)
        jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {config.synthea.jar_url} -> {jar}")
        urllib.request.urlretrieve(config.synthea.jar_url, jar)

    digest = sha256_file(jar)
    if not config.synthea.jar_sha256:
        print(f"jar sha256 (record this in configs/generation.toml): {digest}")
        raise UnpinnedJarError(digest)
    if digest != config.synthea.jar_sha256:
        raise ChecksumMismatchError(
            f"jar checksum {digest} does not match pinned {config.synthea.jar_sha256}"
        )
    return jar
