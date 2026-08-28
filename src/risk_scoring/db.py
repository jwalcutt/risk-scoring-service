"""Database connection settings and the schema migration runner.

Usage:
    python -m risk_scoring.db migrate
    python -m risk_scoring.db status

Schema changes are plain SQL files in ``src/risk_scoring/migrations/``, named
``NNNN_description.sql`` and applied in numeric order. Each applied migration is
recorded in a ``schema_migrations`` table with the SHA-256 checksum of the file
that ran, so an applied file that later disappears or changes is an error, never
a silent divergence. Alembic was considered and rejected: there are no ORM
models to autogenerate from, the schema is a handful of hand-reviewed tables,
and a runner this repository owns executes identically in tests, Compose, and
CI with zero added dependencies (see docs/service-notes.md).

Judgment calls this module fixes:

- Migrations anchor to the package (``Path(__file__)``), not the repo root:
  they are code shipped with the package, unlike runtime artifacts.
- Duplicate migration numbers are an error; gaps are allowed, because
  renumbering committed files causes more churn than a gap costs.
- Each migration runs in its own transaction together with its bookkeeping
  row; a failure rolls back that migration completely and keeps earlier ones.
- Migration 0001 creates ``schema_migrations`` itself; a database without the
  table simply has nothing applied. No advisory locks: the primary key on the
  migration number turns a concurrent double-apply into an error.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Default matches the Compose service in docker-compose.yml: dev credentials,
# host port 5433 to stay clear of a locally installed Postgres on 5432.
DEFAULT_DATABASE_URL = "postgresql://risk:risk@localhost:5433/risk_scoring"
ENV_DATABASE_URL = "RISK_SCORING_DATABASE_URL"

_FILENAME_PATTERN = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """A migration set is inconsistent with itself or with the database."""


@dataclass(frozen=True)
class Migration:
    """One migration file: its order, name, SQL text, and content checksum."""

    number: int
    name: str
    sql: str
    checksum: str


def database_url(env: Mapping[str, str] | None = None) -> str:
    """The target database DSN: ``RISK_SCORING_DATABASE_URL`` or the Compose default."""
    if env is None:
        env = os.environ
    return env.get(ENV_DATABASE_URL, DEFAULT_DATABASE_URL)


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Read every migration file in ``migrations_dir``, ordered by number.

    Raises MigrationError on a filename that does not match
    ``NNNN_description.sql`` or on two files sharing a number.
    """
    migrations: dict[int, Migration] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        match = _FILENAME_PATTERN.match(path.name)
        if match is None:
            raise MigrationError(
                f"Migration filename {path.name!r} does not match NNNN_description.sql"
            )
        number = int(match.group(1))
        if number in migrations:
            raise MigrationError(
                f"Duplicate migration number {number:04d}: "
                f"{migrations[number].name!r} and {match.group(2)!r}"
            )
        contents = path.read_bytes()
        migrations[number] = Migration(
            number=number,
            name=match.group(2),
            sql=contents.decode("utf-8"),
            checksum=hashlib.sha256(contents).hexdigest(),
        )
    return [migrations[number] for number in sorted(migrations)]


def pending_migrations(
    discovered: Sequence[Migration], applied: Mapping[int, str]
) -> list[Migration]:
    """The discovered migrations not yet applied, in order.

    Raises MigrationError when an applied migration's file is missing from
    disk or its checksum no longer matches the recorded one: an applied
    migration is frozen, and fixes belong in a new migration.
    """
    by_number = {migration.number: migration for migration in discovered}
    for number, recorded_checksum in sorted(applied.items()):
        migration = by_number.get(number)
        if migration is None:
            raise MigrationError(
                f"Applied migration {number:04d} has no file on disk; "
                "applied migrations must never be deleted"
            )
        if migration.checksum != recorded_checksum:
            raise MigrationError(
                f"Applied migration {number:04d}_{migration.name} has a different "
                "checksum on disk than was recorded when it ran; applied migrations "
                "must never be edited"
            )
    return [migration for migration in discovered if migration.number not in applied]
