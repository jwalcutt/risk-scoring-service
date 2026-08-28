"""Tests for the migration runner's pure logic; no database required."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from risk_scoring import db


def _write_migration(directory: Path, filename: str, sql: str) -> Path:
    path = directory / filename
    path.write_text(sql)
    return path


def test_discover_migrations_orders_by_number(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0010_zebra.sql", "SELECT 10;")
    _write_migration(tmp_path, "0002_apple.sql", "SELECT 2;")
    _write_migration(tmp_path, "0001_first.sql", "SELECT 1;")

    migrations = db.discover_migrations(tmp_path)

    assert [m.number for m in migrations] == [1, 2, 10]
    assert [m.name for m in migrations] == ["first", "apple", "zebra"]
    assert [m.sql for m in migrations] == ["SELECT 1;", "SELECT 2;", "SELECT 10;"]


def test_discover_migrations_rejects_duplicate_numbers(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_first.sql", "SELECT 1;")
    _write_migration(tmp_path, "0001_second.sql", "SELECT 1;")

    with pytest.raises(db.MigrationError, match="0001"):
        db.discover_migrations(tmp_path)


def test_discover_migrations_rejects_malformed_filenames(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_good.sql", "SELECT 1;")
    _write_migration(tmp_path, "02_short_prefix.sql", "SELECT 2;")

    with pytest.raises(db.MigrationError, match=re.escape("02_short_prefix.sql")):
        db.discover_migrations(tmp_path)


def test_migration_checksum_tracks_file_contents(tmp_path: Path) -> None:
    path = _write_migration(tmp_path, "0001_first.sql", "CREATE TABLE a (id int);")

    (migration,) = db.discover_migrations(tmp_path)
    assert migration.checksum == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_text("CREATE TABLE a (id bigint);")
    (edited,) = db.discover_migrations(tmp_path)
    assert edited.checksum != migration.checksum


def test_pending_migrations_excludes_applied_and_preserves_order(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_first.sql", "SELECT 1;")
    _write_migration(tmp_path, "0002_second.sql", "SELECT 2;")
    _write_migration(tmp_path, "0003_third.sql", "SELECT 3;")
    discovered = db.discover_migrations(tmp_path)
    applied = {1: discovered[0].checksum}

    pending = db.pending_migrations(discovered, applied)

    assert [m.number for m in pending] == [2, 3]


def test_pending_migrations_rejects_applied_file_missing_from_disk(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0002_second.sql", "SELECT 2;")
    discovered = db.discover_migrations(tmp_path)
    applied = {1: "abc123", 2: discovered[0].checksum}

    with pytest.raises(db.MigrationError, match="0001"):
        db.pending_migrations(discovered, applied)


def test_pending_migrations_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = _write_migration(tmp_path, "0001_first.sql", "SELECT 1;")
    original = db.discover_migrations(tmp_path)[0].checksum
    path.write_text("SELECT 999;")
    discovered = db.discover_migrations(tmp_path)

    with pytest.raises(db.MigrationError, match="checksum"):
        db.pending_migrations(discovered, {1: original})


def test_database_url_env_override_and_default() -> None:
    assert db.database_url(env={}) == db.DEFAULT_DATABASE_URL
    override = "postgresql://other:secret@dbhost:5432/elsewhere"
    assert db.database_url(env={db.ENV_DATABASE_URL: override}) == override


def test_packaged_migrations_discoverable() -> None:
    """The default directory ships with the package and holds the bookkeeping migration."""
    migrations = db.discover_migrations()

    assert migrations[0].number == 1
    assert migrations[0].name == "schema_migrations"
    assert "schema_migrations" in migrations[0].sql
