"""Integration tests for the migration runner against a real PostgreSQL server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest

from risk_scoring import db

pytestmark = pytest.mark.db

BOOTSTRAP_SQL = (db.MIGRATIONS_DIR / "0001_schema_migrations.sql").read_text()

EXPECTED_TABLES = {
    "schema_migrations",
    "patients",
    "encounters",
    "medications",
    "conditions",
    "predictions",
    "replay_runs",
    "labels",
}


def _public_tables(conn: psycopg.Connection[Any]) -> set[str]:
    rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    return {row[0] for row in rows}


def test_migrate_empty_database_creates_expected_tables(
    db_conn: psycopg.Connection[Any],
) -> None:
    assert _public_tables(db_conn) == EXPECTED_TABLES

    recorded = db.applied_migrations(db_conn)
    discovered = db.discover_migrations()
    assert recorded == {m.number: m.checksum for m in discovered}


def test_migrate_rerun_is_noop(db_conn: psycopg.Connection[Any]) -> None:
    before = db_conn.execute(
        "SELECT number, checksum, applied_at FROM schema_migrations ORDER BY number"
    ).fetchall()

    assert db.migrate(db_conn) == []

    after = db_conn.execute(
        "SELECT number, checksum, applied_at FROM schema_migrations ORDER BY number"
    ).fetchall()
    assert after == before


def test_migrate_failure_rolls_back_failed_migration_only(
    db_conn: psycopg.Connection[Any], tmp_path: Path
) -> None:
    """A failing migration leaves no trace of itself; earlier ones stay recorded."""
    (tmp_path / "0001_schema_migrations.sql").write_text(BOOTSTRAP_SQL)
    (tmp_path / "0002_broken.sql").write_text(
        "CREATE TABLE partial (id int); SELECT * FROM missing_table;"
    )
    fresh = psycopg.conninfo.make_conninfo(db.database_url(), dbname=db_conn.info.dbname)
    with psycopg.connect(fresh, connect_timeout=2) as conn:
        conn.execute("DROP TABLE schema_migrations")
        conn.commit()

        with pytest.raises(psycopg.errors.UndefinedTable):
            db.migrate(conn, migrations_dir=tmp_path)

        assert db.applied_migrations(conn) == {1: db.discover_migrations(tmp_path)[0].checksum}
        assert _public_tables(conn) == EXPECTED_TABLES


def test_isolation_first_test_writes_scratch_table(
    db_conn: psycopg.Connection[Any],
) -> None:
    db_conn.execute("CREATE TABLE scratch (id int)")
    db_conn.commit()
    assert "scratch" in _public_tables(db_conn)


def test_isolation_second_test_sees_fresh_database(
    db_conn: psycopg.Connection[Any],
) -> None:
    """Runs after the scratch-table test; a shared database would still show it."""
    assert _public_tables(db_conn) == EXPECTED_TABLES
