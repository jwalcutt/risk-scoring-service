"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import psycopg
import pytest
from psycopg import sql

from risk_scoring import db as db_module


@pytest.fixture()
def repo_root(tmp_path: Path) -> Iterator[Path]:
    """A throwaway repo root; restores global MLflow URIs after the test."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    yield tmp_path
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


@pytest.fixture(scope="session")
def _db_admin_url() -> str:
    """The server URL for tests needing Postgres; skips (or fails in CI) if unreachable."""
    url = db_module.database_url()
    try:
        with psycopg.connect(url, connect_timeout=2):
            pass
    except psycopg.OperationalError as error:
        if os.environ.get("RISK_SCORING_REQUIRE_DB"):
            pytest.fail(
                f"RISK_SCORING_REQUIRE_DB is set but PostgreSQL is not reachable at {url}: {error}"
            )
        pytest.skip(
            f"PostgreSQL not reachable at {url}. Start it with "
            "'docker compose up -d postgres'. Set RISK_SCORING_REQUIRE_DB=1 "
            f"to make this a failure instead. ({error})"
        )
    return url


@pytest.fixture()
def db_conn(_db_admin_url: str) -> Iterator[psycopg.Connection[Any]]:
    """A connection to a freshly created, fully migrated throwaway database."""
    database = f"test_risk_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_db_admin_url, connect_timeout=2, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    url = psycopg.conninfo.make_conninfo(_db_admin_url, dbname=database)
    conn: psycopg.Connection[Any] = psycopg.connect(url, connect_timeout=2)
    try:
        db_module.migrate(conn)
        yield conn
    finally:
        conn.close()
        with psycopg.connect(_db_admin_url, connect_timeout=2, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
