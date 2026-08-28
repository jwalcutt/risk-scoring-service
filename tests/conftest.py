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

from factories import write_training_csvs
from risk_scoring import db as db_module
from risk_scoring import train


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
def db_url(_db_admin_url: str) -> Iterator[str]:
    """A freshly created, fully migrated throwaway database, as a DSN.

    Tests that drive the service through HTTP need the DSN rather than a
    connection, because the app opens its own pool.
    """
    database = f"test_risk_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_db_admin_url, connect_timeout=2, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    url = psycopg.conninfo.make_conninfo(_db_admin_url, dbname=database)
    try:
        with psycopg.connect(url, connect_timeout=2) as conn:
            db_module.migrate(conn)
        yield url
    finally:
        with psycopg.connect(_db_admin_url, connect_timeout=2, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


@pytest.fixture()
def db_conn(db_url: str) -> Iterator[psycopg.Connection[Any]]:
    """A connection to a freshly created, fully migrated throwaway database."""
    conn: psycopg.Connection[Any] = psycopg.connect(db_url, connect_timeout=2)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def trained_repo(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, train.TrainingResult]]:
    """One fast population trained and registered once per test module.

    Module scope rather than session scope: the fixture points the global
    MLflow URIs at its own repo root for as long as it lives, so keeping
    that window to one module at a time avoids surprising any other test
    that reads the registry.
    """
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("service-repo")
    write_training_csvs(root / "data" / "baseline" / "csv")
    result = train.train(root / "data" / "baseline" / "csv", root)
    yield root, result
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)
