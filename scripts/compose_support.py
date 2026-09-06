"""What the local check scripts share: the Compose stack and throwaway databases.

Two scripts compare an interrupted run against an uninterrupted one on the
real containers and real generated data: one restarts the service
container, the other pauses and resumes the replay harness. Both need to
bring the stack up against a database that did not exist a moment ago,
read the prediction log back without the columns the database assigns,
and say where two logs first disagree. That lives here once.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from risk_scoring import db as db_module
from risk_scoring import label_log, predictions

REPO_ROOT = Path(__file__).resolve().parent.parent

# The database assigns both, and a bigserial consumes a value even when the
# log's conflict clause drops a re-post, so ids gap after a resume by
# design and say nothing about whether the two runs agree.
VOLATILE_COLUMNS = ("prediction_id", "scored_at")

# The labels table's equivalents: its own bigserial, the wall clock at
# write, and the prediction id the log assigned.
LABEL_VOLATILE_COLUMNS = ("label_id", "prediction_id", "recorded_at")


def compose(arguments: list[str], env: dict[str, str]) -> None:
    subprocess.run(["docker", "compose", *arguments], cwd=REPO_ROOT, env=env, check=True)


def compose_env(*, database: str, port: int) -> dict[str, str]:
    """Compose reads ${PWD} for the registry mounts, so it is set explicitly."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return {
        **os.environ,
        "PWD": str(REPO_ROOT),
        "SERVICE_DB": database,
        "SERVICE_PORT": str(port),
        "GIT_SHA": sha.stdout.strip() if sha.returncode == 0 else "",
    }


@contextmanager
def throwaway_database(prefix: str) -> Iterator[tuple[str, str]]:
    """A fresh database on the local server, dropped on exit; yields (name, DSN).

    The schema is not applied here: the Compose ``migrate`` service does
    that against ``SERVICE_DB`` before the app is allowed to start.
    """
    admin_url = db_module.database_url()
    database = f"{prefix}_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield database, psycopg.conninfo.make_conninfo(admin_url, dbname=database)
    finally:
        with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def read_log(url: str) -> list[dict[str, Any]]:
    """The prediction log, minus the fields the database assigns."""
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = predictions.all_predictions(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in VOLATILE_COLUMNS}
        for row in rows
    ]


def read_labels(url: str) -> list[dict[str, Any]]:
    """The labels table, minus the fields the database assigns."""
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = label_log.all_labels(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in LABEL_VOLATILE_COLUMNS}
        for row in rows
    ]


def compare(
    interrupted: list[dict[str, Any]], uninterrupted: list[dict[str, Any]], *, what: str
) -> str | None:
    """None when the two logs agree, otherwise the first difference found."""
    if len(interrupted) != len(uninterrupted):
        return f"{len(interrupted)} rows after the {what}, {len(uninterrupted)} without it"
    for position, (left, right) in enumerate(zip(interrupted, uninterrupted, strict=True)):
        if left != right:
            differing = sorted(name for name in left if left[name] != right[name])
            return f"row {position} ({left['encounter_id']}) differs on {differing}"
    return None
