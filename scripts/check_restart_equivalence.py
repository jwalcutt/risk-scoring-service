"""Confirm a restarted service container resumes with no log gaps.

The equivalence test in CI restarts the app in process, against a small
synthetic population. This script runs the same comparison against the
real containers and real generated data, which is local-only (verified by
checksum manifest) and therefore cannot run in CI:

    python scripts/check_restart_equivalence.py --population baseline --patients 25

Two arms post the identical event stream to the Compose stack, each
against its own throwaway database. The first arm runs straight through.
The second stops and starts the service container partway, waits for it
to come back healthy, and posts the remainder. The prediction logs must
then be equal.

prediction_id and scored_at are excluded from the comparison. The
database assigns both, and a bigserial consumes a value even when the
log's conflict clause drops a re-post, so ids gap after a resume by
design and say nothing about whether the two runs agree.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from risk_scoring import db as db_module
from risk_scoring import predictions
from risk_scoring.populations import load_population
from risk_scoring.sampling import sample_patients
from risk_scoring.service_client import DEFAULT_SERVICE_PORT, ServiceClient
from risk_scoring.stream import build_stream

REPO_ROOT = Path(__file__).resolve().parent.parent

VOLATILE_COLUMNS = ("prediction_id", "scored_at")


def compose(arguments: list[str], env: dict[str, str]) -> None:
    subprocess.run(["docker", "compose", *arguments], cwd=REPO_ROOT, env=env, check=True)


def app_started_at(env: dict[str, str]) -> str:
    """When the service container's current process started, per Docker."""
    container = subprocess.run(
        ["docker", "compose", "ps", "-q", "app"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not container:
        raise RuntimeError("the service container is not running")
    return subprocess.run(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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


def read_log(url: str) -> list[dict[str, Any]]:
    """The prediction log, minus the fields the database assigns."""
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = predictions.all_predictions(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in VOLATILE_COLUMNS}
        for row in rows
    ]


def run_arm(
    events: list[dict[str, Any]], *, restart_at: int | None, port: int
) -> list[dict[str, Any]]:
    """Post the stream to the real stack against its own throwaway database."""
    admin_url = db_module.database_url()
    database = f"restart_check_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    url = psycopg.conninfo.make_conninfo(admin_url, dbname=database)
    env = compose_env(database=database, port=port)
    try:
        # The migrate service applies the schema to the new database before
        # the app is allowed to start.
        compose(["up", "-d", "--build"], env)
        with ServiceClient(port=port) as client:
            client.wait_for_health()
            if restart_at is None:
                client.post_events(events)
            else:
                client.post_events(events[:restart_at])
                before = app_started_at(env)
                compose(["restart", "app"], env)
                client.wait_for_health()
                # A restart that silently did nothing would let a service
                # holding state in memory pass this check, so require the
                # process cycled.
                if app_started_at(env) == before:
                    raise RuntimeError(
                        f"the service container never restarted (still up since {before})"
                    )
                client.post_events(events[restart_at:])
        return read_log(url)
    finally:
        compose(["stop", "app"], env)
        with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def compare(interrupted: list[dict[str, Any]], uninterrupted: list[dict[str, Any]]) -> str | None:
    """None when the two logs agree, otherwise the first difference found."""
    if len(interrupted) != len(uninterrupted):
        return f"{len(interrupted)} rows after the restart, {len(uninterrupted)} without it"
    for position, (left, right) in enumerate(zip(interrupted, uninterrupted, strict=True)):
        if left != right:
            differing = sorted(name for name in left if left[name] != right[name])
            return f"row {position} ({left['encounter_id']}) differs on {differing}"
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_restart_equivalence.py",
        description="Compare an interrupted and an uninterrupted run against the Compose stack.",
    )
    parser.add_argument("--population", default="baseline")
    parser.add_argument("--patients", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SERVICE_PORT", DEFAULT_SERVICE_PORT)),
        help="host port the stack publishes the service on",
    )
    args = parser.parse_args(argv)

    csv_dir = REPO_ROOT / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")

    frames = sample_patients(load_population(csv_dir), count=args.patients, seed=args.seed)
    events = build_stream(frames)
    restart_at = len(events) // 2
    print(f"population {args.population}, {len(frames['patients'])} patients (seed {args.seed})")
    print(f"{len(events)} events, restarting the service container after {restart_at}")

    uninterrupted = run_arm(events, restart_at=None, port=args.port)
    print(f"uninterrupted run: {len(uninterrupted)} discharges scored")
    interrupted = run_arm(events, restart_at=restart_at, port=args.port)
    print(f"interrupted run:   {len(interrupted)} discharges scored")

    difference = compare(interrupted, uninterrupted)
    if difference is not None:
        print(f"MISMATCH: {difference}")
        sys.exit(1)
    if not uninterrupted:
        print("MISMATCH: both runs scored nothing, so the comparison proves nothing")
        sys.exit(1)
    print(f"MATCH: {len(interrupted)} predictions identical across the restart")


if __name__ == "__main__":
    main()
