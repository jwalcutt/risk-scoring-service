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
from typing import Any

from compose_support import (
    REPO_ROOT,
    compare,
    compose,
    compose_env,
    read_log,
    throwaway_database,
)
from risk_scoring.populations import load_population
from risk_scoring.sampling import sample_patients
from risk_scoring.service_client import DEFAULT_SERVICE_PORT, ServiceClient
from risk_scoring.stream import build_stream


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


def run_arm(
    events: list[dict[str, Any]], *, restart_at: int | None, port: int
) -> list[dict[str, Any]]:
    """Post the stream to the real stack against its own throwaway database."""
    with throwaway_database("restart_check") as (database, url):
        return _post(events, database=database, url=url, restart_at=restart_at, port=port)


def _post(
    events: list[dict[str, Any]], *, database: str, url: str, restart_at: int | None, port: int
) -> list[dict[str, Any]]:
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

    difference = compare(interrupted, uninterrupted, what="restart")
    if difference is not None:
        print(f"MISMATCH: {difference}")
        sys.exit(1)
    if not uninterrupted:
        print("MISMATCH: both runs scored nothing, so the comparison proves nothing")
        sys.exit(1)
    print(f"MATCH: {len(interrupted)} predictions identical across the restart")


if __name__ == "__main__":
    main()
