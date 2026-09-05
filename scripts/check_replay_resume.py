"""Confirm a replay paused by Ctrl-C and resumed logs what a straight run logs.

The byte-identity test in CI pauses and kills the tick loop in process,
against a small synthetic population. This script runs the same comparison
against the real containers and real generated data, which is local-only
(verified by checksum manifest) and therefore cannot run in CI. It drives
the real commands as subprocesses and interrupts one of them the way an
operator would:

    python scripts/check_replay_resume.py --population baseline --patients 25

A seeded sample of patients with a discharge inside the replay span is
written as its own export, which the commands read through --data-root.
Two arms run it at max speed, each against its own throwaway database.
The first runs `start` straight through. The second runs `start`, sends
it SIGINT once the log holds a prediction, waits for it to exit, checks
the run row says paused with the clock strictly inside the span, and runs
`resume` to the end. The prediction logs must then be equal.

prediction_id and scored_at are excluded from the comparison. The
database assigns both, and a bigserial consumes a value even when the
log's conflict clause drops a re-post, so ids gap after a resume by
design and say nothing about whether the two runs agree.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from compose_support import (
    REPO_ROOT,
    compare,
    compose,
    compose_env,
    read_log,
    throwaway_database,
)
from risk_scoring.populations import load_population
from risk_scoring.replay import clock, runs
from risk_scoring.replay.config import DEFAULT_CONFIG_RELPATH, load_config
from risk_scoring.sampling import sample_patients
from risk_scoring.service_client import DEFAULT_SERVICE_PORT

POLL_SECONDS = 0.05
EXIT_TIMEOUT_SECONDS = 600


def write_export(frames: Mapping[str, pd.DataFrame], csv_dir: Path) -> None:
    """The sample as a CSV export with the population's four frames."""
    csv_dir.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(csv_dir / f"{name}.csv", index=False)


def replay(
    command: str,
    *,
    url: str,
    port: int,
    data_root: Path,
    population: str,
    start: date,
    end: date,
) -> subprocess.Popen[bytes]:
    """The real command as a subprocess against ``url``, at max speed."""
    argv = [
        sys.executable,
        "-m",
        "risk_scoring.replay",
        command,
        "--max-speed",
        "--port",
        str(port),
        "--data-root",
        str(data_root),
    ]
    if command == "start":
        argv += [
            "--population",
            population,
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
        ]
    env = {**os.environ, "RISK_SCORING_DATABASE_URL": url}
    return subprocess.Popen(argv, cwd=REPO_ROOT, env=env)


def wait_for(process: subprocess.Popen[bytes], what: str) -> None:
    code = process.wait(timeout=EXIT_TIMEOUT_SECONDS)
    if code != 0:
        raise RuntimeError(f"{what} exited with status {code}")


def count_predictions(url: str) -> int:
    with psycopg.connect(url, connect_timeout=5) as conn:
        row = conn.execute("SELECT count(*) FROM predictions").fetchone()
    return int(row[0]) if row is not None else 0


def open_run(url: str) -> runs.ReplayRun:
    with psycopg.connect(url, connect_timeout=5) as conn:
        run = runs.open_run(conn)
    if run is None:
        raise RuntimeError("no unfinished run in the interrupted arm's database")
    return run


def straight_arm(port: int, **settings: Any) -> list[dict[str, Any]]:
    with throwaway_database("replay_check") as (database, url):
        env = compose_env(database=database, port=port)
        try:
            compose(["up", "-d", "--build"], env)
            wait_for(replay("start", url=url, port=port, **settings), "start")
            return read_log(url)
        finally:
            compose(["stop", "app"], env)


def interrupted_arm(port: int, **settings: Any) -> tuple[list[dict[str, Any]], runs.ReplayRun]:
    """Start, interrupt once something is logged, check the row, resume."""
    with throwaway_database("replay_check") as (database, url):
        env = compose_env(database=database, port=port)
        try:
            compose(["up", "-d", "--build"], env)
            started = replay("start", url=url, port=port, **settings)
            while started.poll() is None and count_predictions(url) == 0:
                time.sleep(POLL_SECONDS)
            if started.poll() is not None:
                raise RuntimeError(
                    "the run finished before it could be interrupted;"
                    " sample more patients or widen the span"
                )
            started.send_signal(signal.SIGINT)
            wait_for(started, "the interrupted start")

            paused = open_run(url)
            if paused.status != "paused":
                raise RuntimeError(f"after Ctrl-C the run row says {paused.status!r}, not paused")
            if not paused.start_at < paused.sim_now < paused.end_at:
                raise RuntimeError(f"paused with the clock at {paused.sim_now}, not inside the run")
            print(
                f"paused at {clock.instant(paused.sim_now)} with"
                f" {count_predictions(url)} predictions logged"
            )

            wait_for(replay("resume", url=url, port=port, **settings), "resume")
            return read_log(url), paused
        finally:
            compose(["stop", "app"], env)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_replay_resume.py",
        description="Compare a paused-and-resumed replay with a straight one on the Compose stack.",
    )
    parser.add_argument("--population", default="baseline")
    parser.add_argument("--patients", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
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
    config = load_config(REPO_ROOT / DEFAULT_CONFIG_RELPATH)
    start = config.start if args.start is None else args.start
    end = config.end if args.end is None else args.end

    frames = sample_patients(
        load_population(csv_dir),
        count=args.patients,
        seed=args.seed,
        discharged_at_or_after=pd.Timestamp(start, tz="UTC"),
    )
    with tempfile.TemporaryDirectory(prefix="replay-check-") as tmp:
        data_root = Path(tmp)
        write_export(frames, data_root / args.population / "csv")
        settings = {
            "data_root": data_root,
            "population": args.population,
            "start": start,
            "end": end,
        }
        print(
            f"population {args.population}, {len(frames['patients'])} patients (seed {args.seed}),"
            f" replaying {start} to {end}"
        )

        straight = straight_arm(args.port, **settings)
        print(f"straight run:    {len(straight)} discharges scored")
        resumed, paused = interrupted_arm(args.port, **settings)
        print(f"interrupted run: {len(resumed)} discharges scored")

    difference = compare(resumed, straight, what="pause")
    if difference is not None:
        print(f"MISMATCH: {difference}")
        sys.exit(1)
    if not straight:
        print("MISMATCH: both runs scored nothing, so the comparison proves nothing")
        sys.exit(1)
    print(
        f"MATCH: {len(resumed)} predictions identical across a pause at"
        f" {clock.instant(paused.sim_now)}"
    )


if __name__ == "__main__":
    main()
