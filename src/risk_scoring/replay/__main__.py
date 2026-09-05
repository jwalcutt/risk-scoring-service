"""Operate a replay: start it, pause it, resume it, or ask where it stands.

Usage (from the repo root):
    python -m risk_scoring.replay start [--config PATH] [--population NAME]
        [--start DATE] [--end DATE] [--acceleration N] [--max-speed]
        [--port PORT] [--data-root DIR]
    python -m risk_scoring.replay resume [--max-speed] [--port PORT] [--data-root DIR]
    python -m risk_scoring.replay pause
    python -m risk_scoring.replay status

The database is ``RISK_SCORING_DATABASE_URL`` or the Compose default, as
everywhere else. The service is the Compose stack on ``--port``.

Judgment calls this module fixes:

- ``start`` preloads before it opens the run row, so a Ctrl-C during the
  preload leaves no row behind and a second ``start`` finishes the
  idempotent load. It refuses before loading anything if a run is open.
- ``start`` runs the clock once the row exists; one command takes a
  database from nothing to a ticking replay. Every later session is
  ``resume`` with no other argument.
- The first Ctrl-C asks the loop to pause after the tick it is in, so the
  checkpoint is written and nothing is re-posted on resume. The handler
  then steps aside, so a second Ctrl-C quits at once through Python's
  own ``KeyboardInterrupt``; the last checkpoint stands and ``resume``
  re-posts the partial tick, which the service answers as no-ops.
- ``resume`` accepts a row that still says ``running``. A process that
  died without pausing leaves that behind, and the checkpoint is as good
  as a paused one. Nothing detects a second harness on the same row; the
  operator runs one.
- The pause contract is one field: the loop stops when the row's status
  reads ``paused``, whoever wrote it. ``pause`` here writes it from
  another terminal; a monitoring alert will write the same field later.
- The run summary describes one invocation. The row's wall timestamps and
  the prediction log hold the whole run.
- Side effects (the service, the desktop, the wall clock, the signal
  handler) are values ``main`` takes, so the commands are tested end to
  end against a database with none of them real.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

import pandas as pd
import psycopg

from risk_scoring.db import database_url
from risk_scoring.populations import load_population
from risk_scoring.replay import clock, harness, runs
from risk_scoring.replay.config import (
    ReplayConfig,
    add_config_arguments,
    apply_overrides,
    load_config,
)
from risk_scoring.replay.notify import Notifier, desktop_notifier
from risk_scoring.replay.preload import PreloadSummary, preload_history
from risk_scoring.replay.runs import ReplayRun
from risk_scoring.service_client import DEFAULT_SERVICE_PORT, ServiceClient
from risk_scoring.stream import StreamEvent, ordered_events

DEFAULT_DATA_ROOT = Path("data")

PosterFactory = Callable[[int], AbstractContextManager[harness.Poster]]
"""A running service to post to, given the port; closed when the run stops."""

PAUSING = "pausing after this tick; press Ctrl-C again to quit now"


@contextmanager
def service_poster(port: int) -> Iterator[harness.Poster]:
    """The Compose service on ``port``, waited for before the first post."""
    with ServiceClient(port=port) as client:
        client.wait_for_health()
        yield client


class InterruptGuard:
    """Turns the first Ctrl-C into a pause request and leaves the second to Python."""

    def __init__(self, announce: Callable[[str], None] = print) -> None:
        self.requested = False
        self._announce = announce
        self._previous: Any = None

    def __enter__(self) -> InterruptGuard:
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        signal.signal(signal.SIGINT, self._previous)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True
        signal.signal(signal.SIGINT, self._previous)
        self._announce(PAUSING)

    def pause_requested(self, sim_now: datetime) -> bool:
        return self.requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.replay",
        description="Stream a frozen population to the scoring service on a simulated clock.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="preload history, open a run, and start its clock")
    add_config_arguments(start)
    _add_run_arguments(start)

    resume = sub.add_parser("resume", help="continue the open run from its checkpoint")
    _add_run_arguments(resume)

    sub.add_parser("pause", help="ask the running harness to pause at its next tick")
    sub.add_parser("status", help="print the open run's row")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-speed", action="store_true", help="never wait between ticks (tests and checks)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_SERVICE_PORT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="directory holding <population>/csv exports",
    )


def pacing_for(acceleration: float, args: argparse.Namespace) -> clock.Pacing:
    """Pacing is the flag and the acceleration; it never comes from the row or the file."""
    return clock.Pacing(acceleration=acceleration, max_speed=args.max_speed)


def main(
    argv: list[str] | None = None,
    *,
    poster_factory: PosterFactory = service_poster,
    notifier: Notifier | None = None,
    wall_clock: clock.WallClock = clock.DEFAULT_WALL_CLOCK,
    sleep: harness.Sleep = time.sleep,
) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()
    notify = desktop_notifier() if notifier is None else notifier

    if args.command == "start":
        config = apply_overrides(load_config(repo_root / args.config), args)
        csv_dir = _csv_dir(repo_root, args.data_root, config.population)
        with _connect() as conn:
            if runs.open_run(conn) is not None:
                sys.exit(
                    "a replay run that is not finished already exists in this database;"
                    " resume it or finish it before starting another"
                )
            frames, events = _load(csv_dir)
            run = _start(conn, config, frames, events)
            _drive(conn, run, events, args, poster_factory, notify, wall_clock, sleep)
    elif args.command == "resume":
        with _connect() as conn:
            run = _open_run(conn, "start one")
            csv_dir = _csv_dir(repo_root, args.data_root, run.population)
            _, events = _load(csv_dir)
            runs.set_status(conn, run.run_id, "running")
            run = runs.read_run(conn, run.run_id)
            print(f"resuming run {run.run_id} from {clock.instant(run.sim_now)}")
            _drive(conn, run, events, args, poster_factory, notify, wall_clock, sleep)
    elif args.command == "pause":
        with _connect() as conn:
            run = _open_run(conn, "nothing to pause")
            if run.status == "paused":
                print(f"run {run.run_id} is already paused at {clock.instant(run.sim_now)}")
                return
            runs.set_status(conn, run.run_id, "paused")
            print(
                f"run {run.run_id} will pause at its next tick"
                f" (clock at {clock.instant(run.sim_now)})"
            )
    elif args.command == "status":
        with _connect() as conn:
            found = runs.open_run(conn)
        print("no unfinished replay run in this database" if found is None else describe(found))


def _connect() -> psycopg.Connection[Any]:
    return psycopg.connect(database_url(), connect_timeout=5)


def _open_run(conn: psycopg.Connection[Any], remedy: str) -> ReplayRun:
    run = runs.open_run(conn)
    if run is None:
        sys.exit(f"no unfinished replay run in this database; {remedy}")
    return run


def _csv_dir(repo_root: Path, data_root: Path, population: str) -> Path:
    csv_dir = repo_root / data_root / population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")
    return csv_dir


def _load(csv_dir: Path) -> tuple[dict[str, pd.DataFrame], list[StreamEvent]]:
    print(f"reading {csv_dir}")
    frames = load_population(csv_dir)
    events = ordered_events(frames["encounters"], frames["medications"], frames["conditions"])
    print(f"{len(events)} clinical events in the stream")
    return frames, events


def _start(
    conn: psycopg.Connection[Any],
    config: ReplayConfig,
    frames: dict[str, pd.DataFrame],
    events: Sequence[StreamEvent],
) -> ReplayRun:
    """Preload, then open the row: a run exists only once its history is in state."""
    start_at = clock.day_start(config.start)
    end_at = clock.day_start(config.end)
    print(f"loading history before {clock.instant(start_at)} into state")
    loaded = preload_history(conn, frames, events, clock.instant(start_at))
    print(preload_report(loaded))
    try:
        run = runs.create_run(
            conn,
            population=config.population,
            start_at=start_at,
            end_at=end_at,
            acceleration=config.acceleration,
        )
    except runs.OpenRunError as error:
        sys.exit(str(error))
    print(f"run {run.run_id} opened: {clock.instant(start_at)} to {clock.instant(end_at)}")
    return run


def _drive(
    conn: psycopg.Connection[Any],
    run: ReplayRun,
    events: Sequence[StreamEvent],
    args: argparse.Namespace,
    poster_factory: PosterFactory,
    notify: Notifier,
    wall_clock: clock.WallClock,
    sleep: harness.Sleep,
) -> None:
    """Run the loop from the row, then record how it stopped and tell the operator."""
    pacing = pacing_for(run.acceleration, args)
    with poster_factory(args.port) as poster, InterruptGuard() as guard:
        summary = harness.run_replay(
            conn,
            run,
            events,
            poster,
            pacing=pacing,
            wall_clock=wall_clock,
            sleep=sleep,
            pause_requested=guard.pause_requested,
        )
    if summary.finished:
        runs.set_status(conn, run.run_id, "finished")
        notify(f"finished at {clock.instant(summary.sim_to)}")
    else:
        if runs.read_status(conn, run.run_id) != "paused":
            runs.set_status(conn, run.run_id, "paused")
        notify(f"paused at {clock.instant(summary.sim_to)}")
    print(harness.report(summary))


def preload_report(loaded: PreloadSummary) -> str:
    by_kind = ", ".join(f"{count} {kind}" for kind, count in sorted(loaded.rows_loaded.items()))
    return "\n".join(
        [
            f"history loaded: {by_kind} ({loaded.rows_already_present} already present)",
            f"{loaded.discharges_unscored} discharges before the start left unscored",
        ]
    )


def describe(run: ReplayRun) -> str:
    """The run row as an operator reads it."""
    covered = (run.sim_now - run.start_at) / (run.end_at - run.start_at)
    cursor = "nothing posted yet" if run.cursor is None else f"last event at {run.cursor[0]}"
    return "\n".join(
        [
            f"run {run.run_id}: population {run.population}, status {run.status}",
            f"simulated span: {clock.instant(run.start_at)} to {clock.instant(run.end_at)}",
            f"clock: {clock.instant(run.sim_now)} ({covered:.1%} of the span)",
            f"cursor: {cursor}",
            f"created {_wall(run.created_at)}, last written {_wall(run.updated_at)}",
        ]
    )


def _wall(moment: datetime) -> str:
    """A wall-clock timestamp, formatted apart from simulated instants on purpose."""
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == "__main__":
    main()
