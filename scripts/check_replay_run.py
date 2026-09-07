"""Audit what a replay left behind against the export and the batch pipeline.

The CI tests prove the replay's exit criteria over a synthetic population.
This script runs the same checks against a real database after a real
run, over the frozen population it streamed, which is local-only and so
cannot run in CI:

    RISK_SCORING_DATABASE_URL=postgresql://risk:risk@localhost:5433/replay_exit \\
        python scripts/check_replay_run.py --population baseline --end 2025-07-01

It prints the run row and the table counts, then checks, in order: the
never-early query returns zero; every released label equals the batch
label for its discharge; every unlabelled discharge lies inside the final
30 days of the span; and realized performance over the labelled window,
the join of the log and the labels, equals what the cohort, feature, and
label modules and the logged model version compute over the same
discharges, with no tolerance. Any failure exits nonzero.

The run must be splice-free. A spliced-in population's ids are rewritten
at load, so its export would have to be read through the same rewrite
before it could meet the log, and this script does not do that.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from risk_scoring.db import database_url
from risk_scoring.labels import READMISSION_WINDOW_DAYS
from risk_scoring.populations import load_population
from risk_scoring.replay import audit, runs
from risk_scoring.replay.config import DEFAULT_CONFIG_RELPATH, load_config
from risk_scoring.replay.realized import RealizedPerformance, realized_performance

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOW = timedelta(days=READMISSION_WINDOW_DAYS)


def _instant(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _count(conn: psycopg.Connection[Any], table: str) -> int:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    row = conn.execute(query).fetchone()
    assert row is not None
    return int(row[0])


def _earliest_unlabelled(conn: psycopg.Connection[Any]) -> datetime | None:
    row = conn.execute(
        "SELECT min(p.event_time) FROM predictions AS p"
        " LEFT JOIN labels AS l USING (prediction_id) WHERE l.label_id IS NULL"
    ).fetchone()
    assert row is not None
    value = row[0]
    return None if value is None else datetime.fromisoformat(str(value))


def _describe(run: runs.ReplayRun) -> str:
    return (
        f"run {run.run_id}: population {run.population}, status {run.status},"
        f" {run.start_at.isoformat()} to {run.end_at.isoformat()},"
        f" clock at {run.sim_now.isoformat()}"
    )


def _metrics(name: str, result: RealizedPerformance) -> str:
    prevalence = "none" if result.prevalence is None else f"{result.prevalence:.4f}"
    auroc = "none" if result.auroc is None else f"{result.auroc:.4f}"
    return f"{name}: {result.count} discharges, prevalence {prevalence}, AUROC {auroc}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_replay_run.py",
        description="Check a replay's tables against the export and the batch pipeline.",
    )
    parser.add_argument("--population", default="baseline")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / DEFAULT_CONFIG_RELPATH)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--database-url",
        default=None,
        help="defaults to RISK_SCORING_DATABASE_URL, then the Compose default",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.splices:
        sys.exit(f"{args.config} schedules a splice; this check reads one export only")
    start = _instant(config.start if args.start is None else args.start)
    end = _instant(config.end if args.end is None else args.end)
    labelled_end = end - WINDOW
    csv_dir = args.data_root / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")
    url = database_url() if args.database_url is None else args.database_url

    failures: list[str] = []
    with psycopg.connect(url, connect_timeout=5) as conn:
        run = runs.latest_run(conn)
        print("no replay run in this database" if run is None else _describe(run))
        model = audit.logged_model(conn)
        print(
            f"prediction log: {_count(conn, 'predictions')} rows,"
            f" model {model.name} version {model.version}"
        )
        print(f"labels table: {_count(conn, 'labels')} rows")

        early = audit.early_labels(conn)
        print(f"labels released within {READMISSION_WINDOW_DAYS} days of their discharge: {early}")
        if early:
            failures.append(f"{early} labels released early")

        print(f"reading {csv_dir}")
        frames = load_population(csv_dir)
        labels = audit.label_audit(conn, frames)
        print(
            f"labels against the export: {labels.checked} checked,"
            f" {len(labels.disagreements)} disagreements"
        )
        if not labels.ok:
            shown = ", ".join(labels.disagreements[:10])
            failures.append(f"labels disagree with the export for {shown}")

        earliest = _earliest_unlabelled(conn)
        if earliest is None:
            print("every scored discharge has a label")
        else:
            inside = earliest >= labelled_end
            where = "inside" if inside else "OUTSIDE"
            print(
                f"earliest unlabelled discharge {earliest.isoformat()}, {where} the final 30 days"
            )
            if not inside:
                failures.append(f"an unlabelled discharge at {earliest.isoformat()} is due a label")

        realized = realized_performance(conn, start, labelled_end)

    window = f"[{start.date()}, {labelled_end.date()})"
    print(_metrics(f"realized over {window}", realized))
    batch = audit.batch_performance(frames, model, start, labelled_end, repo_root=REPO_ROOT)
    print(_metrics("batch pipeline over the same discharges", batch))
    if realized != batch:
        failures.append("realized performance differs from the batch pipeline")

    if failures:
        print("MISMATCH: " + "; ".join(failures))
        sys.exit(1)
    print(
        f"MATCH: {realized.count} labelled discharges agree with the batch pipeline,"
        f" {labels.checked} labels agree with the export, none released early"
    )


if __name__ == "__main__":
    main()
