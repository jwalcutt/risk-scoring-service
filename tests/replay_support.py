"""Helpers shared by the replay tests that drive a real service.

Three test files run the tick loop against an in-process service over a
throwaway database: the harness tests, the byte-identity matrix across
pauses and kills, and the command tests. They need the same synthetic
population, the same poster over the test client, the same preparation
of a run row, and the same view of the prediction log. Those live here so
the files state their rules rather than their plumbing.

Not a conftest: nothing here is a fixture. Each file wraps what it needs
in fixtures of the scope it wants.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from fastapi.testclient import TestClient

from factories import write_skew_population
from risk_scoring import label_log, predictions, train
from risk_scoring.populations import load_population
from risk_scoring.replay import clock, preload, release, runs
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.stream import StreamEvent, ordered_events
from risk_scoring.train import MODEL_NAME

# The skew population's replay span: four cohort discharges fall before
# this start and six after it, and the population's last event is dated
# the day before this end.
START = datetime(2024, 4, 1, tzinfo=UTC)
END = datetime(2024, 8, 7, tzinfo=UTC)
POPULATION = "skew"

MAX_SPEED = clock.Pacing(acceleration=4, max_speed=True)

# Assigned by the database, so they differ between two runs of one stream
# by design and say nothing about whether the runs agree. A bigserial
# consumes a value even when the log's conflict clause drops a re-post,
# so ids gap after a resume; scored_at is the wall clock at write.
VOLATILE_COLUMNS = ("prediction_id", "scored_at")

# The labels table's twins: label_id is a bigserial that a dropped
# re-release still consumes, recorded_at is the wall clock at write, and
# prediction_id is whatever the log assigned.
LABEL_VOLATILE_COLUMNS = ("label_id", "prediction_id", "recorded_at")

Serve = Callable[[str], AbstractContextManager[TestClient]]


class KilledMidTick(RuntimeError):
    """The harness process died after a post and before its checkpoint."""


class ClientPoster:
    """The poster the loop needs, backed by the FastAPI test client.

    ``die_after`` kills the poster after that many posts, so a test can
    stand in for a process that died before its checkpoint. Zero dies
    before the first post is made.
    """

    def __init__(self, client: TestClient, *, die_after: int | None = None) -> None:
        self.client = client
        self.die_after = die_after
        self.posted: list[Mapping[str, Any]] = []
        self.acks: list[dict[str, Any]] = []

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if self.die_after == 0:
            raise KilledMidTick("died before the first post")
        response = self.client.post("/events", json=event)
        if response.status_code != 202:
            raise RuntimeError(f"{event.get('event_type')} refused: {response.status_code}")
        self.posted.append(event)
        ack = dict(response.json())
        self.acks.append(ack)
        if self.die_after is not None and len(self.posted) == self.die_after:
            raise KilledMidTick(f"died after {self.die_after} posts")
        return ack


def skew_frames(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Write the skew population to ``csv_dir`` and read it back as the CLIs do."""
    write_skew_population(csv_dir)
    return load_population(csv_dir)


def stream_of(frames: Mapping[str, pd.DataFrame]) -> list[StreamEvent]:
    return ordered_events(frames["encounters"], frames["medications"], frames["conditions"])


def schedule_of(frames: Mapping[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return release.label_schedule(frames)


def serving(trained_repo: tuple[Path, train.TrainingResult]) -> Serve:
    """A factory of in-process services over the trained fixture, one per DSN."""
    root, trained = trained_repo

    @contextmanager
    def instance(dsn: str) -> Iterator[TestClient]:
        app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, dsn)
        with TestClient(app) as client:
            yield client

    return instance


def prepare(
    dsn: str,
    frames: Mapping[str, pd.DataFrame],
    events: list[StreamEvent],
    *,
    start: datetime = START,
    end: datetime = END,
) -> runs.ReplayRun:
    """Preload history and open the run row, as the start command does."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        preload.preload_history(conn, frames, events, clock.instant(start))
        return runs.create_run(
            conn, population=POPULATION, start_at=start, end_at=end, acceleration=4
        )


def read_log(dsn: str) -> list[dict[str, Any]]:
    """The prediction log minus the columns the database assigns."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        rows = predictions.all_predictions(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in VOLATILE_COLUMNS}
        for row in rows
    ]


def read_labels(dsn: str) -> list[dict[str, Any]]:
    """The labels table minus the columns the database assigns."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        rows = label_log.all_labels(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in LABEL_VOLATILE_COLUMNS}
        for row in rows
    ]


def read_outputs(dsn: str) -> dict[str, list[dict[str, Any]]]:
    """Everything a replay writes that two runs of one stream must agree on."""
    return {"predictions": read_log(dsn), "labels": read_labels(dsn)}


def tick_containing(at: str, *, start: datetime = START) -> datetime:
    """The first tick boundary at or after an event's instant, from ``start``."""
    moment = datetime.fromisoformat(at)
    return start + math.ceil((moment - start) / clock.TICK) * clock.TICK


def first_discharge_index(replayed: list[StreamEvent]) -> int:
    """One past the position of the first inpatient discharge in the replayed stream."""
    for index, event in enumerate(replayed):
        if event.kind == "encounter" and event.row["ENCOUNTERCLASS"] == "inpatient":
            return index + 1
    raise AssertionError("no inpatient discharge after the start")
