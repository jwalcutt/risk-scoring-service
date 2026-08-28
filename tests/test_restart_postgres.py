"""Restart and state rebuild.

The service holds nothing across requests that it did not derive at
startup: the loaded model, the connection pool, the frozen config, and the
git SHA. Everything a score depends on lives in Postgres. That makes the
rebuild path structurally trivial, so these tests exist to keep it that
way and to prove that a restarted service resumes with no log gaps and
correct state, rather than assuming it from the structure.

The rules these tests pin:

- A fresh instance scores correctly from history it never ingested. No
  warm-up, no replay of earlier events, no cache to prime.
- An interrupted run and an uninterrupted run over the same stream produce
  the same prediction log: same rows, same order, same hashes, same
  scores, no gaps and no duplicates.
- A discharge whose state write landed while its score did not is scored
  by the next instance to see it. This is the crash window between the two
  commits, and closing it across a restart is what makes a resume gap-free
  rather than merely idempotent.
- The lifespan writes a fixed set of names onto the app state, so a
  per-patient cache added later breaks a test instead of passing review.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from factories import make_encounter_row, make_patient_row, write_skew_population
from risk_scoring import predictions, state, train
from risk_scoring.cohort import build_cohort
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.stream import EVENT_FIELDS, build_stream
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

# Assigned by the database, so they differ between two runs of one stream
# by design and say nothing about whether the runs agree.
_VOLATILE_COLUMNS = ("prediction_id", "scored_at")


def _event(kind: str, row: dict[str, str]) -> dict[str, Any]:
    """One hand-built row as a posted envelope, projected like the stream."""
    return {"event_type": kind, "payload": {name: row[name] for name in EVENT_FIELDS[kind]}}


@pytest.fixture(scope="module")
def population(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """The boundary population, read exactly as the training pipeline reads it."""
    csv_dir = tmp_path_factory.mktemp("restart-population") / "csv"
    write_skew_population(csv_dir)
    return {
        name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in ("patients", "encounters", "medications", "conditions")
    }


@pytest.fixture(scope="module")
def stream(population: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """The whole population as one ordered event stream, demographics first."""
    return build_stream(population)


@pytest.fixture()
def serve(
    trained_repo: tuple[Path, train.TrainingResult],
) -> Callable[[str], Any]:
    """Build one service instance against a DSN, as a context manager."""
    root, trained = trained_repo

    @contextmanager
    def instance(dsn: str) -> Iterator[TestClient]:
        app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, dsn)
        with TestClient(app) as client:
            yield client

    return instance


def _ingest(
    serve: Callable[[str], Any],
    dsn: str,
    events: Sequence[dict[str, Any]],
    *,
    restart_after: int | None = None,
) -> None:
    """Post the stream, tearing the instance down and rebuilding it once."""
    chunks = (
        [list(events)]
        if restart_after is None
        else [list(events[:restart_after]), list(events[restart_after:])]
    )
    for chunk in chunks:
        with serve(dsn) as client:
            for event in chunk:
                response = client.post("/events", json=event)
                assert response.status_code == 202, response.text


def _log(dsn: str) -> list[dict[str, Any]]:
    """The prediction log, minus the fields the database assigns."""
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        rows = predictions.all_predictions(conn)
    return [
        {name: value for name, value in asdict(row).items() if name not in _VOLATILE_COLUMNS}
        for row in rows
    ]


def _first_admitted_discharge(events: Sequence[dict[str, Any]]) -> int:
    for index, event in enumerate(events):
        payload = event["payload"]
        if (
            event["event_type"] == "encounter"
            and payload["ENCOUNTERCLASS"] == "inpatient"
            and payload["STOP"]
        ):
            return index
    raise AssertionError("the stream carries no inpatient discharge")


RESTART_POINTS: list[Any] = [
    pytest.param(lambda events: 0, id="before-the-first-event"),
    pytest.param(lambda events: 1, id="after-the-first-event"),
    pytest.param(lambda events: len(events) // 2, id="mid-stream"),
    pytest.param(_first_admitted_discharge, id="immediately-before-a-discharge"),
    pytest.param(lambda events: _first_admitted_discharge(events) + 1, id="just-after-a-discharge"),
    pytest.param(lambda events: len(events), id="after-the-last-event"),
]


@pytest.mark.parametrize("choose_restart", RESTART_POINTS)
def test_an_interrupted_run_logs_exactly_what_an_uninterrupted_run_logs(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
    stream: list[dict[str, Any]],
    choose_restart: Callable[[Sequence[dict[str, Any]]], int],
) -> None:
    uninterrupted_dsn = db_url_factory()
    interrupted_dsn = db_url_factory()

    _ingest(serve, uninterrupted_dsn, stream)
    _ingest(serve, interrupted_dsn, stream, restart_after=choose_restart(stream))

    assert _log(interrupted_dsn) == _log(uninterrupted_dsn)


def test_an_interrupted_run_scores_every_discharge_exactly_once(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
    stream: list[dict[str, Any]],
    population: dict[str, pd.DataFrame],
) -> None:
    """No gaps and no duplicates, stated against the batch pipeline's own count."""
    dsn = db_url_factory()

    _ingest(serve, dsn, stream, restart_after=len(stream) // 2)

    admitted = build_cohort(population["encounters"], population["patients"]).frame
    logged = [row["encounter_id"] for row in _log(dsn)]
    assert sorted(logged) == sorted(admitted["encounter_id"].tolist())
    assert len(logged) == len(set(logged))


def test_a_fresh_instance_scores_from_history_it_never_ingested(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
) -> None:
    """The rebuild path: everything the score depends on comes back from Postgres."""
    patient = make_patient_row(Id="patient-1", BIRTHDATE="1951-04-09")
    prior = make_encounter_row(
        Id="encounter-prior",
        PATIENT="patient-1",
        ENCOUNTERCLASS="inpatient",
        START="2024-02-01T08:00:00Z",
        STOP="2024-02-05T08:00:00Z",
    )
    index = make_encounter_row(
        Id="encounter-index",
        PATIENT="patient-1",
        ENCOUNTERCLASS="inpatient",
        START="2024-05-01T08:00:00Z",
        STOP="2024-05-04T17:30:00Z",
    )
    history = [_event("patient", patient), _event("encounter", prior)]
    events = [*history, _event("encounter", index)]

    together_dsn = db_url_factory()
    _ingest(serve, together_dsn, events)

    apart_dsn = db_url_factory()
    _ingest(serve, apart_dsn, events, restart_after=len(history))

    scored = _log(apart_dsn)
    assert len(scored) == 2
    assert scored == _log(together_dsn)
    # The prior stay is only visible through state, so a fresh instance that
    # missed it would score the index discharge with no history at all.
    assert scored[1]["features"]["prior_inpatient_180d"] == 1.0
    assert scored[1]["features"]["days_since_prev_discharge"] == 86.0


def test_a_discharge_stored_without_its_score_is_scored_after_a_restart(
    serve: Callable[[str], Any],
    db_url_factory: Callable[[], str],
) -> None:
    """The crash window between the two commits, reopened across a restart."""
    dsn = db_url_factory()
    patient = make_patient_row(Id="patient-1", BIRTHDATE="1951-04-09")
    discharge = make_encounter_row(
        Id="encounter-1",
        PATIENT="patient-1",
        ENCOUNTERCLASS="inpatient",
        START="2024-05-01T08:00:00Z",
        STOP="2024-05-04T17:30:00Z",
    )
    _ingest(serve, dsn, [_event("patient", patient)])

    # The state write commits on its own, so a process can die here.
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        state.record_encounter(conn, state.EncounterEvent.from_row(discharge))
        assert predictions.all_predictions(conn) == []

    _ingest(serve, dsn, [_event("encounter", discharge)])

    (logged,) = _log(dsn)
    assert logged["encounter_id"] == "encounter-1"


def test_startup_holds_nothing_but_what_it_derives(
    serve: Callable[[str], Any], db_url: str
) -> None:
    """Restated literally: a per-patient cache added later must break a test."""
    with serve(db_url) as client:
        held = set(client.app.state._state)

    assert held == {"model", "pool", "config", "git_sha"}
