"""The scoring path end to end: posted event to persisted prediction.

This is where the ingestion boundary, the state layer, the shared cohort
and feature modules, the loaded model, and the provenance log meet.

The rules these tests pin:

- An admitted discharge produces exactly one log row whose provenance
  fields identify the exact model and the exact input that produced the
  score, and whose stored feature values equal what the batch pipeline
  computes for the same rows.
- Everything that is not a scoring event still updates state and logs
  nothing: non-encounter events, encounters the cohort rules exclude,
  and encounters still open.
- Re-posting a scored discharge writes no second row, so a replay resume
  cannot duplicate history.
- An encounter that reached state while its score did not gets scored on
  the next post. This is the crash window between the two commits, and
  closing it is what makes a resumed run gap-free.
- An inpatient discharge whose patient has never been posted is refused
  loudly rather than scored or silently skipped; the event is still
  stored, so re-posting it after the demographics arrive scores it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring import predictions, state, train
from risk_scoring.cohort import COHORT_VERSION, build_cohort
from risk_scoring.features import FEATURE_COLUMNS, FEATURE_VERSION, build_features
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

PATIENT_FIELDS = ("Id", "BIRTHDATE", "DEATHDATE")
ENCOUNTER_FIELDS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
MEDICATION_FIELDS = ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE")
CONDITION_FIELDS = ("START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION")

PATIENT_ROW = make_patient_row(Id="patient-1", BIRTHDATE="1951-04-09")
DISCHARGE_ROW = make_encounter_row(
    Id="encounter-1",
    PATIENT="patient-1",
    ENCOUNTERCLASS="inpatient",
    START="2024-05-01T08:00:00Z",
    STOP="2024-05-04T17:30:00Z",
)


@pytest.fixture()
def client(trained_repo: tuple[Path, train.TrainingResult], db_url: str) -> Iterator[TestClient]:
    root, trained = trained_repo
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, db_url)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def conn(db_url: str) -> Iterator[psycopg.Connection[Any]]:
    """A second connection for reading what the service wrote."""
    connection: psycopg.Connection[Any] = psycopg.connect(db_url, connect_timeout=2)
    try:
        yield connection
    finally:
        connection.close()


def _event(event_type: str, row: dict[str, str], fields: tuple[str, ...]) -> dict[str, object]:
    return {"event_type": event_type, "payload": {field: row[field] for field in fields}}


def _patient(**overrides: str) -> dict[str, object]:
    return _event("patient", {**PATIENT_ROW, **overrides}, PATIENT_FIELDS)


def _discharge(**overrides: str) -> dict[str, object]:
    return _event("encounter", {**DISCHARGE_ROW, **overrides}, ENCOUNTER_FIELDS)


def _post(client: TestClient, event: dict[str, object], expected: int = 202) -> dict[str, Any]:
    response = client.post("/events", json=event)
    assert response.status_code == expected, response.text
    body: dict[str, Any] = response.json()
    return body


def _prediction_count(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute("SELECT count(*) FROM predictions").fetchone()
    assert row is not None
    return int(row[0])


def _batch_features(rows: dict[str, list[dict[str, str]]]) -> pd.DataFrame:
    """What the training pipeline computes over exactly the same rows."""
    frames = {name: pd.DataFrame(values) for name, values in rows.items()}
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    return build_features(cohort, frames["encounters"], frames["medications"], frames["conditions"])


# --- the scored path ---


def test_admitted_discharge_is_scored_and_logged_with_full_provenance(
    client: TestClient,
    conn: psycopg.Connection[Any],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    _, trained = trained_repo
    _post(client, _patient())
    event = _discharge()

    body = _post(client, event)

    assert body["scored"] is True
    assert body["prediction_id"] is not None
    assert isinstance(body["score"], float)

    stored = predictions.prediction_for_encounter(conn, "encounter-1")
    assert stored is not None
    assert stored.prediction_id == body["prediction_id"]
    assert stored.patient_id == "patient-1"
    assert stored.encounter_id == "encounter-1"
    assert stored.event_time == datetime(2024, 5, 4, 17, 30, tzinfo=UTC)
    assert stored.input_hash == payload_hash(event)
    assert stored.model_name == MODEL_NAME
    assert stored.model_version == trained.model_version
    assert stored.feature_version == FEATURE_VERSION
    assert stored.cohort_version == COHORT_VERSION
    assert stored.score == body["score"]
    assert 0.0 <= stored.score <= 1.0


def test_logged_features_equal_the_batch_pipeline_values(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    """The stored input is the model's actual input, not an approximation of it."""
    medication = make_medication_row(
        PATIENT="patient-1", ENCOUNTER="encounter-1", START="2024-05-01T09:00:00Z", STOP=""
    )
    condition = make_condition_row(
        PATIENT="patient-1", ENCOUNTER="encounter-1", START="2024-05-01", STOP=""
    )
    _post(client, _patient())
    _post(client, _event("medication", medication, MEDICATION_FIELDS))
    _post(client, _event("condition", condition, CONDITION_FIELDS))
    _post(client, _discharge())

    batch = _batch_features(
        {
            "patients": [PATIENT_ROW],
            "encounters": [DISCHARGE_ROW],
            "medications": [medication],
            "conditions": [condition],
        }
    )
    expected = {name: float(batch.iloc[0][name]) for name in FEATURE_COLUMNS[2:]}

    stored = predictions.prediction_for_encounter(conn, "encounter-1")
    assert stored is not None
    assert stored.features == expected
    assert stored.features["active_medication_count"] == 1.0
    assert stored.features["active_disorder_count"] == 1.0


# --- everything that is not a scoring event ---


def test_non_encounter_events_update_state_and_log_nothing(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    _post(client, _patient())
    body = _post(
        client,
        _event(
            "medication",
            make_medication_row(PATIENT="patient-1", ENCOUNTER="encounter-1"),
            MEDICATION_FIELDS,
        ),
    )

    assert body["scored"] is False
    assert body["prediction_id"] is None
    assert body["score"] is None
    assert _prediction_count(conn) == 0
    assert len(state.patient_history(conn, "patient-1").medications) == 1


@pytest.mark.parametrize(
    ("label", "patient_overrides", "encounter_overrides"),
    [
        ("outpatient", {}, {"ENCOUNTERCLASS": "outpatient"}),
        ("emergency", {}, {"ENCOUNTERCLASS": "emergency"}),
        ("minor", {"BIRTHDATE": "2010-01-01"}, {}),
        ("in-hospital death", {"DEATHDATE": "2024-05-03"}, {}),
    ],
)
def test_excluded_encounters_update_state_but_log_nothing(
    client: TestClient,
    conn: psycopg.Connection[Any],
    label: str,
    patient_overrides: dict[str, str],
    encounter_overrides: dict[str, str],
) -> None:
    _post(client, _patient(**patient_overrides))

    body = _post(client, _discharge(**encounter_overrides))

    assert body["scored"] is False, label
    assert _prediction_count(conn) == 0
    assert len(state.patient_history(conn, "patient-1").encounters) == 1


def test_open_encounter_is_stored_but_not_scored(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    """A stay with no STOP has not discharged, so there is nothing to score yet."""
    _post(client, _patient())

    body = _post(client, _discharge(STOP=""))

    assert body["scored"] is False
    assert _prediction_count(conn) == 0
    assert len(state.patient_history(conn, "patient-1").encounters) == 1


# --- idempotency and the crash window ---


def test_reposting_a_scored_discharge_writes_no_second_row(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    _post(client, _patient())
    first = _post(client, _discharge())
    second = _post(client, _discharge())

    assert first["scored"] is True
    assert second["scored"] is False
    assert second["prediction_id"] is None
    assert _prediction_count(conn) == 1


def test_encounter_stored_without_its_score_is_scored_on_repost(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    """The window between the state commit and the prediction commit self-heals."""
    _post(client, _patient())
    state.record_encounter(conn, state.EncounterEvent.from_row(DISCHARGE_ROW))
    assert _prediction_count(conn) == 0

    body = _post(client, _discharge())

    assert body["scored"] is True
    assert _prediction_count(conn) == 1


# --- ordering violations and rejections ---


def test_discharge_for_unknown_patient_is_refused_then_scores_once_demographics_arrive(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    """Demographics must precede the discharge; a violation is loud, not silent."""
    refused = client.post("/events", json=_discharge())
    assert refused.status_code == 422
    assert "patient-1" in refused.text
    assert _prediction_count(conn) == 0
    assert len(state.patient_history(conn, "patient-1").encounters) == 1

    _post(client, _patient())
    body = _post(client, _discharge())

    assert body["scored"] is True
    assert _prediction_count(conn) == 1


def test_rejected_event_writes_nothing(client: TestClient, conn: psycopg.Connection[Any]) -> None:
    _post(client, _patient())
    bad = client.post("/events", json=_discharge(START="2024-05-01"))

    assert bad.status_code == 422
    assert _prediction_count(conn) == 0
    assert len(state.patient_history(conn, "patient-1").encounters) == 0


def test_divergent_repost_is_rejected_and_leaves_the_first_score_standing(
    client: TestClient, conn: psycopg.Connection[Any]
) -> None:
    """Two contradicting versions of one encounter is a producer bug, not a merge."""
    _post(client, _patient())
    _post(client, _discharge())

    conflict = client.post("/events", json=_discharge(STOP="2024-05-05T17:30:00Z"))

    assert conflict.status_code == 409
    stored = predictions.prediction_for_encounter(conn, "encounter-1")
    assert stored is not None
    assert stored.event_time == datetime(2024, 5, 4, 17, 30, tzinfo=UTC)
    assert _prediction_count(conn) == 1
