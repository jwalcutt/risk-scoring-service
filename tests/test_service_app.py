"""Tests for the service app factory, startup, and endpoints.

The rules these tests pin:

- Startup loads the pinned model version from the MLflow registry and
  opens the connection pool, and fails loudly, at startup rather than as
  a later 500, when either the pinned version or the database is absent.
- /health answers ok only once the lifespan has run, so a 200 implies
  the model loaded and the database is reachable.
- /version reports the full provenance set: model name and pinned
  version, FEATURE_VERSION, COHORT_VERSION, and the git SHA when one is
  resolvable.
- POST /events acknowledges a valid event with 202 and the input hash of
  the raw posted object: the hash is computed from the body as received,
  so equivalent JSON texts with different key orders hash identically.
- Bad shape and bad field format are both rejected with 422, never a
  5xx and never a silent drop.

What the accepted events actually do to state and to the prediction log
is pinned separately, in test_service_ingest_postgres.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factories import make_encounter_row, make_patient_row
from risk_scoring import train
from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.app import create_app, resolve_git_sha
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

# The app opens its connection pool at startup and refuses to start without
# a reachable database, so every test that builds one needs Postgres.
pytestmark = pytest.mark.db

ENCOUNTER_FIELDS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
PATIENT_FIELDS = ("Id", "BIRTHDATE", "DEATHDATE")

# --- fixtures ---


@pytest.fixture()
def client(trained_repo: tuple[Path, train.TrainingResult], db_url: str) -> Iterator[TestClient]:
    root, trained = trained_repo
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, db_url)
    with TestClient(app) as test_client:
        yield test_client


def _event(event_type: str, row: dict[str, str], fields: tuple[str, ...]) -> dict[str, object]:
    return {"event_type": event_type, "payload": {field: row[field] for field in fields}}


def _patient_event() -> dict[str, object]:
    return _event("patient", make_patient_row(), PATIENT_FIELDS)


def _encounter_event() -> dict[str, object]:
    row = make_encounter_row(ENCOUNTERCLASS="inpatient")
    return _event("encounter", row, ENCOUNTER_FIELDS)


# --- startup ---


def test_startup_loads_pinned_model_and_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_fails_loudly_when_pinned_version_absent(
    trained_repo: tuple[Path, train.TrainingResult], db_url: str
) -> None:
    root, _ = trained_repo
    app = create_app(ServiceConfig(MODEL_NAME, 999), root, db_url)
    with pytest.raises(RuntimeError, match=r"readmission-risk.*999"), TestClient(app):
        pass


def test_startup_fails_loudly_when_registry_empty(repo_root: Path, db_url: str) -> None:
    app = create_app(ServiceConfig(MODEL_NAME, 1), repo_root, db_url)
    with pytest.raises(RuntimeError, match="version 1"), TestClient(app):
        pass


def test_startup_fails_loudly_when_database_unreachable(
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    """A 200 from /health has to mean the service can actually store an event."""
    root, trained = trained_repo
    unreachable = "postgresql://risk:risk@127.0.0.1:1/risk_scoring"
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, unreachable)
    with pytest.raises(RuntimeError, match="database"), TestClient(app):
        pass


# --- version endpoint ---


def test_version_endpoint_reports_all_provenance_fields(
    client: TestClient, trained_repo: tuple[Path, train.TrainingResult]
) -> None:
    _, trained = trained_repo
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == MODEL_NAME
    assert body["model_version"] == trained.model_version
    assert body["feature_version"] == FEATURE_VERSION
    assert body["cohort_version"] == COHORT_VERSION
    assert "git_sha" in body
    assert body["git_sha"] is None or isinstance(body["git_sha"], str)


def test_resolve_git_sha_in_repo_and_outside(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    commit = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty"]
    subprocess.run([*commit, "-q", "-m", "x"], cwd=repo, check=True)
    sha = resolve_git_sha(repo)
    assert sha is not None
    assert len(sha) == 40

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert resolve_git_sha(bare) is None


# --- ingestion endpoint ---


def test_post_event_returns_202_with_input_hash(client: TestClient) -> None:
    client.post("/events", json=_patient_event())
    event = _encounter_event()

    response = client.post("/events", json=event)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["event_type"] == "encounter"
    assert body["input_hash"] == payload_hash(event)


def test_post_all_four_event_types_accepted(client: TestClient) -> None:
    events: list[dict[str, object]] = [
        _patient_event(),
        _encounter_event(),
        {
            "event_type": "medication",
            "payload": {
                "START": "2024-01-01T08:00:00Z",
                "STOP": "",
                "PATIENT": "patient-1",
                "ENCOUNTER": "encounter-1",
                "CODE": "308136",
            },
        },
        {
            "event_type": "condition",
            "payload": {
                "START": "2024-01-01",
                "STOP": "",
                "PATIENT": "patient-1",
                "ENCOUNTER": "encounter-1",
                "SYSTEM": "SNOMED-CT",
                "CODE": "444814009",
                "DESCRIPTION": "Viral sinusitis (disorder)",
            },
        },
    ]
    for event in events:
        response = client.post("/events", json=event)
        assert response.status_code == 202, response.text
        assert response.json()["event_type"] == event["event_type"]


def test_hash_covers_raw_body_not_model(client: TestClient) -> None:
    event = _patient_event()
    payload = dict(event["payload"])  # type: ignore[arg-type]
    ordered = json.dumps(event)
    reordered = json.dumps(
        {"payload": dict(reversed(list(payload.items()))), "event_type": "patient"}
    )
    headers = {"content-type": "application/json"}
    first = client.post("/events", content=ordered, headers=headers)
    second = client.post("/events", content=reordered, headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["input_hash"] == second.json()["input_hash"]

    altered = json.loads(ordered)
    altered["payload"]["Id"] = "patient-2"
    third = client.post("/events", json=altered)
    assert third.json()["input_hash"] != first.json()["input_hash"]


def test_malformed_payloads_rejected_4xx(client: TestClient) -> None:
    event = _encounter_event()
    payload = dict(event["payload"])  # type: ignore[arg-type]

    missing = {field: value for field, value in payload.items() if field != "PATIENT"}
    unknown_type = {"event_type": "observation", "payload": payload}
    extra_field = {"event_type": "encounter", "payload": {**payload, "PAYER": "payer-1"}}
    bad_format = {"event_type": "encounter", "payload": {**payload, "START": "2024-01-01"}}
    empty_identity = {"event_type": "encounter", "payload": {**payload, "PATIENT": ""}}

    for bad in (
        {"event_type": "encounter", "payload": missing},
        unknown_type,
        extra_field,
        bad_format,
        empty_identity,
    ):
        response = client.post("/events", json=bad)
        assert response.status_code == 422, response.text

    invalid_json = client.post(
        "/events", content="{not json", headers={"content-type": "application/json"}
    )
    assert invalid_json.status_code == 422


def test_resolve_git_sha_prefers_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image carries no .git, so the build stamps the SHA in instead."""
    stamped = "b" * 40
    monkeypatch.setenv("RISK_SCORING_GIT_SHA", stamped)

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert resolve_git_sha(bare) == stamped


def test_an_empty_git_sha_override_falls_back_to_the_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset build argument arrives as an empty string, which is not a SHA."""
    monkeypatch.setenv("RISK_SCORING_GIT_SHA", "")

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert resolve_git_sha(bare) is None
