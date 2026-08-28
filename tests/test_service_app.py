"""Tests for the service app factory, startup model loading, and endpoints.

The rules these tests pin:

- Startup loads the pinned model version from the MLflow registry and
  fails loudly, at startup rather than as a later 500, when the pinned
  version (or the whole registered model) is absent.
- /health answers ok only once the lifespan has run, so a 200 implies
  the model actually loaded.
- /version reports the full provenance set: model name and pinned
  version, FEATURE_VERSION, COHORT_VERSION, and the git SHA when one is
  resolvable.
- POST /events acknowledges a valid event with 202 and the input hash of
  the raw posted object: the hash is computed from the body as received,
  so equivalent JSON texts with different key orders hash identically.
- Malformed payloads are rejected with 422, never a 5xx and never a
  silent drop; the stubbed ingestion has no side effects of any kind.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from factories import make_encounter_row, write_training_csvs
from risk_scoring import train
from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.app import create_app, resolve_git_sha
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

# --- fixtures ---


@pytest.fixture(scope="module")
def trained_repo(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, train.TrainingResult]]:
    """One fast population trained and registered once for this module."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("service-repo")
    csv_dir = root / "data" / "baseline" / "csv"
    write_training_csvs(csv_dir)
    result = train.train(csv_dir, root)
    yield root, result
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


@pytest.fixture()
def client(trained_repo: tuple[Path, train.TrainingResult]) -> Iterator[TestClient]:
    root, trained = trained_repo
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root)
    with TestClient(app) as test_client:
        yield test_client


def _encounter_event() -> dict[str, object]:
    row = make_encounter_row(ENCOUNTERCLASS="inpatient")
    payload = {field: row[field] for field in ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")}
    return {"event_type": "encounter", "payload": payload}


# --- startup and model loading ---


def test_startup_loads_pinned_model_and_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_fails_loudly_when_pinned_version_absent(
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = trained_repo
    app = create_app(ServiceConfig(MODEL_NAME, 999), root)
    with pytest.raises(RuntimeError, match=r"readmission-risk.*999"), TestClient(app):
        pass


def test_startup_fails_loudly_when_registry_empty(repo_root: Path) -> None:
    app = create_app(ServiceConfig(MODEL_NAME, 1), repo_root)
    with pytest.raises(RuntimeError, match="version 1"), TestClient(app):
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
    event = _encounter_event()
    response = client.post("/events", json=event)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["event_type"] == "encounter"
    assert body["input_hash"] == payload_hash(event)


def test_post_all_four_event_types_accepted(client: TestClient) -> None:
    events: list[dict[str, object]] = [
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
        {
            "event_type": "patient",
            "payload": {"Id": "patient-1", "BIRTHDATE": "1970-01-01", "DEATHDATE": ""},
        },
    ]
    for event in events:
        response = client.post("/events", json=event)
        assert response.status_code == 202, response.text
        assert response.json()["event_type"] == event["event_type"]


def test_hash_covers_raw_body_not_model(client: TestClient) -> None:
    event = _encounter_event()
    payload = dict(event["payload"])  # type: ignore[arg-type]
    ordered = json.dumps(event)
    reordered = json.dumps(
        {"payload": dict(reversed(list(payload.items()))), "event_type": "encounter"}
    )
    headers = {"content-type": "application/json"}
    first = client.post("/events", content=ordered, headers=headers)
    second = client.post("/events", content=reordered, headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["input_hash"] == second.json()["input_hash"]

    altered = json.loads(ordered)
    altered["payload"]["Id"] = "encounter-2"
    third = client.post("/events", json=altered)
    assert third.json()["input_hash"] != first.json()["input_hash"]


def test_malformed_payloads_rejected_4xx(client: TestClient) -> None:
    event = _encounter_event()
    payload = dict(event["payload"])  # type: ignore[arg-type]

    missing = {field: value for field, value in payload.items() if field != "PATIENT"}
    unknown_type = {"event_type": "observation", "payload": payload}
    extra_field = {"event_type": "encounter", "payload": {**payload, "PAYER": "payer-1"}}

    for bad in (
        {"event_type": "encounter", "payload": missing},
        unknown_type,
        extra_field,
    ):
        response = client.post("/events", json=bad)
        assert response.status_code == 422, response.text

    invalid_json = client.post(
        "/events", content="{not json", headers={"content-type": "application/json"}
    )
    assert invalid_json.status_code == 422


def test_stubbed_ingestion_has_no_side_effects(client: TestClient) -> None:
    event = _encounter_event()
    first = client.post("/events", json=event)
    second = client.post("/events", json=event)
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json()
