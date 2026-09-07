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
- A body larger than MAX_EVENT_BYTES is refused with 413 before it is
  buffered: on the declared Content-Length without reading a byte, and
  on a running total for a chunked body at the chunk that crosses the
  limit.
- POST /events requires the bearer token from RISK_SCORING_API_TOKEN
  and answers 401 without it or with a wrong one, before reading the
  body. /health and /version stay open. Startup refuses to run when the
  variable is unset or empty.

What the accepted events actually do to state and to the prediction log
is pinned separately, in test_service_ingest_postgres.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from factories import make_encounter_row, make_patient_row
from risk_scoring import train
from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.app import MAX_EVENT_BYTES, create_app, resolve_git_sha
from risk_scoring.service.auth import ENV_API_TOKEN, bearer_headers
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

# The app opens its connection pool at startup and refuses to start without
# a reachable database, so every test that builds one needs Postgres.
pytestmark = pytest.mark.db

ENCOUNTER_FIELDS = ("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS")
PATIENT_FIELDS = ("Id", "BIRTHDATE", "DEATHDATE")

# --- fixtures ---


@pytest.fixture()
def app(trained_repo: tuple[Path, train.TrainingResult], db_url: str) -> FastAPI:
    root, trained = trained_repo
    return create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, db_url)


@pytest.fixture()
def client(app: FastAPI, api_token: str) -> Iterator[TestClient]:
    """A started service whose every request carries the token."""
    with TestClient(app, headers=bearer_headers(api_token)) as test_client:
        yield test_client


@contextmanager
def _serving(
    trained_repo: tuple[Path, train.TrainingResult], db_url: str, pool_size: int, api_token: str
) -> Iterator[TestClient]:
    """A running instance whose pool is capped at ``pool_size`` connections."""
    root, trained = trained_repo
    config = ServiceConfig(MODEL_NAME, trained.model_version, pool_size=pool_size)
    with TestClient(create_app(config, root, db_url), headers=bearer_headers(api_token)) as client:
        yield client


@contextmanager
def _holding_connections(client: TestClient, count: int) -> Iterator[None]:
    """Take ``count`` connections out of the app's pool for the duration."""
    pool = client.app.state.pool
    held = [pool.getconn() for _ in range(count)]
    try:
        yield
    finally:
        for conn in held:
            pool.putconn(conn)


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


def test_resolve_git_sha_reports_unknown_when_git_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No git on the machine means no SHA, even inside a repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    commit = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty"]
    subprocess.run([*commit, "-q", "-m", "x"], cwd=repo, check=True)
    monkeypatch.delenv("RISK_SCORING_GIT_SHA", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd, *args, **kwargs: None)

    assert resolve_git_sha(repo) is None


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


# --- body size cap ---

TOO_LARGE = {"detail": "event body exceeds 1048576 bytes"}
JSON_HEADERS = {"content-type": "application/json"}


def _oversized_event_bytes() -> bytes:
    event = _patient_event()
    payload = dict(event["payload"])  # type: ignore[arg-type]
    payload["Id"] = "p" * (MAX_EVENT_BYTES + 1)
    return json.dumps({"event_type": "patient", "payload": payload}).encode()


def test_event_body_limit_is_one_mebibyte() -> None:
    assert MAX_EVENT_BYTES == 1_048_576


def test_oversized_declared_content_length_is_refused_unread(client: TestClient) -> None:
    """The body is not JSON, so reading it would have produced a 422, not a 413."""
    response = client.post(
        "/events",
        content=b"{not json",
        headers={**JSON_HEADERS, "content-length": str(MAX_EVENT_BYTES + 1)},
    )
    assert response.status_code == 413
    assert response.json() == TOO_LARGE


def test_oversized_chunked_body_is_refused(client: TestClient) -> None:
    """No Content-Length to check, so the running total has to catch it."""
    body = _oversized_event_bytes()
    half = len(body) // 2
    response = client.post(
        "/events", content=iter([body[:half], body[half:]]), headers=JSON_HEADERS
    )
    assert response.status_code == 413
    assert response.json() == TOO_LARGE


def test_chunked_body_is_refused_at_the_chunk_that_crosses_the_limit(repo_root: Path) -> None:
    """Two 512 KiB chunks sit at the limit; the third crosses it; the fourth is never read."""
    app = create_app(ServiceConfig(MODEL_NAME, 1), repo_root, "postgresql://unused")
    chunks = [b"x" * (512 * 1024)] * 4
    consumed = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal consumed
        chunk = chunks[consumed]
        consumed += 1
        return {"type": "http.request", "body": chunk, "more_body": consumed < len(chunks)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/events",
        "raw_path": b"/events",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(app(scope, receive, send))

    assert consumed == 3
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert json.loads(b"".join(m.get("body", b"") for m in sent[1:])) == TOO_LARGE


def test_event_under_the_limit_is_accepted_with_the_same_hash(client: TestClient) -> None:
    """The bounded read hands the handler the exact bytes, so the hash is unchanged."""
    event = _patient_event()
    body = json.dumps(event).encode()
    response = client.post(
        "/events", content=body, headers={**JSON_HEADERS, "content-length": str(len(body))}
    )
    assert response.status_code == 202
    assert response.json()["input_hash"] == payload_hash(event)


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


# --- bearer token ---


def test_post_event_without_a_bearer_token_is_401(client: TestClient) -> None:
    """Whoever can reach the port must not be able to write state."""
    del client.headers["Authorization"]

    response = client.post("/events", json=_patient_event())

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "RISK_SCORING_API_TOKEN" in response.json()["detail"]


def test_post_event_with_the_wrong_token_is_401(client: TestClient) -> None:
    detail = "a bearer token matching RISK_SCORING_API_TOKEN is required"
    for wrong in ("Bearer not-the-token", "Bearer test-token-and-more", "Basic test-token"):
        response = client.post("/events", json=_patient_event(), headers={"Authorization": wrong})
        assert response.status_code == 401, wrong
        assert response.json()["detail"] == detail


def test_the_token_is_checked_before_the_body(client: TestClient) -> None:
    """An unauthenticated caller learns nothing about the event schema."""
    del client.headers["Authorization"]
    response = client.post("/events", json={"event_type": "observation"})
    assert response.status_code == 401


def test_health_and_version_need_no_token(client: TestClient) -> None:
    del client.headers["Authorization"]
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200


def test_startup_fails_loudly_when_the_token_is_unset(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open service is the defect the token exists to prevent, so it must not start."""
    monkeypatch.delenv(ENV_API_TOKEN)
    with pytest.raises(RuntimeError, match=ENV_API_TOKEN), TestClient(app):
        pass


def test_startup_treats_an_empty_token_as_unset(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_API_TOKEN, "  ")
    with pytest.raises(RuntimeError, match=ENV_API_TOKEN), TestClient(app):
        pass


# --- the connection pool ---


def test_the_pool_holds_as_many_connections_as_the_config_allows(
    trained_repo: tuple[Path, train.TrainingResult], db_url: str, api_token: str
) -> None:
    """With one of two connections taken elsewhere, a post still has one to use."""
    with _serving(trained_repo, db_url, 2, api_token) as client:
        client.app.state.pool.timeout = 1.0
        with _holding_connections(client, 1):
            response = client.post("/events", json=_patient_event())

    assert response.status_code == 202


def test_an_exhausted_pool_answers_503_not_500(
    trained_repo: tuple[Path, train.TrainingResult], db_url: str, api_token: str
) -> None:
    """Waiting out the pool is a capacity condition, reported as one."""
    with _serving(trained_repo, db_url, 1, api_token) as client:
        client.app.state.pool.timeout = 0.5
        with _holding_connections(client, 1):
            response = client.post("/events", json=_patient_event())

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert "connection" in response.json()["detail"]
