"""FastAPI app factory: pinned model loading and provenance endpoints.

Judgment calls this module fixes:

- No module-level app: :func:`create_app` takes an already-loaded config
  and a repo root, so tests and the container construct the app the same
  way and nothing reads configuration at import time.
- The model loads once, at startup, by the explicit registered version
  the config pins. A missing version raises a ``RuntimeError`` naming
  the model, version, and registry URI, so a broken pin stops the
  service from starting instead of surfacing as a later 500.
- The ingestion handler hashes the raw request body as received, before
  pydantic touches it; the validated model is never re-serialized, so
  coercion can never contaminate the input hash.
- Ingestion is a stub until the state layer is wired in: validate,
  hash, acknowledge with 202, store nothing. Structurally invalid
  payloads get FastAPI's default 422; format failures surface as
  MalformedEventError from the conversion and are mapped to the same
  422, so the contract's reject-with-4xx holds for both.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException

from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.config import ServiceConfig
from risk_scoring.service.events import Event, to_state_event
from risk_scoring.state import MalformedEventError
from risk_scoring.tracking import configure_tracking, tracking_uri


def resolve_git_sha(repo_root: Path) -> str | None:
    """HEAD commit SHA of the repo at ``repo_root``, or None if unresolvable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def create_app(config: ServiceConfig, repo_root: Path) -> FastAPI:
    """Build the service app; the lifespan loads the pinned model at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_tracking(repo_root)
        model_uri = f"models:/{config.model_name}/{config.model_version}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except MlflowException as exc:
            raise RuntimeError(
                f"model {config.model_name!r} version {config.model_version} is not in "
                f"the registry at {tracking_uri(repo_root)}; the service refuses to "
                f"start without its pinned version"
            ) from exc
        app.state.model = model
        app.state.config = config
        app.state.git_sha = resolve_git_sha(repo_root)
        yield

    app = FastAPI(title="risk-scoring-service", lifespan=lifespan)

    @app.exception_handler(MalformedEventError)
    async def malformed_event(request: Request, exc: Exception) -> JSONResponse:
        """A field that fails its format rule is a 4xx, same as a bad shape."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version(request: Request) -> dict[str, str | int | None]:
        served: ServiceConfig = request.app.state.config
        git_sha: str | None = request.app.state.git_sha
        return {
            "model_name": served.model_name,
            "model_version": served.model_version,
            "feature_version": FEATURE_VERSION,
            "cohort_version": COHORT_VERSION,
            "git_sha": git_sha,
        }

    @app.post("/events", status_code=202)
    async def ingest(event: Event, request: Request) -> dict[str, str]:
        raw_event = json.loads(await request.body())
        to_state_event(event)
        return {
            "status": "accepted",
            "event_type": event.event_type,
            "input_hash": payload_hash(raw_event),
        }

    return app
