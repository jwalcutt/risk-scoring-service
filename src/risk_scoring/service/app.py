"""FastAPI app factory: pinned model loading and the scoring endpoint.

Judgment calls this module fixes:

- No module-level app: :func:`create_app` takes an already-loaded config
  and a repo root, so tests and the container construct the app the same
  way and nothing reads configuration at import time.
- Startup loads the model once, by the explicit registered version the
  config pins, and opens the connection pool. Either failing raises a
  ``RuntimeError`` that names the cause, so a broken pin or an
  unreachable database stops the service from starting instead of
  surfacing as a later 500. A 200 from ``/health`` therefore means the
  service can actually score and store.
- The ingestion handler hashes the raw request body as received, before
  pydantic touches it; the validated model is never re-serialized, so
  coercion can never contaminate the input hash.
- The handler is async only to reach the raw body. Every blocking step
  after that (database, pandas, the model) runs in the threadpool, so
  one slow scoring call cannot stall the event loop.
- Every way an event can be refused is a 4xx: a bad shape is FastAPI's
  422, a bad field format is the same 422 raised from the state layer, a
  discharge arriving before its patient's demographics is a 422 naming
  the patient, and an event contradicting one already stored is a 409.
  None of them is ever a silent drop.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException
from psycopg_pool import ConnectionPool

from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.db import database_url
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.config import ServiceConfig
from risk_scoring.service.events import Event, to_state_event
from risk_scoring.service.ingest import IngestResult, ingest_event
from risk_scoring.serving import UnknownPatientError
from risk_scoring.state import EventConflictError, MalformedEventError
from risk_scoring.tracking import configure_tracking, tracking_uri

POOL_STARTUP_TIMEOUT_SECONDS = 10.0


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


def _load_model(config: ServiceConfig, repo_root: Path) -> Any:
    configure_tracking(repo_root)
    model_uri = f"models:/{config.model_name}/{config.model_version}"
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except MlflowException as exc:
        raise RuntimeError(
            f"model {config.model_name!r} version {config.model_version} is not in "
            f"the registry at {tracking_uri(repo_root)}; the service refuses to "
            f"start without its pinned version"
        ) from exc


def _open_pool(dsn: str) -> ConnectionPool[Any]:
    pool: ConnectionPool[Any] = ConnectionPool(dsn, min_size=1, open=False)
    try:
        pool.open(wait=True, timeout=POOL_STARTUP_TIMEOUT_SECONDS)
    except Exception as exc:
        pool.close()
        raise RuntimeError(
            f"the database is not reachable; the service refuses to start without it ({exc})"
        ) from exc
    return pool


def create_app(config: ServiceConfig, repo_root: Path, dsn: str | None = None) -> FastAPI:
    """Build the service app; the lifespan loads the model and opens the pool."""
    dsn = database_url() if dsn is None else dsn

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = _load_model(config, repo_root)
        pool = _open_pool(dsn)
        app.state.model = model
        app.state.pool = pool
        app.state.config = config
        app.state.git_sha = resolve_git_sha(repo_root)
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(title="risk-scoring-service", lifespan=lifespan)

    @app.exception_handler(MalformedEventError)
    async def malformed_event(request: Request, exc: Exception) -> JSONResponse:
        """A field that fails its format rule is a 4xx, same as a bad shape."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(UnknownPatientError)
    async def unknown_patient(request: Request, exc: Exception) -> JSONResponse:
        """A discharge that outran its patient's demographics: report, never skip."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(EventConflictError)
    async def event_conflict(request: Request, exc: Exception) -> JSONResponse:
        """Two contradicting versions of one event is a conflict, not a merge."""
        return JSONResponse(status_code=409, content={"detail": str(exc)})

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
    async def ingest(event: Event, request: Request) -> dict[str, Any]:
        raw_event = json.loads(await request.body())
        input_hash = payload_hash(raw_event)
        state_event = to_state_event(event)
        result = await run_in_threadpool(
            _store_and_score, request.app.state, config, state_event, input_hash
        )
        return {
            "status": "accepted",
            "event_type": event.event_type,
            "input_hash": input_hash,
            "scored": result.scored,
            "prediction_id": result.prediction_id,
            "score": result.score,
        }

    return app


def _store_and_score(
    app_state: Any, config: ServiceConfig, event: Any, input_hash: str
) -> IngestResult:
    """Run the blocking half of ingestion on a pooled connection."""
    pool: ConnectionPool[Any] = app_state.pool
    with pool.connection() as conn:
        return ingest_event(conn, app_state.model, config, event, input_hash)
