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
  422, a bad field format or a reversed interval is the same 422 raised
  from the state layer, a discharge arriving before its patient's
  demographics is a 422 naming the patient, an event contradicting one
  already stored is a 409 naming the key and the differing columns,
  never the stored values, and a body over ``MAX_EVENT_BYTES`` is a 413.
  None of them is ever a silent drop.
- ``POST /events`` requires the bearer token from
  ``RISK_SCORING_API_TOKEN``. The check is a route dependency, so it
  runs before the body is validated and an unauthenticated caller
  learns nothing about the event schema. Startup reads the token the
  way it loads the model: missing means the service does not start.
- The size cap is enforced in ASGI middleware rather than in the handler,
  because FastAPI buffers the whole body to build the ``Event`` parameter
  before the handler runs. The middleware refuses on the declared
  Content-Length without reading a byte, and counts a chunked body as it
  streams in so the refusal lands at the chunk that crosses the limit.
- The pool's size comes from the config and is passed explicitly, since
  ``psycopg_pool`` otherwise caps it at ``min_size``. A request that waits
  out the pool gets a 503 with a JSON body rather than an unhandled 500.
  Posting the event again is always the right response: an event that
  was stored is a no-op on re-post, and a discharge whose score was not
  written is scored then.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException
from psycopg_pool import ConnectionPool, PoolTimeout
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.db import database_url
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.payload_hash import payload_hash
from risk_scoring.service.auth import ENV_API_TOKEN, authorized, require_api_token
from risk_scoring.service.config import ServiceConfig
from risk_scoring.service.events import Event, to_state_event
from risk_scoring.service.ingest import IngestResult, ingest_event
from risk_scoring.serving import UnknownPatientError
from risk_scoring.state import EventConflictError, MalformedEventError
from risk_scoring.tracking import configure_tracking, tracking_uri

POOL_STARTUP_TIMEOUT_SECONDS = 10.0
ENV_GIT_SHA = "RISK_SCORING_GIT_SHA"

# An event document is a few hundred bytes, so 1 MiB is thousands of times
# the largest legitimate body and still small enough that buffering it, and
# the parsed copies json and pydantic make of it, costs nothing. The cap
# bounds one request's body: a declared Content-Length over it is refused
# unread, and a chunked body is refused at the chunk that crosses it, so at
# most one chunk past the limit is ever held. It does not bound how many
# requests are in flight at once, and it does not replace a proxy limit if
# the service is ever bound to anything but loopback.
MAX_EVENT_BYTES = 1 * 1024 * 1024
EVENTS_PATH = "/events"


def _too_large_detail(limit: int) -> str:
    return f"event body exceeds {limit} bytes"


class EventBodyLimit:
    """ASGI middleware refusing a ``POST /events`` body over the limit with 413.

    Everything else passes through untouched. The refusal body is the same
    ``{"detail": ...}`` shape as every other 4xx the service returns.
    """

    def __init__(self, app: ASGIApp, limit: int = MAX_EVENT_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != EVENTS_PATH:
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.limit:
            response = JSONResponse(
                status_code=413, content={"detail": _too_large_detail(self.limit)}
            )
            await response(scope, receive, send)
            return

        received = 0

        async def bounded_receive() -> Message:
            # FastAPI reads the body through this before validating it. It
            # re-raises an HTTPException from the read as-is, and the app's
            # exception middleware turns it into the 413 response.
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise HTTPException(status_code=413, detail=_too_large_detail(self.limit))
            return message

        await self.app(scope, bounded_receive, send)


def resolve_git_sha(repo_root: Path) -> str | None:
    """HEAD commit SHA: the build's stamp if it has one, else the working tree.

    The image carries no ``.git``, so the build stamps the SHA into
    ``RISK_SCORING_GIT_SHA`` and ``/version`` stays honest in a container.
    An unset build argument arrives as an empty string, which is not a SHA,
    so it falls through to the working tree the same as no variable at all.
    The fallback runs git by absolute path, looked up once, so the
    subprocess never depends on PATH at the moment it starts; no git on
    the machine reads as no SHA.
    """
    stamped = os.environ.get(ENV_GIT_SHA, "").strip()
    if stamped:
        return stamped
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(
            [os.path.abspath(git), "rev-parse", "HEAD"],
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


def _open_pool(dsn: str, max_size: int) -> ConnectionPool[Any]:
    # max_size is explicit because psycopg_pool sets it equal to min_size
    # when omitted, which would cap the service at one connection.
    pool: ConnectionPool[Any] = ConnectionPool(dsn, min_size=1, max_size=max_size, open=False)
    try:
        pool.open(wait=True, timeout=POOL_STARTUP_TIMEOUT_SECONDS)
    except Exception as exc:
        pool.close()
        raise RuntimeError(
            f"the database is not reachable; the service refuses to start without it ({exc})"
        ) from exc
    return pool


async def require_bearer(request: Request) -> None:
    """Refuse with 401 unless the request carries the token startup read."""
    token: str = request.app.state.api_token
    if not authorized(request.headers.get("authorization"), token):
        raise HTTPException(
            status_code=401,
            detail=f"a bearer token matching {ENV_API_TOKEN} is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(config: ServiceConfig, repo_root: Path, dsn: str | None = None) -> FastAPI:
    """Build the service app; the lifespan loads the model and opens the pool."""
    dsn = database_url() if dsn is None else dsn

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        api_token = require_api_token()
        model = _load_model(config, repo_root)
        pool = _open_pool(dsn, config.pool_size)
        app.state.api_token = api_token
        app.state.model = model
        app.state.pool = pool
        app.state.config = config
        app.state.git_sha = resolve_git_sha(repo_root)
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(title="risk-scoring-service", lifespan=lifespan)
    app.add_middleware(EventBodyLimit)

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
        """Two contradicting versions of one event is a conflict, not a merge.

        The body is built from the exception's fields, so what reaches the
        poster is exactly the key it supplied and the names of the columns
        that differ. The stored and posted values are in the server log,
        written where the conflict was detected.
        """
        assert isinstance(exc, EventConflictError)
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    f"{exc.table} key {exc.key} already ingested with different values for "
                    f"{', '.join(exc.columns)}"
                ),
                "key": exc.key,
                "columns": list(exc.columns),
            },
        )

    @app.exception_handler(PoolTimeout)
    async def pool_exhausted(request: Request, exc: Exception) -> JSONResponse:
        """Every pooled connection stayed busy for the whole wait: retry later."""
        return JSONResponse(
            status_code=503,
            content={"detail": f"no database connection became free in time ({exc})"},
        )

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

    @app.post(EVENTS_PATH, status_code=202, dependencies=[Depends(require_bearer)])
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
    """Run the blocking half of ingestion, borrowing pooled connections as needed."""
    pool: ConnectionPool[Any] = app_state.pool
    return ingest_event(pool.connection, app_state.model, config, event, input_hash)
