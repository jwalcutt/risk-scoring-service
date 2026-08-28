"""Posting an event stream to a running scoring service over HTTP.

The service takes events one at a time, so a batch of sixty thousand is
sixty thousand requests. Judgment calls this module fixes:

- One kept-alive connection carries the whole stream. A handshake per
  event would dominate the run.
- Only 202 is success. The service answers 4xx for a malformed, out of
  order, or contradicting event, and every one of those is a defect in
  the caller's stream rather than something to count and continue past,
  so a refusal raises carrying the status and the service's own message.
- The acknowledgement is returned, not discarded. It carries the input
  hash and the score, which lets a caller cross-check the durable log
  against what the service said at the time.
- Requests go through the standard library rather than an HTTP client
  library, because the only one this project depends on is a test-time
  dependency and this module ships.
"""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any

DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8001
HEALTH_TIMEOUT_SECONDS = 180.0

_ACCEPTED = 202
_OK = 200


class ServiceClient:
    """One connection to a running service, reused across a whole stream."""

    def __init__(
        self,
        *,
        port: int = DEFAULT_SERVICE_PORT,
        host: str = DEFAULT_SERVICE_HOST,
        timeout: float = 60.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _open(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
        return self._connection

    def _request(self, method: str, path: str, body: str | None = None) -> tuple[int, bytes]:
        connection = self._open()
        headers = {"content-type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()

    def wait_for_health(self, timeout: float = HEALTH_TIMEOUT_SECONDS) -> None:
        """Block until the service reports ready, or give up saying why."""
        deadline = time.monotonic() + timeout
        last = "no attempt made"
        while time.monotonic() < deadline:
            probe = http.client.HTTPConnection(self.host, self.port, timeout=5)
            try:
                probe.request("GET", "/health")
                response = probe.getresponse()
                response.read()
                if response.status == _OK:
                    return
                last = f"status {response.status}"
            except OSError as error:
                last = str(error)
            finally:
                probe.close()
            time.sleep(0.1)
        raise TimeoutError(f"the service never became healthy on port {self.port} ({last})")

    def version(self) -> dict[str, Any]:
        """The service's reported model, feature, cohort, and commit versions."""
        status, body = self._request("GET", "/version")
        if status != _OK:
            raise RuntimeError(f"/version answered {status}: {body!r}")
        parsed: dict[str, Any] = json.loads(body)
        return parsed

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Post one event, returning the service's acknowledgement."""
        status, body = self._request("POST", "/events", json.dumps(event))
        if status != _ACCEPTED:
            raise RuntimeError(f"{event.get('event_type')} refused: {status} {body.decode()!r}")
        parsed: dict[str, Any] = json.loads(body)
        return parsed

    def post_events(self, events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Post a whole stream in order, returning every acknowledgement."""
        return [self.post_event(event) for event in events]
