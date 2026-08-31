"""Tests for the HTTP client that posts an event stream to a running service.

A real server on a loopback port backs these, rather than a mock, because
the properties under test are HTTP properties: which status codes are
accepted, what a refusal says, and that one connection carries the whole
stream instead of one handshake per event.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from risk_scoring.service_client import ServiceClient

Responder = Callable[[str, str], tuple[int, dict[str, Any]]]

# A responder returns this status to close the connection without
# answering, which is what the client sees when the service it kept a
# connection open to has gone away.
_DROP = -1


class _Counter:
    def __init__(self) -> None:
        self.connections = 0
        self.requests: list[tuple[str, str]] = []


def _serve(responder: Responder, counter: _Counter) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # The real service speaks HTTP/1.1, so the connection is kept open
        # between requests. The 1.0 default would close after every reply
        # and make the reuse assertion below vacuously unreachable.
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            counter.connections += 1
            super().setup()

        def _reply(self, method: str) -> None:
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length).decode() if length else ""
            counter.requests.append((self.path, body))
            status, payload = responder(self.path, body)
            if status == _DROP:
                self.close_connection = True
                return
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._reply("GET")

        def do_POST(self) -> None:
            self._reply("POST")

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the handler's stderr logging."""

    return Handler


@pytest.fixture()
def server() -> Iterator[Callable[[Responder], tuple[int, _Counter]]]:
    running: list[ThreadingHTTPServer] = []

    def start(responder: Responder) -> tuple[int, _Counter]:
        counter = _Counter()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _serve(responder, counter))
        running.append(httpd)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd.server_address[1], counter

    yield start
    for httpd in running:
        httpd.shutdown()
        httpd.server_close()


def _accepts(path: str, body: str) -> tuple[int, dict[str, Any]]:
    if path == "/health":
        return 200, {"status": "ok"}
    if path == "/version":
        return 200, {"model_name": "readmission-risk", "model_version": 3}
    return 202, {"status": "accepted", "scored": False, "input_hash": "abc"}


def _event(kind: str = "patient") -> dict[str, Any]:
    return {"event_type": kind, "payload": {"Id": "p1"}}


def test_a_posted_event_returns_the_parsed_acknowledgement(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    port, _ = server(_accepts)
    with ServiceClient(port=port) as client:
        assert client.post_event(_event())["input_hash"] == "abc"


def test_the_whole_stream_posts_in_order(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    port, counter = server(_accepts)
    events = [_event("patient"), _event("encounter"), _event("condition")]
    with ServiceClient(port=port) as client:
        assert len(client.post_events(events)) == 3
    assert [json.loads(body)["event_type"] for _, body in counter.requests] == [
        "patient",
        "encounter",
        "condition",
    ]


def test_one_connection_serves_the_whole_stream(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    """A handshake per event would dominate a sixty-thousand-event run."""
    port, counter = server(_accepts)
    with ServiceClient(port=port) as client:
        client.post_events([_event() for _ in range(10)])
    assert counter.connections == 1


def test_a_non_202_raises_naming_the_event_type_and_the_body(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    def refuses(path: str, body: str) -> tuple[int, dict[str, Any]]:
        return 409, {"detail": "encounter e1 contradicts a stored event"}

    port, _ = server(refuses)
    with ServiceClient(port=port) as client, pytest.raises(RuntimeError) as raised:
        client.post_event(_event("encounter"))
    message = str(raised.value)
    assert "encounter" in message
    assert "409" in message
    assert "contradicts" in message


def test_version_returns_the_reported_provenance(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    port, _ = server(_accepts)
    with ServiceClient(port=port) as client:
        assert client.version()["model_version"] == 3


def test_wait_for_health_returns_once_the_service_answers_200(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    port, _ = server(_accepts)
    with ServiceClient(port=port) as client:
        client.wait_for_health(timeout=5.0)


def test_wait_for_health_times_out_carrying_the_last_failure(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    def unhealthy(path: str, body: str) -> tuple[int, dict[str, Any]]:
        return 503, {"detail": "still loading"}

    port, _ = server(unhealthy)
    with ServiceClient(port=port) as client, pytest.raises(TimeoutError) as raised:
        client.wait_for_health(timeout=0.5)
    assert "503" in str(raised.value)


def test_posting_no_events_opens_no_connection(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    port, counter = server(_accepts)
    with ServiceClient(port=port) as client:
        assert client.post_events([]) == []
    assert counter.connections == 0


def test_a_2xx_that_is_not_202_is_still_refused(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    """The service acknowledges an accepted event with 202 and nothing else."""

    def surprising(path: str, body: str) -> tuple[int, dict[str, Any]]:
        return 200, {"status": "accepted"}

    port, _ = server(surprising)
    with ServiceClient(port=port) as client, pytest.raises(RuntimeError) as raised:
        client.post_event(_event("encounter"))
    assert "200" in str(raised.value)


class _Drops:
    """A responder that closes the connection on the requests named."""

    def __init__(self, *, on: set[int]) -> None:
        self.on = on
        self.seen = 0

    def __call__(self, path: str, body: str) -> tuple[int, dict[str, Any]]:
        self.seen += 1
        if self.seen in self.on:
            return _DROP, {}
        return _accepts(path, body)


def test_a_dropped_kept_alive_connection_is_reopened(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    """Restarting the service leaves the client holding a dead socket.

    The container check posts half a stream, restarts the service, waits
    for it to report healthy, and posts the rest. The connection carrying
    the stream does not survive that restart, and the service being back
    is what makes reopening it the right answer rather than a guess.
    """
    port, counter = server(_Drops(on={2}))
    with ServiceClient(port=port) as client:
        acknowledgements = client.post_events([_event() for _ in range(3)])
    assert len(acknowledgements) == 3
    assert counter.connections == 2


def test_the_retry_is_bounded_at_one_reconnect(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    """A service that is genuinely gone must fail, not be retried forever."""
    port, counter = server(_Drops(on={2, 3}))
    with ServiceClient(port=port) as client:
        client.post_event(_event())
        with pytest.raises(ConnectionError):
            client.post_event(_event())
    assert counter.connections == 2


def test_a_drop_on_a_fresh_connection_is_not_retried(
    server: Callable[[Responder], tuple[int, _Counter]],
) -> None:
    """Only a connection the server already answered on is worth reopening."""
    port, counter = server(_Drops(on={1}))
    with ServiceClient(port=port) as client, pytest.raises(ConnectionError):
        client.post_event(_event())
    assert counter.connections == 1
