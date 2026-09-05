"""The replay commands against a real database and an in-process service.

Each command is run through ``main`` exactly as an operator would type it,
from a throwaway repo root holding the config file and a synthetic export,
with the service and the notifier injected. The rules these tests pin:

- ``start`` preloads history, opens the run row, runs to the end, marks
  the row finished, and notifies once.
- ``start`` refuses while a run is open, before loading anything.
- ``pause`` marks the row; the harness honors it at its next tick, and
  ``pause`` on a paused run says so. ``pause`` and ``resume`` with no run
  say so and exit non-zero.
- Ctrl-C finishes the tick, checkpoints, marks the row paused, and
  notifies; ``resume`` with no other argument completes the run to a log
  identical to an uninterrupted one.
- A run whose process died leaves ``running`` behind; ``resume`` takes it
  from the checkpoint all the same.
- ``status`` prints the row.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

import risk_scoring.replay.__main__ as replay_main
from replay_support import (
    END,
    MAX_SPEED,
    START,
    ClientPoster,
    KilledMidTick,
    Serve,
    prepare,
    read_log,
    serving,
    skew_frames,
    stream_of,
    tick_containing,
)
from risk_scoring import train
from risk_scoring.replay import clock, harness, preload, runs
from risk_scoring.stream import StreamEvent, envelope

pytestmark = pytest.mark.db

CONFIG = '[replay]\npopulation = "skew"\nstart = 2024-04-01\nend = 2024-08-07\nacceleration = 4\n'


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repo root with the config and the skew export where the commands look."""
    root = tmp_path_factory.mktemp("replay-repo")
    (root / "configs").mkdir()
    (root / "configs" / "replay.toml").write_text(CONFIG)
    skew_frames(root / "data" / "skew" / "csv")
    return root


@pytest.fixture(scope="module")
def frames(repo: Path) -> dict[str, pd.DataFrame]:
    return skew_frames(repo / "data" / "skew" / "csv")


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def replayed(events: list[StreamEvent]) -> list[StreamEvent]:
    return preload.replay_from(events, clock.instant(START))


@pytest.fixture()
def serve(trained_repo: tuple[Any, train.TrainingResult]) -> Serve:
    return serving(trained_repo)


class Operator:
    """Runs the commands as typed, against one database, recording what they said."""

    def __init__(
        self,
        repo: Path,
        dsn: str,
        serve: Serve,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.dsn = dsn
        self.serve = serve
        self.capsys = capsys
        self.notifications: list[str] = []
        self.posters: list[ClientPoster] = []
        self.make_poster: Callable[[TestClient], ClientPoster] = ClientPoster
        monkeypatch.chdir(repo)
        monkeypatch.setenv("RISK_SCORING_DATABASE_URL", dsn)

    @contextmanager
    def _poster(self, port: int) -> Iterator[ClientPoster]:
        with self.serve(self.dsn) as client:
            poster = self.make_poster(client)
            self.posters.append(poster)
            yield poster

    def run(self, *argv: str) -> str:
        replay_main.main(
            [*argv, "--max-speed"] if argv[0] in ("start", "resume") else list(argv),
            poster_factory=self._poster,
            notifier=self.notifications.append,
        )
        return self.capsys.readouterr().out

    def row(self) -> runs.ReplayRun:
        with psycopg.connect(self.dsn, connect_timeout=2) as conn:
            row = conn.execute("SELECT run_id FROM replay_runs ORDER BY run_id DESC").fetchone()
            assert row is not None
            return runs.read_run(conn, row[0])

    def posted(self) -> list[Mapping[str, Any]]:
        return [event for poster in self.posters for event in poster.posted]


@pytest.fixture()
def operate(
    repo: Path, serve: Serve, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[[str], Operator]:
    return lambda dsn: Operator(repo, dsn, serve, monkeypatch, capsys)


def _expected(replayed: list[StreamEvent]) -> list[dict[str, Any]]:
    return [envelope(event.kind, event.row) for event in replayed]


def _straight_log(
    serve: Serve, dsn: str, frames: dict[str, pd.DataFrame], events: list[StreamEvent]
) -> list[dict[str, Any]]:
    """An uninterrupted run through the loop itself, the reference every command run meets."""
    prepare(dsn, frames, events)
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        harness.run_replay(conn, run, events, ClientPoster(client), pacing=MAX_SPEED)
    return read_log(dsn)


# start


def test_start_preloads_opens_the_run_and_runs_it_to_the_end(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    reference = _straight_log(serve, db_url_factory(), frames, events)
    operator = operate(db_url_factory())

    out = operator.run("start")

    row = operator.row()
    assert row.status == "finished"
    assert row.sim_now == END
    assert row.cursor == replayed[-1].sort_key
    assert read_log(operator.dsn) == reference
    assert operator.posted() == _expected(replayed)
    assert operator.notifications == [f"finished at {clock.instant(END)}"]
    assert "history loaded" in out
    assert "finished" in out
    assert "6 discharges scored" in out


def test_start_refuses_while_a_run_is_open_before_loading_anything(
    operate: Callable[[str], Operator], db_url: str
) -> None:
    operator = operate(db_url)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        runs.create_run(conn, population="skew", start_at=START, end_at=END, acceleration=4)

    with pytest.raises(SystemExit, match="not finished already exists"):
        operator.run("start")

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        assert conn.execute("SELECT count(*) FROM patients").fetchone() == (0,)
    assert operator.posters == []


# pause and status


def test_pause_marks_the_row_and_the_harness_stops_at_its_next_tick(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    """The pause command runs from another terminal; here, from inside a post."""
    reference = _straight_log(serve, db_url_factory(), frames, events)
    operator = operate(db_url_factory())
    pause_after = len(replayed) // 2
    paused_tick = tick_containing(replayed[pause_after - 1].at)

    class PausesFromOutside(ClientPoster):
        def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
            ack = super().post_event(event)
            if len(self.posted) == pause_after:
                # A second connection, as the other terminal would open.
                with psycopg.connect(operator.dsn, connect_timeout=2) as conn:
                    run = runs.open_run(conn)
                    assert run is not None
                    runs.set_status(conn, run.run_id, "paused")
            return ack

    operator.make_poster = PausesFromOutside
    out = operator.run("start")

    row = operator.row()
    assert row.status == "paused"
    assert row.sim_now == paused_tick
    assert operator.notifications == [f"paused at {clock.instant(paused_tick)}"]
    assert f"paused at {clock.instant(paused_tick)}" in out

    assert "already paused" in operator.run("pause")

    operator.make_poster = ClientPoster
    out = operator.run("resume")
    assert operator.row().status == "finished"
    assert read_log(operator.dsn) == reference
    assert operator.posted() == _expected(replayed)
    assert "finished" in out


def test_pause_and_resume_with_no_run_say_so(
    operate: Callable[[str], Operator], db_url: str
) -> None:
    operator = operate(db_url)
    with pytest.raises(SystemExit, match="no unfinished replay run"):
        operator.run("pause")
    with pytest.raises(SystemExit, match="no unfinished replay run"):
        operator.run("resume")


def test_pause_marks_a_run_no_harness_is_driving(
    operate: Callable[[str], Operator], db_url: str
) -> None:
    operator = operate(db_url)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        runs.create_run(conn, population="skew", start_at=START, end_at=END, acceleration=4)

    out = operator.run("pause")

    assert operator.row().status == "paused"
    assert "will pause at its next tick" in out


def test_status_prints_the_row(operate: Callable[[str], Operator], db_url: str) -> None:
    operator = operate(db_url)
    assert "no unfinished replay run" in operator.run("status")

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        runs.create_run(conn, population="skew", start_at=START, end_at=END, acceleration=4)
    out = operator.run("status")
    assert "skew" in out
    assert f"{clock.instant(START)} to {clock.instant(END)}" in out
    assert "running" in out


# Ctrl-C, and a process that died


def test_ctrl_c_finishes_the_tick_marks_the_row_paused_and_resume_completes_it(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    reference = _straight_log(serve, db_url_factory(), frames, events)
    operator = operate(db_url_factory())
    interrupt_after = len(replayed) // 3
    interrupted_tick = tick_containing(replayed[interrupt_after - 1].at)
    in_that_tick = [event for event in replayed if event.at <= clock.instant(interrupted_tick)]

    class OperatorPressesCtrlC(ClientPoster):
        def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
            ack = super().post_event(event)
            if len(self.posted) == interrupt_after:
                signal.raise_signal(signal.SIGINT)
            return ack

    operator.make_poster = OperatorPressesCtrlC
    out = operator.run("start")

    row = operator.row()
    assert row.status == "paused"
    assert row.sim_now == interrupted_tick
    assert row.cursor == in_that_tick[-1].sort_key
    assert operator.posters[0].posted == _expected(in_that_tick)
    assert len(in_that_tick) < len(replayed)
    assert operator.notifications == [f"paused at {clock.instant(interrupted_tick)}"]
    assert "pausing after this tick" in out

    operator.make_poster = ClientPoster
    operator.run("resume")
    assert operator.row().status == "finished"
    assert read_log(operator.dsn) == reference
    assert operator.posted() == _expected(replayed)


def test_resume_takes_a_run_whose_process_died_from_its_checkpoint(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    replayed: list[StreamEvent],
) -> None:
    """No pause was ever written; the row still says running, and resume proceeds."""
    reference = _straight_log(serve, db_url_factory(), frames, events)
    operator = operate(db_url_factory())
    die_after = len(replayed) // 2
    operator.make_poster = lambda client: ClientPoster(client, die_after=die_after)

    with pytest.raises(KilledMidTick):
        operator.run("start")
    assert operator.row().status == "running"
    assert operator.notifications == []

    operator.make_poster = ClientPoster
    operator.run("resume")
    assert operator.row().status == "finished"
    assert read_log(operator.dsn) == reference
