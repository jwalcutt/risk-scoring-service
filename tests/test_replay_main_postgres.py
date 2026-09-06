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
- ``start`` with a splice preloads every population's history before its
  own instant and posts the incoming population from the splice on;
  ``resume`` rebuilds the same spliced stream from the config it is given.
  A splice naming a missing export is refused before anything is loaded,
  and a splice outside the row's span is refused on resume.
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
    SPLICE_AT,
    SPLICE_POPULATION,
    START,
    ClientPoster,
    KilledMidTick,
    Serve,
    Source,
    prepare,
    prepare_spliced,
    read_outputs,
    schedule_of,
    serving,
    skew_frames,
    splice_frames,
    stream_of,
    tick_containing,
)
from risk_scoring import train
from risk_scoring.populations import rekeyed
from risk_scoring.replay import clock, harness, preload, runs
from risk_scoring.replay.config import ReplayConfig, Splice
from risk_scoring.replay.splice import (
    segment_events,
    segment_labels,
    segments,
    spliced_events,
    spliced_labels,
)
from risk_scoring.stream import StreamEvent, envelope

pytestmark = pytest.mark.db

CONFIG = '[replay]\npopulation = "skew"\nstart = 2024-04-01\nend = 2024-08-07\nacceleration = 4\n'
SPLICED = (
    CONFIG
    + f'\n[[splice]]\nat = {SPLICE_AT.date().isoformat()}\npopulation = "{SPLICE_POPULATION}"\n'
)
MISSING = CONFIG + f'\n[[splice]]\nat = {SPLICE_AT.date().isoformat()}\npopulation = "ghost"\n'
# Valid on its own terms, so the refusal comes from the row's span, not the file's.
STRANDED = (
    CONFIG.replace("end = 2024-08-07", "end = 2025-01-01")
    + f'\n[[splice]]\nat = 2024-09-01\npopulation = "{SPLICE_POPULATION}"\n'
)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repo root with the configs and the exports where the commands look."""
    root = tmp_path_factory.mktemp("replay-repo")
    (root / "configs").mkdir()
    (root / "configs" / "replay.toml").write_text(CONFIG)
    (root / "configs" / "spliced.toml").write_text(SPLICED)
    (root / "configs" / "missing.toml").write_text(MISSING)
    (root / "configs" / "stranded.toml").write_text(STRANDED)
    skew_frames(root / "data" / "skew" / "csv")
    splice_frames(root / "data" / SPLICE_POPULATION / "csv")
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
) -> dict[str, list[dict[str, Any]]]:
    """An uninterrupted run through the loop itself, the reference every command run meets."""
    prepare(dsn, frames, events)
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        harness.run_replay(
            conn, run, events, ClientPoster(client), labels=schedule_of(frames), pacing=MAX_SPEED
        )
    return read_outputs(dsn)


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
    assert read_outputs(operator.dsn) == reference
    assert operator.posted() == _expected(replayed)
    assert operator.notifications == [f"finished at {clock.instant(END)}"]
    assert "history loaded" in out
    assert "labels to release" in out
    assert "finished" in out
    assert "6 discharges scored" in out
    assert "5 labels released, 1 pending" in out


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
    assert read_outputs(operator.dsn) == reference
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
    assert read_outputs(operator.dsn) == reference
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
    assert read_outputs(operator.dsn) == reference


# Splicing


@pytest.fixture(scope="module")
def spliced_sources(repo: Path, frames: dict[str, pd.DataFrame]) -> list[Source]:
    incoming = rekeyed(splice_frames(repo / "data" / SPLICE_POPULATION / "csv"), SPLICE_POPULATION)
    return [(frames, stream_of(frames), START), (incoming, stream_of(incoming), SPLICE_AT)]


@pytest.fixture(scope="module")
def spliced(spliced_sources: list[Source]) -> tuple[list[StreamEvent], list[Any]]:
    """The spliced stream and schedule the commands must reproduce."""
    config = ReplayConfig(
        population="skew",
        start=START.date(),
        end=END.date(),
        acceleration=4,
        splices=(Splice(at=SPLICE_AT.date(), population=SPLICE_POPULATION),),
    )
    plan = segments(config)
    events = spliced_events(
        [segment_events(s, stream) for s, (_, stream, _) in zip(plan, spliced_sources, strict=True)]
    )
    labels = spliced_labels(
        [
            segment_labels(s, schedule_of(frames))
            for s, (frames, _, _) in zip(plan, spliced_sources, strict=True)
        ]
    )
    return events, labels


def _straight_spliced(
    serve: Serve, dsn: str, sources: list[Source], spliced: tuple[list[StreamEvent], list[Any]]
) -> dict[str, list[dict[str, Any]]]:
    events, labels = spliced
    prepare_spliced(dsn, sources)
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        harness.run_replay(conn, run, events, ClientPoster(client), labels=labels, pacing=MAX_SPEED)
    outputs = read_outputs(dsn)
    assert len(outputs["predictions"]) == 8 and len(outputs["labels"]) == 7
    return outputs


def test_start_with_a_splice_preloads_both_populations_and_posts_the_incoming_after_it(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    spliced_sources: list[Source],
    spliced: tuple[list[StreamEvent], list[Any]],
) -> None:
    reference = _straight_spliced(serve, db_url_factory(), spliced_sources, spliced)
    operator = operate(db_url_factory())

    out = operator.run("start", "--config", "configs/spliced.toml")

    events, _ = spliced
    assert operator.row().status == "finished"
    assert read_outputs(operator.dsn) == reference
    assert operator.posted() == _expected(events)
    assert f"history from skew before {clock.instant(START)}" in out
    assert f"history from {SPLICE_POPULATION} before {clock.instant(SPLICE_AT)}" in out
    before = sum(event.at < clock.instant(SPLICE_AT) for event in events)
    assert f"{before} events from skew" in out
    assert f"{len(events) - before} events from {SPLICE_POPULATION}" in out
    assert "8 discharges scored" in out
    assert "7 labels released, 1 pending" in out


def test_resume_rebuilds_the_spliced_stream_from_the_config_it_is_given(
    operate: Callable[[str], Operator],
    db_url_factory: Callable[[], str],
    serve: Serve,
    spliced_sources: list[Source],
    spliced: tuple[list[StreamEvent], list[Any]],
) -> None:
    reference = _straight_spliced(serve, db_url_factory(), spliced_sources, spliced)
    operator = operate(db_url_factory())
    events, _ = spliced
    pause_after = sum(event.at < clock.instant(SPLICE_AT) for event in events) - 1

    class PausesFromOutside(ClientPoster):
        def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
            ack = super().post_event(event)
            if len(self.posted) == pause_after:
                with psycopg.connect(operator.dsn, connect_timeout=2) as conn:
                    run = runs.open_run(conn)
                    assert run is not None
                    runs.set_status(conn, run.run_id, "paused")
            return ack

    operator.make_poster = PausesFromOutside
    operator.run("start", "--config", "configs/spliced.toml")
    assert operator.row().status == "paused"
    assert operator.row().sim_now < SPLICE_AT

    operator.make_poster = ClientPoster
    operator.run("resume", "--config", "configs/spliced.toml")

    assert operator.row().status == "finished"
    assert read_outputs(operator.dsn) == reference
    assert operator.posted() == _expected(events)


def test_start_with_a_splice_naming_a_missing_export_is_refused_before_loading(
    operate: Callable[[str], Operator], db_url: str
) -> None:
    operator = operate(db_url)
    with pytest.raises(SystemExit, match="no CSV export"):
        operator.run("start", "--config", "configs/missing.toml")

    with psycopg.connect(db_url, connect_timeout=2) as conn:
        assert conn.execute("SELECT count(*) FROM patients").fetchone() == (0,)
        assert runs.open_run(conn) is None
    assert operator.posters == []


def test_resume_refuses_a_splice_outside_the_rows_span(
    operate: Callable[[str], Operator], db_url: str
) -> None:
    operator = operate(db_url)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        runs.create_run(conn, population="skew", start_at=START, end_at=END, acceleration=4)

    with pytest.raises(SystemExit, match="strictly inside"):
        operator.run("resume", "--config", "configs/stranded.toml")

    assert operator.row().status == "running"
    assert operator.posters == []
