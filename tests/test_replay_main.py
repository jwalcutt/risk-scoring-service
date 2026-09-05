"""The replay commands, with no database.

The rules these tests pin:

- The four commands parse, and pacing comes from ``--max-speed`` alone;
  the config file and its overrides never carry it.
- The first Ctrl-C asks the loop to pause after its tick and hands the
  next Ctrl-C back to Python, so a second press quits at once. The
  previous handler is back in place when the guard exits.
- ``start`` refuses a population with no export before touching anything.
- The status text names what an operator needs: the run, its span, where
  the clock stands, and the status.
"""

from __future__ import annotations

import signal
from datetime import UTC, datetime
from pathlib import Path

import pytest

import risk_scoring.replay.__main__ as replay_main
from risk_scoring.replay import clock
from risk_scoring.replay.runs import ReplayRun

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)


def _write_config(root: Path, population: str = "baseline") -> None:
    (root / "configs").mkdir()
    (root / "configs" / "replay.toml").write_text(
        f'[replay]\npopulation = "{population}"\nstart = 2025-01-01\nend = 2026-01-01\n'
        "acceleration = 4\n"
    )


# Parsing


def test_start_and_resume_take_max_speed_and_the_others_do_not() -> None:
    parser = replay_main.build_parser()
    assert parser.parse_args(["start", "--max-speed"]).max_speed is True
    assert parser.parse_args(["resume", "--max-speed"]).max_speed is True
    assert parser.parse_args(["start"]).max_speed is False
    for command in ("pause", "status"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--max-speed"])


def test_pacing_comes_from_the_flag_and_the_configured_acceleration() -> None:
    parser = replay_main.build_parser()
    paced = replay_main.pacing_for(4.0, parser.parse_args(["start"]))
    assert paced == clock.Pacing(acceleration=4.0, max_speed=False)
    flat_out = replay_main.pacing_for(4.0, parser.parse_args(["resume", "--max-speed"]))
    assert flat_out == clock.Pacing(acceleration=4.0, max_speed=True)


def test_start_accepts_the_config_overrides_and_a_data_root() -> None:
    args = replay_main.build_parser().parse_args(
        ["start", "--population", "care_protocol", "--end", "2025-03-01", "--data-root", "/x"]
    )
    assert args.population == "care_protocol"
    assert args.end.isoformat() == "2025-03-01"
    assert args.data_root == Path("/x")
    assert replay_main.build_parser().parse_args(["start"]).data_root == Path("data")


def test_every_command_needs_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        replay_main.build_parser().parse_args([])


# Ctrl-C


def test_the_first_ctrl_c_requests_a_pause_and_the_second_raises() -> None:
    announced: list[str] = []
    original = signal.getsignal(signal.SIGINT)
    guard = replay_main.InterruptGuard(announce=announced.append)

    with guard:
        assert guard.pause_requested(START) is False
        signal.raise_signal(signal.SIGINT)
        assert guard.pause_requested(START) is True
        assert announced == ["pausing after this tick; press Ctrl-C again to quit now"]
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)

    assert signal.getsignal(signal.SIGINT) is original


def test_the_guard_restores_the_handler_when_no_ctrl_c_came() -> None:
    original = signal.getsignal(signal.SIGINT)
    with replay_main.InterruptGuard(announce=lambda text: None) as guard:
        assert signal.getsignal(signal.SIGINT) is not original
        assert guard.pause_requested(START) is False
    assert signal.getsignal(signal.SIGINT) is original


# Refusals before any side effect


def test_start_without_an_export_fails_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, population="nowhere")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RISK_SCORING_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")

    with pytest.raises(SystemExit, match=r"no CSV export at .*nowhere"):
        replay_main.main(["start"], notifier=lambda text: None)


def test_start_without_a_config_file_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        replay_main.main(["start"], notifier=lambda text: None)


# The status text


def test_describe_names_the_run_its_span_its_clock_and_its_status() -> None:
    run = ReplayRun(
        run_id=3,
        population="baseline",
        start_at=START,
        end_at=END,
        acceleration=4.0,
        sim_now=datetime(2025, 4, 2, 6, tzinfo=UTC),
        status="paused",
        cursor=("2025-04-02T05:12:00Z", 3, "row"),
        created_at=datetime(2026, 9, 5, 19, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 9, 5, 20, 11, 41, tzinfo=UTC),
    )
    text = replay_main.describe(run)
    assert "run 3" in text
    assert "baseline" in text
    assert "paused" in text
    assert "2025-01-01T00:00:00Z to 2026-01-01T00:00:00Z" in text
    assert "2025-04-02T06:00:00Z" in text
    assert "25.0%" in text
    assert "2025-04-02T05:12:00Z" in text
    assert "last written 2026-09-05 20:11:41 UTC" in text


def test_describe_says_when_nothing_has_been_posted_yet() -> None:
    run = ReplayRun(
        run_id=1,
        population="baseline",
        start_at=START,
        end_at=END,
        acceleration=4.0,
        sim_now=START,
        status="running",
        cursor=None,
        created_at=START,
        updated_at=START,
    )
    text = replay_main.describe(run)
    assert "0.0%" in text
    assert "nothing posted yet" in text
