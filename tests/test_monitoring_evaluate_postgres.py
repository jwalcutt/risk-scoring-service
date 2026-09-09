"""Evaluations over a real replay: the same boundary always says the same thing.

The property this file exists to prove is the one the monitoring layer
rests on. An evaluation is a pure function of the tables and the boundary,
so it must not depend on how far past the boundary the harness had run when
the evaluation happened, nor on whether the run was interrupted.

Three arms over one population, in three databases:

1. Replay straight through, then evaluate every boundary over the finished
   tables. This is the reference set.
2. Replay in segments, pausing at each boundary and evaluating there before
   resuming. Every evaluation sees a table holding nothing past its own
   boundary, which is the live case.
3. Replay paused and resumed at an instant unrelated to any boundary, then
   evaluate over the finished tables.

Arm two against arm one is the purity: how much the tables hold beyond the
boundary changes nothing. Arm three against arm one is determinism under
interruption, carrying the byte-identity guarantee from the two tables
through to what is derived from them.

The file also reads the substantive numbers once against live data rather
than only comparing them to each other, since three equal wrong answers
would satisfy the arms above.

One thing this file cannot say anything about, and it matters: the
false-alarm rate. The reference here is the training window of the fixture
model, which was fitted on a different synthetic population from the one
replayed, so the two distributions genuinely differ and almost every
window alerts. That is the statistics working, not the thresholds failing.
Whether the placeholder thresholds are sensible is measured over a clean
replay where the reference and the replayed data come from one frozen
population. What is asserted here instead is that the comparison
discriminates: some signals are flagged and others are not, and the
matched case is pinned in the pure tests, where an identical window scores
zero and raises nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from factories import MONITORING_END, MONITORING_START, write_monitoring_population
from replay_support import (
    MAX_SPEED,
    ClientPoster,
    Serve,
    prepare,
    read_outputs,
    schedule_of,
    serving,
    stream_of,
)
from risk_scoring import train
from risk_scoring.monitoring import signals
from risk_scoring.monitoring.config import DEFAULT_CONFIG_RELPATH, MonitoringConfig, load_config
from risk_scoring.monitoring.evaluate import (
    Evaluation,
    boundaries,
    evaluate,
    judge,
    window_for,
)
from risk_scoring.monitoring.reference import (
    Reference,
    build_reference,
    read_reference,
    record_reference,
)
from risk_scoring.replay import harness, runs
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

POPULATION = "monitoring"

PAUSE_AT = MONITORING_START + timedelta(days=17, hours=5)
"""An instant deliberately off the seven-day grid, so no boundary coincides."""


# --- the population and the reference ---


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    from risk_scoring.populations import load_population

    csv_dir = tmp_path_factory.mktemp("monitoring-population") / "csv"
    write_monitoring_population(csv_dir)
    return dict(load_population(csv_dir))


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return schedule_of(frames)


@pytest.fixture(scope="module")
def built_reference(
    trained_repo: tuple[Path, train.TrainingResult],
) -> Reference:
    """The reference from the real path: the training population of the fixture model."""
    root, trained = trained_repo
    return build_reference(
        root / "data" / "baseline" / "csv", root, model_version=trained.model_version
    )


@pytest.fixture()
def config() -> MonitoringConfig:
    return load_config(Path(DEFAULT_CONFIG_RELPATH))


# --- driving a replay ---


def _run_row(dsn: str) -> runs.ReplayRun:
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        run = runs.latest_run(conn)
        assert run is not None
        return run


def _replay(
    serve: Serve,
    dsn: str,
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    *,
    pause_at: datetime | None = None,
) -> harness.RunSummary:
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        return harness.run_replay(
            conn,
            run,
            events,
            ClientPoster(client),
            labels=schedule,
            pacing=MAX_SPEED,
            pause_requested=(
                (lambda sim_now: pause_at is not None and sim_now >= pause_at)
                if pause_at is not None
                else (lambda sim_now: False)
            ),
        )


def _open(
    dsn: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    reference: Reference,
) -> int:
    """Preload, open the run row, and store the reference; returns its id."""
    prepare(
        dsn,
        frames,
        events,
        start=MONITORING_START,
        end=MONITORING_END,
        population=POPULATION,
    )
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        return record_reference(conn, reference)


def _evaluate_all(
    dsn: str, reference: Reference, reference_id: int, config: MonitoringConfig
) -> list[Evaluation]:
    run = _run_row(dsn)
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        return [
            evaluate(conn, run, reference, reference_id, boundary, config)
            for boundary in boundaries(run.start_at, run.end_at, config)
        ]


@pytest.fixture()
def straight(
    trained_repo: tuple[Path, train.TrainingResult],
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    built_reference: Reference,
) -> tuple[str, int]:
    """A complete replay of the monitoring population; yields its DSN and reference id."""
    serve: Serve = serving(trained_repo)
    reference_id = _open(db_url, frames, events, built_reference)
    summary = _replay(serve, db_url, events, schedule)
    assert summary.finished
    return db_url, reference_id


# --- the substantive reads ---


def test_the_run_produces_twelve_boundaries(
    straight: tuple[str, int], config: MonitoringConfig
) -> None:
    """Ninety days on a seven-day grid, with the last six days unevaluated."""
    run = _run_row(straight[0])
    grid = boundaries(run.start_at, run.end_at, config)
    assert len(grid) == 12
    assert grid[0] == MONITORING_START + timedelta(days=7)
    assert grid[-1] == MONITORING_START + timedelta(days=84)


def test_a_window_prediction_count_matches_a_direct_count(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    dsn, reference_id = straight
    run = _run_row(dsn)
    for evaluation in _evaluate_all(dsn, built_reference, reference_id, config):
        window_start, boundary = window_for(evaluation.boundary, run.start_at, config)
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM predictions WHERE event_time >= %s AND event_time < %s",
                [window_start, boundary],
            ).fetchone()
        assert row is not None
        assert evaluation.prediction_count == int(row[0])


def test_the_early_boundaries_are_sparse_and_the_later_ones_are_not(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """One run has to exercise both sides of the minimum-count rule."""
    dsn, reference_id = straight
    evaluations = _evaluate_all(dsn, built_reference, reference_id, config)
    suppressed = [e for e in evaluations if not e.statistics.minimum_met]
    met = [e for e in evaluations if e.statistics.minimum_met]
    assert suppressed and met
    assert all(e.prediction_count < config.minimum_predictions for e in suppressed)
    assert all(e.prediction_count >= config.minimum_predictions for e in met)
    assert [e.boundary for e in suppressed] == [
        MONITORING_START + timedelta(days=7 * k) for k in (1, 2)
    ]


def test_a_suppressed_window_reports_no_statistic_and_a_full_one_reports_finite_ones(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    dsn, reference_id = straight
    evaluations = _evaluate_all(dsn, built_reference, reference_id, config)
    sparse = next(e for e in evaluations if not e.statistics.minimum_met)
    full = next(e for e in evaluations if e.statistics.minimum_met)
    assert all(sparse.p_value_for(signal) is None for signal in signals.DRIFT_SIGNALS)
    assert sparse.statistics.score_psi.value is None
    for signal in signals.DRIFT_SIGNALS:
        p_value = full.p_value_for(signal)
        assert p_value is not None and 0.0 <= p_value <= 1.0
    assert full.statistics.score_psi.value is not None


def test_labels_arrive_only_after_the_maturation_window(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """Nothing has matured at the first boundaries, so realized metrics start empty."""
    dsn, reference_id = straight
    evaluations = _evaluate_all(dsn, built_reference, reference_id, config)
    assert evaluations[0].label_count == 0
    assert evaluations[0].realized.count == 0
    assert evaluations[-1].label_count > 0
    assert evaluations[-1].realized.count > 0


def test_realized_performance_is_cumulative_and_never_shrinks(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """The drift window is 30 days and labels mature in 30, so this has to be cumulative."""
    dsn, reference_id = straight
    counts = [e.realized.count for e in _evaluate_all(dsn, built_reference, reference_id, config)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[len(counts) // 2]


def test_a_realized_auroc_appears_once_both_classes_have_matured(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    dsn, reference_id = straight
    last = _evaluate_all(dsn, built_reference, reference_id, config)[-1]
    assert last.realized.prevalence is not None and 0.0 < last.realized.prevalence < 1.0
    assert last.realized.auroc is not None


def test_no_prediction_disagrees_with_the_reference_versions(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """One model served the whole run, and it is the one the reference names."""
    dsn, reference_id = straight
    assert all(
        e.version_mismatch_count == 0
        for e in _evaluate_all(dsn, built_reference, reference_id, config)
    )


def test_nothing_was_refused_so_the_refusal_count_is_zero(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """Pins the seam: recording refusals has to make this test say something else."""
    dsn, reference_id = straight
    assert all(
        e.refusal_count == 0 for e in _evaluate_all(dsn, built_reference, reference_id, config)
    )


def test_every_evaluation_serializes_as_jsonb_with_no_nan(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    dsn, reference_id = straight
    for evaluation in _evaluate_all(dsn, built_reference, reference_id, config):
        payload = evaluation.statistics_payload()
        json.dumps(payload, allow_nan=False)
        assert set(payload) == {*signals.REPORTED_SIGNALS, "realized"}


def test_every_evaluation_names_the_committed_thresholds(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    dsn, reference_id = straight
    assert all(
        e.thresholds_hash == config.thresholds_hash
        for e in _evaluate_all(dsn, built_reference, reference_id, config)
    )


def test_the_stored_reference_reads_back_equal_to_the_built_one(
    straight: tuple[str, int], built_reference: Reference
) -> None:
    dsn, reference_id = straight
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        stored = read_reference(conn, MODEL_NAME, built_reference.model_version)
    assert stored is not None
    assert stored.reference_id == reference_id
    assert stored.features == built_reference.features


def test_an_off_grid_boundary_is_refused(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """An arbitrary instant would write a row no second evaluation could reproduce."""
    dsn, reference_id = straight
    run = _run_row(dsn)
    with psycopg.connect(dsn, connect_timeout=2) as conn, pytest.raises(ValueError, match="bound"):
        evaluate(conn, run, built_reference, reference_id, PAUSE_AT, config)


# --- the three arms ---


def test_evaluating_at_each_boundary_matches_evaluating_after_the_run(
    trained_repo: tuple[Path, train.TrainingResult],
    straight: tuple[str, int],
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    built_reference: Reference,
    config: MonitoringConfig,
) -> None:
    """How far past a boundary the harness had run cannot change what the boundary says."""
    reference_dsn, reference_id = straight
    expected = _evaluate_all(reference_dsn, built_reference, reference_id, config)

    live_dsn = db_url_factory()
    serve: Serve = serving(trained_repo)
    live_reference_id = _open(live_dsn, frames, events, built_reference)
    grid = boundaries(MONITORING_START, MONITORING_END, config)

    live: list[Evaluation] = []
    for boundary in grid:
        _replay(serve, live_dsn, events, schedule, pause_at=boundary)
        run = _run_row(live_dsn)
        assert run.sim_now >= boundary
        with psycopg.connect(live_dsn, connect_timeout=2) as conn:
            live.append(evaluate(conn, run, built_reference, live_reference_id, boundary, config))
        with psycopg.connect(live_dsn, connect_timeout=2) as conn:
            runs.set_status(conn, run.run_id, "running")
    _replay(serve, live_dsn, events, schedule)

    assert read_outputs(live_dsn) == read_outputs(reference_dsn)
    assert _volatile_free(live) == _volatile_free(expected)


def test_a_paused_and_resumed_replay_evaluates_identically(
    trained_repo: tuple[Path, train.TrainingResult],
    straight: tuple[str, int],
    db_url_factory: Callable[[], str],
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    built_reference: Reference,
    config: MonitoringConfig,
) -> None:
    """An interruption off the grid changes neither table, and so changes no evaluation."""
    reference_dsn, reference_id = straight
    expected = _evaluate_all(reference_dsn, built_reference, reference_id, config)

    resumed_dsn = db_url_factory()
    serve: Serve = serving(trained_repo)
    resumed_reference_id = _open(resumed_dsn, frames, events, built_reference)

    paused = _replay(serve, resumed_dsn, events, schedule, pause_at=PAUSE_AT)
    assert not paused.finished
    with psycopg.connect(resumed_dsn, connect_timeout=2) as conn:
        run = runs.open_run(conn)
        assert run is not None
        runs.set_status(conn, run.run_id, "running")
    assert _replay(serve, resumed_dsn, events, schedule).finished

    assert read_outputs(resumed_dsn) == read_outputs(reference_dsn)
    assert _volatile_free(
        _evaluate_all(resumed_dsn, built_reference, resumed_reference_id, config)
    ) == _volatile_free(expected)


def test_the_comparison_discriminates_between_signals(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """A statistic that flagged everything would pass every other test in this file.

    The replayed population differs from the reference's on the columns the
    two factories build differently, and agrees on the ones they build the
    same way, so a full window must flag some signals and not others.
    """
    dsn, reference_id = straight
    full = next(
        e
        for e in _evaluate_all(dsn, built_reference, reference_id, config)
        if e.statistics.minimum_met
    )
    flagged = {alert.signal for alert in judge(full, config)}
    assert flagged
    quiet = set(signals.DRIFT_SIGNALS) - flagged
    assert quiet, "every drift signal alerted, so the comparison is not discriminating"
    for signal in quiet:
        p_value = full.p_value_for(signal)
        assert p_value is not None and p_value >= config.thresholds.p_floor_for(signal)


def test_judging_the_same_evaluation_twice_gives_the_same_alerts(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """Judging is pure, so equal evaluations must give equal alerts."""
    dsn, reference_id = straight
    evaluations = _evaluate_all(dsn, built_reference, reference_id, config)
    once = [judge(evaluation, config) for evaluation in evaluations]
    again = [judge(evaluation, config) for evaluation in evaluations]
    assert once == again


def test_an_alert_names_its_own_evaluation_boundary(
    straight: tuple[str, int], built_reference: Reference, config: MonitoringConfig
) -> None:
    """What the alerts table's composite foreign key will then make unfalsifiable."""
    dsn, reference_id = straight
    for evaluation in _evaluate_all(dsn, built_reference, reference_id, config):
        assert all(alert.sim_at == evaluation.boundary for alert in judge(evaluation, config))


def _volatile_free(evaluations: list[Evaluation]) -> list[dict[str, object]]:
    """Everything an evaluation says except the ids the database assigned."""
    return [
        {key: value for key, value in asdict(evaluation).items() if key != "reference_id"}
        for evaluation in evaluations
    ]
