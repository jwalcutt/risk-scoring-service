"""Realized performance: the join of the log and the labels over a simulated window.

The rule this file pins is exact equality with the batch pipeline: over a
complete replay of the skew population, the count, prevalence, and AUROC
that the join reconstructs equal what the cohort, feature, and label
modules and the registered model compute over the same discharges. The
scores are identical by the skew check and the labels by the label
proof, so the metrics must be identical too; a tolerance would hide a
join bug. The window is half-open on the discharge instant, and a
discharge without a label is not in it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import psycopg
import pytest
from sklearn.metrics import roc_auc_score

from replay_support import (
    END,
    MAX_SPEED,
    START,
    ClientPoster,
    Serve,
    prepare,
    schedule_of,
    serving,
    skew_frames,
    stream_of,
)
from risk_scoring import train
from risk_scoring.cohort import build_cohort
from risk_scoring.features import MODEL_INPUT_COLUMNS, build_features
from risk_scoring.labels import READMISSION_WINDOW_DAYS, build_labels
from risk_scoring.replay import harness, runs
from risk_scoring.replay.realized import RealizedPerformance, realized_performance
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent
from risk_scoring.tracking import configure_tracking
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

LABELLED_END = END - timedelta(days=READMISSION_WINDOW_DAYS)


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("realized-population") / "csv")


@pytest.fixture(scope="module")
def events(frames: dict[str, pd.DataFrame]) -> list[StreamEvent]:
    return stream_of(frames)


@pytest.fixture(scope="module")
def schedule(frames: dict[str, pd.DataFrame]) -> list[ScheduledLabel]:
    return schedule_of(frames)


@pytest.fixture()
def replayed(
    trained_repo: tuple[Path, train.TrainingResult],
    db_url: str,
    frames: dict[str, pd.DataFrame],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> str:
    """A complete replay of the skew population; yields its DSN."""
    serve: Serve = serving(trained_repo)
    prepare(db_url, frames, events)
    with psycopg.connect(db_url, connect_timeout=2) as conn, serve(db_url) as client:
        run = runs.open_run(conn)
        assert run is not None
        summary = harness.run_replay(
            conn, run, events, ClientPoster(client), labels=schedule, pacing=MAX_SPEED
        )
    assert summary.finished and summary.labels_released == 5
    return db_url


def _realized(dsn: str, start: datetime, end: datetime) -> RealizedPerformance:
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        return realized_performance(conn, start, end)


def _batch(
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
    start: datetime,
    end: datetime,
) -> tuple[int, float, float]:
    """Count, prevalence, and AUROC from the modules training uses, over one window."""
    root, trained = trained_repo
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    stop = cohort["stop"]
    window = cohort.loc[(stop >= start) & (stop < end)].reset_index(drop=True)
    labels = build_labels(window, frames["encounters"])
    features = build_features(
        window, frames["encounters"], frames["medications"], frames["conditions"]
    )
    x = features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    configure_tracking(root)
    model: Any = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{trained.model_version}")
    scores = np.asarray(model.predict(x), dtype=float)
    y = labels["label"].to_numpy(dtype=float)
    return len(y), float(y.mean()), float(roc_auc_score(y, scores))


def test_the_join_reconstructs_the_batch_pipelines_metrics_exactly(
    replayed: str,
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    count, prevalence, auroc = _batch(frames, trained_repo, START, LABELLED_END)
    realized = _realized(replayed, START, LABELLED_END)
    assert count == 5
    assert realized == RealizedPerformance(count=count, prevalence=prevalence, auroc=auroc)


def test_a_window_reaching_past_the_labelled_span_counts_only_labelled_discharges(
    replayed: str,
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    """Six discharges were scored in the run; the sixth has no label yet and is not counted."""
    whole_run = _realized(replayed, START, END)
    assert whole_run == _realized(replayed, START, LABELLED_END)
    assert whole_run.count == 5


def test_the_window_is_half_open_on_the_discharge_instant(replayed: str) -> None:
    first = datetime(2024, 4, 10, 8, tzinfo=UTC)
    second = datetime(2024, 4, 12, 8, tzinfo=UTC)
    assert _realized(replayed, first, second).count == 1
    assert _realized(replayed, first + timedelta(seconds=1), second).count == 0
    assert _realized(replayed, first, second + timedelta(seconds=1)).count == 2


def test_a_window_holding_one_class_has_a_prevalence_and_no_auroc(replayed: str) -> None:
    only_negatives = _realized(replayed, datetime(2024, 6, 1, tzinfo=UTC), LABELLED_END)
    assert only_negatives == RealizedPerformance(count=1, prevalence=0.0, auroc=None)


def test_an_empty_window_has_no_metrics(replayed: str) -> None:
    before_the_run = _realized(replayed, datetime(2024, 1, 1, tzinfo=UTC), START)
    assert before_the_run == RealizedPerformance(count=0, prevalence=None, auroc=None)


def test_the_window_bounds_must_be_ordered(replayed: str) -> None:
    with pytest.raises(ValueError, match="before"):
        _realized(replayed, END, START)
