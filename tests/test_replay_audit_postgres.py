"""The audit of a replay's tables, over the skew population's replay.

These are the checks the exit criteria name, stated once in the audit
module so a script can run them against a real database and CI can run
them here: the join of the log and the labels equals the batch pipeline
over the same discharges, no label is released before its discharge is
30 simulated days old, every released label is the batch label for its
discharge, and the log names exactly one model version. Each check is
also shown to fail on a table edited to break its rule, so a passing
audit means the rule held rather than that the check was vacuous.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg
import pytest

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
from risk_scoring.labels import READMISSION_WINDOW_DAYS
from risk_scoring.replay import audit, harness, runs
from risk_scoring.replay.realized import RealizedPerformance, realized_performance
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.stream import StreamEvent
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

LABELLED_END = END - timedelta(days=READMISSION_WINDOW_DAYS)


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("audit-population") / "csv")


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
    """A complete replay of the skew population: six scored, five labelled."""
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


# The logged model


def test_the_log_names_one_model(
    replayed: str, trained_repo: tuple[Path, train.TrainingResult]
) -> None:
    _, trained = trained_repo
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        assert audit.logged_model(conn) == audit.LoggedModel(MODEL_NAME, trained.model_version)


def test_an_empty_log_names_no_model(db_url: str) -> None:
    with psycopg.connect(db_url, connect_timeout=2) as conn, pytest.raises(LookupError):
        audit.logged_model(conn)


def test_a_log_naming_two_versions_is_refused(replayed: str) -> None:
    """The batch side must load the version the rows name, so two versions is no answer."""
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        conn.execute(
            "UPDATE predictions SET model_version = model_version + 1"
            " WHERE prediction_id = (SELECT min(prediction_id) FROM predictions)"
        )
        conn.commit()
        with pytest.raises(audit.ManyModelsError, match="2 model versions"):
            audit.logged_model(conn)


# Realized performance against the batch pipeline


def test_the_batch_pipeline_reports_what_the_join_reports(
    replayed: str,
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = trained_repo
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        model = audit.logged_model(conn)
        realized = realized_performance(conn, START, LABELLED_END)
    batch = audit.batch_performance(frames, model, START, LABELLED_END, repo_root=root)
    assert batch.count == 5 and batch.auroc is not None
    assert batch == realized


def test_the_batch_side_follows_the_joins_rules_for_thin_windows(
    replayed: str,
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    """Empty and one-class windows report None the same way, so == stays meaningful."""
    root, _ = trained_repo
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        model = audit.logged_model(conn)
    empty = audit.batch_performance(
        frames,
        model,
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2000, 2, 1, tzinfo=UTC),
        repo_root=root,
    )
    assert empty == RealizedPerformance(count=0, prevalence=None, auroc=None)
    one_class = audit.batch_performance(
        frames, model, datetime(2024, 6, 1, tzinfo=UTC), LABELLED_END, repo_root=root
    )
    assert one_class == RealizedPerformance(count=1, prevalence=0.0, auroc=None)


def test_a_rescored_prediction_breaks_the_equality(
    replayed: str,
    frames: dict[str, pd.DataFrame],
    trained_repo: tuple[Path, train.TrainingResult],
) -> None:
    root, _ = trained_repo
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        model = audit.logged_model(conn)
        conn.execute(
            "UPDATE predictions SET score = 1 - score"
            " WHERE prediction_id = (SELECT min(prediction_id) FROM predictions)"
        )
        conn.commit()
        realized = realized_performance(conn, START, LABELLED_END)
    batch = audit.batch_performance(frames, model, START, LABELLED_END, repo_root=root)
    assert batch.count == realized.count and batch != realized


# The never-early rule


def test_no_released_label_is_early(replayed: str) -> None:
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        assert audit.early_labels(conn) == 0


def test_a_label_released_inside_the_window_is_counted(replayed: str) -> None:
    """The table only checks released_at against due_at, so a wrong due_at slips past it."""
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        conn.execute(
            "INSERT INTO labels"
            " (prediction_id, encounter_id, label, label_version, due_at, released_at)"
            " SELECT p.prediction_id, p.encounter_id, 0, 'audit-test',"
            "  p.event_time + interval '1 day', p.event_time + interval '1 day'"
            " FROM predictions p LEFT JOIN labels l USING (prediction_id)"
            " WHERE l.label_id IS NULL"
        )
        conn.commit()
        assert audit.early_labels(conn) == 1


# Labels against the export


def test_every_released_label_is_the_batch_label(
    replayed: str, frames: dict[str, pd.DataFrame]
) -> None:
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        result = audit.label_audit(conn, frames)
    assert result == audit.LabelAudit(checked=5, disagreements=())
    assert result.ok


def test_a_flipped_label_is_named(replayed: str, frames: dict[str, pd.DataFrame]) -> None:
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        flipped = conn.execute(
            "UPDATE labels SET label = 1 - label"
            " WHERE label_id = (SELECT min(label_id) FROM labels) RETURNING encounter_id"
        ).fetchone()
        conn.commit()
        assert flipped is not None
        result = audit.label_audit(conn, frames)
    assert result == audit.LabelAudit(checked=5, disagreements=(flipped[0],))
    assert not result.ok


def test_a_label_for_a_discharge_the_export_lacks_is_named(
    replayed: str, frames: dict[str, pd.DataFrame]
) -> None:
    """A released label the batch pipeline never produced cannot be the batch label."""
    with psycopg.connect(replayed, connect_timeout=2) as conn:
        renamed = conn.execute(
            "UPDATE labels SET encounter_id = 'not-in-the-export'"
            " WHERE label_id = (SELECT min(label_id) FROM labels) RETURNING encounter_id"
        ).fetchone()
        conn.commit()
        assert renamed is not None
        result = audit.label_audit(conn, frames)
    assert result.disagreements == ("not-in-the-export",)
