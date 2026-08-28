"""The held-out batch driver against a real service and a real log.

The driver is exercised end to end: it selects patients, posts their whole
history through a service, reads back what was logged, and recomputes the
provenance of every prediction. The service here is an in-process test
client rather than a container, so the same path CI runs is the one an
operator runs against Compose.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from factories import write_gate_population, write_skew_population
from risk_scoring import predictions, scoring_run, train
from risk_scoring.service.app import create_app
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

pytestmark = pytest.mark.db

# Splits the boundary population four discharges before and six after,
# leaving one of its five patients ineligible, so the sample is a sample.
CUTOFF = pd.Timestamp("2024-04-01", tz="UTC")


class ClientPoster:
    """The EventPoster the driver needs, backed by the FastAPI test client."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.posted: list[Mapping[str, Any]] = []

    def version(self) -> dict[str, Any]:
        response = self.client.get("/version")
        assert response.status_code == 200, response.text
        return dict(response.json())

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.post("/events", json=event)
        if response.status_code != 202:
            raise RuntimeError(f"{event.get('event_type')} refused: {response.status_code}")
        self.posted.append(event)
        return dict(response.json())


@pytest.fixture(scope="module")
def csv_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("batch-population") / "csv"
    write_skew_population(directory)
    return directory


@pytest.fixture(scope="module")
def signal_repo(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, train.TrainingResult]]:
    """A registry whose model's score depends on its input, so checks can bite."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("batch-repo")
    write_gate_population(root / "data" / "baseline" / "csv")
    result = train.train(root / "data" / "baseline" / "csv", root)
    yield root, result
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


@pytest.fixture()
def run(
    csv_dir: Path, signal_repo: tuple[Path, train.TrainingResult], db_url: str
) -> Iterator[Any]:
    """Run one batch against a fresh database, returning result and poster."""
    root, trained = signal_repo
    app = create_app(ServiceConfig(MODEL_NAME, trained.model_version), root, db_url)

    def go(**overrides: Any) -> tuple[scoring_run.BatchRunResult, ClientPoster]:
        with TestClient(app) as client:
            poster = ClientPoster(client)
            options: dict[str, Any] = {
                "cutoff": CUTOFF,
                "count": 10,
                "seed": 20260101,
                "dsn": db_url,
                "poster": poster,
            }
            options.update(overrides)
            return scoring_run.run_batch(csv_dir, root, **options), poster

    yield go


def test_the_run_reports_a_batch_it_actually_scored(run: Any) -> None:
    """Every assertion below would hold vacuously on an empty run."""
    result, _ = run()
    assert result.selection.patients_selected > 0
    assert result.events_posted > 0
    assert len(result.partition.held_out) > 0
    assert result.ok


def test_every_cohort_discharge_of_every_selected_patient_is_scored(run: Any) -> None:
    result, _ = run()
    assert result.partition.unscored_cohort_ids == ()
    assert result.partition.unexpected_logged_ids == ()
    assert result.predictions_logged == len(result.selection.cohort_encounter_ids)


def test_the_run_verifies_provenance_for_every_prediction_it_logged(run: Any) -> None:
    result, _ = run()
    assert len(result.provenance) == result.predictions_logged
    assert [check.describe() for check in result.provenance if not check.ok] == []


def test_each_acknowledgement_agrees_with_its_logged_row(run: Any) -> None:
    """What the service returned at the time and what it durably wrote."""
    result, _ = run()
    assert result.response_mismatches == ()


def test_the_full_history_of_selected_patients_is_posted(run: Any) -> None:
    """Pre-cutoff rows must reach the service, or features would disagree."""
    result, poster = run()
    posted_encounters = {
        str(event["payload"]["Id"]) for event in poster.posted if event["event_type"] == "encounter"
    }
    assert result.selection.pre_cutoff_encounter_ids
    assert result.selection.pre_cutoff_encounter_ids <= posted_encounters
    assert result.selection.held_out_encounter_ids <= posted_encounters


def test_held_out_and_pre_cutoff_predictions_are_reported_apart(run: Any) -> None:
    result, _ = run()
    assert result.held_out_scores.count == len(result.partition.held_out)
    assert result.all_scores.count == result.predictions_logged
    assert result.held_out_scores.count < result.all_scores.count


def test_only_predictions_for_posted_encounters_enter_the_summary(run: Any, db_url: str) -> None:
    """Whatever else the log holds, the summary describes this batch."""
    baseline, _ = run()
    with psycopg.connect(db_url) as conn:
        predictions.record_prediction(
            conn,
            predictions.PredictionRecord(
                patient_id="stranger",
                encounter_id="not-from-this-batch",
                event_time=baseline.partition.held_out[0].event_time,
                input_hash="f" * 64,
                model_name=MODEL_NAME,
                model_version=1,
                feature_version="1.0.0",
                cohort_version="1.0.0",
                score=0.999,
                features=dict(baseline.partition.held_out[0].features),
            ),
        )
    again, _ = run()
    assert again.predictions_logged == baseline.predictions_logged
    assert again.partition.unexpected_logged_ids == ()
    assert again.ok


def test_the_worked_example_is_a_held_out_prediction(run: Any) -> None:
    result, _ = run()
    assert result.example is not None
    assert result.example.encounter_id in result.selection.held_out_encounter_ids
    assert result.example.ok


def test_a_named_example_is_the_one_written_up(run: Any) -> None:
    result, _ = run()
    target = sorted(row.encounter_id for row in result.partition.held_out)[0]
    chosen, _ = run(example=target)
    assert chosen.example is not None
    assert chosen.example.encounter_id == target


def test_the_report_names_the_two_discharge_sets_apart(run: Any) -> None:
    result, _ = run()
    text = scoring_run.report(result)
    assert "held-out discharges" in text
    assert "not held-out evidence" in text
    assert str(len(result.selection.held_out_encounter_ids)) in text


def test_a_broken_provenance_check_makes_the_run_not_ok(run: Any) -> None:
    result, _ = run()
    broken = scoring_run.BatchRunResult(
        selection=result.selection,
        version=result.version,
        events_by_type=result.events_by_type,
        partition=result.partition,
        all_scores=result.all_scores,
        held_out_scores=result.held_out_scores,
        provenance=(
            *result.provenance[:-1],
            type(result.provenance[-1])(
                **{**vars(result.provenance[-1]), "recomputed_hash": "9" * 64}
            ),
        ),
        example=result.example,
    )
    assert not broken.ok
    assert "BROKEN" in scoring_run.report(broken)


def test_the_cli_exits_nonzero_when_the_run_is_not_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: Any
) -> None:
    """A run that failed its own checks must not report success."""
    result, _ = run()
    not_ok = scoring_run.BatchRunResult(
        selection=result.selection,
        version=result.version,
        events_by_type=result.events_by_type,
        partition=scoring_run.LogPartition(
            held_out=result.partition.held_out,
            pre_cutoff=result.partition.pre_cutoff,
            unscored_cohort_ids=("e-never-scored",),
            unexpected_logged_ids=(),
        ),
        all_scores=result.all_scores,
        held_out_scores=result.held_out_scores,
        provenance=result.provenance,
        example=result.example,
    )
    (tmp_path / "data" / "baseline" / "csv").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scoring_run, "run_batch", lambda *a, **k: not_ok)

    with pytest.raises(SystemExit) as raised:
        scoring_run.main(["run"])
    assert raised.value.code == 1


def test_the_cli_refuses_a_population_that_is_not_generated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        scoring_run.main(["run", "--population", "absent"])
    assert "no CSV export" in str(raised.value)
