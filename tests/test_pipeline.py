"""Tests for the chained retrain entrypoint.

The rules these tests pin:

- One call goes from a raw Synthea CSV export to a registered model
  version and a written gate report, with nothing manual in between.
- Retraining an already-populated registry registers the next version
  and gates that version, not an earlier one.
- The command prints the training summary and the gate report, and
  exits non-zero when the gate fails while leaving the candidate
  registered and the report on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow import MlflowClient

from factories import write_gate_population, write_training_csvs
from risk_scoring import pipeline
from risk_scoring.train import MODEL_NAME


def test_retrain_registers_a_model_and_writes_a_gate_report(repo_root: Path) -> None:
    csv_dir = repo_root / "data" / "baseline" / "csv"
    write_gate_population(csv_dir)
    report_path = repo_root / "gate_report.md"

    outcome = pipeline.retrain(csv_dir, repo_root, report_path=report_path)

    assert outcome.training.model_version == 1
    assert outcome.gate.result.verdict == "pass"
    assert "PASS" in report_path.read_text()


def test_retraining_again_gates_the_newly_registered_version(repo_root: Path) -> None:
    csv_dir = repo_root / "data" / "baseline" / "csv"
    write_gate_population(csv_dir)

    first = pipeline.retrain(csv_dir, repo_root, report_path=repo_root / "first.md")
    second = pipeline.retrain(csv_dir, repo_root, report_path=repo_root / "second.md")

    assert (first.training.model_version, second.training.model_version) == (1, 2)
    assert second.gate.model_version == 2
    assert second.gate.candidate_run_id == second.training.run_id
    version = MlflowClient().get_model_version(MODEL_NAME, "2")
    assert version.tags["gate_verdict"] == second.gate.result.verdict
    assert version.tags["gate_run_id"] == second.gate.gate_run_id


def test_cli_retrains_gates_and_prints_both_reports(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_gate_population(repo_root / "data" / "baseline" / "csv")
    report_path = repo_root / "gate_report.md"
    monkeypatch.chdir(repo_root)

    pipeline.main(["retrain", "--report", str(report_path)])

    out = capsys.readouterr().out
    assert "holdout AUROC:" in out  # the training summary
    assert "PASS" in out  # the gate report
    assert "PASS" in report_path.read_text()


def test_cli_exits_nonzero_on_failing_gate_but_keeps_the_evidence(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The deterministic training population separates perfectly, so the
    # honest model lands above the band ceiling and the gate must fail.
    write_training_csvs(repo_root / "data" / "baseline" / "csv")
    report_path = repo_root / "gate_report.md"
    monkeypatch.chdir(repo_root)

    with pytest.raises(SystemExit) as excinfo:
        pipeline.main(["retrain", "--report", str(report_path)])

    assert excinfo.value.code == 1
    # Asserting only that it failed would also pass if the model came back at
    # chance and tripped the floor instead, which is the opposite diagnosis.
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "SUSPECTED LEAKAGE" in output
    assert "FAIL" in report_path.read_text()
    version = MlflowClient().get_model_version(MODEL_NAME, "1")
    assert version.tags["gate_verdict"] == "fail"
