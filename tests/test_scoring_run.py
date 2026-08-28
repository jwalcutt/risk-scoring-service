"""Tests for selecting and summarizing a held-out batch scoring run.

The run posts patients, not encounters. A patient chosen for a discharge
the model never saw is posted with their whole history, so the service
also scores their earlier discharges, which the model was trained
through. Those are not held-out evidence, and the counts must keep the
two apart rather than letting one number stand for both.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from factories import (
    CONDITION_DEFAULTS,
    ENCOUNTER_DEFAULTS,
    MEDICATION_DEFAULTS,
    PATIENT_DEFAULTS,
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
)
from risk_scoring.features import MODEL_INPUT_COLUMNS
from risk_scoring.predictions import StoredPrediction
from risk_scoring.scoring_run import (
    EmptyBatchError,
    partition_log,
    select_held_out_batch,
    summarize_scores,
)

CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")


def population(
    patients: list[dict[str, str]],
    encounters: list[dict[str, str]],
    medications: list[dict[str, str]] | None = None,
    conditions: list[dict[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "patients": pd.DataFrame(patients or None, columns=list(PATIENT_DEFAULTS)),
        "encounters": pd.DataFrame(encounters or None, columns=list(ENCOUNTER_DEFAULTS)),
        "medications": pd.DataFrame(medications or None, columns=list(MEDICATION_DEFAULTS)),
        "conditions": pd.DataFrame(conditions or None, columns=list(CONDITION_DEFAULTS)),
    }


def adult(patient_id: str) -> dict[str, str]:
    return make_patient_row(Id=patient_id, BIRTHDATE="1970-01-01")


def stay(encounter_id: str, patient_id: str, stop: str) -> dict[str, str]:
    return make_encounter_row(
        Id=encounter_id, PATIENT=patient_id, ENCOUNTERCLASS="inpatient", START=stop, STOP=stop
    )


def logged(encounter_id: str, score: float = 0.5) -> StoredPrediction:
    return StoredPrediction(
        prediction_id=abs(hash(encounter_id)) % 10_000,
        scored_at=datetime(2026, 1, 1, tzinfo=UTC),
        patient_id="p1",
        encounter_id=encounter_id,
        event_time=datetime(2025, 6, 6, tzinfo=UTC),
        input_hash="0" * 64,
        model_name="readmission-risk",
        model_version=3,
        feature_version="1.0.0",
        cohort_version="1.0.0",
        score=score,
        features={name: 1.0 for name in MODEL_INPUT_COLUMNS},
    )


# Selection


def test_held_out_and_pre_cutoff_discharges_are_counted_separately() -> None:
    """One number for both would read as held-out evidence it is not."""
    frames = population(
        [adult("p1")],
        [
            stay("e-old-1", "p1", "2021-03-03T00:00:00Z"),
            stay("e-old-2", "p1", "2023-04-04T00:00:00Z"),
            stay("e-new", "p1", "2025-06-06T00:00:00Z"),
        ],
    )
    selection = select_held_out_batch(frames, cutoff=CUTOFF, count=10, seed=20260101)
    assert selection.held_out_encounter_ids == {"e-new"}
    assert selection.pre_cutoff_encounter_ids == {"e-old-1", "e-old-2"}
    assert selection.patients_selected == 1
    assert selection.patients_eligible == 1


def test_the_selection_carries_the_full_history_of_every_chosen_patient() -> None:
    """Truncating at the cutoff would make serving features disagree."""
    frames = population(
        [adult("p1")],
        [
            stay("e-old", "p1", "2019-01-01T00:00:00Z"),
            stay("e-new", "p1", "2025-06-06T00:00:00Z"),
        ],
        [make_medication_row(PATIENT="p1", ENCOUNTER="e-old", START="2019-02-02T00:00:00Z")],
        [make_condition_row(PATIENT="p1", ENCOUNTER="e-old", START="2018-05-05")],
    )
    selection = select_held_out_batch(frames, cutoff=CUTOFF, count=10, seed=20260101)
    assert sorted(selection.frames["encounters"]["Id"]) == ["e-new", "e-old"]
    assert len(selection.frames["medications"]) == 1
    assert len(selection.frames["conditions"]) == 1


def test_only_patients_with_a_post_cutoff_discharge_are_drawn() -> None:
    frames = population(
        [adult("p-old"), adult("p-new")],
        [
            stay("e-old", "p-old", "2020-01-01T00:00:00Z"),
            stay("e-new", "p-new", "2025-06-06T00:00:00Z"),
        ],
    )
    selection = select_held_out_batch(frames, cutoff=CUTOFF, count=10, seed=20260101)
    assert list(selection.frames["patients"]["Id"]) == ["p-new"]
    assert selection.patients_eligible == 1


def test_the_eligible_pool_is_reported_even_when_the_sample_is_smaller() -> None:
    """A sample that is really the whole pool must not read as a sample."""
    frames = population(
        [adult(f"p{index}") for index in range(6)],
        [stay(f"e{index}", f"p{index}", "2025-06-06T00:00:00Z") for index in range(6)],
    )
    selection = select_held_out_batch(frames, cutoff=CUTOFF, count=2, seed=20260101)
    assert selection.patients_eligible == 6
    assert selection.patients_selected == 2


def test_a_population_with_no_post_cutoff_discharge_raises() -> None:
    """A silent no-op must never be written up as a successful run."""
    frames = population([adult("p1")], [stay("e1", "p1", "2020-01-01T00:00:00Z")])
    with pytest.raises(EmptyBatchError, match="no patient"):
        select_held_out_batch(frames, cutoff=CUTOFF, count=10, seed=20260101)


# Summary


def test_the_score_summary_reports_the_distribution() -> None:
    """A linear ramp makes every quantile exact."""
    summary = summarize_scores([value / 100 for value in range(101)])
    assert summary.count == 101
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    assert summary.p50 == pytest.approx(0.5)
    assert summary.p25 == pytest.approx(0.25)
    assert summary.p75 == pytest.approx(0.75)
    assert summary.p05 == pytest.approx(0.05)
    assert summary.p95 == pytest.approx(0.95)
    assert summary.mean == pytest.approx(0.5)


def test_an_empty_score_summary_counts_zero_without_inventing_quantiles() -> None:
    summary = summarize_scores([])
    assert summary.count == 0
    assert summary.mean is None
    assert summary.p50 is None


# Partitioning the log


def selection_of(held_out: set[str], pre_cutoff: set[str]) -> Any:
    frames = population([adult("p1")], [stay("e1", "p1", "2025-06-06T00:00:00Z")])
    selection = select_held_out_batch(frames, cutoff=CUTOFF, count=1, seed=20260101)
    return type(selection)(
        frames=selection.frames,
        cutoff=CUTOFF,
        seed=20260101,
        patients_eligible=1,
        patients_selected=1,
        held_out_encounter_ids=frozenset(held_out),
        pre_cutoff_encounter_ids=frozenset(pre_cutoff),
    )


def test_the_partition_splits_the_log_by_which_side_of_the_cutoff() -> None:
    selection = selection_of({"e-new"}, {"e-old"})
    partition = partition_log(selection, [logged("e-old", 0.1), logged("e-new", 0.9)])
    assert [row.encounter_id for row in partition.held_out] == ["e-new"]
    assert [row.encounter_id for row in partition.pre_cutoff] == ["e-old"]
    assert partition.unscored_cohort_ids == ()
    assert partition.unexpected_logged_ids == ()


def test_a_cohort_discharge_the_service_never_scored_is_named() -> None:
    selection = selection_of({"e-new", "e-missing"}, set())
    partition = partition_log(selection, [logged("e-new")])
    assert partition.unscored_cohort_ids == ("e-missing",)


def test_a_logged_row_outside_both_sets_is_named() -> None:
    """The service and the batch cohort disagreeing is a finding, not a crash."""
    selection = selection_of({"e-new"}, set())
    partition = partition_log(selection, [logged("e-new"), logged("e-surprise")])
    assert partition.unexpected_logged_ids == ("e-surprise",)


def test_every_logged_row_lands_in_exactly_one_bucket() -> None:
    selection = selection_of({"e1"}, {"e2"})
    rows = [logged("e1"), logged("e2"), logged("e3")]
    partition = partition_log(selection, rows)
    assert len(partition.held_out) + len(partition.pre_cutoff) + len(
        partition.unexpected_logged_ids
    ) == len(rows)
