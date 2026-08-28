"""Tests for re-deriving a logged prediction from its source and its model.

A logged prediction claims two things: that the input hash covers the
exact event that produced it, and that the score is what the named model
version returns for the stored feature values. This module checks both by
recomputing them, so the checks here are about reproducing the service's
own construction exactly rather than approximately.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy.typing as npt
import pandas as pd
import pytest

from factories import make_encounter_row
from risk_scoring.features import MODEL_INPUT_COLUMNS
from risk_scoring.payload_hash import payload_hash
from risk_scoring.predictions import StoredPrediction
from risk_scoring.provenance import (
    ProvenanceCheck,
    recompute_input_hash,
    rescore,
    source_event,
)
from risk_scoring.stream import EVENT_FIELDS


class RecordingModel:
    """Records the frame it was asked to predict on."""

    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.seen: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[Any]:
        self.seen = frame.copy()
        return pd.Series([self.value] * len(frame)).to_numpy()


def features(**overrides: float) -> dict[str, float]:
    values = {name: float(index) for index, name in enumerate(MODEL_INPUT_COLUMNS)}
    values.update(overrides)
    return values


def stored(**overrides: Any) -> StoredPrediction:
    defaults: dict[str, Any] = {
        "prediction_id": 1,
        "scored_at": datetime(2026, 1, 1, tzinfo=UTC),
        "patient_id": "p1",
        "encounter_id": "e1",
        "event_time": datetime(2025, 6, 6, tzinfo=UTC),
        "input_hash": "0" * 64,
        "model_name": "readmission-risk",
        "model_version": 3,
        "feature_version": "1.0.0",
        "cohort_version": "1.0.0",
        "score": 0.25,
        "features": features(),
    }
    defaults.update(overrides)
    return StoredPrediction(**defaults)


# The hash


def test_the_recomputed_hash_covers_the_whole_envelope() -> None:
    """The service hashes the posted envelope, not the payload alone."""
    row = make_encounter_row(Id="e1", PATIENT="p1")
    payload = {name: row[name] for name in EVENT_FIELDS["encounter"]}
    assert recompute_input_hash(row) == payload_hash(
        {"event_type": "encounter", "payload": payload}
    )
    assert recompute_input_hash(row) != payload_hash(payload)


def test_the_hash_reads_only_the_contract_fields() -> None:
    """Columns the service never sees cannot change the digest."""
    base = make_encounter_row(Id="e1", TOTAL_CLAIM_COST="100.00")
    other = make_encounter_row(Id="e1", TOTAL_CLAIM_COST="999.99")
    assert recompute_input_hash(base) == recompute_input_hash(other)


def test_a_changed_source_field_changes_the_hash() -> None:
    base = make_encounter_row(Id="e1", START="2025-06-06T08:00:00Z")
    shifted = make_encounter_row(Id="e1", START="2025-06-06T08:00:01Z")
    assert recompute_input_hash(base) != recompute_input_hash(shifted)


def test_the_source_event_is_the_posted_envelope() -> None:
    row = make_encounter_row(Id="e1")
    event = source_event(row)
    assert event["event_type"] == "encounter"
    assert tuple(event["payload"]) == EVENT_FIELDS["encounter"]


# The re-score


def test_rescoring_uses_the_model_input_columns_in_order() -> None:
    model = RecordingModel()
    rescore(model, features())
    assert model.seen is not None
    assert tuple(model.seen.columns) == MODEL_INPUT_COLUMNS
    assert len(model.seen) == 1
    assert set(model.seen.dtypes.astype(str)) == {"float64"}


def test_rescoring_returns_the_models_value_as_a_float() -> None:
    assert rescore(RecordingModel(0.75), features()) == 0.75


def test_rescoring_is_indifferent_to_the_stored_key_order() -> None:
    """JSONB does not preserve insertion order, so the column order cannot come from it."""
    values = features()
    reversed_values = dict(reversed(list(values.items())))
    assert rescore(RecordingModel(), reversed_values) == rescore(RecordingModel(), values)


def test_rescoring_rejects_a_features_dict_missing_a_model_column() -> None:
    """A filled-in default would turn a provenance break into a plausible number."""
    incomplete = features()
    del incomplete["los_days"]
    with pytest.raises(KeyError, match="los_days"):
        rescore(RecordingModel(), incomplete)


# The verdict


def test_a_check_that_reproduces_both_sides_is_ok() -> None:
    check = ProvenanceCheck(
        encounter_id="e1",
        prediction_id=1,
        model_uri="models:/readmission-risk/3",
        logged_hash="a" * 64,
        recomputed_hash="a" * 64,
        logged_score=0.25,
        rescored=0.25,
    )
    assert check.hash_matches and check.score_matches and check.ok


def test_a_nudged_score_is_a_mismatch_with_no_tolerance() -> None:
    """A near-miss is a real defect, so nothing rounds it away."""
    import math

    logged = 0.25
    check = ProvenanceCheck(
        encounter_id="e1",
        prediction_id=1,
        model_uri="models:/readmission-risk/3",
        logged_hash="a" * 64,
        recomputed_hash="a" * 64,
        logged_score=logged,
        rescored=math.nextafter(logged, 1.0),
    )
    assert not check.score_matches
    assert not check.ok


def test_a_mismatch_describes_both_sides() -> None:
    check = ProvenanceCheck(
        encounter_id="e1",
        prediction_id=7,
        model_uri="models:/readmission-risk/3",
        logged_hash="a" * 64,
        recomputed_hash="b" * 64,
        logged_score=0.25,
        rescored=0.75,
    )
    described = check.describe()
    assert "e1" in described
    assert "a" * 64 in described and "b" * 64 in described
    assert "0.25" in described and "0.75" in described


def test_a_matching_check_describes_itself_without_alarm() -> None:
    check = ProvenanceCheck(
        encounter_id="e1",
        prediction_id=7,
        model_uri="models:/readmission-risk/3",
        logged_hash="a" * 64,
        recomputed_hash="a" * 64,
        logged_score=0.25,
        rescored=0.25,
    )
    assert "e1" in check.describe()
    assert "models:/readmission-risk/3" in check.describe()
