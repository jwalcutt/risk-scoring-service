"""Tests for the pinned model's version compatibility check.

The rules these tests pin:

- A version series is the major and minor numbers only. features.py
  defines a patch bump as parsing or validation that moves no value, so a
  patch difference between the model and the code is not a mismatch; a
  minor difference is exactly the case the module says means "a model
  trained under the previous number no longer matches serving".
- The check covers every version the service actually executes with, and
  reports all of them that disagree rather than the first, so one
  refusal names the whole problem.
- A training run that logged no version at all is a refusal, not a pass.
  The same goes for a version string the check cannot parse: an
  unrecognized value is never treated as compatible.
- A registry the check cannot read is a refusal naming the model, not a
  raw MlflowException escaping into the startup path.

The startup wiring, which needs a registry and a database, is pinned in
test_service_app.
"""

from __future__ import annotations

from typing import Any

import pytest
from mlflow.exceptions import MlflowException

from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.service import compatibility
from risk_scoring.service.compatibility import (
    SERVING_VERSIONS,
    VersionMismatch,
    describe,
    incompatible,
    major_minor,
    verify_pinned_versions,
)
from risk_scoring.service.config import ServiceConfig
from risk_scoring.train import MODEL_NAME

RUNNING = {"feature_version": "1.1.0", "cohort_version": "1.0.0"}


# --- the series parse ---


def test_major_minor_keeps_major_and_minor_and_drops_patch() -> None:
    assert major_minor("1.1.0") == (1, 1)
    assert major_minor("1.1.7") == (1, 1)
    assert major_minor("10.20.30") == (10, 20)


# Arabic-Indic digits, written as escapes so the source stays ASCII. The
# case is here because "\d" matches them and "[0-9]" does not: a version
# read back from the registry is whatever was written there, and only the
# ASCII digits this project writes may parse.
NON_ASCII_DIGITS = "\u0661.\u0662.\u0663"


@pytest.mark.parametrize(
    "value",
    ["latest", "1", "1.2", "1.2.3.4", "1.x.0", "", "v1.2.3", "1.2.3 ", NON_ASCII_DIGITS],
)
def test_major_minor_refuses_anything_that_is_not_three_integers(value: str) -> None:
    with pytest.raises(ValueError, match="three-part version"):
        major_minor(value)


# --- the comparison ---


def test_identical_versions_are_compatible() -> None:
    assert incompatible(dict(RUNNING), RUNNING) == []


@pytest.mark.parametrize("pinned_patch", ["1.1.0", "1.1.2", "1.1.99"])
def test_a_patch_difference_is_compatible_in_either_direction(pinned_patch: str) -> None:
    """features.py defines a patch bump as moving no value on existing data."""
    pinned = {**RUNNING, "feature_version": pinned_patch}
    assert incompatible(pinned, RUNNING) == []
    assert incompatible(dict(RUNNING), {**RUNNING, "feature_version": pinned_patch}) == []


def test_a_minor_difference_is_a_mismatch() -> None:
    pinned = {**RUNNING, "feature_version": "1.0.0"}
    assert incompatible(pinned, RUNNING) == [VersionMismatch("feature_version", "1.0.0", "1.1.0")]


def test_a_major_difference_is_a_mismatch() -> None:
    pinned = {**RUNNING, "cohort_version": "2.0.0"}
    assert incompatible(pinned, RUNNING) == [VersionMismatch("cohort_version", "2.0.0", "1.0.0")]


def test_every_disagreeing_signal_is_reported_not_just_the_first() -> None:
    pinned = {"feature_version": "1.0.0", "cohort_version": "0.9.0"}
    assert incompatible(pinned, RUNNING) == [
        VersionMismatch("feature_version", "1.0.0", "1.1.0"),
        VersionMismatch("cohort_version", "0.9.0", "1.0.0"),
    ]


def test_a_version_the_run_never_logged_is_a_mismatch() -> None:
    """A model registered before a version was logged cannot be vouched for."""
    pinned = {"cohort_version": "1.0.0"}
    assert incompatible(pinned, RUNNING) == [VersionMismatch("feature_version", None, "1.1.0")]


@pytest.mark.parametrize("unparseable", ["latest", "", "1.1"])
def test_an_unparseable_pinned_version_is_a_mismatch_not_a_crash(unparseable: str) -> None:
    pinned = {**RUNNING, "feature_version": unparseable}
    assert incompatible(pinned, RUNNING) == [
        VersionMismatch("feature_version", unparseable, "1.1.0")
    ]


def test_the_other_params_a_training_run_logs_are_ignored() -> None:
    """Training logs a cutoff, a seed, and the booster's parameters too."""
    pinned = {**RUNNING, "label_version": "9.9.9", "split_seed": "20260101", "data_dir": "/tmp"}
    assert incompatible(pinned, RUNNING) == []


# --- what the service actually checks ---


def test_the_checked_versions_are_the_ones_serving_executes_with() -> None:
    """Restated literally, so a third version added later must be decided on.

    label_version is deliberately absent: it builds training targets and
    never runs at serving, so a model trained under a different one still
    matches the code that scores.
    """
    assert SERVING_VERSIONS == {
        "feature_version": FEATURE_VERSION,
        "cohort_version": COHORT_VERSION,
    }


# --- the message ---


def test_the_message_names_the_model_the_version_and_both_numbers() -> None:
    message = describe(MODEL_NAME, 3, [VersionMismatch("feature_version", "1.0.0", "1.1.0")])
    assert MODEL_NAME in message
    assert "version 3" in message
    assert "1.0.0" in message
    assert "1.1.0" in message
    assert "feature version" in message


def test_the_message_distinguishes_an_absent_version_from_a_different_one() -> None:
    absent = describe(MODEL_NAME, 3, [VersionMismatch("feature_version", None, "1.1.0")])
    assert "logged no feature version" in absent
    different = describe(MODEL_NAME, 3, [VersionMismatch("feature_version", "1.0.0", "1.1.0")])
    assert "logged no" not in different


def test_the_message_names_every_mismatch() -> None:
    message = describe(
        MODEL_NAME,
        3,
        [
            VersionMismatch("feature_version", "1.0.0", "1.1.0"),
            VersionMismatch("cohort_version", "0.9.0", "1.0.0"),
        ],
    )
    assert "feature version" in message
    assert "cohort version" in message
    assert "0.9.0" in message


# --- reading the registry ---


class _RaisingClient:
    """Stands in for MlflowClient when the registry cannot answer."""

    def __init__(self) -> None:
        pass

    def get_model_version(self, name: str, version: str) -> Any:
        raise MlflowException("run not found")

    def get_run(self, run_id: str) -> Any:  # pragma: no cover - never reached
        raise AssertionError("the lookup should have failed before this")


def test_a_registry_that_cannot_be_read_refuses_by_name(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw MlflowException must never escape into the startup path."""
    monkeypatch.setattr(compatibility, "configure_tracking", lambda root: "0")
    monkeypatch.setattr(compatibility, "MlflowClient", _RaisingClient)
    with pytest.raises(RuntimeError, match=rf"{MODEL_NAME}.*3"):
        verify_pinned_versions(ServiceConfig(MODEL_NAME, 3), tmp_path)
