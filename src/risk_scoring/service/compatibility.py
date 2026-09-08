"""Refuse a pinned model whose feature definitions differ from this code's.

The service loads a model version pinned in ``configs/service.toml`` and
computes features through ``risk_scoring.features`` and cohort membership
through ``risk_scoring.cohort`` on every request. Nothing tied the two
together: a model fitted under one set of feature definitions went on
serving after those definitions changed, and the only record of the
disagreement was two fields of the prediction log that nobody compared.

This module makes the pair a startup precondition. Training logs its
code versions as run parameters, so the pinned version's own training run
says what it was fitted under, and that is compared against what this
process will compute with. A disagreement stops startup, alongside the
missing token, the absent registry version, and the unreachable database.

Two judgment calls, both deliberate.

The comparison is on the major and minor numbers only. ``features.py``
defines its patch number as moving when parsing or validation changes and
no value does, and its minor number as the case where a model trained
under the previous number no longer matches serving. Comparing full
strings would force a retrain for a parsing fix the module itself says is
value-preserving, which would put pressure on nobody ever bumping the
patch number.

``label_version`` is not checked. It builds training targets and never
runs at serving, so a model trained under a different one still matches
the code that scores. ``SERVING_VERSIONS`` states the covered set once so
that a version added to training later is a decision rather than an
omission.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION
from risk_scoring.service.config import ServiceConfig
from risk_scoring.tracking import configure_tracking, tracking_uri

# The versions this process will actually compute with, keyed by the run
# parameter training logs them under.
SERVING_VERSIONS: Mapping[str, str] = {
    "feature_version": FEATURE_VERSION,
    "cohort_version": COHORT_VERSION,
}

_SERIES = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


@dataclass(frozen=True)
class VersionMismatch:
    """One version the pinned model and this code disagree on.

    ``pinned`` is None when the model's training run logged no such
    version at all, which is a refusal for the same reason a differing
    one is: the service cannot show that the model matches its code.
    """

    signal: str
    pinned: str | None
    running: str


def major_minor(version: str) -> tuple[int, int]:
    """The comparable series of a version string, dropping the patch number."""
    if not _SERIES.match(version):
        raise ValueError(f"expected a three-part version like '1.2.3', got {version!r}")
    major, minor, _ = version.split(".")
    return int(major), int(minor)


def _same_series(pinned: str, running: str) -> bool:
    try:
        return major_minor(pinned) == major_minor(running)
    except ValueError:
        # A value this code cannot read is never treated as compatible.
        return False


def incompatible(pinned: Mapping[str, str], running: Mapping[str, str]) -> list[VersionMismatch]:
    """Every version in ``running`` that ``pinned`` does not match.

    ``pinned`` is a training run's full parameter mapping, so it carries
    the cutoff, the split seed, and the booster's settings as well; only
    the keys in ``running`` are read.
    """
    found = []
    for signal, running_value in running.items():
        pinned_value = pinned.get(signal)
        if pinned_value is None or not _same_series(pinned_value, running_value):
            found.append(VersionMismatch(signal, pinned_value, running_value))
    return found


def describe(model_name: str, model_version: int, found: list[VersionMismatch]) -> str:
    """The refusal message, naming every version that disagreed."""
    clauses = []
    for mismatch in found:
        label = mismatch.signal.replace("_", " ")
        if mismatch.pinned is None:
            clauses.append(f"it logged no {label}, against this service's {mismatch.running}")
        else:
            clauses.append(
                f"it was trained at {label} {mismatch.pinned}, "
                f"against this service's {mismatch.running}"
            )
    return (
        f"model {model_name!r} version {model_version} was not fitted under this "
        f"service's code: {'; '.join(clauses)}. The service refuses to score through "
        f"definitions its model was not fitted on; retrain and re-pin, or pin a "
        f"version trained at the current versions."
    )


def verify_pinned_versions(config: ServiceConfig, repo_root: Path) -> None:
    """Raise unless the pinned model was fitted under this code's versions."""
    configure_tracking(repo_root)
    client = MlflowClient()
    try:
        registered = client.get_model_version(config.model_name, str(config.model_version))
        training_run = client.get_run(registered.run_id or "")
    except MlflowException as exc:
        raise RuntimeError(
            f"the training run behind model {config.model_name!r} version "
            f"{config.model_version} cannot be read from the registry at "
            f"{tracking_uri(repo_root)}, so the service cannot confirm the model was "
            f"fitted under its own feature and cohort definitions; it refuses to "
            f"start without that ({exc})"
        ) from exc
    found = incompatible(training_run.data.params, SERVING_VERSIONS)
    if found:
        raise RuntimeError(describe(config.model_name, config.model_version, found))
