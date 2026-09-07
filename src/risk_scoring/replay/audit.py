"""Checks a replay's tables must pass, stated over any database and export.

The replay's exit criteria are properties of what a run leaves behind:
the join of the prediction log and the labels table reconstructs realized
30-day performance, no label is released before its discharge is 30
simulated days old, and every released label is the label the batch
pipeline computes for that discharge. The CI tests prove them over a
synthetic replay; this module states them once so the same checks run
against a real database after a real run, through
``scripts/check_replay_run.py``.

Judgment calls this module fixes:

- The batch side loads the model version the log names, never the one
  the service is pinned to, and refuses a log that names more than one.
  Loading the pin would let predictions from another model pass, which is
  the provenance rule restated.
- ``batch_performance`` returns the same dataclass as
  ``realized_performance`` with the same ``None`` rules, so the comparison
  is equality with no tolerance. A tolerance would hide a join bug.
- The label audit walks the released labels, not the export: a discharge
  the harness never labelled is the maturation boundary at work, not a
  disagreement, and a released label the export cannot account for is.
- Nothing here reads a spliced-in export. A variant's ids are rewritten
  at load, so its rows would have to be read through ``populations.rekeyed``
  before they could meet the log; the script refuses a spliced config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import roc_auc_score

from risk_scoring import label_log
from risk_scoring.cohort import build_cohort
from risk_scoring.features import MODEL_INPUT_COLUMNS, build_features
from risk_scoring.labels import READMISSION_WINDOW_DAYS, build_labels
from risk_scoring.replay.realized import RealizedPerformance
from risk_scoring.tracking import configure_tracking


class ManyModelsError(RuntimeError):
    """The prediction log names more than one model version."""


@dataclass(frozen=True)
class LoggedModel:
    """The one registered model version the prediction log names."""

    name: str
    version: int


@dataclass(frozen=True)
class LabelAudit:
    """Released labels checked against the export, and the ones that disagreed."""

    checked: int
    disagreements: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.disagreements


def logged_model(conn: psycopg.Connection[Any]) -> LoggedModel:
    """The model version every logged prediction names; read-only.

    Raises ``LookupError`` on an empty log and ``ManyModelsError`` when the
    log names more than one version, since the batch side can only load
    one and must load the one the rows name.
    """
    rows = conn.execute(
        "SELECT DISTINCT model_name, model_version FROM predictions ORDER BY 1, 2"
    ).fetchall()
    if not rows:
        raise LookupError("the prediction log is empty; nothing names a model")
    if len(rows) > 1:
        named = ", ".join(f"{name} v{version}" for name, version in rows)
        raise ManyModelsError(f"the prediction log names {len(rows)} model versions: {named}")
    (name, version), *_ = rows
    return LoggedModel(name=str(name), version=int(version))


def early_labels(conn: psycopg.Connection[Any]) -> int:
    """How many labels were released inside their discharge's readmission window.

    The exit criterion in one query: it must return zero. The labels table
    checks ``released_at >= due_at``, but ``due_at`` is what the harness
    wrote, so this compares against the prediction's own discharge instant.
    """
    row = conn.execute(
        "SELECT count(*) FROM labels AS l JOIN predictions AS p USING (prediction_id)"
        " WHERE l.released_at < p.event_time + %s",
        [timedelta(days=READMISSION_WINDOW_DAYS)],
    ).fetchone()
    assert row is not None
    return int(row[0])


def batch_labels(frames: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    """Every cohort discharge's label from the export, by encounter id."""
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    labelled = build_labels(cohort, frames["encounters"])
    return dict(zip(labelled["encounter_id"], labelled["label"].astype(int), strict=True))


def label_audit(conn: psycopg.Connection[Any], frames: Mapping[str, pd.DataFrame]) -> LabelAudit:
    """Compare every released label to the batch label for its discharge; read-only."""
    expected = batch_labels(frames)
    released = label_log.all_labels(conn)
    disagreements = tuple(
        row.encounter_id for row in released if expected.get(row.encounter_id) != row.label
    )
    return LabelAudit(checked=len(released), disagreements=disagreements)


def batch_performance(
    frames: Mapping[str, pd.DataFrame],
    model: LoggedModel,
    start: datetime,
    end: datetime,
    *,
    repo_root: Path,
) -> RealizedPerformance:
    """Count, prevalence, and AUROC from the modules training uses, over one window.

    The window is half-open on the discharge instant, as the join's is.
    ``repo_root`` is where the registry lives; configuring tracking there
    is process-global, as everywhere else the registry is read.
    """
    if end <= start:
        raise ValueError(f"the window's start must be before its end; got {start} and {end}")
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    stop = cohort["stop"]
    window = cohort.loc[(stop >= start) & (stop < end)].reset_index(drop=True)
    if window.empty:
        return RealizedPerformance(count=0, prevalence=None, auroc=None)
    y = build_labels(window, frames["encounters"])["label"].to_numpy(dtype=float)
    features = build_features(
        window, frames["encounters"], frames["medications"], frames["conditions"]
    )
    x = features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    configure_tracking(repo_root)
    loaded: Any = mlflow.pyfunc.load_model(f"models:/{model.name}/{model.version}")
    scores = np.asarray(loaded.predict(x), dtype=float)
    auroc = float(roc_auc_score(y, scores)) if 0.0 < y.mean() < 1.0 else None
    return RealizedPerformance(count=len(y), prevalence=float(y.mean()), auroc=auroc)
