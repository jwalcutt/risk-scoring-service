"""What a monitoring window is compared against.

A reference is the registered model's own training window: the feature
rows it was fitted on, and the scores it produced on the patient-grouped
holdout. It is built once per registered version and stored, so the live
monitor, an offline audit, and Grafana all read one copy, and so a
monitor needs no data root at evaluation time.

The rebuild reads the training cutoff, split seed, and holdout fraction
off that version's own training run and then runs the cohort, label, and
feature modules over the frozen export, exactly as ``risk_scoring.gate``
rebuilds its holdout. Reusing ``train.grouped_split`` rather than
re-deriving a split is what keeps the reference from disagreeing with
what training actually did.

Two halves, drawn from different rows on purpose. The feature arrays
span the whole training window, which is the largest honest sample of
what the model was fitted on. The score array is the holdout only:
in-sample scores are optimistically shifted, and a reference built from
them would make live score drift read low.

Raw arrays are stored rather than summaries, so a statistic sees the
whole sample instead of an earlier guess about what mattered about it.
At roughly 12,000 rows by 14 columns that is under a couple of megabytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import psycopg
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from risk_scoring.cohort import COHORT_VERSION, build_cohort, filter_training_window
from risk_scoring.features import FEATURE_VERSION, MODEL_INPUT_COLUMNS, build_features
from risk_scoring.populations import load_population
from risk_scoring.tracking import configure_tracking
from risk_scoring.train import MODEL_NAME, grouped_split

_WRITE_COLUMNS = (
    "model_name",
    "model_version",
    "feature_version",
    "cohort_version",
    "training_cutoff",
    "split_seed",
    "n_train_rows",
    "n_holdout_rows",
    "features",
    "scores",
)

_READ_COLUMNS = ("reference_id", *_WRITE_COLUMNS, "created_at")


class ReferenceExistsError(RuntimeError):
    """A reference for this model version is already stored."""


@dataclass(frozen=True)
class Reference:
    """The training window a model version was fitted on, as arrays."""

    model_name: str
    model_version: int
    feature_version: str
    cohort_version: str
    training_cutoff: datetime
    split_seed: int
    n_train_rows: int
    n_holdout_rows: int
    features: dict[str, list[float]]
    scores: list[float]


@dataclass(frozen=True)
class StoredReference(Reference):
    """A reference as read back, carrying the id the database assigned."""

    reference_id: int
    created_at: datetime


def build_reference(csv_dir: Path, repo_root: Path, *, model_version: int) -> Reference:
    """Rebuild one registered version's training window from the frozen export."""
    configure_tracking(repo_root)
    client = MlflowClient()
    try:
        registered = client.get_model_version(MODEL_NAME, str(model_version))
        training_run = client.get_run(registered.run_id or "")
    except MlflowException as exc:
        raise LookupError(
            f"model {MODEL_NAME!r} version {model_version} is not in the registry, or its"
            f" training run cannot be read; register it before building a reference ({exc})"
        ) from exc

    params = training_run.data.params
    cutoff = pd.Timestamp(params["training_cutoff"], tz="UTC")
    seed = int(params["split_seed"])
    holdout_fraction = float(params["holdout_fraction"])

    frames = load_population(csv_dir)
    encounters, patients = frames["encounters"], frames["patients"]
    cohort = filter_training_window(build_cohort(encounters, patients).frame, cutoff)
    features = build_features(cohort, encounters, frames["medications"], frames["conditions"])
    x = features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    _, holdout_idx = grouped_split(features["patient_id"], holdout_fraction, seed)

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version}")
    scores = np.asarray(model.predict(x.iloc[holdout_idx]), dtype=float)

    return Reference(
        model_name=MODEL_NAME,
        model_version=model_version,
        feature_version=params.get("feature_version", FEATURE_VERSION),
        cohort_version=params.get("cohort_version", COHORT_VERSION),
        training_cutoff=cutoff.to_pydatetime(),
        split_seed=seed,
        n_train_rows=len(x),
        n_holdout_rows=len(holdout_idx),
        features={column: x[column].tolist() for column in MODEL_INPUT_COLUMNS},
        scores=scores.tolist(),
    )


def record_reference(conn: psycopg.Connection[Any], reference: Reference) -> int:
    """Store one reference and commit; refuse a version that already has one."""
    try:
        row = conn.execute(
            f"INSERT INTO monitoring_reference ({', '.join(_WRITE_COLUMNS)})"
            f" VALUES ({', '.join(['%s'] * len(_WRITE_COLUMNS))})"
            " ON CONFLICT (model_name, model_version) DO NOTHING"
            " RETURNING reference_id",
            [
                reference.model_name,
                reference.model_version,
                reference.feature_version,
                reference.cohort_version,
                reference.training_cutoff,
                reference.split_seed,
                reference.n_train_rows,
                reference.n_holdout_rows,
                # allow_nan=False so a NaN raises here, where the value has a
                # name, rather than as a jsonb parse error at the INSERT.
                json.dumps(reference.features, allow_nan=False),
                json.dumps(reference.scores, allow_nan=False),
            ],
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if row is None:
        raise ReferenceExistsError(
            f"model {reference.model_name!r} version {reference.model_version} already has a"
            f" reference; a promotion writes a row for the new version rather than editing one"
        )
    return int(row[0])


def read_reference(
    conn: psycopg.Connection[Any], model_name: str, model_version: int
) -> StoredReference | None:
    """The stored reference for one registered version, or None."""
    row = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM monitoring_reference"
        " WHERE model_name = %s AND model_version = %s",
        [model_name, model_version],
    ).fetchone()
    if row is None:
        return None
    values = dict(zip(_READ_COLUMNS, row, strict=True))
    return StoredReference(**values)


def describe(reference: Reference, reference_id: int) -> str:
    """One block naming what was stored, for the command's output."""
    return "\n".join(
        [
            f"reference:          {reference_id}",
            f"model:              {reference.model_name} version {reference.model_version}",
            f"versions:           cohort {reference.cohort_version},"
            f" features {reference.feature_version}",
            f"training cutoff:    {reference.training_cutoff.date().isoformat()}"
            f" (STOP strictly before)",
            f"split seed:         {reference.split_seed}",
            f"training rows:      {reference.n_train_rows} over"
            f" {len(reference.features)} feature columns",
            f"holdout scores:     {reference.n_holdout_rows}",
        ]
    )
