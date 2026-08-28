"""The provenance log: reading and writing one row per scored discharge.

Every prediction the service makes is recorded here with the model
version, feature-pipeline version, cohort version, input hash, and the
feature values the model actually saw. Provenance written at scoring
time is cheap; provenance reconstructed later is impossible, which is
why the feature values are stored rather than recomputed on demand.

Judgment calls this module fixes:

- The encounter id is the log's idempotency key, and this table is the
  authority on whether a discharge has been scored. The state tables
  commit per event, so a process can die after an encounter is durably
  stored but before its prediction is. Deciding "score it" by the
  absence of a prediction row, rather than by whether the state write
  was new, makes that window self-healing: the re-posted encounter is a
  no-op in state but still has no score, so it gets one.
- The first score for an encounter stands. A conflicting write is
  dropped, never merged and never overwritten, so a replay that revisits
  a discharge cannot rewrite history. :func:`record_prediction` reports
  that by returning None instead of raising, because a re-post is
  expected traffic during a resume, not an error.
- Each write commits on its own, matching the state layer: an
  acknowledged score is a durable score.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

_WRITE_COLUMNS = (
    "patient_id",
    "encounter_id",
    "event_time",
    "input_hash",
    "model_name",
    "model_version",
    "feature_version",
    "cohort_version",
    "score",
    "features",
)

_READ_COLUMNS = ("prediction_id", "scored_at", *_WRITE_COLUMNS)


@dataclass(frozen=True)
class PredictionRecord:
    """One scored discharge, as the service hands it to the log.

    ``event_time`` is the discharge instant in simulated time;
    ``scored_at`` is assigned by the database at write time and so is
    absent here. ``features`` maps feature name to the value the model
    was given, which is every column of the model input.
    """

    patient_id: str
    encounter_id: str
    event_time: datetime
    input_hash: str
    model_name: str
    model_version: int
    feature_version: str
    cohort_version: str
    score: float
    features: dict[str, float]


@dataclass(frozen=True)
class StoredPrediction(PredictionRecord):
    """A logged prediction as read back, with the fields the database assigned."""

    prediction_id: int
    scored_at: datetime


def record_prediction(conn: psycopg.Connection[Any], record: PredictionRecord) -> int | None:
    """Write one prediction and commit; None if this encounter already has one."""
    try:
        row = conn.execute(
            f"INSERT INTO predictions ({', '.join(_WRITE_COLUMNS)})"
            f" VALUES ({', '.join(['%s'] * len(_WRITE_COLUMNS))})"
            " ON CONFLICT (encounter_id) DO NOTHING"
            " RETURNING prediction_id",
            [
                record.patient_id,
                record.encounter_id,
                record.event_time,
                record.input_hash,
                record.model_name,
                record.model_version,
                record.feature_version,
                record.cohort_version,
                record.score,
                json.dumps(record.features),
            ],
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return None if row is None else int(row[0])


def has_prediction(conn: psycopg.Connection[Any], encounter_id: str) -> bool:
    """Whether this discharge has already been scored; read-only."""
    row = conn.execute(
        "SELECT 1 FROM predictions WHERE encounter_id = %s", [encounter_id]
    ).fetchone()
    return row is not None


def prediction_for_encounter(
    conn: psycopg.Connection[Any], encounter_id: str
) -> StoredPrediction | None:
    """The logged prediction for one discharge, or None if it was never scored."""
    row = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM predictions WHERE encounter_id = %s",
        [encounter_id],
    ).fetchone()
    if row is None:
        return None
    return _stored(row)


def all_predictions(conn: psycopg.Connection[Any]) -> list[StoredPrediction]:
    """The whole log, oldest write first; read-only.

    Ordering by ``prediction_id`` is ordering by write, since the sequence
    only ever advances. Callers that compare two runs of the same stream
    read the log this way and compare position by position.
    """
    rows = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM predictions ORDER BY prediction_id"
    ).fetchall()
    return [_stored(row) for row in rows]


def _stored(row: Sequence[Any]) -> StoredPrediction:
    return StoredPrediction(**dict(zip(_READ_COLUMNS, row, strict=True)))
