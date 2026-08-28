"""One event in, state updated and a prediction logged if it earned one.

This is the whole scoring path as a plain function over a connection and
a loaded model, deliberately free of HTTP: the endpoint is a thin wrapper
around it, and the replay harness can drive the same path directly.

Judgment calls this module fixes:

- Only an encounter can be a scoring event, and only after the shared
  cohort rules admit it. Nothing here re-expresses those rules;
  ``serving.serving_features`` narrows the same functions the training
  pipeline calls, and a ``None`` from it means "state updated, nothing to
  score" for every reason at once (still open, wrong class, in-hospital
  death, under 18).
- The prediction log, not the state write, decides whether to score.
  State commits per event, so an encounter can be durable while its score
  is not; asking the log instead makes that window self-healing and costs
  one indexed lookup per encounter. It also means a discharge is never
  scored twice, no matter how often the stream replays it.
- The score is computed from the same frame the training pipeline builds,
  cast the same way, so the model sees the columns its signature
  declares and the logged feature values are the model's actual input.
- The stored feature values are the model input columns only. The
  encounter and patient ids are not features; they have their own
  columns in the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg

from risk_scoring import predictions, serving, state
from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_COLUMNS, FEATURE_VERSION
from risk_scoring.service.config import ServiceConfig

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS[2:]


@dataclass(frozen=True)
class IngestResult:
    """What one ingested event did."""

    stored: bool
    """True if the event was new to state; False on an identical re-post."""

    scored: bool
    """True if this call wrote a prediction row."""

    prediction_id: int | None
    score: float | None


_NOT_SCORED = (False, None, None)


def ingest_event(
    conn: psycopg.Connection[Any],
    model: Any,
    config: ServiceConfig,
    event: state.AnyEvent,
    input_hash: str,
) -> IngestResult:
    """Persist one event and score it if it is an admitted discharge.

    Raises :class:`risk_scoring.state.EventConflictError` when the event
    contradicts one already stored, and
    :class:`risk_scoring.serving.UnknownPatientError` when a discharge
    arrives before its patient's demographics.
    """
    stored = state.record_event(conn, event)
    if not isinstance(event, state.EncounterEvent):
        return IngestResult(stored, *_NOT_SCORED)
    if predictions.has_prediction(conn, event.id):
        return IngestResult(stored, *_NOT_SCORED)

    history = state.patient_history(conn, event.patient)
    scoring_input = serving.serving_features(history, event.id)
    if scoring_input is None:
        return IngestResult(stored, *_NOT_SCORED)

    model_input = scoring_input.features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    score = float(np.asarray(model.predict(model_input), dtype=float).ravel()[0])
    prediction_id = predictions.record_prediction(
        conn,
        predictions.PredictionRecord(
            patient_id=event.patient,
            encounter_id=event.id,
            event_time=state.parse_timestamp(event.stop),
            input_hash=input_hash,
            model_name=config.model_name,
            model_version=config.model_version,
            feature_version=FEATURE_VERSION,
            cohort_version=COHORT_VERSION,
            score=score,
            features={name: float(model_input.iloc[0][name]) for name in MODEL_INPUT_COLUMNS},
        ),
    )
    if prediction_id is None:
        # Another writer scored this discharge between the check and the
        # insert. Theirs stands; this call logged nothing.
        return IngestResult(stored, *_NOT_SCORED)
    return IngestResult(stored, True, prediction_id, score)
