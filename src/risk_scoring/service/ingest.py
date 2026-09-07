"""One event in, state updated and a prediction logged if it earned one.

This is the whole scoring path as a plain function over a source of
connections and a loaded model, deliberately free of HTTP: the endpoint
is a thin wrapper around it, and the replay harness can drive the same
path directly.

Judgment calls this module fixes:

- A connection is held only while the database is in use: once for the
  state write, the log check, and the history read, and again for the log
  write. The cohort and feature build and the model call run with no
  connection held, so a slow score never keeps a pooled connection from
  another request. The caller hands over a way to get a connection rather
  than a connection, and each acquisition is bounded by the pool's timeout.
- Dropping the connection between the two halves opens a window in which
  another request can score the same discharge first. That is the case the
  log's uniqueness constraint already covers: the second write is dropped
  and this call reports it scored nothing.

- Only an encounter can be a scoring event, and only after the shared
  cohort rules admit it. Nothing here re-expresses those rules;
  ``serving.serving_features`` narrows the same functions the training
  pipeline calls, and a ``None`` from it means "state updated, nothing to
  score" for every reason at once (still open, wrong class, in-hospital
  death, under 18).
- A closed encounter whose patient has no demographics is refused before
  it is written. The state write commits on its own, so a refusal raised
  after it would answer 4xx for an event the service had kept, and a
  caller who believed the refusal would leave that discharge stored and
  never scored. Checking first costs one primary-key lookup, and a
  re-post after the demographics arrive stores and scores it as normal.
- The prediction log, not the state write, decides whether to score.
  State commits per event, so an encounter can be durable while its score
  is not; asking the log instead makes that window self-healing and costs
  one indexed lookup per encounter. It also means a discharge is never
  scored twice, no matter how often the stream replays it.
- History is read through ``serving.history_window``, the rows the
  discharge's features can see, rather than the patient's whole record.
  The cost of scoring one discharge is then bounded by the window, not
  by how many events the patient has accumulated.
- The score is computed from the same frame the training pipeline builds,
  cast the same way, so the model sees the columns its signature
  declares and the logged feature values are the model's actual input.
- The stored feature values are the model input columns only. The
  encounter and patient ids are not features; they have their own
  columns in the log.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg

from risk_scoring import predictions, serving, state
from risk_scoring.cohort import COHORT_VERSION
from risk_scoring.features import FEATURE_VERSION, MODEL_INPUT_COLUMNS
from risk_scoring.service.config import ServiceConfig

ConnectionSource = Callable[[], AbstractContextManager[psycopg.Connection[Any]]]
"""A way to borrow a connection for one block, such as ``pool.connection``."""


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
    connect: ConnectionSource,
    model: Any,
    config: ServiceConfig,
    event: state.AnyEvent,
    input_hash: str,
) -> IngestResult:
    """Persist one event and score it if it is an admitted discharge.

    ``connect`` is called once for the state write and history read and,
    when there is a score to log, once more for the log write; no
    connection is held while the model runs. Raises
    :class:`risk_scoring.state.EventConflictError` when the event
    contradicts one already stored,
    :class:`risk_scoring.state.PatientEventLimitError` when storing it
    would take its patient past ``config.max_events_per_patient``, and
    :class:`risk_scoring.serving.UnknownPatientError` when a discharge
    arrives before its patient's demographics. Neither refusal stores the
    event.
    """
    with connect() as conn:
        if isinstance(event, state.EncounterEvent) and event.stop != "":
            serving.require_demographics(
                state.has_patient(conn, event.patient), event.id, event.patient
            )
        stored = state.record_event(conn, event, max_patient_rows=config.max_events_per_patient)
        if not isinstance(event, state.EncounterEvent):
            return IngestResult(stored, *_NOT_SCORED)
        if predictions.has_prediction(conn, event.id):
            return IngestResult(stored, *_NOT_SCORED)

        window = serving.history_window(event)
        if window is None:
            # Still open: no discharge instant to read history against, and
            # serving_features would say the same of the same row.
            return IngestResult(stored, *_NOT_SCORED)
        history = state.patient_history(conn, event.patient, window)

    scoring_input = serving.serving_features(history, event.id)
    if scoring_input is None:
        return IngestResult(stored, *_NOT_SCORED)

    model_input = scoring_input.features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    score = float(np.asarray(model.predict(model_input), dtype=float).ravel()[0])
    record = predictions.PredictionRecord(
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
    )
    with connect() as conn:
        prediction_id = predictions.record_prediction(conn, record)
    if prediction_id is None:
        # Another writer scored this discharge between the check and the
        # insert. Theirs stands; this call logged nothing.
        return IngestResult(stored, *_NOT_SCORED)
    return IngestResult(stored, True, prediction_id, score)
