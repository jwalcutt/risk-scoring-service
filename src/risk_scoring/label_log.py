"""The labels table: reading and writing one released label per scored discharge.

A label is ground truth the replay harness already holds and releases
30 simulated days after the discharge. This module is the table's only
writer; what a label is and when it falls due are decided elsewhere.

Judgment calls this module fixes:

- A label is attached by encounter id and the write looks the prediction
  up itself, in the insert. The caller never handles a prediction id, so
  a label cannot be pinned to the wrong row by a mismatched pair.
- One label per prediction, enforced by the table. A second write for
  the same discharge adds nothing and returns None rather than raising,
  because a harness resuming from a checkpoint re-releases the labels
  of the tick that was never checkpointed, exactly as it re-posts that
  tick's events.
- A label for a discharge that was never scored is a defect, not a
  no-op. The schedule is built by the same cohort module the service
  scores with, so a missing prediction means the two disagree on the
  cohort, and :class:`UnscoredDischargeError` says so loudly.
- Each write commits on its own, matching the log and the run row: a
  released label is a durable label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

from risk_scoring.predictions import has_prediction

_WRITE_COLUMNS = (
    "prediction_id",
    "encounter_id",
    "label",
    "label_version",
    "due_at",
    "released_at",
)

_READ_COLUMNS = ("label_id", "recorded_at", *_WRITE_COLUMNS)


class UnscoredDischargeError(LookupError):
    """A label fell due for a discharge the log holds no prediction for."""


@dataclass(frozen=True)
class LabelRecord:
    """One label as the harness releases it.

    ``due_at`` is the discharge instant plus the readmission window;
    ``released_at`` is the simulated instant of the tick that released
    it. Both are simulated time. The prediction id and the wall clock are
    assigned by the write and so are absent here.
    """

    encounter_id: str
    label: int
    label_version: str
    due_at: datetime
    released_at: datetime


@dataclass(frozen=True)
class StoredLabel(LabelRecord):
    """A released label as read back, with the fields the database assigned."""

    label_id: int
    recorded_at: datetime
    prediction_id: int


def record_label(conn: psycopg.Connection[Any], record: LabelRecord) -> int | None:
    """Write one label against its prediction and commit; None if it already has one.

    Raises :class:`UnscoredDischargeError` when the discharge has no
    prediction to attach to, leaving nothing written.
    """
    try:
        row = conn.execute(
            f"INSERT INTO labels ({', '.join(_WRITE_COLUMNS)})"
            " SELECT prediction_id, %s, %s, %s, %s, %s FROM predictions WHERE encounter_id = %s"
            " ON CONFLICT (prediction_id) DO NOTHING"
            " RETURNING label_id",
            [
                record.encounter_id,
                record.label,
                record.label_version,
                record.due_at,
                record.released_at,
                record.encounter_id,
            ],
        ).fetchone()
        if row is None and not has_prediction(conn, record.encounter_id):
            raise UnscoredDischargeError(
                f"no prediction to attach a label to for encounter {record.encounter_id}"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return None if row is None else int(row[0])


def all_labels(conn: psycopg.Connection[Any]) -> list[StoredLabel]:
    """Every released label, oldest write first; read-only.

    Ordering by ``label_id`` is ordering by release, since the sequence
    only ever advances. Callers that compare two runs of one stream read
    the table this way and compare position by position.
    """
    rows = conn.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM labels ORDER BY label_id"
    ).fetchall()
    return [_stored(row) for row in rows]


def _stored(row: Sequence[Any]) -> StoredLabel:
    return StoredLabel(**dict(zip(_READ_COLUMNS, row, strict=True)))
