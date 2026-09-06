"""Realized performance: what the prediction log and the labels table say together.

The exit criterion for the replay is that the join of the two tables
reconstructs realized 30-day performance. This module is that join over
a simulated window, returning the same three numbers the batch pipeline
reports over a set of discharges, so the two can be compared exactly.

Judgment calls this module fixes:

- The window is over the discharge instant (``event_time`` on the log),
  half-open at the end, which is how the training cutoff and the
  monitoring windows are stated everywhere else.
- A scored discharge whose label has not been released is not in the
  window's population. Realized performance describes what is known,
  and the maturation boundary means the last 30 days of a run are never
  known; the caller who wants a count of the unlabelled reads the log.
- A window that cannot support a metric reports ``None`` for it rather
  than raising: prevalence over nothing, AUROC over one class. A
  monitoring job reads early windows on every evaluation, and an
  exception there would be an error made of a fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RealizedPerformance:
    """The labelled discharges in a window, and how the scores did on them."""

    count: int
    prevalence: float | None
    auroc: float | None


def realized_performance(
    conn: psycopg.Connection[Any], start: datetime, end: datetime
) -> RealizedPerformance:
    """Count, prevalence, and AUROC over the labelled discharges in ``[start, end)``."""
    if end <= start:
        raise ValueError(f"the window's start must be before its end; got {start} and {end}")
    rows = conn.execute(
        "SELECT p.score, l.label FROM predictions AS p JOIN labels AS l USING (prediction_id)"
        " WHERE p.event_time >= %s AND p.event_time < %s ORDER BY p.prediction_id",
        [start, end],
    ).fetchall()
    if not rows:
        return RealizedPerformance(count=0, prevalence=None, auroc=None)
    scores = np.asarray([score for score, _ in rows], dtype=float)
    y = np.asarray([label for _, label in rows], dtype=float)
    auroc = float(roc_auc_score(y, scores)) if 0.0 < y.mean() < 1.0 else None
    return RealizedPerformance(count=len(rows), prevalence=float(y.mean()), auroc=auroc)
