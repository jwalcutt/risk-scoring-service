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
- ``released_by`` bounds the join by the instant a label became
  available, which is what lets a monitoring evaluation be a pure
  function of its boundary: without it, the same window read later would
  pick up labels that had not been released yet when the boundary passed,
  and a run evaluated live and evaluated afterwards would disagree. It is
  a bound on this join rather than a second join in the monitoring code,
  because the ``None`` rules above have to hold identically either way.
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
    conn: psycopg.Connection[Any],
    start: datetime,
    end: datetime,
    *,
    released_by: datetime | None = None,
) -> RealizedPerformance:
    """Count, prevalence, and AUROC over the labelled discharges in ``[start, end)``.

    ``released_by`` restricts the join to labels released at or before that
    simulated instant, so a caller asking what was known at a boundary gets
    the same answer whenever it asks.
    """
    if end <= start:
        raise ValueError(f"the window's start must be before its end; got {start} and {end}")
    released_clause = "" if released_by is None else " AND l.released_at <= %s"
    parameters: list[datetime] = [start, end]
    if released_by is not None:
        parameters.append(released_by)
    rows = conn.execute(
        "SELECT p.score, l.label FROM predictions AS p JOIN labels AS l USING (prediction_id)"
        f" WHERE p.event_time >= %s AND p.event_time < %s{released_clause}"
        " ORDER BY p.prediction_id",
        parameters,
    ).fetchall()
    if not rows:
        return RealizedPerformance(count=0, prevalence=None, auroc=None)
    scores = np.asarray([score for score, _ in rows], dtype=float)
    y = np.asarray([label for _, label in rows], dtype=float)
    auroc = float(roc_auc_score(y, scores)) if 0.0 < y.mean() < 1.0 else None
    return RealizedPerformance(count=len(rows), prevalence=float(y.mean()), auroc=auroc)
