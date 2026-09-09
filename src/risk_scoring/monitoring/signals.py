"""The signal set a monitoring evaluation reports on, and its shared names.

A signal is one named thing a window is judged on. Fourteen of them are
the model's own input columns, split by what kind of value they hold, one
is the score the model produced, and the rest are operational facts about
the window rather than distributions: how many predictions it held, how
many events the service refused, and how many predictions came from a
model or feature version other than the reference's.

Judgment calls this module fixes:

- The feature signals are derived from ``features.MODEL_INPUT_COLUMNS``
  and ``features.FLAG_CODES`` rather than restated, so a column added to
  the feature pipeline cannot leave a signal unnamed or unjudged. The
  order follows the feature frame's, not a dict's insertion order, so the
  tuples are stable whatever the flag definitions do.
- ``features.FLAG_CODES`` is the source of which columns are flags, not
  ``gate.FLAG_COLUMNS``, which restates the same seven names. Reading the
  lighter module also keeps mlflow and scikit-learn off this import path.
- The score's population stability index is a signal name of its own and
  is deliberately absent from :data:`ALERTABLE_SIGNALS`. It is a
  threshold-free number for the dashboard: it has no p-value, and the
  conventional 0.1 and 0.25 bands are folklore rather than a test.
"""

from __future__ import annotations

from risk_scoring.features import FLAG_CODES, MODEL_INPUT_COLUMNS

CONTINUOUS_SIGNALS: tuple[str, ...] = tuple(
    column for column in MODEL_INPUT_COLUMNS if column not in FLAG_CODES
)
"""Feature columns holding a continuous or count value, compared by a KS test."""

FLAG_SIGNALS: tuple[str, ...] = tuple(
    column for column in MODEL_INPUT_COLUMNS if column in FLAG_CODES
)
"""Binary feature columns, compared by a two-proportion test."""

SCORE_SIGNAL = "score"
"""The model's output, compared by the same KS test as a continuous feature."""

SCORE_PSI_SIGNAL = "score_psi"
"""Population stability index on the score. Reported, never judged."""

VOLUME_SIGNAL = "volume"
"""Predictions in the window against the rate the window's length implies."""

REFUSAL_SIGNAL = "refusals"
"""Events the service refused, counted by their stream instant."""

VERSION_MISMATCH_SIGNAL = "version_mismatch"
"""Predictions whose model or feature version is not the reference's."""

DRIFT_SIGNALS: tuple[str, ...] = (*CONTINUOUS_SIGNALS, *FLAG_SIGNALS, SCORE_SIGNAL)
"""Every signal carrying a p-value, and so every signal a p-value floor covers."""

COUNT_SIGNALS: tuple[str, ...] = (VOLUME_SIGNAL, REFUSAL_SIGNAL, VERSION_MISMATCH_SIGNAL)
"""Signals judged on a count rather than on a distribution."""

ALERTABLE_SIGNALS: tuple[str, ...] = (*DRIFT_SIGNALS, *COUNT_SIGNALS)
"""Every signal that can raise an alert. The score's PSI is not among them."""

REPORTED_SIGNALS: tuple[str, ...] = (*ALERTABLE_SIGNALS, SCORE_PSI_SIGNAL)
"""Every signal an evaluation stores a statistic for."""
