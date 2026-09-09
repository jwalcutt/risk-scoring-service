"""The signal set, restated literally.

These assertions duplicate what the module derives, on purpose. A feature
column added to the pipeline, or a flag moved between the two kinds,
changes what every threshold covers and what the multiple-comparison
arithmetic counts, so it has to be a deliberate edit in two places rather
than a silent widening of the signal set.

The rules these tests pin:

- Fourteen feature signals, split seven continuous and seven flags.
- Fifteen signals carry a p-value; eighteen can raise an alert.
- The score's population stability index is reported and never alertable.
- Every feature column is a signal, and no signal name is used twice.
"""

from __future__ import annotations

from risk_scoring.features import MODEL_INPUT_COLUMNS
from risk_scoring.monitoring import signals

EXPECTED_CONTINUOUS = (
    "age_at_discharge",
    "los_days",
    "prior_inpatient_180d",
    "days_since_prev_discharge",
    "prior_ed_180d",
    "active_medication_count",
    "active_disorder_count",
)

EXPECTED_FLAGS = (
    "flag_chf",
    "flag_chronic_pulmonary",
    "flag_dementia",
    "flag_diabetes",
    "flag_malignancy",
    "flag_mi",
    "flag_renal_disease",
)


def test_continuous_signals_are_the_seven_value_columns() -> None:
    assert signals.CONTINUOUS_SIGNALS == EXPECTED_CONTINUOUS


def test_flag_signals_are_the_seven_binary_columns() -> None:
    assert signals.FLAG_SIGNALS == EXPECTED_FLAGS


def test_every_feature_column_is_a_signal() -> None:
    """A column the pipeline computes and nothing monitors would drift unseen."""
    covered = set(signals.CONTINUOUS_SIGNALS) | set(signals.FLAG_SIGNALS)
    assert covered == set(MODEL_INPUT_COLUMNS)


def test_drift_signals_are_the_fourteen_features_plus_the_score() -> None:
    """Fifteen is the number the multiple-comparison arithmetic counts."""
    assert len(signals.DRIFT_SIGNALS) == 15
    assert signals.DRIFT_SIGNALS[-1] == signals.SCORE_SIGNAL


def test_alertable_signals_add_the_three_counts() -> None:
    assert signals.COUNT_SIGNALS == ("volume", "refusals", "version_mismatch")
    assert len(signals.ALERTABLE_SIGNALS) == 18


def test_score_psi_is_reported_but_never_alertable() -> None:
    """It has no p-value, and its conventional bands are folklore, not a test."""
    assert signals.SCORE_PSI_SIGNAL in signals.REPORTED_SIGNALS
    assert signals.SCORE_PSI_SIGNAL not in signals.ALERTABLE_SIGNALS


def test_no_signal_name_is_used_twice() -> None:
    assert len(set(signals.REPORTED_SIGNALS)) == len(signals.REPORTED_SIGNALS)
