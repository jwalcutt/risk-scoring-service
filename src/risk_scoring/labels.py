"""30-day readmission labels for cohort discharges.

Each cohort row (an adult inpatient discharge) is labeled 1 when the same
patient has a qualifying inpatient readmission within 30 days of the
discharge, and 0 otherwise. Labels are derived from the raw encounters
frame, not from the cohort, so candidate stays are judged before any
cohort exclusion applies to them.

Judgment calls this module fixes:

- A readmission is any inpatient encounter for the same patient whose
  START is strictly after the index discharge's STOP and at or before
  STOP plus 30 days. Equality at exactly 30 days counts, matching the
  feature module's inclusive far edge.
- An inpatient encounter whose START is at or before the index STOP is a
  continuation or transfer of the same episode, never a readmission,
  regardless of how that stay ends.
- Candidate readmissions come from the raw encounters frame: a
  readmission stay that ends in death never becomes a cohort row, yet it
  still labels the index discharge 1.
- Death within 30 days without a qualifying readmission labels 0. The
  model predicts readmission, not death; the competing risk is accepted
  and recorded here rather than modeled, and the module never reads
  patient records.
- Candidate starts are parsed under ``%Y-%m-%dT%H:%M:%SZ``, the exact
  format the Synthea export writes. Pandas would otherwise guess a
  format from the first element and fall back to per-element dateutil
  parsing when the guess fails, so whether a malformed value is rejected
  or quietly reinterpreted would depend on the first row. A start read
  under a different reading of the same digits moves an encounter across
  the 30-day boundary, flipping a label.
- This module is the only source of training labels. The crude rate in
  ``risk_scoring.datagen.sanity`` is a data-adequacy check and
  deliberately shares no code with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_VERSION = "1.0.0"

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

READMISSION_WINDOW_DAYS = 30

LABEL_COLUMNS = ("encounter_id", "label")


def build_labels(cohort: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    """Label each cohort discharge for 30-day inpatient readmission.

    ``cohort`` is a frame of cohort rows (``encounter_id``, ``patient_id``,
    ``stop`` are used); ``encounters`` is the raw Synthea encounters frame
    with string columns. Returns one row per cohort row, in cohort order,
    with columns ``LABEL_COLUMNS`` and an integer ``label``.
    """
    inpatient = encounters.loc[encounters["ENCOUNTERCLASS"] == "inpatient"]
    candidates = pd.DataFrame(
        {
            "candidate_id": inpatient["Id"],
            "patient_id": inpatient["PATIENT"],
            "candidate_start": pd.to_datetime(
                inpatient["START"], format=TIMESTAMP_FORMAT, utc=True
            ),
        }
    )

    index_rows = cohort.loc[:, ["encounter_id", "patient_id", "stop"]].reset_index(drop=True)
    merged = index_rows.merge(candidates, on="patient_id", how="left")

    window = pd.Timedelta(np.timedelta64(READMISSION_WINDOW_DAYS, "D"))
    is_readmission = (
        (merged["candidate_id"] != merged["encounter_id"])
        & (merged["candidate_start"] > merged["stop"])
        & (merged["candidate_start"] <= merged["stop"] + window)
    )
    labeled = (
        is_readmission.groupby(merged["encounter_id"])
        .any()
        .reindex(index_rows["encounter_id"], fill_value=False)
    )

    return pd.DataFrame(
        {
            "encounter_id": index_rows["encounter_id"],
            "label": labeled.astype(int).to_numpy(),
        }
    )
