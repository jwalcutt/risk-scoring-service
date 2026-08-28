"""Sampling a frozen population down to a runnable size.

The frozen populations are local-only, verified by checksum manifest, so
every check that runs against them is a script rather than a test. Both
of those checks read their population with
``risk_scoring.populations.load_population`` and then narrow it here.
"""

from __future__ import annotations

import pandas as pd

from risk_scoring.cohort import build_cohort


def sample_patients(
    frames: dict[str, pd.DataFrame], *, count: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Restrict every frame to a seeded sample of patients with a cohort discharge.

    Sampling from cohort patients keeps the run informative: a sample of
    all patients would be mostly people the service never scores. It is
    also exact rather than approximate, because every feature reads only
    the scored patient's own rows, so a patient's features do not depend
    on which other patients share the frame.
    """
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    eligible = pd.Index(cohort["patient_id"].unique())
    chosen = set(
        eligible.to_series().sample(n=min(count, len(eligible)), random_state=seed).tolist()
    )
    return {
        "patients": frames["patients"].loc[frames["patients"]["Id"].isin(chosen)],
        "encounters": frames["encounters"].loc[frames["encounters"]["PATIENT"].isin(chosen)],
        "medications": frames["medications"].loc[frames["medications"]["PATIENT"].isin(chosen)],
        "conditions": frames["conditions"].loc[frames["conditions"]["PATIENT"].isin(chosen)],
    }
