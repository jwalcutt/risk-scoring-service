"""Reading and sampling a frozen population, shared by the check scripts.

The frozen populations are local-only, verified by checksum manifest, so
every check that runs against them is a script rather than a test. Both
of those checks need the same two things: the CSVs read exactly as the
training pipeline reads them, and a seeded patient sample that is worth
running against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from risk_scoring.cohort import build_cohort  # noqa: E402

POPULATION_FRAMES = ("patients", "encounters", "medications", "conditions")


def load_population(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Read a population's CSVs exactly as the training pipeline reads them."""
    return {
        name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in POPULATION_FRAMES
    }


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
