"""Confirm the training-serving skew check on a frozen population.

The skew test in CI runs on a small synthetic population built to hit
every feature boundary. This script runs the same comparison against real
generated data, which is local-only (verified by checksum manifests) and
therefore cannot run in CI:

    python scripts/check_serving_skew.py --population baseline --patients 500

A seeded sample of patients is ingested event by event into a throwaway
database and every cohort discharge is scored on arrival, exactly as the
service will. The result is compared to the batch pipeline's features for
the same encounters, with exact equality.

Sampling by patient is exact, not approximate: every feature reads only
the scored patient's own rows, so the batch pipeline's output for a
patient is the same whether or not other patients are in the frame. Full
population runs are possible but pointless at per-event commit cost.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

from risk_scoring import db as db_module
from risk_scoring import serving, state
from risk_scoring.cohort import build_cohort
from risk_scoring.features import FEATURE_COLUMNS, build_features
from risk_scoring.populations import load_population
from risk_scoring.sampling import sample_patients
from risk_scoring.stream import ordered_events

REPO_ROOT = Path(__file__).resolve().parent.parent


def replay(conn: psycopg.Connection[Any], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ingest the stream, scoring each discharge the cohort rules admit."""
    for _, demographics in frames["patients"].iterrows():
        state.record_patient(conn, state.PatientEvent.from_row(dict(demographics)))

    scored: list[pd.DataFrame] = []
    stream = ordered_events(frames["encounters"], frames["medications"], frames["conditions"])
    for event in stream:
        if event.kind == "medication":
            state.record_medication(conn, state.MedicationEvent.from_row(event.row))
            continue
        if event.kind == "condition":
            state.record_condition(conn, state.ConditionEvent.from_row(event.row))
            continue
        state.record_encounter(conn, state.EncounterEvent.from_row(event.row))
        result = serving.serving_features(
            state.patient_history(conn, event.row["PATIENT"]), event.row["Id"]
        )
        if result is not None:
            scored.append(result.features)

    if not scored:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    return pd.concat(scored, ignore_index=True)


def batch_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cohort = build_cohort(frames["encounters"], frames["patients"]).frame
    return build_features(cohort, frames["encounters"], frames["medications"], frames["conditions"])


def _by_encounter(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("encounter_id").reset_index(drop=True)


def compare(frames: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    """Run both paths against a throwaway database and report the verdict."""
    admin_url = db_module.database_url()
    database = f"skew_check_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    url = psycopg.conninfo.make_conninfo(admin_url, dbname=database)
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            db_module.migrate(conn)
            served = _by_encounter(replay(conn, frames))
    finally:
        with psycopg.connect(admin_url, connect_timeout=5, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))

    expected = _by_encounter(batch_features(frames))
    if served["encounter_id"].tolist() != expected["encounter_id"].tolist():
        missing = set(expected["encounter_id"]) - set(served["encounter_id"])
        extra = set(served["encounter_id"]) - set(expected["encounter_id"])
        return False, f"scored set differs; missing {sorted(missing)}, extra {sorted(extra)}"
    try:
        pd.testing.assert_frame_equal(served, expected, check_exact=True)
    except AssertionError as mismatch:
        return False, str(mismatch)
    return True, f"{len(served)} discharges match exactly"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_serving_skew.py",
        description="Compare serving-time features to batch features on a frozen population.",
    )
    parser.add_argument("--population", default="baseline")
    parser.add_argument("--patients", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args(argv)

    csv_dir = REPO_ROOT / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")

    frames = sample_patients(load_population(csv_dir), count=args.patients, seed=args.seed)
    counts = {name: len(frame) for name, frame in frames.items()}
    print(f"population {args.population}, {counts['patients']} patients (seed {args.seed})")
    print(
        f"events to ingest: {counts['encounters']} encounters, "
        f"{counts['medications']} medications, {counts['conditions']} conditions"
    )

    matched, detail = compare(frames)
    print(("MATCH: " if matched else "SKEW: ") + detail)
    if not matched:
        sys.exit(1)


if __name__ == "__main__":
    main()
