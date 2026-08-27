"""Rough sanity statistics over raw Synthea CSV output.

These numbers gate the freeze decision for a generated population: enough
inpatient volume, a nonzero readmission base rate, a mostly-adult population,
and replay-orderable timestamps. The readmission rate here is deliberately
crude (any inpatient encounter followed by another inpatient encounter for the
same patient within 30 days). The real cohort definition is separate,
versioned code and is not derived from this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


def is_iso8601(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SanityStats:
    total_patients: int
    total_encounters: int
    inpatient_encounters: int
    crude_readmission_rate: float
    adult_share: float
    first_start: str
    last_stop: str
    invalid_timestamp_rows: int


def _adult_share(patients_csv: Path, as_of: date) -> tuple[int, float]:
    total = 0
    adults = 0
    with patients_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            birth = date.fromisoformat(row["BIRTHDATE"])
            age = (as_of - birth).days / 365.25
            if age >= 18:
                adults += 1
    return total, adults / total if total else 0.0


def compute_sanity_stats(csv_dir: Path, as_of: date) -> SanityStats:
    total_patients, adult_share = _adult_share(csv_dir / "patients.csv", as_of)

    total_encounters = 0
    invalid_rows = 0
    first_start = ""
    last_stop = ""
    inpatient: list[tuple[str, datetime, datetime]] = []

    with (csv_dir / "encounters.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            total_encounters += 1
            start_raw, stop_raw = row["START"], row["STOP"]
            if not (is_iso8601(start_raw) and is_iso8601(stop_raw)):
                invalid_rows += 1
                continue
            if not first_start or start_raw < first_start:
                first_start = start_raw
            if stop_raw > last_stop:
                last_stop = stop_raw
            if row["ENCOUNTERCLASS"] == "inpatient":
                inpatient.append(
                    (
                        row["PATIENT"],
                        datetime.fromisoformat(start_raw),
                        datetime.fromisoformat(stop_raw),
                    )
                )

    by_patient: dict[str, list[tuple[datetime, datetime]]] = {}
    for patient, start, stop in inpatient:
        by_patient.setdefault(patient, []).append((start, stop))

    readmitted = 0
    window = timedelta(days=30)
    for stays in by_patient.values():
        stays.sort()
        for i, (_, stop) in enumerate(stays):
            if any(stop < later_start <= stop + window for later_start, _ in stays[i + 1 :]):
                readmitted += 1

    rate = readmitted / len(inpatient) if inpatient else 0.0
    return SanityStats(
        total_patients=total_patients,
        total_encounters=total_encounters,
        inpatient_encounters=len(inpatient),
        crude_readmission_rate=rate,
        adult_share=adult_share,
        first_start=first_start,
        last_stop=last_stop,
        invalid_timestamp_rows=invalid_rows,
    )
