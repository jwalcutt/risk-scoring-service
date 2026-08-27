"""Builders for synthetic Synthea CSV rows used in tests.

Column sets and ordering mirror the frozen Synthea v4.0.0 CSV export
exactly, so fixtures exercise the same schema the real populations carry.
Defaults are neutral: no builder default satisfies a cohort inclusion rule
on its own (the default encounter class is ``wellness``, the default
patient is living and adult).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

PATIENT_DEFAULTS: dict[str, str] = {
    "Id": "patient-1",
    "BIRTHDATE": "1970-01-01",
    "DEATHDATE": "",
    "SSN": "999-99-9999",
    "DRIVERS": "",
    "PASSPORT": "",
    "PREFIX": "",
    "FIRST": "Test",
    "MIDDLE": "",
    "LAST": "Patient",
    "SUFFIX": "",
    "MAIDEN": "",
    "MARITAL": "",
    "RACE": "white",
    "ETHNICITY": "nonhispanic",
    "GENDER": "F",
    "BIRTHPLACE": "Boston  Massachusetts  US",
    "ADDRESS": "1 Test Way",
    "CITY": "Boston",
    "STATE": "Massachusetts",
    "COUNTY": "Suffolk County",
    "FIPS": "25025",
    "ZIP": "02108",
    "LAT": "42.36",
    "LON": "-71.06",
    "HEALTHCARE_EXPENSES": "0.00",
    "HEALTHCARE_COVERAGE": "0.00",
    "INCOME": "50000",
}

ENCOUNTER_DEFAULTS: dict[str, str] = {
    "Id": "encounter-1",
    "START": "2024-01-01T08:00:00Z",
    "STOP": "2024-01-03T08:00:00Z",
    "PATIENT": "patient-1",
    "ORGANIZATION": "org-1",
    "PROVIDER": "provider-1",
    "PAYER": "payer-1",
    "ENCOUNTERCLASS": "wellness",
    "CODE": "162673000",
    "DESCRIPTION": "General examination of patient (procedure)",
    "BASE_ENCOUNTER_COST": "0.00",
    "TOTAL_CLAIM_COST": "0.00",
    "PAYER_COVERAGE": "0.00",
    "REASONCODE": "",
    "REASONDESCRIPTION": "",
}


MEDICATION_DEFAULTS: dict[str, str] = {
    "START": "2024-01-01T08:00:00Z",
    "STOP": "2024-01-08T08:00:00Z",
    "PATIENT": "patient-1",
    "PAYER": "payer-1",
    "ENCOUNTER": "encounter-1",
    "CODE": "308136",
    "DESCRIPTION": "amLODIPine 2.5 MG Oral Tablet",
    "BASE_COST": "0.00",
    "PAYER_COVERAGE": "0.00",
    "DISPENSES": "1",
    "TOTALCOST": "0.00",
    "REASONCODE": "",
    "REASONDESCRIPTION": "",
}

CONDITION_DEFAULTS: dict[str, str] = {
    "START": "2024-01-01",
    "STOP": "2024-01-08",
    "PATIENT": "patient-1",
    "ENCOUNTER": "encounter-1",
    "SYSTEM": "SNOMED-CT",
    "CODE": "444814009",
    "DESCRIPTION": "Viral sinusitis (disorder)",
}


def _build(defaults: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"unknown column overrides: {sorted(unknown)}")
    return {column: overrides.get(column, default) for column, default in defaults.items()}


def make_patient_row(**overrides: str) -> dict[str, str]:
    return _build(PATIENT_DEFAULTS, overrides)


def make_encounter_row(**overrides: str) -> dict[str, str]:
    return _build(ENCOUNTER_DEFAULTS, overrides)


def make_medication_row(**overrides: str) -> dict[str, str]:
    return _build(MEDICATION_DEFAULTS, overrides)


def make_condition_row(**overrides: str) -> dict[str, str]:
    return _build(CONDITION_DEFAULTS, overrides)


def write_rows_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Write builder rows to a CSV file with the schema header."""
    rows = list(rows)
    if not rows:
        raise ValueError("write_rows_csv needs at least one row to derive the header")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
