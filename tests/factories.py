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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

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


def iso_timestamp(moment: datetime) -> str:
    """Format a datetime the way the Synthea encounter export does."""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_training_csvs(csv_dir: Path) -> int:
    """Synthetic population with a learnable signal; returns the cohort row count.

    Forty adult patients each have one index inpatient stay well before the
    cutoff. Even-numbered patients carry a prior emergency visit (the
    signal) and an inpatient readmission 10 days after the index discharge;
    the readmission stays are cohort rows themselves, so the pre-cutoff
    cohort holds 60 rows. One extra patient discharges after the cutoff and
    must be excluded. The signal is deterministic, so an honestly trained
    model on this population scores near-perfect AUROC.
    """
    patients = []
    encounters = []
    for i in range(40):
        pid = f"p{i:02d}"
        patients.append(make_patient_row(Id=pid, BIRTHDATE="1960-01-01"))
        index_start = datetime(2024, 3, 1, 8, 0, tzinfo=UTC) + timedelta(days=i)
        index_stop = index_start + timedelta(days=3)
        encounters.append(
            make_encounter_row(
                Id=f"e-index-{pid}",
                PATIENT=pid,
                ENCOUNTERCLASS="inpatient",
                START=iso_timestamp(index_start),
                STOP=iso_timestamp(index_stop),
            )
        )
        if i % 2 == 0:
            ed_visit = index_start - timedelta(days=30)
            encounters.append(
                make_encounter_row(
                    Id=f"e-ed-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="emergency",
                    START=iso_timestamp(ed_visit),
                    STOP=iso_timestamp(ed_visit + timedelta(hours=4)),
                )
            )
            readmit_start = index_stop + timedelta(days=10)
            encounters.append(
                make_encounter_row(
                    Id=f"e-readmit-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="inpatient",
                    START=iso_timestamp(readmit_start),
                    STOP=iso_timestamp(readmit_start + timedelta(days=2)),
                )
            )

    patients.append(make_patient_row(Id="p-late", BIRTHDATE="1960-01-01"))
    encounters.append(
        make_encounter_row(
            Id="e-late",
            PATIENT="p-late",
            ENCOUNTERCLASS="inpatient",
            START="2025-01-30T08:00:00Z",
            STOP="2025-02-02T08:00:00Z",
        )
    )

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(
        csv_dir / "medications.csv",
        [make_medication_row(PATIENT="p00", ENCOUNTER="e-index-p00")],
    )
    write_rows_csv(
        csv_dir / "conditions.csv",
        [make_condition_row(PATIENT="p00", ENCOUNTER="e-index-p00")],
    )
    return 60


def write_gate_population(csv_dir: Path, *, seed: int = 20260101, n_patients: int = 2000) -> None:
    """Synthetic population a fair model passes the gate on.

    Each patient carries two index inpatient stays 420 days apart:
    farther than the 365-day days-since-previous cap plus the readmission
    window, so no stay's history features see an earlier stay or its
    readmission. A hidden per-patient risk coin drives everything
    observable: a risky patient probably shows a prior emergency visit
    before each stay (the learnable, noisy signal) and one readmission
    coin decides both stays' outcomes together. No feature identifies a
    patient (length of stay is constant, ages repeat across many
    patients), so an honestly trained model learns the emergency rates,
    lands mid-band on AUROC, and stays calibrated.

    Ages cycle across all four gate age bands and sexes alternate, so
    every subgroup column is populated. All draws come from one seeded
    generator: the population, and every metric computed from it, is
    exactly reproducible.
    """
    rng = np.random.default_rng(seed)
    patients = []
    encounters = []
    for i in range(n_patients):
        pid = f"g{i:04d}"
        # Eight age levels spanning all four gate bands: enough patients
        # share each age that the model cannot use it to overfit noise.
        age = (22, 31, 40, 49, 58, 67, 76, 85)[i % 8]
        patients.append(
            make_patient_row(
                Id=pid,
                BIRTHDATE=f"{2024 - age}-01-01",
                GENDER="M" if i % 2 else "F",
            )
        )

        risky = rng.random() < 0.35
        p_ed = 0.75 if risky else 0.15
        p_readmit = 0.6 if risky else 0.08
        readmits = rng.random() < p_readmit
        los = timedelta(days=3)

        stay1_start = datetime(2022, 6, 5, 8, 0, tzinfo=UTC) + timedelta(days=int(i % 60))
        for stay_no, stay_start in ((1, stay1_start), (2, stay1_start + timedelta(days=420))):
            stay_stop = stay_start + los
            encounters.append(
                make_encounter_row(
                    Id=f"e{stay_no}-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="inpatient",
                    START=iso_timestamp(stay_start),
                    STOP=iso_timestamp(stay_stop),
                )
            )
            if rng.random() < p_ed:
                ed_start = stay_start - timedelta(days=20)
                encounters.append(
                    make_encounter_row(
                        Id=f"d{stay_no}-{pid}",
                        PATIENT=pid,
                        ENCOUNTERCLASS="emergency",
                        START=iso_timestamp(ed_start),
                        STOP=iso_timestamp(ed_start + timedelta(hours=4)),
                    )
                )
            if readmits:
                readmit_start = stay_stop + timedelta(days=12)
                encounters.append(
                    make_encounter_row(
                        Id=f"r{stay_no}-{pid}",
                        PATIENT=pid,
                        ENCOUNTERCLASS="inpatient",
                        START=iso_timestamp(readmit_start),
                        STOP=iso_timestamp(readmit_start + timedelta(days=2)),
                    )
                )

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(
        csv_dir / "medications.csv",
        [make_medication_row(PATIENT="g0000", ENCOUNTER="e1-g0000")],
    )
    write_rows_csv(
        csv_dir / "conditions.csv",
        [make_condition_row(PATIENT="g0000", ENCOUNTER="e1-g0000")],
    )


def write_leak_population(csv_dir: Path, *, seed: int = 20260101, n_patients: int = 300) -> None:
    """Synthetic population built to expose split leakage, and nothing else.

    Outcomes are a pure per-patient coin: one draw decides every stay's
    readmission, and no feature carries any cross-patient signal, so a
    patient-grouped model can do no better than chance. Each patient has
    ten index stays and a near-unique length-of-stay fingerprint shared
    by all of them. A candidate trained on a row-level split therefore
    sees most of each evaluation patient's rows during training and can
    read the outcome straight off the fingerprint: its apparent AUROC
    lands far above the signal-band ceiling, which is exactly the
    inflated score the gate must flag.

    Stays sit 420 days apart, beyond every history window, so the
    fingerprint is the only within-patient link.
    """
    rng = np.random.default_rng(seed)
    patients = []
    encounters = []
    for i in range(n_patients):
        pid = f"l{i:04d}"
        age = 20 + (i % 72)
        patients.append(make_patient_row(Id=pid, BIRTHDATE=f"{2024 - age}-01-01"))

        readmits = rng.random() < 0.5
        los = timedelta(days=2.0 + i * 0.01)

        stay1_start = datetime(2014, 3, 5, 8, 0, tzinfo=UTC) + timedelta(days=int(i % 60))
        for stay_no in range(10):
            stay_start = stay1_start + timedelta(days=420 * stay_no)
            stay_stop = stay_start + los
            encounters.append(
                make_encounter_row(
                    Id=f"e{stay_no}-{pid}",
                    PATIENT=pid,
                    ENCOUNTERCLASS="inpatient",
                    START=iso_timestamp(stay_start),
                    STOP=iso_timestamp(stay_stop),
                )
            )
            if readmits:
                readmit_start = stay_stop + timedelta(days=12)
                encounters.append(
                    make_encounter_row(
                        Id=f"r{stay_no}-{pid}",
                        PATIENT=pid,
                        ENCOUNTERCLASS="inpatient",
                        START=iso_timestamp(readmit_start),
                        STOP=iso_timestamp(readmit_start + timedelta(days=2)),
                    )
                )

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(
        csv_dir / "medications.csv",
        [make_medication_row(PATIENT="l0000", ENCOUNTER="e0-l0000")],
    )
    write_rows_csv(
        csv_dir / "conditions.csv",
        [make_condition_row(PATIENT="l0000", ENCOUNTER="e0-l0000")],
    )
