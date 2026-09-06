"""Builders for synthetic Synthea CSV rows used in tests.

Column sets and ordering mirror the frozen Synthea v4.0.0 CSV export
exactly, so fixtures exercise the same schema the real populations carry.
Defaults are neutral: no builder default satisfies a cohort inclusion rule
on its own (the default encounter class is ``wellness``, the default
patient is living and adult).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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


def write_training_csvs(csv_dir: Path, *, n_patients: int = 100) -> int:
    """Synthetic population with a learnable signal; returns the cohort row count.

    Each adult patient has one index inpatient stay well before the cutoff.
    Even-numbered patients carry a prior emergency visit (the signal) and an
    inpatient readmission 10 days after the index discharge; the readmission
    stays are cohort rows themselves, so the pre-cutoff cohort holds one and
    a half rows per patient. One extra patient discharges after the cutoff
    and must be excluded. The signal is deterministic, so an honestly
    trained model on this population separates perfectly and lands above the
    signal band ceiling.

    The default patient count is what makes that true rather than
    aspirational. Training uses ``min_data_in_leaf`` 20, so a population
    small enough to leave fewer than roughly a hundred training rows gives
    LightGBM no split it will take, and the booster comes back constant at
    the base rate: every feature vector scores the same number, AUROC is
    exactly 0.5, and any test comparing scores holds no matter what the
    scoring path does.
    """
    patients = []
    encounters = []
    for i in range(n_patients):
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
    return n_patients + n_patients // 2


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


def payload_frame(rows: Iterable[Mapping[str, str]], columns: Sequence[str]) -> pd.DataFrame:
    """Project builder rows onto an ingestion payload column set.

    The result is shaped exactly like ``state.patient_history`` read-back:
    uppercase Synthea names in payload order, verbatim string values, empty
    strings for missing. Passing no rows yields the empty frame with those
    columns, which is what a patient with no medications reads back as.
    """
    return pd.DataFrame(
        [{name: row[name] for name in columns} for row in rows], columns=list(columns)
    )


def write_skew_population(csv_dir: Path) -> int:
    """Population built to exercise every feature boundary at once.

    Eight patients cover the cases the batch feature tests pin
    individually: the 180-day window's inclusive far edge and the day
    beyond it, the days-since-previous cap and its no-history sentinel,
    overlapping stays flooring the gap at zero, a readmission whose
    features must see the index stay, medications stopping exactly at the
    discharge instant, conditions starting and stopping on the discharge
    date, a finding that is not a disorder, two prescriptions of one drug
    sharing an encounter and a start instant, history-based flags from a
    resolved situation code and from an ICD10 malignancy, and events
    dated after the discharge that no feature may read. Three patients
    are excluded outright: a minor, an in-hospital death, and one with no
    inpatient encounter.

    Returns the number of encounters the cohort rules admit.
    """
    patients = [
        make_patient_row(Id="p-fresh", BIRTHDATE="1970-05-15"),
        make_patient_row(Id="p-edge", BIRTHDATE="1955-02-20"),
        make_patient_row(Id="p-gap", BIRTHDATE="1948-11-03"),
        make_patient_row(Id="p-readmit", BIRTHDATE="1962-07-07"),
        make_patient_row(Id="p-full", BIRTHDATE="1951-09-30"),
        make_patient_row(Id="p-minor", BIRTHDATE="2008-06-01"),
        make_patient_row(Id="p-died", BIRTHDATE="1940-01-01", DEATHDATE="2024-03-12"),
        make_patient_row(Id="p-outpatient", BIRTHDATE="1975-01-01"),
    ]

    def stay(
        encounter_id: str, patient: str, start: str, stop: str, encounter_class: str = "inpatient"
    ) -> dict[str, str]:
        return make_encounter_row(
            Id=encounter_id,
            PATIENT=patient,
            ENCOUNTERCLASS=encounter_class,
            START=start,
            STOP=stop,
        )

    encounters = [
        # No history at all: the days-since-previous sentinel and zero counts.
        stay("e-fresh", "p-fresh", "2024-03-01T09:00:00Z", "2024-03-04T09:00:00Z"),
        # The index discharge's 180-day window opens 2023-12-17T07:00:00Z.
        stay("e-edge-out", "p-edge", "2023-12-16T03:00:00Z", "2023-12-16T07:00:00Z", "emergency"),
        stay("e-edge-in", "p-edge", "2023-12-17T01:00:00Z", "2023-12-17T07:00:00Z"),
        stay("e-edge-ed", "p-edge", "2024-01-05T10:00:00Z", "2024-01-05T13:00:00Z", "emergency"),
        stay("e-edge-index", "p-edge", "2024-06-10T07:00:00Z", "2024-06-14T07:00:00Z"),
        # A discharge far enough back to hit the cap, then two overlapping stays.
        stay("e-gap-ancient", "p-gap", "2022-01-05T08:00:00Z", "2022-01-10T08:00:00Z"),
        stay("e-gap-index", "p-gap", "2024-02-01T08:00:00Z", "2024-02-05T08:00:00Z"),
        stay("e-gap-overlap-a", "p-gap", "2024-04-01T08:00:00Z", "2024-04-10T08:00:00Z"),
        stay("e-gap-overlap-b", "p-gap", "2024-04-05T08:00:00Z", "2024-04-12T08:00:00Z"),
        # A readmission ten days after its index discharge; both are scored.
        stay("e-readmit-1", "p-readmit", "2024-05-01T08:00:00Z", "2024-05-05T08:00:00Z"),
        stay("e-readmit-2", "p-readmit", "2024-05-15T08:00:00Z", "2024-05-18T08:00:00Z"),
        # The medication and condition boundaries all land on this discharge.
        stay("e-full-index", "p-full", "2024-08-01T06:00:00Z", "2024-08-05T06:00:00Z"),
        stay("e-minor", "p-minor", "2024-01-02T08:00:00Z", "2024-01-05T08:00:00Z"),
        stay("e-died", "p-died", "2024-03-08T08:00:00Z", "2024-03-12T08:00:00Z"),
        stay(
            "e-out-wellness",
            "p-outpatient",
            "2024-02-02T08:00:00Z",
            "2024-02-02T09:00:00Z",
            "wellness",
        ),
        stay(
            "e-out-ed", "p-outpatient", "2024-02-20T08:00:00Z", "2024-02-20T11:00:00Z", "emergency"
        ),
    ]

    def prescription(code: str, start: str, stop: str) -> dict[str, str]:
        return make_medication_row(
            PATIENT="p-full", ENCOUNTER="e-full-index", CODE=code, START=start, STOP=stop
        )

    medications = [
        # Stopping exactly at the discharge instant means inactive.
        prescription("308136", "2024-07-01T06:00:00Z", "2024-08-05T06:00:00Z"),
        prescription("310798", "2024-07-02T06:00:00Z", ""),
        # Same drug, encounter, and instant as the row above, differing only in
        # stop: a single dispense beside a continuing course, which is how
        # Synthea records a renewal. Both are active and both must be counted.
        prescription("310798", "2024-07-02T06:00:00Z", "2024-09-15T06:00:00Z"),
        prescription("314076", "2024-07-03T06:00:00Z", "2024-09-01T06:00:00Z"),
        prescription("861007", "2024-06-01T06:00:00Z", "2024-07-15T06:00:00Z"),
        # Prescribed the day after discharge: invisible to the scored row.
        prescription("197361", "2024-08-06T06:00:00Z", ""),
    ]

    def diagnosis(
        code: str, start: str, stop: str, description: str, system: str = "SNOMED-CT"
    ) -> dict[str, str]:
        return make_condition_row(
            PATIENT="p-full",
            ENCOUNTER="e-full-index",
            SYSTEM=system,
            CODE=code,
            START=start,
            STOP=stop,
            DESCRIPTION=description,
        )

    conditions = [
        diagnosis("444814009", "2024-01-10", "", "Viral sinusitis (disorder)"),
        # Recorded on the discharge date: active. Resolved on it: not active.
        diagnosis("195662009", "2024-08-05", "", "Acute viral pharyngitis (disorder)"),
        diagnosis("10509002", "2024-02-01", "2024-08-05", "Acute bronchitis (disorder)"),
        diagnosis("160903007", "2024-03-01", "", "Full-time employment (finding)"),
        diagnosis("39848009", "2024-08-06", "", "Whiplash injury to neck (disorder)"),
        diagnosis("88805009", "2023-05-01", "", "Chronic congestive heart failure (disorder)"),
        # Resolved situation code: sets the history-based flag, counts as no disorder.
        diagnosis(
            "399211009", "2020-01-01", "2020-02-01", "History of myocardial infarction (situation)"
        ),
        diagnosis("C50.9", "2022-06-01", "", "Malignant neoplasm of breast (disorder)", "ICD10"),
    ]

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(csv_dir / "medications.csv", medications)
    write_rows_csv(csv_dir / "conditions.csv", conditions)
    return 10


def write_splice_population(csv_dir: Path) -> int:
    """The population a replay splices to, built against the skew population.

    Designed for a splice at 2024-05-10T00:00:00Z in the skew population's
    span (2024-04-01 to 2024-08-07). One patient reuses a skew id with a
    different birthdate, the collision a module-variant export produces
    in practice, so the two can share state only once this population's
    ids are rewritten. Its stays cover: a pre-splice discharge that is
    preloaded, never posted, and never labelled; an index discharge whose
    180-day count and days-since-previous are right only if that history
    reached state; a discharge at exactly the splice instant; a
    readmission pair after the splice; and a discharge inside the final
    30 days, so one label is pending. A medication and a condition lie on
    each side of the splice so every event kind crosses it.

    Returns the number of encounters the cohort rules admit after the
    splice.
    """
    patients = [
        make_patient_row(Id="p-fresh", BIRTHDATE="1958-03-03"),
        make_patient_row(Id="p-b-boundary", BIRTHDATE="1965-01-01"),
        make_patient_row(Id="p-b-readmit", BIRTHDATE="1950-06-06"),
        make_patient_row(Id="p-b-late", BIRTHDATE="1972-02-02"),
    ]

    def stay(encounter_id: str, patient: str, start: str, stop: str) -> dict[str, str]:
        return make_encounter_row(
            Id=encounter_id, PATIENT=patient, ENCOUNTERCLASS="inpatient", START=start, STOP=stop
        )

    encounters = [
        # History before the splice: preloaded, never posted, never labelled.
        stay("e-b-prior", "p-fresh", "2024-04-16T08:00:00Z", "2024-04-20T08:00:00Z"),
        # Admitted 38 days after e-b-prior: one prior inpatient stay in the window.
        stay("e-b-index", "p-fresh", "2024-05-28T08:00:00Z", "2024-06-01T08:00:00Z"),
        # Discharged at exactly the splice instant: this population's.
        stay("e-b-boundary", "p-b-boundary", "2024-05-07T00:00:00Z", "2024-05-10T00:00:00Z"),
        # A pre-splice stay readmitted after it, then a readmission pair.
        stay("e-b-history", "p-b-readmit", "2024-05-02T08:00:00Z", "2024-05-08T08:00:00Z"),
        stay("e-b-r1", "p-b-readmit", "2024-05-17T08:00:00Z", "2024-05-20T08:00:00Z"),
        stay("e-b-r2", "p-b-readmit", "2024-06-05T08:00:00Z", "2024-06-08T08:00:00Z"),
        # Inside the final 30 days: scored, label pending at the end.
        stay("e-b-late", "p-b-late", "2024-07-29T08:00:00Z", "2024-08-01T08:00:00Z"),
    ]
    medications = [
        make_medication_row(
            PATIENT="p-fresh",
            ENCOUNTER="e-b-prior",
            CODE="308136",
            START="2024-04-16T08:00:00Z",
            STOP="2024-04-20T08:00:00Z",
        ),
        make_medication_row(
            PATIENT="p-fresh",
            ENCOUNTER="e-b-index",
            CODE="310798",
            START="2024-05-28T08:00:00Z",
            STOP="",
        ),
    ]
    conditions = [
        make_condition_row(
            PATIENT="p-fresh",
            ENCOUNTER="e-b-prior",
            CODE="444814009",
            START="2024-04-16",
            STOP="2024-04-30",
            DESCRIPTION="Viral sinusitis (disorder)",
        ),
        make_condition_row(
            PATIENT="p-fresh",
            ENCOUNTER="e-b-index",
            CODE="88805009",
            START="2024-05-29",
            STOP="",
            DESCRIPTION="Chronic congestive heart failure (disorder)",
        ),
    ]

    csv_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(csv_dir / "patients.csv", patients)
    write_rows_csv(csv_dir / "encounters.csv", encounters)
    write_rows_csv(csv_dir / "medications.csv", medications)
    write_rows_csv(csv_dir / "conditions.csv", conditions)
    return 5
