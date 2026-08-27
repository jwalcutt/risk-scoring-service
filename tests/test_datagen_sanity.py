"""Tests for the rough sanity statistics computed over raw Synthea CSV output.

These stats gate the freeze decision only. They are deliberately crude
(ENCOUNTERCLASS-based) and are not the cohort definition.
"""

from datetime import date
from pathlib import Path

import pytest

from risk_scoring.datagen.sanity import compute_sanity_stats, is_iso8601

PATIENTS_CSV = """Id,BIRTHDATE,DEATHDATE,GENDER
p1,1950-01-01,,M
p2,2015-06-15,,F
p3,1980-03-10,,F
"""

# e1 (p1) is followed by e2 (p1 inpatient) 15 days after discharge -> readmission.
# e3 (p3) has no later inpatient encounter -> no readmission.
# e2 itself has no later encounter -> no readmission.
# e4 is ambulatory and never counts as inpatient.
ENCOUNTERS_CSV = """Id,START,STOP,PATIENT,ENCOUNTERCLASS
e1,2025-01-01T08:00:00Z,2025-01-05T10:00:00Z,p1,inpatient
e2,2025-01-20T09:00:00Z,2025-01-22T09:00:00Z,p1,inpatient
e3,2025-03-01T00:00:00Z,2025-03-02T12:00:00Z,p3,inpatient
e4,2025-02-10T10:00:00Z,2025-02-10T11:00:00Z,p2,ambulatory
"""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    (tmp_path / "patients.csv").write_text(PATIENTS_CSV)
    (tmp_path / "encounters.csv").write_text(ENCOUNTERS_CSV)
    return tmp_path


def test_counts_inpatient_encounters_only(csv_dir: Path) -> None:
    stats = compute_sanity_stats(csv_dir, as_of=date(2026, 1, 1))
    assert stats.total_encounters == 4
    assert stats.inpatient_encounters == 3


def test_crude_readmission_rate_counts_30_day_inpatient_returns(csv_dir: Path) -> None:
    stats = compute_sanity_stats(csv_dir, as_of=date(2026, 1, 1))
    assert stats.crude_readmission_rate == pytest.approx(1 / 3)


def test_adult_share_measured_at_as_of_date(csv_dir: Path) -> None:
    stats = compute_sanity_stats(csv_dir, as_of=date(2026, 1, 1))
    assert stats.total_patients == 3
    assert stats.adult_share == pytest.approx(2 / 3)


def test_date_span_covers_min_start_and_max_stop(csv_dir: Path) -> None:
    stats = compute_sanity_stats(csv_dir, as_of=date(2026, 1, 1))
    assert stats.first_start == "2025-01-01T08:00:00Z"
    assert stats.last_stop == "2025-03-02T12:00:00Z"


def test_invalid_timestamps_are_counted(tmp_path: Path) -> None:
    (tmp_path / "patients.csv").write_text(PATIENTS_CSV)
    (tmp_path / "encounters.csv").write_text(
        "Id,START,STOP,PATIENT,ENCOUNTERCLASS\n"
        "e1,2025-01-01T08:00:00Z,2025-01-05T10:00:00Z,p1,inpatient\n"
        "e2,01/20/2025,2025-01-22T09:00:00Z,p1,inpatient\n"
        "e3,2025-03-01T00:00:00Z,,p3,inpatient\n"
    )
    stats = compute_sanity_stats(tmp_path, as_of=date(2026, 1, 1))
    assert stats.invalid_timestamp_rows == 2


@pytest.mark.parametrize(
    "value",
    ["2025-01-01T08:00:00Z", "2025-01-01T08:00:00-05:00", "2025-01-01T08:00:00.123Z"],
)
def test_is_iso8601_accepts_timestamped_formats(value: str) -> None:
    assert is_iso8601(value)


@pytest.mark.parametrize("value", ["01/01/2025", "", "2025-13-01T00:00:00Z", "not-a-date"])
def test_is_iso8601_rejects_malformed_values(value: str) -> None:
    assert not is_iso8601(value)
