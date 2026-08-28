"""Tests for the single reader every raw-CSV path goes through.

Two properties matter here. The read must hold every cell as its verbatim
source text, because that is what makes a state read-back byte-identical
to a batch load (docs/service-notes.md). And it must be the only such read
in the shipped code, because a read that drifts in one caller and not the
others is a skew bug that produces no error, only wrong numbers.
"""

from pathlib import Path

import pandas as pd
import pytest

import risk_scoring
from factories import (
    make_condition_row,
    make_encounter_row,
    make_medication_row,
    make_patient_row,
    write_rows_csv,
)
from risk_scoring.populations import POPULATION_FRAMES, load_population


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    """A minimal four-file export with one row in each frame."""
    write_rows_csv(tmp_path / "patients.csv", [make_patient_row()])
    write_rows_csv(tmp_path / "encounters.csv", [make_encounter_row()])
    write_rows_csv(tmp_path / "medications.csv", [make_medication_row()])
    write_rows_csv(tmp_path / "conditions.csv", [make_condition_row()])
    return tmp_path


def test_reads_every_frame_by_default(csv_dir: Path) -> None:
    frames = load_population(csv_dir)
    assert set(frames) == set(POPULATION_FRAMES)
    assert all(len(frame) == 1 for frame in frames.values())


def test_every_column_is_read_as_text(csv_dir: Path) -> None:
    for frame in load_population(csv_dir).values():
        assert all(pd.api.types.is_object_dtype(dtype) for dtype in frame.dtypes)
        assert all(isinstance(value, str) for value in frame.to_numpy().ravel())


def test_missing_values_are_empty_strings_not_nan(csv_dir: Path) -> None:
    patients = load_population(csv_dir, frames=("patients",))["patients"]
    assert patients.loc[0, "DEATHDATE"] == ""
    assert not patients.isna().to_numpy().any()


@pytest.mark.parametrize("literal", ["NA", "N/A", "NaN", "null", "None", "nan"])
def test_values_that_pandas_would_call_missing_survive_verbatim(
    tmp_path: Path, literal: str
) -> None:
    write_rows_csv(tmp_path / "patients.csv", [make_patient_row(MARITAL=literal)])
    patients = load_population(tmp_path, frames=("patients",))["patients"]
    assert patients.loc[0, "MARITAL"] == literal


def test_numeric_looking_values_keep_their_source_text(tmp_path: Path) -> None:
    write_rows_csv(tmp_path / "patients.csv", [make_patient_row(ZIP="02108", FIPS="0025")])
    patients = load_population(tmp_path, frames=("patients",))["patients"]
    assert patients.loc[0, "ZIP"] == "02108"
    assert patients.loc[0, "FIPS"] == "0025"


def test_frame_matches_a_direct_all_string_read(csv_dir: Path) -> None:
    """The contract state read-back is compared against, spelled out once."""
    expected = pd.read_csv(csv_dir / "encounters.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(load_population(csv_dir)["encounters"], expected)


def test_requesting_a_subset_reads_only_those_files(csv_dir: Path) -> None:
    (csv_dir / "medications.csv").unlink()
    (csv_dir / "conditions.csv").unlink()
    frames = load_population(csv_dir, frames=("encounters", "patients"))
    assert set(frames) == {"encounters", "patients"}


def test_unknown_frame_name_is_rejected(csv_dir: Path) -> None:
    with pytest.raises(ValueError, match="observations"):
        load_population(csv_dir, frames=("patients", "observations"))


def test_shipped_code_reads_csvs_only_through_this_module() -> None:
    """No second copy of the read may exist in src or scripts.

    Tests are deliberately exempt: a test that read through this module
    could not notice this module changing. They spell the options out
    themselves and compare.
    """
    package = Path(risk_scoring.__file__).resolve().parent
    scripts = package.parent.parent / "scripts"
    offenders = sorted(
        str(path.relative_to(package.parent.parent))
        for path in [*package.rglob("*.py"), *scripts.glob("*.py")]
        if path.name != "populations.py" and "read_csv" in path.read_text()
    )
    assert offenders == []
