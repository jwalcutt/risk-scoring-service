"""Tests for the single reader every raw-CSV path goes through.

Two properties matter here. The read must hold every cell as its verbatim
source text, because that is what makes a state read-back byte-identical
to a batch load (docs/service-notes.md). And it must be the only such read
in the shipped code, because a read that drifts in one caller and not the
others is a skew bug that produces no error, only wrong numbers.
"""

import uuid
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
    write_skew_population,
)
from risk_scoring.cohort import build_cohort
from risk_scoring.features import build_features
from risk_scoring.labels import build_labels
from risk_scoring.populations import (
    ID_COLUMNS,
    POPULATION_FRAMES,
    load_population,
    rekey_id,
    rekeyed,
)


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


# Rekeying a spliced-in population
#
# A variant export generated from the same seed reuses the baseline's
# patient and encounter ids, sometimes for the same person with divergent
# rows and sometimes for a different person altogether. A population that
# joins a replay mid-stream therefore has its ids rewritten at load, so
# its patients are distinct people in state. The rewrite must be pure,
# deterministic, confined to the id columns, and invisible to the shared
# modules, which only ever use ids to group and join.

# Restated literally, so a column added to or dropped from the rewrite
# breaks a test rather than silently changing which rows collide.
EXPECTED_ID_COLUMNS = {
    "patients": ("Id",),
    "encounters": ("Id", "PATIENT"),
    "medications": ("PATIENT", "ENCOUNTER"),
    "conditions": ("PATIENT", "ENCOUNTER"),
}


@pytest.fixture()
def skew(tmp_path: Path) -> dict[str, pd.DataFrame]:
    csv_dir = tmp_path / "skew"
    write_skew_population(csv_dir)
    return load_population(csv_dir)


def test_the_id_columns_are_exactly_these() -> None:
    assert ID_COLUMNS == EXPECTED_ID_COLUMNS


def test_rekeyed_rewrites_every_id_column_and_nothing_else(skew: dict[str, pd.DataFrame]) -> None:
    out = rekeyed(skew, "variant")
    assert set(out) == set(skew)
    for name, frame in skew.items():
        assert list(out[name].columns) == list(frame.columns)
        for column in frame.columns:
            if column in EXPECTED_ID_COLUMNS[name]:
                assert (out[name][column] != frame[column]).all(), (name, column)
            else:
                pd.testing.assert_series_equal(out[name][column], frame[column])


def test_rekeyed_is_deterministic_and_namespace_specific(skew: dict[str, pd.DataFrame]) -> None:
    once = rekeyed(skew, "variant")
    again = rekeyed(skew, "variant")
    other = rekeyed(skew, "other")
    for name in skew:
        pd.testing.assert_frame_equal(once[name], again[name])
        assert not set(once[name]["Id" if name == "patients" else "PATIENT"]) & set(
            other[name]["Id" if name == "patients" else "PATIENT"]
        )


def test_rekeyed_ids_are_uuid_shaped(skew: dict[str, pd.DataFrame]) -> None:
    out = rekeyed(skew, "variant")
    for name, columns in EXPECTED_ID_COLUMNS.items():
        for column in columns:
            for value in out[name][column]:
                assert str(uuid.UUID(value)) == value


def test_rekey_id_keeps_an_empty_id_empty() -> None:
    assert rekey_id("variant", "") == ""
    assert rekey_id("variant", "p-1") == rekey_id("variant", "p-1")
    assert rekey_id("variant", "p-1") != rekey_id("other", "p-1")
    assert rekey_id("variant", "p-1") != "p-1"


def test_rekeyed_preserves_the_joins(skew: dict[str, pd.DataFrame]) -> None:
    out = rekeyed(skew, "variant")
    patients = set(out["patients"]["Id"])
    encounters = set(out["encounters"]["Id"])
    assert set(out["encounters"]["PATIENT"]) <= patients
    for name in ("medications", "conditions"):
        assert set(out[name]["PATIENT"]) <= patients
        assert set(out[name]["ENCOUNTER"]) <= encounters


def test_rekeyed_leaves_the_source_frames_untouched(skew: dict[str, pd.DataFrame]) -> None:
    before = {name: frame.copy() for name, frame in skew.items()}
    rekeyed(skew, "variant")
    for name, frame in skew.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_rekeyed_is_invisible_to_the_shared_modules(skew: dict[str, pd.DataFrame]) -> None:
    """Cohort, features, and labels over rekeyed frames equal the originals up to the ids."""
    out = rekeyed(skew, "variant")

    cohort = build_cohort(skew["encounters"], skew["patients"]).frame
    cohort_out = build_cohort(out["encounters"], out["patients"]).frame
    features = build_features(cohort, skew["encounters"], skew["medications"], skew["conditions"])
    features_out = build_features(
        cohort_out, out["encounters"], out["medications"], out["conditions"]
    )
    labels = build_labels(cohort, skew["encounters"])
    labels_out = build_labels(cohort_out, out["encounters"])

    def mapped(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in ("encounter_id", "patient_id"):
            if column in frame:
                frame[column] = frame[column].map(lambda value: rekey_id("variant", value))
        return frame.sort_values("encounter_id").reset_index(drop=True)

    assert len(cohort) == 10
    for original, rewritten in (
        (cohort, cohort_out),
        (features, features_out),
        (labels, labels_out),
    ):
        pd.testing.assert_frame_equal(
            mapped(original), rewritten.sort_values("encounter_id").reset_index(drop=True)
        )
