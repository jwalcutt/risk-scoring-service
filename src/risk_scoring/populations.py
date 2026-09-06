"""Reading a frozen population's CSV export.

Every path that starts from raw Synthea CSVs comes through here: batch
training, the evaluation gate, the cohort builder, and the local check
scripts. The read options are load-bearing rather than incidental, so
they are written once and shared instead of being repeated per caller.

Judgment calls this module fixes:

- Columns are read as text and missing cells as empty strings
  (``dtype=str, keep_default_na=False``). Pandas would otherwise infer a
  type per column and turn empty cells into NaN, so a leading-zero ZIP
  would lose its zero and an empty DEATHDATE would stop matching the
  empty string the state tables hold. Keeping every value as its verbatim
  source text is what makes a state read-back byte-identical to a batch
  load (docs/service-notes.md); the shared modules parse timestamps and
  numbers themselves, from those strings.
- Callers name the frames they need, because the cohort builder reads two
  files while training and the gate read four. A name outside
  ``POPULATION_FRAMES`` raises here rather than surfacing later as a
  missing file, since the export defines no such frame to read.
- A population that joins a replay mid-stream is read through
  :func:`rekeyed`, which rewrites its patient and encounter ids. Synthea
  reproduces the id sequence from the seed, so a module-variant export
  reuses most of the baseline's ids: sometimes for the same person with a
  few divergent rows, sometimes for a different person altogether. Either
  way its history cannot sit beside the baseline's in state under the
  same ids. The rewrite is a pure function of the population name and the
  id, confined to the id columns, and invisible to the shared modules,
  which use ids only to group and join; it lives here rather than in the
  replay package because anything that later reads a spliced-in export
  against the log (provenance, realized performance) must read it through
  the same view.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

POPULATION_FRAMES: tuple[str, ...] = ("patients", "encounters", "medications", "conditions")

# The columns holding patient and encounter ids, per frame. Organization,
# provider, and payer ids are shared reference data, not keys, and stay.
ID_COLUMNS: dict[str, tuple[str, ...]] = {
    "patients": ("Id",),
    "encounters": ("Id", "PATIENT"),
    "medications": ("PATIENT", "ENCOUNTER"),
    "conditions": ("PATIENT", "ENCOUNTER"),
}

# Fixed so the rewrite is the same in every process and every run.
REKEY_NAMESPACE = uuid.UUID("6f3c2f0e-9d3b-4c6a-8f1e-2b7a9c4d5e61")


def load_population(
    csv_dir: Path, *, frames: Sequence[str] = POPULATION_FRAMES
) -> dict[str, pd.DataFrame]:
    """Read the named frames of a Synthea CSV export as all-string frames."""
    unknown = sorted({name for name in frames if name not in POPULATION_FRAMES})
    if unknown:
        raise ValueError(
            f"no such population frame: {', '.join(unknown)}; "
            f"expected any of {', '.join(POPULATION_FRAMES)}"
        )
    return {
        name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in frames
    }


def rekey_id(namespace: str, value: str) -> str:
    """One id rewritten for a population; deterministic, UUID-shaped, empty stays empty."""
    if value == "":
        return value
    return str(uuid.uuid5(REKEY_NAMESPACE, f"{namespace}/{value}"))


def rekeyed(frames: Mapping[str, pd.DataFrame], namespace: str) -> dict[str, pd.DataFrame]:
    """The frames with every patient and encounter id rewritten under ``namespace``.

    Every other column is returned byte-identical, and the given frames
    are not modified. Frames outside ``ID_COLUMNS`` pass through copied.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        copy = frame.copy()
        for column in ID_COLUMNS.get(name, ()):
            copy[column] = copy[column].map(lambda value: rekey_id(namespace, value))
        out[name] = copy
    return out
