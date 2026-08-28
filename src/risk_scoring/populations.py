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
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

POPULATION_FRAMES: tuple[str, ...] = ("patients", "encounters", "medications", "conditions")


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
