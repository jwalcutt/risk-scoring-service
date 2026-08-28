"""Re-deriving a logged prediction from its source row and its own model.

Every logged prediction makes two claims. Its input hash covers the exact
event that produced it, and its score is what the named model version
returns for the stored feature values. Both are checkable after the fact,
and this module checks them by recomputation rather than by trusting the
row.

Judgment calls this module fixes:

- The source event is rebuilt from the population export, never from
  whatever the caller happens to hold in memory. Hashing the same object
  twice inside one process would prove nothing; rebuilding from the
  export proves the chain from source row through payload projection and
  envelope to the digest that was stored.
- The model comes from the version the row names, never from the version
  the service is currently pinned to. Loading the pinned version would
  make the check pass for a row written by an entirely different model,
  which is the failure it exists to catch.
- Comparison is exact, with no tolerance. Both stored columns round-trip
  losslessly, and the mistakes this check exists to catch (a wrong column
  order, a rounded stored value, the wrong version) either change the
  score visibly or not at all. A tolerance would only widen the band
  where something subtly wrong looks fine.
- A features dict missing a model column raises rather than filling in a
  default, which would turn a provenance break into a plausible number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from risk_scoring.features import MODEL_INPUT_COLUMNS
from risk_scoring.payload_hash import payload_hash
from risk_scoring.predictions import StoredPrediction
from risk_scoring.stream import EVENT_FIELDS
from risk_scoring.tracking import configure_tracking


@dataclass(frozen=True)
class ProvenanceCheck:
    """One logged prediction, recomputed from its source and its model."""

    encounter_id: str
    prediction_id: int
    model_uri: str
    logged_hash: str
    recomputed_hash: str
    logged_score: float
    rescored: float

    @property
    def hash_matches(self) -> bool:
        return self.logged_hash == self.recomputed_hash

    @property
    def score_matches(self) -> bool:
        return self.logged_score == self.rescored

    @property
    def ok(self) -> bool:
        return self.hash_matches and self.score_matches

    def describe(self) -> str:
        """One line naming what was checked, and both sides of any break."""
        head = f"{self.encounter_id} (prediction {self.prediction_id}, {self.model_uri})"
        if self.ok:
            return f"{head}: hash {self.logged_hash} and score {self.logged_score!r} reproduced"
        parts = []
        if not self.hash_matches:
            parts.append(f"hash logged {self.logged_hash}, recomputed {self.recomputed_hash}")
        if not self.score_matches:
            gap = abs(self.logged_score - self.rescored)
            parts.append(
                f"score logged {self.logged_score!r}, rescored {self.rescored!r} "
                f"(absolute {gap!r}, {_ulps(self.logged_score, self.rescored)} ulp)"
            )
        return f"{head}: " + "; ".join(parts)


def _ulps(left: float, right: float) -> int:
    """Representable doubles between two values, so a last-bit drift is legible."""
    if left == right:
        return 0
    steps = 0
    low, high = (left, right) if left < right else (right, left)
    while low < high and steps < 1000:
        low = math.nextafter(low, high)
        steps += 1
    return steps


def source_event(encounter_row: Mapping[str, str]) -> dict[str, Any]:
    """The posted envelope for one encounter, projected as the stream projects it."""
    payload = {name: encounter_row[name] for name in EVENT_FIELDS["encounter"]}
    return {"event_type": "encounter", "payload": payload}


def recompute_input_hash(encounter_row: Mapping[str, str]) -> str:
    """The digest the service would have stored for this source row."""
    return payload_hash(source_event(encounter_row))


def rescore(model: Any, features: Mapping[str, float]) -> float:
    """The model's score for stored feature values, built as the service builds it."""
    missing = [name for name in MODEL_INPUT_COLUMNS if name not in features]
    if missing:
        raise KeyError(f"stored features are missing model input columns: {', '.join(missing)}")
    frame = pd.DataFrame(
        [{name: features[name] for name in MODEL_INPUT_COLUMNS}],
        columns=list(MODEL_INPUT_COLUMNS),
    ).astype("float64")
    return float(np.asarray(model.predict(frame), dtype=float).ravel()[0])


def verify_predictions(
    predictions: Sequence[StoredPrediction],
    encounters: pd.DataFrame,
    repo_root: Path,
) -> list[ProvenanceCheck]:
    """Recompute the hash and the score of every prediction, in log order.

    Raises ``KeyError`` when a prediction names an encounter the export
    does not contain, since that is a broken chain rather than a mismatch
    to report.
    """
    if not predictions:
        return []
    configure_tracking(repo_root)
    rows: dict[str, dict[str, str]] = {}
    for _, row in encounters.iterrows():
        rows[str(row["Id"])] = {str(name): str(value) for name, value in row.items()}
    loaded: dict[str, Any] = {}
    checks = []
    for prediction in predictions:
        if prediction.encounter_id not in rows:
            raise KeyError(f"no source row for encounter {prediction.encounter_id}")
        uri = f"models:/{prediction.model_name}/{prediction.model_version}"
        if uri not in loaded:
            loaded[uri] = mlflow.pyfunc.load_model(uri)
        checks.append(
            ProvenanceCheck(
                encounter_id=prediction.encounter_id,
                prediction_id=prediction.prediction_id,
                model_uri=uri,
                logged_hash=prediction.input_hash,
                recomputed_hash=recompute_input_hash(rows[prediction.encounter_id]),
                logged_score=prediction.score,
                rescored=rescore(loaded[uri], prediction.features),
            )
        )
    return checks
