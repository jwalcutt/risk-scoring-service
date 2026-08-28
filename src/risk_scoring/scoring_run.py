"""Score a batch of held-out encounters through a running service.

    python -m risk_scoring.scoring_run run [--population baseline]
        [--patients 250] [--seed 20260101] [--cutoff 2025-01-01]

Held out means discharged at or after the training cutoff, so the model
has never seen the encounter. The run posts a seeded sample of patients
holding such a discharge, reads the resulting log back, and recomputes
the provenance of every prediction it produced.

Judgment calls this module fixes:

- Patients are selected, encounters are reported. A patient's features
  read their whole history, so posting only their post-cutoff rows would
  make serving features disagree with the training pipeline for a reason
  that is not a defect. The whole history goes, and the consequence is
  that the service also scores that patient's pre-cutoff discharges.
- Those two sets are counted separately and never summed into one
  headline. Pre-cutoff discharges are scored by the service but are not
  held-out evidence, and one number covering both would read as though
  they were.
- The service must already be running. Bringing the stack up is an
  operator action, not this module's business, so there is no Compose
  orchestration here.
- The log is read filtered to the encounters this run posted, so the
  summary describes this batch whatever else the database holds.
- A batch with no eligible patient raises. A run that silently scored
  nothing would otherwise be written up as a successful one.
- A cohort discharge the service never scored, or a logged row the batch
  cohort does not admit, is named with its encounter ids and fails the
  run. Either is a disagreement worth reading, not an assertion to crash
  on.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from risk_scoring import predictions as predictions_module
from risk_scoring.cohort import build_cohort, split_at_cutoff
from risk_scoring.populations import load_population
from risk_scoring.provenance import ProvenanceCheck, verify_predictions
from risk_scoring.sampling import eligible_patients, select_patients
from risk_scoring.service_client import DEFAULT_SERVICE_PORT, ServiceClient
from risk_scoring.stream import build_stream
from risk_scoring.train import TRAINING_CUTOFF

DEFAULT_PATIENTS = 250
DEFAULT_SEED = 20260101


class EmptyBatchError(RuntimeError):
    """No patient in the population holds a discharge past the cutoff."""


class EventPoster(Protocol):
    """What this module needs from a running service."""

    def version(self) -> dict[str, Any]: ...

    def post_event(self, event: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BatchSelection:
    """The patients chosen, and which of their discharges are held out."""

    frames: dict[str, pd.DataFrame]
    cutoff: pd.Timestamp
    seed: int
    patients_eligible: int
    patients_selected: int
    held_out_encounter_ids: frozenset[str]
    pre_cutoff_encounter_ids: frozenset[str]

    @property
    def cohort_encounter_ids(self) -> frozenset[str]:
        return self.held_out_encounter_ids | self.pre_cutoff_encounter_ids


@dataclass(frozen=True)
class ScoreSummary:
    """A score distribution, or an empty one that invents no quantiles."""

    count: int
    mean: float | None = None
    minimum: float | None = None
    p05: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class LogPartition:
    """The log this run produced, split by which side of the cutoff."""

    held_out: list[predictions_module.StoredPrediction]
    pre_cutoff: list[predictions_module.StoredPrediction]
    unscored_cohort_ids: tuple[str, ...]
    unexpected_logged_ids: tuple[str, ...]

    @property
    def agrees(self) -> bool:
        return not self.unscored_cohort_ids and not self.unexpected_logged_ids


@dataclass(frozen=True)
class BatchRunResult:
    """Everything one held-out batch produced, ready to be written up."""

    selection: BatchSelection
    version: dict[str, Any]
    events_by_type: dict[str, int]
    partition: LogPartition
    all_scores: ScoreSummary
    held_out_scores: ScoreSummary
    provenance: tuple[ProvenanceCheck, ...]
    example: ProvenanceCheck | None
    response_mismatches: tuple[str, ...] = field(default=())

    @property
    def events_posted(self) -> int:
        return sum(self.events_by_type.values())

    @property
    def predictions_logged(self) -> int:
        return len(self.partition.held_out) + len(self.partition.pre_cutoff)

    @property
    def ok(self) -> bool:
        return (
            self.partition.agrees
            and not self.response_mismatches
            and bool(self.provenance)
            and all(check.ok for check in self.provenance)
        )


def select_held_out_batch(
    frames: Mapping[str, pd.DataFrame], *, cutoff: pd.Timestamp, count: int, seed: int
) -> BatchSelection:
    """A seeded sample of patients holding a post-cutoff discharge, in full."""
    eligible = eligible_patients(frames, discharged_at_or_after=cutoff)
    if len(eligible) == 0:
        raise EmptyBatchError(f"no patient holds a cohort discharge at or after {cutoff.date()}")
    chosen = eligible.to_series().sample(n=min(count, len(eligible)), random_state=seed)
    narrowed = select_patients(frames, set(chosen.tolist()))

    cohort = build_cohort(narrowed["encounters"], narrowed["patients"]).frame
    split = split_at_cutoff(cohort, cutoff)
    return BatchSelection(
        frames=narrowed,
        cutoff=cutoff,
        seed=seed,
        patients_eligible=len(eligible),
        patients_selected=len(narrowed["patients"]),
        held_out_encounter_ids=frozenset(split.at_or_after["encounter_id"]),
        pre_cutoff_encounter_ids=frozenset(split.before["encounter_id"]),
    )


def summarize_scores(scores: Sequence[float]) -> ScoreSummary:
    """Count, mean, and five quantiles, or a bare count when there is nothing."""
    if not scores:
        return ScoreSummary(count=0)
    values = np.asarray(scores, dtype=float)
    p05, p25, p50, p75, p95 = (float(value) for value in np.percentile(values, [5, 25, 50, 75, 95]))
    return ScoreSummary(
        count=len(scores),
        mean=statistics.fmean(scores),
        minimum=float(values.min()),
        p05=p05,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        maximum=float(values.max()),
    )


def partition_log(
    selection: BatchSelection, logged: Iterable[predictions_module.StoredPrediction]
) -> LogPartition:
    """Split the log by cutoff side, naming anything either side does not explain."""
    held_out = []
    pre_cutoff = []
    unexpected = []
    seen = set()
    for row in logged:
        seen.add(row.encounter_id)
        if row.encounter_id in selection.held_out_encounter_ids:
            held_out.append(row)
        elif row.encounter_id in selection.pre_cutoff_encounter_ids:
            pre_cutoff.append(row)
        else:
            unexpected.append(row.encounter_id)
    return LogPartition(
        held_out=held_out,
        pre_cutoff=pre_cutoff,
        unscored_cohort_ids=tuple(sorted(selection.cohort_encounter_ids - seen)),
        unexpected_logged_ids=tuple(sorted(unexpected)),
    )


def choose_example(
    checks: Sequence[ProvenanceCheck],
    partition: LogPartition,
    encounter_id: str | None = None,
) -> ProvenanceCheck | None:
    """One prediction to write up: the median held-out score, or a named one."""
    by_encounter = {check.encounter_id: check for check in checks}
    if encounter_id is not None:
        if encounter_id not in by_encounter:
            raise KeyError(f"{encounter_id} is not among this run's predictions")
        return by_encounter[encounter_id]
    if not partition.held_out:
        return checks[0] if checks else None
    ranked = sorted(partition.held_out, key=lambda row: (row.score, row.encounter_id))
    return by_encounter.get(ranked[len(ranked) // 2].encounter_id)


def _read_log(
    dsn: str | None, encounter_ids: frozenset[str]
) -> list[predictions_module.StoredPrediction]:
    """This run's predictions, in write order, whatever else the log holds."""
    import psycopg

    from risk_scoring.db import database_url

    with psycopg.connect(dsn or database_url(), connect_timeout=5) as conn:
        rows = predictions_module.all_predictions(conn)
    return [row for row in rows if row.encounter_id in encounter_ids]


def run_batch(
    csv_dir: Path,
    repo_root: Path,
    *,
    cutoff: pd.Timestamp | None = None,
    count: int = DEFAULT_PATIENTS,
    seed: int = DEFAULT_SEED,
    dsn: str | None = None,
    port: int = DEFAULT_SERVICE_PORT,
    poster: EventPoster | None = None,
    example: str | None = None,
    announce: bool = False,
) -> BatchRunResult:
    """Post a held-out batch to a running service and verify what it logged."""
    cutoff = pd.Timestamp(TRAINING_CUTOFF, tz="UTC") if cutoff is None else cutoff
    frames = load_population(csv_dir)
    selection = select_held_out_batch(frames, cutoff=cutoff, count=count, seed=seed)
    stream = build_stream(selection.frames)

    events_by_type: dict[str, int] = {}
    for event in stream:
        kind = str(event["event_type"])
        events_by_type[kind] = events_by_type.get(kind, 0) + 1
    if announce:
        print(
            f"{selection.patients_selected} of {selection.patients_eligible} eligible patients, "
            f"{len(stream)} events, {len(selection.held_out_encounter_ids)} held-out discharges"
        )

    client = ServiceClient(port=port) if poster is None else None
    active: EventPoster = client if client is not None else poster  # type: ignore[assignment]
    try:
        version = active.version()
        acknowledged: dict[str, tuple[str, float | None]] = {}
        for event in stream:
            ack = active.post_event(event)
            if ack.get("scored"):
                encounter_id = str(event["payload"]["Id"])
                acknowledged[encounter_id] = (str(ack["input_hash"]), ack.get("score"))
    finally:
        if client is not None:
            client.close()

    logged = _read_log(dsn, selection.cohort_encounter_ids)
    partition = partition_log(selection, logged)
    mismatches = tuple(
        sorted(
            row.encounter_id
            for row in [*partition.held_out, *partition.pre_cutoff]
            if row.encounter_id in acknowledged
            and acknowledged[row.encounter_id] != (row.input_hash, row.score)
        )
    )

    scored = [*partition.held_out, *partition.pre_cutoff]
    checks = verify_predictions(scored, selection.frames["encounters"], repo_root)
    return BatchRunResult(
        selection=selection,
        version=version,
        events_by_type=events_by_type,
        partition=partition,
        all_scores=summarize_scores([row.score for row in scored]),
        held_out_scores=summarize_scores([row.score for row in partition.held_out]),
        provenance=tuple(checks),
        example=choose_example(checks, partition, example),
        response_mismatches=mismatches,
    )


def _format(summary: ScoreSummary, label: str) -> str:
    if summary.count == 0 or summary.p50 is None:
        return f"{label}: none"
    return (
        f"{label}: n={summary.count} mean={summary.mean:.6f} min={summary.minimum:.6f} "
        f"p05={summary.p05:.6f} p25={summary.p25:.6f} p50={summary.p50:.6f} "
        f"p75={summary.p75:.6f} p95={summary.p95:.6f} max={summary.maximum:.6f}"
    )


def report(result: BatchRunResult) -> str:
    """The run as plain text, with the held-out figures leading."""
    selection = result.selection
    lines = [
        f"population sample: {selection.patients_selected} of "
        f"{selection.patients_eligible} eligible patients (seed {selection.seed}, "
        f"cutoff {selection.cutoff.date()})",
        "events posted: "
        + ", ".join(f"{count} {kind}" for kind, count in sorted(result.events_by_type.items()))
        + f" ({result.events_posted} total)",
        f"held-out discharges: {len(selection.held_out_encounter_ids)} admitted, "
        f"{len(result.partition.held_out)} scored",
        f"pre-cutoff discharges (in sample, not held-out evidence): "
        f"{len(selection.pre_cutoff_encounter_ids)} admitted, "
        f"{len(result.partition.pre_cutoff)} scored",
        _format(result.held_out_scores, "held-out scores"),
        _format(result.all_scores, "all scores"),
        f"service reported: {result.version}",
        f"provenance: {sum(check.ok for check in result.provenance)} of "
        f"{len(result.provenance)} predictions reproduced",
    ]
    if result.example is not None:
        lines.append(f"worked example: {result.example.describe()}")
    for encounter_id in result.partition.unscored_cohort_ids:
        lines.append(f"NOT SCORED: {encounter_id} was admitted by the cohort rules")
    for encounter_id in result.partition.unexpected_logged_ids:
        lines.append(f"UNEXPECTED: {encounter_id} was logged but the cohort rules exclude it")
    for encounter_id in result.response_mismatches:
        lines.append(f"DISAGREES: {encounter_id} logged something other than what was returned")
    for check in result.provenance:
        if not check.ok:
            lines.append(f"BROKEN: {check.describe()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.scoring_run",
        description="Score a batch of held-out encounters through a running service.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="post a held-out batch and verify what it logged")
    run_parser.add_argument("--population", default="baseline")
    run_parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS)
    run_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run_parser.add_argument("--cutoff", default=TRAINING_CUTOFF)
    run_parser.add_argument("--port", type=int, default=DEFAULT_SERVICE_PORT)
    run_parser.add_argument("--example", default=None, help="encounter id to write up")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    csv_dir = repo_root / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")

    result = run_batch(
        csv_dir,
        repo_root,
        cutoff=pd.Timestamp(args.cutoff, tz="UTC"),
        count=args.patients,
        seed=args.seed,
        port=args.port,
        example=args.example,
        announce=True,
    )
    print()
    print(report(result))
    if not result.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
