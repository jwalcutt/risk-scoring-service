"""Evaluation gate: pre-registered checks a candidate model must pass.

The gate turns held-out scores into a verdict and a written report. It
exists before any retraining automation so it acts as a fixed standard,
not a retrofit around a model that already passes it.

Judgment calls this module fixes:

- The gate blocks on an out-of-band AUROC in both directions. Training
  discloses an out-of-band score but never blocks (docs/signal-band.md);
  the gate's whole job is to stop a bad candidate, so above the ceiling
  it fails loudly as suspected leakage and below the floor it fails
  citing the fallback ladder. The raw number is always reported, never
  adjusted.
- Calibration is judged by expected calibration error over equal-count
  bins, threshold 0.05.
- Subgroups are report-only: four age bands, both sexes (joined from the
  patients frame), and the seven comorbidity flags. A subgroup with
  fewer than 50 holdout rows or a single label class has its metrics
  suppressed with a note; no subgroup moves the verdict.
- Confidence intervals come from the patient-level bootstrap in
  risk_scoring.evaluation, so every interval respects the patient
  grouping the split was made with.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from risk_scoring.cohort import COHORT_VERSION, build_cohort
from risk_scoring.evaluation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ECE_BINS,
    CalibrationBin,
    MetricCI,
    bootstrap_ci,
    calibration_bins,
    expected_calibration_error,
)
from risk_scoring.features import FEATURE_VERSION, MODEL_INPUT_COLUMNS, build_features
from risk_scoring.labels import LABEL_VERSION, build_labels
from risk_scoring.populations import load_population
from risk_scoring.tracking import configure_tracking
from risk_scoring.train import MODEL_NAME, SIGNAL_BAND, filter_training_window, grouped_split

ECE_THRESHOLD = 0.05

SUBGROUP_MIN_ROWS = 50

AUROC_REPRODUCTION_TOLERANCE = 1e-6

AGE_BAND_EDGES = (18, 50, 65, 80)

FLAG_COLUMNS = (
    "flag_chf",
    "flag_chronic_pulmonary",
    "flag_dementia",
    "flag_diabetes",
    "flag_malignancy",
    "flag_mi",
    "flag_renal_disease",
)


@dataclass(frozen=True)
class GateCheck:
    """One named pass/fail criterion with its measured value."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


@dataclass(frozen=True)
class SubgroupMetrics:
    """Report-only metrics for one subgroup of the holdout."""

    name: str
    n_rows: int
    n_patients: int
    prevalence: float
    auroc: float | None
    note: str


@dataclass(frozen=True)
class GateResult:
    """Everything one gate evaluation produced."""

    verdict: str
    checks: tuple[GateCheck, ...]
    auroc: MetricCI
    pr_auc: MetricCI
    ece: MetricCI
    brier: MetricCI
    calibration: tuple[CalibrationBin, ...]
    subgroups: tuple[SubgroupMetrics, ...]
    n_rows: int
    n_patients: int
    prevalence: float


def build_subgroups(features: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Boolean subgroup membership columns aligned to the feature rows."""
    age = features["age_at_discharge"].astype(int).reset_index(drop=True)
    gender = (
        features[["patient_id"]]
        .reset_index(drop=True)
        .merge(
            patients[["Id", "GENDER"]].rename(columns={"Id": "patient_id"}),
            on="patient_id",
            how="left",
        )["GENDER"]
    )
    subgroups = pd.DataFrame(
        {
            "age_18_49": (age >= AGE_BAND_EDGES[0]) & (age < AGE_BAND_EDGES[1]),
            "age_50_64": (age >= AGE_BAND_EDGES[1]) & (age < AGE_BAND_EDGES[2]),
            "age_65_79": (age >= AGE_BAND_EDGES[2]) & (age < AGE_BAND_EDGES[3]),
            "age_80_plus": age >= AGE_BAND_EDGES[3],
            "sex_male": (gender == "M").astype(bool),
            "sex_female": (gender == "F").astype(bool),
        }
    )
    for flag in FLAG_COLUMNS:
        subgroups[flag] = features[flag].reset_index(drop=True).astype(int) == 1
    return subgroups


def _subgroup_metrics(
    name: str,
    mask: npt.NDArray[np.bool_],
    y: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    patient_ids: pd.Series[str],
) -> SubgroupMetrics:
    y_sub = y[mask]
    n_rows = int(mask.sum())
    n_patients = int(patient_ids[mask].nunique())
    prevalence = float(y_sub.mean()) if n_rows else float("nan")
    if n_rows < SUBGROUP_MIN_ROWS:
        note = f"insufficient rows ({n_rows} < {SUBGROUP_MIN_ROWS}); metrics suppressed"
        return SubgroupMetrics(name, n_rows, n_patients, prevalence, None, note)
    if np.unique(y_sub).size < 2:
        note = "single label class; metrics suppressed"
        return SubgroupMetrics(name, n_rows, n_patients, prevalence, None, note)
    auroc = float(roc_auc_score(y_sub, scores[mask]))
    return SubgroupMetrics(name, n_rows, n_patients, prevalence, auroc, "")


def _band_checks(auroc: float) -> tuple[GateCheck, GateCheck]:
    floor, ceiling = SIGNAL_BAND
    above_floor = auroc >= floor
    below_ceiling = auroc <= ceiling
    floor_detail = (
        f"holdout AUROC {auroc:.4f} meets the pre-registered floor {floor}"
        if above_floor
        else (
            f"holdout AUROC {auroc:.4f} is below the pre-registered floor {floor}; "
            "the fallback ladder applies (docs/signal-band.md)"
        )
    )
    ceiling_detail = (
        f"holdout AUROC {auroc:.4f} is within the pre-registered ceiling {ceiling}"
        if below_ceiling
        else (
            f"SUSPECTED LEAKAGE: holdout AUROC {auroc:.4f} exceeds the "
            f"pre-registered ceiling {ceiling} (docs/signal-band.md)"
        )
    )
    return (
        GateCheck("auroc_above_band_floor", above_floor, auroc, floor, floor_detail),
        GateCheck("auroc_below_band_ceiling", below_ceiling, auroc, ceiling, ceiling_detail),
    )


def evaluate(
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    patient_ids: pd.Series[str],
    subgroups: pd.DataFrame | None = None,
    *,
    expected_auroc: float | None = None,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> GateResult:
    """Judge held-out scores against the pre-registered gate checks."""
    ids = patient_ids.reset_index(drop=True)

    auroc = bootstrap_ci(
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        y,
        scores,
        ids,
        n_replicates=n_replicates,
        seed=seed,
    )
    pr_auc = bootstrap_ci(
        lambda yt, ys: float(average_precision_score(yt, ys)),
        y,
        scores,
        ids,
        n_replicates=n_replicates,
        seed=seed,
    )
    ece = bootstrap_ci(
        expected_calibration_error, y, scores, ids, n_replicates=n_replicates, seed=seed
    )
    brier = bootstrap_ci(
        lambda yt, ys: float(brier_score_loss(yt, ys)),
        y,
        scores,
        ids,
        n_replicates=n_replicates,
        seed=seed,
    )

    checks = list(_band_checks(auroc.value))
    ece_passed = ece.value <= ECE_THRESHOLD
    ece_detail = (
        f"ECE {ece.value:.4f} is within the threshold {ECE_THRESHOLD}"
        if ece_passed
        else f"ECE {ece.value:.4f} exceeds the threshold {ECE_THRESHOLD}"
    )
    checks.append(
        GateCheck("ece_within_threshold", ece_passed, ece.value, ECE_THRESHOLD, ece_detail)
    )
    if expected_auroc is not None:
        gap = abs(auroc.value - expected_auroc)
        reproduced = gap <= AUROC_REPRODUCTION_TOLERANCE
        repro_detail = (
            f"recomputed holdout AUROC {auroc.value:.6f} reproduces the training run's "
            f"logged {expected_auroc:.6f}"
            if reproduced
            else (
                f"recomputed holdout AUROC {auroc.value:.6f} differs from the training "
                f"run's logged {expected_auroc:.6f}; the rebuilt holdout does not match "
                "the one training evaluated"
            )
        )
        checks.append(
            GateCheck(
                "holdout_reproduced", reproduced, gap, AUROC_REPRODUCTION_TOLERANCE, repro_detail
            )
        )

    subgroup_metrics: list[SubgroupMetrics] = []
    if subgroups is not None:
        for name in subgroups.columns:
            mask = subgroups[name].to_numpy(dtype=bool)
            subgroup_metrics.append(_subgroup_metrics(name, mask, y, scores, ids))

    return GateResult(
        verdict="pass" if all(check.passed for check in checks) else "fail",
        checks=tuple(checks),
        auroc=auroc,
        pr_auc=pr_auc,
        ece=ece,
        brier=brier,
        calibration=calibration_bins(y, scores),
        subgroups=tuple(subgroup_metrics),
        n_rows=len(y),
        n_patients=int(ids.nunique()),
        prevalence=float(y.mean()),
    )


def render_report(
    result: GateResult,
    *,
    model_version: int,
    candidate_run_id: str,
    data_dir: str,
    cutoff: str,
    seed: int,
) -> str:
    """Render one gate evaluation as a markdown report."""
    lines = [
        f"# Gate report: {MODEL_NAME} version {model_version}",
        "",
        f"**Verdict: {result.verdict.upper()}**",
        "",
    ]
    leakage = [
        check
        for check in result.checks
        if check.name == "auroc_below_band_ceiling" and not check.passed
    ]
    if leakage:
        lines += [f"**{leakage[0].detail}**", ""]
    lines += [
        f"- candidate run: {candidate_run_id}",
        f"- data: {data_dir}",
        f"- training cutoff: {cutoff}",
        f"- split seed: {seed}",
        f"- versions: cohort {COHORT_VERSION}, features {FEATURE_VERSION}, labels {LABEL_VERSION}",
        f"- holdout: {result.n_rows} rows, {result.n_patients} patients, "
        f"prevalence {result.prevalence:.4f}",
        "",
        "## Checks",
        "",
        "| check | status | value | threshold | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in result.checks:
        status = "pass" if check.passed else "FAIL"
        lines.append(
            f"| {check.name} | {status} | {check.value:.6f} | {check.threshold} | {check.detail} |"
        )
    lines += [
        "",
        "## Metrics",
        "",
        "| metric | value | 95% CI | replicates used |",
        "| --- | --- | --- | --- |",
    ]
    for label, metric in (
        ("AUROC", result.auroc),
        ("PR-AUC", result.pr_auc),
        ("ECE", result.ece),
        ("Brier", result.brier),
    ):
        lines.append(
            f"| {label} | {metric.value:.4f} | [{metric.ci_low:.4f}, {metric.ci_high:.4f}] "
            f"| {metric.n_replicates_used} |"
        )
    lines += [
        "",
        "## Calibration",
        "",
        "| bin | score range | mean score | observed rate | rows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, cal_bin in enumerate(result.calibration, start=1):
        lines.append(
            f"| {i} | [{cal_bin.lower:.4f}, {cal_bin.upper:.4f}] | {cal_bin.mean_score:.4f} "
            f"| {cal_bin.observed_rate:.4f} | {cal_bin.count} |"
        )
    if result.subgroups:
        lines += [
            "",
            "## Subgroups",
            "",
            "| subgroup | rows | patients | prevalence | AUROC | note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for subgroup in result.subgroups:
            auroc_cell = f"{subgroup.auroc:.4f}" if subgroup.auroc is not None else "-"
            lines.append(
                f"| {subgroup.name} | {subgroup.n_rows} | {subgroup.n_patients} "
                f"| {subgroup.prevalence:.4f} | {auroc_cell} | {subgroup.note} |"
            )
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class GateRunOutcome:
    """What one gate execution produced, for the CLI and for tests."""

    result: GateResult
    report: str
    model_version: int
    candidate_run_id: str
    gate_run_id: str


def _resolve_model_version(client: MlflowClient, model_version: int | None) -> int:
    versions = [int(v.version) for v in client.search_model_versions(f"name = '{MODEL_NAME}'")]
    if not versions:
        raise ValueError(f"no registered versions of {MODEL_NAME}; train a model first")
    if model_version is None:
        return max(versions)
    if model_version not in versions:
        raise ValueError(f"{MODEL_NAME} has no version {model_version}; found {sorted(versions)}")
    return model_version


def run_gate(
    csv_dir: Path,
    repo_root: Path,
    *,
    model_version: int | None = None,
    report_path: Path | None = None,
) -> GateRunOutcome:
    """Gate one registered model version against its re-derived holdout.

    The holdout is rebuilt from the raw CSVs with the split seed, holdout
    fraction, and cutoff read from the model version's own training run,
    so the gate cannot drift from what training did; the reproduction
    check confirms the rebuilt holdout scores exactly what training
    logged. Results are written to a new MLflow run tagged run_type=gate
    and onto the model version itself, never onto the finished training
    run.
    """
    configure_tracking(repo_root)
    client = MlflowClient()
    version = _resolve_model_version(client, model_version)
    registered = client.get_model_version(MODEL_NAME, str(version))
    candidate_run_id = registered.run_id or ""
    training_run = client.get_run(candidate_run_id)
    cutoff = pd.Timestamp(training_run.data.params["training_cutoff"], tz="UTC")
    seed = int(training_run.data.params["split_seed"])
    holdout_fraction = float(training_run.data.params["holdout_fraction"])
    expected_auroc = training_run.data.metrics["holdout_auroc"]

    frames = load_population(csv_dir)
    encounters, patients = frames["encounters"], frames["patients"]
    medications, conditions = frames["medications"], frames["conditions"]

    cohort = filter_training_window(build_cohort(encounters, patients).frame, cutoff)
    labels = build_labels(cohort, encounters)
    features = build_features(cohort, encounters, medications, conditions)
    x = features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    y = labels["label"].to_numpy(dtype=float)
    _, holdout_idx = grouped_split(features["patient_id"], holdout_fraction, seed)

    features_holdout = features.iloc[holdout_idx].reset_index(drop=True)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    scores = np.asarray(model.predict(x.iloc[holdout_idx]), dtype=float)

    result = evaluate(
        scores,
        y[holdout_idx],
        features_holdout["patient_id"],
        build_subgroups(features_holdout, patients),
        expected_auroc=expected_auroc,
    )
    report = render_report(
        result,
        model_version=version,
        candidate_run_id=candidate_run_id,
        data_dir=str(csv_dir),
        cutoff=cutoff.date().isoformat(),
        seed=seed,
    )

    with mlflow.start_run() as gate_run:
        mlflow.set_tags(
            {
                "run_type": "gate",
                "candidate_run_id": candidate_run_id,
                "candidate_model_version": str(version),
                "gate_verdict": result.verdict,
            }
        )
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "model_version": version,
                "data_dir": str(csv_dir),
                "training_cutoff": cutoff.date().isoformat(),
                "split_seed": seed,
                "holdout_fraction": holdout_fraction,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "ece_bins": ECE_BINS,
            }
        )
        metrics = {
            "gate_auroc": result.auroc.value,
            "gate_auroc_ci_low": result.auroc.ci_low,
            "gate_auroc_ci_high": result.auroc.ci_high,
            "gate_pr_auc": result.pr_auc.value,
            "gate_pr_auc_ci_low": result.pr_auc.ci_low,
            "gate_pr_auc_ci_high": result.pr_auc.ci_high,
            "gate_ece": result.ece.value,
            "gate_ece_ci_low": result.ece.ci_low,
            "gate_ece_ci_high": result.ece.ci_high,
            "gate_brier": result.brier.value,
            "gate_brier_ci_low": result.brier.ci_low,
            "gate_brier_ci_high": result.brier.ci_high,
            "gate_n_rows": float(result.n_rows),
            "gate_n_patients": float(result.n_patients),
            "gate_prevalence": result.prevalence,
        }
        for subgroup in result.subgroups:
            if subgroup.auroc is not None:
                metrics[f"gate_subgroup_auroc__{subgroup.name}"] = subgroup.auroc
        mlflow.log_metrics(metrics)
        mlflow.log_text(report, "gate_report.md")
        gate_run_id = gate_run.info.run_id

    client.set_model_version_tag(MODEL_NAME, str(version), "gate_verdict", result.verdict)
    client.set_model_version_tag(MODEL_NAME, str(version), "gate_run_id", gate_run_id)

    if report_path is not None:
        report_path.write_text(report)
    return GateRunOutcome(
        result=result,
        report=report,
        model_version=version,
        candidate_run_id=candidate_run_id,
        gate_run_id=gate_run_id,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.gate",
        description="Gate a registered readmission model against its holdout.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="gate a registered model version")
    run_parser.add_argument("--population", default="baseline")
    run_parser.add_argument("--model-version", type=int, default=None)
    run_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    csv_dir = repo_root / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")
    outcome = run_gate(
        csv_dir, repo_root, model_version=args.model_version, report_path=args.report
    )
    print(outcome.report)
    if outcome.result.verdict != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
