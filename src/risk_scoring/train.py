"""Training entrypoint: raw Synthea CSVs to a registered readmission model.

One command trains the gradient-boosted fixture model from a frozen
population's CSV export, evaluates it on a patient-grouped holdout, and
registers the result in the MLflow registry:

    python -m risk_scoring.train run [--population baseline]
        [--cutoff 2025-01-01] [--seed 20260101] [--holdout 0.2]

Judgment calls this module fixes:

- Only cohort discharges with STOP strictly before the training cutoff
  enter training and holdout. The cutoff must sit at least the 30-day
  label window before the end of generated history so every label is
  fully mature; the default cutoff leaves roughly eleven months of slack
  against the generator's 2026-01-01 reference date.
- Label derivation may read encounters recorded after the cutoff: a
  late-December discharge legitimately looks up to 30 days ahead for its
  readmission. That lookahead is label maturation, not leakage, because
  no feature sees anything after its own discharge instant.
- The holdout is grouped by patient: no patient contributes rows to both
  sides, so evaluation measures generalization to unseen patients.
- Hyperparameters are fixed constants with no tuning loop. The model is
  a deliberately simple fixture; the operational loop around it is the
  project's subject.
- The signal band (docs/signal-band.md) is checked, logged, and printed
  but never blocks: an out-of-band result is disclosed, not suppressed.
- Training registers a new model version but sets no registry alias.
  Promoting a version to serve traffic is a separate, deliberate act.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from risk_scoring.cohort import COHORT_VERSION, build_cohort
from risk_scoring.features import FEATURE_VERSION, MODEL_INPUT_COLUMNS, build_features
from risk_scoring.labels import LABEL_VERSION, build_labels
from risk_scoring.populations import load_population
from risk_scoring.tracking import configure_tracking

MODEL_NAME = "readmission-risk"

TRAINING_CUTOFF = "2025-01-01"

HOLDOUT_FRACTION = 0.2

SPLIT_SEED = 20260101

SIGNAL_BAND = (0.65, 0.92)

LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 20260101,
    "deterministic": True,
    "verbosity": -1,
}

NUM_BOOST_ROUND = 300


@dataclass(frozen=True)
class TrainingResult:
    """What one training run produced, for reporting and tests."""

    run_id: str
    model_version: int
    auroc: float
    pr_auc: float
    n_train_rows: int
    n_holdout_rows: int
    n_train_patients: int
    n_holdout_patients: int
    holdout_prevalence: float
    in_band: bool


def filter_training_window(cohort: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep cohort rows whose discharge STOP is strictly before the cutoff."""
    stop = cohort["stop"]
    return cohort.loc[stop < cutoff].reset_index(drop=True)


def grouped_split(
    patient_ids: pd.Series[str], holdout_fraction: float, seed: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Patient-grouped train/holdout split over row positions."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(patient_ids.to_frame(), groups=patient_ids))
    return np.asarray(train_idx, dtype=np.intp), np.asarray(holdout_idx, dtype=np.intp)


def train(
    csv_dir: Path,
    repo_root: Path,
    *,
    cutoff: pd.Timestamp | None = None,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> TrainingResult:
    """Train, evaluate, and register one fixture model from a CSV export."""
    cutoff = pd.Timestamp(TRAINING_CUTOFF, tz="UTC") if cutoff is None else cutoff

    frames = load_population(csv_dir)
    encounters, patients = frames["encounters"], frames["patients"]
    medications, conditions = frames["medications"], frames["conditions"]

    cohort = filter_training_window(build_cohort(encounters, patients).frame, cutoff)
    labels = build_labels(cohort, encounters)
    features = build_features(cohort, encounters, medications, conditions)

    # Float inputs keep the logged model signature all-double, so schema
    # enforcement at serving time tolerates missing values.
    x = features.loc[:, list(MODEL_INPUT_COLUMNS)].astype("float64")
    y = labels["label"]
    train_idx, holdout_idx = grouped_split(features["patient_id"], holdout_fraction, seed)
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_holdout, y_holdout = x.iloc[holdout_idx], y.iloc[holdout_idx]

    booster = lgb.train(
        LGBM_PARAMS, lgb.Dataset(x_train, label=y_train), num_boost_round=NUM_BOOST_ROUND
    )
    holdout_scores = np.asarray(booster.predict(x_holdout), dtype=float)
    auroc = float(roc_auc_score(y_holdout, holdout_scores))
    pr_auc = float(average_precision_score(y_holdout, holdout_scores))
    in_band = SIGNAL_BAND[0] <= auroc <= SIGNAL_BAND[1]

    configure_tracking(repo_root)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "cohort_version": COHORT_VERSION,
                "feature_version": FEATURE_VERSION,
                "label_version": LABEL_VERSION,
                "training_cutoff": cutoff.date().isoformat(),
                "split_seed": seed,
                "holdout_fraction": holdout_fraction,
                "num_boost_round": NUM_BOOST_ROUND,
                "data_dir": str(csv_dir),
                **{f"lgbm_{key}": value for key, value in LGBM_PARAMS.items()},
            }
        )
        mlflow.log_metrics(
            {
                "holdout_auroc": auroc,
                "holdout_pr_auc": pr_auc,
                "holdout_prevalence": float(y_holdout.mean()),
                "n_train_rows": len(x_train),
                "n_holdout_rows": len(x_holdout),
                "n_train_patients": features["patient_id"].iloc[train_idx].nunique(),
                "n_holdout_patients": features["patient_id"].iloc[holdout_idx].nunique(),
            }
        )
        mlflow.set_tags(
            {
                "signal_band": f"{SIGNAL_BAND[0]}-{SIGNAL_BAND[1]}",
                "auroc_in_band": "true" if in_band else "false",
            }
        )
        mlflow.lightgbm.log_model(
            booster,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=x_train.head(5),
        )
        run_id = run.info.run_id

    client = MlflowClient()
    version = max(
        int(v.version)
        for v in client.search_model_versions(f"name = '{MODEL_NAME}'")
        if v.run_id == run_id
    )

    result = TrainingResult(
        run_id=run_id,
        model_version=version,
        auroc=auroc,
        pr_auc=pr_auc,
        n_train_rows=len(x_train),
        n_holdout_rows=len(x_holdout),
        n_train_patients=int(features["patient_id"].iloc[train_idx].nunique()),
        n_holdout_patients=int(features["patient_id"].iloc[holdout_idx].nunique()),
        holdout_prevalence=float(y_holdout.mean()),
        in_band=in_band,
    )
    _print_report(result, csv_dir, cutoff, seed)
    return result


def _print_report(result: TrainingResult, csv_dir: Path, cutoff: pd.Timestamp, seed: int) -> None:
    band_low, band_high = SIGNAL_BAND
    if result.in_band:
        verdict = f"inside the pre-registered band [{band_low}, {band_high}]"
    elif result.auroc > band_high:
        verdict = (
            f"ABOVE the pre-registered band ceiling {band_high}: "
            "suspected generator leakage, see docs/signal-band.md"
        )
    else:
        verdict = (
            f"BELOW the pre-registered band floor {band_low}: "
            "fallback ladder applies, see docs/signal-band.md"
        )
    print(f"data:               {csv_dir}")
    print(
        f"versions:           cohort {COHORT_VERSION}, features {FEATURE_VERSION}, "
        f"labels {LABEL_VERSION}"
    )
    print(f"training cutoff:    {cutoff.date().isoformat()} (STOP strictly before)")
    print(f"split seed:         {seed}")
    print(f"train rows:         {result.n_train_rows} ({result.n_train_patients} patients)")
    print(f"holdout rows:       {result.n_holdout_rows} ({result.n_holdout_patients} patients)")
    print(f"holdout prevalence: {result.holdout_prevalence:.4f}")
    print(f"holdout AUROC:      {result.auroc:.4f} -- {verdict}")
    print(f"holdout PR-AUC:     {result.pr_auc:.4f}")
    print(f"registered:         {MODEL_NAME} version {result.model_version} (run {result.run_id})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m risk_scoring.train",
        description="Train and register the readmission fixture model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="train from a frozen population's CSV export")
    run_parser.add_argument("--population", default="baseline")
    run_parser.add_argument("--cutoff", default=TRAINING_CUTOFF)
    run_parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    run_parser.add_argument("--holdout", type=float, default=HOLDOUT_FRACTION)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    csv_dir = repo_root / "data" / args.population / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"no CSV export at {csv_dir}; generate the population first")
    train(
        csv_dir,
        repo_root,
        cutoff=pd.Timestamp(args.cutoff, tz="UTC"),
        seed=args.seed,
        holdout_fraction=args.holdout,
    )


if __name__ == "__main__":
    main()
