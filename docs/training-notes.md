# Fixture model training notes

The first fixture model was trained and registered on 2026-08-27. The sections through the gate verdict below describe that run: its exact inputs and where its held-out discrimination landed relative to the pre-registered band in [signal-band.md](signal-band.md). Later retrains are appended as dated sections at the end.

## Provenance

Command: `python -m risk_scoring.train run --population baseline`, run from the repository root. The frozen baseline was verified against its committed manifest (`python -m risk_scoring.datagen verify baseline`, 18 files matching) immediately before the run. Code versions: cohort 1.0.0, features 1.0.0, labels 1.0.0. The classifier is LightGBM with the fixed parameters committed in `risk_scoring.train` (binary objective, 300 boosting rounds, seed 20260101); there is no tuning loop.

Only discharges with STOP strictly before 2025-01-01 entered training and holdout, which reserves the most recent year of generated history for replay. The cutoff sits eleven months before the generator's 2026-01-01 reference date, so every training label had its full 30-day maturation window. The holdout is grouped by patient with seed 20260101 at a 0.2 fraction, so no patient appears on both sides of the split. One detail worth preserving: the days-since-previous-discharge feature uses 365 as a real cap value rather than a missing marker, so nothing in this model depends on LightGBM's missing-value handling.

The model is registered as version 1 of `readmission-risk` (MLflow run `aa20db08f1fc434db193b80f344b2624`), with the versions, cutoff, seed, and metrics below logged on the run.

## Results

| Statistic | Value |
| --- | --- |
| Training rows | 9,049 |
| Training patients | 3,564 |
| Holdout rows | 2,245 |
| Holdout patients | 892 |
| Holdout prevalence | 12.0% |
| Holdout AUROC | 0.8617 |
| Holdout PR-AUC | 0.4491 |

## Band verdict

The held-out patient-grouped AUROC of 0.8617 lands inside the band of 0.65 to 0.92 that [signal-band.md](signal-band.md) committed before any training run, so neither fallback applies and no leakage disclosure is triggered. The score sits in the upper half of the band, which is consistent with Synthea's documented tendency toward clean, rule-driven histories, but it stays below the 0.92 ceiling that would mark suspected generator leakage. The number describes discrimination on generator output and carries no clinical meaning.

## Gate verdict

Version 1 passed the evaluation gate ([gate-notes.md](gate-notes.md)) on 2026-08-27, MLflow gate run `911bbf950a9643eb83a5d5f30989d732`. Every check passed: AUROC 0.8617 with a 95% patient-bootstrap interval of [0.8087, 0.9103], expected calibration error 0.0371 [0.0195, 0.0589], Brier score 0.0838 [0.0550, 0.1100], and an exact reproduction of the training run's logged holdout score. The subgroup table recorded near-chance discrimination for patients with the heart-failure flag (AUROC 0.48 on 85 holdout rows) and a below-average 0.73 in the 65-to-79 age band; subgroups are report-only, so neither affects the verdict, and both numbers are kept here as honest context for later monitoring.

## Retrain at features 1.1.0, 2026-09-08

Versions 1 through 3 were trained under `FEATURE_VERSION` 1.0.0. The prior-encounter predicate became strict on 2026-09-07: an encounter counts as prior only when its `STOP` is before the scoring discharge's `STOP`, which is what a causally replayed stream can know, and `FEATURE_VERSION` moved to 1.1.0. That left the registered models claiming a feature definition the pipeline no longer used, so the chained command was re-run against the frozen baseline after all three populations reverified clean:

```bash
python -m risk_scoring.pipeline retrain --population baseline --report gate_report.md
```

It registered version 4 of `readmission-risk` (MLflow run `32252e6d9476445f8515bd8127d6284e`) with code versions cohort 1.0.0, features 1.1.0, labels 1.0.0, and a passing gate report (gate run `23a9b0780b2047b4abafae4693f53844`). `configs/service.toml` now pins version 4.

Every number reproduces version 1's exactly: 9,049 training rows across 3,564 patients, 2,245 holdout rows across 892 patients, holdout prevalence 0.1203, AUROC 0.8617, PR-AUC 0.4491, ECE 0.0371, Brier 0.0838, and all thirteen subgroup AUROCs to four decimals.

## Why the numbers did not move

That identity is the useful result, and it is worth stating plainly rather than reading as a null. The strict rule can only change a discharge that shares its `STOP` instant with another inpatient encounter of the same patient, and no such pair exists in any frozen population: 0 of 12,308 baseline cohort discharges, 0 of 12,250 in `care_protocol`, and 0 of 53,663 in `demographic_shift`. The synthetic population the skew check uses in CI carries such a pair deliberately, which is where the rule is exercised.

So the version bump followed the rule stated in the `features` module docstring, that the minor number moves when a value on existing data *can* differ, and on this data none does. Two consequences worth keeping. Versions 1 through 4 are numerically the same model, so the retrain was about provenance rather than a wrong model in production: version 3's run recorded `feature_version = 1.0.0`, and anything derived from the registered model would have inherited that claim. And gating the incumbent version 3 under the new code passed, including `holdout_reproduced` at a difference of 0.000000, which is the gate's reproduction check reporting honestly rather than failing to notice.

## Registry schema note

This retrain was the first run after mlflow moved from 3.15.1 to 3.16.0, which advances the tracking-store schema. The existing `mlflow.db` had to be migrated on the host with `mlflow db upgrade` before any command could open the registry; the three registered versions and their gate tags came through unchanged. `docs/service-notes.md` records what this means for the container.
