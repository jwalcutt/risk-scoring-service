# Fixture model training notes

The first fixture model was trained and registered on 2026-08-27. This note records the exact inputs behind that run and where its held-out discrimination landed relative to the pre-registered band in [signal-band.md](signal-band.md).

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
