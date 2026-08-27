# Evaluation gate

Recorded 2026-08-27, alongside the gate's first implementation. The gate is the fixed standard every candidate model must pass before it can serve traffic. It exists before any retraining automation so that later automation is judged against a standard set in advance, not one retrofitted around a system that already passes. This note records the criteria, the measurement method, and the verdict semantics; the code lives in `risk_scoring.gate` and `risk_scoring.evaluation`.

## How to run it

```bash
python -m risk_scoring.gate run --population baseline --report gate_report.md
```

The command gates the latest registered version of `readmission-risk` (or a specific one via `--model-version`), prints the full report, and exits nonzero on a failing verdict, which is what lets CI block on it. The holdout is rebuilt from the raw CSVs using the split seed, holdout fraction, and cutoff read from the model version's own training run, so the gate cannot drift from what training did.

## Checks

The verdict is pass only when every check passes.

| Check | Criterion |
| --- | --- |
| `auroc_above_band_floor` | Patient-grouped holdout AUROC at least 0.65 |
| `auroc_below_band_ceiling` | Patient-grouped holdout AUROC at most 0.92 |
| `ece_within_threshold` | Expected calibration error at most 0.05 |
| `holdout_reproduced` | Recomputed holdout AUROC matches the training run's logged value within 1e-6 |

The band checks reuse the range committed in [signal-band.md](signal-band.md). One asymmetry with training is deliberate: training discloses an out-of-band score but never blocks, while the gate blocks in both directions. A score above the ceiling fails the gate with a SUSPECTED LEAKAGE banner naming the raw number, because an inflated score is exactly what a candidate trained on a leaking pipeline would show; the raw number is always reported and never adjusted. A test proves this path works by training a candidate on a deliberately leaked row-level split (the same patients on both sides) and confirming the gate flags it.

The reproduction check guards the rebuild: if the cohort, feature, or label code drifted since the model was trained, the re-derived holdout would no longer score what training logged, and the gate refuses rather than silently evaluating a different dataset.

## Measurement method

Discrimination is patient-grouped AUROC and PR-AUC on the holdout. Calibration is expected calibration error over 10 equal-count bins: rows are ranked by score and split into bins of equal size, and ECE is the count-weighted mean absolute gap between each bin's mean score and its observed readmission rate. Equal-count binning is used because scores skew low at this prevalence and equal-width bins run empty. The Brier score and the full calibration table are reported alongside.

Every headline metric carries a 95% confidence interval from a percentile bootstrap with 1,000 replicates (seed 20260101) that resamples unique patients with replacement, keeping every row of a drawn patient. Rows of one patient are correlated, so row-level resampling would understate the intervals. Replicates whose resampled labels collapse to a single class are skipped and the count actually used is reported.

## Subgroups

The report breaks the holdout into thirteen subgroups: four age bands (18 to 49, 50 to 64, 65 to 79, 80 and over), both sexes, and the seven comorbidity flags. Subgroups are report-only in the current gate: they inform review but never move the verdict. A subgroup with fewer than 50 holdout rows, or with a single label class, has its metrics suppressed and the reason noted, because an AUROC on a handful of rows is noise presented as signal. The result type carries each subgroup's row count, patient count, prevalence, and AUROC, so a later per-subgroup comparison against an incumbent can be added without reshaping the report.

## Where results land

Each gate execution writes a new MLflow run tagged `run_type=gate` in the same experiment as training, carrying the metrics with their intervals, the per-subgroup AUROCs, and the full markdown report as an artifact. The gated model version itself is tagged with `gate_verdict` and `gate_run_id`, so the registry entry shows its gate status directly. The gate never modifies the finished training run, and it sets no registry alias: promotion remains a separate, deliberate act.

## First gate run

The registered version 1 (training notes in [training-notes.md](training-notes.md)) passed the gate on 2026-08-27: AUROC 0.8617 with a 95% interval of [0.8087, 0.9103], ECE 0.0371 [0.0195, 0.0589], Brier 0.0838 [0.0550, 0.1100], and an exact holdout reproduction. The subgroup table surfaced one honest weakness worth carrying forward: discrimination among patients with the heart-failure flag is near chance (AUROC 0.48 on 85 rows), and the 65-to-79 age band trails the overall score at 0.73. Neither moves the verdict under the current report-only policy, but both are the kind of number the subgroup table exists to surface.
