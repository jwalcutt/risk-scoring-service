# Frozen population notes

Three synthetic populations were generated on 2026-08-27 and frozen; their checksum manifests live in `data_manifests/`. This note records the generation environment and the sanity numbers that supported the freeze decision.

## Provenance

Generator: Synthea v4.0.0 (`synthea-with-dependencies.jar`, SHA-256 pinned in [configs/generation.toml](../configs/generation.toml)), run under OpenJDK 21.0.12.1 (Homebrew) on macOS. All parameters come from the committed config: seed 20260101, clinician seed 20260101, reference date 2026-01-01, 10,000 living patients, Massachusetts. A 10-patient smoke run confirmed before full generation that CSV export produces encounters with ISO8601 `START` and `STOP` timestamps suitable for replay ordering.

The `care_protocol` variant was generated with the two-line hypertension module change described in [synthea_modules/README.md](../synthea_modules/README.md). The `demographic_shift` variant was generated with the age range restricted to 55 through 100. Both variants otherwise share every baseline parameter.

## Sanity numbers

| Statistic | baseline | care_protocol | demographic_shift |
| --- | --- | --- | --- |
| Patients exported | 11,557 | 11,564 | 18,762 |
| Encounters | 712,833 | 713,541 | 2,312,820 |
| Inpatient encounters | 13,890 | 13,844 | 55,740 |
| Crude 30-day readmission rate | 11.2% | 10.7% | 19.2% |
| Adult share at reference date | 82.5% | 82.5% | 100% |
| Rows with invalid timestamps | 0 | 0 | 0 |

The crude readmission rate counts any inpatient encounter followed by another inpatient encounter for the same patient within 30 days. It exists only to confirm the outcome is present at a workable base rate; the service's actual cohort and label definitions are separate, versioned code.

Patient counts exceed 10,000 because Synthea generates until the living population reaches the target, and deceased patients are exported too. Deceased patients also carry their complete encounter history, which is why the earliest encounter dates precede the 10-year history window applied to living patients. The demographic-shift variant is much larger on disk because an exclusively 55-and-older population accumulates far more clinical activity per patient.

## Adequacy

Baseline inpatient volume runs at roughly 630 encounters per calendar year across the recent decade. A multi-year training window therefore holds several thousand inpatient encounters with several hundred readmissions, which is sufficient for the planned model and its evaluation. The binding constraint is monitoring: a 30-day window over one year of replayed traffic holds roughly 50 scored encounters, enough to surface blunt data failures but statistically weak against subtle drift. The population was frozen at 10,000 with that tradeoff accepted and documented, rather than regenerated larger.
