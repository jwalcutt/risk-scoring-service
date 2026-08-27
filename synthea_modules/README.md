# Modified Synthea modules

This directory holds locally modified Synthea disease modules used to pre-generate variant patient populations. Each subdirectory is passed to Synthea's local-modules flag (`-d`), which overrides the built-in module of the same relative path.

## care_protocol

`care_protocol/medications/hypertension_medication.json` was extracted from `synthea-with-dependencies.jar` v4.0.0 and differs from the pristine module by exactly two lines. The step-1 `Lisinopril` and `Losartan` medication orders transition to the `HCTZ` order instead of `Terminal`, so first-line hypertension treatment becomes combination therapy (a thiazide added to the ACE inhibitor, or to the ARB on the allergy branch) rather than monotherapy. The chaining pattern copies what the pristine module already does in its very-high-blood-pressure branch.

A population generated with this module models a care-protocol change upstream of the scoring service: prescription patterns shift while every schema and unit stays identical. Because Synthea consumes randomness differently once a module changes, this population is statistically comparable to the baseline rather than patient-identical, which is why it is generated ahead of time instead of regenerated during a replay.
