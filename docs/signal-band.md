# Pre-registered signal band

Recorded 2026-08-27, before any model has been trained. The model this project trains is a fixture: its job is to be observed under operating conditions, not to make a clinical claim. A fixture can fail in two opposite directions, and both would be easy to rationalize after the fact, so the acceptable range is fixed here in advance and the commit date is the evidence.

## The band

Held-out discrimination must fall between a patient-grouped AUROC of 0.65 and 0.92, measured on patients absent from training.

Below 0.65, the synthetic population carries too little readmission signal for drift and realized-performance monitoring to mean anything, and the fallback ladder applies.

Above 0.92, the score is treated as evidence that the generator's rules have leaked into the features rather than as evidence of a good model. A model scoring above the ceiling is still used, and the leakage is disclosed wherever the model's performance is reported.

## Leakage disclosure

Any held-out AUROC above 0.92 is reported as suspected generator leakage in every public description of the model, alongside the raw number. The number is never adjusted, and no noise is added to the data or the labels to manufacture a more realistic-looking result. Fudging the fixture would corrupt the honesty of every claim built on top of it more than an implausibly clean fixture does.

One exception permits a change to the model rather than the disclosure. If scores concentrate so tightly at the extremes that prediction-distribution drift loses the resolution to detect anything, dropping the single most leaking feature is justified. Such a drop is documented with the feature name, the before-and-after score, and the drift-resolution problem that motivated it.

## Fallback ladder

If held-out AUROC lands below 0.65, remedies are attempted in this order, and each attempt is recorded whether or not it succeeds:

1. Tune the Synthea modules to widen risk variation across the population. This preserves the replay architecture and the frozen-population workflow, so it is tried first.
2. Switch training and replay to the UCI Diabetes 130-Hospitals dataset. This dataset carries no event timestamps, so the replay layer would have to synthesize them, and that cost is accepted only if the first remedy fails.

Neither step is a license to retune the band. The floor and ceiling stated here hold for the life of the project.
