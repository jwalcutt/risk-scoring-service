# Monitoring notes

Decisions and evidence for the monitoring layer: where it runs, what it compares a window against, and the tables it reads and writes. The replay harness is unchanged by any of this. It keeps posting events on the simulated clock and reading one field of its run row to decide whether to stop; monitoring writes that field and nothing else of the harness's.

One property runs through every decision below. An evaluation is a pure function of the tables and the boundary it evaluates. Every query is bounded by the boundary, never by "now", so the same boundary evaluated live during a run and evaluated afterwards over the finished tables must produce equal rows. Whatever the poll interval did, and however far ahead the harness had already run, the answer is the same.

## Where the monitor runs

A separate host process. `python -m risk_scoring.monitoring` opens its own connection, polls the open run row on the wall clock, and evaluates every 7-day boundary the simulated clock has passed.

The alternative was a hook inside `harness.drive`, fired when a tick crosses a boundary. It has no lag and needs no second process to run, and it was rejected anyway.

The harness is the generator layer, and it is where the failure injectors will live. The thing that detects a failure should not share a process, a connection, or a call stack with the thing that injects it: an injected fault and its detector inside one loop is an arrangement where a bug in either can quietly excuse the other. Two further reasons point the same way. The pause contract was already written for an outside writer, since `harness.run_replay` reads `runs.read_status` once per tick and never writes it, so a monitor that pauses the clock needs no new mechanism. And the retraining trigger built later consumes the alerts table with no harness involvement at all, so the harness would have been the wrong home for that reader too.

The cost is real and is accepted: an alert does not stop the clock instantly. The lag is one poll interval plus one tick, and the harness keeps posting through it. That is how a production monitor behaves, it changes no output, and the alert's own recorded instant is the boundary rather than the moment the clock happened to stop, so detection time does not depend on it.

## The evaluation grid

Fixed here, because everything downstream depends on it being fixed.

Boundaries sit at the run's `start_at` plus 7k days, for k from 1. Each boundary evaluates the window `[max(start_at, boundary - 30 days), boundary)`, half-open on `predictions.event_time`. Labels count toward a boundary when their `released_at` is at or before it. Refusals, once that table exists, count by their stream instant.

Two consequences worth stating. The first four boundaries see windows shorter than 30 days, because the window is clipped at the run's start rather than reaching back before it; what a short window is allowed to say is a separate decision and is not taken here. And a database holds at most one evaluation per run per boundary, enforced by a unique index, so a monitor that was killed and restarted neither skips a boundary nor writes a second row for one.

That index is on `(run_id, boundary)` rather than on `boundary` alone. Only one *unfinished* run may exist per database, but finished runs accumulate, and two runs over the same simulated span would produce the same boundaries. Keying on the boundary alone would refuse the second run's evaluations for no reason.

## What a window is compared against

The registered model's own training window, stored once in `monitoring_reference` and keyed to the model version.

Two alternatives were rejected. Comparing each window to the previous one needs no reference table and no data root, but it only sees sudden changes: a slow drift carries the reference along with the data and never alerts. Comparing against the replay's first 30 days is cheap and self-contained, but it anchors on roughly fifty discharges of whatever the run happened to open with, and says nothing about what the model was trained on.

The training window is what the model was actually fitted on, which is the comparison that means something when the question is whether the world has moved away from the model. It is fixed by the registered version, so it cannot drift underneath a run, and a later promotion writes a new row keyed to the new version rather than mutating this one. It lives in the database, so the live monitor, an offline audit, and Grafana all read one copy and the monitor needs no data root at evaluation time.

The reference is rebuilt the way `risk_scoring.gate` rebuilds its holdout: the training cutoff, split seed, and holdout fraction are read from the registered version's own training run, and the cohort, label, and feature modules are then run over the frozen export. Reusing `train.grouped_split` rather than re-deriving a split is what keeps the reference and training from disagreeing.

Two halves, deliberately drawn from different rows. The feature arrays come from the whole training window, which is the largest honest sample of what the model was fitted on. The score array comes from the patient-grouped holdout only, because in-sample scores are optimistically shifted and a reference built from them would make live score drift read low. Raw per-feature arrays are stored rather than summaries, roughly 12,000 rows by 14 columns, so that a statistic can see the whole sample instead of somebody's earlier idea of what mattered about it.

## Hand-rolled statistics

The statistics are written in this repository, with no new dependency.

scipy is the obvious alternative and is already installed, arriving transitively through lightgbm, mlflow, and scikit-learn. It was not chosen. Importing it directly would mean depending on something `pyproject.toml` does not declare, and the honest fix for that is to declare it, which is a dependency decision taken for two functions. An off-the-shelf drift library, evidently or nannyml or alibi-detect, was rejected more firmly: it brings a dependency tree and its own opinions about binning and reference windows, on a problem that is fourteen feature columns and one score.

The signal set is small enough that the arithmetic is short. A two-sample Kolmogorov-Smirnov statistic is a maximum difference of two empirical distribution functions; its p-value uses the asymptotic Kolmogorov series. A two-proportion test needs a normal CDF, which `math.erf` gives. Population stability index is a sum over bins. Each of those is a handful of lines with an exact expected value to test against, and each stays reproducible from its stored evaluation row, which a library's internal defaults would not guarantee.

`risk_scoring.evaluation` already holds the calibration bins, the patient-level bootstrap, and the equal-count binning rule, and was factored out of the gate so that monitoring could reuse the primitives without inheriting gate policy. One difference to watch when reusing it: `evaluation.bootstrap_ci` raises when a metric cannot be computed, while `replay.realized_performance` returns `None` for the same situation. A monitor reads early and sparse windows on every evaluation, where an uncomputable metric is a fact rather than an error, so it follows the `None` convention and guards the bootstrap path itself.

## The tables

Migration `0007_monitoring.sql` adds three.

`monitoring_reference` holds one row per registered model version: the versions it was trained under, the split seed and cutoff that define its holdout, the row counts, and the feature and score arrays as `jsonb`. `UNIQUE (model_name, model_version)` keeps it one row per version.

`monitoring_evaluations` holds one row per boundary per run: the window, the reference it was compared against, the hash of the threshold file in force, the counts the window saw, and the per-signal statistics as `jsonb`. Storing the thresholds hash on every row is what makes a changed threshold visible in the data rather than only in a commit.

`alerts` holds one row per signal that crossed its threshold at a boundary, with `sim_at` equal to the boundary, `raised_at` on the wall clock, and a nullable `acknowledged_at`.

`sim_at` is the instant the detection-time measurement reads, so it must be the boundary and not a wall-clock read. That is enforced by the schema rather than by the writer: `monitoring_evaluations` carries `UNIQUE (evaluation_id, boundary)` purely as a foreign-key target, and `alerts` references that pair, so an alert whose instant is not its own evaluation's boundary cannot be written at all. It is the same move the labels table makes with `released_at >= due_at`, where a rule that matters is a property of the table instead of a convention in the code. `UNIQUE (evaluation_id, signal)` says the rest: at one boundary a signal either alerts or it does not.

## The signals and their statistics

Nineteen signals, derived from the feature pipeline's own columns rather
than listed again, so a column added there cannot leave a signal unnamed.
Seven feature columns hold a continuous or count value and are compared by
a two-sample Kolmogorov-Smirnov test. Seven are binary comorbidity flags
and are compared by a pooled two-proportion test. The score is compared by
the same KS test as a continuous feature, which makes fifteen signals
carrying a p-value: the number the multiple-comparison arithmetic has to
account for at the freeze. Three more are counts rather than distributions,
namely prediction volume, refused events, and predictions from a model or
feature version other than the reference's. The nineteenth is population
stability index on the score, which is reported and never judged: it has no
p-value, and its conventional 0.1 and 0.25 bands are folklore rather than a
test.

The arithmetic is in `risk_scoring.monitoring.statistics`, and the module's
own docstring carries the judgment calls. Three are worth repeating here
because they were nearly got wrong.

The KS statistic is the largest gap between two right-continuous
distribution functions evaluated over the merged sample's distinct values.
The familiar shortcut, a running sum of steps over the concatenated sort,
is exact only when the two samples share no values. Both samples here are
heavily tied: five of the seven columns are small integers, and
`days_since_prev_discharge` has a mass point at 365.0 that means both
"capped" and "no prior discharge". For a reference of `[0, 0, 1, 1]`
against a window of `[0, 1]` the true gap is zero and the running sum
wanders to a half inside the tied block, and which half depends on the
sort's tie order, so the shortcut is not even a function of the two
samples. The cap itself costs no power, which is worth stating because it
looks as though it should: both distribution functions reach one at 365.0,
so a change in the no-prior-discharge rate is detected at the largest value
below the cap, where the gap equals the difference in cap mass exactly.

The two-sided normal tail goes through `math.erfc` rather than
`1 - normal_cdf`. The naive form returns exactly zero past about eight
standard errors, which is reachable: a flag going from absent to universal
in a twenty-row window against a hundred-row reference gives a z near
eleven. Against a real eleven-thousand-row reference the tail underflows to
exactly zero anyway, at a z near 106. That is recorded rather than clamped.
A floor invented to keep a logarithmic axis readable would be a number
nobody measured, and a p-value of zero still judges correctly.

PSI bins are the reference's own quantiles, deduplicated exactly. A tree
ensemble's leaf values repeat, so a collapsed bin is the ordinary case and
the realized bin count is stored: a sum over seven bins is not comparable
to one over ten. The epsilon substituted for an empty bin is applied to
both sides, not only to the window, because a reference bin can be empty
even after deduplication when an interpolated quantile edge brackets no
reference row, and a one-sided epsilon then takes the logarithm of a ratio
with zero underneath. With both sides floored every ratio lies between the
epsilon and its reciprocal, so the sum is finite as a property of the rule
rather than by luck. The value 1e-4 is the largest conventional choice that
still sits far below the smallest mass a sixty-row window can express,
1/60, so a bin holding one real observation can never read as empty.

One caveat that has to travel with every stored p-value. Because the
effective sample size is the harmonic combination of the two, it never
exceeds the window's own: 11,294 reference rows against a sixty-row window
give 59.68, half a percent better than an infinite reference. The window
alone sets the resolution, and at sixty rows the smallest gap the KS test
can flag at the asymptotic five percent point is 0.176 of the distribution.
Two further biases both push toward under-alerting: the asymptotic series
is roughly ten to fifteen percent too generous near p = 0.05 at these
sizes, and ties shrink the gap achievable under the null, which also gives
the sparse count columns a p-value with a handful of atoms rather than a
smooth null. So the p-value is a monotone drift score, not a calibrated
false-alarm rate, and it is not exchangeable across columns with different
tie structure. The thresholds come from the measured distribution over a
clean run for exactly that reason. If that distribution turns out unusable,
the dependency-free fix is a fixed-seed permutation p-value, which is exact
under ties and stays deterministic. It is named here and not built.

The two-proportion test has a matching weakness in the other direction. Its
normal approximation wants the pooled count at five or more, which at a
sixty-row window needs a rate above eight percent, and several comorbidity
flags are rarer than that. So the flag test sits outside its validity range
exactly where a case-mix shift would show. It is computed anyway and all
four counts are stored beside it, so a reader can apply the rule; declining
to report a rare flag is policy and does not belong in the arithmetic.
Fisher's exact test would remove the weakness through `math.lgamma` with no
new dependency, and is the named alternative rather than a silent
substitution for what was decided.

## What a short window may say

Decided here, and left open when the grid was fixed. Below the minimum
prediction count every drift statistic is `None`, the score's PSI with
them, and only volume, refusals, and version mismatches can raise an
alert. Twenty is the recommended count and a placeholder like every other
number in `configs/monitoring.toml`.

Two details of how that is expressed. The signal keys stay present with
`None` statistics rather than being left out, so a sparse window leaves a
gap in a dashboard series instead of removing the series, and the counts on
each suppressed result still say how sparse the window was. And the count
signals are deliberately unaffected by the rule, because an outage is
precisely the window too thin to compute drift on: a monitor that went
quiet exactly when the stream stopped would be useless.

The alternative was to skip judging a sparse window entirely. Suppressing
the statistics instead means `judge` has one rule for a missing number, and
the same rule covers a window where a statistic is uncomputable for its own
reasons, such as a flag that is absent from both samples.

## Alert rules as data

`configs/monitoring.toml` carries the cadence, the window, the minimum
count, the expected discharge rate, and one threshold per kind of signal.
Its digest is stored on every evaluation row, so a changed threshold shows
up in the data and not only in a commit.

One p-value floor covers all fifteen drift tests rather than fifteen floors
being written out. The arithmetic the freeze does is one calculation over
fifteen signals and fifty-two windows, not fifteen separate ones, and an
optional `signal_overrides` table is where a single signal's floor can be
raised later without restating the other fourteen.

That table is the one place a config in this repository refuses a key it
does not recognize. Every other loader ignores an unknown key. This one
cannot: these values become the evidence that the thresholds were fixed
before any failure was injected, and a misspelled threshold key that
silently took the shared default would make that evidence say something
untrue. Only signals carrying a p-value may be overridden, since volume and
the two counts are judged on counts and a p-value floor for them would
parse and mean nothing. Nothing in the file has a default, either: a
threshold nobody wrote down is not a threshold.

The expected discharge rate is a departure worth naming. The plan called
for comparing volume against the reference's own discharges-per-30-days,
but `monitoring_reference` stores no window span, only the training cutoff,
so that rate is not computable from the stored row. The alternative was a
migration adding span columns and a rebuilt reference row, which reopens a
table committed alongside the reference itself, for a number the clean
replay measures anyway. Rejected as well was the median of earlier windows,
which is blind to a slow decline and says nothing at the first boundary. So
the rate is a configured value, scaled by the window's actual length so the
short windows at the start of a run are compared fairly rather than reading
as an outage every time.

`judge` is pure and does nothing but compare. A statistic exactly at its
threshold does not alert and one past it does, which is the case an
operator will argue about, so it is pinned in a test rather than left to a
comparison operator nobody reread. A `None` statistic never alerts. And
`judge` refuses a config whose digest is not the one the evaluation was
computed under, because an evaluation row names its rules and judging it
against different rules would produce alerts the row cannot account for.

## What an evaluation reads

Two readings had to be settled before the queries made sense.

`prediction_count` is the window's, while `label_count` counts the labels
released inside the window. They answer different questions: how much the
service scored lately, and how much ground truth arrived lately. A label
pipeline that stalls shows in the second and not the first.

Realized performance is cumulative over the run to the boundary, not over
the drift window. Labels mature thirty simulated days after discharge and
the drift window is the last thirty days, so a realized metric over that
window would be empty at every boundary by construction. Cumulative with a
`released_by` bound is both non-empty and reproducible, and it is the
series a reader wants as labels mature. That bound is a new keyword on
`replay.realized_performance` rather than a second join written in the
monitoring code, because the rules about what an empty or single-class
window may report have to hold identically either way. Without it the same
window read after the run would pick up labels that had not been released
when the boundary passed, and a run evaluated live would disagree with the
same run evaluated afterwards.

Realized metrics stay report-only in this phase. At around fifty labelled
discharges and eight positives a window AUROC has a confidence interval too
wide to alert on. The freeze decides whether that stays true.

A boundary handed to the evaluator must lie on the grid. An arbitrary
instant would produce a perfectly valid-looking row that no second
evaluation could reproduce, and the unique index on `(run_id, boundary)`
would then be guarding a set nobody had defined. A run whose length is not
a whole number of cadences leaves its last few days unevaluated, which is
what a fixed grid does; stretching the final window to reach the end would
make one window a different size from the other fifty-one and quietly
change what its statistics mean.

Feature versions are compared by major and minor, reusing
`service.compatibility.major_minor`, because the feature module defines a
patch bump as moving no value. A version string this code cannot parse
counts as a mismatch, matching how the startup guard treats the same case.

Refusals are counted by a function that returns zero, because nothing
records a refusal yet: the harness stops on one. The signal has a column, a
threshold, and that seam now, so recording refusals changes one function
and no schema, and a test pins the zero so that change cannot be forgotten.

## Determinism against a replay, 2026-09-08

`tests/test_monitoring_evaluate_postgres.py` replays 162 cohort discharges
of a purpose-built population over ninety simulated days, 2025-01-01 to
2025-04-01, in three databases: straight through; paused at each of the
twelve boundaries and evaluated there before resuming; and paused and
resumed at an instant deliberately off the grid. All three produce equal
evaluations apart from the reference id the database assigned, and the
prediction and label tables are equal too. The first comparison is the
purity property, that how far past a boundary the harness had run cannot
change what the boundary says. The second carries the byte-identity
guarantee from the two tables through to what is derived from them.

The population is new because the existing ones do not suit a boundary
grid. The skew population scores single digits over four simulated months.
The gate population lives in 2022 and 2023 with a fourteen-month hole
between its two encounter clusters, sits entirely inside the training
window, and is what the fixture model is fitted on. One index inpatient
stay every sixteen hours from the training cutoff gives the density the
grid needs, and comorbidity codes are drawn from the feature module's own
lists by sorted order rather than by iterating a frozenset, because string
hashing is randomized between processes and the export has to be identical
every time.

Twelve boundaries, 2,160 ticks, 557 stream events posted, 99 labels
released, 5.8 seconds at max speed. The window counts are 5, 15, 27, 40,
then 50 to 57 for the rest, so the first two boundaries fall below the
minimum count and report nothing while the remaining ten report real
statistics. That is the point of the sizing: one run exercises both sides
of the rule. Cumulative realized count climbs from 0 to 90 and the realized
AUROC appears at the sixth boundary once both classes have matured, running
0.5000, 0.5562, 0.7263, 0.7143, 0.6886, 0.6434, 0.6623.

The worked evaluation at 2025-02-26, window 2025-01-27 to 2025-02-26: 57
predictions against 50.0 expected, 35 labels released in the window, 0
refusals, 0 version mismatches, realized count 36 at prevalence 0.2500 and
AUROC 0.7263. Its p-values are 0.03227 for `age_at_discharge`, 7.6e-10 for
`los_days`, 1 for `prior_inpatient_180d` and `days_since_prev_discharge`,
7.3e-06 for `prior_ed_180d`, 2.2e-49 for `active_medication_count`,
5.3e-27 for `active_disorder_count`, between 1e-98 and 1e-137 for the seven
flags, and 0.00054 for `score`; PSI 0.7382 over 10 bins. The whole payload
is twenty keys and 5.3 kB of JSON.

One finding that has to travel with those numbers: **this run says nothing
about the false-alarm rate.** Eleven of its twelve windows alert, on ten to
twelve signals each. That is the statistics working, not the thresholds
failing. The reference is the training window of the fixture model, which
was fitted on a different synthetic population from the one replayed, so
the two distributions genuinely differ, and they differ most on the columns
the two factories build differently: the gate population carries one
medication row and one condition row in total, so its medication count,
disorder count, and every flag rate are effectively zero against a
population where three quarters of patients carry a comorbidity. The
false-alarm rate is measured over a clean replay where the reference and
the replayed data come from one frozen population. What this file asserts
instead is that the comparison discriminates: `prior_inpatient_180d` and
`days_since_prev_discharge` both sit at p = 1 and `age_at_discharge` at
0.032, above the floor, so some signals are flagged and others are not. The
matched case is pinned in the pure tests, where an identical window scores
a KS statistic of exactly zero, a PSI of exactly zero, and raises nothing.

## The first reference, 2026-09-08

Built against the frozen baseline and the registry as it stands:

```bash
python -m risk_scoring.monitoring reference --population baseline
```

It stored one row for `readmission-risk` version 4, the pinned version, read from `configs/service.toml` rather than named on the command line so the reference and the model actually serving cannot silently disagree. Training cutoff 2025-01-01, split seed 20260101, 11,294 training-window rows over the 14 model input columns, and 2,245 holdout scores. Those two numbers reconcile with the training run's own: 9,049 train rows plus 2,245 holdout rows is the whole window, and the holdout is the same 2,245 the gate report describes.

The stored row is 312 kB including indexes, against the couple of megabytes budgeted for raw arrays, so nothing here argues for storing summaries instead. Re-running the command prints what is stored and writes nothing.

One thing the boot check caught that is worth knowing. The service image copies `src` at build time and installs it non-editable, so a new migration file is not in the container until the image is rebuilt. Running the `migrate` service against an empty database on a stale image applied 0001 through 0006 and stopped, with no error: the runner correctly applied every migration it could see, and it could not see the new one. A migration therefore lands in two places, the repository and the image, and `docker compose build` belongs between them. Proven both ways here, against a throwaway database on the same server: stale image reached 0006, rebuilt image reached 0007.
