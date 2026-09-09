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

## The first reference, 2026-09-08

Built against the frozen baseline and the registry as it stands:

```bash
python -m risk_scoring.monitoring reference --population baseline
```

It stored one row for `readmission-risk` version 4, the pinned version, read from `configs/service.toml` rather than named on the command line so the reference and the model actually serving cannot silently disagree. Training cutoff 2025-01-01, split seed 20260101, 11,294 training-window rows over the 14 model input columns, and 2,245 holdout scores. Those two numbers reconcile with the training run's own: 9,049 train rows plus 2,245 holdout rows is the whole window, and the holdout is the same 2,245 the gate report describes.

The stored row is 312 kB including indexes, against the couple of megabytes budgeted for raw arrays, so nothing here argues for storing summaries instead. Re-running the command prints what is stored and writes nothing.

One thing the boot check caught that is worth knowing. The service image copies `src` at build time and installs it non-editable, so a new migration file is not in the container until the image is rebuilt. Running the `migrate` service against an empty database on a stale image applied 0001 through 0006 and stopped, with no error: the runner correctly applied every migration it could see, and it could not see the new one. A migration therefore lands in two places, the repository and the image, and `docker compose build` belongs between them. Proven both ways here, against a throwaway database on the same server: stale image reached 0006, rebuilt image reached 0007.
