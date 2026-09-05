# Replay notes

Decisions and evidence for the replay harness: the simulated clock, its configuration, and how a frozen population reaches the scoring service as a stream. The service itself is unchanged by any of this; it keeps taking events one at a time, in timestamp order, over HTTP, with a re-post answered as a no-op. Everything the harness writes must be a function of simulated time and the configured stream, never of wall time, tick size, or pacing. That property is what makes a paused-and-resumed replay byte-identical to an uninterrupted one, and every section below protects it.

## Where simulated time lives

Simulated time is a row in Postgres. Migration `0005_replay_runs.sql` creates `replay_runs`, one row per replay, and that row is both the clock and the checkpoint:

| Column | Holds |
| --- | --- |
| `run_id` | `bigserial`, assigned by the database |
| `population` | the frozen population streamed at the start |
| `start_at`, `end_at` | the simulated span, midnight UTC on the configured dates |
| `acceleration` | simulated days per wall minute, as the run was attended |
| `sim_now` | the run's simulated instant |
| `status` | `running`, `paused`, or `finished` |
| `cursor_at`, `cursor_kind`, `cursor_row` | the sort key of the last event posted |
| `created_at`, `updated_at` | wall clock, for the run summary and outage diagnosis |

A checkpoint file on disk was the alternative and was rejected because it would put the clock somewhere only the harness can see. The monitoring job and Grafana read `sim_now` from the same place they read everything else, pause and resume are writes to `status`, and a paused run survives a machine restart with nothing to reconcile between a file and the tables.

The cursor is the stream's own three-part sort key (arrival instant, kind order, the row itself), not an index into a list. A resumed harness rebuilds the stream from the population export and continues at the first event past the cursor, so a spliced or rebuilt stream resumes at the same event. The three columns are text and integer because Python compares them against `StreamEvent.sort_key`; the database never orders by them. A check constraint keeps them null together, since a half-written cursor is no checkpoint.

A partial unique index allows at most one row whose status is not `finished`. The prediction log is unique per encounter, so one database can host one scoring run at a time anyway; the index makes `resume` and `status` with no arguments unambiguous, and lets monitoring find the current run without being told its id. Starting a second run while one is open raises `OpenRunError` and names the remedy.

Deliberately absent from the row: tick size and max-speed mode, because pacing must not change what a run writes and so belongs to a process invocation, not to the run; and the splice list, which is plain data in the config file until stream splicing needs more.

`risk_scoring.replay.runs` is the only writer. Each of `create_run`, `checkpoint`, and `set_status` commits on its own, matching the state layer and the log: a checkpoint that a later failure could roll back would let the harness re-post from a point earlier than the one it reported.

## The wall clock

The harness reads `time.time()`. `time.monotonic()` was the alternative and was rejected because it stops while a Mac sleeps, so a laptop closed mid-replay would freeze simulated time silently and the run would look healthy when it was not. With `time.time()` the harness wakes to find simulated time far ahead and posts everything due in one burst. That burst is the stream outage the operating record expects to happen on its own the first time the machine sleeps; it leaves a visible gap in `scored_at` while the simulated record stays correct, which is exactly the diagnosis the log should support.

The clock is a value the tick loop takes as an argument, with `time.time` as the default in `risk_scoring.replay.clock`. The stream-emission tests drive the loop under a fake clock that jumps forward by a week mid-run and assert the posted sequence is unchanged, which is what makes this choice unable to affect outputs.

## The tick and the merged order

One tick is one simulated hour. At the default four simulated days per wall minute that is 0.625 wall seconds, short enough that a pause request is honored within a second and long enough that the process is not spinning. The last tick is clamped so the clock lands exactly on `end_at`.

At each tick the harness processes everything due at or before the new `sim_now` in one merged simulated-time order: events by their arrival instant, label releases by their due instant. When a label's due instant equals an event's arrival instant, the label goes first: what was already determined lands before what is new at that instant, for the same reason medications and conditions starting at a stay's end reach the service before that discharge in the stream order. An event at exactly `sim_now` is due.

Tick size is a constant, not a configuration value, because it must not change outputs: the same stream driven at one-hour and at seven-day ticks must post the identical sequence, and the stream-emission tests assert that.

A simulated instant is formatted with the stream's own timestamp format (`clock.instant`), after normalizing to UTC because the database driver returns timestamps in the session zone. "Due at or before `sim_now`" is then a string comparison against an event's arrival instant, with no parsing in the loop. Synthea timestamps are whole seconds, so `instant` refuses sub-second input rather than truncating it.

## Configuration and overrides

`configs/replay.toml` holds the population (default `baseline`), the start (default `2025-01-01`, the training cutoff, so every replayed discharge is held out), the end (default `2026-01-01`, the generation reference date, where the data stops), the acceleration (default 4), and an optional `[[splice]]` list. A test loads the committed file and asserts the start equals `train.TRAINING_CUTOFF` and the end equals the reference date in `configs/generation.toml`, read from those sources, so the three cannot drift apart.

Dates are unquoted TOML dates. The parser types them and refuses a malformed one before the loader sees it; the loader refuses a quoted string or a datetime by name, matching how the service config treats its model version. Range rules (start before end, positive acceleration, splices strictly inside the span and strictly increasing) live on the dataclass, so a config assembled from the file plus command-line overrides is validated by the same code as one read from the file alone. `add_config_arguments` and `apply_overrides` supply `--config`, `--population`, `--start`, `--end`, and `--acceleration`; a given override replaces the file's value and the whole is re-validated.

`--max-speed` is not configuration. It never sleeps between ticks, for tests and the byte-identity checks, and it must not change what a run writes, so it lives beside the commands as pacing and the config dataclass has no such field.

## Pre-start history

A discharge in January 2025 reads events from 2024: the 180-day counts and days-since-previous look back across the start. State must therefore hold every event dated before the start before the first replayed discharge arrives.

Two ways were considered. Posting the pre-start rows through the service at max speed would score every pre-start discharge, roughly a decade of in-sample ones, and at the measured posting rate of about 400 events per second take over an hour for the full baseline. Instead, `replay.preload_history` writes them straight into the state tables through `state.record_batch`, a batched insert with the same conflict clause the per-event path uses. A batch that comes up short re-runs its events one at a time, so an identical re-post stays a no-op and a divergent one raises `EventConflictError` exactly as it does per event; the batch has committed by then, so nothing acknowledged is rolled back. The preload is idempotent, and a load that died partway is resumed by running it again.

Pre-start means the same thing on both sides of the partition: an event is history when its arrival instant, the one the stream posts it at, is strictly before the start instant, and the replay posts everything at or after. Every patient row is loaded whatever its dates, since a discharge that outran its patient is refused by the service. The summary reports rows loaded by kind, rows already present, and the count of cohort discharges left unscored, the last taken from the shared cohort module through the same cutoff split training uses.

The baseline's pre-start volume is about 1.7 million rows (712,833 encounters, 553,590 medications, 411,201 conditions, 11,557 patients). The wall time of a real preload is to be recorded by the first end-to-end run. One known cost ahead of it: `stream.ordered_events` builds a dictionary per row through `iterrows`, which over that many rows will take minutes; a rewrite over `to_dict("records")` with identical output is the likely fix if the `start` command feels it.

Verified on the synthetic skew population in CI: after a preload, per-patient state read back byte-identical to per-event ingestion of the same rows, nothing dated at or after the start was in state, no prediction had been written, and a second preload loaded nothing new.

## The tick loop

`risk_scoring.replay.harness` is the loop. Each tick advances `sim_now` by one step, posts every event due at or before it through the service's existing contract, then writes the clock and the cursor to the run row. `risk_scoring.replay.emission.due_events` decides what is due: the events whose sort key is after the cursor and whose arrival instant is at or before the tick's instant, in stream order. The resume point is found by bisection over the stream's sort keys, so a stream rebuilt from the export, or spliced so that different rows precede the cursor, resumes at the same next event, and a year of hourly ticks does not scan the stream from the top each time.

What the loop posts is a function of the stream and simulated time only. `tests/test_replay_harness.py` drives it with no database and no real clock and asserts that one-hour and seven-day ticks post the identical sequence, that a wall clock jumping forward a week mid-run posts the identical sequence, and that pacing shows up only as time asked of `sleep`. `next_tick` takes the step as an argument for that test alone; production passes the constant.

Pacing is a schedule anchored when the loop starts, not a fixed sleep per tick. Tick `n` is due `n` tick-lengths of wall time after the anchor, and the loop sleeps only while it is ahead of that schedule. A fixed sleep was the alternative and was rejected because it could never produce the burst the wall-clock decision describes: after the machine sleeps, `time.time()` is far ahead of the schedule, so the loop runs ticks back to back with no waiting until it has caught up, and the burst lands in `scored_at` as the outage record.

The checkpoint is written after the tick's posts, never before. A refusal from the service propagates unchanged, so a 4xx stops the run rather than being counted and skipped; because the tick was never checkpointed, the next invocation re-posts it from the last checkpoint, and the service answers what it has already stored as a no-op. `tests/test_replay_harness_postgres.py` proves that against a real service: a run killed after a post and before its checkpoint, then resumed from the run row, re-posts the killed tick, every repeat comes back unscored, and the prediction log equals an uninterrupted run's row for row. The same file shows a replay after a preload logging exactly what posting the post-start events one at a time logs. The comparisons exclude `prediction_id` and `scored_at`, which the database assigns and which differ between two runs of one stream by design.

Only clinical events are posted. The preload loads every patient row, so demographics are in state before the first tick, and the harness stream is `preload.replay_from`, the exact complement of what the preload took. A splice has to keep that true for the population it brings in.

The run summary an invocation returns covers the simulated span it advanced, events posted by kind, discharges scored as the service acknowledged them, ticks, wall time, and the largest wall gap between consecutive ticks. Label counts are not on it yet: a field that always reads zero would misdescribe a run, so the labels substep adds those fields when it adds the numbers. `harness.report` renders the summary as text for the commands that arrive next; the loop itself does not touch the run's status, and pause, resume, and the commands are that substep's work.

## Where the harness runs

The harness is a host process, `python -m risk_scoring.replay`, reading the local population export and posting to the Compose service. A fourth Compose service with `data/` mounted was the alternative. The data is local-only by design and the pause notification needs the desktop, so the host is where both already are.

## Why the prediction log needs no new column

The log records `event_time`, the discharge instant, and `scored_at`, the wall clock at write. An encounter arrives at its own `STOP`, so `event_time` already is the simulated instant it was scored at, and no simulated-scoring-time column is needed. `scored_at` stays the wall clock on purpose: a burst after the machine wakes shows there as a cluster of writes, which is how a stream outage is diagnosed after the fact while the simulated record stays correct.
