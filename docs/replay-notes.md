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

Deliberately absent from the row: tick size and max-speed mode, because pacing must not change what a run writes and so belongs to a process invocation, not to the run; and the splice list, which stays in the config file so that the date of a scheduled change never sits in a table anyone can read (see "Stream splicing").

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

Only clinical events are posted. The preload loads every patient row, so demographics are in state before the first tick, and the harness stream is `preload.replay_from`, the exact complement of what the preload took. A splice keeps that true for the population it brings in by preloading that population's history before its own splice instant, the same partition at a different boundary.

The run summary an invocation returns covers the simulated span it advanced, events posted by kind, discharges scored as the service acknowledged them, labels released and labels pending, ticks, wall time, the largest wall gap between consecutive ticks, and whether it stopped on a pause. The summary describes one invocation, not the whole run. Counters accumulated on the run row were considered and rejected: they would need a migration to duplicate what the prediction log and the row's wall timestamps already hold. `harness.report` renders the summary as text; the loop itself reads the run's status and never writes it.

## Where labels come from

A label is computed from the population export, through the same `labels.build_labels` that training uses, and released 30 simulated days after the discharge. The harness is the generator layer: it holds ground truth from the first tick and withholds only the release.

Deriving labels from service state was the alternative and was rejected because it would disagree with training. An encounter reaches the service at its own `STOP`, so a readmission stay that is still open on day 30 has not arrived yet, and a label read from state at that instant would say no readmission where the training label says yes. The two would then differ on exactly the discharges whose labels matter most, and realized performance would drift from the batch pipeline's numbers for a reason that has nothing to do with the model. `tests/test_replay_labels_postgres.py` asserts that every label the harness releases equals the batch label for that discharge, over a replayed synthetic population.

`risk_scoring.replay.release` computes the whole schedule up front: every cohort discharge in the export, its label, the instant it was discharged, and the instant its label falls due, sorted by due instant and then encounter id so the order is total. The `start` and `resume` commands build it beside the stream from the same frames. Instants are the stream's own timestamp strings, so "due at or before `sim_now`" is the same string comparison the events use.

## The labels table

Migration `0006_labels.sql` creates `labels`, one row per released label:

| Column | Holds |
| --- | --- |
| `label_id` | `bigserial`, assigned by the database |
| `prediction_id` | the scored discharge, unique and a foreign key to the log |
| `encounter_id` | the discharge, for reading without a join |
| `label` | 0 or 1, checked by the table |
| `label_version` | `labels.LABEL_VERSION` at release |
| `due_at` | the discharge instant plus 30 simulated days |
| `released_at` | the simulated instant of the tick that wrote the row |
| `recorded_at` | wall clock at write, the twin of `scored_at` |

`prediction_id` unique and a foreign key means a label attaches to exactly one scored discharge and an unscored discharge cannot have one. The uniqueness is also the release's idempotency key, the way the log's unique encounter id is the score's: a re-release is dropped by the conflict clause and reported as a no-op, never as an error.

The table carries a check that `released_at` is at or after `due_at`. That is the never-early rule as a property of the table rather than only of the harness: a label released before its discharge has had 30 days to mature is unrepresentable. The query that proves the property over a real run, restated in the tests and to be run against the real tables by the end-to-end replay, is

```sql
SELECT count(*) FROM labels l JOIN predictions p USING (prediction_id)
WHERE l.released_at < p.event_time + interval '30 days';
```

and must return zero.

`risk_scoring.label_log` is the only writer. `record_label` looks the prediction up inside the insert, so the caller never handles a prediction id; it commits on its own, matching the log and the run row. A label that falls due for a discharge the log holds no prediction for raises `UnscoredDischargeError` rather than being skipped: the schedule is built by the same cohort module the service scores with, so a missing prediction means the two disagree on the cohort, which is the kind of silent divergence this project refuses everywhere else. The run stops with that tick uncheckpointed, as it does on a refusal.

## Label release

A label is released in the first tick whose `sim_now` is at or past its due instant, and `released_at` is that tick's `sim_now`. After a wall-clock gap the loop catches up in a burst of ticks, and each label in the burst still carries the `sim_now` of its own tick, so the table records when a label became available on the simulated clock and never when the machine woke. Because ticks are one simulated hour and the due instant is on the second, a label is released at most an hour after it falls due, and exactly at it when the due instant lands on a tick boundary.

Within a tick, labels and events are merged by simulated instant and processed in that order: a label due at 13:10 lands before an event at 13:45 in the same tick and after one at 13:05. At an equal instant the label goes first, the tie rule the clock section fixed. The pure tests in `tests/test_replay_harness.py` pin the order, and pin that one-hour and seven-day ticks release the same labels in the same order relative to the posts, and that a wall clock jumping a week mid-run leaves both the labels and their release instants unchanged.

No label cursor is kept on the run row. The checkpoint follows a whole tick, so every label due at or before the checkpoint's `sim_now` was released by the tick that ended there, and what a resumed run still owes is exactly what falls due after it. A label due exactly at the run's start cannot exist, since the schedule holds only discharges at or after the start and a due instant is 30 days later. A run killed between a label write and the checkpoint re-releases that tick's labels on resume, and the table drops the repeats; the summary counts only rows actually written, so a re-release is not counted twice, as a re-posted discharge acknowledged unscored is not.

The run's share of the schedule is the discharges inside its span, inclusive at both ends. Earlier discharges are the ones the preload deliberately left unscored, and a discharge after the end is never posted; neither can be labelled, and counting them as pending would misdescribe the run. `labels_pending` on the summary is the count of scheduled labels due after the instant the invocation stopped: for a finished run that is exactly the discharges inside the final 30 days, the maturation boundary, and for a paused invocation it is what remains owed. Over the skew population's replay, six discharges are scored, five labels are released, and the discharge two days before the end is reported pending; a boundary population built for the purpose, with readmissions on day 29, on day 30 exactly, and one second past day 30, has an empty labels table when paused on day 29 and labels 1, 1, 0 once the clock passes day 30, which is the exit criterion in its own words.

Against the containers on 2026-09-05, a 200-patient sample of the baseline (seed 20260101) replayed at max speed from 2025-01-01 to 2025-07-01: 3,125 events posted, 73 discharges scored, 63 labels released and 10 pending, the loop itself taking 15 wall seconds. The never-early query returned zero rows; the earliest unlabelled discharge was `2025-06-03T00:45:03Z`, inside the final 30 days as every unlabelled one must be; and the lag from due instant to release instant ranged from 107 seconds to 3,447 seconds, never a full tick. Realized performance over `[2025-01-01, 2025-06-01)` came back as 63 labelled discharges, prevalence 0.0952, AUROC 0.8977; the comparison of those numbers against the batch pipeline over the real tables belongs to the end-to-end run.

## Realized performance

`risk_scoring.replay.realized.realized_performance(conn, start, end)` joins the log and the labels on `prediction_id` over the discharges whose `event_time` lies in `[start, end)` and returns the count, the prevalence, and the AUROC. A scored discharge whose label has not been released is not in the window's population; the maturation boundary means the last 30 days of a run are never known, and realized performance describes what is. A window with nothing in it reports no prevalence and no AUROC, and one holding a single class reports a prevalence and no AUROC, as `None` rather than an exception, because a monitoring job will read early windows on every evaluation.

`tests/test_replay_realized_postgres.py` asserts exact equality with the batch pipeline: after a complete replay of the skew population, the join's count, prevalence, and AUROC equal what the cohort, feature, and label modules and the registered model compute over the same discharges, with no tolerance. The scores are identical by the skew check and the labels by the label proof, so the metrics must be identical too, and a tolerance would hide a join bug.

## The commands

`python -m risk_scoring.replay` has four subcommands. `start` reads `configs/replay.toml` and its overrides, preloads history from before the start, opens the run row, and begins ticking. `resume` finds the open run, rebuilds the stream from the row's population, marks the row `running`, and continues from the checkpoint. `pause` marks the open run `paused` from any terminal. `status` prints the row: the run, its span, where the clock stands as an instant and a percentage of the span, the cursor's instant, and the wall times it was created and last written. `start` and `resume` take `--max-speed`, `--port` for the Compose service, and `--data-root` (default `data`) for the directory holding `<population>/csv`, so a test or a check script can point the real command at a synthetic or sampled export. The database is `RISK_SCORING_DATABASE_URL` or the Compose default, as everywhere else.

`start` preloads before it opens the row. A Ctrl-C during a preload that takes minutes therefore leaves no row behind, and a second `start` finishes the idempotent load; the run exists only once its history is in state. It refuses, before loading anything, if the database already holds an unfinished run.

`start` runs the clock as soon as the row exists, so one command takes a database from nothing to a ticking replay, and every later session is `resume` with no other argument. The alternative, a `start` that only prepared and left the first tick to `resume`, was rejected as one more step for the operator with nothing to show for it.

`resume` accepts a row whose status still reads `running`. A harness that died without pausing (a kill signal, a crash, a lost machine) leaves that behind, and its checkpoint is as good as a paused one. Nothing detects a second harness on the same row: the row cannot tell a live harness from a dead one, and a liveness heuristic on `updated_at` would have to be told the pacing to be right. The operator runs one harness per database, and the partial unique index already holds one run per database.

Every invocation ends by writing the row's status (`finished` at the end, `paused` otherwise, unless something outside already wrote it) and printing the summary.

## The pause contract

The harness pauses when the run row's status reads `paused`, whoever wrote it. That is the whole contract, and it is one field. The `pause` command writes it from another terminal now; a monitoring alert will write the same field later, and nothing alert-shaped exists until then.

`run_replay` reads the status once per tick, after the pacing sleep and before the tick's posts, so nothing is posted once a pause is observed and a pause written while the loop sleeps costs at most that one tick (0.625 wall seconds at the default acceleration). The checkpoint a pause leaves behind is the last complete tick's, so a resume posts nothing twice. Proven against a real service: a poster double writes `paused` from a second connection mid-tick, the loop finishes that tick, checkpoints it, and stops with the status untouched; resuming from that row completes the run to a log identical to an uninterrupted one.

Wall time spent paused never advances simulated time. The loop starts from the `sim_now` it is given and anchors its pacing schedule at that moment, so a run resumed a day later continues at the checkpoint with the next tick one simulated hour on, and the day off is not treated as a burst to catch up on. Tested with a fake wall clock jumped a day between two invocations.

## Ctrl-C

The first Ctrl-C asks the loop to pause. A signal handler installed for the length of the run sets a flag the loop reads where it reads the row's status, so the tick in progress finishes its posts and is checkpointed before the process exits, and the row is marked `paused`. Nothing is re-posted on resume. The alternative, letting `KeyboardInterrupt` propagate at once, was rejected because every Ctrl-C would then leave a re-post overlap in the wall-clock record for no gain: at one simulated hour per tick the wait is under a second.

The handler steps aside as soon as it has fired, so a second Ctrl-C reaches Python's own handler and the process dies at once. The last checkpoint stands, the row still says `running`, and `resume` re-posts the partial tick, which the service answers as no-ops; that path is the killed-and-restarted arm of the byte-identity test.

## Notifications

Every pause and every finish sends a desktop notification: on macOS one `osascript -e 'display notification ...'` call through `subprocess`, with the text quoted through `json.dumps`, no new dependency. Anywhere else the notifier prints the text to stderr, so a run on a CI box or a Linux host records the pause all the same. A notification that fails to display never fails the run; the row and the printed summary are the record. The notifier is a value the commands take, and the tests assert the call and its text through an injected double; no test touches a real desktop.

## Byte identity across a pause

`tests/test_replay_resume_postgres.py` is the proof for the exit criterion. Three arms run the synthetic skew population at max speed against an in-process service, each over its own throwaway database: straight through; paused at a chosen simulated instant and resumed from the row; killed after a chosen number of posts, before that tick's checkpoint, and restarted from the row. The pause points are the tick before the first event, the tick containing it, mid-stream, the tick before a discharge, the tick containing it, the tick containing the last event, the tick before the first label is released, and the instant it is released (a discharge plus 30 simulated days). The kill points are zero posts, one, mid-stream, one short of a discharge, just after it, and after the last. Every arm's log must equal the straight arm's row for row, the paused arm's two invocations must cover the stream exactly once between them, and the killed arm's overlap must be exactly the uncheckpointed tick, every repeat acknowledged unscored.

`prediction_id` and `scored_at` are excluded, for the reason the restart check recorded: the database assigns both, and a bigserial consumes a value even when the log's conflict clause drops a re-post. The labels table is compared at every one of those points too, minus `label_id`, `recorded_at`, and `prediction_id`, which the database assigns the same way. Two of the pause points exist for the labels: the tick before the first label falls due, so the resumed invocation has to release it, and the due instant itself, so the resume must not release it twice. No post-counted kill lands in a tick that releases a label (the skew labels fall due in quiet hours), so one more arm dies inside the release itself, after the first label's row is written and before the checkpoint: the resumed run re-releases it, the table drops the repeat, and both tables equal the straight run's.

`tests/test_replay_main_postgres.py` runs the commands themselves, as typed, from a throwaway repo root: `start` to the end, `start` refused while a run is open, `pause` honored at the next tick, Ctrl-C (a real `SIGINT` raised from inside a post) finishing the tick and marking the row, `resume` completing both a paused run and one whose process died, and `status`.

`scripts/check_replay_resume.py` is the same comparison against the Compose stack and real data, which is local-only and cannot run in CI. It writes a seeded sample of patients as its own export, runs `start` straight through against one throwaway database, then against another runs `start` as a subprocess, sends it `SIGINT` once the labels table holds a row, checks the row says `paused` with the clock inside the span, and runs `resume`. It compares the prediction log and the labels table separately and names the table on a mismatch. Waiting for a label rather than a prediction puts the pause at least 30 simulated days in, so the resumed invocation has to continue a label stream already under way; the earlier version paused on the first prediction, which was always before any label could be due and so proved nothing about them. Run on 2026-09-05 over 25 sampled baseline patients (seed 20260101) across the default span, after the labels table existed: the interrupted arm paused at `2025-02-10T06:00:00Z` with 3 predictions logged and 1 label released, resumed, and the two runs matched on all 15 predictions and all 14 labels, the fifteenth discharge being inside the final 30 days; the whole check took about two minutes, most of it the container builds. The earlier run of the same check, before labels, paused at `2025-01-10T16:00:00Z` with one prediction and matched on 15.

## Stream splicing

From a configured simulated date, a pre-generated variant export replaces the population that has been streaming as the source of every event, and nothing before that date changes. This is how a care-protocol change enters the stream without regenerating anything mid-replay, which Synthea cannot do reproducibly.

`risk_scoring.replay.splice` is pure. `segments` turns the config's start population and its `[[splice]]` list into half-open stretches, each naming one population: the first from the start instant, each later one from midnight of its splice date to the next. A segment owns the events of its population dated inside it, in stream order, and the labels of the discharges dated inside it. The pieces are joined into one stream and one schedule, and the tick loop runs over them unchanged; it knows nothing about splices. At the splice instant itself the incoming population wins: an event at exactly that instant is the incoming population's, and the outgoing population's event at that instant is dropped. The boundary matches the start's: a segment's population is preloaded before the segment's own instant and posted from it, so preload and stream are exact complements per population.

### Variant exports reuse the baseline's ids

Splicing was planned on the assumption that a variant population is different people with no rows in state. The exports say otherwise:

| Population | Patient ids shared with the baseline | Encounter ids shared | Shared rows differing in a column state stores |
| --- | --- | --- | --- |
| `care_protocol` | 11,203 of 11,564 | 561,745 of 713,541 | 253 patients (birth or death date), 1,120 encounters (STOP), 1,319 conditions (STOP), scattered from 1939 to 2026 |
| `demographic_shift` | 10,008 of 18,762 | 2,118 of 2,312,820 | every shared row: different people behind the same ids |

Synthea derives its id sequence from the seed, so a module edit leaves most ids in place and perturbs a scattering of later draws, and a demographic change produces new people under the first ten thousand of the same ids. Either way a variant's history cannot be loaded beside the baseline's under the same ids: the first shared key with a different stored value raises `EventConflictError`, as it should. Two ways of tolerating the overlap were rejected. Refusing only divergent keys would refuse `care_protocol` on its 1,120 divergent encounters, so no splice could run. Keeping the baseline's row where a shared key diverges would give post-splice discharges a history that is neither export's, so their features would not equal the batch pipeline over the variant export and the skew proof across the splice would need a tolerance.

Instead, every population other than the run's own is read through `populations.rekeyed`: patient and encounter ids are rewritten as `uuid5` over a fixed namespace and the population name, every other column stays byte-identical, and the shared modules cannot tell, since they use ids only to group and join. Variant patients are then distinct people in state, ids stay UUID-shaped so the prediction log does not name the population, and a splice back to the starting population is the same people under their own ids. The cost is that anything reading a spliced-in export against the log later, provenance verification or realized performance by encounter, must read that export through the same function; nothing does yet, and the function lives beside `load_population` so that it can.

### Where the history is loaded, and when

Every population the config names is preloaded by `start`, before the run row opens: the starting population before the start instant, each spliced-in one before its splice instant, through the same `preload_history` at a different boundary. With distinct ids, when a variant's history reaches state cannot change what the run writes, since nothing in state is read except by that patient's own discharges. Loading at the splice instant with the clock held was the alternative and was rejected: it needs a hook in the tick loop, a resume that re-runs an interrupted load, and it stalls the loop mid-run for minutes, which would land in `scored_at` as an outage that never happened. Loading everything up front makes `start` slower and nothing else.

Each population is read once and its full stream dropped before the next is read, so the memory held during the run is the run's own events and labels, not two whole populations. The `start` output says, per segment, how many events and labels the population contributes and from when.

### What resume trusts

`resume` rebuilds the spliced stream from the splice list in the config it is given (`--config`, the same file `start` used) and the population, span, and acceleration on the row, re-validated through the config's own range rules so a splice outside the row's span is refused. The row records no splices, on purpose: the date of a scheduled change must never sit in a table anyone can read, since the sealed incident schedule will later supply the same list. The limitation is that a splice list edited between sessions changes the stream silently, and nothing on the row can detect it. The operator keeps the file as it was; the check scripts pass the same file to both commands.

### Labels follow the population of origin

A discharge posted from the outgoing population keeps the outgoing export's label even when the readmission that decides it falls after the splice and is never posted. The label is the batch label over that export, held from the start as every label is; the splice changes the source of events, not of truth about events already posted. An incoming population's discharges from before the splice were preloaded, never posted, and so never labelled; the preload report counts them as left unscored.

### Proof

`tests/test_replay_splice.py` pins the segment rules pure: the boundary at the splice instant, the tie going to the incoming population, chained splices, rekeying only populations other than the run's own, labels by discharge instant, and the cursor crossing from the outgoing population's last event to the incoming one's first. `tests/test_populations.py` pins the rewrite: exactly the id columns, deterministic, namespace-specific, UUID-shaped, joins preserved, and invisible to the cohort, feature, and label modules.

`tests/test_replay_splice_postgres.py` runs the skew population until a splice at `2024-05-10` and a second synthetic population from it, one whose `p-fresh` reuses a skew id with a different birthdate, the collision the real exports produce. Against an in-process service: only the outgoing population is posted before the splice and only the incoming after, including a discharge at exactly the instant, and the outgoing population's three later discharges never; the log equals posting the spliced events one at a time after both preloads; the incoming population's index discharge logs a 180-day count of one and 38 days since a stay that was only ever preloaded, equal to the batch pipeline over the incoming export; every incoming patient id is absent from the outgoing export; pre-splice labels equal the outgoing export's, including a discharge readmitted after the splice by a stay never posted, and post-splice labels equal the incoming export's; the never-early query returns zero; and paused the tick before the splice, at it, or after it, or killed one post before or after the first incoming post, both tables equal a straight run's. `tests/test_replay_main_postgres.py` runs the commands over the same two exports: `start` with a splice reports both preloads and both segments, `resume` given the same config finishes to the straight log, a splice naming a missing export is refused before anything is read, and a splice outside the row's span is refused on resume.

### Against the containers

On 2026-09-06 the full exports were replayed at max speed from 2025-01-01 to 2025-03-01 with a splice to `care_protocol` at 2025-02-01, against the Compose stack and a throwaway database, from one `start`:

| | baseline, before 2025-01-01 | care_protocol, before 2025-02-01 |
| --- | --- | --- |
| history loaded | 632,255 encounters, 483,075 medications, 366,111 conditions, 11,557 patients | 635,968 encounters, 515,394 medications, 370,741 conditions, 11,564 patients |
| discharges left unscored | 11,294 | 11,279 |
| preload wall time | 47 s | 52 s |
| events contributed to the stream | 9,532 (January) | 190,196 (from February to the export's end) |

The two segments together are 199,728 events and 1,020 scheduled labels, of which the run's span holds 18,921 events (7,720 encounters, 6,912 medications, 4,289 conditions) and 93 scored discharges: 49 from the baseline before the splice and 44 from `care_protocol` after it. The loop took 57.8 wall seconds over 1,416 ticks with a largest gap of 0.5 s; the whole `start`, two reads and stream builds and two preloads included, took 250 s, with a peak resident set of 2.3 GB. All 47 released labels belong to pre-splice discharges, since a February discharge cannot mature before a March 1 end; 46 are pending, the earliest unlabelled discharge is `2025-01-30T00:18:24Z`, inside the final 30 days, and the never-early query returned zero. None of the 44 post-splice `patient_id` values appears in the baseline export, and all 44 post-splice predictions' logged features equal `build_features` over the rekeyed `care_protocol` export, which is the skew check across a real splice.

## Where the harness runs

The harness is a host process, `python -m risk_scoring.replay`, reading the local population export and posting to the Compose service. A fourth Compose service with `data/` mounted was the alternative. The data is local-only by design and the pause notification needs the desktop, so the host is where both already are.

## Why the prediction log needs no new column

The log records `event_time`, the discharge instant, and `scored_at`, the wall clock at write. An encounter arrives at its own `STOP`, so `event_time` already is the simulated instant it was scored at, and no simulated-scoring-time column is needed. `scored_at` stays the wall clock on purpose: a burst after the machine wakes shows there as a cluster of writes, which is how a stream outage is diagnosed after the fact while the simulated record stays correct.
