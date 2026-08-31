# Service notes

Recorded 2026-08-27, alongside the first infrastructure for the scoring service: a Compose-managed Postgres, the schema migration runner, per-patient event state, the serving feature seam, and the HTTP service that loads a pinned model and validates incoming events. The endpoint does not yet write state or produce scores; those land when the ingestion path is wired to the state layer. This note records the storage decisions, the ingestion contract, the input hash definition, the version-pinning rules, and the registry-access decision, and it will grow as the service does.

## Why Postgres

Prediction logs, per-patient state, labels, and alerts all land in one Postgres instance run by Docker Compose. Postgres is chosen for Grafana compatibility, not scale: Grafana has a first-class Postgres datasource and no equivalent for SQLite, and the dashboards later read these tables directly. The event volume (roughly 630 inpatient encounters per simulated year) would fit in anything.

## Schema management

Schema changes are plain SQL files in `src/risk_scoring/migrations/`, named `NNNN_description.sql` and applied in numeric order by a small runner in `risk_scoring.db`. The runner records each applied migration in a `schema_migrations` table with a SHA-256 checksum of the file it executed. Alembic was considered and rejected: there are no ORM models to autogenerate from, the schema is a handful of hand-reviewed tables, Alembic ships no type information (so strict type checking would need a blanket exemption), and a runner the repository owns executes identically in tests, Compose, and CI with zero added dependencies.

The runner's rules:

- Migrations apply in filename-number order. Duplicate numbers are an error; gaps are allowed, since renumbering old files causes more churn than a gap costs.
- An applied migration is frozen. If a recorded migration's file disappears or its checksum changes, the runner refuses to proceed rather than guessing. Schema fixes go in a new migration.
- Each migration runs in its own transaction together with its bookkeeping row. Postgres DDL is transactional, so a failing migration rolls back completely and earlier migrations stay recorded.
- Migration 0001 creates the `schema_migrations` table itself; the runner treats a database without that table as having nothing applied. Advisory locking is deliberately omitted: this is a single-operator system, and the primary key on the migration number makes a concurrent double-apply an error instead of a corruption.

```bash
python -m risk_scoring.db migrate
```

`migrate` applies whatever is pending and prints what it did; `status` lists applied and pending migrations without changing anything. Both read the target database from `RISK_SCORING_DATABASE_URL` and default to the local Compose instance.

## Running Postgres locally

```bash
docker compose up -d postgres
```

The image is pinned to `postgres:17.6-alpine`, digest `sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94`; bump the tag and the digest together. Data persists in the named volume `postgres-data`. Credentials come from the environment with committed development defaults (`risk`/`risk`, database `risk_scoring`), overridable via a local `.env` file that stays out of git. The container listens on host port 5433, not 5432, so a Postgres already installed on the machine never collides with it. The healthcheck runs `pg_isready`, which is what later Compose services wait on before starting.

## Database tests

The migration runner's pure logic (ordering, duplicate detection, checksum freezing) is tested without any database. Tests that need a real server request the `db_conn` fixture, which creates a uniquely named database on the configured server, applies every migration, and drops the database with FORCE on teardown so a leaked connection can never hang the suite. Each test therefore starts from a fresh, fully migrated database, and the isolation is itself asserted by a pair of tests that would see each other's tables on a shared database.

When no server is reachable at `RISK_SCORING_DATABASE_URL` (probed once per session with a 2-second timeout), the database tests skip, and the skip reason says how to start Compose. Setting `RISK_SCORING_REQUIRE_DB=1` turns that skip into a failure. CI sets it and provides a Postgres service container pinned to the same image as Compose, so the database tests run on every push and can never be skipped silently there. A local run without Docker still executes everything else and reports the skips in the summary.

## Per-patient state

The service keeps each patient's raw event history, not incremental aggregates. Four tables (`patients`, `encounters`, `medications`, `conditions`) hold exactly the Synthea columns the shared cohort and feature modules read, and serving-time features are recomputed from that history through the same `risk_scoring.features` module the training pipeline runs. Incremental aggregates were rejected because they would re-encode the window logic outside the shared module and then need an independent equivalence proof against the batch pipeline; recomputing from raw history makes "one feature module, shared verbatim" a structural property rather than something a test chases. The event volume (roughly 630 inpatient encounters per simulated year) is nowhere near large enough to force aggregates.

Every column is text and every value is a verbatim string from the source row. Batch training, the evaluation gate, and the cohort builder all read CSVs through one function, `risk_scoring.populations.load_population`, which keeps missing values as empty strings rather than NaN, and the shared modules do their own timestamp parsing, so state read-back returns frames byte-identical to a batch CSV load: uppercase Synthea column names, export column order, empty string (never NULL) for missing. ISO8601 timestamps sort lexicographically in time order, so plain btree indexes on text serve the by-patient chronological access pattern; encounters get a `(patient, start)` index, while medications and conditions are covered by composite primary keys that lead with the patient id.

## Event ingestion and idempotency

`risk_scoring.state` defines one typed event per table: patient (`Id`, `BIRTHDATE`, `DEATHDATE`), encounter (`Id`, `START`, `STOP`, `PATIENT`, `ENCOUNTERCLASS`), medication (`START`, `STOP`, `PATIENT`, `ENCOUNTER`, `CODE`), and condition (`START`, `STOP`, `PATIENT`, `ENCOUNTER`, `SYSTEM`, `CODE`, `DESCRIPTION`). Events validate at construction: key fields must be non-empty, and dates and timestamps must round-trip through their exact format, so a non-zero-padded near-miss is rejected rather than normalized. Optional fields (`STOP`, `DEATHDATE`) accept the empty string. Malformed payloads raise immediately and never reach the database; the HTTP layer will translate that into a 4xx response rather than a silent drop.

Ingestion is keyed on deterministic identifiers from the payload: encounters and patients on their Synthea row `Id`, conditions (which have no row id) on the composite `(PATIENT, ENCOUNTER, CODE, START)`, and medications on that composite plus `STOP`. Re-posting an already-ingested event, which happens on crash retries and replay resumes, is a silent no-op. A re-post whose key exists with different field values is rejected loudly instead of overwritten: replays are byte-identical by design, so divergence signals a harness bug that must surface, not be absorbed. Medications are the exception: their key is their whole payload, so a medication event cannot conflict and any difference makes it a distinct row.

The wider medication key was forced by the data. Synthea records a renewal as two rows for the same drug, ordered at the same encounter and the same instant, differing only in `STOP` and the cost columns the payload drops: a single dispense beside a continuing course. The frozen populations hold 1,001 such pairs in baseline, 1,192 in care_protocol, and 2,136 in demographic_shift. Under the narrower key the second row of every pair was rejected as a divergent re-post, and absorbing it silently instead would have undercounted active medications on 33 of the 12,308 baseline discharges. The five-column key is unique across all three populations (zero duplicates in 4.0 million medication rows), which migration `0003_medication_stop_in_key` installs. Conditions keep the four-column key and its conflict detection, because they carry `SYSTEM` and `DESCRIPTION` outside the key and collide in no population.

Each record call commits its own row. An acknowledged event must be a persisted event, or crash retries and replay resumes could observe acknowledged-but-lost writes; callers therefore must not wrap record calls in a larger transaction they intend to roll back. At this volume, per-event commits cost nothing.


## Serving-time features and the skew check

`risk_scoring.serving.serving_features` computes one discharge's scoring input from that patient's recorded history. It calls the same two functions the training pipeline calls: `cohort.build_cohort` decides admission and `features.build_features` computes the row. Neither rule is restated. The cohort check runs over the single encounter being scored, since every cohort rule is per-encounter; feature computation receives the whole history, because prior encounters, medications, and conditions are what the features read. Nothing was extracted from the shared modules to make this work, so "one cohort module and one feature module, shared verbatim" stays structural rather than a claim a test has to chase.

Two behaviors the serving seam fixes on its own. An encounter still open at ingestion, with an empty `STOP`, is not a scoring event and yields nothing: a completed CSV export contains no such row, so the cohort module never sees one, and admitting it would anchor a feature row to a missing timestamp. Asking to score an encounter that state has no row for raises instead of returning nothing, because a silent exclusion there would make a lost ingestion look like a routine cohort rejection.

The skew check runs both paths over one population and compares with exact equality, never a tolerance: a feature that differs by a rounding step between training and serving is still a skew bug. The batch path reads the population's CSVs the way the training pipeline reads them. The serving path interleaves the same rows into one timestamp-ordered stream, ingests them one at a time, and scores each discharge on arrival from persisted state alone. Arrival times follow the event's own meaning: an encounter is a discharge notification and arrives at its `STOP`, medications arrive at their `START`, and conditions, whose dates are date-only, arrive at midnight of their start date. That last rule is what puts a condition recorded on a discharge date in front of that day's discharge, matching how the feature module judges condition activity against the discharge date rather than the discharge instant. Where two arrive at the same instant, medications and conditions precede the discharge they were in effect for.

The CI population is synthetic and built to hit every boundary at once: the 180-day window's inclusive far edge and the day past it, the days-since-previous cap and its no-history sentinel, overlapping stays flooring the gap at zero, a readmission that must see its index stay, a medication stopping exactly at the discharge instant, two prescriptions of one drug sharing an encounter and a start instant, conditions starting and stopping on the discharge date, a finding that is not a disorder, history-based flags from a resolved situation code and from an ICD10 malignancy, and events dated after the discharge that no feature may read. A separate test asserts the population still produces those values, so a fixture that quietly collapsed could not make the skew comparison vacuous.

## Skew confirmed on generated data

Recorded 2026-08-27. The frozen populations are local-only and verified by checksum manifest, so this comparison cannot run in CI:

```bash
python scripts/check_serving_skew.py --population baseline --patients 500
```

A seeded sample of 500 baseline patients holding at least one cohort discharge: 48,054 encounters, 46,502 medications, and 24,904 conditions ingested as 119,460 individual events, 1,202 discharges scored on arrival, and every feature value on every one of them equal to the batch pipeline's. Runtime 4 minutes 39 seconds, dominated by the per-event commit. Sampling by patient is exact rather than approximate: every feature reads only the scored patient's own rows, so the batch pipeline's output for a patient does not depend on which other patients share the frame.

Two things this run surfaced:

Medications and conditions are posted once, at their `START`, carrying the `STOP` they will eventually have. State therefore knows a prescription's end date before it arrives. No feature reads it early, because `STOP` is only ever compared against the discharge instant, so there is no leak today. A replay harness that instead posted a medication open and closed it later would collide with the key design, since a medication's key is its whole payload and the close would land as a second row rather than an update. Whether ingestion needs update semantics is a question for the replay harness, not for the data spine.

Pandas emitted one `UserWarning` per run about falling back from datetime format inference, raised by a single patient in the 500-patient sample whose medication start timestamps defeat the format guess: `guess_datetime_format` returns `None` when a value's minute field repeats two digits of its year, as in `2007-02-09T20:07:18Z`, so the whole column goes through `dateutil` element by element. Seven of the 11,064 baseline patients with medications hit that shape. The fallback parsed every value identically, and explicit-format parsing was checked against inferred parsing for all eight timestamp columns of all three frozen populations with no value differing, so this was a latent hazard rather than a present defect. Inference also made rejection depend on the luck of the first row: with a guessable first element pandas applies the guess strictly and rejects a malformed value, while with an unguessable one the same value reaches `dateutil` and is quietly accepted. `risk_scoring.cohort`, `risk_scoring.features`, and `risk_scoring.labels` now pin the export's formats, `%Y-%m-%dT%H:%M:%SZ` for encounter and medication timestamps and `%Y-%m-%d` for condition dates and patient birth and death dates, matching what `risk_scoring.state` already enforces at event construction. Every `pd.to_datetime` call in `src` and `scripts` now carries an explicit format, and a non-conforming value raises instead of being reinterpreted. The full baseline feature frame, 12,308 discharges over 16 columns, and the full baseline label frame, the same 12,308 discharges carrying 1,557 positives, are both bit-identical before and after the change, and this run reproduces with warnings escalated to errors.

## Running the service

```bash
python -m risk_scoring.service run
```

The command reads `configs/service.toml` from the working directory, loads the pinned model version from the repo-local MLflow registry, opens a connection pool against `RISK_SCORING_DATABASE_URL` (defaulting to the Compose instance), and serves on port 8000 (`--port` overrides it). Startup fails with a nonzero exit if the pinned version is absent from the registry or the database is unreachable; the service never starts without either.

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok"}`. A 200 implies startup completed, so the model is loaded and the database is reachable. |
| `GET /version` | Model name and pinned version from the config, `FEATURE_VERSION`, `COHORT_VERSION`, and the git SHA (null when unresolvable). |
| `POST /events` | Ingests one event and answers 202 with the event type, the input hash, whether the event was scored, and the prediction id and score when it was. |

The git SHA comes from `git rev-parse HEAD` at startup and is null when the working directory is not a repository. The container image will not contain `.git`, so a build-time override must be added when the image exists.

## The HTTP ingestion boundary

Events arrive over one endpoint, in timestamp order, one at a time, wrapped in an envelope that names the type: `{"event_type": "encounter", "payload": {...}}`. The four accepted types and their payload columns are the four `risk_scoring.state` events, carrying the same Synthea column names and the same verbatim-string rule.

The split between the two layers is deliberate and narrow. The request models own shape: which fields exist, that no unknown field sneaks in, and which event type an envelope names. `state` owns every value rule: required-and-non-empty, the exact timestamp and date formats, and which fields may be empty. Both layers were built in parallel and independently arrived at the same four column sets and the same format checks, which is exactly the kind of duplication that drifts, so the format rules now live once and a test asserts each payload's field set equals the matching column tuple in `state`.

Every refusal is a 4xx, and none is a silent drop:

| Refusal | Status | Cause |
| --- | --- | --- |
| Unknown event type, unknown field, missing field | 422 | FastAPI's own validation, with field-level detail |
| A value that fails its exact format, or an empty identity field | 422 | `MalformedEventError` from the state layer, raised at conversion |
| An inpatient discharge whose patient has no recorded demographics | 422 | `UnknownPatientError`, naming the patient |
| An event contradicting one already stored under the same key | 409 | `EventConflictError`, naming the differing fields |

The unknown-patient case is an ordering violation rather than a bad payload: the cohort rules need a birthdate, so demographics must precede a patient's first discharge. Reporting it rather than skipping the score is the point, since a silently unscored discharge would look identical to a cohort exclusion. The event is still stored before the refusal, so re-posting that discharge once the demographics arrive scores it normally.

Accepted events answer 202 rather than 200, which stays honest for the events that are not scoring events: a medication, a condition, an open stay, or a cohort-excluded encounter all update state and produce no score.

## The scoring path

`risk_scoring.service.ingest.ingest_event` is the whole path, as a plain function over a connection and a loaded model. The endpoint is a thin wrapper around it, so the replay harness can drive the same code without HTTP.

1. Persist the event through `state.record_event`, which commits it on its own.
2. If it is not an encounter, stop. Nothing but a discharge can be a scoring event.
3. If the predictions log already holds a row for this encounter, stop.
4. Read the patient's history and call `serving.serving_features`, which narrows the same `build_cohort` and `build_features` the training pipeline calls. A `None` means "state updated, nothing to score" for every reason at once: still open, wrong encounter class, in-hospital death, under 18.
5. Cast the feature row to the model input columns as float64, exactly as training does, score it, and write the log row.

Nothing in that sequence re-expresses a cohort or feature rule, which is what keeps "one cohort module and one feature module, shared verbatim" structural.

The endpoint handler is async only long enough to read the raw body for the hash. Everything after that (database, pandas, the model) runs in the threadpool, so one slow scoring call cannot stall the event loop. Connections come from a `psycopg_pool` pool opened at startup rather than one connection per request, since a replay posts events continuously.

## Predictions log

One row per scored discharge, in the `predictions` table (migration `0004_predictions`):

| Column | Type | Notes |
| --- | --- | --- |
| `prediction_id` | `bigserial` | primary key |
| `patient_id`, `encounter_id` | `text` | `encounter_id` is unique |
| `event_time` | `timestamptz` | the discharge instant in simulated time |
| `scored_at` | `timestamptz` | wall clock at write, defaulted by the database |
| `input_hash` | `text` | SHA-256 over the exact posted event |
| `model_name`, `model_version` | `text`, `integer` | the registered version that scored it |
| `feature_version`, `cohort_version` | `text` | the shared modules' own versions |
| `score` | `double precision` | |
| `features` | `jsonb` | the 14 model input values, as the model received them |

The table uses real column types, unlike the verbatim strings in the state tables. State stores exact bytes because serving-time recompute must be byte-identical to the batch path; the log is read instead by rolling time window and by numeric score, so `event_time` is indexed and typed. The parse from the encounter's verbatim STOP is lossless, since every stored value has already round-tripped through the same exact format.

The feature values are stored rather than recomputed on demand, so diagnosing a suspect score later never requires rebuilding the patient's state as it was.

### Why the log is the idempotency authority

The state tables commit per event, and the prediction commits separately. A process can therefore die with an encounter durably stored and its score not, which would leave a permanent gap in the log across a restart.

Deciding whether to score by the *absence of a prediction row*, rather than by whether the state write was new, closes that window. A re-posted encounter is a no-op in state but still has no score, so it gets one; `ON CONFLICT (encounter_id) DO NOTHING` settles the race if two writers reach it at once. The first score for an encounter stands: a conflicting write is dropped, never merged and never overwritten, so a replay that revisits a discharge cannot rewrite history.

A separate test restates the whole schema literally, column names, types, nullability, and both constraints, so a migration cannot change this substrate without someone changing it in two places on purpose.

## Input hash

Every accepted event is hashed so a logged prediction can be tied to its exact input later. The definition, implemented as pure functions in `risk_scoring.payload_hash`:

1. Parse the request body as JSON.
2. Serialize the parsed object with keys sorted lexicographically at every nesting level, minimal separators (`","` and `":"`), `ensure_ascii=False`, and `allow_nan=False`.
3. Encode as UTF-8 and take the SHA-256 digest, rendered as 64 lowercase hex characters.

The handler hashes the body as received, before the request model parses it. Re-serializing the validated model was rejected because any coercion the model applied would contaminate the hash; with the raw-object route, "unmodified field values" is structural rather than something a test has to chase. The hash covers the whole envelope, including `event_type`, and is computed before any parsing into frames or feature work.

Three consequences of the definition are worth stating. Formatting and key order of the wire text never change the digest, since the hash covers the parsed object; duplicate keys in the raw text collapse to the last occurrence. No unicode normalization is applied, so NFC and NFD spellings of one value hash differently, which is what unmodified values require. NaN and infinities raise a `ValueError` instead of producing unparseable output. A known-vector test pins the algorithm, so any change to the canonicalization breaks a test rather than silently rewriting history.

## Model loading and version pinning

The service loads its model once, at startup, from the MLflow registry that `risk_scoring.tracking.configure_tracking` points at, using `models:/readmission-risk/<version>` with the version read from `configs/service.toml`:

```toml
[model]
name = "readmission-risk"
version = 3
```

The pin must be an explicit positive integer. The loader rejects strings (including `"latest"` and numeric strings), booleans, zero, and negatives, and no code path in the service resolves "newest", so serving an unpinned model is structurally impossible. This deliberately differs from the gate, which defaults to the newest registered version when no pin is given: a gate wants the latest candidate, a service wants exactly what was promoted. A missing pin or an absent version stops startup with an error naming the model, the version, and the registry URI. The committed pin is version 3, the version the last full retrain from raw data produced, and a test asserts the committed name matches `risk_scoring.train.MODEL_NAME` so the two cannot drift.

## How the container reaches the registry

Decision, taken before the container existed: bind-mount `mlflow.db` and `mlruns/` rather than bake the model artifact into the image. One registry stays the single source of truth for training, gating, and serving, and promotion remains an edit to one committed config value. Baking the artifact in at build time forks the registry into per-image copies; it remains the recorded fallback.

Two risks were recorded against that decision. Building the container settled both, and turned up a third that had not been anticipated.

The absolute-path risk was real and is handled by construction. `configure_tracking` pins each experiment's artifact location as an absolute `file://` URI, and every registered version's `storage_location` carries the training machine's own path, so a mount at any other path resolves to nothing. Compose therefore interpolates `${PWD}` into both the mounts and the container's working directory, which puts `mlflow.db` and `mlruns/` at exactly the paths the stored URIs name. The service's repo root is its working directory, so `configure_tracking` derives the same SQLite path inside the container as on the host. This is why the stack must be started from the repository root, and why any clone works without editing the file: the interpolation follows the clone.

The SQLite risk did not materialize. The database is in `delete` journal mode, not WAL, so a read needs no writable sidecar, and a read-only mount serves model resolution without complaint. `mlflow.db` is mounted read-only and stays that way.

The third risk was the one that bit. Loading a `models:/name/version` URI is not a pure read: MLflow writes a derived `registered_model_meta` file beside the artifact on every load, recording the registered name and version the local copy came from. A read-only `mlruns/` fails that write with `OSError: [Errno 30] Read-only file system` before the model ever loads. `mlruns/` is therefore mounted writable while `mlflow.db` stays read-only, which is the narrowest thing that works: the container can rewrite a 48-byte derived sidecar that the host service already writes with identical content, and it cannot alter the registry itself, which is what "single source of truth" means here. Copying the artifact store into the container at startup would restore full immutability at the cost of an entrypoint script and a per-start copy; that is the recorded escalation if the artifact store ever needs to be genuinely untouchable.

## The container and the Compose stack

`docker compose up -d --build`, run from the repository root, brings up three services:

| Service | Role |
| --- | --- |
| `postgres` | State, the prediction log, and whatever is built on them. Host port 5433. |
| `migrate` | One-shot `python -m risk_scoring.db migrate`, gated on Postgres reporting healthy. |
| `app` | The service, gated on `migrate` completing successfully. Host port 8001. |

Schema changes are a separate one-shot service rather than part of the service's own start, so restarting the service is a pure restart and never issues DDL. That keeps the restart check honest: what it exercises is a process coming back, not a schema being reapplied.

Host port 8001, not 8000, for the same reason Postgres publishes on 5433: an application already listening on the usual port must never collide with the stack.

The image is `python:3.12-slim` with dependencies installed from the same `uv.lock` CI installs, so the container runs the versions the tests ran against. LightGBM's wheel links against `libgomp`, which the slim base omits, so the image installs it. The service's own source is copied in; the registry, `configs/`, and the working directory arrive by mount.

Two things the container forced into the service code:

- The CLI gained `--host`, defaulting to `127.0.0.1`. A host process should not be reachable off the machine; the image's command asks for `0.0.0.0` explicitly, because its port is published.
- `resolve_git_sha` now prefers `RISK_SCORING_GIT_SHA` over `git rev-parse`. An image carries no `.git`, so `/version` was reporting a null commit for the code it was built from. The build passes the SHA as an argument, placed after the dependency layers so a new commit rebuilds one cheap layer. An unset argument arrives as an empty string, which is not a SHA and falls through to the working tree, so a host process is unaffected.

## Restart and state rebuild

The service holds nothing across requests that it did not derive at startup: the loaded model, the connection pool, the frozen config, and the commit SHA. There is no cache, no mutable module global, and no per-patient memory anywhere in the package. Every value a score depends on is read back from Postgres on the request that needs it, so the rebuild path is structurally trivial: a fresh instance scores correctly from history it never saw arrive, with no warm-up and nothing to prime.

That is a property worth keeping rather than rediscovering, so a test restates the set of names the lifespan writes onto the app state. Adding a per-patient cache later breaks that test instead of passing review.

The equivalence check is the substantive one. It posts one event stream twice, once straight through and once with the app torn down and rebuilt partway, and requires the two prediction logs to be equal row for row. Six restart points are covered: before the first event, after the first, mid-stream, immediately before a discharge, immediately after one, and after the last event. A separate case reopens the crash window across the restart, writing an encounter through the state layer directly so it is durable while its score is not, then bringing a fresh instance up and posting that encounter; the log is what decides whether to score, so the new instance scores it. Two mutations confirm the tests bite: deciding to score on the state write rather than on the log breaks the crash-window case, and caching a patient's history in the process breaks every equivalence case.

`prediction_id` and `scored_at` are excluded from the comparison, and the reason is worth stating because it looks like a gap. The database assigns both. A `bigserial` evaluates `nextval` before the conflict check, so a dropped re-post consumes an id: writing `encounter-1`, re-posting it, then writing `encounter-2` leaves ids 1 and 3. A resume that revisits scored discharges therefore gaps the sequence by design, and comparing two runs by id would fail on a system behaving correctly. What must match is the content and the order, and a test pins the gap behavior so the exclusion is a recorded fact rather than a convenience.

## Restart equivalence confirmed against the containers

Recorded 2026-08-28. The frozen populations are local-only, so this cannot run in CI:

```bash
python scripts/check_restart_equivalence.py --population baseline --patients 25
```

A seeded sample of 25 baseline patients, 5,641 events posted one at a time over HTTP to the running stack, in two arms against two throwaway databases. The first arm ran straight through. The second stopped and started the service container after 2,820 events, waited for it to report healthy again, and posted the remaining 2,821. Both arms scored 42 discharges, and every logged field of every row was identical: same encounters, same order, same input hashes, same scores, same feature values. Runtime 55 seconds for both arms including the image build check.

The script asserts the container actually cycled, comparing Docker's `StartedAt` for the service container across the restart. Without that check, a restart that silently did nothing would let a service holding state in memory print a match, which is precisely the failure the run exists to catch; skipping the restart call makes the script fail rather than pass.


## Provenance confirmed on generated data

The scoring path was run against the real local registry and a real PostgreSQL database, using one patient drawn from the frozen baseline population rather than a fixture. The patient's full history up to the target discharge was posted as an event stream in timestamp order, one event per request: 1 demographics event, 689 encounters, 76 medications, and 60 conditions, 826 events in all. Three of those encounters are adult inpatient discharges, and the service scored exactly those three on arrival.

`GET /version` reported model `readmission-risk` version 3, `FEATURE_VERSION` 1.0.0, `COHORT_VERSION` 1.0.0, and the serving commit's SHA.

The worked example traces the target discharge, encounter `31576c3d-e3a7-4eef-65ed-76d4163d1611`, discharged 2025-11-01T04:08:50Z, logged as prediction 3 with score 0.013610957904194856:

- Recomputing SHA-256 over the exact posted event reproduced the logged `input_hash` `5e945dd96340053624d11903d252f852ec58a857600dd54d5caa27b42ff5260c`.
- Loading `models:/readmission-risk/3` from MLflow and re-scoring the logged feature values reproduced the logged score exactly, to the last bit, with no tolerance.
- The stored features are `age_at_discharge` 63, `los_days` 1, `prior_inpatient_180d` 0, `days_since_prev_discharge` 365 (the no-history sentinel), `prior_ed_180d` 1, `active_medication_count` 7, `active_disorder_count` 11, and the diabetes, MI, and renal flags set.

Two properties were checked alongside it. Re-posting the same discharge answered 202 with `scored: false` and left the table at three rows. Running the batch pipeline over the same rows produced the same three cohort discharges, and every one of the 42 logged feature values matched its batch value exactly. That trace was performed by hand on one encounter; the sections below turn the same two recomputations into code that runs over every prediction a batch produces.

## Recomputing provenance

A logged prediction makes two claims, and both are checkable after the fact. Its input hash covers the exact event that produced it, and its score is what the named model version returns for the stored feature values. `risk_scoring/provenance.py` recomputes both rather than trusting the row.

The source event is rebuilt from the population export, never from whatever the caller holds in memory. Hashing the same object twice inside one process would prove nothing. Rebuilding it proves the whole chain: source CSV row, payload projection, posted envelope, stored digest. The envelope is the part worth naming, because the service hashes the whole posted body rather than the payload inside it, and a recomputation over the payload alone would disagree with every row ever written.

The model comes from the version the row itself names, never the version the service happens to be pinned to. That is the failure the check exists to catch, since a row written by an entirely different model would otherwise pass without comment.

Comparison is exact and carries no tolerance. The score column is `double precision` and the feature values are `jsonb`, so both round-trip losslessly, and two loads of one artifact deserialize the same booster. The mistakes worth catching here, a wrong column order or a rounded stored value or the wrong version, either move the score visibly or not at all, so a tolerance would only widen the band where something subtly wrong looks fine. A mismatch reports both values with their absolute and last-bit difference, so a run on different hardware would be diagnosable rather than merely failed.

`tests/test_provenance_postgres.py` runs the same recomputation in CI over a synthetic population.

Writing that check surfaced a weakness in the fixture underneath it. The model every service test shared was trained on 46 rows against a `min_data_in_leaf` of 20, so LightGBM took no split at all and the booster returned the base rate for every feature vector it was ever given: one score across all ten discharges the service tests ingest, and a holdout AUROC of exactly 0.5. Nothing failed, because a constant model satisfies an equality between two runs exactly as well as a real one does, and the assertion that should have caught it checked only that AUROC was a probability. Two gate tests were affected in a subtler way, failing on the band floor while their comments explained they were exercising the ceiling.

The fixture now trains on the signal-bearing population, which lands mid-band and produces six distinct scores across those ten discharges. `tests/test_trained_fixture.py` holds it there, asserting that the model is not constant, that it lands inside the pre-registered band, and that it spreads scores across the population the service tests actually post. The smaller deterministic population was enlarged until it delivers the perfect separation its docstring always claimed, and the two gate tests now assert the leakage banner rather than merely a failure.

## The held-out batch

`python -m risk_scoring.scoring_run run` posts encounters the model was never trained on. Held out means discharged at or after the 2025-01-01 training cutoff, which is the exact complement of the training window, computed by one function in the cohort module so the two halves cannot drift apart at the boundary.

Selection works on patients, not encounters, and that distinction carries the run. A patient's features read their whole history, so posting only their post-cutoff rows would make serving features disagree with the training pipeline for a reason that is not a defect in either. Every selected patient is posted in full, back to their earliest row. The consequence is that the service also scores their pre-cutoff discharges, which the model was trained through, and those are reported as their own count and their own distribution. Summing the two would present in-sample discharges as held-out evidence.

The run reads its log back filtered to the encounters it posted, so the summary describes that batch whatever else the database holds. It fails, naming encounter ids, when a discharge the cohort rules admit was never scored, when a logged row the cohort rules exclude appears anyway, or when what the service returned at the time disagrees with what it durably wrote.

## Held-out batch scored against the containers

Recorded 2026-08-28. The frozen populations are local-only, so this cannot run in CI:

```bash
python -m risk_scoring.scoring_run run --population baseline --patients 250
```

A seeded sample of 250 patients drawn from the 639 in the frozen baseline holding a post-cutoff cohort discharge, posted to the Compose stack over HTTP one event at a time: 250 demographics events, 25,101 encounters, 19,958 medications, and 12,611 conditions, 57,920 events in all. The service admitted and scored 326 held-out discharges and 409 pre-cutoff discharges of the same patients, 735 predictions in total, with no cohort discharge left unscored and no logged row the cohort rules exclude. Every acknowledgement the service returned matched the row it durably wrote. Runtime 179 seconds. `GET /version` reported model `readmission-risk` version 3, `FEATURE_VERSION` 1.0.0, `COHORT_VERSION` 1.0.0, and the serving commit's SHA.

Held-out scores span 0.000155 to 0.788526, with median 0.017349, quartiles 0.006704 and 0.062074, fifth and ninety-fifth percentiles 0.000537 and 0.362723, and mean 0.071047. Across all 735 predictions the median is 0.014145 and the mean 0.069563. The pre-cutoff figures are recorded for completeness and are not evidence of held-out performance.

All 735 predictions reproduced. The worked example is the held-out prediction at the median score, encounter `7609e8c3-fbdc-cb44-d2ba-6cff1456f1a1` for patient `7609e8c3-fbdc-cb44-6ebb-a555935aa5c4`, discharged 2026-01-08T15:46:48Z and logged as prediction 598 with score 0.01740727240699904:

- Rebuilding the posted event from the source CSV row and taking SHA-256 over it reproduced the logged `input_hash` `68bf989f14fe55c4f3a5c5c85165efe88b579022fa91af88b5a5660d487b0f09`.
- Loading `models:/readmission-risk/3` and re-scoring the logged feature values reproduced the logged score exactly, to the last bit, with no tolerance.
- The stored features are `age_at_discharge` 59, `los_days` 7.345879629629629, `prior_inpatient_180d` 0, `days_since_prev_discharge` 365 (the no-history sentinel), `prior_ed_180d` 0, `active_medication_count` 7, `active_disorder_count` 10, and the diabetes, malignancy, and renal flags set.

Running the same command a second time against the same database logged nothing new and still reported 735 predictions reproduced, since a discharge already carrying a prediction is not scored again.

## All three checks re-run together

Recorded 2026-08-28, at commit `5cc09c5`, against the same containers and the same frozen baseline population, in one sitting:

```bash
python scripts/check_serving_skew.py --population baseline --patients 500
python scripts/check_restart_equivalence.py --population baseline --patients 25
python -m risk_scoring.scoring_run run --population baseline --patients 250
```

Serving features matched the batch pipeline on all 1,202 discharges of the 500-patient sample, 119,460 events ingested, 4 minutes 59 seconds. The restarted container produced a prediction log identical to the uninterrupted arm on all 42 discharges of the 25-patient sample, 5,641 events with the restart after 2,820, 89 seconds. The held-out batch posted 57,920 events for 250 patients and scored 326 held-out and 409 pre-cutoff discharges, and all 735 predictions reproduced their input hash and their score, 173 seconds. The batch ran against an empty database and reproduced the earlier run exactly, down to the worked example landing on the same prediction id.

The restart check failed the first time it ran, and the reason is worth keeping. The client that posts the stream holds one connection open for the whole run, which is what makes sixty thousand requests affordable. Restarting the service container kills that connection. The script waits for health before posting the rest of the stream, so the service was back and ready, but the first post after the restart went down a dead socket and raised `RemoteDisconnected`.

That connection was hoisted to run scope when the posting code moved out of the check script and into `risk_scoring.service_client`, and the check had not been run against the containers since. Nothing in CI could catch it: the restart test there drives the app in process through a test client, which has no socket to lose. The equivalent hazard for a caller that never restarts anything is an idle service closing a kept-alive connection on its own timeout.

`ServiceClient` now reopens the connection once and repeats the request, but only when the connection had already carried a request, since one that fails on its first use was never established rather than dropped. Repeating a post is safe because the service treats a re-posted event as a no-op, and the retry is bounded at one so a service that is genuinely gone fails rather than spins. Three tests in `tests/test_service_client.py` cover the reopen, the bound, and the refusal to retry a fresh connection.
