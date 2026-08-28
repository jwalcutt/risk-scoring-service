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

Every column is text and every value is a verbatim string from the source row. The training pipeline reads CSVs as all-string with empty strings for missing values, and the shared modules do their own timestamp parsing, so state read-back returns frames byte-identical to a batch CSV load: uppercase Synthea column names, export column order, empty string (never NULL) for missing. ISO8601 timestamps sort lexicographically in time order, so plain btree indexes on text serve the by-patient chronological access pattern; encounters get a `(patient, start)` index, while medications and conditions are covered by composite primary keys that lead with the patient id.

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

Two things this run surfaced that are recorded rather than acted on:

Medications and conditions are posted once, at their `START`, carrying the `STOP` they will eventually have. State therefore knows a prescription's end date before it arrives. No feature reads it early, because `STOP` is only ever compared against the discharge instant, so there is no leak today. A replay harness that instead posted a medication open and closed it later would collide with the key design, since a medication's key is its whole payload and the close would land as a second row rather than an update. Whether ingestion needs update semantics is a question for the replay harness, not for the data spine.

Pandas emits one `UserWarning` per run about falling back from datetime format inference, raised by a single patient in the 500-patient sample whose medication start timestamps make the first-element format guess ambiguous. The fallback parses every value identically: explicit-format parsing was checked against inferred parsing for all eight timestamp columns of all three frozen populations and no value differs. The warning is a latent hazard rather than a present defect, and pinning explicit formats in the shared feature module is a change on its own, not part of this one.

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

A separate test restates the whole schema literally, column names, types, nullability, and both constraints, so a migration cannot change the substrate every later phase reads without someone changing it in two places on purpose.

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

Decision: when the service gets its own Compose block, the container bind-mounts `mlflow.db` and `mlruns/` read-only. One registry stays the single source of truth for training, gating, and serving, and promotion remains an edit to one committed config value. The alternative, baking the model artifact into the image at build time, was rejected as the default because it forks the registry into per-image copies; it remains the recorded fallback if the mount proves brittle.

Two known risks may trigger that fallback, and the restart tests against real containers will decide. First, `configure_tracking` pins each experiment's artifact location as an absolute `file://` URI of the training machine, and model loading resolves artifacts through the URI stored in the database, so the mount must reproduce the host's absolute path inside the container or the stored URIs must be regenerated. Second, SQLite on a read-only bind mount can fail on lock or journal acquisition depending on journal state; opening the database with `mode=ro&immutable=1`, or copying it in at startup, are the candidate mitigations. No Dockerfile exists yet: nothing in the service shell requires a container, and the service's Compose block waits until the ingestion path writes state, so the shared Compose file grows in one place at a time.

## Provenance confirmed on generated data

The scoring path was run against the real local registry and a real PostgreSQL database, using one patient drawn from the frozen baseline population rather than a fixture. The patient's full history up to the target discharge was posted as an event stream in timestamp order, one event per request: 1 demographics event, 689 encounters, 76 medications, and 60 conditions, 826 events in all. Three of those encounters are adult inpatient discharges, and the service scored exactly those three on arrival.

`GET /version` reported model `readmission-risk` version 3, `FEATURE_VERSION` 1.0.0, `COHORT_VERSION` 1.0.0, and the serving commit's SHA.

The worked example traces the target discharge, encounter `31576c3d-e3a7-4eef-65ed-76d4163d1611`, discharged 2025-11-01T04:08:50Z, logged as prediction 3 with score 0.013610957904194856:

- Recomputing SHA-256 over the exact posted event reproduced the logged `input_hash` `5e945dd96340053624d11903d252f852ec58a857600dd54d5caa27b42ff5260c`.
- Loading `models:/readmission-risk/3` from MLflow and re-scoring the logged feature values reproduced the logged score exactly, to the last bit, with no tolerance.
- The stored features are `age_at_discharge` 63, `los_days` 1, `prior_inpatient_180d` 0, `days_since_prev_discharge` 365 (the no-history sentinel), `prior_ed_180d` 1, `active_medication_count` 7, `active_disorder_count` 11, and the diabetes, MI, and renal flags set.

Two properties were checked alongside it. Re-posting the same discharge answered 202 with `scored: false` and left the table at three rows. Running the batch pipeline over the same rows produced the same three cohort discharges, and every one of the 42 logged feature values matched its batch value exactly.
