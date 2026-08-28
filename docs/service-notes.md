# Service notes

Recorded 2026-08-27, alongside the first infrastructure for the scoring service: a Compose-managed Postgres and the schema migration runner. The service itself does not exist yet. This note records the storage decisions made before any service code, and it will grow as the service does.

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
