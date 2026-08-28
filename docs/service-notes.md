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
