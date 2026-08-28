-- Bookkeeping table for the migration runner in risk_scoring.db.
-- The runner treats a database without this table as having nothing applied,
-- so this migration bootstraps the mechanism itself.
CREATE TABLE schema_migrations (
    number     integer PRIMARY KEY,
    name       text NOT NULL,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
