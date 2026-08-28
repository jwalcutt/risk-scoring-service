-- Per-patient raw event state: verbatim Synthea strings, restricted to the
-- columns the cohort and feature modules read. Empty string means the source
-- field was empty; NULL never appears, mirroring how the training pipeline
-- reads CSVs (dtype=str, keep_default_na=False). Natural keys make ingestion
-- idempotent; divergent re-posts are rejected in application code.
--
-- Timestamps stay text: ISO8601 sorts lexicographically, so plain btree
-- indexes already serve by-patient chronological access, and storing the
-- exact bytes keeps serving-time recompute identical to the batch path.

CREATE TABLE patients (
    id        text PRIMARY KEY CHECK (id <> ''),
    birthdate text NOT NULL CHECK (birthdate <> ''),
    deathdate text NOT NULL
);

CREATE TABLE encounters (
    id              text PRIMARY KEY CHECK (id <> ''),
    start           text NOT NULL CHECK (start <> ''),
    stop            text NOT NULL,
    patient         text NOT NULL CHECK (patient <> ''),
    encounter_class text NOT NULL CHECK (encounter_class <> '')
);

CREATE INDEX encounters_patient_start_idx ON encounters (patient, start);

-- Medications and conditions carry no Synthea row Id; their composite
-- natural keys lead with patient, so the primary-key btree already covers
-- by-patient access and no separate index is needed.
CREATE TABLE medications (
    start     text NOT NULL CHECK (start <> ''),
    stop      text NOT NULL,
    patient   text NOT NULL CHECK (patient <> ''),
    encounter text NOT NULL CHECK (encounter <> ''),
    code      text NOT NULL CHECK (code <> ''),
    PRIMARY KEY (patient, encounter, code, start)
);

CREATE TABLE conditions (
    start       text NOT NULL CHECK (start <> ''),
    stop        text NOT NULL,
    patient     text NOT NULL CHECK (patient <> ''),
    encounter   text NOT NULL CHECK (encounter <> ''),
    system      text NOT NULL CHECK (system <> ''),
    code        text NOT NULL CHECK (code <> ''),
    description text NOT NULL,
    PRIMARY KEY (patient, encounter, code, start)
);
