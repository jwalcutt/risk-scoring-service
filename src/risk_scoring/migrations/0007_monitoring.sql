-- What monitoring compares against, what it found, and what it raised.
--
-- monitoring_reference is one row per registered model version: the rows
-- that version was fitted on, kept as raw per-feature arrays rather than
-- summaries so a statistic sees the whole sample instead of an earlier
-- guess about what mattered. The feature arrays span the whole training
-- window; the score array is the patient-grouped holdout only, because
-- in-sample scores are optimistically shifted and would make live score
-- drift read low. A promotion writes a new row for the new version rather
-- than editing this one, so a finished run's evaluations keep naming the
-- reference they were actually measured against.
--
-- monitoring_evaluations is one row per boundary per run. Every query
-- behind it is bounded by the boundary and never by the wall clock, so
-- the same boundary evaluated live and evaluated afterwards over the
-- finished tables gives equal rows. The unique key is (run_id, boundary):
-- a monitor killed between boundaries neither skips one nor writes a
-- second row for one, and a database that has hosted more than one
-- finished run can still hold both runs' grids, which a unique key on the
-- boundary alone would refuse. thresholds_hash is on every row so that a
-- changed threshold is visible in the data and not only in a commit.
--
-- alerts carries sim_at, the boundary, because detection time is measured
-- in simulated days from an injection to its alert and must not depend on
-- when a poll happened to land. That equality is a property of the schema
-- rather than a rule the writer remembers: monitoring_evaluations carries
-- a unique pair purely as a foreign-key target, and an alert references
-- it, so an alert whose instant is not its own evaluation's boundary
-- cannot be written. raised_at is the wall clock at write, the twin of
-- scored_at and recorded_at, for diagnosis and nothing else.

CREATE TABLE monitoring_reference (
    reference_id    bigserial PRIMARY KEY,
    model_name      text NOT NULL CHECK (model_name <> ''),
    model_version   integer NOT NULL CHECK (model_version > 0),
    feature_version text NOT NULL CHECK (feature_version <> ''),
    cohort_version  text NOT NULL CHECK (cohort_version <> ''),
    training_cutoff timestamptz NOT NULL,
    split_seed      integer NOT NULL,
    n_train_rows    integer NOT NULL CHECK (n_train_rows > 0),
    n_holdout_rows  integer NOT NULL CHECK (n_holdout_rows > 0),
    features        jsonb NOT NULL,
    scores          jsonb NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_name, model_version)
);

CREATE TABLE monitoring_evaluations (
    evaluation_id           bigserial PRIMARY KEY,
    run_id                  bigint NOT NULL REFERENCES replay_runs (run_id),
    reference_id            bigint NOT NULL REFERENCES monitoring_reference (reference_id),
    window_start            timestamptz NOT NULL,
    boundary                timestamptz NOT NULL CHECK (boundary > window_start),
    thresholds_hash         text NOT NULL CHECK (thresholds_hash <> ''),
    prediction_count        integer NOT NULL CHECK (prediction_count >= 0),
    label_count             integer NOT NULL CHECK (label_count >= 0),
    refusal_count           integer NOT NULL CHECK (refusal_count >= 0),
    version_mismatch_count  integer NOT NULL CHECK (version_mismatch_count >= 0),
    statistics              jsonb NOT NULL,
    evaluated_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, boundary),
    -- Not a business rule. It is the target an alert's (evaluation, instant)
    -- pair references, which is what makes sim_at unable to disagree.
    UNIQUE (evaluation_id, boundary)
);

-- Grafana and the offline audit both read a run's evaluations in boundary order.
CREATE INDEX monitoring_evaluations_boundary_idx ON monitoring_evaluations (boundary);

CREATE TABLE alerts (
    alert_id        bigserial PRIMARY KEY,
    evaluation_id   bigint NOT NULL REFERENCES monitoring_evaluations (evaluation_id),
    signal          text NOT NULL CHECK (signal <> ''),
    statistic       double precision NOT NULL,
    threshold       double precision NOT NULL,
    sim_at          timestamptz NOT NULL,
    raised_at       timestamptz NOT NULL DEFAULT now(),
    acknowledged_at timestamptz,
    note            text NOT NULL,
    -- At one boundary a signal either alerts or it does not.
    UNIQUE (evaluation_id, signal),
    FOREIGN KEY (evaluation_id, sim_at)
        REFERENCES monitoring_evaluations (evaluation_id, boundary)
);

-- The operator's question is which alerts are still outstanding.
CREATE INDEX alerts_unacknowledged_idx ON alerts (sim_at) WHERE acknowledged_at IS NULL;
