-- The provenance log: one row per scored discharge, carrying everything
-- needed to trace a score back to the exact model and the exact input that
-- produced it, without recomputing anything.
--
-- Unlike the state tables, this one uses real types. State stores verbatim
-- Synthea strings because serving-time recompute must be byte-identical to
-- the batch path; the log is read instead by monitoring, dashboards, and
-- the later comparison work, all of which query by time window and by
-- numeric score. event_time is the discharge instant in simulated time,
-- parsed losslessly from the encounter's verbatim STOP; scored_at is the
-- wall clock at write.
--
-- encounter_id is unique, so one discharge has exactly one score. That
-- constraint is the log's idempotency key: a re-posted event whose state
-- write already landed but whose score did not is rescored and inserted,
-- and one whose score did land is dropped by the conflict clause. The
-- state tables commit per event, so this is what closes the gap between
-- an event becoming durable and its prediction becoming durable.
--
-- The feature values are stored as written, so diagnosing a suspect score
-- later never requires rebuilding the patient's state as it was.

CREATE TABLE predictions (
    prediction_id   bigserial PRIMARY KEY,
    patient_id      text NOT NULL,
    encounter_id    text NOT NULL UNIQUE,
    event_time      timestamptz NOT NULL,
    scored_at       timestamptz NOT NULL DEFAULT now(),
    input_hash      text NOT NULL,
    model_name      text NOT NULL,
    model_version   integer NOT NULL,
    feature_version text NOT NULL,
    cohort_version  text NOT NULL,
    score           double precision NOT NULL,
    features        jsonb NOT NULL
);

-- Monitoring reads this table by rolling time window on every evaluation.
CREATE INDEX predictions_event_time_idx ON predictions (event_time);
