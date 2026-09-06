-- Ground truth, released late: one row per scored discharge whose 30-day
-- readmission label has become available on the simulated clock. The
-- replay harness computes every label from the population export up
-- front and withholds only the release, so what this table records is
-- when a label became knowable, never what it is.
--
-- prediction_id is unique and a foreign key, so a label attaches to
-- exactly one scored discharge and an unscored discharge gets none. The
-- uniqueness is also the release's idempotency key: a harness that died
-- after writing a label and before its checkpoint re-releases the tick
-- on resume, and the conflict clause drops the repeat.
--
-- due_at is the discharge instant plus 30 simulated days; released_at is
-- the simulated instant of the tick that wrote the row. The check between
-- them is the never-early rule as a property of the table: a label that
-- appears before its discharge has had 30 days to mature is
-- unrepresentable. recorded_at is the wall clock at write, the twin of
-- scored_at on the log, for outage diagnosis and nothing else.
--
-- Real types, as in predictions: monitoring reads this table by simulated
-- time as labels mature and joins it to the log by prediction id.

CREATE TABLE labels (
    label_id      bigserial PRIMARY KEY,
    prediction_id bigint NOT NULL UNIQUE REFERENCES predictions (prediction_id),
    encounter_id  text NOT NULL,
    label         integer NOT NULL CHECK (label IN (0, 1)),
    label_version text NOT NULL,
    due_at        timestamptz NOT NULL,
    released_at   timestamptz NOT NULL CHECK (released_at >= due_at),
    recorded_at   timestamptz NOT NULL DEFAULT now()
);

-- Realized performance is read as labels arrive, by the instant they were released.
CREATE INDEX labels_released_at_idx ON labels (released_at);
