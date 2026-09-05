-- One row per replay run: the simulated clock and the checkpoint are the
-- same row. sim_now is the run's simulated instant. The cursor is the sort
-- key of the last event posted, so a resumed, rebuilt, or spliced stream
-- continues at the same event rather than at an index into a list that
-- may have changed. Pause and resume are writes to status; whoever writes
-- 'paused' (an operator now, a monitoring alert later) stops the harness
-- at its next tick.
--
-- Real types, as in predictions: this table is read by monitoring and
-- dashboards by simulated time and never feeds feature computation. The
-- cursor columns stay text because they are compared in Python against
-- the stream's own sort key and only need to round-trip exactly.
--
-- Pacing is deliberately absent. Tick size and max-speed mode must not
-- change what a run writes, so they belong to a process invocation, not
-- to the run. Acceleration is recorded because it describes the run an
-- operator attended, not because anything reads it back.

CREATE TABLE replay_runs (
    run_id        bigserial PRIMARY KEY,
    population    text NOT NULL CHECK (population <> ''),
    start_at      timestamptz NOT NULL,
    end_at        timestamptz NOT NULL CHECK (end_at > start_at),
    acceleration  double precision NOT NULL CHECK (acceleration > 0),
    sim_now       timestamptz NOT NULL CHECK (sim_now >= start_at),
    status        text NOT NULL CHECK (status IN ('running', 'paused', 'finished')),
    cursor_at     text,
    cursor_kind   integer,
    cursor_row    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    -- A half-written cursor is no checkpoint.
    CHECK ((cursor_at IS NULL) = (cursor_kind IS NULL)
       AND (cursor_at IS NULL) = (cursor_row IS NULL))
);

-- The prediction log is unique per discharge, so one database hosts one
-- scoring run at a time; resume and status must find exactly one row.
CREATE UNIQUE INDEX replay_runs_one_open_idx
    ON replay_runs ((true)) WHERE status <> 'finished';
