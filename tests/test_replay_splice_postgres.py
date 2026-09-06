"""Stream splicing against a real service, a real log, and the labels table.

The skew population is the stream until the splice instant; from it, the
splice population, its ids rewritten, is the source of every event. The
rules these tests pin:

- Before the splice only the outgoing population is posted; at or after
  it only the incoming one, including a discharge at exactly the instant.
  The outgoing population's later discharges are never posted.
- A spliced replay logs exactly what posting the spliced events one at a
  time logs, after both populations' history is in state.
- The skew check across the splice: an incoming discharge's logged
  features equal the batch pipeline's over the incoming export, which is
  only true if its pre-splice history reached state under its rewritten
  ids. No incoming patient id is an outgoing one.
- Labels follow the population of origin: an outgoing discharge keeps the
  outgoing export's label though its readmission fell after the splice,
  and an incoming one has the incoming export's. No label is early, an
  incoming discharge from before the splice has no prediction and no
  label, and a discharge inside the final 30 days is pending.
- Byte identity across the splice: paused the tick before it, at it, and
  after it, or killed one post before the first incoming post and one
  after, both tables equal an uninterrupted run's.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import psycopg
import pytest

from replay_support import (
    END,
    MAX_SPEED,
    POPULATION,
    SPLICE_AT,
    SPLICE_POPULATION,
    START,
    ClientPoster,
    KilledMidTick,
    Serve,
    Source,
    prepare_spliced,
    read_log,
    read_outputs,
    schedule_of,
    serving,
    skew_frames,
    splice_frames,
    stream_of,
    tick_containing,
)
from risk_scoring import label_log, predictions, train
from risk_scoring.cohort import build_cohort
from risk_scoring.features import MODEL_INPUT_COLUMNS, build_features
from risk_scoring.populations import rekey_id, rekeyed
from risk_scoring.replay import audit, clock, harness, preload, runs
from risk_scoring.replay.config import ReplayConfig, Splice
from risk_scoring.replay.release import ScheduledLabel
from risk_scoring.replay.splice import (
    Segment,
    segment_events,
    segment_labels,
    segments,
    spliced_events,
    spliced_labels,
)
from risk_scoring.stream import StreamEvent, envelope

pytestmark = pytest.mark.db

SPLICE = clock.instant(SPLICE_AT)


@pytest.fixture(scope="module")
def outgoing(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    return skew_frames(tmp_path_factory.mktemp("splice-outgoing") / "csv")


@pytest.fixture(scope="module")
def incoming(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """The splice population as the harness reads it: ids rewritten."""
    return rekeyed(
        splice_frames(tmp_path_factory.mktemp("splice-incoming") / "csv"), SPLICE_POPULATION
    )


@pytest.fixture(scope="module")
def plan() -> list[Segment]:
    config = ReplayConfig(
        population=POPULATION,
        start=START.date(),
        end=END.date(),
        acceleration=4,
        splices=(Splice(at=SPLICE_AT.date(), population=SPLICE_POPULATION),),
    )
    return segments(config)


@pytest.fixture(scope="module")
def sources(outgoing: dict[str, pd.DataFrame], incoming: dict[str, pd.DataFrame]) -> list[Source]:
    return [(outgoing, stream_of(outgoing), START), (incoming, stream_of(incoming), SPLICE_AT)]


@pytest.fixture(scope="module")
def events(plan: list[Segment], sources: list[Source]) -> list[StreamEvent]:
    """The spliced stream: what the run posts, nothing else."""
    return spliced_events(
        [
            segment_events(segment, stream)
            for segment, (_, stream, _) in zip(plan, sources, strict=True)
        ]
    )


@pytest.fixture(scope="module")
def schedule(plan: list[Segment], sources: list[Source]) -> list[ScheduledLabel]:
    return spliced_labels(
        [
            segment_labels(segment, schedule_of(frames))
            for segment, (frames, _, _) in zip(plan, sources, strict=True)
        ]
    )


@pytest.fixture()
def serve(trained_repo: tuple[Any, train.TrainingResult]) -> Serve:
    return serving(trained_repo)


def _replay(
    serve: Serve,
    dsn: str,
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    *,
    die_after: int | None = None,
    pause_at: datetime | None = None,
) -> tuple[harness.RunSummary | None, ClientPoster]:
    """One invocation from wherever the run row stands; None if it died."""
    with psycopg.connect(dsn, connect_timeout=2) as conn, serve(dsn) as client:
        run = runs.open_run(conn)
        assert run is not None
        poster = ClientPoster(client, die_after=die_after)
        try:
            summary = harness.run_replay(
                conn,
                run,
                events,
                poster,
                labels=schedule,
                pacing=MAX_SPEED,
                pause_requested=lambda sim_now: pause_at is not None and sim_now >= pause_at,
            )
        except KilledMidTick:
            return None, poster
        return summary, poster


def _incoming_id(name: str) -> str:
    return rekey_id(SPLICE_POPULATION, name)


def test_the_fixture_crosses_the_splice_with_every_kind(
    sources: list[Source], events: list[StreamEvent]
) -> None:
    """Every kind arrives after the splice, and every kind of outgoing row is dropped."""
    after = {event.kind for event in events if event.at >= SPLICE}
    assert after == {"encounter", "medication", "condition"}
    outgoing_after = {event.kind for event in sources[0][1] if event.at >= SPLICE}
    assert outgoing_after == {"encounter", "medication", "condition"}
    assert all(event.row["Id"] != "e-full-index" for event in events if event.kind == "encounter")


def _keys(events: list[StreamEvent]) -> set[str]:
    """The row ids a poster's payloads carry: an encounter's own, otherwise the patient's."""
    return {
        event.row["Id"] if event.kind == "encounter" else event.row["PATIENT"] for event in events
    }


def test_only_the_outgoing_population_is_posted_before_the_splice_and_only_the_incoming_after(
    serve: Serve,
    db_url: str,
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    prepare_spliced(db_url, sources)
    summary, poster = _replay(serve, db_url, events, schedule)

    assert summary is not None and summary.finished
    outgoing_keys, incoming_keys = _keys(sources[0][1]), _keys(sources[1][1])
    assert not outgoing_keys & incoming_keys
    assert poster.posted == [envelope(event.kind, event.row) for event in events]
    for event in events:
        key = event.row["Id"] if event.kind == "encounter" else event.row["PATIENT"]
        assert key in (outgoing_keys if event.at < SPLICE else incoming_keys), event.at
    boundary = [event for event in events if event.at == SPLICE]
    assert [event.row["Id"] for event in boundary] == [_incoming_id("e-b-boundary")]

    scored = {row["encounter_id"] for row in read_log(db_url)}
    assert scored == {
        "e-gap-overlap-a",
        "e-gap-overlap-b",
        "e-readmit-1",
        *(_incoming_id(n) for n in ("e-b-boundary", "e-b-r1", "e-b-index", "e-b-r2", "e-b-late")),
    }
    assert summary.discharges_scored == 8


def test_a_spliced_replay_logs_what_per_event_posting_logs(
    serve: Serve,
    db_url_factory: Any,
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    reference_dsn = db_url_factory()
    replay_dsn = db_url_factory()

    with psycopg.connect(reference_dsn, connect_timeout=2) as conn:
        for frames, stream, before in sources:
            preload.preload_history(conn, frames, stream, clock.instant(before))
    with serve(reference_dsn) as client:
        poster = ClientPoster(client)
        for event in events:
            poster.post_event(envelope(event.kind, event.row))
    reference = read_log(reference_dsn)

    prepare_spliced(replay_dsn, sources)
    summary, _ = _replay(serve, replay_dsn, events, schedule)

    assert summary is not None and summary.finished
    assert read_log(replay_dsn) == reference
    assert len(reference) == 8


def test_an_incoming_discharge_computes_over_its_preloaded_history(
    serve: Serve,
    db_url: str,
    outgoing: dict[str, pd.DataFrame],
    incoming: dict[str, pd.DataFrame],
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    """The skew check restated across the splice."""
    prepare_spliced(db_url, sources)
    _replay(serve, db_url, events, schedule)

    cohort = build_cohort(incoming["encounters"], incoming["patients"]).frame
    batch = build_features(
        cohort, incoming["encounters"], incoming["medications"], incoming["conditions"]
    ).set_index("encounter_id")
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        logged = {row.encounter_id: row for row in predictions.all_predictions(conn)}

    index = logged[_incoming_id("e-b-index")]
    expected = {
        name: float(batch.loc[_incoming_id("e-b-index"), name]) for name in MODEL_INPUT_COLUMNS
    }
    assert index.features == expected
    # Independent of the pipeline: one prior stay, discharged 42 days earlier,
    # one active prescription and one active disorder from the incoming export.
    assert index.features["prior_inpatient_180d"] == 1
    assert index.features["days_since_prev_discharge"] == 38
    assert index.features["active_medication_count"] == 1
    assert index.features["active_disorder_count"] == 1
    assert index.features["flag_chf"] == 1

    for name in ("e-b-boundary", "e-b-r1", "e-b-r2", "e-b-late"):
        row = logged[_incoming_id(name)]
        assert row.features == {
            column: float(batch.loc[_incoming_id(name), column]) for column in MODEL_INPUT_COLUMNS
        }
    incoming_patients = {row.patient_id for row in logged.values() if row.event_time >= SPLICE_AT}
    assert incoming_patients
    assert not incoming_patients & set(outgoing["patients"]["Id"])
    # The id clash the fixture plants: the same export id, two different people.
    assert "p-fresh" in set(outgoing["patients"]["Id"])
    assert _incoming_id("p-fresh") in incoming_patients


def test_labels_follow_the_population_of_origin(
    serve: Serve,
    db_url: str,
    outgoing: dict[str, pd.DataFrame],
    incoming: dict[str, pd.DataFrame],
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> None:
    prepare_spliced(db_url, sources)
    summary, _ = _replay(serve, db_url, events, schedule)

    assert summary is not None
    assert (summary.labels_released, summary.labels_pending) == (7, 1)
    outgoing_batch = audit.batch_labels(outgoing)
    incoming_batch = audit.batch_labels(incoming)
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        released = {row.encounter_id: row for row in label_log.all_labels(conn)}
        discharged = {row.encounter_id: row.event_time for row in predictions.all_predictions(conn)}

    for name, row in released.items():
        batch = outgoing_batch if discharged[name] < SPLICE_AT else incoming_batch
        assert row.label == batch[name], name
    # Readmitted on 2024-05-15, after the splice, by a stay never posted.
    assert released["e-readmit-1"].label == 1
    assert "e-readmit-2" not in discharged
    assert released[_incoming_id("e-b-r1")].label == 1
    with psycopg.connect(db_url, connect_timeout=2) as conn:
        assert audit.early_labels(conn) == 0
    assert _incoming_id("e-b-prior") not in discharged
    assert _incoming_id("e-b-history") not in discharged
    assert set(discharged) - set(released) == {_incoming_id("e-b-late")}


# Byte identity across the splice

Point = Callable[[list[StreamEvent]], datetime]


def _first_incoming_index(events: list[StreamEvent]) -> int:
    return next(index for index, event in enumerate(events) if event.at >= SPLICE)


def _tick_before_splice(events: list[StreamEvent]) -> datetime:
    return SPLICE_AT - clock.TICK


def _splice_tick(events: list[StreamEvent]) -> datetime:
    return SPLICE_AT


def _after_splice(events: list[StreamEvent]) -> datetime:
    return tick_containing(events[_first_incoming_index(events) + 2].at) + timedelta(hours=1)


PAUSE_POINTS: list[tuple[str, Point]] = [
    ("the tick before the splice", _tick_before_splice),
    ("the splice tick", _splice_tick),
    ("after the first incoming events", _after_splice),
]

KILL_POINTS: list[tuple[str, Callable[[list[StreamEvent]], int]]] = [
    ("one post before the first incoming post", lambda events: _first_incoming_index(events)),
    ("one post after the first incoming post", lambda events: _first_incoming_index(events) + 1),
]


def _straight(
    serve: Serve,
    dsn: str,
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
) -> dict[str, list[dict[str, Any]]]:
    prepare_spliced(dsn, sources)
    summary, _ = _replay(serve, dsn, events, schedule)
    assert summary is not None and summary.finished
    outputs = read_outputs(dsn)
    assert len(outputs["predictions"]) == 8 and len(outputs["labels"]) == 7
    return outputs


@pytest.mark.parametrize(("name", "point"), PAUSE_POINTS, ids=[name for name, _ in PAUSE_POINTS])
def test_paused_and_resumed_across_the_splice_is_byte_identical(
    serve: Serve,
    db_url_factory: Callable[[], str],
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    name: str,
    point: Point,
) -> None:
    straight = _straight(serve, db_url_factory(), sources, events, schedule)
    db_url = db_url_factory()
    prepare_spliced(db_url, sources)
    first, before = _replay(serve, db_url, events, schedule, pause_at=point(events))
    assert first is not None and first.paused and not first.finished
    second, after = _replay(serve, db_url, events, schedule)
    assert second is not None and second.finished

    assert read_outputs(db_url) == straight
    posted = [*before.posted, *after.posted]
    assert posted == [envelope(event.kind, event.row) for event in events]


@pytest.mark.parametrize(("name", "count"), KILL_POINTS, ids=[name for name, _ in KILL_POINTS])
def test_killed_and_restarted_across_the_splice_is_byte_identical(
    serve: Serve,
    db_url_factory: Callable[[], str],
    sources: list[Source],
    events: list[StreamEvent],
    schedule: list[ScheduledLabel],
    name: str,
    count: Callable[[list[StreamEvent]], int],
) -> None:
    straight = _straight(serve, db_url_factory(), sources, events, schedule)
    db_url = db_url_factory()
    prepare_spliced(db_url, sources)
    died, before = _replay(serve, db_url, events, schedule, die_after=count(events))
    assert died is None
    second, after = _replay(serve, db_url, events, schedule)
    assert second is not None and second.finished

    assert read_outputs(db_url) == straight
    expected = [envelope(event.kind, event.row) for event in events]
    assert before.posted == expected[: len(before.posted)]
    assert after.posted == expected[len(expected) - len(after.posted) :]
    assert len(before.posted) + len(after.posted) >= len(expected)
