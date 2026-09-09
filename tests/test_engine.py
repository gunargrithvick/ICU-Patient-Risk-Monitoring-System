"""The monitoring engine: one tick, and everything that has to be true after it.

The engine is where the pieces meet, so these tests are mostly about *composition* rather
than about any single calculation - that a tick produces one assessment per bed, that history
accumulates and stays bounded, that the simulator controls actually move the physiology, and
that back-dated warm-up ticks produce a plausible past instead of a pile of observations all
stamped "now".

Vision is left off for most of these. A camera-less ward is the default deployment and the
one a reviewer will run first, so it is the configuration worth exercising by default.
"""

from __future__ import annotations

import itertools
from datetime import timedelta

import pytest

from icu_monitor.config import Settings
from icu_monitor.core.types import RiskLevel, utcnow
from icu_monitor.monitoring import MonitoringEngine, build_engine

from .conftest import EPOCH


@pytest.fixture
def engine(config: Settings) -> MonitoringEngine:
    instance = MonitoringEngine(config=config, load_vision=False)
    try:
        yield instance
    finally:
        instance.close()


# ------------------------------------------------------------------------------ a tick


def test_a_tick_assesses_every_bed(engine: MonitoringEngine, config: Settings) -> None:
    snapshot = engine.tick()
    assert len(snapshot.beds) == config.bed_count
    assert snapshot.tick == 1
    assert {bed.patient.patient_id for bed in snapshot.beds} == {
        patient.patient_id for patient in engine.patients
    }
    for bed in snapshot.beds:
        assert bed.assessment.patient_id == bed.patient.patient_id
        assert 0.0 <= bed.assessment.composite_score <= 100.0


def test_the_tick_counter_advances(engine: MonitoringEngine) -> None:
    assert [engine.tick().tick for _ in range(3)] == [1, 2, 3]


def test_a_tick_reports_how_long_it_took(engine: MonitoringEngine) -> None:
    """Surfaced on the dashboard: a ward that takes 4 s per tick is not monitoring anything."""
    snapshot = engine.tick()
    assert snapshot.duration_ms >= 0.0
    assert snapshot.duration_ms < 5000.0


def test_a_tick_can_be_stamped_with_an_explicit_time(engine: MonitoringEngine) -> None:
    """The snapshot's clock and its observations' clock have to be the same clock.

    Warm-up back-dates its ticks, and ``as_dict`` publishes both ``at`` and each bed's
    ``recorded_at``; a snapshot stamped *now* around vitals recorded minutes ago reads as
    stale data rather than as history.
    """
    snapshot = engine.tick(at=EPOCH)
    assert snapshot.at == EPOCH
    assert all(bed.vitals.recorded_at == EPOCH for bed in snapshot.beds)
    assert all(bed.assessment.assessed_at == EPOCH for bed in snapshot.beds)


def test_snapshot_lookup_by_patient(engine: MonitoringEngine) -> None:
    snapshot = engine.tick()
    known = snapshot.beds[0].patient.patient_id
    assert snapshot.bed(known) is not None
    assert snapshot.bed("P999") is None


def test_snapshot_is_json_serialisable(engine: MonitoringEngine) -> None:
    """``icu-monitor tick --json`` and the API both lean on this."""
    import json

    payload = engine.tick().as_dict()
    assert json.dumps(payload, default=str)
    assert payload["beds"]


# ----------------------------------------------------------------------- ward summaries


def test_level_counts_account_for_every_bed(engine: MonitoringEngine, config: Settings) -> None:
    snapshot = engine.tick()
    counts = snapshot.level_counts()
    assert sum(counts.values()) == config.bed_count
    assert set(counts) <= set(RiskLevel)


def test_worst_bed_is_the_highest_scoring_one(engine: MonitoringEngine) -> None:
    """The ward list is sorted by this, so a wrong answer sends staff to the wrong bed."""
    snapshot = engine.run(6)
    worst = snapshot.worst
    assert worst is not None
    assert worst.assessment.composite_score == max(
        bed.assessment.composite_score for bed in snapshot.beds
    )


def test_mean_score_is_the_mean(engine: MonitoringEngine) -> None:
    snapshot = engine.tick()
    scores = [bed.assessment.composite_score for bed in snapshot.beds]
    assert snapshot.mean_score == pytest.approx(sum(scores) / len(scores))


def test_new_alerts_are_only_this_ticks(engine: MonitoringEngine) -> None:
    """A tick reports what it raised; the manager keeps the running total."""
    engine.run(10)
    snapshot = engine.tick()
    assert all(alert in engine.alerts.history for alert in snapshot.new_alerts)
    assert len(snapshot.new_alerts) <= len(engine.alerts.history)


# ----------------------------------------------------------------------------- history


def test_history_accumulates_per_patient(engine: MonitoringEngine) -> None:
    engine.run(5)
    for patient in engine.patients:
        assert len(engine.history(patient.patient_id)) == 5


def test_history_is_oldest_first(engine: MonitoringEngine) -> None:
    """The charts plot it straight through; reversed history draws time backwards."""
    engine.run(6)
    stamps = [v.recorded_at for v in engine.history(engine.patients[0].patient_id)]
    assert stamps == sorted(stamps)


def test_history_is_bounded_by_the_configured_window(config: Settings) -> None:
    """A monitor left running for a week must not accumulate a week of vitals in RAM."""
    tight = config.with_overrides(history_window=20)
    engine = MonitoringEngine(config=tight, load_vision=False)
    try:
        engine.run(25)
        assert len(engine.history(engine.patients[0].patient_id)) == 20
    finally:
        engine.close()


def test_score_history_tracks_the_composite(engine: MonitoringEngine) -> None:
    engine.run(4)
    series = engine.score_history(engine.patients[0].patient_id)
    assert len(series) == 4
    assert all(0.0 <= score <= 100.0 for _at, score in series)
    assert [at for at, _score in series] == sorted(at for at, _score in series)


def test_history_of_an_unknown_patient_is_empty_not_an_error(engine: MonitoringEngine) -> None:
    assert engine.history("P999") == ()
    assert engine.score_history("P999") == ()


# ---------------------------------------------------------------------------- warm-up


def test_backfill_lays_the_warmup_out_in_the_past(engine: MonitoringEngine) -> None:
    """A warm-up must look like history, not like twelve observations at the same instant.

    The dashboard's first paint shows a trend line. Without back-dating, every warm-up tick
    carries the wall-clock time it was computed at, so the chart opens as a vertical stripe
    and the trend arrows are meaningless.
    """
    before = utcnow()
    snapshot = engine.run(12, backfill=True)
    after = utcnow()

    stamps = [v.recorded_at for v in engine.history(engine.patients[0].patient_id)]
    assert len(stamps) == 12
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 12
    assert stamps[0] < before
    assert stamps[-1] <= after
    assert snapshot.at is not None

    step = timedelta(seconds=engine._config.tick_seconds)
    gaps = {round((b - a).total_seconds(), 6) for a, b in itertools.pairwise(stamps)}
    assert gaps == {round(step.total_seconds(), 6)}


def test_a_plain_run_uses_wall_clock(engine: MonitoringEngine) -> None:
    before = utcnow()
    engine.run(3)
    stamps = [v.recorded_at for v in engine.history(engine.patients[0].patient_id)]
    assert all(stamp >= before for stamp in stamps)


def test_run_of_zero_ticks_still_returns_a_usable_snapshot(engine: MonitoringEngine) -> None:
    """``run`` returns a snapshot, never ``None``, so callers need no optional handling.

    ``build_engine(warmup_ticks=0)``, the API warm-up and ``icu-monitor tick --ticks 0`` all
    take this path. With no prior state there is nothing to return, so one tick is taken
    rather than handing back a null the dashboard would have to guard on every read.
    """
    snapshot = engine.run(0)
    assert snapshot.tick == 1
    assert len(snapshot.beds) == len(engine.patients)


def test_zero_ticks_does_not_advance_an_already_running_ward(engine: MonitoringEngine) -> None:
    engine.run(3)
    before = engine.last_snapshot
    assert engine.run(0) is before
    assert engine.tick_count == 3


# ------------------------------------------------------------------- simulator controls


def test_the_advertised_events_are_the_ones_that_work(engine: MonitoringEngine) -> None:
    """The dashboard builds its dropdown from this map, so every entry has to be injectable."""
    events = engine.available_events()
    assert events
    patient_id = engine.patients[0].patient_id
    for slug in events:
        assert engine.inject_event(patient_id, slug), slug


def test_injecting_an_unknown_event_is_refused_not_raised(engine: MonitoringEngine) -> None:
    assert engine.inject_event(engine.patients[0].patient_id, "spontaneous_recovery") is None
    assert engine.inject_event("P999", "desaturation") is None


def test_an_injected_event_shows_as_running(engine: MonitoringEngine) -> None:
    patient_id = engine.patients[0].patient_id
    assert engine.active_events(patient_id) == ()
    engine.inject_event(patient_id, "desaturation")
    assert engine.active_events(patient_id) != ()


def test_a_desaturation_actually_desaturates(engine: MonitoringEngine) -> None:
    """The control surface has to move the physiology, or the demo is a puppet show."""
    patient_id = engine.patients[0].patient_id
    engine.run(6)
    baseline = min(v.spo2 for v in engine.history(patient_id) if v.spo2 is not None)

    engine.inject_event(patient_id, "desaturation")
    engine.run(20)
    during = [v.spo2 for v in engine.history(patient_id) if v.spo2 is not None]
    assert min(during) < baseline


def test_setting_the_trajectory_takes_effect(engine: MonitoringEngine) -> None:
    patient_id = engine.patients[0].patient_id
    assert engine.set_state(patient_id, "deteriorating") is True
    assert engine.patient(patient_id).state.value == "deteriorating"
    assert engine.set_state(patient_id, "levitating") is False
    assert engine.set_state("P999", "stable") is False


def test_deterioration_raises_the_score_over_time(config: Settings) -> None:
    """The point of the trajectory control: a deteriorating patient must actually deteriorate."""
    engine = MonitoringEngine(config=config, load_vision=False)
    try:
        patient_id = engine.patients[0].patient_id
        engine.set_state(patient_id, "stable")
        engine.run(20)
        stable_mean = sum(s for _at, s in engine.score_history(patient_id)) / 20

        engine.set_state(patient_id, "critical")
        engine.run(40)
        later = engine.score_history(patient_id)[-20:]
        critical_mean = sum(s for _at, s in later) / len(later)
        assert critical_mean > stable_mean
    finally:
        engine.close()


def test_oxygen_and_target_scale_can_be_changed(engine: MonitoringEngine) -> None:
    patient_id = engine.patients[0].patient_id
    assert engine.set_oxygen(patient_id, on=True, scale=2) is True

    record = engine.patient(patient_id)
    assert record.on_supplemental_oxygen is True
    assert record.spo2_scale == 2

    snapshot = engine.tick()
    bed = snapshot.bed(patient_id)
    assert bed.vitals.on_supplemental_oxygen is True
    assert bed.assessment.news2.scale == 2


def test_oxygen_change_for_an_unknown_bed_is_refused(engine: MonitoringEngine) -> None:
    assert engine.set_oxygen("P999", on=True) is False


def test_patient_lookup_is_by_id(engine: MonitoringEngine) -> None:
    """``engine.patients`` is a tuple, so lookup goes through this method rather than indexing."""
    known = engine.patients[0].patient_id
    assert engine.patient(known) is not None
    assert engine.patient("P999") is None


# ----------------------------------------------------------------- degraded and wiring


def test_the_ward_runs_with_no_model_and_no_camera(config: Settings) -> None:
    """The bare-clone configuration: nothing trained, nothing plugged in, still monitoring."""
    engine = MonitoringEngine(config=config, model=None, load_vision=False)
    try:
        snapshot = engine.run(4)
        assert len(snapshot.beds) == config.bed_count
        assert all(bed.assessment.model_available is False for bed in snapshot.beds)
        assert all(bed.assessment.news2 is not None for bed in snapshot.beds)
        assert snapshot.model_version in (None, "")
    finally:
        engine.close()


def test_vision_is_reported_per_bed_not_duplicated(config: Settings) -> None:
    """One camera serves the ward, so only the focused bed carries a live signal."""
    engine = MonitoringEngine(config=config, load_vision=False)
    try:
        snapshot = engine.tick()
        assert snapshot.focus_bed in {bed.patient.patient_id for bed in snapshot.beds}
        assert all(bed.assessment.vision is not None for bed in snapshot.beds)
    finally:
        engine.close()


def test_the_simulation_seed_makes_a_run_reproducible(config: Settings) -> None:
    """A demo that cannot be replayed cannot be debugged."""

    def scores() -> list[float]:
        engine = MonitoringEngine(config=config, load_vision=False)
        try:
            snapshot = engine.run(8)
            return [round(bed.assessment.composite_score, 6) for bed in snapshot.beds]
        finally:
            engine.close()

    assert scores() == scores()


def test_a_different_seed_gives_a_different_ward(config: Settings) -> None:
    def scores(seed: int) -> list[float]:
        engine = MonitoringEngine(
            config=config.with_overrides(simulation_seed=seed), load_vision=False
        )
        try:
            snapshot = engine.run(8)
            return [round(bed.assessment.composite_score, 6) for bed in snapshot.beds]
        finally:
            engine.close()

    assert scores(1) != scores(2)


def test_build_engine_can_warm_up_on_construction(config: Settings) -> None:
    engine = build_engine(config=config, warmup_ticks=5)
    try:
        assert len(engine.history(engine.patients[0].patient_id)) == 5
    finally:
        engine.close()


def test_close_is_idempotent(config: Settings) -> None:
    engine = MonitoringEngine(config=config, load_vision=False)
    engine.close()
    engine.close()
