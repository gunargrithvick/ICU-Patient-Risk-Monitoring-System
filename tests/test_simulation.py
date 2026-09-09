"""The synthetic ward: physiology that behaves, and a replay path that degrades.

A simulator is easy to write and easy to write *badly*. Uncorrelated noise produces
numbers that look like vitals and never produce the coupled, gradually-worsening pattern
an early-warning score exists to detect - so the tests that matter here are the ones about
**shape**: that a deteriorating patient's saturation actually falls, that hypoxia drags
respiratory rate up, that nothing ever leaves the plausible range, and that the same seed
replays the same stay.

The second half covers the replay provider and the factory. The contract worth pinning is
that a missing dataset is a *fallback*, not a crash: a reviewer with no PhysioNet download
still gets a working ward.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from icu_monitor.config import Settings
from icu_monitor.core.types import ClinicalState, Consciousness, Vitals
from icu_monitor.simulation.patient import (
    BASELINE,
    CLAMPS,
    EVENT_LIBRARY,
    PatientSimulator,
    SimulationEvent,
)
from icu_monitor.simulation.ward import (
    REPLAY_STALENESS_HOURS,
    ReplayWard,
    SimulatedWard,
    VitalsProvider,
    _resample_stay,
    build_provider,
    state_mix,
)

# --------------------------------------------------------------------------- reproducible


def steady(seed: int = 7, **kwargs: object) -> PatientSimulator:
    """A simulator with dropouts off, so a test sees every channel it asks for."""
    options: dict[str, object] = {"seed": seed, "dropouts": False}
    options.update(kwargs)
    return PatientSimulator(**options)  # type: ignore[arg-type]


def test_the_same_seed_replays_the_same_stay() -> None:
    """A demo that cannot be replayed cannot be debugged, and neither can a test."""

    def trace() -> list[float | None]:
        simulator = PatientSimulator(seed=99)
        return [simulator.step().heart_rate for _ in range(20)]

    assert trace() == trace()


def test_a_different_seed_gives_a_different_stay() -> None:
    def trace(seed: int) -> list[float | None]:
        simulator = PatientSimulator(seed=seed)
        return [simulator.step().heart_rate for _ in range(10)]

    assert trace(1) != trace(2)


def test_step_returns_a_bedside_observation() -> None:
    vitals = steady().step()
    assert isinstance(vitals, Vitals)
    assert vitals.recorded_at is not None
    assert vitals.consciousness in set(Consciousness)


def test_steps_are_counted() -> None:
    simulator = steady()
    for _ in range(5):
        simulator.step()
    assert simulator.steps_taken == 5


# ------------------------------------------------------------------------------- bounds


@pytest.mark.parametrize("state", list(ClinicalState))
def test_no_channel_ever_leaves_its_plausible_range(state: ClinicalState) -> None:
    """A simulator must never emit an impossible number, whatever the trajectory."""
    simulator = steady(state=state)
    for _ in range(400):
        simulator.step()
        for channel, value in simulator.snapshot().items():
            low, high = CLAMPS[channel]
            assert low <= value <= high, channel


def test_observations_are_rounded_like_a_chart() -> None:
    """Bedside monitors show integers, and a temperature to one decimal place."""
    vitals = steady().step()
    assert vitals.heart_rate == round(vitals.heart_rate)
    assert vitals.spo2 == round(vitals.spo2)
    assert vitals.temperature == round(vitals.temperature, 1)


def test_diastolic_stays_below_systolic() -> None:
    simulator = steady(state=ClinicalState.CRITICAL)
    for _ in range(120):
        vitals = simulator.step()
        if vitals.bp_systolic is not None and vitals.bp_diastolic is not None:
            assert vitals.bp_diastolic < vitals.bp_systolic
            assert vitals.bp_diastolic >= 28.0


# --------------------------------------------------------------------------- trajectory


def mean_of(simulator: PatientSimulator, channel: str, steps: int = 120) -> float:
    values = []
    for _ in range(steps):
        simulator.step()
        values.append(simulator.snapshot()[channel])
    return sum(values) / len(values)


def test_a_deteriorating_patient_deteriorates() -> None:
    """The whole point of modelling trajectory rather than sampling noise."""
    stable = mean_of(steady(state=ClinicalState.STABLE), "spo2")
    deteriorating = mean_of(steady(state=ClinicalState.DETERIORATING), "spo2")
    critical = mean_of(steady(state=ClinicalState.CRITICAL), "spo2")
    assert stable > deteriorating > critical


def test_deterioration_moves_every_channel_the_clinical_way() -> None:
    stable = steady(state=ClinicalState.STABLE)
    critical = steady(state=ClinicalState.CRITICAL)
    for _ in range(150):
        stable.step()
        critical.step()
    calm, sick = stable.snapshot(), critical.snapshot()
    assert sick["heart_rate"] > calm["heart_rate"]
    assert sick["resp_rate"] > calm["resp_rate"]
    assert sick["bp_systolic"] < calm["bp_systolic"]
    assert sick["temperature"] > calm["temperature"]
    assert sick["gcs"] < calm["gcs"]


def test_everyone_starts_near_the_stable_baseline() -> None:
    """A deteriorating patient must visibly deteriorate, not arrive already broken.

    Otherwise the first frame of the demo is the end of the story and no trend is visible.
    """
    for state in ClinicalState:
        opening = steady(state=state).snapshot()
        assert abs(opening["spo2"] - BASELINE["spo2"][0]) < 4.0
        assert abs(opening["heart_rate"] - BASELINE["heart_rate"][0]) < 12.0


def test_state_can_be_changed_mid_stay() -> None:
    simulator = steady(state=ClinicalState.STABLE)
    simulator.set_state("critical")
    assert simulator.state is ClinicalState.CRITICAL
    simulator.set_state(ClinicalState.STABLE)
    assert simulator.state is ClinicalState.STABLE


# ------------------------------------------------------------------------- the couplings


def test_hypoxia_drives_respiratory_rate_up() -> None:
    """Compensatory tachypnoea. Twin simulators, one insult, same random draws.

    Both twins consume the same number of random numbers per step, so the divergence is
    attributable to the coupling rather than to a different noise sequence.
    """
    control, hypoxic = steady(), steady()
    hypoxic.events.append(
        SimulationEvent(label="test", remaining_steps=60, offsets={"spo2": -18.0})
    )
    for _ in range(40):
        control.step()
        hypoxic.step()
    assert hypoxic.snapshot()["spo2"] < control.snapshot()["spo2"]
    assert hypoxic.snapshot()["resp_rate"] > control.snapshot()["resp_rate"]


def test_hypotension_drives_heart_rate_up() -> None:
    """Compensatory tachycardia, isolated the same way."""
    control, bleeding = steady(), steady()
    bleeding.events.append(
        SimulationEvent(label="test", remaining_steps=60, offsets={"bp_systolic": -40.0})
    )
    for _ in range(40):
        control.step()
        bleeding.step()
    assert bleeding.snapshot()["bp_systolic"] < control.snapshot()["bp_systolic"]
    assert bleeding.snapshot()["heart_rate"] > control.snapshot()["heart_rate"]


def test_consciousness_follows_gcs() -> None:
    simulator = steady(state=ClinicalState.CRITICAL)
    for _ in range(200):
        vitals = simulator.step()
        if vitals.gcs is not None:
            assert vitals.consciousness is Consciousness.from_gcs(vitals.gcs)


def test_supplemental_oxygen_raises_the_saturation_target() -> None:
    on_air = mean_of(steady(state=ClinicalState.DETERIORATING), "spo2")
    on_oxygen = mean_of(
        steady(state=ClinicalState.DETERIORATING, on_supplemental_oxygen=True), "spo2"
    )
    assert on_oxygen > on_air


def test_oxygen_is_reported_on_every_observation() -> None:
    vitals = steady(on_supplemental_oxygen=True).step()
    assert vitals.on_supplemental_oxygen is True


def test_age_shifts_the_baseline() -> None:
    """Older patients run stiffer. Not clinical truth - it keeps the cohort varied."""
    young = steady(age=30).snapshot()
    old = steady(age=88).snapshot()
    assert old["bp_systolic"] > young["bp_systolic"]


# ---------------------------------------------------------------------------- step size


def test_a_longer_step_reverts_further() -> None:
    """Reversion is calibrated per five minutes and has to scale with the interval.

    GCS is the channel to test it on: its noise term is zero in every profile, so the
    movement is pure reversion and the comparison is exact rather than probabilistic.
    """
    short, long = steady(state=ClinicalState.CRITICAL), steady(state=ClinicalState.CRITICAL)
    short.step(minutes=5.0)
    long.step(minutes=30.0)
    assert long.snapshot()["gcs"] < short.snapshot()["gcs"]


def test_the_step_scale_is_bounded() -> None:
    """An absurd interval must not overshoot the target or explode the noise."""
    simulator = steady(state=ClinicalState.CRITICAL)
    simulator.step(minutes=10_000.0)
    low, high = CLAMPS["gcs"]
    assert low <= simulator.snapshot()["gcs"] <= high


# ------------------------------------------------------------------------------- events


def test_every_advertised_event_can_be_injected() -> None:
    simulator = steady()
    for slug in EVENT_LIBRARY:
        assert simulator.inject(slug) is not None


def test_an_unknown_event_is_refused() -> None:
    assert steady().inject("spontaneous_recovery") is None


def test_an_injected_event_is_listed_then_expires() -> None:
    simulator = steady()
    label = simulator.inject("arrhythmia")
    assert simulator.active_events == (label,)

    steps = EVENT_LIBRARY["arrhythmia"][1]
    for _ in range(steps + 1):
        simulator.step()
    assert simulator.active_events == ()


def test_a_desaturation_event_desaturates() -> None:
    control, subject = steady(), steady()
    subject.inject("desaturation")
    for _ in range(12):
        control.step()
        subject.step()
    assert subject.snapshot()["spo2"] < control.snapshot()["spo2"]


def test_a_recovery_event_improves_things() -> None:
    """The library has to contain a way *down* as well as up, or the demo only worsens."""
    control, treated = (
        steady(state=ClinicalState.DETERIORATING),
        steady(state=ClinicalState.DETERIORATING),
    )
    treated.inject("recovery")
    for _ in range(20):
        control.step()
        treated.step()
    assert treated.snapshot()["spo2"] > control.snapshot()["spo2"]


def test_events_are_additive() -> None:
    simulator = steady()
    simulator.inject("sepsis")
    simulator.inject("haemorrhage")
    assert len(simulator.active_events) == 2


# ----------------------------------------------------------------------------- dropouts


def test_dropouts_produce_missing_channels() -> None:
    """Leads come off and cuffs cycle. A monitor that never loses a channel is fiction."""
    simulator = PatientSimulator(seed=3, dropouts=True)
    readings = [simulator.step() for _ in range(400)]
    assert any(vitals.spo2 is None for vitals in readings)
    assert any(vitals.bp_systolic is None for vitals in readings)


def test_dropouts_can_be_switched_off_for_determinism() -> None:
    readings = [steady().step() for _ in range(50)]
    assert all(vitals.heart_rate is not None for vitals in readings)
    assert all(vitals.temperature is not None for vitals in readings)


def test_the_cuff_drops_out_as_one_reading() -> None:
    """Systolic and diastolic come from the same measurement, so they vanish together."""
    simulator = PatientSimulator(seed=11, dropouts=True)
    for _ in range(400):
        vitals = simulator.step()
        if vitals.bp_systolic is None:
            assert vitals.bp_diastolic is None


# ------------------------------------------------------------------------ simulated ward


@pytest.fixture
def ward(config: Settings) -> SimulatedWard:
    return SimulatedWard(config=config)


def test_the_ward_fills_every_bed(ward: SimulatedWard, config: Settings) -> None:
    assert len(ward.patients) == config.bed_count
    assert list(ward.patients) == [f"P{index + 1:03d}" for index in range(config.bed_count)]
    assert [p.bed for p in ward.patients.values()] == [
        f"BED-{index + 1:02d}" for index in range(config.bed_count)
    ]


def test_the_first_two_beds_are_fixed(ward: SimulatedWard) -> None:
    """A reviewer opening the app sees one normal and one abnormal patient immediately."""
    occupants = list(ward.patients.values())
    assert occupants[0].state is ClinicalState.STABLE
    assert occupants[1].state is ClinicalState.DETERIORATING


def test_every_patient_is_labelled_synthetic(ward: SimulatedWard) -> None:
    """Nothing in this project may be mistaken for a real person's record."""
    for patient in ward.patients.values():
        assert "not a real person" in patient.notes.lower()


def test_the_ward_is_reproducible(config: Settings) -> None:
    def cohort() -> list[tuple[str, int, str]]:
        ward = SimulatedWard(config=config)
        return [(p.display_name, p.age, p.state.value) for p in ward.patients.values()]

    assert cohort() == cohort()


def test_an_explicit_seed_overrides_the_configured_one(config: Settings) -> None:
    first = SimulatedWard(config=config, seed=1)
    second = SimulatedWard(config=config, seed=2)
    assert [p.age for p in first.patients.values()] != [p.age for p in second.patients.values()]


def test_advance_reports_one_observation_per_bed(ward: SimulatedWard) -> None:
    readings = ward.advance()
    assert set(readings) == set(ward.patients)
    assert all(isinstance(vitals, Vitals) for vitals in readings.values())


def test_some_patients_are_on_the_scale_2_target(config: Settings) -> None:
    """Scale 2 exists in the code, so the demo ward has to contain someone on it."""
    ward = SimulatedWard(config=config.with_overrides(bed_count=24), seed=5)
    scales = {patient.spo2_scale for patient in ward.patients.values()}
    assert scales == {1, 2}


def test_the_source_is_named_for_the_reader(ward: SimulatedWard) -> None:
    assert "imulat" in ward.source_label


# ------------------------------------------------------------------- ward control surface


def test_the_ward_forwards_a_state_change(ward: SimulatedWard) -> None:
    assert ward.set_state("P001", "critical") is True
    assert ward.patients["P001"].state is ClinicalState.CRITICAL


def test_the_ward_forwards_an_oxygen_change(ward: SimulatedWard) -> None:
    assert ward.set_oxygen("P001", True) is True
    assert ward.patients["P001"].on_supplemental_oxygen is True


def test_the_ward_forwards_an_event(ward: SimulatedWard) -> None:
    assert ward.inject("P001", "sepsis") is not None
    assert ward.active_events("P001") != ()


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda w: w.set_state("P999", "stable"), False),
        (lambda w: w.set_oxygen("P999", True), False),
        (lambda w: w.inject("P999", "sepsis"), None),
        (lambda w: w.active_events("P999"), ()),
    ],
)
def test_an_unknown_bed_is_refused_not_raised(ward: SimulatedWard, call, expected) -> None:
    assert call(ward) == expected


def test_the_advertised_events_are_the_library(ward: SimulatedWard) -> None:
    assert ward.available_events == {slug: entry[0] for slug, entry in EVENT_LIBRARY.items()}


def test_state_mix_counts_every_trajectory(ward: SimulatedWard) -> None:
    counts = state_mix(ward.patients.values())
    assert set(counts) == {state.value for state in ClinicalState}
    assert sum(counts.values()) == len(ward.patients)


# ------------------------------------------------------------------------- replay ward


def write_replay(config: Settings, *, records: int = 2, hours: int = 6) -> None:
    """A minimal ``replay_series.csv`` in the long format the ETL emits."""
    config.ensure_directories()
    rows = []
    for record_id in range(1, records + 1):
        for hour in range(hours):
            for channel, value in (
                ("heart_rate", 80.0 + hour),
                ("spo2", 97.0 - hour),
                ("bp_systolic", 120.0 - hour),
                ("bp_diastolic", 70.0),
                ("resp_rate", 17.0),
                ("temperature", 36.9),
                ("gcs", 15.0),
                ("fio2", 0.4),
            ):
                rows.append(
                    {
                        "record_id": record_id,
                        "channel": channel,
                        "hours": float(hour),
                        "value": value,
                        "acuity": "medium",
                    }
                )
    pd.DataFrame(rows).to_csv(config.replay_path, index=False)


@pytest.fixture
def replay_config(config: Settings) -> Settings:
    cfg = config.with_overrides(vitals_source="replay")
    write_replay(cfg)
    return cfg


def test_replay_streams_recorded_stays(replay_config: Settings) -> None:
    ward = ReplayWard(config=replay_config)
    assert ward.patients
    assert all(pid.startswith("R") for pid in ward.patients)
    readings = ward.advance()
    assert set(readings) == set(ward.patients)
    first = next(iter(readings.values()))
    assert first.heart_rate == pytest.approx(80.0)
    assert first.spo2 == pytest.approx(97.0)


def test_replay_reads_oxygen_from_fio2(replay_config: Settings) -> None:
    """Room air is 21 %. Anything above it is supplemental, and NEWS2 scores it."""
    ward = ReplayWard(config=replay_config)
    assert all(vitals.on_supplemental_oxygen for vitals in ward.advance().values())


def test_replay_honours_the_bed_count(config: Settings) -> None:
    cfg = config.with_overrides(vitals_source="replay", bed_count=2)
    write_replay(cfg, records=8)
    assert len(ReplayWard(config=cfg).patients) == 2


def test_replay_says_where_the_data_came_from(replay_config: Settings) -> None:
    label = ReplayWard(config=replay_config).source_label
    assert "PhysioNet" in label


def test_replayed_patients_are_marked_de_identified(replay_config: Settings) -> None:
    for patient in ReplayWard(config=replay_config).patients.values():
        assert "de-identified" in patient.notes.lower()


def test_replay_loops_rather_than_running_out(replay_config: Settings) -> None:
    """A six-hour stay must not end the demo six ticks in."""
    ward = ReplayWard(config=replay_config)
    for _ in range(40):
        assert ward.advance()


def test_replay_refuses_the_controls_without_raising(replay_config: Settings) -> None:
    """Recorded history cannot be steered, but the dashboard offers the same buttons.

    Returning falsy rather than raising is what lets one UI serve both providers - the
    controls simply report that the source does not support them.
    """
    ward = ReplayWard(config=replay_config)
    patient_id = next(iter(ward.patients))
    assert ward.set_state(patient_id, "critical") is False
    assert ward.set_oxygen(patient_id, True) is False
    assert ward.inject(patient_id, "sepsis") is None
    assert ward.active_events(patient_id) == ()
    assert ward.available_events == {}


def test_replay_without_a_dataset_is_a_clear_error(config: Settings) -> None:
    """The message has to name the fix; a bare KeyError teaches the reader nothing."""
    with pytest.raises(FileNotFoundError, match="etl"):
        ReplayWard(config=config.with_overrides(vitals_source="replay"))


def test_an_empty_replay_file_is_rejected(config: Settings) -> None:
    cfg = config.with_overrides(vitals_source="replay")
    cfg.ensure_directories()
    pd.DataFrame(columns=["record_id", "channel", "hours", "value", "acuity"]).to_csv(
        cfg.replay_path, index=False
    )
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        ReplayWard(config=cfg)


# ------------------------------------------------------------------------- resampling


def test_resampling_carries_a_value_forward_while_it_is_fresh() -> None:
    frame = pd.DataFrame(
        [
            {"record_id": 1, "channel": "heart_rate", "hours": 0.0, "value": 80.0},
            {"record_id": 1, "channel": "heart_rate", "hours": 5.0, "value": 95.0},
        ]
    )
    frames = _resample_stay(frame, step_hours=1.0)
    assert frames[0]["heart_rate"] == pytest.approx(80.0)
    assert frames[int(REPLAY_STALENESS_HOURS)]["heart_rate"] == pytest.approx(80.0)


def test_resampling_lets_real_gaps_through() -> None:
    """A stale value is not a current one. Smoothing the gap away invents data."""
    frame = pd.DataFrame(
        [
            {"record_id": 1, "channel": "heart_rate", "hours": 0.0, "value": 80.0},
            {"record_id": 1, "channel": "heart_rate", "hours": 20.0, "value": 95.0},
        ]
    )
    frames = _resample_stay(frame, step_hours=1.0)
    stale = int(REPLAY_STALENESS_HOURS) + 2
    assert frames[stale]["heart_rate"] is None


def test_resampling_reports_nothing_before_the_first_sample() -> None:
    frame = pd.DataFrame([{"record_id": 1, "channel": "spo2", "hours": 4.0, "value": 92.0}])
    frames = _resample_stay(frame, step_hours=1.0)
    assert frames[0]["spo2"] is None
    assert frames[4]["spo2"] == pytest.approx(92.0)


def test_resampling_an_empty_stay_is_empty() -> None:
    assert _resample_stay(pd.DataFrame(columns=["hours", "value", "channel"])) == []


# ---------------------------------------------------------------------------- the factory


def test_the_factory_builds_the_simulator_by_default(config: Settings) -> None:
    provider = build_provider(config)
    assert isinstance(provider, SimulatedWard)
    assert isinstance(provider, VitalsProvider)


def test_the_factory_builds_replay_when_asked(replay_config: Settings) -> None:
    provider = build_provider(replay_config)
    assert isinstance(provider, ReplayWard)
    assert isinstance(provider, VitalsProvider)


def test_a_missing_dataset_falls_back_with_a_warning(
    config: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The difference between a deployable app and one that only runs on my machine.

    A reviewer who cloned the repo has no PhysioNet download. Asking for replay must give
    them a working ward and an explanation, not a traceback on startup.
    """
    with caplog.at_level(logging.WARNING):
        provider = build_provider(config.with_overrides(vitals_source="replay"))
    assert isinstance(provider, SimulatedWard)
    assert any("simulator" in record.message.lower() for record in caplog.records)
