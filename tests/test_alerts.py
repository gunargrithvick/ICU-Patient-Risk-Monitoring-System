"""Alerting: the rules, and the alarm-fatigue machinery around them.

The rules themselves are the easy part. What this module is really about is **not** alerting:
a monitor that raises the same hypoxia alarm on every tick trains staff to ignore it, which
is the mechanism behind the Joint Commission's alarm-management goal (NPSG 06.01.01). So the
tests here spend most of their effort on the suppression paths - dedupe, cooldown, refresh -
and on the one case where suppression must lose: an alert that gets *worse*.

The other distinction worth stating plainly is **active vs open**. Active means the condition
is true right now, which is what a wall display shows. Open means raised and not yet
acknowledged, which is what an audit trail shows. They are different sets, and adding them
together would be meaningless.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from icu_monitor.config import Settings
from icu_monitor.core.fusion import fuse_risk
from icu_monitor.core.news2 import calculate_news2
from icu_monitor.core.types import Alert, AlertKind, Patient, RiskLevel, Vitals
from icu_monitor.monitoring.alerts import (
    RULES,
    SENSOR_SILENCE_SECONDS,
    THRESHOLDS,
    AlertManager,
)

from .conftest import EPOCH, make_patient, make_vitals


def assess_for(config: Settings, patient: Patient, vitals: Vitals):
    return fuse_risk(
        patient_id=patient.patient_id,
        vitals=vitals,
        news2=calculate_news2(vitals, spo2_scale=patient.spo2_scale),
        config=config,
    )


def tick(
    manager: AlertManager,
    config: Settings,
    *,
    patient: Patient | None = None,
    at=EPOCH,
    **vitals_kwargs: object,
):
    """One evaluation pass at a chosen instant. Returns the alerts raised *this* tick."""
    subject = patient or make_patient()
    observation = make_vitals(at=at, **vitals_kwargs)
    assessment = assess_for(config, subject, observation)
    return manager.evaluate(subject, observation, assessment, now=at)


@pytest.fixture
def manager(config: Settings) -> AlertManager:
    return AlertManager(config=config)


# ----------------------------------------------------------------------------- rules


def test_a_healthy_patient_raises_nothing(manager: AlertManager, config: Settings) -> None:
    assert tick(manager, config) == ()
    assert manager.active == ()
    assert manager.open_alerts == ()


def test_every_rule_has_a_kind_a_severity_and_a_reason() -> None:
    """The detail text is what the dashboard shows the reader; an empty one is a bug."""
    assert RULES
    for rule in RULES:
        assert isinstance(rule.kind, AlertKind)
        assert isinstance(rule.severity, RiskLevel)
        assert rule.detail.strip()


def test_rule_kinds_are_unique() -> None:
    """Two rules sharing a kind would fight over one dedupe key."""
    kinds = [rule.kind for rule in RULES]
    assert len(kinds) == len(set(kinds))


@pytest.mark.parametrize(
    ("vitals_kwargs", "expected"),
    [
        ({"spo2": THRESHOLDS["spo2_scale1"] - 1}, AlertKind.HYPOXIA),
        ({"heart_rate": THRESHOLDS["pulse_high"]}, AlertKind.TACHYCARDIA),
        ({"heart_rate": THRESHOLDS["pulse_low"]}, AlertKind.BRADYCARDIA),
        ({"bp_systolic": THRESHOLDS["systolic_low"]}, AlertKind.HYPOTENSION),
        ({"bp_systolic": THRESHOLDS["systolic_high"]}, AlertKind.HYPERTENSION),
        ({"resp_rate": THRESHOLDS["resp_high"]}, AlertKind.TACHYPNOEA),
        ({"resp_rate": THRESHOLDS["resp_low"]}, AlertKind.TACHYPNOEA),
        ({"temperature": THRESHOLDS["temp_high"]}, AlertKind.PYREXIA),
        ({"temperature": THRESHOLDS["temp_low"]}, AlertKind.HYPOTHERMIA),
    ],
)
def test_each_physiological_rule_fires_at_its_threshold(
    manager: AlertManager, config: Settings, vitals_kwargs: dict, expected: AlertKind
) -> None:
    raised = tick(manager, config, **vitals_kwargs)
    assert expected in {alert.kind for alert in raised}


def test_hypoxia_uses_the_patients_own_target_range(config: Settings) -> None:
    """Scale 2 patients live at 88-92%. Alarming at 91% would be alarming at target.

    This is the clinical reason the alert manager takes the patient and not just the vitals:
    the same SpO₂ is an emergency for one bed and the plan for the next one along.
    """
    scale1 = make_patient("P001", spo2_scale=1)
    scale2 = make_patient("P002", spo2_scale=2)
    manager = AlertManager(config=config)

    assert AlertKind.HYPOXIA in {a.kind for a in tick(manager, config, patient=scale1, spo2=90.0)}
    assert AlertKind.HYPOXIA not in {
        a.kind for a in tick(manager, config, patient=scale2, spo2=90.0)
    }
    assert AlertKind.HYPOXIA in {a.kind for a in tick(manager, config, patient=scale2, spo2=86.0)}


def test_a_reading_exactly_at_target_does_not_alarm(config: Settings) -> None:
    """The hypoxia test is strictly below the floor, so the floor itself is acceptable."""
    manager = AlertManager(config=config)
    raised = tick(manager, config, spo2=THRESHOLDS["spo2_scale1"])
    assert AlertKind.HYPOXIA not in {alert.kind for alert in raised}


def test_news2_trigger_severity_tracks_the_total(manager: AlertManager, config: Settings) -> None:
    """One rule, two severities: urgent review and emergency response are not the same call."""
    medium = tick(manager, config, resp_rate=22.0, spo2=93.0, heart_rate=115.0)
    news2_alerts = [a for a in medium if a.kind is AlertKind.NEWS2_TRIGGER]
    assert news2_alerts and news2_alerts[0].severity is RiskLevel.MEDIUM

    manager.clear()
    high = tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    news2_alerts = [a for a in high if a.kind is AlertKind.NEWS2_TRIGGER]
    assert news2_alerts and news2_alerts[0].severity is RiskLevel.HIGH


def test_silent_sensors_raise_a_system_alert(manager: AlertManager, config: Settings) -> None:
    """A monitor that stops receiving data has to say so, loudly, and not just go quiet."""
    raised = tick(
        manager,
        config,
        heart_rate=None,
        spo2=None,
        bp_systolic=None,
        resp_rate=None,
        temperature=None,
    )
    assert AlertKind.SENSOR_FAILURE in {alert.kind for alert in raised}


def test_the_sensor_silence_window_is_declared_in_seconds() -> None:
    assert SENSOR_SILENCE_SECONDS > 0


# ------------------------------------------------------------------------ deduplication


def test_a_persisting_condition_raises_once(manager: AlertManager, config: Settings) -> None:
    """Ten ticks of the same hypoxia is one alarm, not ten."""
    first = tick(manager, config, spo2=85.0)
    assert AlertKind.HYPOXIA in {alert.kind for alert in first}

    for index in range(1, 10):
        again = tick(manager, config, spo2=85.0, at=EPOCH + timedelta(seconds=index))
        assert AlertKind.HYPOXIA not in {alert.kind for alert in again}

    hypoxia = [a for a in manager.history if a.kind is AlertKind.HYPOXIA]
    assert len(hypoxia) == 1


def test_a_persisting_alert_stays_current(manager: AlertManager, config: Settings) -> None:
    """Suppressed is not the same as stale.

    The alert is not re-raised, but its message and ``last_seen_at`` are refreshed, so the
    card reads "SpO₂ 82%" rather than the 85% it happened to open at nine minutes ago.
    """
    tick(manager, config, spo2=85.0)
    alert = next(a for a in manager.active if a.kind is AlertKind.HYPOXIA)
    opened_at = alert.created_at
    opening_message = alert.message

    later = EPOCH + timedelta(seconds=30)
    tick(manager, config, spo2=82.0, at=later)
    assert alert.created_at == opened_at
    assert alert.last_seen_at == later
    assert alert.message != opening_message
    assert "82" in alert.message
    assert alert.duration_seconds(now=later) == pytest.approx(30.0)


def test_two_patients_with_the_same_problem_get_two_alerts(
    manager: AlertManager, config: Settings
) -> None:
    """Dedupe is per patient *and* kind, not per kind."""
    tick(manager, config, patient=make_patient("P001"), spo2=85.0)
    tick(manager, config, patient=make_patient("P002"), spo2=85.0)
    hypoxia = [a for a in manager.active if a.kind is AlertKind.HYPOXIA]
    assert {a.patient_id for a in hypoxia} == {"P001", "P002"}


# ---------------------------------------------------------------------------- cooldown


def test_a_flapping_value_does_not_re_alarm_immediately(
    manager: AlertManager, config: Settings
) -> None:
    """A SpO₂ hovering on the threshold must not ring once per tick.

    This is the oscillation case: the condition clears, comes back a second later, and would
    raise a fresh alarm every time without a cooldown.
    """
    tick(manager, config, spo2=85.0)
    tick(manager, config, spo2=98.0, at=EPOCH + timedelta(seconds=1))  # clears
    again = tick(manager, config, spo2=85.0, at=EPOCH + timedelta(seconds=2))
    assert AlertKind.HYPOXIA not in {alert.kind for alert in again}


def test_the_cooldown_expires(manager: AlertManager, config: Settings) -> None:
    tick(manager, config, spo2=85.0)
    tick(manager, config, spo2=98.0, at=EPOCH + timedelta(seconds=1))
    beyond = EPOCH + timedelta(seconds=config.alert_cooldown_seconds + 2)
    again = tick(manager, config, spo2=85.0, at=beyond)
    assert AlertKind.HYPOXIA in {alert.kind for alert in again}


def test_escalation_beats_the_cooldown(manager: AlertManager, config: Settings) -> None:
    """The one case where suppression must lose.

    Suppressing a *worsening* condition is the failure mode that makes alarm-fatigue
    engineering dangerous rather than merely annoying, so a higher severity re-raises even
    inside the cooldown window.
    """
    tick(manager, config, resp_rate=22.0, spo2=93.0, heart_rate=115.0)
    opened = next(a for a in manager.active if a.kind is AlertKind.NEWS2_TRIGGER)
    assert opened.severity is RiskLevel.MEDIUM

    worse = tick(
        manager,
        config,
        resp_rate=26.0,
        spo2=90.0,
        heart_rate=135.0,
        at=EPOCH + timedelta(seconds=1),
    )
    escalated = [a for a in worse if a.kind is AlertKind.NEWS2_TRIGGER]
    assert escalated and escalated[0].severity is RiskLevel.HIGH


def test_an_improving_condition_does_not_re_alarm(manager: AlertManager, config: Settings) -> None:
    """The mirror of escalation: getting better is not news."""
    tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    improving = tick(
        manager,
        config,
        resp_rate=22.0,
        spo2=93.0,
        heart_rate=115.0,
        at=EPOCH + timedelta(seconds=1),
    )
    assert AlertKind.NEWS2_TRIGGER not in {alert.kind for alert in improving}


# ------------------------------------------------------------------- risk escalation


def test_risk_escalation_fires_on_the_transition_not_the_state(
    manager: AlertManager, config: Settings
) -> None:
    """ "Rose to high" is an event. Being high is a state, and states do not need re-announcing."""
    first = tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    assert AlertKind.RISK_ESCALATION in {alert.kind for alert in first}

    for index in range(1, 4):
        again = tick(
            manager,
            config,
            resp_rate=26.0,
            spo2=90.0,
            heart_rate=135.0,
            at=EPOCH + timedelta(seconds=index),
        )
        assert AlertKind.RISK_ESCALATION not in {alert.kind for alert in again}


def test_a_first_look_at_an_already_sick_patient_still_escalates(config: Settings) -> None:
    """No previous level is not the same as a previous level of LOW - but it must still alarm.

    A patient who is already deteriorating when the monitor is switched on is exactly the
    patient a "only alert on change" rule would silently miss.
    """
    manager = AlertManager(config=config)
    raised = tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    assert AlertKind.RISK_ESCALATION in {alert.kind for alert in raised}


def test_recovering_then_deteriorating_escalates_again(
    manager: AlertManager, config: Settings
) -> None:
    tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    tick(manager, config, at=EPOCH + timedelta(seconds=1))  # back to normal
    beyond = EPOCH + timedelta(seconds=config.alert_cooldown_seconds + 2)
    again = tick(manager, config, resp_rate=26.0, spo2=90.0, heart_rate=135.0, at=beyond)
    assert AlertKind.RISK_ESCALATION in {alert.kind for alert in again}


# --------------------------------------------------------------- active vs open vs history


def test_active_reflects_now_and_open_reflects_the_audit_trail(
    manager: AlertManager, config: Settings
) -> None:
    """The distinction the dashboard depends on, and must never add together.

    An alert whose condition has resolved but which nobody has acknowledged is *open* and
    not *active*: the wall display should stop shouting about it, while the audit trail must
    still show that it happened and was never signed off.

    Pyrexia is the subject because it fires exactly one rule - a low SpO₂ would also trip
    the NEWS2 and escalation rules and muddle the arithmetic.
    """
    tick(manager, config, temperature=39.5)
    assert [a.kind for a in manager.active] == [AlertKind.PYREXIA]
    assert [a.kind for a in manager.open_alerts] == [AlertKind.PYREXIA]

    tick(manager, config, temperature=36.8, at=EPOCH + timedelta(seconds=1))
    assert manager.active == ()
    assert [a.kind for a in manager.open_alerts] == [AlertKind.PYREXIA]
    assert len(manager.history) == 1


def test_history_is_newest_first(manager: AlertManager, config: Settings) -> None:
    tick(manager, config, spo2=85.0)
    tick(manager, config, heart_rate=135.0, at=EPOCH + timedelta(seconds=5))
    created = [alert.created_at for alert in manager.history]
    assert created == sorted(created, reverse=True)


def test_counts_are_reported_by_severity(manager: AlertManager, config: Settings) -> None:
    tick(manager, config, spo2=85.0, temperature=39.5)
    counts = manager.counts()
    assert counts
    assert sum(counts.values()) >= 2
    assert all(isinstance(value, int) for value in counts.values())


# ------------------------------------------------------------------- acknowledgement


def test_acknowledging_closes_one_alert(manager: AlertManager, config: Settings) -> None:
    tick(manager, config, spo2=85.0, heart_rate=135.0)
    target = manager.open_alerts[0]
    assert target.alert_id is not None

    acknowledged = manager.acknowledge(target.alert_id, by="charge-nurse")
    assert acknowledged is not None
    assert acknowledged.is_open is False
    assert acknowledged.acknowledged_by == "charge-nurse"
    assert acknowledged.acknowledged_at is not None
    assert target not in manager.open_alerts


def test_acknowledging_an_unknown_id_is_not_an_error(manager: AlertManager) -> None:
    assert manager.acknowledge(9999) is None


def test_acknowledge_all_can_be_scoped_to_one_patient(
    manager: AlertManager, config: Settings
) -> None:
    tick(manager, config, patient=make_patient("P001"), temperature=39.5)
    tick(manager, config, patient=make_patient("P002"), temperature=39.5)
    assert len(manager.open_alerts) == 2

    closed = manager.acknowledge_all(patient_id="P001", by="nurse")
    assert closed == 1
    assert [a.patient_id for a in manager.open_alerts] == ["P002"]

    assert manager.acknowledge_all(by="nurse") == 1
    assert manager.open_alerts == ()


def test_acknowledging_does_not_stop_the_condition_being_active(
    manager: AlertManager, config: Settings
) -> None:
    """Signing an alarm off does not fix the patient.

    The alert leaves the open list but the condition is still true, so it stays active - and
    because it is still tracked, it must not immediately re-raise either.
    """
    tick(manager, config, temperature=39.5)
    manager.acknowledge_all(by="nurse")
    assert manager.open_alerts == ()
    assert [a.kind for a in manager.active] == [AlertKind.PYREXIA]

    again = tick(manager, config, temperature=39.5, at=EPOCH + timedelta(seconds=1))
    assert AlertKind.PYREXIA not in {alert.kind for alert in again}


# ------------------------------------------------------------------------- housekeeping


def test_forget_clears_live_state_but_keeps_the_record(
    manager: AlertManager, config: Settings
) -> None:
    """Called when a bed is discharged.

    The next occupant must not inherit the previous patient's live alarms - but discharge is
    not a reason to erase what happened, so the history keeps both patients.
    """
    tick(manager, config, patient=make_patient("P001"), temperature=39.5)
    tick(manager, config, patient=make_patient("P002"), temperature=39.5)

    manager.forget("P001")
    assert {a.patient_id for a in manager.active} == {"P002"}
    assert {a.patient_id for a in manager.history} == {"P001", "P002"}


def test_a_discharged_bed_can_alarm_again_immediately(
    manager: AlertManager, config: Settings
) -> None:
    """Forgetting has to drop the cooldown too, or the new patient starts out muted."""
    tick(manager, config, spo2=85.0)
    manager.forget("P001")
    again = tick(manager, config, spo2=85.0, at=EPOCH + timedelta(seconds=1))
    assert AlertKind.HYPOXIA in {alert.kind for alert in again}


def test_clear_empties_everything(manager: AlertManager, config: Settings) -> None:
    tick(manager, config, spo2=85.0)
    manager.clear()
    assert manager.active == ()
    assert manager.open_alerts == ()
    assert manager.history == ()


def test_the_open_list_is_bounded(config: Settings, tmp_path) -> None:
    """An unbounded list is a memory leak on a ward that runs for weeks."""
    tight = config.with_overrides(alert_max_open=10, alert_cooldown_seconds=0.0)
    manager = AlertManager(config=tight)
    for index in range(40):
        patient = make_patient(f"P{index:03d}")
        tick(manager, tight, patient=patient, spo2=85.0, at=EPOCH + timedelta(seconds=index))
    assert len(manager.history) <= tight.alert_max_open


# --------------------------------------------------------------- restore after a restart


def _stored_alert(
    kind: AlertKind,
    *,
    patient_id: str = "P001",
    severity: RiskLevel = RiskLevel.HIGH,
    alert_id: int = 1,
    created_at=EPOCH,
    acknowledged: bool = False,
) -> Alert:
    """An alert as it comes back off the ledger after a restart."""
    alert = Alert(
        patient_id=patient_id,
        kind=kind,
        severity=severity,
        message=kind.label,
        created_at=created_at,
        last_seen_at=created_at,
        alert_id=alert_id,
    )
    if acknowledged:
        alert.acknowledge("nurse")
    return alert


def test_hydrate_rebuilds_the_open_ledger(manager: AlertManager) -> None:
    """A restart restores the full audit trail, but only the unacknowledged alerts go live.

    The acknowledged pyrexia belongs in history for the record; it must not reappear on the
    wall display. The still-open hypoxia seeds the per-condition dedup map so that a condition
    which is *still* true refreshes its restored alert instead of raising a duplicate.
    """
    stored = [
        _stored_alert(AlertKind.PYREXIA, alert_id=1, created_at=EPOCH, acknowledged=True),
        _stored_alert(AlertKind.HYPOXIA, alert_id=2, created_at=EPOCH + timedelta(seconds=5)),
    ]

    restored = manager.hydrate(stored)

    assert restored == 2
    assert {a.kind for a in manager.history} == {AlertKind.PYREXIA, AlertKind.HYPOXIA}
    assert [a.kind for a in manager.open_alerts] == [AlertKind.HYPOXIA]
    assert [a.kind for a in manager.active] == [AlertKind.HYPOXIA]


def test_hydrate_advances_the_id_counter_past_restored_ids(
    manager: AlertManager, config: Settings
) -> None:
    """A restored id of 50 must never be handed out again to a freshly raised alert."""
    manager.hydrate([_stored_alert(AlertKind.HYPOXIA, patient_id="P001", alert_id=50)])

    raised = tick(manager, config, patient=make_patient("P002"), spo2=85.0)
    hypoxia = next(a for a in raised if a.kind is AlertKind.HYPOXIA)
    assert hypoxia.alert_id is not None
    assert hypoxia.alert_id > 50


def test_seed_levels_prevents_a_spurious_re_escalation(config: Settings) -> None:
    """Priming the last-known level stops a still-high bed re-announcing on the first tick."""
    patient = make_patient("P001")
    observation = make_vitals(at=EPOCH, resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    level = assess_for(config, patient, observation).level
    assert level.rank >= RiskLevel.HIGH.rank  # the observation really is high-risk

    seeded = AlertManager(config=config)
    seeded.seed_levels({patient.patient_id: level})
    raised = seeded.evaluate(
        patient, observation, assess_for(config, patient, observation), now=EPOCH
    )
    assert AlertKind.RISK_ESCALATION not in {a.kind for a in raised}

    # Without the seed the identical first look reads as a fresh escalation.
    cold = AlertManager(config=config)
    raised_cold = cold.evaluate(
        patient, observation, assess_for(config, patient, observation), now=EPOCH
    )
    assert AlertKind.RISK_ESCALATION in {a.kind for a in raised_cold}
