"""Alert rules, de-duplication, and the open-alert ledger.

The original project's alerting was a ``print`` plus ``winsound.Beep`` on every frame
where a threshold was crossed. A patient resting at SpO₂ 91 % therefore produced one
alert per tick - thousands an hour - and the only rational response is to stop looking at
the channel. That is alarm fatigue, and reducing it is a Joint Commission National
Patient Safety Goal rather than a matter of taste.

Three rules make the channel usable:

**One alert per condition per patient.** ``Alert.dedupe_key`` is ``patient:kind``, and a
condition already active does not re-fire.

**A cooldown after it clears.** Re-arming waits
:attr:`~icu_monitor.config.Settings.alert_cooldown_seconds`, so a value oscillating
across a threshold does not chatter.

**Escalation always gets through.** The same condition returning at a higher severity
bypasses the cooldown, because suppressing a deterioration is the one failure worse
than noise.

Thresholds are the NEWS2 score-3 band edges rather than invented numbers, so an alert
and the score that explains it can never contradict each other.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import (
    Alert,
    AlertKind,
    Patient,
    RiskAssessment,
    RiskLevel,
    Vitals,
    utcnow,
)

logger = logging.getLogger(__name__)

#: Physiological cut points, taken from the NEWS2 score-3 band edges (RCP 2017). Pyrexia
#: uses the score-2 edge, which is the highest temperature band NEWS2 defines.
THRESHOLDS: dict[str, float] = {
    "spo2_scale1": 92.0,
    "spo2_scale2": 88.0,
    "pulse_high": 131.0,
    "pulse_low": 40.0,
    "systolic_low": 90.0,
    "systolic_high": 220.0,
    "resp_high": 25.0,
    "resp_low": 8.0,
    "temp_high": 39.1,
    "temp_low": 35.0,
}

#: Seconds a bed may report no measured channel at all before it reads as a fault.
SENSOR_SILENCE_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may look at for one bed on one tick."""

    patient: Patient
    vitals: Vitals
    assessment: RiskAssessment
    previous_level: RiskLevel = RiskLevel.UNKNOWN
    now: datetime = field(default_factory=utcnow)

    @property
    def spo2_floor(self) -> float:
        """The hypoxaemia threshold for this patient's oxygen target range.

        A patient on Scale 2 - chronic hypercapnic respiratory failure - is *targeted* at
        88-92 %, so alerting at 92 % would fire continuously on correct management.
        """
        key = "spo2_scale2" if self.patient.spo2_scale == 2 else "spo2_scale1"
        return THRESHOLDS[key]

    @property
    def seconds_since_reading(self) -> float:
        recorded = self.vitals.recorded_at
        if recorded.tzinfo is None:  # pragma: no cover - defensive
            return 0.0
        return max(0.0, (self.now - recorded).total_seconds())


@dataclass(frozen=True, slots=True)
class AlertRule:
    """One condition, its severity, and how it describes itself."""

    kind: AlertKind
    severity: RiskLevel
    test: Callable[[RuleContext], bool]
    message: Callable[[RuleContext], str]
    detail: str = ""
    severity_of: Callable[[RuleContext], RiskLevel] | None = None

    def severity_for(self, context: RuleContext) -> RiskLevel:
        return self.severity_of(context) if self.severity_of is not None else self.severity


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------


def _vision_flag(context: RuleContext, attribute: str) -> bool:
    vision = context.assessment.vision
    return bool(vision is not None and vision.available and getattr(vision, attribute))


def _below(value: float | None, limit: float) -> bool:
    return value is not None and value < limit


def _at_or_below(value: float | None, limit: float) -> bool:
    return value is not None and value <= limit


def _at_or_above(value: float | None, limit: float) -> bool:
    return value is not None and value >= limit


SAFETY_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        kind=AlertKind.FALL_SUSPECTED,
        severity=RiskLevel.CRITICAL,
        test=lambda c: _vision_flag(c, "fall_suspected"),
        message=lambda c: "Possible fall - patient appears to be out of bed and down",
        detail=(
            "Recumbent posture outside the bed region, held for the configured number of "
            "consecutive frames. Attend the bed; the camera cannot confirm injury."
        ),
    ),
    AlertRule(
        kind=AlertKind.BED_EXIT,
        severity=RiskLevel.HIGH,
        test=lambda c: _vision_flag(c, "bed_exit_suspected")
        and not _vision_flag(c, "fall_suspected"),
        message=lambda c: "Patient appears to have left the bed",
        detail="Upright or seated posture outside the bed region. Falls risk.",
    ),
    AlertRule(
        kind=AlertKind.PATIENT_ABSENT,
        severity=RiskLevel.MEDIUM,
        test=lambda c: (
            c.assessment.vision is not None
            and c.assessment.vision.available
            and not c.assessment.vision.patient_present
        ),
        message=lambda c: "No patient visible in the camera view",
        detail="Expected if the patient is off the unit; otherwise check the bed.",
    ),
)


PHYSIOLOGY_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        kind=AlertKind.HYPOXIA,
        severity=RiskLevel.HIGH,
        test=lambda c: _below(c.vitals.spo2, c.spo2_floor),
        message=lambda c: f"SpO₂ {c.vitals.spo2:.0f}% (below {c.spo2_floor:.0f}%)",
        detail="Check the probe, then oxygen delivery and airway.",
    ),
    AlertRule(
        kind=AlertKind.TACHYCARDIA,
        severity=RiskLevel.HIGH,
        test=lambda c: _at_or_above(c.vitals.heart_rate, THRESHOLDS["pulse_high"]),
        message=lambda c: f"Pulse {c.vitals.heart_rate:.0f} bpm (≥131)",
        detail="NEWS2 scores 3 for this band. Consider sepsis, pain, hypovolaemia.",
    ),
    AlertRule(
        kind=AlertKind.BRADYCARDIA,
        severity=RiskLevel.HIGH,
        test=lambda c: _at_or_below(c.vitals.heart_rate, THRESHOLDS["pulse_low"]),
        message=lambda c: f"Pulse {c.vitals.heart_rate:.0f} bpm (≤40)",
        detail="NEWS2 scores 3 for this band. Check perfusion and rhythm.",
    ),
    AlertRule(
        kind=AlertKind.HYPOTENSION,
        severity=RiskLevel.HIGH,
        test=lambda c: _at_or_below(c.vitals.bp_systolic, THRESHOLDS["systolic_low"]),
        message=lambda c: f"Systolic {c.vitals.bp_systolic:.0f} mmHg (≤90)",
        detail="NEWS2 scores 3 for this band. Assess volume status and lactate.",
    ),
    AlertRule(
        kind=AlertKind.HYPERTENSION,
        severity=RiskLevel.MEDIUM,
        test=lambda c: _at_or_above(c.vitals.bp_systolic, THRESHOLDS["systolic_high"]),
        message=lambda c: f"Systolic {c.vitals.bp_systolic:.0f} mmHg (≥220)",
        detail="NEWS2 scores 3 for this band.",
    ),
    AlertRule(
        kind=AlertKind.TACHYPNOEA,
        severity=RiskLevel.HIGH,
        test=lambda c: _at_or_above(c.vitals.resp_rate, THRESHOLDS["resp_high"])
        or _at_or_below(c.vitals.resp_rate, THRESHOLDS["resp_low"]),
        message=lambda c: f"Respiratory rate {c.vitals.resp_rate:.0f} /min (outside 9-24)",
        detail="Respiratory rate is the earliest and most predictive NEWS2 parameter.",
    ),
    AlertRule(
        kind=AlertKind.PYREXIA,
        severity=RiskLevel.MEDIUM,
        test=lambda c: _at_or_above(c.vitals.temperature, THRESHOLDS["temp_high"]),
        message=lambda c: f"Temperature {c.vitals.temperature:.1f}°C (≥39.1)",
        detail="Consider cultures and the sepsis screening pathway.",
    ),
    AlertRule(
        kind=AlertKind.HYPOTHERMIA,
        severity=RiskLevel.MEDIUM,
        test=lambda c: _at_or_below(c.vitals.temperature, THRESHOLDS["temp_low"]),
        message=lambda c: f"Temperature {c.vitals.temperature:.1f}°C (≤35.0)",
        detail="NEWS2 scores 3 for this band. Also a sepsis presentation.",
    ),
)


def _news2_total(context: RuleContext) -> int:
    news2 = context.assessment.news2
    return news2.total if news2 is not None else 0


def _news2_severity(context: RuleContext) -> RiskLevel:
    news2 = context.assessment.news2
    if news2 is None:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH if news2.total >= 7 else RiskLevel.MEDIUM


SYSTEM_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        kind=AlertKind.NEWS2_TRIGGER,
        severity=RiskLevel.HIGH,
        severity_of=_news2_severity,
        test=lambda c: _news2_total(c) >= 5
        or bool(c.assessment.news2 and c.assessment.news2.has_red_score),
        message=lambda c: (
            f"NEWS2 {_news2_total(c)}"
            + (
                " with a red score"
                if c.assessment.news2 and c.assessment.news2.has_red_score
                else ""
            )
        ),
        detail="Escalate per the RCP graded response shown on the patient page.",
    ),
    AlertRule(
        kind=AlertKind.RISK_ESCALATION,
        severity=RiskLevel.HIGH,
        severity_of=lambda c: c.assessment.level,
        # Fire on the transition into HIGH or CRITICAL, not on every tick spent there.
        test=lambda c: c.assessment.level.rank >= RiskLevel.HIGH.rank
        and c.assessment.level.rank > c.previous_level.rank,
        message=lambda c: (
            f"Composite risk rose to {c.assessment.level.label.lower()} "
            f"({c.assessment.composite_score:.0f}/100)"
        ),
        detail="Open the patient page for the itemised contributions.",
    ),
    AlertRule(
        kind=AlertKind.SENSOR_FAILURE,
        severity=RiskLevel.MEDIUM,
        test=lambda c: c.vitals.measured_channels == 0
        or c.seconds_since_reading > SENSOR_SILENCE_SECONDS,
        message=lambda c: (
            "No vital signs are being measured"
            if c.vitals.measured_channels == 0
            else f"No new observation for {c.seconds_since_reading:.0f}s"
        ),
        detail="Risk scoring is degraded until at least one channel returns.",
    ),
)

#: The full rule set, safety first so the most urgent alerts are raised first.
RULES: tuple[AlertRule, ...] = SAFETY_RULES + PHYSIOLOGY_RULES + SYSTEM_RULES


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------


class AlertManager:
    """Evaluates the rules and keeps the de-duplicated ledger.

    Two collections, because they answer different questions. *Active* alerts are the
    conditions still true right now - that is the wall display. *Open* alerts are the
    ones nobody has acknowledged yet, whether or not the condition has since resolved -
    that is the audit trail, and a transient critical event must not vanish from it just
    because the number came back.
    """

    def __init__(
        self,
        *,
        config: Settings | None = None,
        rules: Iterable[AlertRule] = RULES,
    ) -> None:
        self._config = config or default_settings
        self._rules = tuple(rules)
        self._active: dict[str, Alert] = {}
        self._cleared_at: dict[str, datetime] = {}
        self._previous_level: dict[str, RiskLevel] = {}
        self._history: deque[Alert] = deque(maxlen=self._config.alert_max_open)
        self._next_id = 1

    # -- read-side ---------------------------------------------------------------------

    @property
    def active(self) -> tuple[Alert, ...]:
        """Conditions currently true, most severe first."""
        return tuple(sorted(self._active.values(), key=lambda a: (-a.severity.rank, a.created_at)))

    @property
    def history(self) -> tuple[Alert, ...]:
        """Every alert raised recently, newest first."""
        return tuple(reversed(self._history))

    @property
    def open_alerts(self) -> tuple[Alert, ...]:
        """Raised but not yet acknowledged, newest first."""
        return tuple(alert for alert in self.history if alert.is_open)

    def counts(self) -> dict[str, int]:
        by_severity: dict[str, int] = {}
        for alert in self._active.values():
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
        return {
            "active": len(self._active),
            "open": sum(1 for alert in self._history if alert.is_open),
            "history": len(self._history),
            **by_severity,
        }

    # -- evaluation --------------------------------------------------------------------

    def evaluate(
        self,
        patient: Patient,
        vitals: Vitals,
        assessment: RiskAssessment,
        *,
        now: datetime | None = None,
    ) -> tuple[Alert, ...]:
        """Run every rule for one bed and return only the *newly* raised alerts."""
        moment = now or utcnow()
        context = RuleContext(
            patient=patient,
            vitals=vitals,
            assessment=assessment,
            previous_level=self._previous_level.get(patient.patient_id, RiskLevel.UNKNOWN),
            now=moment,
        )

        raised: list[Alert] = []
        for rule in self._rules:
            key = f"{patient.patient_id}:{rule.kind.value}"
            try:
                fires = bool(rule.test(context))
            except Exception:  # pragma: no cover - a broken rule must not stop monitoring
                logger.exception("Alert rule %s failed for %s", rule.kind.value, patient.bed)
                continue

            if not fires:
                if self._active.pop(key, None) is not None:
                    self._cleared_at[key] = moment
                continue

            severity = rule.severity_for(context)
            existing = self._active.get(key)
            if existing is not None:
                if severity.rank <= existing.severity.rank:
                    # Still true, not worse: keep the one alert but let its text follow the
                    # live numbers, so the display never shows "SpO₂ 91%" for a patient who
                    # has since dropped to 84%.
                    existing.refresh(self._message_for(rule, context), moment)
                    continue
                self._cleared_at.pop(key, None)  # escalation bypasses the cooldown

            elif self._in_cooldown(key, moment):
                continue

            alert = self._raise(patient.patient_id, rule, severity, context, moment)
            raised.append(alert)

        self._previous_level[patient.patient_id] = assessment.level
        return tuple(raised)

    def _in_cooldown(self, key: str, moment: datetime) -> bool:
        cleared = self._cleared_at.get(key)
        if cleared is None:
            return False
        return (moment - cleared).total_seconds() < self._config.alert_cooldown_seconds

    @staticmethod
    def _message_for(rule: AlertRule, context: RuleContext) -> str:
        """Render a rule's message, falling back to its label if formatting fails."""
        try:
            return rule.message(context)
        except Exception:  # pragma: no cover - defensive
            return rule.kind.label

    def _raise(
        self,
        patient_id: str,
        rule: AlertRule,
        severity: RiskLevel,
        context: RuleContext,
        moment: datetime,
    ) -> Alert:
        alert = Alert(
            patient_id=patient_id,
            kind=rule.kind,
            severity=severity,
            message=self._message_for(rule, context),
            detail=rule.detail,
            created_at=moment,
            last_seen_at=moment,
            alert_id=self._next_id,
        )
        self._next_id += 1
        self._active[f"{patient_id}:{rule.kind.value}"] = alert
        self._history.append(alert)
        logger.info("[%s] %s - %s", severity.value, rule.kind.label, alert.message)
        return alert

    # -- write-side --------------------------------------------------------------------

    def acknowledge(self, alert_id: int, *, by: str = "operator") -> Alert | None:
        """Acknowledge one alert by id. Acknowledging does not clear the condition."""
        for alert in self._history:
            if alert.alert_id == alert_id:
                alert.acknowledge(by)
                return alert
        return None

    def acknowledge_all(self, *, patient_id: str | None = None, by: str = "operator") -> int:
        """Acknowledge every open alert, optionally for one patient only."""
        count = 0
        for alert in self._history:
            if not alert.is_open:
                continue
            if patient_id is not None and alert.patient_id != patient_id:
                continue
            alert.acknowledge(by)
            count += 1
        return count

    def forget(self, patient_id: str) -> None:
        """Drop all state for a discharged bed."""
        for key in [k for k in self._active if k.startswith(f"{patient_id}:")]:
            self._active.pop(key, None)
        for key in [k for k in self._cleared_at if k.startswith(f"{patient_id}:")]:
            self._cleared_at.pop(key, None)
        self._previous_level.pop(patient_id, None)

    def clear(self) -> None:
        self._active.clear()
        self._cleared_at.clear()
        self._previous_level.clear()
        self._history.clear()
        self._next_id = 1


__all__ = [
    "PHYSIOLOGY_RULES",
    "RULES",
    "SAFETY_RULES",
    "SENSOR_SILENCE_SECONDS",
    "SYSTEM_RULES",
    "THRESHOLDS",
    "AlertManager",
    "AlertRule",
    "RuleContext",
]
