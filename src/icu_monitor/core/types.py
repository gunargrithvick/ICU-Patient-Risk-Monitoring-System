"""Domain enumerations and value objects.

These are plain :mod:`dataclasses` on purpose: they sit in the per-tick hot loop
for every bed, so they avoid validation overhead. Pydantic models live at the API
boundary (:mod:`icu_monitor.api.schemas`) where untrusted input arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Acuity ladder shared by the model, the fusion engine, and the UI.

    The machine-learning model is trained on the first three levels. ``CRITICAL``
    is reserved for the fusion engine, which reaches it either by composite score
    or by a hard clinical override (for example a suspected fall).
    """

    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _RISK_RANK[self]

    @property
    def label(self) -> str:
        return _RISK_LABEL[self]

    @property
    def status_token(self) -> str:
        """Design-system status role: good / warning / serious / critical."""
        return _RISK_STATUS[self]

    def escalate_to(self, other: RiskLevel) -> RiskLevel:
        """Return whichever level is more severe (``UNKNOWN`` always loses)."""
        if self is RiskLevel.UNKNOWN:
            return other
        if other is RiskLevel.UNKNOWN:
            return self
        return self if self.rank >= other.rank else other

    @classmethod
    def coerce(cls, value: object) -> RiskLevel:
        """Best-effort parse from model output, JSON, or user input."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper()
        aliases = {"MODERATE": "MEDIUM", "SEVERE": "HIGH", "NONE": "UNKNOWN", "": "UNKNOWN"}
        text = aliases.get(text, text)
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN


_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.UNKNOWN: -1,
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}

_RISK_LABEL: dict[RiskLevel, str] = {
    RiskLevel.UNKNOWN: "No data",
    RiskLevel.LOW: "Low",
    RiskLevel.MEDIUM: "Medium",
    RiskLevel.HIGH: "High",
    RiskLevel.CRITICAL: "Critical",
}

# Status roles never carry meaning by colour alone; the UI always pairs them with
# an icon and this text label.
_RISK_STATUS: dict[RiskLevel, str] = {
    RiskLevel.UNKNOWN: "muted",
    RiskLevel.LOW: "good",
    RiskLevel.MEDIUM: "warning",
    RiskLevel.HIGH: "serious",
    RiskLevel.CRITICAL: "critical",
}

# ML training targets, in the fixed order used for probability vectors.
ML_RISK_CLASSES: tuple[str, ...] = (
    RiskLevel.LOW.value,
    RiskLevel.MEDIUM.value,
    RiskLevel.HIGH.value,
)


class Consciousness(str, Enum):
    """ACVPU scale used by NEWS2."""

    ALERT = "A"
    CONFUSION = "C"
    VOICE = "V"
    PAIN = "P"
    UNRESPONSIVE = "U"

    @property
    def label(self) -> str:
        return {
            Consciousness.ALERT: "Alert",
            Consciousness.CONFUSION: "New confusion",
            Consciousness.VOICE: "Responds to voice",
            Consciousness.PAIN: "Responds to pain",
            Consciousness.UNRESPONSIVE: "Unresponsive",
        }[self]

    @classmethod
    def parse(cls, value: object) -> Consciousness | None:
        """Parse an ACVPU letter or its English name; ``None`` when unrecognised.

        Unlike :meth:`RiskLevel.coerce` this refuses rather than defaulting. The channel
        is only ever supplied by a caller describing a patient in front of them, and
        reading an unrecognised word as ``ALERT`` would silently erase the three NEWS2
        points that "unresponsive" is worth. Returning ``None`` lets the caller decide -
        the API turns it into a 422 naming the accepted set.
        """
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        if not token:
            return None
        for level in cls:
            if token in {level.value.lower(), level.name.lower(), level.label.lower()}:
                return level
        return _ACVPU_ALIASES.get(token)

    @classmethod
    def from_gcs(cls, gcs: float | None) -> Consciousness:
        """Map a Glasgow Coma Scale total onto ACVPU.

        ICU datasets record GCS; NEWS2 expects ACVPU. The mapping below is the
        pragmatic one used for retrospective scoring: only a fully alert patient
        (GCS 15) scores zero, and the ACVPU rungs follow decreasing GCS.
        """
        if gcs is None or (isinstance(gcs, float) and math.isnan(gcs)):
            return cls.ALERT
        if gcs >= 15:
            return cls.ALERT
        if gcs >= 13:
            return cls.CONFUSION
        if gcs >= 9:
            return cls.VOICE
        if gcs >= 6:
            return cls.PAIN
        return cls.UNRESPONSIVE


#: Spellings a caller may reasonably type that are neither the ACVPU letter, the enum
#: name, nor the human label - all three of which :meth:`Consciousness.parse` already
#: accepts on its own.
_ACVPU_ALIASES: dict[str, Consciousness] = {
    "awake": Consciousness.ALERT,
    "confused": Consciousness.CONFUSION,
    "verbal": Consciousness.VOICE,
    "painful": Consciousness.PAIN,
    "unconscious": Consciousness.UNRESPONSIVE,
}

#: The canonical spellings, in ACVPU order, for the message a rejection comes back with.
ACVPU_TOKENS: tuple[str, ...] = tuple(
    token for level in Consciousness for token in (level.value, level.name.lower())
)


class ClinicalState(str, Enum):
    """Simulation trajectory for a synthetic patient."""

    STABLE = "stable"
    RECOVERING = "recovering"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @classmethod
    def coerce(cls, value: object) -> ClinicalState:
        """Parse a state from anything, defaulting to ``STABLE``.

        Used at the boundaries - stored rows, query strings, UI selections - so a stale
        value from an older schema degrades to the benign default instead of raising.
        """
        if isinstance(value, ClinicalState):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.STABLE


class Posture(str, Enum):
    """Coarse body position inferred from the detection bounding box."""

    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SEATED = "seated"
    RECUMBENT = "recumbent"
    COLLAPSED = "collapsed"

    @property
    def label(self) -> str:
        return self.value.capitalize()


class AlertKind(str, Enum):
    RISK_ESCALATION = "risk_escalation"
    NEWS2_TRIGGER = "news2_trigger"
    HYPOXIA = "hypoxia"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    HYPOTENSION = "hypotension"
    HYPERTENSION = "hypertension"
    PYREXIA = "pyrexia"
    HYPOTHERMIA = "hypothermia"
    TACHYPNOEA = "tachypnoea"
    FALL_SUSPECTED = "fall_suspected"
    BED_EXIT = "bed_exit"
    PATIENT_ABSENT = "patient_absent"
    SENSOR_FAILURE = "sensor_failure"

    @property
    def label(self) -> str:
        return {
            AlertKind.RISK_ESCALATION: "Risk escalation",
            AlertKind.NEWS2_TRIGGER: "NEWS2 threshold",
            AlertKind.HYPOXIA: "Hypoxaemia",
            AlertKind.TACHYCARDIA: "Tachycardia",
            AlertKind.BRADYCARDIA: "Bradycardia",
            AlertKind.HYPOTENSION: "Hypotension",
            AlertKind.HYPERTENSION: "Hypertension",
            AlertKind.PYREXIA: "Pyrexia",
            AlertKind.HYPOTHERMIA: "Hypothermia",
            AlertKind.TACHYPNOEA: "Tachypnoea",
            AlertKind.FALL_SUSPECTED: "Suspected fall",
            AlertKind.BED_EXIT: "Bed exit",
            AlertKind.PATIENT_ABSENT: "Patient not visible",
            AlertKind.SENSOR_FAILURE: "Sensor failure",
        }[self]


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used everywhere so timestamps are comparable."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Vitals:
    """One bedside monitor observation.

    Every field is optional because real monitors drop channels. Downstream code
    treats ``None`` as "not measured" rather than substituting a normal value,
    which keeps missingness visible instead of silently reassuring.
    """

    heart_rate: float | None = None
    spo2: float | None = None
    bp_systolic: float | None = None
    bp_diastolic: float | None = None
    resp_rate: float | None = None
    temperature: float | None = None
    consciousness: Consciousness = Consciousness.ALERT
    on_supplemental_oxygen: bool = False
    gcs: float | None = None
    recorded_at: datetime = field(default_factory=utcnow)

    @property
    def map_mmhg(self) -> float | None:
        """Mean arterial pressure estimated as ``(SBP + 2·DBP) / 3``."""
        if self.bp_systolic is None or self.bp_diastolic is None:
            return None
        return (self.bp_systolic + 2 * self.bp_diastolic) / 3

    @property
    def shock_index(self) -> float | None:
        """Heart rate divided by systolic pressure; > 0.9 suggests occult shock."""
        if self.heart_rate is None or not self.bp_systolic:
            return None
        return self.heart_rate / self.bp_systolic

    @property
    def measured_channels(self) -> int:
        return sum(
            value is not None
            for value in (
                self.heart_rate,
                self.spo2,
                self.bp_systolic,
                self.resp_rate,
                self.temperature,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "bp_systolic": self.bp_systolic,
            "bp_diastolic": self.bp_diastolic,
            "resp_rate": self.resp_rate,
            "temperature": self.temperature,
            "map": self.map_mmhg,
            "shock_index": self.shock_index,
            "consciousness": self.consciousness.value,
            "on_supplemental_oxygen": self.on_supplemental_oxygen,
            "gcs": self.gcs,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ParameterScore:
    """A single NEWS2 row: what was measured, what it scored, and why."""

    parameter: str
    display_name: str
    value: float | str | None
    unit: str
    score: int
    band: str
    is_red: bool = False

    @property
    def measured(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class NEWS2Result:
    """Outcome of a National Early Warning Score 2 calculation."""

    total: int
    components: tuple[ParameterScore, ...]
    risk_level: RiskLevel
    clinical_response: str
    has_red_score: bool
    missing_parameters: tuple[str, ...]
    scale: int = 1

    @property
    def max_total(self) -> int:
        """Ceiling of the NEWS2 scale (used to normalise into the composite)."""
        return 20

    @property
    def scored_components(self) -> tuple[ParameterScore, ...]:
        return tuple(c for c in self.components if c.score > 0)

    @property
    def normalised(self) -> float:
        """NEWS2 total mapped to 0-1 against the practical ceiling."""
        return min(1.0, self.total / float(self.max_total))

    @property
    def is_complete(self) -> bool:
        return not self.missing_parameters


@dataclass(frozen=True, slots=True)
class Detection:
    """One bounding box from the patient detector."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "patient"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "aspect_ratio": round(self.aspect_ratio, 3),
        }


@dataclass(slots=True)
class VisionSignal:
    """Everything the vision pipeline contributes to one tick."""

    available: bool = False
    patient_present: bool = False
    detections: tuple[Detection, ...] = ()
    posture: Posture = Posture.UNKNOWN
    fall_suspected: bool = False
    bed_exit_suspected: bool = False
    motion_index: float = 0.0
    backend: str = "off"
    source: str = "off"
    latency_ms: float = 0.0
    note: str = ""

    @property
    def person_count(self) -> int:
        return len(self.detections)

    @property
    def best_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "patient_present": self.patient_present,
            "person_count": self.person_count,
            "posture": self.posture.value,
            "fall_suspected": self.fall_suspected,
            "bed_exit_suspected": self.bed_exit_suspected,
            "motion_index": round(self.motion_index, 4),
            "backend": self.backend,
            "source": self.source,
            "latency_ms": round(self.latency_ms, 2),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RiskFactor:
    """One explainable contribution to the composite risk score."""

    source: str
    description: str
    points: float
    severity: str = "info"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "description": self.description,
            "points": round(self.points, 2),
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Fused, explainable risk output for one patient at one instant."""

    patient_id: str
    level: RiskLevel
    composite_score: float
    ml_level: RiskLevel
    ml_confidence: float | None
    ml_probabilities: dict[str, float]
    news2: NEWS2Result | None
    vision: VisionSignal | None
    factors: tuple[RiskFactor, ...]
    overrides: tuple[str, ...] = ()
    model_available: bool = True
    assessed_at: datetime = field(default_factory=utcnow)

    @property
    def top_factors(self) -> tuple[RiskFactor, ...]:
        return tuple(sorted(self.factors, key=lambda f: -f.points))

    @property
    def news2_total(self) -> int | None:
        return self.news2.total if self.news2 else None

    @property
    def summary(self) -> str:
        parts = [f"{self.level.label} risk ({self.composite_score:.0f}/100)"]
        if self.news2 is not None:
            parts.append(f"NEWS2 {self.news2.total}")
        if self.overrides:
            parts.append("override: " + ", ".join(self.overrides))
        return " · ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "level": self.level.value,
            "composite_score": round(self.composite_score, 2),
            "ml_level": self.ml_level.value,
            "ml_confidence": self.ml_confidence,
            "ml_probabilities": {k: round(v, 4) for k, v in self.ml_probabilities.items()},
            "news2_total": self.news2_total,
            "news2_response": self.news2.clinical_response if self.news2 else None,
            "vision": self.vision.as_dict() if self.vision else None,
            "factors": [f.as_dict() for f in self.top_factors],
            "overrides": list(self.overrides),
            "model_available": self.model_available,
            "assessed_at": self.assessed_at.isoformat(),
        }


@dataclass(slots=True)
class Patient:
    """An occupied ICU bed."""

    patient_id: str
    bed: str
    display_name: str
    age: int
    sex: str
    admitted_at: datetime
    primary_diagnosis: str
    state: ClinicalState = ClinicalState.STABLE
    on_supplemental_oxygen: bool = False
    spo2_scale: int = 1
    notes: str = ""

    @property
    def initials(self) -> str:
        return "".join(part[0] for part in self.display_name.split()[:2]).upper()

    def los_hours(self, now: datetime | None = None) -> float:
        reference = now or utcnow()
        admitted = self.admitted_at
        if admitted.tzinfo is None:
            admitted = admitted.replace(tzinfo=timezone.utc)
        return max(0.0, (reference - admitted).total_seconds() / 3600.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "bed": self.bed,
            "display_name": self.display_name,
            "age": self.age,
            "sex": self.sex,
            "admitted_at": self.admitted_at.isoformat(),
            "primary_diagnosis": self.primary_diagnosis,
            "state": self.state.value,
            "on_supplemental_oxygen": self.on_supplemental_oxygen,
            "spo2_scale": self.spo2_scale,
            "los_hours": round(self.los_hours(), 2),
        }


@dataclass(slots=True)
class Alert:
    """A raised clinical or technical alert.

    ``created_at`` is when the condition was first detected and never moves - that is the
    audit fact. ``last_seen_at`` tracks the most recent tick on which the condition was
    still true, which lets the wall display show current numbers without re-raising the
    alert and re-starting the alarm-fatigue problem de-duplication exists to solve.
    """

    patient_id: str
    kind: AlertKind
    severity: RiskLevel
    message: str
    detail: str = ""
    created_at: datetime = field(default_factory=utcnow)
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    alert_id: int | None = None
    last_seen_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.acknowledged_at is None

    @property
    def dedupe_key(self) -> str:
        return f"{self.patient_id}:{self.kind.value}"

    def duration_seconds(self, now: datetime | None = None) -> float:
        """How long this condition has been true, in seconds."""
        end = self.last_seen_at or now or utcnow()
        return max(0.0, (end - self.created_at).total_seconds())

    def acknowledge(self, by: str = "operator") -> None:
        if self.acknowledged_at is None:
            self.acknowledged_at = utcnow()
            self.acknowledged_by = by

    def refresh(self, message: str, moment: datetime) -> None:
        """Update the live text of an already-raised alert. Does not re-notify."""
        self.message = message
        self.last_seen_at = moment

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "kind": self.kind.value,
            "kind_label": self.kind.label,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "duration_seconds": round(self.duration_seconds(), 1),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "is_open": self.is_open,
        }


@dataclass(slots=True)
class BedSnapshot:
    """Everything the UI needs to render one bed after a tick."""

    patient: Patient
    vitals: Vitals
    assessment: RiskAssessment
    new_alerts: tuple[Alert, ...] = ()

    @property
    def level(self) -> RiskLevel:
        return self.assessment.level

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient": self.patient.as_dict(),
            "vitals": self.vitals.as_dict(),
            "assessment": self.assessment.as_dict(),
            "new_alerts": [a.as_dict() for a in self.new_alerts],
        }


__all__ = [
    "ACVPU_TOKENS",
    "ML_RISK_CLASSES",
    "Alert",
    "AlertKind",
    "BedSnapshot",
    "ClinicalState",
    "Consciousness",
    "Detection",
    "NEWS2Result",
    "ParameterScore",
    "Patient",
    "Posture",
    "RiskAssessment",
    "RiskFactor",
    "RiskLevel",
    "VisionSignal",
    "Vitals",
    "utcnow",
]
