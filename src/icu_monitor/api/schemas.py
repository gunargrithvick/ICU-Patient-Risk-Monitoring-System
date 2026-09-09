"""Request and response models for the HTTP API.

These are deliberately *not* the domain dataclasses. A public schema and an internal
model change for different reasons, and coupling them means every refactor of
:mod:`icu_monitor.core.types` is a breaking API change. The domain objects already know
how to describe themselves (``as_dict``), so most responses are thin adapters over that.

Only the *request* models do real work: :class:`VitalsIn` validates ranges before the
values reach the scoring code, because a systolic pressure of 4,000 is a client bug and
should come back as a 422 rather than a plausible-looking risk score.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from icu_monitor.core.types import ACVPU_TOKENS, Consciousness, Vitals, utcnow

# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class VitalsIn(BaseModel):
    """One observation as posted by a client. Every channel is optional."""

    heart_rate: float | None = Field(default=None, ge=0, le=350, description="bpm")
    spo2: float | None = Field(default=None, ge=0, le=100, description="%")
    bp_systolic: float | None = Field(default=None, ge=0, le=350, description="mmHg")
    bp_diastolic: float | None = Field(default=None, ge=0, le=250, description="mmHg")
    resp_rate: float | None = Field(default=None, ge=0, le=90, description="breaths/min")
    temperature: float | None = Field(default=None, ge=20, le=45, description="°C")
    gcs: float | None = Field(default=None, ge=3, le=15, description="Glasgow Coma Scale")
    consciousness: str | None = Field(
        default=None,
        description="ACVPU letter or name: A/alert, C/confusion, V/voice, P/pain, U/unresponsive",
    )
    on_supplemental_oxygen: bool = False
    recorded_at: datetime | None = None

    @field_validator("consciousness")
    @classmethod
    def _known_consciousness(cls, value: str | None) -> str | None:
        """Normalise to the ACVPU letter, or refuse and name what is accepted.

        Refusing rather than defaulting is deliberate: ``ALERT`` is worth zero NEWS2
        points and ``U`` is worth three, so quietly reading an unrecognised word as
        "alert" would understate exactly the patient this system exists to escalate.
        """
        if value is None:
            return None
        level = Consciousness.parse(value)
        if level is None:
            raise ValueError(f"consciousness must be one of {list(ACVPU_TOKENS)}")
        return level.value

    def to_domain(self) -> Vitals:
        """Convert to the internal value object, deriving ACVPU from GCS if needed."""
        if self.consciousness is not None:
            consciousness = Consciousness(self.consciousness)
        elif self.gcs is not None:
            consciousness = Consciousness.from_gcs(self.gcs)
        else:
            consciousness = Consciousness.ALERT
        return Vitals(
            heart_rate=self.heart_rate,
            spo2=self.spo2,
            bp_systolic=self.bp_systolic,
            bp_diastolic=self.bp_diastolic,
            resp_rate=self.resp_rate,
            temperature=self.temperature,
            consciousness=consciousness,
            on_supplemental_oxygen=self.on_supplemental_oxygen,
            gcs=self.gcs,
            recorded_at=self.recorded_at or utcnow(),
        )


class ScoreRequest(BaseModel):
    """Score one observation without touching ward state."""

    vitals: VitalsIn
    patient_id: str = Field(default="adhoc", max_length=32)
    spo2_scale: Literal[1, 2] = Field(
        default=1,
        description="NEWS2 SpO₂ target scale. 2 = chronic hypercapnic respiratory failure.",
    )
    age: int | None = Field(default=None, ge=0, le=120)
    include_model: bool = Field(
        default=True, description="Set false to score on NEWS2 alone (no ML channel)."
    )


class BatchScoreRequest(BaseModel):
    items: list[ScoreRequest] = Field(min_length=1, max_length=500)


class AcknowledgeRequest(BaseModel):
    by: str = Field(default="operator", max_length=64)


class StateRequest(BaseModel):
    state: Literal["stable", "recovering", "deteriorating", "critical"]


class OxygenRequest(BaseModel):
    on: bool
    scale: Literal[1, 2] | None = None


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str
    at: datetime


class ComponentStatus(BaseModel):
    name: str
    ready: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    ready: bool
    components: list[ComponentStatus]
    at: datetime


class ScoreResponse(BaseModel):
    patient_id: str
    level: str
    composite_score: float
    news2_total: int | None
    news2_response: str | None
    factors: list[dict[str, Any]]
    overrides: list[str]
    model_available: bool
    ml_level: str
    ml_probabilities: dict[str, float]
    assessed_at: datetime


class BatchScoreResponse(BaseModel):
    count: int
    results: list[ScoreResponse]


class AcknowledgeResponse(BaseModel):
    acknowledged: int
    alert_id: int | None = None
    by: str


class ModelInfoResponse(BaseModel):
    available: bool
    version: str | None
    algorithm: str | None = None
    trained_at: str | None = None
    feature_count: int | None = None
    classes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    card: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    ok: bool
    message: str


__all__ = [
    "AcknowledgeRequest",
    "AcknowledgeResponse",
    "BatchScoreRequest",
    "BatchScoreResponse",
    "ComponentStatus",
    "HealthResponse",
    "MessageResponse",
    "ModelInfoResponse",
    "OxygenRequest",
    "ReadyResponse",
    "ScoreRequest",
    "ScoreResponse",
    "StateRequest",
    "VitalsIn",
]
