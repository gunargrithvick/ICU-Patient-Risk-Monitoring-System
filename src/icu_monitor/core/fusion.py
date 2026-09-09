"""Risk fusion: combine the ML model, NEWS2, and the vision channel into one score.

The design goal is **explainability**. A single number is useless to a clinician
unless they can see what drove it, so the engine returns the composite *and* the
itemised contributions that add up to it, plus any hard override that fired.

Composite score
---------------
Each channel produces a normalised severity in ``[0, 1]``. The composite is the
weighted sum scaled to ``0-100``::

    composite = 100 * (w_ml·ml + w_news2·news2 + w_vision·vision)

Weights come from configuration and are normalised to sum to 1. NEWS2 is further
decomposed into one factor per physiological parameter, so the factor list
literally accounts for the composite rather than approximating it.

**A channel that is switched off surrenders its weight rather than voting zero.** The
weights are renormalised across the channels actually reporting, because the alternative
silently caps the scale: with no camera, ``w_vision`` would contribute a permanent 0 and no
patient could ever exceed 85, making ``CRITICAL`` unreachable on the default configuration -
which ships with ``frame_source="off"``. Counting a missing camera as reassurance is the same
mistake as counting a missing SpO₂ as 98%, and :mod:`icu_monitor.core.news2` already refuses
that one. See :func:`_effective_weights`.

Overrides
---------
Weighted averages dilute single catastrophic findings, so a small set of hard
rules can raise (never lower) the final level regardless of the composite:

* a suspected fall or collapse -> ``CRITICAL``
* NEWS2 total at or above the emergency threshold -> at least ``HIGH``
* a NEWS2 red score (any single parameter scoring 3) -> at least ``MEDIUM``
* extreme single values (SpO₂ ≤ 85%, pulse ≤ 40 or ≥ 140, SBP ≤ 80) -> at least ``HIGH``

An override that mandates a level also mandates a floor under the composite, and that
floor is *graded across the band* rather than pinned to its edge - otherwise every
escalated patient would read exactly 65 and the ward list would lose its ordering where
it matters most. See :func:`_graded_floor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import (
    ML_RISK_CLASSES,
    NEWS2Result,
    Posture,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    VisionSignal,
    Vitals,
)


@dataclass(frozen=True, slots=True)
class MLPrediction:
    """Output of the bedside risk model, or a clearly-marked absence of one."""

    level: RiskLevel = RiskLevel.UNKNOWN
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    available: bool = False
    model_version: str = "unavailable"

    @property
    def severity(self) -> float:
        """Expected severity in ``[0, 1]``: MEDIUM counts half, HIGH counts full.

        Using the full probability vector rather than the ``argmax`` label keeps the
        composite smooth - a patient at ``P(high)=0.49`` should not read the same as
        one at ``P(high)=0.02`` merely because both argmax to ``LOW``.
        """
        if not self.available or not self.probabilities:
            return 0.0
        low = self.probabilities.get(RiskLevel.LOW.value, 0.0)
        medium = self.probabilities.get(RiskLevel.MEDIUM.value, 0.0)
        high = self.probabilities.get(RiskLevel.HIGH.value, 0.0)
        total = low + medium + high
        if total <= 0:
            return 0.0
        return (0.5 * medium + 1.0 * high) / total

    @classmethod
    def unavailable(cls, reason: str = "model not loaded") -> MLPrediction:
        return cls(available=False, model_version=reason)


# --------------------------------------------------------------------------------------
# Vision severity
# --------------------------------------------------------------------------------------

#: Severity weight per vision finding, highest wins. Keyed for display.
VISION_SEVERITY: dict[str, tuple[float, str]] = {
    "fall": (1.00, "Fall or collapse posture detected"),
    "bed_exit": (0.60, "Patient appears to have left the bed"),
    "absent": (0.40, "No patient visible in frame"),
    "agitation": (0.30, "Sustained high movement (possible agitation)"),
    "recumbent_still": (0.05, "Recumbent and settled"),
    "settled": (0.00, "Patient present and settled"),
}

#: Motion index above this is treated as agitation rather than normal movement.
#: Calibrated against the synthetic ward: a settled patient runs at 0.11-0.18 and a
#: recovering one peaks near 0.33, while sustained agitation averages 0.50. The cut sits
#: above the former and below the latter, so ordinary repositioning does not read as
#: distress.
AGITATION_THRESHOLD = 0.42


def vision_severity(vision: VisionSignal | None) -> tuple[float, str, str]:
    """Reduce a vision signal to ``(severity, key, description)``."""
    if vision is None or not vision.available:
        return 0.0, "unavailable", "Vision channel offline"

    if vision.fall_suspected or vision.posture is Posture.COLLAPSED:
        weight, text = VISION_SEVERITY["fall"]
        return weight, "fall", text
    if vision.bed_exit_suspected:
        weight, text = VISION_SEVERITY["bed_exit"]
        return weight, "bed_exit", text
    if not vision.patient_present:
        weight, text = VISION_SEVERITY["absent"]
        return weight, "absent", text
    if vision.motion_index >= AGITATION_THRESHOLD:
        weight, text = VISION_SEVERITY["agitation"]
        return weight, "agitation", f"{text} (index {vision.motion_index:.2f})"
    if vision.posture is Posture.RECUMBENT:
        weight, text = VISION_SEVERITY["recumbent_still"]
        return weight, "recumbent_still", text
    weight, text = VISION_SEVERITY["settled"]
    return weight, "settled", text


# --------------------------------------------------------------------------------------
# Hard physiological overrides
# --------------------------------------------------------------------------------------


def _extreme_value_overrides(vitals: Vitals) -> list[tuple[str, RiskLevel]]:
    """Single values severe enough to escalate on their own."""
    triggers: list[tuple[str, RiskLevel]] = []
    if vitals.spo2 is not None and vitals.spo2 <= 85:
        triggers.append((f"SpO₂ {vitals.spo2:.0f}% (≤85)", RiskLevel.HIGH))
    if vitals.heart_rate is not None:
        if vitals.heart_rate <= 40:
            triggers.append((f"Pulse {vitals.heart_rate:.0f} bpm (≤40)", RiskLevel.HIGH))
        elif vitals.heart_rate >= 140:
            triggers.append((f"Pulse {vitals.heart_rate:.0f} bpm (≥140)", RiskLevel.HIGH))
    if vitals.bp_systolic is not None and vitals.bp_systolic <= 80:
        triggers.append((f"Systolic {vitals.bp_systolic:.0f} mmHg (≤80)", RiskLevel.HIGH))
    if vitals.temperature is not None and vitals.temperature <= 34.0:
        triggers.append((f"Temperature {vitals.temperature:.1f}°C (≤34)", RiskLevel.HIGH))
    return triggers


def band_for_score(score: float, config: Settings | None = None) -> RiskLevel:
    """Map a 0-100 composite onto the acuity ladder using configured cut points."""
    cfg = config or default_settings
    if score >= cfg.composite_critical_threshold:
        return RiskLevel.CRITICAL
    if score >= cfg.composite_high_threshold:
        return RiskLevel.HIGH
    if score >= cfg.composite_medium_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _graded_floor(
    *,
    band_floor: float,
    band_ceiling: float,
    value: float,
    low: float,
    high: float,
) -> float:
    """A mandated floor that still ranks patients within the band it lifts them into.

    ``max(composite, threshold)`` is monotone but not injective: every patient an override
    catches lands on exactly the band edge, so NEWS2 20 reads identically to NEWS2 7 and
    the ward list loses its ordering precisely where ordering matters most. Spreading the
    floor across the band keeps the guarantee - "this patient is at least HIGH" - while
    preserving rank, and stops a point short of the next edge so the composite can never
    contradict the level printed beside it.
    """
    span = max(1e-9, high - low)
    fraction = min(1.0, max(0.0, (value - low) / span))
    headroom = max(0.0, band_ceiling - band_floor - 1.0)
    return band_floor + fraction * headroom


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------


def _effective_weights(
    weights: dict[str, float], *, ml: bool, news2: bool, vision: bool
) -> dict[str, float]:
    """Redistribute the weight of any channel that is not reporting.

    Without this the composite is quietly bounded by whatever is plugged in: a deployment
    with no camera can reach at most 85 and therefore never shows ``CRITICAL``, and one with
    no trained artefact caps out at 55. Renormalising over the live channels keeps the whole
    0-100 scale meaningful in every supported configuration, and keeps a switched-off channel
    from acting as evidence of wellbeing.

    If nothing is reporting the weights are returned untouched - there is no severity to
    scale, and :func:`fuse_risk` reports ``UNKNOWN`` for that case anyway.
    """
    available = {"ml": ml, "news2": news2, "vision": vision}
    live = sum(weight for name, weight in weights.items() if available.get(name))
    if live <= 0.0:
        return dict(weights)
    scale = sum(weights.values()) / live
    return {
        name: (weight * scale if available.get(name) else 0.0) for name, weight in weights.items()
    }


def fuse_risk(
    *,
    patient_id: str,
    vitals: Vitals,
    news2: NEWS2Result | None = None,
    ml: MLPrediction | None = None,
    vision: VisionSignal | None = None,
    config: Settings | None = None,
) -> RiskAssessment:
    """Combine every available channel into one explainable assessment."""
    cfg = config or default_settings
    prediction = ml or MLPrediction.unavailable()
    weights = _effective_weights(
        cfg.fusion_weights,
        ml=prediction.available,
        news2=news2 is not None,
        vision=vision is not None and vision.available,
    )
    #: Appended to an offline channel's factor so the reader knows its share went elsewhere
    #: rather than counting as a reassuring zero.
    redistributed = " - weight redistributed" if any(w > 0.0 for w in weights.values()) else ""

    factors: list[RiskFactor] = []
    composite = 0.0

    # -- Machine learning --------------------------------------------------------------
    ml_budget = weights["ml"] * 100.0
    if prediction.available:
        ml_points = prediction.severity * ml_budget
        composite += ml_points
        p_high = prediction.probabilities.get(RiskLevel.HIGH.value, 0.0)
        p_medium = prediction.probabilities.get(RiskLevel.MEDIUM.value, 0.0)
        factors.append(
            RiskFactor(
                source="model",
                description=(
                    f"Risk model: P(high) {p_high:.0%}, P(medium) {p_medium:.0%} "
                    f"→ {prediction.level.label.lower()}"
                ),
                points=ml_points,
                severity=prediction.level.status_token,
            )
        )
    else:
        factors.append(
            RiskFactor(
                source="model",
                description=f"Risk model unavailable ({prediction.model_version}){redistributed}",
                points=0.0,
                severity="muted",
            )
        )

    # -- NEWS2, decomposed per parameter -----------------------------------------------
    news2_budget = weights["news2"] * 100.0
    if news2 is not None:
        points_per_unit = news2_budget / float(news2.max_total)
        for component in news2.components:
            if component.score <= 0:
                continue
            points = component.score * points_per_unit
            composite += points
            value_text = (
                f"{component.value:.1f}" if isinstance(component.value, float) else component.value
            )
            factors.append(
                RiskFactor(
                    source="news2",
                    description=(
                        f"{component.display_name} {value_text}{component.unit} "
                        f"[{component.band}] → NEWS2 {component.score}"
                    ),
                    points=points,
                    severity="critical" if component.is_red else "warning",
                )
            )
        if news2.total == 0:
            factors.append(
                RiskFactor(
                    source="news2",
                    description="NEWS2 total 0 - all parameters within range",
                    points=0.0,
                    severity="good",
                )
            )
        if news2.missing_parameters:
            factors.append(
                RiskFactor(
                    source="news2",
                    description="Not measured: " + ", ".join(news2.missing_parameters),
                    points=0.0,
                    severity="muted",
                )
            )

    # -- Vision ------------------------------------------------------------------------
    vision_budget = weights["vision"] * 100.0
    severity, vision_key, vision_text = vision_severity(vision)
    if vision is not None and vision.available:
        vision_points = severity * vision_budget
        composite += vision_points
        factors.append(
            RiskFactor(
                source="vision",
                description=vision_text,
                points=vision_points,
                severity=(
                    "critical"
                    if vision_key == "fall"
                    else "serious"
                    if vision_key in {"bed_exit", "absent"}
                    else "warning"
                    if vision_key == "agitation"
                    else "good"
                ),
            )
        )
    else:
        factors.append(
            RiskFactor(
                source="vision",
                description=f"{vision_text}{redistributed}",
                points=0.0,
                severity="muted",
            )
        )

    composite = max(0.0, min(100.0, composite))
    level = band_for_score(composite, cfg)

    # -- Hard overrides ----------------------------------------------------------------
    # Each override does two independent things: it raises the *level* (never lowers it),
    # and it may mandate a *floor* under the composite. Only the highest floor binds, so
    # the reasons stay additive while the arithmetic stays single-valued.
    overrides: list[str] = []
    floors: list[tuple[float, str]] = []

    if vision_key == "fall":
        level = level.escalate_to(RiskLevel.CRITICAL)
        reason = "suspected fall → CRITICAL"
        floors.append((cfg.composite_critical_threshold, reason))
        overrides.append(reason)

    if news2 is not None:
        if news2.total >= cfg.news2_high_threshold:
            level = level.escalate_to(RiskLevel.HIGH)
            reason = f"NEWS2 {news2.total} ≥ {cfg.news2_high_threshold} → HIGH"
            floors.append(
                (
                    _graded_floor(
                        band_floor=cfg.composite_high_threshold,
                        band_ceiling=cfg.composite_critical_threshold,
                        value=float(news2.total),
                        low=float(cfg.news2_high_threshold),
                        high=float(news2.max_total),
                    ),
                    reason,
                )
            )
            overrides.append(reason)
        elif news2.total >= cfg.news2_medium_threshold:
            level = level.escalate_to(RiskLevel.MEDIUM)
            reason = f"NEWS2 {news2.total} ≥ {cfg.news2_medium_threshold} → MEDIUM"
            floors.append(
                (
                    _graded_floor(
                        band_floor=cfg.composite_medium_threshold,
                        band_ceiling=cfg.composite_high_threshold,
                        value=float(news2.total),
                        low=float(cfg.news2_medium_threshold),
                        high=float(cfg.news2_high_threshold),
                    ),
                    reason,
                )
            )
            overrides.append(reason)
        if news2.has_red_score:
            # A red score speaks to the *level* only: one parameter at 3 warrants review
            # even when the total is low, but it does not by itself set a magnitude.
            level = level.escalate_to(RiskLevel.MEDIUM)
            overrides.append("NEWS2 red score (single parameter = 3) → MEDIUM")

    for reason, floor_level in _extreme_value_overrides(vitals):
        level = level.escalate_to(floor_level)
        text = f"{reason} → {floor_level.value}"
        floors.append((cfg.composite_high_threshold, text))
        overrides.append(text)

    binding = max(floors, key=lambda item: item[0]) if floors else None
    if binding is not None and binding[0] > composite:
        composite = min(100.0, binding[0])

    # -- No usable data at all ---------------------------------------------------------
    if vitals.measured_channels == 0:
        if vision is not None and vision.available and not vision.patient_present:
            level = RiskLevel.UNKNOWN
            overrides.append("no vitals and no patient visible → UNKNOWN")
        else:
            level = RiskLevel.UNKNOWN
            overrides.append("no vitals measured → UNKNOWN")
        composite = 0.0

    # Overrides can raise the composite above the weighted sum. Record the lift as
    # its own factor so the itemised list still reconciles with the final score.
    weighted_total = sum(factor.points for factor in factors)
    lift = composite - weighted_total
    if lift > 0.05:
        factors.append(
            RiskFactor(
                source="override",
                description=(
                    f"Clinical override: {binding[1]}"
                    if binding is not None
                    else "Clinical override raised the score to its mandated floor"
                ),
                points=lift,
                severity="critical",
            )
        )

    probabilities = dict(prediction.probabilities) or dict.fromkeys(ML_RISK_CLASSES, 0.0)

    return RiskAssessment(
        patient_id=patient_id,
        level=level,
        composite_score=composite,
        ml_level=prediction.level,
        ml_confidence=prediction.confidence,
        ml_probabilities=probabilities,
        news2=news2,
        vision=vision,
        factors=tuple(factors),
        overrides=tuple(overrides),
        model_available=prediction.available,
        assessed_at=vitals.recorded_at,
    )


__all__ = [
    "AGITATION_THRESHOLD",
    "VISION_SEVERITY",
    "MLPrediction",
    "band_for_score",
    "fuse_risk",
    "vision_severity",
]
