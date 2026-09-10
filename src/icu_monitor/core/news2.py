"""National Early Warning Score 2 (NEWS2).

NEWS2 is the aggregate physiological scoring system recommended by the Royal
College of Physicians for detecting clinical deterioration in acutely ill adults.
Each of seven parameters scores 0-3; the total drives a graded clinical response.

This module implements the published scoring tables, including both SpO₂ scales
(Scale 2 applies to patients with a target saturation of 88-92%, typically
hypercapnic respiratory failure), and returns a fully itemised result so the UI
can explain *why* a score is what it is rather than just showing a number.

The published tables are defined on the precision a clinician actually records:
whole numbers for rates and pressures, one decimal for temperature. Continuous
sensor values are therefore rounded to that precision before lookup, which makes
the bands exhaustive - no value can fall into a gap between rows.

Reference: Royal College of Physicians, *National Early Warning Score (NEWS) 2*,
2017 update. Thresholds for the graded response follow the same document.

.. warning::
   This is an educational implementation. It is not a medical device and must not
   be used to make clinical decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from icu_monitor.core.types import (
    Consciousness,
    NEWS2Result,
    ParameterScore,
    RiskLevel,
    Vitals,
)

# A band is (inclusive_lower, inclusive_upper, score, human_label).
Band = tuple[float, float, int, str]

NEG_INF = float("-inf")
POS_INF = float("inf")

#: Recording precision (decimal places) per parameter, applied before band lookup.
RECORDING_PRECISION: dict[str, int] = {
    "resp_rate": 0,
    "spo2": 0,
    "bp_systolic": 0,
    "heart_rate": 0,
    "temperature": 1,
}


# --------------------------------------------------------------------------------------
# Published scoring tables
# --------------------------------------------------------------------------------------

RESP_RATE_BANDS: tuple[Band, ...] = (
    (NEG_INF, 8, 3, "≤8"),
    (9, 11, 1, "9-11"),
    (12, 20, 0, "12-20"),
    (21, 24, 2, "21-24"),
    (25, POS_INF, 3, "≥25"),
)

SPO2_SCALE1_BANDS: tuple[Band, ...] = (
    (NEG_INF, 91, 3, "≤91"),
    (92, 93, 2, "92-93"),
    (94, 95, 1, "94-95"),
    (96, POS_INF, 0, "≥96"),
)

# Scale 2 applies when the prescribed target range is 88-92%. Scoring differs
# depending on whether the patient is breathing air or receiving oxygen, so it is
# resolved by :func:`_score_spo2_scale2` rather than a flat band table.
SPO2_SCALE2_AIR_BANDS: tuple[Band, ...] = (
    (NEG_INF, 83, 3, "≤83"),
    (84, 85, 2, "84-85"),
    (86, 87, 1, "86-87"),
    (88, POS_INF, 0, "≥88 on air"),
)

SPO2_SCALE2_OXYGEN_BANDS: tuple[Band, ...] = (
    (NEG_INF, 83, 3, "≤83"),
    (84, 85, 2, "84-85"),
    (86, 87, 1, "86-87"),
    (88, 92, 0, "88-92 (target)"),
    (93, 94, 1, "93-94 on O₂"),
    (95, 96, 2, "95-96 on O₂"),
    (97, POS_INF, 3, "≥97 on O₂"),
)

SYSTOLIC_BANDS: tuple[Band, ...] = (
    (NEG_INF, 90, 3, "≤90"),
    (91, 100, 2, "91-100"),
    (101, 110, 1, "101-110"),
    (111, 219, 0, "111-219"),
    (220, POS_INF, 3, "≥220"),
)

PULSE_BANDS: tuple[Band, ...] = (
    (NEG_INF, 40, 3, "≤40"),
    (41, 50, 1, "41-50"),
    (51, 90, 0, "51-90"),
    (91, 110, 1, "91-110"),
    (111, 130, 2, "111-130"),
    (131, POS_INF, 3, "≥131"),
)

TEMPERATURE_BANDS: tuple[Band, ...] = (
    (NEG_INF, 35.0, 3, "≤35.0"),
    (35.1, 36.0, 1, "35.1-36.0"),
    (36.1, 38.0, 0, "36.1-38.0"),
    (38.1, 39.0, 1, "38.1-39.0"),
    (39.1, POS_INF, 2, "≥39.1"),
)

#: Graded clinical response by NEWS2 total (RCP 2017, Table 3).
RESPONSE_BY_TOTAL: tuple[tuple[int, float, str, str], ...] = (
    (0, 0, "Routine", "Routine monitoring, minimum 12-hourly observations."),
    (1, 4, "Ward-based", "Registered nurse review; minimum 4-6 hourly observations."),
    (
        5,
        6,
        "Urgent",
        "Urgent review by a clinician with acute-illness competencies; "
        "minimum hourly observations.",
    ),
    (
        7,
        POS_INF,
        "Emergency",
        "Emergency assessment by a critical-care-competent team; "
        "continuous monitoring; consider transfer to a higher level of care.",
    ),
)

#: Response when any single parameter scores 3 but the total is still low.
RED_SCORE_RESPONSE = (
    "Urgent ward-based review: a single parameter scoring 3 is a red score "
    "regardless of the aggregate."
)


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------


def _clean(value: float | None) -> float | None:
    """Drop ``None`` and NaN so they are reported as missing, not scored as zero."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def round_clinical(value: float, precision: int) -> float:
    """Round half **up**, the convention used when charting observations.

    Python's built-in :func:`round` and ``format`` use round-half-to-even, which
    would send a respiratory rate of 8.5 down to 8 (score 3) instead of up to 9
    (score 1). Charting rounds away from zero at the halfway point, so we do too.
    """
    quantum = Decimal(1).scaleb(-precision)
    try:
        return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))
    except InvalidOperation:  # pragma: no cover - defensive
        return float(value)


def score_in_bands(
    value: float | None,
    bands: Sequence[Band],
    *,
    precision: int = 0,
) -> tuple[int, str]:
    """Return ``(score, band_label)`` for ``value``, or ``(0, "not measured")``.

    ``value`` is rounded to ``precision`` decimal places first, so the published
    integer/one-decimal tables cover the whole real line without gaps.
    """
    numeric = _clean(value)
    if numeric is None:
        return 0, "not measured"
    rounded = round_clinical(numeric, precision)
    for lower, upper, score, label in bands:
        if lower <= rounded <= upper:
            return score, label
    return 0, "out of range"


def _score_spo2_scale2(value: float | None, on_oxygen: bool) -> tuple[int, str]:
    bands = SPO2_SCALE2_OXYGEN_BANDS if on_oxygen else SPO2_SCALE2_AIR_BANDS
    return score_in_bands(value, bands, precision=RECORDING_PRECISION["spo2"])


def _consciousness_score(level: Consciousness | None) -> tuple[int, str]:
    if level is None:
        return 0, "not measured"
    if level is Consciousness.ALERT:
        return 0, "Alert"
    return 3, level.label


def _response_for(total: int, has_red_score: bool) -> tuple[RiskLevel, str]:
    """Map a NEWS2 total onto our acuity ladder plus the RCP response text."""
    tier, response = "Routine", RESPONSE_BY_TOTAL[0][3]
    for lower, upper, name, text in RESPONSE_BY_TOTAL:
        if lower <= total <= upper:
            tier, response = name, text
            break

    if tier == "Emergency":
        return RiskLevel.HIGH, response
    if tier == "Urgent":
        return RiskLevel.MEDIUM, response
    # Totals of 0-4 normally sit at LOW, but a single parameter scoring 3 is a
    # "red score" that mandates urgent review on its own.
    if has_red_score:
        return RiskLevel.MEDIUM, RED_SCORE_RESPONSE
    return RiskLevel.LOW, response


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def calculate_news2(vitals: Vitals, *, spo2_scale: int = 1) -> NEWS2Result:
    """Score ``vitals`` with NEWS2.

    Args:
        vitals: The bedside observation. Missing channels are reported in
            :attr:`NEWS2Result.missing_parameters` and contribute zero, matching
            how the score is used at the bedside when a channel is unavailable.
        spo2_scale: ``1`` for the standard target of ≥94%, ``2`` when the
            prescribed target is 88-92%.

    Returns:
        A fully itemised :class:`NEWS2Result`.
    """
    scale = 2 if spo2_scale == 2 else 1

    resp_score, resp_band = score_in_bands(
        vitals.resp_rate, RESP_RATE_BANDS, precision=RECORDING_PRECISION["resp_rate"]
    )
    if scale == 2:
        spo2_score, spo2_band = _score_spo2_scale2(vitals.spo2, vitals.on_supplemental_oxygen)
    else:
        spo2_score, spo2_band = score_in_bands(
            vitals.spo2, SPO2_SCALE1_BANDS, precision=RECORDING_PRECISION["spo2"]
        )
    oxygen_score = 2 if vitals.on_supplemental_oxygen else 0
    sbp_score, sbp_band = score_in_bands(
        vitals.bp_systolic, SYSTOLIC_BANDS, precision=RECORDING_PRECISION["bp_systolic"]
    )
    pulse_score, pulse_band = score_in_bands(
        vitals.heart_rate, PULSE_BANDS, precision=RECORDING_PRECISION["heart_rate"]
    )
    acvpu_score, acvpu_band = _consciousness_score(vitals.consciousness)
    temp_score, temp_band = score_in_bands(
        vitals.temperature, TEMPERATURE_BANDS, precision=RECORDING_PRECISION["temperature"]
    )

    components = (
        ParameterScore(
            parameter="resp_rate",
            display_name="Respiration rate",
            value=_clean(vitals.resp_rate),
            unit="/min",
            score=resp_score,
            band=resp_band,
            is_red=resp_score == 3,
        ),
        ParameterScore(
            parameter="spo2",
            display_name=f"SpO₂ (scale {scale})",
            value=_clean(vitals.spo2),
            unit="%",
            score=spo2_score,
            band=spo2_band,
            is_red=spo2_score == 3,
        ),
        ParameterScore(
            parameter="supplemental_oxygen",
            display_name="Air or oxygen",
            value="Oxygen" if vitals.on_supplemental_oxygen else "Air",
            unit="",
            score=oxygen_score,
            band="Oxygen" if vitals.on_supplemental_oxygen else "Air",
            is_red=False,
        ),
        ParameterScore(
            parameter="bp_systolic",
            display_name="Systolic blood pressure",
            value=_clean(vitals.bp_systolic),
            unit="mmHg",
            score=sbp_score,
            band=sbp_band,
            is_red=sbp_score == 3,
        ),
        ParameterScore(
            parameter="heart_rate",
            display_name="Pulse",
            value=_clean(vitals.heart_rate),
            unit="bpm",
            score=pulse_score,
            band=pulse_band,
            is_red=pulse_score == 3,
        ),
        ParameterScore(
            parameter="consciousness",
            display_name="Consciousness (ACVPU)",
            value=vitals.consciousness.label if vitals.consciousness else None,
            unit="",
            score=acvpu_score,
            band=acvpu_band,
            is_red=acvpu_score == 3,
        ),
        ParameterScore(
            parameter="temperature",
            display_name="Temperature",
            value=_clean(vitals.temperature),
            unit="°C",
            score=temp_score,
            band=temp_band,
            is_red=temp_score == 3,
        ),
    )

    total = sum(component.score for component in components)
    has_red = any(component.is_red for component in components)

    missing = tuple(
        component.display_name
        for component in components
        if component.value is None and component.parameter != "supplemental_oxygen"
    )

    level, response = _response_for(total, has_red)

    return NEWS2Result(
        total=total,
        components=components,
        risk_level=level,
        clinical_response=response,
        has_red_score=has_red,
        missing_parameters=missing,
        scale=scale,
    )


def news2_band_table(parameter: str) -> tuple[Band, ...]:
    """Expose a scoring table for display on the model/reference page."""
    tables: dict[str, tuple[Band, ...]] = {
        "resp_rate": RESP_RATE_BANDS,
        "spo2": SPO2_SCALE1_BANDS,
        "spo2_scale2_air": SPO2_SCALE2_AIR_BANDS,
        "spo2_scale2_oxygen": SPO2_SCALE2_OXYGEN_BANDS,
        "bp_systolic": SYSTOLIC_BANDS,
        "heart_rate": PULSE_BANDS,
        "temperature": TEMPERATURE_BANDS,
    }
    return tables.get(parameter, ())


__all__ = [
    "PULSE_BANDS",
    "RECORDING_PRECISION",
    "RESPONSE_BY_TOTAL",
    "RESP_RATE_BANDS",
    "SPO2_SCALE1_BANDS",
    "SPO2_SCALE2_AIR_BANDS",
    "SPO2_SCALE2_OXYGEN_BANDS",
    "SYSTOLIC_BANDS",
    "TEMPERATURE_BANDS",
    "calculate_news2",
    "news2_band_table",
    "round_clinical",
    "score_in_bands",
]
