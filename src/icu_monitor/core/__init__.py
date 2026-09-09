"""Domain core: types, clinical scoring, and risk fusion. No I/O, no heavy deps."""

from __future__ import annotations

from icu_monitor.core.constants import (
    CORE_VITALS,
    TREND_VITALS,
    VITAL_SPECS,
    VitalSpec,
)
from icu_monitor.core.fusion import MLPrediction, band_for_score, fuse_risk, vision_severity
from icu_monitor.core.news2 import calculate_news2, news2_band_table
from icu_monitor.core.types import (
    ML_RISK_CLASSES,
    Alert,
    AlertKind,
    BedSnapshot,
    ClinicalState,
    Consciousness,
    Detection,
    NEWS2Result,
    ParameterScore,
    Patient,
    Posture,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    VisionSignal,
    Vitals,
    utcnow,
)

__all__ = [
    "CORE_VITALS",
    "ML_RISK_CLASSES",
    "TREND_VITALS",
    "VITAL_SPECS",
    "Alert",
    "AlertKind",
    "BedSnapshot",
    "ClinicalState",
    "Consciousness",
    "Detection",
    "MLPrediction",
    "NEWS2Result",
    "ParameterScore",
    "Patient",
    "Posture",
    "RiskAssessment",
    "RiskFactor",
    "RiskLevel",
    "VisionSignal",
    "VitalSpec",
    "Vitals",
    "band_for_score",
    "calculate_news2",
    "fuse_risk",
    "news2_band_table",
    "utcnow",
    "vision_severity",
]
