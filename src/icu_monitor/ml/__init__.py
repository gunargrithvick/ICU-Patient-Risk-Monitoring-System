"""Machine learning: shared features, training, evaluation, and the model registry.

The layering matters. :mod:`~icu_monitor.ml.features` is imported by both the ETL and
the live dashboard, which is what keeps training and serving from drifting apart;
:mod:`~icu_monitor.ml.registry` is the only module that reads or writes the artefact,
so the "no trained model yet" case is handled in exactly one place.
"""

from __future__ import annotations

from icu_monitor.ml.features import (
    AGGREGATES,
    FEATURE_CHANNELS,
    FEATURE_NAMES,
    build_feature_row,
    features_for_patient,
    features_to_frame,
    summarise_channel,
)
from icu_monitor.ml.registry import (
    ModelMetadata,
    RiskModel,
    build_model_card,
    clear_cache,
    load_metrics,
    load_model,
    load_model_card,
    save_model,
)

__all__ = [
    "AGGREGATES",
    "FEATURE_CHANNELS",
    "FEATURE_NAMES",
    "ModelMetadata",
    "RiskModel",
    "build_feature_row",
    "build_model_card",
    "clear_cache",
    "features_for_patient",
    "features_to_frame",
    "load_metrics",
    "load_model",
    "load_model_card",
    "save_model",
    "summarise_channel",
]
