"""Feature engineering shared by training and inference.

Train/serve skew is the classic way a working notebook becomes a broken product,
so both paths call the *same* functions here:

* the ETL (:mod:`icu_monitor.data.physionet`) slices a PhysioNet stay into
  overlapping windows and summarises each one;
* the live engine (:mod:`icu_monitor.monitoring.engine`) summarises the rolling
  buffer of bedside observations for the same window length.

A window is described by six aggregates per channel - ``last``, ``mean``, ``min``,
``max``, ``std``, ``slope`` - because deterioration shows up as *trend and
variability*, not just a snapshot. ``slope`` is a least-squares fit in units per
hour, which is what a clinician reads off a chart.

Missing channels stay ``NaN``. The estimator
(:class:`~sklearn.ensemble.HistGradientBoostingClassifier`) handles missing values
natively, so nothing is imputed with a fake "normal" value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from icu_monitor.core.types import Patient, Vitals

#: Physiological channels summarised for the model, in a fixed order.
FEATURE_CHANNELS: tuple[str, ...] = (
    "heart_rate",
    "spo2",
    "bp_systolic",
    "bp_diastolic",
    "resp_rate",
    "temperature",
    "gcs",
)

#: Window aggregates computed per channel, in a fixed order.
AGGREGATES: tuple[str, ...] = ("last", "mean", "min", "max", "std", "slope")

#: Stay-level descriptors that do not vary within a window.
STATIC_FEATURES: tuple[str, ...] = (
    "age",
    "sex_male",
    "icu_type",
    "weight_kg",
    "height_cm",
    "bmi",
)

#: Derived cross-channel features that carry real clinical signal.
DERIVED_FEATURES: tuple[str, ...] = ("shock_index", "pulse_pressure", "map_estimate")

#: The full design matrix column order. Never reorder in place - retrain instead.
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"{channel}_{aggregate}" for channel in FEATURE_CHANNELS for aggregate in AGGREGATES)
    + STATIC_FEATURES
    + DERIVED_FEATURES
)

NAN = float("nan")


# --------------------------------------------------------------------------------------
# Channel summarisation
# --------------------------------------------------------------------------------------


def summarise_channel(hours: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Summarise one channel's samples inside a window.

    Args:
        hours: Sample times in hours, relative to anything (only spacing matters).
        values: Sample values, same length as ``hours``. May contain NaN.

    Returns:
        Mapping of aggregate name to value; ``NaN`` where undefined.
    """
    if len(values) == 0:
        return dict.fromkeys(AGGREGATES, NAN)

    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(hours)
    if not mask.any():
        return dict.fromkeys(AGGREGATES, NAN)

    hours, values = hours[mask], values[mask]
    order = np.argsort(hours, kind="stable")
    hours, values = hours[order], values[order]

    summary = {
        "last": float(values[-1]),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        # Population std of a single sample is 0, which is truthful: no variability
        # was observed. NaN would wrongly imply "unknown".
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "slope": _slope_per_hour(hours, values),
    }
    return summary


def _slope_per_hour(hours: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope in value-units per hour; ``NaN`` if under-determined."""
    if len(values) < 2:
        return NAN
    span = float(hours[-1] - hours[0])
    if span <= 0:
        return NAN
    centred = hours - hours.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0:
        return NAN
    return float((centred * (values - values.mean())).sum() / denominator)


# --------------------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------------------


def _finite(value: Any) -> float:
    """Coerce to float, mapping ``None``/non-numeric/inf to ``NaN``."""
    if value is None:
        return NAN
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return NAN
    return numeric if math.isfinite(numeric) else NAN


def _bmi(weight_kg: float, height_cm: float) -> float:
    if not math.isfinite(weight_kg) or not math.isfinite(height_cm) or height_cm <= 0:
        return NAN
    metres = height_cm / 100.0
    value = weight_kg / (metres * metres)
    # Reject implausible values that arise from unit errors in the source data.
    return value if 8.0 <= value <= 90.0 else NAN


def _derived(summaries: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Cross-channel features computed from the window's last observed values."""
    hr = summaries.get("heart_rate", {}).get("last", NAN)
    sbp = summaries.get("bp_systolic", {}).get("last", NAN)
    dbp = summaries.get("bp_diastolic", {}).get("last", NAN)

    shock_index = hr / sbp if math.isfinite(hr) and math.isfinite(sbp) and sbp > 0 else NAN
    pulse_pressure = sbp - dbp if math.isfinite(sbp) and math.isfinite(dbp) else NAN
    map_estimate = (sbp + 2 * dbp) / 3 if math.isfinite(sbp) and math.isfinite(dbp) else NAN
    return {
        "shock_index": shock_index,
        "pulse_pressure": pulse_pressure,
        "map_estimate": map_estimate,
    }


def build_feature_row(
    channels: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    static: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build one row of the design matrix.

    Args:
        channels: ``{channel_key: (hours, values)}``. Channels absent from the
            mapping are reported as missing rather than zero.
        static: Stay-level descriptors (``age``, ``sex_male``, ``icu_type``,
            ``weight_kg``, ``height_cm``). ``bmi`` is derived if absent.

    Returns:
        A dict keyed by every name in :data:`FEATURE_NAMES`.
    """
    static = dict(static or {})
    row: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}

    for channel in FEATURE_CHANNELS:
        hours, values = channels.get(channel, ((), ()))
        summary = summarise_channel(np.asarray(hours, dtype=float), np.asarray(values, dtype=float))
        summaries[channel] = summary
        for aggregate in AGGREGATES:
            row[f"{channel}_{aggregate}"] = summary[aggregate]

    weight = _finite(static.get("weight_kg"))
    height = _finite(static.get("height_cm"))
    row["age"] = _finite(static.get("age"))
    row["sex_male"] = _finite(static.get("sex_male"))
    row["icu_type"] = _finite(static.get("icu_type"))
    row["weight_kg"] = weight
    row["height_cm"] = height
    supplied_bmi = _finite(static.get("bmi"))
    row["bmi"] = supplied_bmi if math.isfinite(supplied_bmi) else _bmi(weight, height)

    row.update(_derived(summaries))
    return row


def features_to_frame(rows: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    """Stack feature dicts into a DataFrame with the canonical column order."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=list(FEATURE_NAMES), dtype=float)
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            frame[name] = NAN
    return frame.loc[:, list(FEATURE_NAMES)].astype(float)


# --------------------------------------------------------------------------------------
# Live inference path
# --------------------------------------------------------------------------------------


def channels_from_history(
    history: Sequence[Vitals],
    *,
    window_hours: float | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """Convert a rolling buffer of bedside observations into channel arrays.

    Args:
        history: Observations in chronological order (oldest first).
        window_hours: Keep only samples within this many hours of the newest one.
            ``None`` keeps everything supplied.
    """
    channels: dict[str, tuple[list[float], list[float]]] = {
        channel: ([], []) for channel in FEATURE_CHANNELS
    }
    if not history:
        return channels

    newest = history[-1].recorded_at
    for observation in history:
        delta_hours = (observation.recorded_at - newest).total_seconds() / 3600.0
        if window_hours is not None and abs(delta_hours) > window_hours:
            continue
        readings = {
            "heart_rate": observation.heart_rate,
            "spo2": observation.spo2,
            "bp_systolic": observation.bp_systolic,
            "bp_diastolic": observation.bp_diastolic,
            "resp_rate": observation.resp_rate,
            "temperature": observation.temperature,
            "gcs": observation.gcs,
        }
        for channel, value in readings.items():
            numeric = _finite(value)
            if math.isfinite(numeric):
                hours, values = channels[channel]
                hours.append(delta_hours)
                values.append(numeric)
    return channels


def static_from_patient(
    patient: Patient,
    *,
    weight_kg: float | None = None,
    height_cm: float | None = None,
    icu_type: int = 3,
) -> dict[str, float]:
    """Stay-level descriptors for a live patient."""
    return {
        "age": float(patient.age),
        "sex_male": 1.0 if patient.sex.upper().startswith("M") else 0.0,
        "icu_type": float(icu_type),
        "weight_kg": _finite(weight_kg),
        "height_cm": _finite(height_cm),
    }


def features_for_patient(
    patient: Patient,
    history: Sequence[Vitals],
    *,
    window_hours: float | None = None,
    weight_kg: float | None = None,
    height_cm: float | None = None,
    icu_type: int = 3,
) -> pd.DataFrame:
    """One-row design matrix for a live patient, ready for ``model.predict``."""
    channels = channels_from_history(history, window_hours=window_hours)
    static = static_from_patient(
        patient, weight_kg=weight_kg, height_cm=height_cm, icu_type=icu_type
    )
    return features_to_frame([build_feature_row(channels, static)])


__all__ = [
    "AGGREGATES",
    "DERIVED_FEATURES",
    "FEATURE_CHANNELS",
    "FEATURE_NAMES",
    "STATIC_FEATURES",
    "build_feature_row",
    "channels_from_history",
    "features_for_patient",
    "features_to_frame",
    "static_from_patient",
    "summarise_channel",
]
