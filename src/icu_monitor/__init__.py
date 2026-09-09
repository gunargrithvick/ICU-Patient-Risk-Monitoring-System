"""ICU patient deterioration monitoring system.

A layered application:

``icu_monitor.core``
    Domain types, the NEWS2 clinical early-warning score, and the risk fusion
    engine. Pure Python, no I/O, no heavy dependencies.
``icu_monitor.data``
    PhysioNet Challenge 2012 extraction and acuity labelling.
``icu_monitor.ml``
    Feature engineering, model training, evaluation, and the model registry.
``icu_monitor.simulation``
    Physiologically-shaped bedside monitor simulation for demo/offline use.
``icu_monitor.vision``
    Pluggable frame sources and patient detectors (YOLO with graceful fallback).
``icu_monitor.monitoring``
    The tick engine that binds the ward together, plus alert rules.
``icu_monitor.storage``
    SQLite persistence for readings, assessments, and alerts.
``icu_monitor.api``
    FastAPI service exposing scoring, patients, and alerts.
``icu_monitor.ui``
    Streamlit operator dashboard.
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
