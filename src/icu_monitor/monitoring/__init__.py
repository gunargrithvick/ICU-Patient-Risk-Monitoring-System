"""Live monitoring: the per-tick orchestrator and the alert ledger.

:class:`~icu_monitor.monitoring.engine.MonitoringEngine` is the single place a tick
happens, and every surface - dashboard, API, tests - consumes the same
:class:`~icu_monitor.monitoring.engine.WardSnapshot`. Alerting is separated into
:mod:`~icu_monitor.monitoring.alerts` because de-duplication and cooldown are the whole
difference between a usable channel and alarm fatigue.
"""

from __future__ import annotations

from icu_monitor.monitoring.alerts import (
    PHYSIOLOGY_RULES,
    RULES,
    SAFETY_RULES,
    SENSOR_SILENCE_SECONDS,
    SYSTEM_RULES,
    THRESHOLDS,
    AlertManager,
    AlertRule,
    RuleContext,
)
from icu_monitor.monitoring.engine import (
    MonitoringEngine,
    Recorder,
    WardSnapshot,
    build_engine,
)

__all__ = [
    "PHYSIOLOGY_RULES",
    "RULES",
    "SAFETY_RULES",
    "SENSOR_SILENCE_SECONDS",
    "SYSTEM_RULES",
    "THRESHOLDS",
    "AlertManager",
    "AlertRule",
    "MonitoringEngine",
    "Recorder",
    "RuleContext",
    "WardSnapshot",
    "build_engine",
]
