"""Synthetic and replayed patient data sources.

The dashboard must be demonstrable with no ward, no monitor, and no camera, so the
data layer is pluggable: :class:`~icu_monitor.simulation.ward.SimulatedWard` generates
physiology, :class:`~icu_monitor.simulation.ward.ReplayWard` streams real recorded
stays, and both satisfy the same
:class:`~icu_monitor.simulation.ward.VitalsProvider` protocol.
"""

from __future__ import annotations

from icu_monitor.simulation.patient import (
    EVENT_LIBRARY,
    PatientSimulator,
    SimulationEvent,
)
from icu_monitor.simulation.ward import (
    ReplayWard,
    SimulatedWard,
    VitalsProvider,
    build_provider,
    state_mix,
)

__all__ = [
    "EVENT_LIBRARY",
    "PatientSimulator",
    "ReplayWard",
    "SimulatedWard",
    "SimulationEvent",
    "VitalsProvider",
    "build_provider",
    "state_mix",
]
