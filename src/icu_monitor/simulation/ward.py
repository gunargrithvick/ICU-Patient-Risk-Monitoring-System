"""The ward: many patients, one clock, two interchangeable data sources.

Everything above this layer (the monitoring engine, the API, the dashboard) talks to
a :class:`VitalsProvider`. Two implementations ship:

``SimulatedWard``
    Synthetic patients driven by :class:`~icu_monitor.simulation.patient.PatientSimulator`.
    Trajectories and events are controllable from the UI, which is what makes the
    alerting logic demonstrable on demand.

``ReplayWard``
    Real PhysioNet stays streamed from ``replay_series.csv`` at wall-clock speed.
    Values, gaps, and artefacts are whatever the ICU actually recorded - a much
    harsher and more honest test of the pipeline than any generator.

Selecting between them is configuration (``ICU_VITALS_SOURCE``), not a code change,
and :func:`build_provider` degrades to the simulator with a warning if the replay
file has not been built yet. That is the difference between a deployable app and a
demo that only runs on the author's machine.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.constants import DEMO_DIAGNOSES, DEMO_NAMES
from icu_monitor.core.types import (
    ClinicalState,
    Consciousness,
    Patient,
    Vitals,
    utcnow,
)
from icu_monitor.simulation.patient import EVENT_LIBRARY, PatientSimulator

logger = logging.getLogger(__name__)

#: Trajectory mix for a freshly generated ward - mostly stable, as a real unit is.
DEFAULT_STATE_MIX: tuple[tuple[ClinicalState, float], ...] = (
    (ClinicalState.STABLE, 0.42),
    (ClinicalState.RECOVERING, 0.20),
    (ClinicalState.DETERIORATING, 0.26),
    (ClinicalState.CRITICAL, 0.12),
)

#: A replayed value older than this is treated as no longer current.
REPLAY_STALENESS_HOURS = 3.0


@runtime_checkable
class VitalsProvider(Protocol):
    """Anything that can supply the ward's patients and their next observations.

    The control surface below is part of the contract even though a replay of recorded
    data cannot honour it: ``ReplayWard`` returns ``False``/``None`` rather than raising,
    so the dashboard can offer the same controls everywhere and simply show them as
    unavailable. A caller never has to ask which provider it is talking to.
    """

    @property
    def patients(self) -> dict[str, Patient]: ...

    @property
    def source_label(self) -> str: ...

    @property
    def available_events(self) -> dict[str, str]:
        """Injectable clinical events as ``{slug: label}``; empty when unsupported."""
        ...

    def advance(self) -> dict[str, Vitals]:
        """Produce the next observation for every patient."""
        ...

    def set_state(self, patient_id: str, state: ClinicalState | str) -> bool: ...

    def set_oxygen(self, patient_id: str, on_oxygen: bool) -> bool: ...

    def inject(self, patient_id: str, slug: str) -> str | None: ...

    def active_events(self, patient_id: str) -> tuple[str, ...]: ...


# --------------------------------------------------------------------------------------
# Bed helpers
# --------------------------------------------------------------------------------------


def _bed_label(index: int) -> str:
    return f"BED-{index + 1:02d}"


def _patient_id(index: int) -> str:
    return f"P{index + 1:03d}"


@dataclass(slots=True)
class _Occupant:
    patient: Patient
    simulator: PatientSimulator


# --------------------------------------------------------------------------------------
# Simulated ward
# --------------------------------------------------------------------------------------


class SimulatedWard:
    """A reproducible ward of synthetic patients."""

    def __init__(self, *, config: Settings | None = None, seed: int | None = None) -> None:
        self._config = config or default_settings
        self._rng = np.random.default_rng(self._config.simulation_seed if seed is None else seed)
        self._occupants: dict[str, _Occupant] = {}
        for index in range(self._config.bed_count):
            occupant = self._make_occupant(index)
            self._occupants[occupant.patient.patient_id] = occupant

    # -- construction ------------------------------------------------------------------

    def _make_occupant(self, index: int) -> _Occupant:
        states = [state for state, _ in DEFAULT_STATE_MIX]
        weights = [weight for _, weight in DEFAULT_STATE_MIX]
        # Bed 1 is always stable and bed 2 always deteriorating, so a reviewer opening
        # the app sees both a normal and an abnormal patient immediately.
        if index == 0:
            state = ClinicalState.STABLE
        elif index == 1:
            state = ClinicalState.DETERIORATING
        else:
            state = ClinicalState(str(self._rng.choice([s.value for s in states], p=weights)))

        age = int(self._rng.integers(24, 91))
        on_oxygen = state in {ClinicalState.DETERIORATING, ClinicalState.CRITICAL} or bool(
            self._rng.random() < 0.25
        )
        name = DEMO_NAMES[index % len(DEMO_NAMES)]
        diagnosis = DEMO_DIAGNOSES[int(self._rng.integers(0, len(DEMO_DIAGNOSES)))]

        patient = Patient(
            patient_id=_patient_id(index),
            bed=_bed_label(index),
            display_name=name,
            age=age,
            sex="M" if self._rng.random() < 0.52 else "F",
            admitted_at=utcnow() - timedelta(hours=float(self._rng.uniform(3, 96))),
            primary_diagnosis=diagnosis,
            state=state,
            on_supplemental_oxygen=on_oxygen,
            # Scale 2 applies to a minority with chronic hypercapnic risk.
            spo2_scale=2 if self._rng.random() < 0.12 else 1,
            notes="Synthetic patient - not a real person.",
        )
        simulator = PatientSimulator(
            seed=int(self._rng.integers(0, 2**31 - 1)),
            state=state,
            age=age,
            on_supplemental_oxygen=on_oxygen,
        )
        # Warm up so patients do not all start from an identical baseline.
        for _ in range(int(self._rng.integers(4, 40))):
            simulator.step()
        return _Occupant(patient=patient, simulator=simulator)

    # -- provider protocol -------------------------------------------------------------

    @property
    def patients(self) -> dict[str, Patient]:
        return {pid: occupant.patient for pid, occupant in self._occupants.items()}

    @property
    def source_label(self) -> str:
        return "Simulated ward (synthetic physiology)"

    def advance(self) -> dict[str, Vitals]:
        readings: dict[str, Vitals] = {}
        for patient_id, occupant in self._occupants.items():
            readings[patient_id] = occupant.simulator.step()
        return readings

    # -- controls used by the dashboard ------------------------------------------------

    def set_state(self, patient_id: str, state: ClinicalState | str) -> bool:
        occupant = self._occupants.get(patient_id)
        if occupant is None:
            return False
        resolved = ClinicalState(state) if not isinstance(state, ClinicalState) else state
        occupant.simulator.set_state(resolved)
        occupant.patient.state = resolved
        return True

    def set_oxygen(self, patient_id: str, on_oxygen: bool) -> bool:
        occupant = self._occupants.get(patient_id)
        if occupant is None:
            return False
        occupant.patient.on_supplemental_oxygen = bool(on_oxygen)
        occupant.simulator.on_supplemental_oxygen = bool(on_oxygen)
        return True

    def inject(self, patient_id: str, slug: str) -> str | None:
        occupant = self._occupants.get(patient_id)
        if occupant is None:
            return None
        return occupant.simulator.inject(slug)

    def active_events(self, patient_id: str) -> tuple[str, ...]:
        occupant = self._occupants.get(patient_id)
        return occupant.simulator.active_events if occupant else ()

    @property
    def available_events(self) -> dict[str, str]:
        return {slug: entry[0] for slug, entry in EVENT_LIBRARY.items()}


# --------------------------------------------------------------------------------------
# Replay ward
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _ReplayStay:
    """One real stay, resampled onto an hourly grid of observations."""

    record_id: int
    patient: Patient
    frames: list[dict[str, float | None]] = field(default_factory=list)
    cursor: int = 0

    def next_frame(self) -> dict[str, float | None]:
        if not self.frames:
            return {}
        frame = self.frames[self.cursor % len(self.frames)]
        self.cursor += 1
        return frame


def _resample_stay(
    group: pd.DataFrame, *, step_hours: float = 1.0
) -> list[dict[str, float | None]]:
    """Turn long-format samples into a list of per-timestep channel readings.

    A channel reports its most recent value while that value is still fresh
    (:data:`REPLAY_STALENESS_HOURS`) and ``None`` afterwards, so genuine ICU gaps
    survive into the replay instead of being smoothed away.
    """
    if group.empty:
        return []
    channels = {
        str(channel): sub.sort_values("hours")[["hours", "value"]].to_numpy(dtype=float)
        for channel, sub in group.groupby("channel")
    }
    horizon = float(group["hours"].max())
    frames: list[dict[str, float | None]] = []
    grid = np.arange(0.0, horizon + step_hours, step_hours)
    for now in grid:
        frame: dict[str, float | None] = {}
        for channel, samples in channels.items():
            times = samples[:, 0]
            position = int(np.searchsorted(times, now, side="right")) - 1
            if position < 0:
                frame[channel] = None
                continue
            age = now - float(times[position])
            frame[channel] = float(samples[position, 1]) if age <= REPLAY_STALENESS_HOURS else None
        frames.append(frame)
    return frames


class ReplayWard:
    """Streams recorded PhysioNet stays as if they were live."""

    def __init__(self, *, config: Settings | None = None) -> None:
        self._config = config or default_settings
        path = self._config.replay_path
        if not path.exists():
            raise FileNotFoundError(
                f"Replay series not found at {path}. Run the ETL "
                "(`python -m icu_monitor etl`) or set ICU_VITALS_SOURCE=simulator."
            )
        series = pd.read_csv(path)
        cohort = self._load_cohort()

        self._stays: dict[str, _ReplayStay] = {}
        record_ids = list(dict.fromkeys(series["record_id"].tolist()))[: self._config.bed_count]
        for index, record_id in enumerate(record_ids):
            group = series[series["record_id"] == record_id]
            static = cohort.get(int(record_id), {})
            patient = self._make_patient(index, int(record_id), group, static)
            stay = _ReplayStay(
                record_id=int(record_id),
                patient=patient,
                frames=_resample_stay(group),
            )
            if stay.frames:
                self._stays[patient.patient_id] = stay

        if not self._stays:
            raise RuntimeError(f"{path} contained no usable stays.")

    def _load_cohort(self) -> dict[int, dict[str, float]]:
        path = self._config.cohort_path
        if not path.exists():
            return {}
        frame = pd.read_csv(path)
        return {
            int(row["record_id"]): {
                key: float(row[key])
                for key in ("age", "sex_male", "icu_type")
                if key in row and pd.notna(row[key])
            }
            for _, row in frame.iterrows()
        }

    def _make_patient(
        self,
        index: int,
        record_id: int,
        group: pd.DataFrame,
        static: dict[str, float],
    ) -> Patient:
        acuity = str(group["acuity"].iloc[0]) if "acuity" in group.columns else "unknown"
        age = int(static.get("age", 65) or 65)
        sex = "M" if static.get("sex_male", 1.0) >= 0.5 else "F"
        return Patient(
            patient_id=f"R{record_id}",
            bed=_bed_label(index),
            display_name=f"Record {record_id}",
            age=age,
            sex=sex,
            admitted_at=utcnow() - timedelta(hours=float(group["hours"].max())),
            primary_diagnosis=f"PhysioNet 2012 stay · stay acuity {acuity}",
            state=ClinicalState.STABLE,
            notes=(
                "De-identified PhysioNet/CinC Challenge 2012 record replayed at "
                "wall-clock speed. Stay-level acuity shown for reference only."
            ),
        )

    # -- provider protocol -------------------------------------------------------------

    @property
    def patients(self) -> dict[str, Patient]:
        return {pid: stay.patient for pid, stay in self._stays.items()}

    @property
    def source_label(self) -> str:
        return f"Replay of {len(self._stays)} PhysioNet 2012 stays"

    def advance(self) -> dict[str, Vitals]:
        readings: dict[str, Vitals] = {}
        now = utcnow()
        for patient_id, stay in self._stays.items():
            frame = stay.next_frame()
            gcs = frame.get("gcs")
            readings[patient_id] = Vitals(
                heart_rate=frame.get("heart_rate"),
                spo2=frame.get("spo2"),
                bp_systolic=frame.get("bp_systolic"),
                bp_diastolic=frame.get("bp_diastolic"),
                resp_rate=frame.get("resp_rate"),
                temperature=frame.get("temperature"),
                consciousness=(
                    Consciousness.from_gcs(gcs) if gcs is not None else Consciousness.ALERT
                ),
                on_supplemental_oxygen=bool(frame.get("fio2") and frame["fio2"] > 0.21),
                gcs=gcs,
                recorded_at=now,
            )
        return readings

    # -- controls (replay is read-only; keep the interface uniform) ---------------------

    def set_state(self, patient_id: str, state: ClinicalState | str) -> bool:
        return False

    def set_oxygen(self, patient_id: str, on_oxygen: bool) -> bool:
        return False

    def inject(self, patient_id: str, slug: str) -> str | None:
        return None

    def active_events(self, patient_id: str) -> tuple[str, ...]:
        return ()

    @property
    def available_events(self) -> dict[str, str]:
        return {}


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_provider(config: Settings | None = None) -> VitalsProvider:
    """Build the configured provider, falling back to the simulator if replay is absent."""
    cfg = config or default_settings
    if cfg.vitals_source == "replay":
        try:
            return ReplayWard(config=cfg)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Replay source unavailable (%s); using the simulator.", exc)
    return SimulatedWard(config=cfg)


def state_mix(patients: Iterable[Patient]) -> dict[str, int]:
    """Count patients per trajectory - used by the ward header."""
    counts: dict[str, int] = {state.value: 0 for state in ClinicalState}
    for patient in patients:
        counts[patient.state.value] = counts.get(patient.state.value, 0) + 1
    return counts


__all__ = [
    "DEFAULT_STATE_MIX",
    "REPLAY_STALENESS_HOURS",
    "ReplayWard",
    "SimulatedWard",
    "VitalsProvider",
    "build_provider",
    "state_mix",
]
