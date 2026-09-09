"""Physiologically-shaped vital sign simulation.

Why simulate at all? A monitoring dashboard has to be demonstrable without a real
ward, and a random-number generator makes a *bad* demo: uncorrelated noise never
produces the coupled, gradually-worsening pattern that early-warning scores exist
to detect. So each channel here is a mean-reverting process pulled toward a target
that depends on the patient's trajectory, with the couplings that matter clinically:

* falling saturation drags respiratory rate **up** (compensatory tachypnoea);
* falling blood pressure drags heart rate **up** (compensatory tachycardia);
* sustained deterioration depresses GCS, which drives the ACVPU term in NEWS2;
* diastolic pressure tracks systolic with a realistic pulse pressure.

The result is a patient whose NEWS2 climbs in a recognisable way rather than
flickering, which is what makes the alerting logic testable.

Nothing here is a physiological model in the scientific sense. It is a plausible
*shape*, seeded for reproducibility, and clearly labelled as synthetic everywhere it
surfaces in the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from icu_monitor.core.types import ClinicalState, Consciousness, Vitals, utcnow

#: Per-state channel targets and how hard the process is pulled toward them.
#: ``(target, reversion, noise)`` - reversion in [0, 1] per 5-minute step.
StateProfile = dict[str, tuple[float, float, float]]

BASELINE: StateProfile = {
    "heart_rate": (78.0, 0.06, 1.6),
    "spo2": (97.5, 0.10, 0.5),
    "bp_systolic": (122.0, 0.06, 2.4),
    "resp_rate": (16.0, 0.08, 0.7),
    "temperature": (36.8, 0.05, 0.06),
    "gcs": (15.0, 0.10, 0.0),
}

STATE_TARGETS: dict[ClinicalState, StateProfile] = {
    ClinicalState.STABLE: BASELINE,
    ClinicalState.RECOVERING: {
        "heart_rate": (84.0, 0.05, 1.8),
        "spo2": (96.0, 0.08, 0.6),
        "bp_systolic": (116.0, 0.05, 2.6),
        "resp_rate": (18.0, 0.07, 0.8),
        "temperature": (37.2, 0.05, 0.08),
        "gcs": (15.0, 0.08, 0.0),
    },
    ClinicalState.DETERIORATING: {
        "heart_rate": (118.0, 0.035, 2.6),
        "spo2": (91.0, 0.045, 0.9),
        "bp_systolic": (99.0, 0.035, 3.2),
        "resp_rate": (25.0, 0.05, 1.2),
        "temperature": (38.4, 0.035, 0.12),
        "gcs": (13.0, 0.02, 0.0),
    },
    ClinicalState.CRITICAL: {
        "heart_rate": (136.0, 0.05, 3.4),
        "spo2": (85.0, 0.06, 1.3),
        "bp_systolic": (82.0, 0.05, 4.0),
        "resp_rate": (31.0, 0.07, 1.6),
        "temperature": (39.3, 0.05, 0.16),
        "gcs": (9.0, 0.035, 0.0),
    },
}

#: Hard plausibility clamps - a simulator must never emit an impossible number.
CLAMPS: dict[str, tuple[float, float]] = {
    "heart_rate": (28.0, 190.0),
    "spo2": (60.0, 100.0),
    "bp_systolic": (55.0, 215.0),
    "resp_rate": (5.0, 46.0),
    "temperature": (33.2, 41.2),
    "gcs": (3.0, 15.0),
}

#: Probability per step that a given channel drops out (lead off, cuff cycling).
DROPOUT_PROBABILITY: dict[str, float] = {
    "heart_rate": 0.004,
    "spo2": 0.012,
    "bp_systolic": 0.045,
    "resp_rate": 0.020,
    "temperature": 0.055,
    "gcs": 0.030,
}

#: Nominal step used to calibrate reversion rates.
REFERENCE_STEP_MINUTES = 5.0


@dataclass
class SimulationEvent:
    """A transient physiological insult injected into a patient's trajectory."""

    label: str
    remaining_steps: int
    offsets: dict[str, float] = field(default_factory=dict)

    def decay(self) -> None:
        self.remaining_steps -= 1

    @property
    def active(self) -> bool:
        return self.remaining_steps > 0


#: Injectable events offered in the UI, keyed by a short slug.
EVENT_LIBRARY: dict[str, tuple[str, int, dict[str, float]]] = {
    "desaturation": (
        "Desaturation episode",
        24,
        {"spo2": -9.0, "resp_rate": 6.0, "heart_rate": 12.0},
    ),
    "sepsis": (
        "Suspected sepsis",
        60,
        {"temperature": 1.6, "heart_rate": 24.0, "bp_systolic": -22.0, "resp_rate": 5.0},
    ),
    "haemorrhage": (
        "Acute blood loss",
        36,
        {"bp_systolic": -32.0, "heart_rate": 30.0, "spo2": -3.0},
    ),
    "arrhythmia": (
        "Tachyarrhythmia",
        18,
        {"heart_rate": 46.0, "bp_systolic": -14.0},
    ),
    "bradycardia": (
        "Bradycardic episode",
        18,
        {"heart_rate": -40.0, "bp_systolic": -10.0},
    ),
    "neuro": (
        "Reduced consciousness",
        40,
        {"gcs": -5.0, "resp_rate": -3.0},
    ),
    "recovery": (
        "Response to treatment",
        48,
        {"spo2": 4.0, "heart_rate": -18.0, "bp_systolic": 14.0, "resp_rate": -5.0},
    ),
}


class PatientSimulator:
    """A single patient's evolving physiology.

    Args:
        seed: Reproducibility. The same seed always produces the same stay.
        state: Trajectory, as a :class:`ClinicalState` or its string value.
        age: Used to shift baselines slightly (older patients run stiffer and cooler).
        on_supplemental_oxygen: Sets the NEWS2 oxygen term and raises the SpO₂ target.
        dropouts: Whether to simulate missing channels. Off for deterministic tests.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        state: ClinicalState | str = ClinicalState.STABLE,
        age: int = 65,
        on_supplemental_oxygen: bool = False,
        dropouts: bool = True,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.state = ClinicalState(state) if not isinstance(state, ClinicalState) else state
        self.age = int(age)
        self.on_supplemental_oxygen = bool(on_supplemental_oxygen)
        self.dropouts = bool(dropouts)
        self.events: list[SimulationEvent] = []
        self.steps_taken = 0

        # Start near the *stable* baseline regardless of trajectory, so a
        # deteriorating patient visibly deteriorates instead of arriving broken.
        self._values: dict[str, float] = {}
        for channel, (target, _, noise) in BASELINE.items():
            offset = self._age_offset(channel)
            self._values[channel] = float(target + offset + self._rng.normal(0.0, noise))
        self._clamp_all()

    # -- internals ---------------------------------------------------------------------

    def _age_offset(self, channel: str) -> float:
        """Small, monotone age effects. Not clinical truth; keeps cohorts varied."""
        years = self.age - 60
        return {
            "bp_systolic": 0.22 * years,
            "heart_rate": -0.06 * years,
            "spo2": -0.015 * years,
            "temperature": -0.004 * years,
            "resp_rate": 0.02 * years,
            "gcs": 0.0,
        }.get(channel, 0.0)

    def _clamp_all(self) -> None:
        for channel, (low, high) in CLAMPS.items():
            if channel in self._values:
                self._values[channel] = float(min(high, max(low, self._values[channel])))

    def _active_offsets(self) -> dict[str, float]:
        offsets: dict[str, float] = {}
        for event in self.events:
            for channel, delta in event.offsets.items():
                offsets[channel] = offsets.get(channel, 0.0) + delta
        return offsets

    def _target_for(self, channel: str, profile: StateProfile) -> float:
        target, _, _ = profile[channel]
        target += self._age_offset(channel)
        if channel == "spo2" and self.on_supplemental_oxygen:
            target += 3.0
        return target

    # -- public API --------------------------------------------------------------------

    def set_state(self, state: ClinicalState | str) -> None:
        self.state = ClinicalState(state) if not isinstance(state, ClinicalState) else state

    def inject(self, slug: str) -> str | None:
        """Queue an event from :data:`EVENT_LIBRARY`. Returns its display label."""
        entry = EVENT_LIBRARY.get(slug)
        if entry is None:
            return None
        label, steps, offsets = entry
        self.events.append(
            SimulationEvent(label=label, remaining_steps=steps, offsets=dict(offsets))
        )
        return label

    @property
    def active_events(self) -> tuple[str, ...]:
        return tuple(event.label for event in self.events if event.active)

    def step(self, minutes: float = REFERENCE_STEP_MINUTES) -> Vitals:
        """Advance the simulation and return the resulting observation."""
        self.steps_taken += 1
        profile = STATE_TARGETS[self.state]
        # Reversion rates are calibrated per 5 minutes; scale for other step sizes.
        scale = max(0.05, min(6.0, float(minutes) / REFERENCE_STEP_MINUTES))
        offsets = self._active_offsets()

        for channel in BASELINE:
            _, reversion, noise = profile[channel]
            target = self._target_for(channel, profile) + offsets.get(channel, 0.0)
            pull = 1.0 - math.pow(1.0 - min(0.9, reversion), scale)
            drift = (target - self._values[channel]) * pull
            jitter = self._rng.normal(0.0, noise * math.sqrt(scale))
            self._values[channel] += drift + jitter

        # -- clinical couplings ---------------------------------------------------------
        hypoxia = max(0.0, 94.0 - self._values["spo2"])
        self._values["resp_rate"] += 0.28 * hypoxia * min(1.0, scale)
        hypotension = max(0.0, 105.0 - self._values["bp_systolic"])
        self._values["heart_rate"] += 0.16 * hypotension * min(1.0, scale)
        self._clamp_all()

        for event in self.events:
            event.decay()
        self.events = [event for event in self.events if event.active]

        return self._observe()

    def _observe(self) -> Vitals:
        """Read the current state as a bedside observation, with dropouts."""

        def maybe(channel: str, value: float) -> float | None:
            if self.dropouts and self._rng.random() < DROPOUT_PROBABILITY[channel]:
                return None
            return value

        systolic = self._values["bp_systolic"]
        # Pulse pressure narrows as patients decompensate; keep DBP physiological.
        pulse_pressure = float(
            np.clip(self._rng.normal(0.34 * systolic, 4.0), 18.0, 0.55 * systolic)
        )
        diastolic = max(28.0, systolic - pulse_pressure)

        gcs = float(round(self._values["gcs"]))
        raw_systolic = maybe("bp_systolic", round(systolic))

        return Vitals(
            heart_rate=maybe("heart_rate", round(self._values["heart_rate"])),
            spo2=maybe("spo2", round(self._values["spo2"])),
            bp_systolic=raw_systolic,
            # Diastolic comes from the same cuff reading, so it drops out together.
            bp_diastolic=None if raw_systolic is None else round(diastolic),
            resp_rate=maybe("resp_rate", round(self._values["resp_rate"])),
            temperature=maybe("temperature", round(self._values["temperature"], 1)),
            consciousness=Consciousness.from_gcs(gcs),
            on_supplemental_oxygen=self.on_supplemental_oxygen,
            gcs=maybe("gcs", gcs),
            recorded_at=utcnow(),
        )

    def snapshot(self) -> dict[str, float]:
        """Current latent values, before dropout - useful in tests."""
        return dict(self._values)


__all__ = [
    "BASELINE",
    "CLAMPS",
    "DROPOUT_PROBABILITY",
    "EVENT_LIBRARY",
    "STATE_TARGETS",
    "PatientSimulator",
    "SimulationEvent",
]
