"""The per-tick orchestrator: one pass over the ward, one assessment per bed.

This is the module the original project lacked. There, the Streamlit script itself read
the camera, scored vitals, called the model, and drew the UI in one function, so nothing
could be tested and nothing else could reuse the result. Here a tick is a pure-ish
function of the ward state:

    provider → vitals → NEWS2 → model → vision → fuse_risk → alerts → snapshot

The engine owns no presentation and no I/O it cannot skip. The dashboard, the API, and
the tests all call :meth:`MonitoringEngine.tick` and read the same
:class:`WardSnapshot`, which is what keeps the three surfaces honest about each other.

Degradation is deliberate at every stage: no trained model, no camera, and no database
are each a supported configuration, reported in the snapshot rather than raised.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.fusion import MLPrediction, fuse_risk
from icu_monitor.core.news2 import calculate_news2
from icu_monitor.core.types import (
    Alert,
    BedSnapshot,
    ClinicalState,
    NEWS2Result,
    Patient,
    RiskAssessment,
    RiskLevel,
    VisionSignal,
    Vitals,
    utcnow,
)
from icu_monitor.ml.registry import RiskModel, load_model
from icu_monitor.monitoring.alerts import AlertManager
from icu_monitor.simulation.ward import VitalsProvider, build_provider
from icu_monitor.vision.analyzer import VisionPipeline

logger = logging.getLogger(__name__)


class Recorder(Protocol):
    """The subset of the storage layer the engine uses.

    Declared here as a Protocol so the engine never imports the database. A
    ``None`` recorder is a supported configuration - the app runs entirely in memory.
    """

    def record_tick(
        self,
        patient: Patient,
        vitals: Vitals,
        assessment: RiskAssessment,
        alerts: Sequence[Alert],
    ) -> None: ...

    def upsert_patient(self, patient: Patient) -> None: ...


class HistoryStore(Protocol):
    """The read side of storage the engine uses to restore itself on startup.

    Kept separate from :class:`Recorder` because restoring is a distinct capability from
    recording, and again declared here so the engine never imports SQLAlchemy. The
    repository satisfies both protocols.
    """

    def recent_vitals(self, patient_id: str, *, limit: int = ...) -> list[Vitals]: ...

    def score_series(
        self, patient_id: str, *, limit: int = ...
    ) -> list[tuple[datetime, float, str]]: ...

    def alerts(
        self, *, patient_id: str | None = ..., open_only: bool = ..., limit: int = ...
    ) -> list[Alert]: ...


@dataclass(slots=True)
class WardSnapshot:
    """The whole ward after one tick - everything the UI or API needs, nothing more."""

    beds: tuple[BedSnapshot, ...]
    vision: VisionSignal
    focus_bed: str
    tick: int
    at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    model_version: str | None = None
    source_label: str = ""
    vision_label: str = ""

    @property
    def new_alerts(self) -> tuple[Alert, ...]:
        return tuple(alert for bed in self.beds for alert in bed.new_alerts)

    @property
    def worst(self) -> BedSnapshot | None:
        if not self.beds:
            return None
        return max(self.beds, key=lambda bed: (bed.level.rank, bed.assessment.composite_score))

    @property
    def mean_score(self) -> float:
        if not self.beds:
            return 0.0
        return sum(bed.assessment.composite_score for bed in self.beds) / len(self.beds)

    def level_counts(self) -> dict[RiskLevel, int]:
        counts = dict.fromkeys(
            (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL), 0
        )
        for bed in self.beds:
            if bed.level in counts:
                counts[bed.level] += 1
        return counts

    def bed(self, patient_id: str) -> BedSnapshot | None:
        for snapshot in self.beds:
            if snapshot.patient.patient_id == patient_id:
                return snapshot
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "at": self.at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "focus_bed": self.focus_bed,
            "model_version": self.model_version,
            "source_label": self.source_label,
            "vision_label": self.vision_label,
            "mean_score": round(self.mean_score, 2),
            "level_counts": {level.value: count for level, count in self.level_counts().items()},
            "beds": [bed.as_dict() for bed in self.beds],
        }


class MonitoringEngine:
    """Holds the ward's live state and advances it one tick at a time."""

    def __init__(
        self,
        *,
        config: Settings | None = None,
        provider: VitalsProvider | None = None,
        model: RiskModel | None = None,
        vision: VisionPipeline | None = None,
        alerts: AlertManager | None = None,
        recorder: Recorder | None = None,
        load_vision: bool = True,
    ) -> None:
        self._config = config or default_settings
        self.provider = provider or build_provider(self._config)
        self.model = model if model is not None else load_model(config=self._config)
        if vision is not None:
            self.vision: VisionPipeline | None = vision
        else:
            self.vision = VisionPipeline(config=self._config) if load_vision else None
        self.alerts = alerts or AlertManager(config=self._config)
        self.recorder = recorder
        self.tick_count = 0
        self.last_snapshot: WardSnapshot | None = None

        beds = self.provider.patients
        self._history: dict[str, deque[Vitals]] = {
            patient_id: deque(maxlen=self._config.history_window) for patient_id in beds
        }
        self._scores: dict[str, deque[tuple[datetime, float]]] = {
            patient_id: deque(maxlen=self._config.history_window) for patient_id in beds
        }
        self._focus_bed: str = next(iter(beds), "")

    # -- accessors ---------------------------------------------------------------------

    @property
    def patients(self) -> tuple[Patient, ...]:
        return tuple(self.provider.patients.values())

    @property
    def focus_bed(self) -> str:
        """The bed the single camera is pointed at."""
        return self._focus_bed

    @focus_bed.setter
    def focus_bed(self, patient_id: str) -> None:
        if patient_id not in self.provider.patients:
            return
        if patient_id != self._focus_bed and self.vision is not None:
            # Re-point the camera: posture streaks belong to a bed, not to the ward.
            self.vision.analyzer.reset()
        self._focus_bed = patient_id

    @property
    def model_version(self) -> str | None:
        return self.model.version if self.model is not None else None

    @property
    def vision_label(self) -> str:
        return self.vision.description if self.vision is not None else "Vision disabled"

    def history(self, patient_id: str) -> tuple[Vitals, ...]:
        return tuple(self._history.get(patient_id, ()))

    def score_history(self, patient_id: str) -> tuple[tuple[datetime, float], ...]:
        return tuple(self._scores.get(patient_id, ()))

    def patient(self, patient_id: str) -> Patient | None:
        return self.provider.patients.get(patient_id)

    # -- restore & reset ---------------------------------------------------------------

    def hydrate(self, store: HistoryStore) -> bool:
        """Repopulate per-patient history, score trends, and the alert ledger from storage.

        Called once at startup. A restart then resumes the ward it was monitoring - the trend
        charts, the open-alert wall, and the de-duplication state - instead of beginning from
        a blank history and re-raising every still-true condition as though it were new. Only
        beds the provider currently serves are restored; a bed the DB knows but this ward does
        not is ignored. Returns ``True`` if anything was restored.
        """
        window = self._config.history_window
        restored_any = False
        levels: dict[str, RiskLevel] = {}

        for patient_id in self.provider.patients:
            try:
                vitals = store.recent_vitals(patient_id, limit=window)
                scores = store.score_series(patient_id, limit=window)
            except Exception as exc:  # pragma: no cover - backend specific
                logger.warning("Could not restore history for %s (%s).", patient_id, exc)
                continue
            if vitals:
                history = self._history.setdefault(patient_id, deque(maxlen=window))
                history.clear()
                history.extend(vitals[-window:])
                restored_any = True
            if scores:
                trend = self._scores.setdefault(patient_id, deque(maxlen=window))
                trend.clear()
                trend.extend((at, score) for at, score, _level in scores[-window:])
                levels[patient_id] = RiskLevel.coerce(scores[-1][2])
                restored_any = True

        try:
            ledger = store.alerts(limit=self._config.alert_max_open)
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("Could not restore the alert ledger (%s).", exc)
            ledger = []
        if ledger:
            self.alerts.hydrate(ledger)
            restored_any = True
        if levels:
            self.alerts.seed_levels(levels)

        if restored_any:
            logger.info("Restored ward state from storage for %d bed(s).", len(self._history))
        return restored_any

    def reset(self) -> None:
        """Drop all in-memory history and the alert ledger, keeping the beds.

        The Settings page's purge empties the database; without this the engine would keep
        serving the trend charts and open alerts it had already loaded, so the dashboard would
        show data the operator had just deleted. Beds and the tick counter are left alone -
        the ward is still the same ward, it just has no past.
        """
        for buffer in self._history.values():
            buffer.clear()
        for trend in self._scores.values():
            trend.clear()
        self.alerts.clear()

    # -- one tick ----------------------------------------------------------------------

    def tick(self, *, at: datetime | None = None) -> WardSnapshot:
        """Advance the ward by one interval and assess every bed.

        ``at`` overrides the observation timestamp. Only warm-up uses it: forty ticks
        executed in one second would otherwise all carry the same ``recorded_at``, which
        collapses every trend chart to a single point and makes the alert cooldown
        meaningless. Back-dating on a synthetic clock gives history the shape it would
        have had if the app had been running all along.
        """
        started = time.perf_counter()
        self.tick_count += 1

        readings = self.provider.advance()
        if at is not None:
            for vitals in readings.values():
                vitals.recorded_at = at
        vision_signal = self._step_vision()

        beds: list[BedSnapshot] = []
        for patient_id, patient in self.provider.patients.items():
            vitals = readings.get(patient_id)
            if vitals is None:
                continue
            beds.append(self._assess(patient, vitals, vision_signal))

        snapshot = WardSnapshot(
            beds=tuple(beds),
            vision=vision_signal,
            focus_bed=self._focus_bed,
            tick=self.tick_count,
            # The snapshot carries the same instant as the observations inside it. Falling
            # back to the default factory here would date a back-filled tick to *now* while
            # its own vitals sit minutes in the past, and ``as_dict`` publishes both.
            at=at if at is not None else utcnow(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            model_version=self.model_version,
            source_label=self.provider.source_label,
            vision_label=self.vision_label,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _step_vision(self) -> VisionSignal:
        if self.vision is None:
            return VisionSignal(available=False, note="Vision disabled by configuration")
        try:
            return self.vision.step()
        except Exception as exc:  # pragma: no cover - vision must never take the ward down
            logger.warning("Vision step failed (%s); continuing without it.", exc)
            return VisionSignal(available=False, note=f"Vision error: {exc}")

    def _assess(
        self,
        patient: Patient,
        vitals: Vitals,
        vision_signal: VisionSignal,
    ) -> BedSnapshot:
        history = self._history.setdefault(
            patient.patient_id, deque(maxlen=self._config.history_window)
        )
        history.append(vitals)

        news2 = self._score_news2(patient, vitals)
        prediction = self._predict(patient, list(history))
        # One camera, one bed: every other bed is explicitly "not observed" rather than
        # silently inheriting a signal from someone else's body.
        vision = vision_signal if patient.patient_id == self._focus_bed else _unobserved()

        assessment = fuse_risk(
            patient_id=patient.patient_id,
            vitals=vitals,
            news2=news2,
            ml=prediction,
            vision=vision,
            config=self._config,
        )
        self._scores.setdefault(
            patient.patient_id, deque(maxlen=self._config.history_window)
        ).append((vitals.recorded_at, assessment.composite_score))

        new_alerts = self.alerts.evaluate(patient, vitals, assessment, now=vitals.recorded_at)
        if self.recorder is not None:
            try:
                self.recorder.record_tick(patient, vitals, assessment, new_alerts)
            except Exception as exc:  # pragma: no cover - persistence is best-effort
                logger.warning("Could not persist tick for %s (%s).", patient.bed, exc)

        return BedSnapshot(
            patient=patient, vitals=vitals, assessment=assessment, new_alerts=new_alerts
        )

    def _score_news2(self, patient: Patient, vitals: Vitals) -> NEWS2Result | None:
        try:
            return calculate_news2(vitals, spo2_scale=patient.spo2_scale)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("NEWS2 scoring failed for %s (%s).", patient.bed, exc)
            return None

    def _predict(self, patient: Patient, history: list[Vitals]) -> MLPrediction:
        if self.model is None:
            return MLPrediction.unavailable("no trained model")
        try:
            return self.model.predict_for_patient(patient, history)
        except Exception as exc:  # pragma: no cover - model/runtime specific
            logger.warning("Model inference failed for %s (%s).", patient.bed, exc)
            return MLPrediction.unavailable(f"inference error: {exc}")

    # -- controls the UI exposes -------------------------------------------------------

    def reload_model(self) -> RiskModel | None:
        """Pick up a newly trained artefact without restarting the process."""
        self.model = load_model(config=self._config, refresh=True)
        return self.model

    def annotated_frame(self):
        """The focus bed's latest frame with the bed region and boxes drawn on."""
        return self.vision.annotated_frame() if self.vision is not None else None

    def inject_event(self, patient_id: str, event: str) -> str | None:
        """Ask the provider to start a clinical event on one bed."""
        return self.provider.inject(patient_id, event)

    def _persist_patient(self, patient_id: str) -> None:
        """Best-effort write-back of a bed's record after a control change.

        Mirrors :meth:`record_tick`'s contract: the control has already taken effect in
        memory, so a failed database write is logged and swallowed rather than reported as a
        refused change. Without this, oxygen and trajectory changes made on one process were
        invisible to the other and lost on restart - the record on disk still said "room air".
        """
        if self.recorder is None:
            return
        patient = self.patient(patient_id)
        if patient is None:  # pragma: no cover - defensive
            return
        try:
            self.recorder.upsert_patient(patient)
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            logger.warning("Could not persist control change for %s (%s).", patient_id, exc)

    def set_state(self, patient_id: str, state: str) -> bool:
        """Set a bed's trajectory, refusing anything the provider cannot honour.

        Returns ``False`` rather than raising, because both callers treat this as a
        predicate: the dashboard shows "Provider refused the change" and the API turns a
        falsy result into a 409. Letting a ``ValueError`` out of here would take the
        patient view down over a bad dropdown value.

        Note that an unrecognised state is *refused*, not coerced. ``ClinicalState.coerce``
        exists for boundaries where a stale value should degrade to the benign default -
        reading an old database row - but a control surface must not report success for a
        change it did not make.
        """
        try:
            resolved = ClinicalState(str(state).strip().lower())
        except ValueError:
            logger.warning("Refused unknown clinical state %r for %s.", state, patient_id)
            return False
        changed = bool(self.provider.set_state(patient_id, resolved))
        if changed:
            self._persist_patient(patient_id)
        return changed

    def set_oxygen(self, patient_id: str, *, on: bool, scale: int | None = None) -> bool:
        """Set supplemental oxygen, and optionally the SpO₂ target scale.

        Scale 2 is a *prescription* - the 88-92 % target for chronic hypercapnic
        respiratory failure - so it lives on the patient record, not the simulator.
        """
        changed = bool(self.provider.set_oxygen(patient_id, on))
        patient = self.patient(patient_id)
        if patient is not None and scale in {1, 2}:
            patient.spo2_scale = int(scale)
            changed = True
        if changed:
            self._persist_patient(patient_id)
        return changed

    def active_events(self, patient_id: str) -> tuple[str, ...]:
        return tuple(self.provider.active_events(patient_id))

    def available_events(self) -> dict[str, str]:
        return dict(self.provider.available_events)

    def run(
        self,
        ticks: int,
        *,
        progress: Any | None = None,
        backfill: bool = False,
    ) -> WardSnapshot:
        """Advance several ticks in a row - used for warm-up and by the tests.

        With ``backfill`` the run is stamped on a synthetic clock ending at *now*, so the
        resulting history looks like a ward that has been monitored for
        ``ticks × tick_seconds`` rather than a burst inside one second.
        """
        count = max(0, ticks)
        snapshot = self.last_snapshot
        step = timedelta(seconds=self._config.tick_seconds)
        origin = utcnow() - step * count if backfill else None
        for index in range(count):
            at = origin + step * (index + 1) if origin is not None else None
            snapshot = self.tick(at=at)
            if progress is not None and index % 10 == 0:
                progress(f"tick {index + 1}/{count}")
        if snapshot is None:  # pragma: no cover - ticks <= 0 with no prior state
            snapshot = self.tick()
        return snapshot

    def close(self) -> None:
        if self.vision is not None:
            self.vision.close()


def _unobserved() -> VisionSignal:
    return VisionSignal(available=False, note="No camera assigned to this bed")


def build_engine(
    *,
    config: Settings | None = None,
    recorder: Recorder | None = None,
    warmup_ticks: int = 0,
) -> MonitoringEngine:
    """Construct the engine from configuration, optionally pre-filling history."""
    engine = MonitoringEngine(config=config, recorder=recorder)
    if warmup_ticks > 0:
        engine.run(warmup_ticks, backfill=True)
    return engine


__all__ = [
    "HistoryStore",
    "MonitoringEngine",
    "Recorder",
    "WardSnapshot",
    "build_engine",
]
