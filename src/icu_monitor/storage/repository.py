"""The only module that talks to the database.

Everything above this layer works in domain objects
(:class:`~icu_monitor.core.types.Patient`, ``Vitals``, ``RiskAssessment``, ``Alert``);
the mapping to and from rows lives here and nowhere else. That is what keeps SQLAlchemy
out of the engine, the API, and the UI.

Two design points worth stating:

**Writes never raise into the caller.** :meth:`Repository.record_tick` satisfies the
engine's ``Recorder`` protocol, and a monitoring system must not stop monitoring because
a disk filled up. Failures are counted and logged; the ward keeps ticking.

**Retention is enforced on write.** A demo left running for a weekend would otherwise
grow without bound. ``ICU_DB_RETENTION_ROWS`` caps the two high-volume tables, trimming
oldest-first every few hundred inserts rather than on every one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import (
    Alert,
    AlertKind,
    ClinicalState,
    Consciousness,
    Patient,
    RiskAssessment,
    RiskLevel,
    Vitals,
    utcnow,
)
from icu_monitor.storage.database import (
    AlertRow,
    AssessmentRow,
    PatientRow,
    VitalsRow,
    build_engine,
    build_session_factory,
    create_all,
    session_scope,
)

logger = logging.getLogger(__name__)

#: Inserts between retention sweeps. Counting is far cheaper than a DELETE per tick.
TRIM_EVERY = 250


class Repository:
    """Persistence for patients, observations, assessments, and alerts."""

    def __init__(
        self,
        *,
        config: Settings | None = None,
        engine: Engine | None = None,
        create: bool = True,
    ) -> None:
        self._config = config or default_settings
        self.engine = engine or build_engine(self._config)
        if create:
            create_all(self.engine)
        self._sessions = build_session_factory(self.engine)
        self._writes = 0
        self.write_errors = 0

    # -- upserts -----------------------------------------------------------------------

    def upsert_patient(self, patient: Patient) -> None:
        with session_scope(self._sessions) as session:
            row = session.get(PatientRow, patient.patient_id)
            if row is None:
                row = PatientRow(patient_id=patient.patient_id)
                session.add(row)
            row.bed = patient.bed
            row.display_name = patient.display_name
            row.age = patient.age
            row.sex = patient.sex
            row.primary_diagnosis = patient.primary_diagnosis
            row.state = patient.state.value
            row.on_supplemental_oxygen = bool(patient.on_supplemental_oxygen)
            row.spo2_scale = int(patient.spo2_scale)
            row.admitted_at = patient.admitted_at
            row.notes = patient.notes
            row.updated_at = utcnow()

    def sync_patients(self, patients: Sequence[Patient]) -> int:
        for patient in patients:
            self.upsert_patient(patient)
        return len(patients)

    # -- the Recorder protocol ---------------------------------------------------------

    def record_tick(
        self,
        patient: Patient,
        vitals: Vitals,
        assessment: RiskAssessment,
        alerts: Sequence[Alert],
    ) -> None:
        """Persist one bed's tick. Swallows its own failures by contract."""
        try:
            self._record_tick(patient, vitals, assessment, alerts)
        except Exception as exc:  # pragma: no cover - storage backend specific
            self.write_errors += 1
            logger.warning("Persistence failed for %s (%s).", patient.bed, exc)

    def _record_tick(
        self,
        patient: Patient,
        vitals: Vitals,
        assessment: RiskAssessment,
        alerts: Sequence[Alert],
    ) -> None:
        with session_scope(self._sessions) as session:
            if session.get(PatientRow, patient.patient_id) is None:
                session.add(
                    PatientRow(
                        patient_id=patient.patient_id,
                        bed=patient.bed,
                        display_name=patient.display_name,
                        age=patient.age,
                        sex=patient.sex,
                        primary_diagnosis=patient.primary_diagnosis,
                        state=patient.state.value,
                        on_supplemental_oxygen=bool(patient.on_supplemental_oxygen),
                        spo2_scale=int(patient.spo2_scale),
                        admitted_at=patient.admitted_at,
                        notes=patient.notes,
                    )
                )
                session.flush()

            session.add(
                VitalsRow(
                    patient_id=patient.patient_id,
                    recorded_at=vitals.recorded_at,
                    heart_rate=vitals.heart_rate,
                    spo2=vitals.spo2,
                    bp_systolic=vitals.bp_systolic,
                    bp_diastolic=vitals.bp_diastolic,
                    resp_rate=vitals.resp_rate,
                    temperature=vitals.temperature,
                    consciousness=(
                        vitals.consciousness.value if vitals.consciousness is not None else None
                    ),
                    gcs=vitals.gcs,
                    on_supplemental_oxygen=bool(vitals.on_supplemental_oxygen),
                    measured_channels=int(vitals.measured_channels),
                )
            )
            news2 = assessment.news2
            session.add(
                AssessmentRow(
                    patient_id=patient.patient_id,
                    assessed_at=assessment.assessed_at,
                    level=assessment.level.value,
                    composite_score=float(assessment.composite_score),
                    news2_total=int(news2.total) if news2 is not None else None,
                    news2_red_score=bool(news2.has_red_score) if news2 is not None else False,
                    ml_level=assessment.ml_level.value if assessment.ml_level else None,
                    ml_confidence=assessment.ml_confidence,
                    model_available=bool(assessment.model_available),
                    factors=[
                        {
                            "source": f.source,
                            "description": f.description,
                            "points": round(float(f.points), 3),
                            "severity": f.severity,
                        }
                        for f in assessment.factors
                    ],
                    overrides=list(assessment.overrides),
                )
            )
            for alert in alerts:
                session.add(
                    AlertRow(
                        alert_id=alert.alert_id,
                        patient_id=alert.patient_id,
                        kind=alert.kind.value,
                        severity=alert.severity.value,
                        message=alert.message,
                        detail=alert.detail,
                        created_at=alert.created_at,
                        acknowledged_at=alert.acknowledged_at,
                        acknowledged_by=alert.acknowledged_by,
                    )
                )

        self._writes += 1
        if self._writes % TRIM_EVERY == 0:
            self.trim()

    # -- acknowledgement ---------------------------------------------------------------

    def acknowledge(self, alert_id: int, *, by: str = "operator") -> int:
        """Mark a stored alert acknowledged. Returns the number of rows touched."""
        moment = utcnow()
        with session_scope(self._sessions) as session:
            rows = session.scalars(
                select(AlertRow).where(
                    AlertRow.alert_id == alert_id, AlertRow.acknowledged_at.is_(None)
                )
            ).all()
            for row in rows:
                row.acknowledged_at = moment
                row.acknowledged_by = by
            return len(rows)

    def acknowledge_all(self, *, patient_id: str | None = None, by: str = "operator") -> int:
        moment = utcnow()
        with session_scope(self._sessions) as session:
            statement = select(AlertRow).where(AlertRow.acknowledged_at.is_(None))
            if patient_id is not None:
                statement = statement.where(AlertRow.patient_id == patient_id)
            rows = session.scalars(statement).all()
            for row in rows:
                row.acknowledged_at = moment
                row.acknowledged_by = by
            return len(rows)

    # -- reads -------------------------------------------------------------------------

    def patients(self) -> list[Patient]:
        with session_scope(self._sessions) as session:
            rows = session.scalars(select(PatientRow).order_by(PatientRow.bed)).all()
            return [_to_patient(row) for row in rows]

    def patient(self, patient_id: str) -> Patient | None:
        with session_scope(self._sessions) as session:
            row = session.get(PatientRow, patient_id)
            return _to_patient(row) if row is not None else None

    def recent_vitals(self, patient_id: str, *, limit: int = 120) -> list[Vitals]:
        """The newest ``limit`` observations, returned oldest-first for plotting."""
        with session_scope(self._sessions) as session:
            rows = session.scalars(
                select(VitalsRow)
                .where(VitalsRow.patient_id == patient_id)
                .order_by(VitalsRow.recorded_at.desc(), VitalsRow.id.desc())
                .limit(max(1, limit))
            ).all()
        return [_to_vitals(row) for row in reversed(rows)]

    def score_series(
        self, patient_id: str, *, limit: int = 240
    ) -> list[tuple[datetime, float, str]]:
        """``(timestamp, composite, level)`` oldest-first - the trend line's data."""
        with session_scope(self._sessions) as session:
            rows = session.execute(
                select(
                    AssessmentRow.assessed_at,
                    AssessmentRow.composite_score,
                    AssessmentRow.level,
                )
                .where(AssessmentRow.patient_id == patient_id)
                .order_by(AssessmentRow.assessed_at.desc(), AssessmentRow.id.desc())
                .limit(max(1, limit))
            ).all()
        return [(_as_utc(at), float(score), level) for at, score, level in reversed(rows)]

    def alerts(
        self,
        *,
        patient_id: str | None = None,
        open_only: bool = False,
        limit: int = 100,
    ) -> list[Alert]:
        with session_scope(self._sessions) as session:
            statement = select(AlertRow)
            if patient_id is not None:
                statement = statement.where(AlertRow.patient_id == patient_id)
            if open_only:
                statement = statement.where(AlertRow.acknowledged_at.is_(None))
            rows = session.scalars(
                statement.order_by(AlertRow.created_at.desc(), AlertRow.id.desc()).limit(
                    max(1, limit)
                )
            ).all()
        return [_to_alert(row) for row in rows]

    def alert_counts_by_kind(self, *, since_hours: float = 24.0) -> dict[str, int]:
        cutoff = utcnow() - timedelta(hours=max(0.0, since_hours))
        with session_scope(self._sessions) as session:
            rows = session.execute(
                select(AlertRow.kind, func.count())
                .where(AlertRow.created_at >= cutoff)
                .group_by(AlertRow.kind)
                .order_by(func.count().desc())
            ).all()
        return {kind: int(count) for kind, count in rows}

    def stats(self) -> dict[str, int]:
        with session_scope(self._sessions) as session:
            return {
                "patients": int(session.scalar(select(func.count()).select_from(PatientRow)) or 0),
                "vitals": int(session.scalar(select(func.count()).select_from(VitalsRow)) or 0),
                "assessments": int(
                    session.scalar(select(func.count()).select_from(AssessmentRow)) or 0
                ),
                "alerts": int(session.scalar(select(func.count()).select_from(AlertRow)) or 0),
                "open_alerts": int(
                    session.scalar(
                        select(func.count())
                        .select_from(AlertRow)
                        .where(AlertRow.acknowledged_at.is_(None))
                    )
                    or 0
                ),
                "write_errors": self.write_errors,
            }

    def healthy(self) -> bool:
        """A cheap round-trip, used by ``/ready``."""
        try:
            with session_scope(self._sessions) as session:
                session.execute(select(func.count()).select_from(PatientRow))
            return True
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("Database health check failed (%s).", exc)
            return False

    # -- retention ---------------------------------------------------------------------

    def trim(self, *, keep: int | None = None) -> dict[str, int]:
        """Delete the oldest rows beyond the retention cap. Returns rows removed."""
        limit = keep if keep is not None else self._config.db_retention_rows
        removed = {"vitals": 0, "assessments": 0}
        try:
            with session_scope(self._sessions) as session:
                removed["vitals"] = _trim_table(session, VitalsRow, limit)
                removed["assessments"] = _trim_table(session, AssessmentRow, limit)
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("Retention sweep failed (%s).", exc)
        return removed

    def purge(self) -> None:
        """Empty every table. Used by the tests and the Settings page's reset button."""
        with session_scope(self._sessions) as session:
            for table in (AlertRow, AssessmentRow, VitalsRow, PatientRow):
                session.execute(delete(table))

    def close(self) -> None:
        self.engine.dispose()


# --------------------------------------------------------------------------------------
# Row <-> domain mapping
# --------------------------------------------------------------------------------------


def _trim_table(session, model, limit: int) -> int:
    total = int(session.scalar(select(func.count()).select_from(model)) or 0)
    excess = total - max(100, limit)
    if excess <= 0:
        return 0
    doomed = session.scalars(select(model.id).order_by(model.id.asc()).limit(excess)).all()
    if not doomed:
        return 0
    session.execute(delete(model).where(model.id.in_(doomed)))
    return len(doomed)


def _as_utc(value: datetime) -> datetime:
    """Re-attach UTC to a timestamp read back out of a row.

    SQLite has no timezone-aware storage: SQLAlchemy writes an aware datetime as a
    wall-clock string and hands it back naive. Every timestamp above this layer is aware -
    ``utcnow`` is the only clock that writes them - so a naive value escaping the mapper
    would raise ``can't subtract offset-naive and offset-aware datetimes`` in whichever
    caller first compared it with the present. Restoring the tzinfo here keeps that
    knowledge in the one module that knows it is talking to a database.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _to_patient(row: PatientRow) -> Patient:
    return Patient(
        patient_id=row.patient_id,
        bed=row.bed,
        display_name=row.display_name or "",
        age=row.age or 0,
        sex=row.sex or "U",
        admitted_at=_as_utc(row.admitted_at) if row.admitted_at else utcnow(),
        primary_diagnosis=row.primary_diagnosis or "",
        state=ClinicalState.coerce(row.state),
        on_supplemental_oxygen=bool(row.on_supplemental_oxygen),
        spo2_scale=int(row.spo2_scale or 1),
        notes=row.notes or "",
    )


def _to_vitals(row: VitalsRow) -> Vitals:
    return Vitals(
        heart_rate=row.heart_rate,
        spo2=row.spo2,
        bp_systolic=row.bp_systolic,
        bp_diastolic=row.bp_diastolic,
        resp_rate=row.resp_rate,
        temperature=row.temperature,
        consciousness=(
            Consciousness(row.consciousness) if row.consciousness else Consciousness.ALERT
        ),
        on_supplemental_oxygen=bool(row.on_supplemental_oxygen),
        gcs=row.gcs,
        recorded_at=_as_utc(row.recorded_at),
    )


def _to_alert(row: AlertRow) -> Alert:
    return Alert(
        patient_id=row.patient_id,
        kind=AlertKind(row.kind),
        severity=RiskLevel.coerce(row.severity),
        message=row.message or "",
        detail=row.detail or "",
        created_at=_as_utc(row.created_at),
        acknowledged_at=_as_utc(row.acknowledged_at) if row.acknowledged_at else None,
        acknowledged_by=row.acknowledged_by,
        alert_id=row.alert_id,
    )


def build_repository(config: Settings | None = None) -> Repository | None:
    """Build a repository, or ``None`` if the database cannot be opened.

    Returning ``None`` rather than raising is the whole point: persistence is a
    convenience here, and a broken database must degrade to in-memory operation instead
    of preventing the ward from being monitored at all.
    """
    try:
        return Repository(config=config)
    except Exception as exc:
        logger.warning("Database unavailable (%s); running without persistence.", exc)
        return None


__all__ = ["TRIM_EVERY", "Repository", "build_repository"]
