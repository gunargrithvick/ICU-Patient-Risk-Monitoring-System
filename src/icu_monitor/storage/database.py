"""SQLAlchemy schema and engine/session plumbing.

Four tables, one purpose each:

``patients``
    One row per bed occupant. Static demographics plus the current trajectory.
``vitals_readings``
    Every observation, exactly as it arrived - including the ``NULL``s. Missingness is
    clinical information and the table must not launder it away.
``risk_assessments``
    One row per tick per bed: the composite, the level, the NEWS2 total, and the
    itemised factors as JSON so a score can be explained months later.
``alerts``
    The alert ledger, with acknowledgement columns so the audit trail survives a restart.

The database is optional. Nothing above this layer requires it, and
:class:`~icu_monitor.storage.repository.Repository` is handed to the engine as a
``Recorder`` only when persistence is wanted. SQLite is the default because it needs no
server; ``ICU_DATABASE_URL`` swaps in Postgres without a code change.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    event,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.orm import Session as SASession
from sqlalchemy.pool import StaticPool

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import utcnow

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for every table in the application."""


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


class PatientRow(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    bed: Mapped[str] = mapped_column(String(32), index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(1), nullable=True)
    primary_diagnosis: Mapped[str] = mapped_column(String(200), default="")
    state: Mapped[str] = mapped_column(String(24), default="stable")
    on_supplemental_oxygen: Mapped[bool] = mapped_column(Boolean, default=False)
    spo2_scale: Mapped[int] = mapped_column(Integer, default=1)
    admitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str] = mapped_column(String(400), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class VitalsRow(Base):
    __tablename__ = "vitals_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.patient_id", ondelete="CASCADE"), index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    bp_systolic: Mapped[float | None] = mapped_column(Float, nullable=True)
    bp_diastolic: Mapped[float | None] = mapped_column(Float, nullable=True)
    resp_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    consciousness: Mapped[str | None] = mapped_column(String(16), nullable=True)
    gcs: Mapped[float | None] = mapped_column(Float, nullable=True)
    on_supplemental_oxygen: Mapped[bool] = mapped_column(Boolean, default=False)
    measured_channels: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("ix_vitals_patient_time", "patient_id", "recorded_at"),)


class AssessmentRow(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.patient_id", ondelete="CASCADE"), index=True
    )
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    level: Mapped[str] = mapped_column(String(16), index=True)
    composite_score: Mapped[float] = mapped_column(Float)
    news2_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    news2_red_score: Mapped[bool] = mapped_column(Boolean, default=False)
    ml_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_available: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON, not a child table: factors are read as a whole or not at all, and a
    # normalised design would trade a genuinely useful blob for a costly join.
    factors: Mapped[list] = mapped_column(JSON, default=list)
    overrides: Mapped[list] = mapped_column(JSON, default=list)

    __table_args__ = (Index("ix_assessment_patient_time", "patient_id", "assessed_at"),)


class AlertRow(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.patient_id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str] = mapped_column(String(400), default="")
    detail: Mapped[str] = mapped_column(String(600), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SchemaMeta(Base):
    """One row per housekeeping key. Currently just ``version`` for the migration runner.

    A tiny key/value table rather than SQLite's ``PRAGMA user_version`` so the same version
    stamp works on Postgres too, and so migrations that need to remember more than an integer
    have somewhere to write it.
    """

    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[str] = mapped_column(String(200), default="")


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _is_memory(url: str) -> bool:
    return ":memory:" in url or url.endswith("sqlite://")


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    """WAL + foreign keys for SQLite; a no-op for every other backend.

    Without ``journal_mode=WAL`` the dashboard's writer blocks the API's reader on the
    same file, which shows up as random UI stalls rather than as an obvious error.
    """
    if type(dbapi_connection).__module__.split(".")[0] != "sqlite3":
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def build_engine(config: Settings | None = None, *, url: str | None = None) -> Engine:
    """Create the SQLAlchemy engine, making the SQLite directory if needed."""
    cfg = config or default_settings
    resolved = url or cfg.database_url or "sqlite://"

    kwargs: dict[str, object] = {"future": True, "echo": False}
    if _is_sqlite(resolved):
        kwargs["connect_args"] = {"check_same_thread": False}
        if _is_memory(resolved):
            # One shared connection, or each session would get its own empty database.
            kwargs["poolclass"] = StaticPool
        else:
            target = resolved.split("sqlite:///", 1)[-1]
            if target and target != resolved:
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(resolved, **kwargs)
    logger.debug("Database engine ready (%s).", engine.url.render_as_string(hide_password=True))
    return engine


def create_all(engine: Engine) -> None:
    """Create any missing tables. Idempotent, so it is safe on every start."""
    Base.metadata.create_all(engine)


#: Bump this when a migration is added below. ``migrate`` walks from the database's stored
#: version up to here, so a change without a matching migration step is a visible mistake.
SCHEMA_VERSION = 1


def _migrate_to_v1(session: SASession) -> None:
    """Adopt each alert's row primary key as its ``alert_id``.

    Older builds stored the ``AlertManager``'s per-process counter, which restarts at 1 in
    every process, so an acknowledgement could address more than one event. The row PK is
    unique across processes and restarts; copying it over makes historical rows consistent
    with what the repository writes now. Idempotent and a no-op on a fresh database.
    """
    session.execute(
        update(AlertRow)
        .where((AlertRow.alert_id.is_(None)) | (AlertRow.alert_id != AlertRow.id))
        .values(alert_id=AlertRow.id)
    )


#: Ordered migration steps, keyed by the version each one produces.
_MIGRATIONS: dict[int, "Callable[[SASession], None]"] = {
    1: _migrate_to_v1,
}


def _read_schema_version(session: SASession) -> int:
    row = session.get(SchemaMeta, "version")
    if row is None:
        return 0
    try:
        return int(row.value)
    except (TypeError, ValueError):  # pragma: no cover - defensive against a corrupt stamp
        return 0


def _stamp_schema_version(session: SASession, version: int) -> None:
    row = session.get(SchemaMeta, "version")
    if row is None:
        session.add(SchemaMeta(key="version", value=str(version)))
    else:
        row.value = str(version)


def migrate(engine: Engine) -> int:
    """Create missing tables, then bring the schema up to :data:`SCHEMA_VERSION`.

    ``create_all`` alone creates new *tables* but cannot alter an existing one or backfill
    data, so a schema change would silently leave a half-old database in place. This walks
    ordered, idempotent steps and records the result in ``schema_meta`` - a stamp that works
    on SQLite and Postgres alike. A database written by a newer build is left untouched with
    a loud warning rather than "migrated" backwards. Returns the version now in force.
    """
    create_all(engine)
    factory = build_session_factory(engine)
    with session_scope(factory) as session:
        current = _read_schema_version(session)
        if current > SCHEMA_VERSION:
            logger.warning(
                "Database schema is v%d but this build understands only v%d; "
                "continuing without migrating.",
                current,
                SCHEMA_VERSION,
            )
            return current
        for version in range(current + 1, SCHEMA_VERSION + 1):
            _MIGRATIONS[version](session)
            logger.info("Applied database migration to v%d.", version)
        if current < SCHEMA_VERSION:
            _stamp_schema_version(session, SCHEMA_VERSION)
    return SCHEMA_VERSION


def build_session_factory(engine: Engine) -> sessionmaker[SASession]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[SASession]) -> Iterator[SASession]:
    """Transactional scope: commit on success, roll back on failure, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "SCHEMA_VERSION",
    "AlertRow",
    "AssessmentRow",
    "Base",
    "PatientRow",
    "SchemaMeta",
    "VitalsRow",
    "build_engine",
    "build_session_factory",
    "create_all",
    "migrate",
    "session_scope",
]
