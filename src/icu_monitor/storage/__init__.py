"""Optional persistence: SQLAlchemy tables and the single repository that maps them.

The application runs perfectly without a database - :func:`build_repository` returns
``None`` when one cannot be opened and the engine simply keeps its history in memory.
Nothing outside this package imports SQLAlchemy.
"""

from __future__ import annotations

from icu_monitor.storage.database import (
    AlertRow,
    AssessmentRow,
    Base,
    PatientRow,
    VitalsRow,
    build_engine,
    build_session_factory,
    create_all,
    session_scope,
)
from icu_monitor.storage.repository import TRIM_EVERY, Repository, build_repository

__all__ = [
    "TRIM_EVERY",
    "AlertRow",
    "AssessmentRow",
    "Base",
    "PatientRow",
    "Repository",
    "VitalsRow",
    "build_engine",
    "build_repository",
    "build_session_factory",
    "create_all",
    "session_scope",
]
