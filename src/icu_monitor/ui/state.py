"""Engine lifecycle for the dashboard.

The dashboard reuses :class:`~icu_monitor.api.deps.AppState` rather than growing its own
engine owner. That class already solves the two problems a Streamlit front end has - build
the engine once, and advance the ward only when the last snapshot has gone stale - and
sharing it means the dashboard and the HTTP API cannot drift into two different definitions
of "the current state of the ward".

Streamlit reruns the whole script on every interaction, so the engine must live outside the
script run. ``st.cache_resource`` is the right primitive: one instance per process, keyed on
the settings that would invalidate it. ``st.session_state`` would rebuild the ward for every
browser tab, and a module global would not survive a code reload.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st
from pydantic import ValidationError

from icu_monitor.api.deps import AppState
from icu_monitor.config import Settings, get_settings
from icu_monitor.logging_setup import configure_logging
from icu_monitor.monitoring.engine import MonitoringEngine, WardSnapshot

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Starting the monitoring engine…")
def get_app_state(fingerprint: str) -> AppState:
    """One :class:`AppState` per distinct configuration.

    ``fingerprint`` is the cache key: changing a setting on the Settings page produces a
    different key and therefore a rebuilt ward, rather than silently reusing an engine
    configured the old way. It is also the key under which the built instance is tracked in
    :data:`_LIVE_STATES`, so a rebuild can close the engine it supersedes.
    """
    configure_logging()
    settings = st.session_state.get("icu_settings") or get_settings()
    state = AppState(settings)
    state.engine()  # build eagerly: a spinner here beats a stall on first paint
    # Streamlit hashes ``fingerprint`` because it does not start with an underscore, but it
    # never closes what it evicts. Track the live instance so a settings change can dispose
    # it - the vision camera handle and the SQLAlchemy engine both need closing.
    _LIVE_STATES[fingerprint] = state
    return state


#: Every AppState built this process, by fingerprint. Streamlit's cache holds its own
#: references; this exists only so we can *close* them, which the cache will not do for us.
_LIVE_STATES: dict[str, AppState] = {}


def _dispose_cached_states() -> None:
    """Close and forget every AppState built so far.

    ``st.cache_resource.clear()`` drops Streamlit's references but never calls ``close``, so
    the camera handle and the database engine would leak on every settings change or reset.
    """
    while _LIVE_STATES:
        _, old = _LIVE_STATES.popitem()
        try:
            old.close()
        except Exception:  # pragma: no cover - close is best-effort
            logger.warning("Could not close a superseded AppState cleanly.", exc_info=True)


def settings_fingerprint(settings: Settings) -> str:
    """A cache key covering *every* setting.

    Deliberately not a hand-picked subset. Thresholds, weights, bed count, and the vision
    source all reach the engine through the ``Settings`` instance it was constructed with,
    so any of them going stale produces a dashboard that shows one number and explains it
    with another. Hashing the whole object costs nothing and cannot be wrong.
    """
    return settings.model_dump_json()


def current_settings() -> Settings:
    """The settings this session is running with, editable from the Settings page."""
    if "icu_settings" not in st.session_state:
        st.session_state["icu_settings"] = get_settings()
    return st.session_state["icu_settings"]


def apply_settings(**changes: Any) -> bool:
    """Replace the session's settings and drop the cached engine so it is rebuilt.

    Returns ``False`` and leaves the session untouched if the change would not validate.
    The forms already bound their own inputs, so this is the belt to that braces - but a
    rejected value has to leave the previous settings intact, because half-applying a
    configuration would give a dashboard that draws one number and explains it with another.
    """
    current = current_settings()
    try:
        updated = current.with_overrides(**changes)
    except ValidationError as exc:
        st.error("\n".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
        return False
    st.session_state["icu_settings"] = updated
    _dispose_cached_states()
    get_app_state.clear()
    return True


def reset_to_defaults() -> None:
    """Forget session overrides and rebuild from environment defaults.

    Closes the current engine first, for the same reason :func:`apply_settings` does: the
    cache is about to forget it, and only this layer will ever call ``close`` on it.
    """
    st.session_state.pop("icu_settings", None)
    _dispose_cached_states()
    get_app_state.clear()


def state() -> AppState:
    """The live engine owner for this session."""
    return get_app_state(settings_fingerprint(current_settings()))


def engine() -> MonitoringEngine:
    return state().engine()


def acknowledge_alert(alert_id: int, *, by: str = "dashboard") -> None:
    """Acknowledge one alert in the live ledger *and*, when present, on disk.

    Both surfaces must agree. The in-memory manager drives the wall display this instant; the
    repository is what survives a restart and what the HTTP API reads. Writing only the former
    is the bug where an ack vanished the moment the dashboard rebuilt. The alert id is the
    database row's primary key, so it addresses the same event in both.
    """
    current = state()
    current.engine().alerts.acknowledge(alert_id, by=by)
    repository = current.repository()
    if repository is not None:
        repository.acknowledge(alert_id, by=by)


def acknowledge_all_alerts(*, patient_id: str | None = None, by: str = "dashboard") -> int:
    """Acknowledge every open alert (optionally for one bed) in the ledger and on disk."""
    current = state()
    cleared = current.engine().alerts.acknowledge_all(patient_id=patient_id, by=by)
    repository = current.repository()
    if repository is not None:
        repository.acknowledge_all(patient_id=patient_id, by=by)
    return cleared


def snapshot(*, force: bool = False) -> WardSnapshot:
    """The current ward, advanced first if ``tick_seconds`` have passed."""
    return state().snapshot(force=force)


def selected_patient(snap: WardSnapshot) -> str:
    """The patient the Patient Monitor is focused on, defaulting to the worst bed.

    Defaulting to the sickest patient rather than the first bed is the difference between a
    dashboard that answers "who needs me" and one that answers "who is in bed 1".
    """
    ids = [bed.patient.patient_id for bed in snap.beds]
    chosen = st.session_state.get("icu_selected_patient")
    if chosen in ids:
        return chosen
    worst = snap.worst
    fallback = worst.patient.patient_id if worst is not None else (ids[0] if ids else "")
    st.session_state["icu_selected_patient"] = fallback
    return fallback


def select_patient(patient_id: str) -> None:
    st.session_state["icu_selected_patient"] = patient_id
    st.session_state["icu_view"] = "Patient monitor"


__all__ = [
    "acknowledge_alert",
    "acknowledge_all_alerts",
    "apply_settings",
    "current_settings",
    "engine",
    "get_app_state",
    "reset_to_defaults",
    "select_patient",
    "selected_patient",
    "settings_fingerprint",
    "snapshot",
    "state",
]
