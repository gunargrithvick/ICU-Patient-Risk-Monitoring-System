"""FastAPI service: stateless scoring plus read access to live ward state.

Import :data:`icu_monitor.api.main.app` for ASGI, or call
:func:`icu_monitor.api.main.create_app` to build an isolated instance in tests.
"""

from __future__ import annotations

from icu_monitor.api.deps import AppState, get_state, require_api_key, reset_state
from icu_monitor.api.main import app, create_app

__all__ = ["AppState", "app", "create_app", "get_state", "require_api_key", "reset_state"]
