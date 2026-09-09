"""Streamlit dashboard for the ICU monitor.

Layered so each piece has one job:

* :mod:`~icu_monitor.ui.theme` - colour and typography tokens, and the packaged stylesheet.
* :mod:`~icu_monitor.ui.charts` - Altair builders. No Streamlit calls, so they are testable.
* :mod:`~icu_monitor.ui.components` - reusable blocks built from real Streamlit containers.
* :mod:`~icu_monitor.ui.state` - the cached engine, the snapshot, and the selected patient.
* :mod:`~icu_monitor.ui.views` - one module per screen.
* :mod:`~icu_monitor.ui.app` - page config, sidebar, routing, refresh loop.

``run`` is resolved lazily. Importing it pulls in every view, and the things that import this
package for its tokens alone - the tests, and anything reading :data:`theme.SERIES` - should
not pay for that.
"""

from __future__ import annotations

from typing import Any

from icu_monitor.ui import charts, components, state, theme

__all__ = ["charts", "components", "run", "state", "theme"]


def __getattr__(name: str) -> Any:
    if name == "run":
        from icu_monitor.ui.app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
