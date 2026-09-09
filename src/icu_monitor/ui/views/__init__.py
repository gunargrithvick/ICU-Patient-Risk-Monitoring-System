"""Dashboard views, one module per screen.

Each ``render`` takes the snapshot it needs as an argument rather than fetching one itself,
so a view cannot advance the ward as a side effect of being drawn. The single tick per
script run happens in :mod:`icu_monitor.ui.app`.
"""

from __future__ import annotations

from icu_monitor.ui.views import alerts, model, overview, patient, settings

__all__ = ["alerts", "model", "overview", "patient", "settings"]
