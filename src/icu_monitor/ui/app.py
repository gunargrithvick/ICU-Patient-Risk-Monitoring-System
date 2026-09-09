"""Streamlit entry point.

Run it with ``streamlit run app.py`` from the project root, or ``python -m icu_monitor
dashboard``.

**On refreshing.** The ward advances once per script run, at the top, and the view functions
receive that snapshot. Live updates come from a plain rerun loop rather than
``st.fragment(run_every=...)``: a fragment only reruns itself, so a button inside one that
changes which view is showing needs an app-scoped rerun to take effect, and threading that
through every card is a lot of machinery for a dashboard whose whole render is a few
milliseconds. A full rerun keeps every interaction working the way Streamlit documents.

The cost is that a click during the sleep waits for it to finish, which is why the interval
is capped at the tick interval rather than something leisurely.
"""

from __future__ import annotations

import time

import streamlit as st

from icu_monitor import __version__
from icu_monitor.logging_setup import configure_logging
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state
from icu_monitor.ui import theme
from icu_monitor.ui.views import alerts, model, overview, patient, settings

VIEWS = ("Ward overview", "Patient monitor", "Alerts", "Model insights", "Settings")


def _sidebar(snapshot) -> str:
    cfg = app_state.current_settings()
    with st.sidebar:
        st.markdown(
            f"<div style='font-size:1.05rem;font-weight:680;color:{theme.INK}'>"
            f"{cfg.app_name}</div>"
            f"<div style='color:{theme.INK_MUTED};font-size:0.76rem;margin-bottom:0.9rem'>"
            f"{cfg.app_tagline}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.setdefault("icu_view", VIEWS[0]) not in VIEWS:
            st.session_state["icu_view"] = VIEWS[0]
        view = st.radio(
            "View",
            options=VIEWS,
            key="icu_view",
            label_visibility="collapsed",
        )

        st.divider()
        st.session_state.setdefault("icu_live", True)
        live = st.toggle(
            "Live",
            key="icu_live",
            help=f"Re-runs every {cfg.tick_seconds:g} s, advancing the ward by one tick.",
        )
        if not live and st.button("Advance one tick", width="stretch"):
            app_state.snapshot(force=True)
            st.rerun()

        st.divider()
        counts = snapshot.level_counts()
        worst = snapshot.worst
        ui.definition_list(
            {
                "Beds": len(snapshot.beds),
                "Needing review": counts.get("HIGH", 0) + counts.get("CRITICAL", 0),
                "Active alerts": len(app_state.engine().alerts.active),
                "Highest risk": "—" if worst is None else worst.patient.bed,
                "Tick": snapshot.tick,
            }
        )

        st.divider()
        ui.caption(
            f"v{__version__} · {snapshot.source_label}<br>"
            f"model {snapshot.model_version or 'none'}<br>"
            f"vision {snapshot.vision_label}"
        )
        ui.caption(
            "<b>Not a medical device.</b> Educational demonstration only; no output here "
            "should inform patient care.",
            colour=theme.STATUS["warning"],
        )
    return view


def main() -> None:
    configure_logging()
    cfg = app_state.current_settings()
    st.set_page_config(
        page_title=f"{cfg.app_name} · ICU monitoring",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui.inject_theme()

    snapshot = app_state.snapshot()
    view = _sidebar(snapshot)

    if view == "Ward overview":
        overview.render(snapshot)
    elif view == "Patient monitor":
        patient.render(snapshot)
    elif view == "Alerts":
        alerts.render(snapshot)
    elif view == "Model insights":
        model.render()
    else:
        settings.render()

    if st.session_state.get("icu_live", True):
        time.sleep(max(0.5, min(float(cfg.tick_seconds), 5.0)))
        st.rerun()


if __name__ == "__main__":
    main()
