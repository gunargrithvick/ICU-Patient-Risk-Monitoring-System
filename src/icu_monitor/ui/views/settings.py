"""Settings: change the ward's configuration and see the change take effect.

Every control here is wired to something the engine actually reads. Editing a threshold
rebuilds the engine with the new ``Settings`` object rather than patching a copy, because a
dashboard that shows 65 while the fusion layer still uses 70 is worse than one with no
controls at all.

Changes are session-scoped. They do not write to ``.env``, and they do not leak into the
HTTP API's process - that reads its own environment, which is the correct behaviour for a
service whose configuration should come from deployment rather than from a browser.
"""

from __future__ import annotations

from typing import get_args

import streamlit as st
from sqlalchemy.engine import make_url

from icu_monitor.config import DetectorName, FrameSourceName, Settings, VitalsSourceName
from icu_monitor.storage.repository import Repository
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state
from icu_monitor.ui import theme

# The option lists are read off the ``Literal`` aliases rather than retyped, and the numeric
# bounds off the field constraints, so a widget cannot offer a value the schema will refuse.
# It could before: this page listed a "none" detector that is spelled "off" in the config, and
# a bed count up to 32 against a field that stops at 24.
VITALS_SOURCES: tuple[str, ...] = get_args(VitalsSourceName)
FRAME_SOURCES: tuple[str, ...] = get_args(FrameSourceName)
DETECTORS: tuple[str, ...] = get_args(DetectorName)


def _safe_database_url(url: str | None) -> str:
    """Display the database target without exposing a password in the dashboard."""
    if not url:
        return "unset"
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        # Never fall back to the raw value: a malformed URL can still contain credentials.
        return "<configured URL could not be parsed>"


def _bounds(field: str, cap: float) -> tuple[float, float]:
    """The field's own ``ge`` as the widget minimum, and its ``le`` as the maximum.

    ``cap`` is the maximum for the fields that declare a floor and no ceiling
    (``history_window``, the two NEWS2 thresholds). That top end is a presentational choice,
    and it cannot contradict validation because there is nothing there to contradict.
    """
    meta = Settings.model_fields[field].metadata
    low = next((m.ge for m in meta if hasattr(m, "ge")), 0)
    high = next((m.le for m in meta if hasattr(m, "le")), cap)
    return float(low), float(high)


# The purge confirmation flips its flag from ``on_click`` rather than from the script body.
# Streamlit runs callbacks *before* the rerun they trigger, so the panel is gone by the time
# the body decides whether to draw it. Setting the flag inline instead left "Cancel" looking
# broken: the flag was already False, but the panel above it had been drawn from the old value
# and stayed on screen until something else happened to rerun the page.
def _ask_purge() -> None:
    st.session_state["icu_confirm_purge"] = True


def _cancel_purge() -> None:
    st.session_state["icu_confirm_purge"] = False


def _do_purge(repository: Repository) -> None:
    repository.purge()
    # The database is empty now; drop the history and alert ledger the engine still holds in
    # memory, or the dashboard would keep drawing trend charts and open alerts for data the
    # operator just deleted.
    app_state.engine().reset()
    st.session_state["icu_confirm_purge"] = False
    st.session_state["icu_purged"] = True


def _system_panel() -> None:
    state = app_state.state()
    settings = app_state.current_settings()
    with st.container(border=True):
        st.markdown("**System status**")
        for component in state.readiness():
            ready = bool(component["ready"])
            colour = theme.STATUS["good"] if ready else theme.STATUS["warning"]
            glyph = "●" if ready else "▲"
            st.markdown(
                f"<div style='display:flex;gap:0.5rem;align-items:baseline;padding:0.2rem 0'>"
                f"<span style='color:{colour};font-weight:700'>{glyph}</span>"
                f"<span style='color:{theme.INK};font-weight:600;font-size:0.85rem;"
                f"min-width:5.5rem'>{component['name']}</span>"
                f"<span style='color:{theme.INK_MUTED};font-size:0.8rem'>"
                f"{component['detail']}</span></div>",
                unsafe_allow_html=True,
            )

        repository = state.repository()
        st.divider()
        st.markdown("**Persistence**")
        if repository is None:
            ui.caption(
                "Running in memory only. Persistence is a convenience here, so a database "
                "that cannot be reached degrades to in-memory operation instead of stopping "
                "the monitor."
            )
        else:
            ui.definition_list({k: f"{v:,}" for k, v in repository.stats().items()})
            ui.caption(f"`{_safe_database_url(settings.database_url)}`")
            trim_col, purge_col = st.columns(2)
            with trim_col:
                if st.button("Trim to retention limit", width="stretch"):
                    removed = repository.trim()
                    total = sum(removed.values())
                    breakdown = ", ".join(f"{count:,} {name}" for name, count in removed.items())
                    st.success(f"Removed {total:,} row(s): {breakdown}.")
            with purge_col:
                st.button("Purge all rows", width="stretch", on_click=_ask_purge)
            if st.session_state.pop("icu_purged", False):
                st.success("Database emptied.")
            if st.session_state.get("icu_confirm_purge"):
                st.warning(
                    "This deletes every stored patient, observation, assessment, and alert. "
                    "It cannot be undone."
                )
                yes, no = st.columns(2)
                with yes:
                    st.button(
                        "Yes, purge",
                        type="primary",
                        width="stretch",
                        on_click=_do_purge,
                        args=(repository,),
                    )
                with no:
                    st.button("Cancel", width="stretch", on_click=_cancel_purge)


def _thresholds_form() -> None:
    settings = app_state.current_settings()
    with st.container(border=True):
        st.markdown("**Escalation thresholds**")
        ui.caption(
            "NEWS2 thresholds follow the Royal College of Physicians 2017 defaults (5 and 7). "
            "Composite thresholds define the bands an override lifts a patient into."
        )
        with st.form("icu_thresholds"):
            news_col, comp_col = st.columns(2)
            with news_col:
                low, high = _bounds("news2_medium_threshold", 20)
                news2_medium = st.number_input(
                    "NEWS2 → MEDIUM", int(low), int(high), settings.news2_medium_threshold, 1
                )
                low, high = _bounds("news2_high_threshold", 20)
                news2_high = st.number_input(
                    "NEWS2 → HIGH", int(low), int(high), settings.news2_high_threshold, 1
                )
            with comp_col:
                low, high = _bounds("composite_medium_threshold", 99)
                comp_medium = st.number_input(
                    "Composite → MEDIUM", low, high, settings.composite_medium_threshold, 1.0
                )
                low, high = _bounds("composite_high_threshold", 99)
                comp_high = st.number_input(
                    "Composite → HIGH", low, high, settings.composite_high_threshold, 1.0
                )
                low, high = _bounds("composite_critical_threshold", 100)
                comp_critical = st.number_input(
                    "Composite → CRITICAL", low, high, settings.composite_critical_threshold, 1.0
                )
            if st.form_submit_button("Apply thresholds", type="primary", width="stretch"):
                ordered = comp_medium < comp_high < comp_critical
                if news2_medium >= news2_high:
                    st.error("The NEWS2 MEDIUM threshold must be below the HIGH threshold.")
                elif not ordered:
                    st.error("Composite thresholds must increase: MEDIUM < HIGH < CRITICAL.")
                elif app_state.apply_settings(
                    news2_medium_threshold=int(news2_medium),
                    news2_high_threshold=int(news2_high),
                    composite_medium_threshold=float(comp_medium),
                    composite_high_threshold=float(comp_high),
                    composite_critical_threshold=float(comp_critical),
                ):
                    st.success("Thresholds applied; the ward was rebuilt with them.")
                    st.rerun()


def _weights_form() -> None:
    settings = app_state.current_settings()
    with st.container(border=True):
        st.markdown("**Fusion weights**")
        ui.caption(
            "Relative, not absolute: the three are normalised over whichever channels are "
            "available, so removing the camera does not silently shrink every score."
        )
        with st.form("icu_weights"):
            ml = st.slider("Model", 0.0, 1.0, float(settings.weight_ml), 0.05)
            news2 = st.slider("NEWS2", 0.0, 1.0, float(settings.weight_news2), 0.05)
            vision = st.slider("Vision", 0.0, 1.0, float(settings.weight_vision), 0.05)
            total = ml + news2 + vision
            ui.caption(
                f"Normalised split: model {ml / total:.0%} · NEWS2 {news2 / total:.0%} · "
                f"vision {vision / total:.0%}"
                if total > 0
                else "At least one weight must be above zero."
            )
            if st.form_submit_button("Apply weights", type="primary", width="stretch"):
                if total <= 0:
                    st.error("At least one weight must be above zero.")
                elif app_state.apply_settings(
                    weight_ml=ml, weight_news2=news2, weight_vision=vision
                ):
                    st.success("Weights applied.")
                    st.rerun()


def _ward_form() -> None:
    settings = app_state.current_settings()
    with st.container(border=True):
        st.markdown("**Ward and sensing**")
        ui.caption(
            "Changing any of these rebuilds the ward, which resets the observation history "
            "and the alert ledger."
        )
        with st.form("icu_ward"):
            col_a, col_b = st.columns(2)
            with col_a:
                low, high = _bounds("bed_count", 24)
                beds = st.number_input("Beds", int(low), int(high), int(settings.bed_count), 1)
                low, high = _bounds("tick_seconds", 30.0)
                tick = st.number_input(
                    "Tick interval (s)", low, high, float(settings.tick_seconds), 0.25
                )
                low, high = _bounds("history_window", 2000)
                history = st.number_input(
                    "History window (observations)",
                    int(low),
                    int(high),
                    int(settings.history_window),
                    10,
                )
                low, high = _bounds("api_warmup_ticks", 600)
                warmup = st.number_input(
                    "Warm-up ticks",
                    int(low),
                    int(high),
                    int(settings.api_warmup_ticks),
                    5,
                    help="Back-dated on a synthetic clock so the trend charts start populated.",
                )
            with col_b:
                source = st.selectbox(
                    "Vitals source",
                    VITALS_SOURCES,
                    index=VITALS_SOURCES.index(settings.vitals_source)
                    if settings.vitals_source in VITALS_SOURCES
                    else 0,
                )
                frame_source = st.selectbox(
                    "Frame source",
                    FRAME_SOURCES,
                    index=FRAME_SOURCES.index(settings.frame_source)
                    if settings.frame_source in FRAME_SOURCES
                    else 0,
                    help="'auto' prefers a camera, falls back to a synthetic ward bay. "
                    "'off' drops the channel entirely, and the fusion layer renormalises "
                    "over the two that remain. The app is fully functional with no camera.",
                )
                detector = st.selectbox(
                    "Detector",
                    DETECTORS,
                    index=DETECTORS.index(settings.detector)
                    if settings.detector in DETECTORS
                    else 0,
                    help="'auto' uses YOLO when ultralytics is installed, otherwise a NumPy "
                    "blob heuristic.",
                )
                low, high = _bounds("alert_cooldown_seconds", 600.0)
                cooldown = st.number_input(
                    "Alert cooldown (s)",
                    low,
                    high,
                    float(settings.alert_cooldown_seconds),
                    5.0,
                    help="How long a cleared alert stays quiet before it can re-fire. "
                    "Escalation to a higher severity always breaks through.",
                )
            submitted = st.form_submit_button("Rebuild ward", type="primary", width="stretch")
            # `and` short-circuits, so nothing is applied unless the button was pressed, and
            # nothing is announced unless the new settings validated.
            if submitted and app_state.apply_settings(
                bed_count=int(beds),
                tick_seconds=float(tick),
                history_window=int(history),
                api_warmup_ticks=int(warmup),
                vitals_source=source,
                frame_source=frame_source,
                detector=detector,
                alert_cooldown_seconds=float(cooldown),
            ):
                st.session_state.pop("icu_selected_patient", None)
                st.success("Ward rebuilt.")
                st.rerun()


def render() -> None:
    settings = app_state.current_settings()
    ui.page_header(
        "Settings",
        "Session-scoped. Every control below is read by the engine, not just displayed.",
        right=f"{settings.app_name} · {settings.environment}",
    )

    left, right = st.columns([0.52, 0.48])
    with left:
        _thresholds_form()
        _weights_form()
    with right:
        _system_panel()
        _ward_form()

    with st.container(border=True):
        st.markdown("**Security**")
        if settings.api_key:
            ui.caption(
                "`ICU_API_KEY` is set, so every `/api/v1` route requires a matching "
                "`X-API-Key` header. Health and readiness probes stay open so an "
                "orchestrator can reach them without a secret.",
                colour=theme.STATUS["good"],
            )
        else:
            st.warning(
                "`ICU_API_KEY` is not set: the HTTP API's `/api/v1` routes are "
                "unauthenticated. That is a reasonable default for a local demo and the "
                "wrong one for anything reachable from a network — set the variable before "
                "exposing the service beyond localhost.",
                icon="⚠",
            )
        ui.caption(f"API base URL: `{settings.api_base_url}` · docs at `/docs`")

    if st.button("Reset all settings to environment defaults"):
        app_state.reset_to_defaults()
        st.rerun()


__all__ = ["render"]
