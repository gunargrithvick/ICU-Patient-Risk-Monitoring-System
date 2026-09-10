"""Alerts: the ledger, and the two questions it has to answer separately.

*Active* means the condition is true right now — that is what belongs on a wall display.
*Open* means an alert was raised and nobody has acknowledged it — that is the audit trail.
They are different sets and conflating them is how a monitor ends up either silent about a
deteriorating patient or shouting about one who recovered ten minutes ago, so this view
labels them separately and never adds them together.

The de-duplication that makes the ledger readable is a clinical requirement, not a
convenience: alarm fatigue is a recognised patient-safety hazard (Joint Commission National
Patient Safety Goal 06.01.01). One alert per patient-and-kind, a cooldown after it clears,
and escalation to a higher severity always breaks through.
"""

from __future__ import annotations

import streamlit as st

from icu_monitor.core.types import AlertKind
from icu_monitor.monitoring.engine import WardSnapshot
from icu_monitor.ui import charts
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state


def render(snapshot: WardSnapshot) -> None:
    engine = app_state.engine()
    manager = engine.alerts
    counts = manager.counts()
    history = app_state.persisted_alerts()
    open_alerts = tuple(alert for alert in history if alert.is_open)
    counts["open"] = len(open_alerts)
    counts["history"] = len(history)

    ui.page_header(
        "Alerts",
        "De-duplicated per patient and kind, with a cooldown after clearing.",
        right=f"tick {snapshot.tick} · {snapshot.at:%H:%M:%S} UTC",
    )

    ui.kpi_row(
        [
            ("Active now", str(counts.get("active", 0)), "Conditions true on this tick."),
            ("Open", str(counts.get("open", 0)), "Raised and not yet acknowledged."),
            ("Raised (session)", str(counts.get("history", 0)), "Total entries in the ledger."),
            ("Critical", str(counts.get("CRITICAL", 0)), "Active alerts at CRITICAL severity."),
            ("High", str(counts.get("HIGH", 0)), "Active alerts at HIGH severity."),
        ]
    )

    by_kind: dict[str, int] = {}
    for alert in history:
        by_kind[alert.kind.label] = by_kind.get(alert.kind.label, 0) + 1
    ui.chart_panel(
        "Alert volume by kind",
        charts.alert_kind_bars(by_kind),
        note="The alarm-fatigue view: a kind that dominates this chart is a threshold that "
        "needs tuning, not a ward that is sicker than it looks.",
        fallback="No alerts raised yet this session.",
    )

    with st.container(border=True):
        scope_col, patient_col, kind_col, action_col = st.columns([0.24, 0.26, 0.3, 0.2])
        with scope_col:
            scope = st.radio(
                "Scope",
                options=["Active", "Open", "All raised"],
                key="icu_alert_scope",
                horizontal=False,
            )
        with patient_col:
            beds = {bed.patient.patient_id: bed.patient.bed for bed in snapshot.beds}
            patient = st.selectbox(
                "Bed",
                options=["All beds", *beds],
                format_func=lambda pid: beds.get(pid, pid),
                key="icu_alert_patient",
            )
        with kind_col:
            kinds = st.multiselect(
                "Kind",
                options=[k.label for k in AlertKind],
                key="icu_alert_kinds",
                placeholder="Any kind",
            )
        with action_col:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Acknowledge all", width="stretch", type="primary"):
                target = None if patient == "All beds" else patient
                cleared = app_state.acknowledge_all_alerts(patient_id=target)
                st.success(f"Acknowledged {cleared} alert(s).")
                # The button action happens during this same Streamlit run. Refresh the
                # ledger-backed collections before rendering cards, otherwise the just-cleared
                # alerts remain visible until the next unrelated interaction.
                history = app_state.persisted_alerts()
                open_alerts = tuple(alert for alert in history if alert.is_open)

    if scope == "Active":
        alerts = list(manager.active)
    elif scope == "Open":
        alerts = list(open_alerts)
    else:
        alerts = list(history)

    if patient != "All beds":
        alerts = [a for a in alerts if a.patient_id == patient]
    if kinds:
        wanted = set(kinds)
        alerts = [a for a in alerts if a.kind.label in wanted]

    st.markdown(f"#### {len(alerts)} alert(s) · {scope.lower()}")
    if not alerts:
        ui.empty_state(
            "Nothing matches this filter."
            if scope != "Active"
            else "No conditions are currently true. The ward is stable.",
            icon="✓",
        )
        return

    for alert in alerts[:60]:
        ui.alert_card(
            alert, on_acknowledge=app_state.acknowledge_alert, key_prefix=f"alerts_{scope}"
        )
    if len(alerts) > 60:
        ui.caption(f"Showing the first 60 of {len(alerts)}.")


__all__ = ["render"]
