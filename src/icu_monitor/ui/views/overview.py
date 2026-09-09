"""Ward Overview: the "who needs me" view.

Ordered by risk, never by bed number. A ward list sorted by bed is a filing system; a ward
list sorted by composite score is a triage tool, and the whole point of computing a
composite score is to be able to sort by it.
"""

from __future__ import annotations

import streamlit as st

from icu_monitor.core.types import RiskLevel
from icu_monitor.monitoring.engine import WardSnapshot
from icu_monitor.ui import charts
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state


def _timeline_rows(snapshot: WardSnapshot, patient_ids: list[str]) -> list[dict[str, object]]:
    engine = app_state.engine()
    rows: list[dict[str, object]] = []
    beds = {bed.patient.patient_id: bed.patient.bed for bed in snapshot.beds}
    for patient_id in patient_ids:
        for moment, score in engine.score_history(patient_id):
            rows.append({"bed": beds.get(patient_id, patient_id), "at": moment, "score": score})
    return rows


def render(snapshot: WardSnapshot) -> None:
    settings = app_state.current_settings()
    engine = app_state.engine()

    ui.page_header(
        "Ward overview",
        f"{len(snapshot.beds)} beds · {snapshot.source_label}",
        right=(
            f"tick {snapshot.tick} · {snapshot.duration_ms:.0f} ms<br>{snapshot.at:%H:%M:%S} UTC"
        ),
    )

    counts = snapshot.level_counts()
    escalated = counts.get("HIGH", 0) + counts.get("CRITICAL", 0)
    worst = snapshot.worst
    active = engine.alerts.active
    ui.kpi_row(
        [
            ("Beds", str(len(snapshot.beds)), "Occupied beds under monitoring."),
            (
                "Needing review",
                str(escalated),
                "Beds at HIGH or CRITICAL composite risk.",
            ),
            (
                "Mean composite",
                f"{snapshot.mean_score:.0f}",
                "Mean of the fused 0-100 risk score across the ward.",
            ),
            (
                "Active alerts",
                str(len(active)),
                "Conditions true right now, de-duplicated per patient and kind.",
            ),
            (
                "Highest risk",
                "—"
                if worst is None
                else f"{worst.patient.bed} · {worst.assessment.composite_score:.0f}",
                "The bed the ward list is sorted by.",
            ),
        ]
    )

    left, right = st.columns([0.38, 0.62])
    with left:
        ui.chart_panel(
            "Risk distribution",
            charts.risk_distribution(counts),
            note="Beds per fused risk level. Glyph and label carry the level, not colour alone.",
        )
    with right:
        ranked = sorted(snapshot.beds, key=lambda b: b.assessment.composite_score, reverse=True)
        tracked = [bed.patient.patient_id for bed in ranked[:3]]
        ui.chart_panel(
            "Composite risk trend · three highest-risk beds",
            charts.score_timeline(
                _timeline_rows(snapshot, tracked),
                thresholds={
                    "medium": settings.composite_medium_threshold,
                    "high": settings.composite_high_threshold,
                    "critical": settings.composite_critical_threshold,
                },
            ),
            note="Capped at three beds: only three palette hues survive an all-pairs "
            "colour-vision check. Hover for exact values.",
            fallback="No history yet - the ward is still warming up.",
        )

    st.markdown("#### Beds, ordered by risk")
    level_filter = (
        st.segmented_control(
            "Show",
            options=["All", "Needs review", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default="All",
            key="icu_overview_filter",
            label_visibility="collapsed",
        )
        or "All"
    )

    beds = sorted(snapshot.beds, key=lambda b: b.assessment.composite_score, reverse=True)
    if level_filter == "Needs review":
        beds = [b for b in beds if b.assessment.level.rank >= RiskLevel.HIGH.rank]
    elif level_filter != "All":
        beds = [b for b in beds if b.assessment.level.value == level_filter]

    if not beds:
        ui.empty_state(f"No beds match “{level_filter}”.", icon="✓")
        return

    columns = 3
    for start in range(0, len(beds), columns):
        row = st.columns(columns, gap="small")
        for col, bed in zip(row, beds[start : start + columns], strict=False):
            with col:
                ui.bed_card(bed, on_open=app_state.select_patient, key_prefix="overview")


__all__ = ["render"]
