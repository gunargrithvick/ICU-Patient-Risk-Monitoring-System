"""Reusable dashboard pieces.

The one hard rule: **containers come from Streamlit, never from injected HTML.** Every card
here is a ``st.container(border=True)`` that real widgets are placed into. The previous
dashboard emitted ``<div class="card">`` through ``st.markdown`` and then called
``st.metric``; Streamlit renders each element into its own DOM node, so the div was closed
before the widget existed and the sanitiser discarded the rest. Cards that looked right and
did nothing.

Inline HTML *is* used, but only for runs of text - a coloured badge, a dim caption. Styling
a span of prose is not the same as trying to wrap an interactive widget, and it degrades to
plain text rather than to a broken layout.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from icu_monitor.core.types import Alert, BedSnapshot, RiskLevel, Vitals
from icu_monitor.ui import theme


def inject_theme() -> None:
    """Apply the stylesheet once per script run."""
    css = theme.load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None, *, right: str | None = None) -> None:
    """A view title, with optional right-aligned status text.

    ``subtitle`` is escaped; ``right`` is a markup slot - the overview passes a ``<br>`` to
    stack the tick and the clock. Only operator-controlled strings (tick counts, timestamps,
    the model version, the environment name) may go there. Patient data goes in ``subtitle``.
    """
    if right:
        left_col, right_col = st.columns([0.72, 0.28])
    else:
        left_col, right_col = st.container(), None
    with left_col:
        st.markdown(f"### {title}")
        if subtitle:
            st.markdown(
                f"<div style='color:{theme.INK_MUTED};font-size:0.86rem;margin-top:-0.5rem'>"
                f"{html.escape(subtitle)}</div>",
                unsafe_allow_html=True,
            )
    if right_col is not None:
        with right_col:
            st.markdown(
                f"<div style='text-align:right;color:{theme.INK_MUTED};font-size:0.8rem;"
                f"font-variant-numeric:tabular-nums;padding-top:0.5rem'>{right}</div>",
                unsafe_allow_html=True,
            )


def status_badge(level: RiskLevel | str, *, size: str = "0.78rem") -> str:
    """A risk level as glyph + word + colour.

    All three channels, always. The status palette's worst colour-vision pair sits in the
    band that is only acceptable with secondary encoding, so the glyph and the word are not
    decoration - they are what makes the colour legal.
    """
    key = str(getattr(level, "value", level)).upper()
    colour = theme.level_color(key)
    glyph = theme.level_glyph(key)
    label = RiskLevel.coerce(key).label
    return (
        f"<span style='display:inline-flex;align-items:center;gap:0.34em;"
        f"background:{colour}22;border:1px solid {colour}66;color:{colour};"
        f"border-radius:999px;padding:0.12em 0.6em;font-size:{size};font-weight:650;"
        f"letter-spacing:0.02em;white-space:nowrap'>{glyph} {label}</span>"
    )


def caption(text: str, *, colour: str | None = None) -> None:
    st.markdown(
        f"<div style='color:{colour or theme.INK_MUTED};font-size:0.78rem;"
        f"line-height:1.45'>{text}</div>",
        unsafe_allow_html=True,
    )


def empty_state(message: str, *, icon: str = "○") -> None:
    """What to show when there is genuinely nothing - never a blank panel."""
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center;padding:1.6rem 0.5rem;color:{theme.INK_MUTED}'>"
            f"<div style='font-size:1.6rem;line-height:1'>{icon}</div>"
            f"<div style='margin-top:0.5rem;font-size:0.86rem'>{html.escape(message)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


#: (attribute, label, unit, format). Order matches how a bedside chart is read.
VITAL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("heart_rate", "Heart rate", "bpm", "{:.0f}"),
    ("spo2", "SpO₂", "%", "{:.0f}"),
    ("bp_systolic", "Systolic", "mmHg", "{:.0f}"),
    ("bp_diastolic", "Diastolic", "mmHg", "{:.0f}"),
    ("map_mmhg", "MAP", "mmHg", "{:.0f}"),
    ("resp_rate", "Respiratory rate", "/min", "{:.0f}"),
    ("temperature", "Temperature", "°C", "{:.1f}"),
    ("shock_index", "Shock index", "", "{:.2f}"),
)


def format_vital(vitals: Vitals, attribute: str, spec: str) -> str:
    value = getattr(vitals, attribute, None)
    return "—" if value is None else spec.format(value)


def vitals_grid(vitals: Vitals, *, columns: int = 4) -> None:
    """The current observation as a grid of metrics.

    An absent channel renders as an em dash rather than a zero. A monitoring system that
    displays 0 mmHg for "not measured" is worse than one that admits it does not know.
    """
    cells = [
        (label, format_vital(vitals, attr, spec), unit) for attr, label, unit, spec in VITAL_ROWS
    ]
    for start in range(0, len(cells), columns):
        row = st.columns(columns)
        for col, (label, value, unit) in zip(row, cells[start : start + columns], strict=False):
            with col:
                st.metric(label, value if not unit else f"{value} {unit}".strip())


def consciousness_line(vitals: Vitals) -> str:
    """ACVPU, GCS and oxygen as one line.

    The letter *and* the word, because ``A`` is the notation on the paper NEWS2 chart while
    ``V``, ``P`` and ``U`` are not self-evident to anyone reading the dashboard over a
    clinician's shoulder. A plain string off a database row has no label and degrades to the
    letter alone. GCS is stored as a float and shown as the integer it actually is.
    """
    acvpu = vitals.consciousness
    letter = str(getattr(acvpu, "value", acvpu) or "—")
    word = str(getattr(acvpu, "label", "") or "")
    shown = f"{letter} ({word})" if word and word != letter else letter
    oxygen = "on supplemental O₂" if vitals.on_supplemental_oxygen else "room air"
    gcs = "—" if vitals.gcs is None else f"{vitals.gcs:.0f}"
    return f"ACVPU <b>{html.escape(shown)}</b> · GCS <b>{gcs}</b> · {oxygen}"


def bed_card(bed: BedSnapshot, *, on_open: Callable[[str], None], key_prefix: str = "bed") -> None:
    """One bed, as a card the reader can act on.

    Composite score is the headline because it is the one number that orders the ward; the
    level badge sits beside it so the number is never interpreted without its band.
    """
    patient = bed.patient
    assessment = bed.assessment
    with st.container(border=True):
        head, badge = st.columns([0.58, 0.42])
        with head:
            st.markdown(
                f"<div style='font-size:0.72rem;letter-spacing:0.08em;color:{theme.INK_MUTED};"
                f"font-weight:650'>{html.escape(patient.bed)}</div>"
                f"<div style='font-size:0.98rem;color:{theme.INK};font-weight:640;"
                f"margin-top:0.1rem'>{html.escape(patient.display_name)}</div>",
                unsafe_allow_html=True,
            )
        with badge:
            st.markdown(
                f"<div style='text-align:right;padding-top:0.35rem'>"
                f"{status_badge(assessment.level)}</div>",
                unsafe_allow_html=True,
            )

        score_col, news_col = st.columns(2)
        with score_col:
            st.metric("Composite", f"{assessment.composite_score:.0f}")
        with news_col:
            total = assessment.news2_total
            st.metric("NEWS2", "—" if total is None else str(total))

        vital_bits = " · ".join(
            f"{label} <b>{format_vital(bed.vitals, attr, spec)}</b>"
            for attr, label, _unit, spec in VITAL_ROWS[:3]
        )
        caption(vital_bits)
        caption(
            f"{html.escape(patient.primary_diagnosis)} · {patient.age}y · "
            f"LOS {patient.los_hours():.0f} h"
        )
        if assessment.overrides:
            caption(
                f"<span style='color:{theme.STATUS['serious']}'>⚑ "
                f"{html.escape(assessment.overrides[0])}</span>"
            )
        st.button(
            "Open monitor",
            key=f"{key_prefix}_{patient.patient_id}",
            width="stretch",
            on_click=on_open,
            args=(patient.patient_id,),
        )


def relative_age(moment: datetime | None, *, now: datetime | None = None) -> str:
    """ "4 m 12 s ago" - an absolute timestamp makes the reader do arithmetic."""
    if moment is None:
        return "—"
    reference = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (reference - moment).total_seconds())
    if seconds < 60:
        return f"{seconds:.0f} s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} m {int(seconds % 60)} s ago"
    return f"{seconds / 3600:.1f} h ago"


def alert_card(
    alert: Alert,
    *,
    on_acknowledge: Callable[[int], None] | None = None,
    key_prefix: str = "alert",
    show_patient: bool = True,
) -> None:
    """One alert, with its age and an acknowledge action.

    Age is shown because a two-second-old hypoxia alert and a nine-minute-old one call for
    different responses, and the raw timestamp buries that.
    """
    with st.container(border=True):
        body, action = st.columns([0.76, 0.24])
        with body:
            heading = alert.kind.label
            if show_patient:
                heading = f"{alert.patient_id} · {heading}"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap'>"
                f"{status_badge(alert.severity, size='0.72rem')}"
                f"<span style='color:{theme.INK};font-weight:640;font-size:0.9rem'>"
                f"{html.escape(heading)}</span></div>"
                f"<div style='color:{theme.INK_SECONDARY};font-size:0.84rem;"
                f"margin-top:0.3rem'>{html.escape(alert.message)}</div>",
                unsafe_allow_html=True,
            )
            trail = f"raised {relative_age(alert.created_at)}"
            if alert.last_seen_at and alert.last_seen_at != alert.created_at:
                trail += f" · still true {relative_age(alert.last_seen_at)}"
            if alert.acknowledged_by:
                trail += f" · acknowledged by {html.escape(alert.acknowledged_by)}"
            caption(trail)
        with action:
            if alert.is_open and on_acknowledge is not None and alert.alert_id is not None:
                st.button(
                    "Acknowledge",
                    key=f"{key_prefix}_{alert.alert_id}",
                    width="stretch",
                    on_click=on_acknowledge,
                    args=(alert.alert_id,),
                )
            elif not alert.is_open:
                caption("✓ acknowledged", colour=theme.STATUS["good"])


def kpi_row(items: Sequence[tuple[str, Any, str | None]]) -> None:
    """A row of headline numbers inside one bordered container."""
    with st.container(border=True):
        columns = st.columns(len(items))
        for col, (label, value, helptext) in zip(columns, items, strict=False):
            with col:
                st.metric(label, value, help=helptext)


def definition_list(rows: Mapping[str, Any]) -> None:
    """Label/value pairs as text, for metadata panels where a table is overkill."""
    body = "".join(
        f"<div style='display:flex;justify-content:space-between;gap:1rem;"
        f"padding:0.28rem 0;border-bottom:1px solid {theme.BORDER}'>"
        f"<span style='color:{theme.INK_MUTED};font-size:0.8rem'>{html.escape(str(k))}</span>"
        f"<span style='color:{theme.INK_SECONDARY};font-size:0.82rem;font-weight:600;"
        f"font-variant-numeric:tabular-nums;text-align:right'>{html.escape(str(v))}</span></div>"
        for k, v in rows.items()
    )
    st.markdown(body, unsafe_allow_html=True)


def chart_panel(
    title: str, chart: Any, *, note: str | None = None, fallback: str = "No data yet."
) -> None:
    """A titled chart in a card, with an honest empty state when there is nothing to draw."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if note:
            caption(note)
        if chart is None:
            st.markdown(
                f"<div style='color:{theme.INK_MUTED};font-size:0.82rem;padding:1.2rem 0;"
                f"text-align:center'>{html.escape(fallback)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.altair_chart(chart, width="stretch")


__all__ = [
    "VITAL_ROWS",
    "alert_card",
    "bed_card",
    "caption",
    "chart_panel",
    "consciousness_line",
    "definition_list",
    "empty_state",
    "format_vital",
    "inject_theme",
    "kpi_row",
    "page_header",
    "relative_age",
    "status_badge",
    "vitals_grid",
]
