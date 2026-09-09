"""Patient Monitor: one bed, in full, with the score taken apart.

The organising idea is that a risk score nobody can interrogate is a risk score nobody will
act on. So this view never shows the composite alone: beside it are the NEWS2 parameters
that fed it, the itemised fusion contributions that sum to it, the model's class
probabilities, and any clinical override that lifted it. A clinician who disagrees with the
number can see exactly which input to argue with.
"""

from __future__ import annotations

import html

import streamlit as st

from icu_monitor.core.types import BedSnapshot
from icu_monitor.monitoring.engine import WardSnapshot
from icu_monitor.ui import charts, theme
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state


def _selector(snapshot: WardSnapshot) -> str:
    ids = [bed.patient.patient_id for bed in snapshot.beds]
    labels = {
        bed.patient.patient_id: (
            f"{bed.patient.bed} · {bed.patient.display_name} · "
            f"{theme.level_glyph(bed.assessment.level.value)} "
            f"{bed.assessment.level.label} {bed.assessment.composite_score:.0f}"
        )
        for bed in snapshot.beds
    }
    current = app_state.selected_patient(snapshot)
    chosen = st.selectbox(
        "Bed",
        options=ids,
        index=ids.index(current) if current in ids else 0,
        format_func=lambda pid: labels.get(pid, pid),
        key="icu_patient_selector",
    )
    st.session_state["icu_selected_patient"] = chosen
    return chosen


def _identity(bed: BedSnapshot) -> None:
    patient = bed.patient
    assessment = bed.assessment
    with st.container(border=True):
        head, score, news = st.columns([0.5, 0.25, 0.25])
        with head:
            st.markdown(
                f"<div style='font-size:1.15rem;color:{theme.INK};font-weight:660'>"
                f"{html.escape(patient.display_name)}</div>"
                f"<div style='margin-top:0.35rem'>{ui.status_badge(assessment.level)}</div>",
                unsafe_allow_html=True,
            )
            ui.caption(
                f"{html.escape(patient.bed)} · {patient.age}y {html.escape(patient.sex)} · "
                f"{html.escape(patient.primary_diagnosis)} · LOS {patient.los_hours():.0f} h · "
                f"state {html.escape(patient.state.label)}"
            )
        with score:
            st.metric(
                "Composite risk", f"{assessment.composite_score:.1f}", help=assessment.summary
            )
        with news:
            total = assessment.news2_total
            st.metric(
                "NEWS2",
                "—"
                if total is None
                else f"{total}/{bed.assessment.news2.max_total if bed.assessment.news2 else 20}",
                help="Royal College of Physicians National Early Warning Score 2 (2017).",
            )

        if assessment.news2 is not None:
            ui.caption(
                f"<b>Recommended response.</b> {html.escape(assessment.news2.clinical_response)}"
            )
        if assessment.overrides:
            items = "".join(
                f"<li style='margin:0.12rem 0'>{html.escape(text)}</li>"
                for text in assessment.overrides
            )
            st.markdown(
                f"<div style='margin-top:0.5rem;border-left:3px solid {theme.STATUS['serious']};"
                f"padding:0.35rem 0.7rem;background:{theme.STATUS['serious']}14;border-radius:0 6px 6px 0'>"
                f"<div style='color:{theme.STATUS['serious']};font-size:0.76rem;font-weight:680;"
                f"letter-spacing:0.05em'>⚑ CLINICAL OVERRIDES APPLIED</div>"
                f"<ul style='margin:0.3rem 0 0 1rem;padding:0;color:{theme.INK_SECONDARY};"
                f"font-size:0.82rem'>{items}</ul></div>",
                unsafe_allow_html=True,
            )


def _controls(patient_id: str) -> None:
    """The simulator's control surface.

    A provider that replays recorded data cannot be steered, and the whole panel goes with
    it - so the reason is stated in place of the controls rather than left as an empty card.
    Saying which provider is in charge matters more than the controls do: a reader who sees
    no Inject button and no explanation cannot tell a missing feature from an inapplicable
    one.
    """
    engine = app_state.engine()
    events = engine.available_events()
    with st.container(border=True):
        st.markdown("**Inject a clinical event**")
        if not events:
            ui.caption(
                f"Unavailable: {html.escape(engine.provider.source_label)} replays recorded "
                "data, so neither an injected event nor a bedside adjustment can steer its "
                "physiology."
            )
            return
        ui.caption(
            "Drives the synthetic physiology so the scoring, alerting, and escalation path "
            "can be demonstrated without waiting for a patient to deteriorate."
        )
        active = engine.active_events(patient_id)
        if active:
            ui.caption(
                "Running: " + ", ".join(html.escape(e) for e in active),
                colour=theme.STATUS["warning"],
            )

        slugs = list(events)
        chosen = st.selectbox(
            "Event",
            options=slugs,
            format_func=lambda s: events.get(s, s),
            key=f"icu_event_{patient_id}",
        )
        if st.button("Inject", key=f"icu_inject_{patient_id}", type="primary", width="stretch"):
            # ``inject_event`` returns the event's display label, not a sentence. Naming the
            # bed it landed on matters when the reader has several tabs open on the ward.
            label = engine.inject_event(patient_id, chosen)
            if label:
                st.success(f"{label} started for {patient_id}.")
            else:
                st.warning("That event could not be started for this patient.")

        st.divider()
        st.markdown("**Bedside adjustments**")
        record = engine.patient(patient_id)
        states = ["stable", "recovering", "deteriorating", "critical"]
        state_col, oxy_col = st.columns(2)
        with state_col:
            state_key = f"icu_state_{patient_id}"
            if record is not None:
                st.session_state.setdefault(state_key, str(record.state.value))
            new_state = st.selectbox(
                "Clinical trajectory",
                options=states,
                format_func=str.title,
                key=state_key,
            )
            if st.button("Apply", key=f"icu_setstate_{patient_id}", width="stretch"):
                if engine.set_state(patient_id, new_state):
                    st.success(f"{patient_id} set to {new_state}.")
                else:
                    st.warning("Provider refused the change.")
        with oxy_col:
            oxy_key = f"icu_oxy_{patient_id}"
            scale_key = f"icu_scale_{patient_id}"
            if record is not None:
                st.session_state.setdefault(oxy_key, bool(record.on_supplemental_oxygen))
                st.session_state.setdefault(scale_key, int(record.spo2_scale))
            on_oxygen = st.toggle("Supplemental O₂", key=oxy_key)
            scale = st.radio(
                "SpO₂ target scale",
                options=[1, 2],
                horizontal=True,
                key=scale_key,
                format_func=lambda s: f"Scale {s}",
                help="Scale 2 is the 88-92% target for chronic hypercapnic respiratory failure.",
            )
            if st.button("Update O₂", key=f"icu_setoxy_{patient_id}", width="stretch"):
                if engine.set_oxygen(patient_id, on=on_oxygen, scale=scale):
                    st.success("Oxygen settings updated.")
                else:
                    st.warning("Provider refused the change.")


def _vision_panel(snapshot: WardSnapshot, patient_id: str) -> None:
    engine = app_state.engine()
    signal = snapshot.vision
    with st.container(border=True):
        st.markdown("**Bedside camera**")
        ui.caption(html.escape(engine.vision_label))
        if signal is None or not signal.available:
            note = (signal.note if signal is not None else None) or "No frame source available."
            ui.caption(f"Inactive — {html.escape(note)}")
            return

        focused = snapshot.focus_bed == patient_id
        if not focused:
            ui.caption(
                f"Currently watching {html.escape(snapshot.focus_bed or '—')}. One camera "
                "serves the ward, so it is assigned to a bed rather than duplicated per bed."
            )
            if st.button("Point camera here", key=f"icu_focus_{patient_id}", width="stretch"):
                engine.focus_bed(patient_id)
                st.rerun()
            return

        frame = engine.annotated_frame()
        if frame is not None:
            st.image(frame, channels="BGR", width="stretch")
        ui.definition_list(
            {
                "Posture": signal.posture.value.title(),
                "Person present": "yes" if signal.patient_present else "no",
                "People detected": signal.person_count,
                "Motion index": f"{signal.motion_index:.3f}",
                "Fall suspected": "YES" if signal.fall_suspected else "no",
                "Bed exit suspected": "YES" if signal.bed_exit_suspected else "no",
                "Detector": signal.backend,
                "Latency": f"{signal.latency_ms:.0f} ms",
            }
        )


def render(snapshot: WardSnapshot) -> None:
    engine = app_state.engine()
    patient_id = _selector(snapshot)
    bed = snapshot.bed(patient_id)
    if bed is None:
        ui.empty_state(f"No bed for {patient_id}.")
        return

    ui.page_header(
        "Patient monitor",
        "Every input that produced the composite score, shown beside it.",
        right=f"assessed {ui.relative_age(bed.assessment.assessed_at)}",
    )
    _identity(bed)

    with st.container(border=True):
        st.markdown("**Current observation**")
        ui.caption(ui.consciousness_line(bed.vitals))
        ui.vitals_grid(bed.vitals)
        missing = [
            label
            for attr, label, _u, _f in ui.VITAL_ROWS[:7]
            if getattr(bed.vitals, attr, None) is None
        ]
        if missing:
            ui.caption(
                f"Not measured this tick: {', '.join(missing)}. NEWS2 is reported as "
                "incomplete rather than assuming a normal value.",
                colour=theme.STATUS["warning"],
            )

    ui.chart_panel(
        f"Vital signs · last {len(engine.history(patient_id))} observations",
        charts.vitals_facets(engine.history(patient_id), spo2_scale=bed.patient.spo2_scale),
        note="One panel per channel, each on its own scale. The green band is the NEWS2 "
        "zero-score range for that parameter.",
        fallback="No observations recorded yet.",
    )

    left, right = st.columns(2)
    with left:
        ui.chart_panel(
            "NEWS2 parameter scores",
            charts.news2_breakdown(bed.assessment.news2.components)
            if bed.assessment.news2
            else None,
            note="A red outline marks a red score — a single parameter at 3, which warrants "
            "review on its own regardless of the total.",
            fallback="NEWS2 could not be scored for this observation.",
        )
    with right:
        ui.chart_panel(
            "Fusion contributions",
            charts.factor_bars(bed.assessment.factors),
            note="These points reconcile to the composite score above, including any "
            "override lift.",
            fallback="No contributing factors recorded.",
        )

    prob_col, ctrl_col = st.columns([0.45, 0.55])
    with prob_col:
        ui.chart_panel(
            "Model class probabilities",
            # With no artefact the fusion layer still returns the class map, filled with
            # zeros, so passing it through drew three 0% bars underneath a note saying
            # nothing was loaded. Withhold it unless a model actually produced it, and the
            # panel falls back to saying so in words.
            charts.probability_bars(
                bed.assessment.ml_probabilities if bed.assessment.model_available else {}
            ),
            note=(
                f"{engine.model_version or 'no artefact'} · confidence "
                f"{bed.assessment.ml_confidence:.0%}"
                if bed.assessment.model_available
                else "No trained artefact loaded — scoring runs on NEWS2 and vision alone."
            ),
            fallback="No model prediction for this bed.",
        )
        _vision_panel(snapshot, patient_id)
    with ctrl_col:
        _controls(patient_id)

    patient_alerts = [a for a in engine.alerts.history if a.patient_id == patient_id]
    st.markdown("#### Alerts for this bed")
    if not patient_alerts:
        ui.empty_state("No alerts raised for this patient.", icon="✓")
    else:
        for alert in patient_alerts[:12]:
            ui.alert_card(
                alert,
                on_acknowledge=lambda aid: engine.alerts.acknowledge(aid, by="dashboard"),
                key_prefix="patient_alert",
                show_patient=False,
            )


__all__ = ["render"]
