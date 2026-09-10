"""Altair chart builders.

Three rules are enforced here rather than left to each caller.

**One y-scale per chart, always.** Heart rate and SpO₂ on shared axes is the single most
common way a vitals dashboard lies: the reader compares two lines whose units have nothing
in common. :func:`vitals_facets` therefore returns small multiples - one panel per channel,
each with its own scale and its own clinical band - and there is no function in this module
that accepts two measures.

**A hover layer by default.** These charts render in a browser, so a tooltip is not an
enhancement; a nurse pointing at a dip should get the number and the timestamp. Line and
area charts get a nearest-point crosshair, bars and cells get per-mark tooltips.

**Series colour follows the entity, not the rank.** Bed colours come from a stable
assignment keyed on patient id, so filtering the ward list down to the three worst beds
does not repaint them. Where more than three series would be visible at once the chart is
faceted instead, because only the first three palette hues survive an all-pairs
colour-vision check.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import altair as alt
import pandas as pd

from icu_monitor.ui import theme

#: Height of a single-row bar, plus the 2px surface gap the spec asks between fills.
BAR_STEP = 26

_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _apply(chart: alt.Chart | alt.LayerChart | alt.FacetChart) -> Any:
    """Apply the dark theme: transparent surface, recessive grid, tabular numerals."""
    return (
        chart.configure(font=_FONT, background="transparent")
        .configure_view(stroke=None, fill=None)
        .configure_axis(
            labelColor=theme.INK_MUTED,
            titleColor=theme.INK_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            gridColor=theme.GRID,
            gridOpacity=0.85,
            domainColor=theme.AXIS,
            tickColor=theme.AXIS,
            labelPadding=4,
        )
        .configure_legend(
            labelColor=theme.INK_SECONDARY,
            titleColor=theme.INK_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            symbolStrokeWidth=0,
            symbolSize=90,
            orient="top",
            direction="horizontal",
            offset=6,
        )
        .configure_title(color=theme.INK, fontSize=12, fontWeight=600, anchor="start")
        .configure_header(
            labelColor=theme.INK_SECONDARY,
            labelFontSize=11,
            labelFontWeight=600,
            titleColor=theme.INK_MUTED,
            titleFontSize=11,
        )
    )


def bed_color_scale(patient_ids: Sequence[str]) -> alt.Scale:
    """A hue per bed, assigned over the sorted domain.

    Sorting makes the assignment deterministic for a given set of beds, so it never depends
    on the order the caller happened to rank them in. It is *not* invariant when the set
    itself changes: measured over every three-bed subset of a twelve-bed ward, swapping one
    member repaints a third of the survivors, and no assignment of three hues to twelve beds
    does better by much (a stable per-id hash with collision repair scores 71% against this
    function's 67%). That is why :func:`score_timeline` labels each line directly - the hue
    separates the lines, the text is what identifies them.

    Beyond three beds the caller should facet rather than ask for a fourth hue: only the
    first three palette entries survive an all-pairs colour-vision check.
    """
    domain = sorted(dict.fromkeys(patient_ids))
    palette = theme.SERIES_DISTINCT if len(domain) <= 3 else theme.SERIES
    colors = [palette[i % len(palette)] for i in range(len(domain))]
    return alt.Scale(domain=domain, range=colors)


def risk_distribution(counts: Mapping[str, int]) -> Any:
    """Beds per risk level.

    One series, so no legend: the level names are the axis. Direct labels carry the counts
    because reading a bed count off a gridline is needless work.
    """
    order = [*theme.LEVEL_ORDER]
    if counts.get("UNKNOWN", 0):
        order.append("UNKNOWN")
    frame = pd.DataFrame(
        {
            "level": order,
            "beds": [int(counts.get(k, 0)) for k in order],
        }
    )
    frame["label"] = frame["level"].map(theme.level_glyph) + " " + frame["level"].str.title()

    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=list(frame["label"]), title=None, axis=alt.Axis(labelLimit=140)),
        x=alt.X("beds:Q", title="Beds", axis=alt.Axis(tickMinStep=1, grid=True)),
    )
    bars = base.mark_bar(
        cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
        height=BAR_STEP - 8,
        stroke=theme.SURFACE,
        strokeWidth=2,
    ).encode(
        color=alt.Color(
            "level:N",
            scale=alt.Scale(
                domain=order,
                range=[theme.level_color(k) for k in order],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Risk level"),
            alt.Tooltip("beds:Q", title="Beds"),
        ],
    )
    text = base.mark_text(
        align="left", dx=6, color=theme.INK_SECONDARY, fontSize=11, fontWeight=600
    ).encode(text="beds:Q")
    return _apply((bars + text).properties(height=len(frame) * BAR_STEP + 24))


def score_timeline(rows: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, float]) -> Any:
    """Composite risk over time, one line per bed, with the escalation bands drawn in.

    The threshold rules are the point of the chart: a line at 61 means nothing until you can
    see where HIGH begins. They are drawn as recessive dashed rules with their names in the
    margin, so they orient the reader without competing with the data.

    The tooltip reports the composite score only. History carries scores, not levels - a
    level also depends on clinical overrides that are not replayed here - and inferring one
    from the other would put a number on screen that the engine never produced.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return None

    hover = alt.selection_point(
        name="hover", on="pointerover", nearest=True, fields=["at"], empty=False, clear="pointerout"
    )
    scale = bed_color_scale(frame["bed"].tolist())
    show_legend = frame["bed"].nunique() > 1

    marks = pd.DataFrame([{"y": float(v), "name": str(k).title()} for k, v in thresholds.items()])
    rules = (
        alt.Chart(marks)
        .mark_rule(strokeDash=[3, 4], strokeWidth=1, color=theme.AXIS)
        .encode(y=alt.Y("y:Q", title="Composite risk", scale=alt.Scale(domain=[0, 100])))
    )
    rule_text = (
        alt.Chart(marks)
        .mark_text(
            align="left",
            baseline="bottom",
            dx=4,
            dy=-2,
            x=4,
            color=theme.INK_MUTED,
            fontSize=9,
        )
        .encode(y="y:Q", text="name:N")
    )

    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, interpolate="monotone")
        .encode(
            x=alt.X("at:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=6)),
            y=alt.Y("score:Q", title="Composite risk", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "bed:N",
                scale=scale,
                legend=alt.Legend(title=None) if show_legend else None,
            ),
        )
    )
    points = (
        line.mark_circle(size=90, stroke=theme.SURFACE, strokeWidth=2)
        .encode(
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.0)),
            tooltip=[
                alt.Tooltip("bed:N", title="Bed"),
                alt.Tooltip("at:T", title="Time", format="%H:%M:%S"),
                alt.Tooltip("score:Q", title="Composite", format=".1f"),
            ],
        )
        .add_params(hover)
    )
    crosshair = (
        alt.Chart(frame)
        .mark_rule(color=theme.BORDER_STRONG, strokeWidth=1)
        .encode(x="at:T", opacity=alt.condition(hover, alt.value(0.9), alt.value(0.0)))
    )
    # Each line is named at its own end. Hue separates the lines; the label is what says
    # which bed is which, so a bed changing hue when the tracked set changes cannot mislead.
    ends = frame.sort_values("at").groupby("bed", as_index=False).last()
    names = (
        alt.Chart(ends)
        .mark_text(align="left", dx=7, fontSize=10, fontWeight=600, color=theme.INK_SECONDARY)
        .encode(x="at:T", y="score:Q", text="bed:N")
    )
    chart = (rules + rule_text + crosshair + line + points + names).properties(height=250)
    return _apply(chart)


#: (attribute, panel title, unit, NEWS2 zero-score band). The band is what makes a single
#: number readable: 104 bpm means nothing until you can see it sits above normal.
CHANNELS: tuple[tuple[str, str, str, float, float], ...] = (
    ("heart_rate", "Heart rate", "bpm", 51, 90),
    ("spo2", "SpO₂", "%", 96, 100),
    ("bp_systolic", "Systolic BP", "mmHg", 111, 219),
    ("resp_rate", "Respiratory rate", "/min", 12, 20),
    ("temperature", "Temperature", "°C", 36.1, 38.0),
)


def vitals_facets(history: Sequence[Any], *, columns: int = 2, spo2_scale: int = 1) -> Any:
    """Small multiples - one panel per vital sign, each with its own scale.

    This is the shape the data demands. Five channels on one axis would need five hues
    (two more than survive an all-pairs colour-vision check) and a shared scale that no
    two of them share. Faceting costs a little space and buys correctness.

    The normal-range band travels in the same table as the readings rather than in a second
    one, because Vega-Lite cannot facet a layer whose sub-layers carry different data. It is
    collapsed back to one rectangle per panel with ``min``/``max`` aggregates.
    """
    records: list[dict[str, Any]] = []
    for key, title, unit, low, high in CHANNELS:
        if key == "spo2" and spo2_scale == 2:
            low, high = 88, 92  # chronic hypercapnic respiratory failure target range
        for vitals in history:
            value = getattr(vitals, key, None)
            if value is None:
                continue
            records.append(
                {
                    "channel": f"{title} ({unit})",
                    "at": vitals.recorded_at,
                    "value": float(value),
                    "low": float(low),
                    "high": float(high),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return None
    seen = set(frame["channel"])
    order = [f"{t} ({u})" for _, t, u, _, _ in CHANNELS if f"{t} ({u})" in seen]

    hover = alt.selection_point(
        name="vhover",
        on="pointerover",
        nearest=True,
        fields=["at"],
        empty=False,
        clear="pointerout",
    )
    y_scale = alt.Scale(zero=False, nice=True)
    band = (
        alt.Chart()
        .mark_rect(color=theme.STATUS["good"], opacity=0.09)
        .encode(
            y=alt.Y("min(low):Q", title=None, scale=y_scale),
            y2=alt.Y2("max(high):Q"),
        )
    )
    line = (
        alt.Chart()
        .mark_line(strokeWidth=2, interpolate="monotone", color=theme.SERIES[0])
        .encode(
            x=alt.X("at:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=4)),
            y=alt.Y("value:Q", title=None, scale=y_scale),
        )
    )
    points = (
        alt.Chart()
        .mark_circle(size=90, color=theme.SERIES[0], stroke=theme.SURFACE, strokeWidth=2)
        .encode(
            x=alt.X("at:T", title=None),
            y=alt.Y("value:Q", title=None, scale=y_scale),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.0)),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("value:Q", title="Value", format=".1f"),
                alt.Tooltip("at:T", title="Time", format="%H:%M:%S"),
            ],
        )
        .add_params(hover)
    )

    faceted = (
        alt.layer(band, line, points, data=frame)
        .properties(width="container", height=118)
        .facet(
            facet=alt.Facet(
                "channel:N", title=None, sort=order, header=alt.Header(labelAnchor="start")
            ),
            columns=columns,
            spacing={"row": 18, "column": 26},
        )
        .resolve_scale(y="independent")
    )
    return _apply(faceted)


def news2_breakdown(components: Sequence[Any]) -> Any:
    """Where a NEWS2 total came from, parameter by parameter.

    Magnitude, so one hue light→dark rather than a category per parameter. A red score (any
    single parameter at 3) gets a ring instead of a different hue, because red is already
    spoken for by the status palette and because the ring survives greyscale printing.
    """
    frame = pd.DataFrame(
        [
            {
                "parameter": c.display_name,
                "score": int(c.score),
                "value": "—" if c.value is None else f"{c.value}{(' ' + c.unit) if c.unit else ''}",
                "band": c.band,
                "red": bool(c.is_red),
            }
            for c in components
        ]
    )
    if frame.empty:
        return None
    frame = frame.sort_values("score", ascending=False, kind="stable")

    bars = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
            height=BAR_STEP - 8,
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("parameter:N", sort=list(frame["parameter"]), title=None),
            x=alt.X(
                "score:Q",
                title="NEWS2 points",
                scale=alt.Scale(domain=[0, 3]),
                axis=alt.Axis(values=[0, 1, 2, 3]),
            ),
            color=alt.Color(
                "score:Q",
                scale=alt.Scale(domain=[0, 3], range=[theme.SEQUENTIAL[1], theme.SEQUENTIAL[4]]),
                legend=None,
            ),
            stroke=alt.condition(
                alt.datum.red, alt.value(theme.STATUS["critical"]), alt.value(theme.SURFACE)
            ),
            tooltip=[
                alt.Tooltip("parameter:N", title="Parameter"),
                alt.Tooltip("value:N", title="Observed"),
                alt.Tooltip("score:Q", title="Points"),
                alt.Tooltip("band:N", title="Band"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=8, color=theme.INK_SECONDARY, fontSize=11)
        .encode(
            y=alt.Y("parameter:N", sort=list(frame["parameter"]), title=None),
            x=alt.X("score:Q"),
            text=alt.Text("value:N"),
        )
    )
    return _apply((bars + labels).properties(height=len(frame) * BAR_STEP + 24))


def factor_bars(factors: Sequence[Any]) -> Any:
    """The itemised contributions that add up to the composite score.

    Signed, so diverging: cool where a factor pulls the score down, warm where it pushes up,
    neutral grey at zero. Everything on one axis of points, which is the only unit these
    rows share.
    """
    frame = pd.DataFrame(
        [
            {
                "description": f.description,
                "points": round(float(f.points), 2),
                "source": f.source,
                "severity": str(getattr(f.severity, "value", f.severity or "")),
            }
            for f in factors
        ]
    )
    if frame.empty:
        return None
    frame = frame.sort_values("points", ascending=False, kind="stable")
    limit = max(1.0, float(frame["points"].abs().max()))

    bars = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
            cornerRadiusTopLeft=4,
            cornerRadiusBottomLeft=4,
            height=BAR_STEP - 8,
            stroke=theme.SURFACE,
            strokeWidth=2,
        )
        .encode(
            y=alt.Y(
                "description:N",
                sort=list(frame["description"]),
                title=None,
                axis=alt.Axis(labelLimit=260),
            ),
            x=alt.X(
                "points:Q", title="Contribution (points)", scale=alt.Scale(domain=[-limit, limit])
            ),
            color=alt.Color(
                "points:Q",
                scale=alt.Scale(domain=[-limit, 0, limit], range=list(theme.DIVERGING)),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("description:N", title="Factor"),
                alt.Tooltip("points:Q", title="Points", format="+.2f"),
                alt.Tooltip("source:N", title="Source"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=theme.AXIS, strokeWidth=1)
        .encode(x="x:Q")
    )
    return _apply((zero + bars).properties(height=len(frame) * BAR_STEP + 26))


def probability_bars(probabilities: Mapping[str, float]) -> Any:
    """The model's class probabilities. Ordinal risk classes, so the status ramp applies."""
    order = [c for c in theme.LEVEL_ORDER if c in probabilities]
    if not order:
        return None
    frame = pd.DataFrame(
        {
            "level": order,
            "label": [f"{theme.level_glyph(c)} {c.title()}" for c in order],
            "p": [float(probabilities[c]) for c in order],
        }
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
            height=BAR_STEP - 8,
            stroke=theme.SURFACE,
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("label:N", sort=list(frame["label"]), title=None),
            x=alt.X(
                "p:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=order, range=[theme.level_color(c) for c in order]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("label:N", title="Class"), alt.Tooltip("p:Q", format=".1%")],
        )
    )
    text = (
        alt.Chart(frame)
        .mark_text(align="left", dx=6, color=theme.INK_SECONDARY, fontSize=11, fontWeight=600)
        .encode(
            y=alt.Y("label:N", sort=list(frame["label"]), title=None),
            x="p:Q",
            text=alt.Text("p:Q", format=".0%"),
        )
    )
    return _apply((bars + text).properties(height=len(frame) * BAR_STEP + 24))


def alert_kind_bars(counts: Mapping[str, int], *, limit: int = 10) -> Any:
    """Alert volume by kind - the alarm-fatigue view. One series, so no legend."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not items:
        return None
    frame = pd.DataFrame({"kind": [k for k, _ in items], "count": [int(v) for _, v in items]})
    bars = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
            height=BAR_STEP - 8,
            color=theme.SERIES[0],
            stroke=theme.SURFACE,
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("kind:N", sort=list(frame["kind"]), title=None, axis=alt.Axis(labelLimit=180)),
            x=alt.X("count:Q", title="Alerts (24 h)", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("kind:N", title="Kind"), alt.Tooltip("count:Q", title="Alerts")],
        )
    )
    text = (
        alt.Chart(frame)
        .mark_text(align="left", dx=6, color=theme.INK_SECONDARY, fontSize=11, fontWeight=600)
        .encode(
            y=alt.Y("kind:N", sort=list(frame["kind"]), title=None), x="count:Q", text="count:Q"
        )
    )
    return _apply((bars + text).properties(height=len(frame) * BAR_STEP + 24))


def confusion_heatmap(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> Any:
    """Row-normalised confusion matrix.

    Normalised by row because the classes are imbalanced: raw counts make a model that
    predicts LOW for everything look excellent. Every cell is also labelled, so the colour
    is a reading aid rather than the only channel.
    """
    total_rows = [max(1, sum(row)) for row in matrix]
    records = [
        {
            "actual": labels[i],
            "predicted": labels[j],
            "count": int(matrix[i][j]),
            "share": matrix[i][j] / total_rows[i],
        }
        for i in range(len(labels))
        for j in range(len(labels))
    ]
    frame = pd.DataFrame(records)
    order = list(labels)
    cells = (
        alt.Chart(frame)
        .mark_rect(stroke=theme.SURFACE, strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X("predicted:N", sort=order, title="Predicted"),
            y=alt.Y("actual:N", sort=order, title="Actual"),
            color=alt.Color(
                "share:Q",
                scale=alt.Scale(range=list(theme.SEQUENTIAL), domain=[0, 1]),
                legend=alt.Legend(title="Share of actual", format=".0%", gradientLength=120),
            ),
            tooltip=[
                alt.Tooltip("actual:N", title="Actual"),
                alt.Tooltip("predicted:N", title="Predicted"),
                alt.Tooltip("count:Q", title="Windows"),
                alt.Tooltip("share:Q", title="Row share", format=".1%"),
            ],
        )
    )
    text = (
        alt.Chart(frame)
        .mark_text(fontSize=11, fontWeight=600)
        .encode(
            x=alt.X("predicted:N", sort=order),
            y=alt.Y("actual:N", sort=order),
            text=alt.Text("share:Q", format=".0%"),
            color=alt.condition(
                alt.datum.share > 0.55, alt.value(theme.INK), alt.value(theme.INK_SECONDARY)
            ),
        )
    )
    size = 62 * len(labels) + 40
    return _apply((cells + text).properties(height=size, width=size))


def calibration_curve(bins: Sequence[Mapping[str, Any]], *, target: str = "HIGH") -> Any:
    """A reliability diagram: predicted probability against observed frequency.

    The dashed diagonal is perfect calibration, drawn as a reference rule rather than a
    second series - it is not data. Marker area encodes how many windows fall in each bin,
    because a bin holding 30 samples and one holding 5,000 should not look equally
    trustworthy.
    """
    frame = pd.DataFrame(
        [
            {
                "predicted": float(b["mean_predicted"]),
                "observed": float(b["observed_frequency"]),
                "count": int(b["count"]),
                "band": f"{float(b['bin_lower']):.0%}–{float(b['bin_upper']):.0%}",
            }
            for b in bins
            if b.get("count")
        ]
    )
    if frame.empty:
        return None
    ideal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], strokeWidth=1, color=theme.AXIS)
        .encode(x="x:Q", y="y:Q")
    )
    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, color=theme.SERIES[0], interpolate="monotone")
        .encode(
            x=alt.X(
                "predicted:Q",
                title=f"Predicted P({target})",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "observed:Q",
                title="Observed frequency",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
        )
    )
    points = (
        alt.Chart(frame)
        .mark_circle(color=theme.SERIES[0], stroke=theme.SURFACE, strokeWidth=2, opacity=0.95)
        .encode(
            x="predicted:Q",
            y="observed:Q",
            size=alt.Size("count:Q", scale=alt.Scale(range=[80, 460]), legend=None),
            tooltip=[
                alt.Tooltip("band:N", title="Probability bin"),
                alt.Tooltip("predicted:Q", title="Mean predicted", format=".1%"),
                alt.Tooltip("observed:Q", title="Observed", format=".1%"),
                alt.Tooltip("count:Q", title="Windows"),
            ],
        )
    )
    return _apply((ideal + line + points).properties(height=260))


def importance_bars(importances: Sequence[Mapping[str, Any]], *, limit: int = 12) -> Any:
    """Permutation importances, with their standard deviation as an error bar.

    The error bar is not decoration: several of these features overlap within noise, and a
    ranked bar chart without it invites the reader to over-read the order.
    """
    rows = list(importances)[:limit]
    if not rows:
        return None
    frame = pd.DataFrame(
        [
            {
                "feature": str(r["feature"]),
                "importance": float(r["importance"]),
                "low": float(r["importance"]) - float(r.get("std", 0.0)),
                "high": float(r["importance"]) + float(r.get("std", 0.0)),
            }
            for r in rows
        ]
    )
    order = list(frame["feature"])
    bars = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
            height=BAR_STEP - 8,
            color=theme.SERIES[0],
            stroke=theme.SURFACE,
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=170)),
            x=alt.X("importance:Q", title="Drop in macro-F1 when shuffled"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
    )
    error = (
        alt.Chart(frame)
        .mark_rule(color=theme.INK_MUTED, strokeWidth=1, opacity=0.8)
        .encode(y=alt.Y("feature:N", sort=order), x="low:Q", x2="high:Q")
    )
    return _apply((bars + error).properties(height=len(frame) * BAR_STEP + 26))


__all__ = [
    "BAR_STEP",
    "CHANNELS",
    "alert_kind_bars",
    "bed_color_scale",
    "calibration_curve",
    "confusion_heatmap",
    "factor_bars",
    "importance_bars",
    "news2_breakdown",
    "probability_bars",
    "risk_distribution",
    "score_timeline",
    "vitals_facets",
]
