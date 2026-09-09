"""The dashboard: the three failures that make a UI worse than no UI.

**A card that looks right and does nothing.** v1 built its layout by emitting
``<div class="card">`` from ``st.markdown`` and then calling ``st.metric`` inside it.
Streamlit renders every element into its own DOM node, so the div closed before the widget
existed and the sanitiser discarded the rest - cards that rendered and could not be clicked.
Everything here goes through :class:`streamlit.testing.v1.AppTest`, which walks the real
element tree and runs real callbacks, so a button that is not really there fails a test.

**A chart that lies.** Two measures sharing one pair of axes, a legend as the only way to
tell series apart, a risk level carried by fill alone. These are checked as properties of
the generated Vega-Lite specs, and the colour claims are *computed*: the sequential ramp's
luminance ordering and the status palette's 3:1 floor against the dashboard surface are
asserted here rather than believed.

**A dashboard that only runs on the maintainer's machine.** Every view is exercised with no
camera, no trained artefact and no database, because that is what a bare clone gets.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

from icu_monitor.config import DetectorName, FrameSourceName, Settings, VitalsSourceName
from icu_monitor.core.news2 import RED_SCORE_RESPONSE, RESPONSE_BY_TOTAL
from icu_monitor.core.types import (
    Alert,
    AlertKind,
    BedSnapshot,
    ClinicalState,
    Consciousness,
    NEWS2Result,
    ParameterScore,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
)
from icu_monitor.ml import pipeline as pipeline_module
from icu_monitor.ml.train import train
from icu_monitor.monitoring.engine import MonitoringEngine
from icu_monitor.ui import app as app_module
from icu_monitor.ui import charts, theme
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state
from icu_monitor.ui.views import patient as patient_view
from icu_monitor.ui.views import settings as settings_view

from .conftest import EPOCH, make_patient, make_vitals, make_window_frame

# --------------------------------------------------------------------- computed colour facts


def relative_luminance(colour: str) -> float:
    """WCAG 2.1 relative luminance of a ``#rrggbb`` string.

    Present so the accessibility claims in this file are arithmetic rather than opinion: a
    palette edit that breaks the contrast floor fails a test instead of shipping.
    """
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    first, second = relative_luminance(foreground), relative_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def channel_spread(colour: str) -> int:
    """Largest minus smallest RGB channel - zero for a true grey."""
    channels = [int(colour[index : index + 2], 16) for index in (1, 3, 5)]
    return max(channels) - min(channels)


# --------------------------------------------------------------------------------- builders


def make_news2(
    total: int = 0,
    *,
    red: bool = False,
    scale: int = 1,
    missing: tuple[str, ...] = (),
) -> NEWS2Result:
    components = (
        ParameterScore("resp_rate", "Respiration rate", 16.0, "/min", 0, "12-20"),
        ParameterScore("spo2", "SpO₂", 98.0, "%", 0, "≥96"),
        ParameterScore("heart_rate", "Heart rate", 122.0, "bpm", 2, "111-130"),
        ParameterScore("temperature", "Temperature", 39.2, "°C", 2, "≥39.1"),
        ParameterScore("bp_systolic", "Systolic BP", 88.0, "mmHg", 3, "≤90", is_red=red),
    )
    return NEWS2Result(
        total=total,
        components=components,
        risk_level=RiskLevel.MEDIUM,
        clinical_response="Hourly observations; urgent review by a clinician.",
        has_red_score=red,
        missing_parameters=missing,
        scale=scale,
    )


def make_assessment(**overrides: Any) -> RiskAssessment:
    values: dict[str, Any] = {
        "patient_id": "P001",
        "level": RiskLevel.HIGH,
        "composite_score": 72.5,
        "ml_level": RiskLevel.MEDIUM,
        "ml_confidence": 0.61,
        "ml_probabilities": {"LOW": 0.14, "MEDIUM": 0.61, "HIGH": 0.25},
        "news2": make_news2(7),
        "vision": None,
        "factors": (
            RiskFactor("news2", "NEWS2 total 7 of 20", 24.0, "serious"),
            RiskFactor("model", "Model leans MEDIUM", 8.5, "warning"),
            RiskFactor("trend", "Composite falling over the last hour", -6.0, "info"),
        ),
        "overrides": (),
        "model_available": True,
        "assessed_at": EPOCH,
    }
    values.update(overrides)
    return RiskAssessment(**values)


def make_bed(patient_id: str = "P001", **overrides: Any) -> BedSnapshot:
    patient = overrides.pop("patient", None) or make_patient(patient_id)
    values: dict[str, Any] = {
        "patient": patient,
        "vitals": make_vitals(),
        "assessment": make_assessment(patient_id=patient.patient_id),
    }
    values.update(overrides)
    return BedSnapshot(**values)


def make_alert(**overrides: Any) -> Alert:
    values: dict[str, Any] = {
        "patient_id": "P001",
        "kind": AlertKind.HYPOXIA,
        "severity": RiskLevel.HIGH,
        "message": "SpO₂ 88% on room air",
        "created_at": EPOCH,
        "alert_id": 1,
    }
    values.update(overrides)
    return Alert(**values)


def history(count: int = 6, **overrides: Any) -> list[Any]:
    """A run of observations one minute apart, for the trend charts."""
    return [make_vitals(at=EPOCH + timedelta(minutes=index), **overrides) for index in range(count)]


# -------------------------------------------------------------------------------- harnesses


def ui_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """The configuration a bare clone produces: no camera, no detector, no database file."""
    values: dict[str, Any] = {
        "project_root": tmp_path,
        "database_url": "sqlite://",
        "bed_count": 4,
        "tick_seconds": 0.25,
        "simulation_seed": 424242,
        "frame_source": "off",
        "detector": "off",
        "api_warmup_ticks": 12,
        "audible_alerts": False,
        "api_key": None,
    }
    values.update(overrides)
    return Settings(**values)


def run_app(cfg: Settings, **session: Any) -> AppTest:
    """The whole dashboard, one script run, with the live rerun loop switched off.

    ``icu_live`` has to be false: with it on, ``main`` ends in ``time.sleep`` followed by
    ``st.rerun``, which is a live display in a browser and an unbounded loop under test.
    """
    app = AppTest.from_string(
        "from icu_monitor.ui.app import main\n\nmain()\n", default_timeout=120
    )
    app.session_state["icu_settings"] = cfg
    app.session_state["icu_live"] = False
    for key, value in session.items():
        app.session_state[key] = value
    return app.run()


def screen_text(app: AppTest) -> str:
    """Every string a run put on screen - markdown, callouts, metrics and button labels.

    Wording assertions go through this rather than one element list, because which Streamlit
    primitive carries a message is an implementation detail; whether the reader sees it is not.
    """
    chunks = [
        str(element.value)
        for element in (*app.markdown, *app.warning, *app.error, *app.success, *app.info)
    ]
    chunks += [f"{metric.label} {metric.value}" for metric in app.metric]
    chunks += [button.label for button in app.button]
    return "\n".join(chunks)


# =============================================================================== the palette


def test_only_the_three_all_pairs_safe_hues_are_offered_as_distinct() -> None:
    """``SERIES_DISTINCT`` is a promise: every member is tellable from every *other* member,
    not just from its neighbour in the list. Only the first three hues pass that check, so a
    fourth simultaneous series has to be faceted rather than coloured."""
    assert theme.SERIES[:3] == theme.SERIES_DISTINCT
    assert len(set(theme.SERIES)) == len(theme.SERIES)


def test_every_risk_level_carries_a_shape_as_well_as_a_colour() -> None:
    """The status palette's worst colour-vision pair is only legal with a second channel.
    Two levels sharing a glyph would collapse that channel exactly where it is needed."""
    glyphs = [glyph for _, glyph in theme.LEVEL_STYLE.values()]
    colours = [colour for colour, _ in theme.LEVEL_STYLE.values()]
    assert len(set(glyphs)) == len(glyphs)
    assert len(set(colours)) == len(colours)
    assert all(glyph.strip() for glyph in glyphs)


@pytest.mark.parametrize("level", ["low", "LOW", "Low", RiskLevel.LOW.value])
def test_a_level_lookup_tolerates_however_the_caller_spells_it(level: str) -> None:
    assert theme.level_color(level) == theme.STATUS["good"]
    assert theme.level_glyph(level) == "●"


@pytest.mark.parametrize("level", ["", "MODERATE", "nonsense", "None"])
def test_an_unrecognised_level_renders_as_unknown_rather_than_low(level: str) -> None:
    """Falling back to the first entry would paint a level the engine never produced in the
    colour that means 'fine'. ``UNKNOWN`` says what is actually true."""
    assert theme.level_color(level) == theme.STATUS["muted"]
    assert theme.level_glyph(level) == "○"


def test_the_sequential_ramp_darkens_monotonically() -> None:
    """A magnitude ramp is read by lightness. One step out of order and the reader cannot
    tell which end is 'more' - the property is computed here, not eyeballed."""
    luminances = [relative_luminance(step) for step in theme.SEQUENTIAL]
    assert luminances == sorted(luminances, reverse=True)
    assert all(earlier - later > 0.02 for earlier, later in pairwise(luminances))


def test_the_diverging_midpoint_is_neutral_and_the_poles_are_not() -> None:
    """A hue at the midpoint reads as a third category and destroys the polarity encoding."""
    cool, middle, warm = theme.DIVERGING
    assert channel_spread(middle) < 40
    assert channel_spread(cool) > 120
    assert channel_spread(warm) > 120


@pytest.mark.parametrize("role", ["good", "warning", "serious", "critical", "muted"])
def test_every_status_colour_clears_three_to_one_on_the_dashboard_surface(role: str) -> None:
    """WCAG 1.4.11 asks 3:1 of a graphical object against its background. ``critical`` sits at
    3.08 - the tightest of the five - so this assertion is load-bearing: darkening the red to
    make it feel more urgent is exactly the edit that would break it."""
    assert contrast_ratio(theme.STATUS[role], theme.SURFACE) >= 3.0


@pytest.mark.parametrize("ink", ["INK", "INK_SECONDARY", "INK_MUTED"])
def test_every_ink_token_clears_the_text_contrast_floor(ink: str) -> None:
    """4.5:1 for body text on both surfaces a card can sit on - ``INK_MUTED`` carries captions
    and axis labels, which are text however small they are."""
    colour = getattr(theme, ink)
    assert contrast_ratio(colour, theme.SURFACE) >= 4.5
    assert contrast_ratio(colour, theme.SURFACE_RAISED) >= 4.5


@pytest.mark.parametrize("hue", theme.SERIES)
def test_every_series_hue_clears_three_to_one_on_the_surface(hue: str) -> None:
    assert contrast_ratio(hue, theme.SURFACE) >= 3.0


def test_the_status_palette_is_reserved() -> None:
    """A status colour reused as 'series 4' makes an ordinary line look like a warning. The
    two sets are kept disjoint so that cannot happen by accident."""
    assert not set(theme.STATUS.values()) & set(theme.SERIES)


def test_the_level_ladder_excludes_unknown_but_the_style_map_covers_it() -> None:
    """``LEVEL_ORDER`` is a chart domain - an axis with a 'No data' band on it is noise. But a
    bed with no assessment still has to render, so the *style* map keeps ``UNKNOWN``."""
    assert theme.LEVEL_ORDER == ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "UNKNOWN" not in theme.LEVEL_ORDER
    assert set(theme.LEVEL_STYLE) == {*theme.LEVEL_ORDER, "UNKNOWN"}
    assert (
        tuple(level.value for level in RiskLevel if level is not RiskLevel.UNKNOWN)
        == theme.LEVEL_ORDER
    )


def test_the_stylesheet_ships_inside_the_package() -> None:
    """``load_css`` reads through ``importlib.resources`` so the dashboard is styled from a
    wheel or a container, not only from a source checkout."""
    css = theme.load_css()
    assert len(css) > 500
    assert theme.PAGE_BG in css
    assert "<script" not in css.lower()


# ================================================================================ the charts


def rows_of(spec: dict[str, Any], layer: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The rows a layer actually plots.

    Altair hoists shared data to the top level and names it, so a layer carries its own
    ``data`` key only when it differs. Following the reference rather than assuming either
    shape is the difference between testing the chart and testing Altair's serialiser.
    """
    reference = (layer or {}).get("data") or spec.get("data")
    assert reference is not None, "neither the layer nor the spec names a dataset"
    return list(spec["datasets"][reference["name"]])


def marks_of(spec: dict[str, Any]) -> list[str]:
    return [
        layer["mark"] if isinstance(layer["mark"], str) else layer["mark"]["type"]
        for layer in spec["layer"]
    ]


EMPTY_CALLS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
    ("score_timeline", ([],), {"thresholds": {"high": 61.0}}),
    ("vitals_facets", ([],), {}),
    ("news2_breakdown", ([],), {}),
    ("factor_bars", ([],), {}),
    ("probability_bars", ({},), {}),
    ("alert_kind_bars", ({},), {}),
    ("calibration_curve", ([],), {}),
    ("importance_bars", ([],), {}),
)


@pytest.mark.parametrize(
    ("name", "args", "kwargs"), EMPTY_CALLS, ids=[call[0] for call in EMPTY_CALLS]
)
def test_a_builder_with_nothing_to_draw_returns_none(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """``chart_panel`` prints its fallback sentence when the chart is ``None``. A builder that
    returned an empty chart instead would draw axes around no data, which a reader takes as
    'measured zero' rather than 'nothing recorded yet'."""
    assert getattr(charts, name)(*args, **kwargs) is None


def test_a_bin_holding_no_windows_is_dropped_rather_than_drawn_at_zero() -> None:
    """An empty probability bin has no observed frequency. Plotting it at 0 invents a point
    that says the model was perfectly wrong there."""
    bins = [
        {
            "bin_lower": 0.0,
            "bin_upper": 0.2,
            "mean_predicted": 0.1,
            "observed_frequency": 0.0,
            "count": 0,
        }
    ]
    assert charts.calibration_curve(bins) is None


def test_the_risk_ladder_is_drawn_even_when_no_bed_is_at_any_level() -> None:
    """The deliberate exception to the rule above: here the levels *are* the axis, so an empty
    ward is four zero-length bars - which is information - not a missing panel."""
    spec = charts.risk_distribution({}).to_dict()
    rows = rows_of(spec)
    assert [row["level"] for row in rows] == list(theme.LEVEL_ORDER)
    assert {row["beds"] for row in rows} == {0}
    assert all(theme.level_glyph(row["level"]) in row["label"] for row in rows)


@pytest.fixture(scope="module")
def every_chart() -> dict[str, Any]:
    """One populated instance of every builder, for the invariants that hold across all of
    them. Module-scoped because these specs are the most expensive objects in this file."""
    beds = ["BED-01", "BED-02"]
    rows = [
        {"bed": bed, "at": EPOCH + timedelta(minutes=index), "score": 40.0 + index}
        for bed in beds
        for index in range(4)
    ]
    return {
        "risk_distribution": charts.risk_distribution({"HIGH": 2, "LOW": 1}),
        "score_timeline": charts.score_timeline(rows, thresholds={"high": 61.0}),
        "vitals_facets": charts.vitals_facets(history()),
        "news2_breakdown": charts.news2_breakdown(make_news2(7, red=True).components),
        "factor_bars": charts.factor_bars(make_assessment().factors),
        "probability_bars": charts.probability_bars({"LOW": 0.2, "HIGH": 0.8}),
        "alert_kind_bars": charts.alert_kind_bars({"Hypoxaemia": 4, "Bed exit": 1}),
        "confusion_heatmap": charts.confusion_heatmap(
            [[8, 2, 0], [1, 5, 1], [0, 1, 3]], ["LOW", "MEDIUM", "HIGH"]
        ),
        "calibration_curve": charts.calibration_curve(
            [
                {
                    "bin_lower": 0.0,
                    "bin_upper": 0.5,
                    "mean_predicted": 0.25,
                    "observed_frequency": 0.3,
                    "count": 40,
                }
            ]
        ),
        "importance_bars": charts.importance_bars(
            [{"feature": "hr_mean", "importance": 0.08, "std": 0.01}]
        ),
    }


CHART_NAMES: tuple[str, ...] = (
    "risk_distribution",
    "score_timeline",
    "vitals_facets",
    "news2_breakdown",
    "factor_bars",
    "probability_bars",
    "alert_kind_bars",
    "confusion_heatmap",
    "calibration_curve",
    "importance_bars",
)


@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_builder_produces_a_chart_altair_can_serialise(
    every_chart: dict[str, Any], name: str
) -> None:
    """``to_dict`` is where a malformed encoding surfaces, and ``st.altair_chart`` calls it.
    A chart that only fails at render time fails in the browser, where no test is looking."""
    spec = every_chart[name].to_dict()
    assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert "layer" in spec or "spec" in spec


@pytest.mark.parametrize("name", CHART_NAMES)
def test_no_layered_chart_resolves_its_y_scale_independently(
    every_chart: dict[str, Any], name: str
) -> None:
    """**The dual-axis guard.** Two measures on one pair of axes is the single most misleading
    thing a chart can do, and in Vega-Lite it is spelled exactly one way: an independent ``y``
    resolution on a *layered* spec. Faceting is the legitimate use of the same key - each panel
    is its own chart with its own axis - so the exemption is granted to facets alone."""
    spec = every_chart[name].to_dict()
    resolved = (spec.get("resolve") or {}).get("scale", {})
    if "facet" in spec:
        assert resolved == {"y": "independent"}
    else:
        assert resolved.get("y") != "independent"


def test_the_bed_palette_is_assigned_over_a_sorted_deduplicated_domain() -> None:
    """Sorting is what makes the assignment independent of the order the caller ranked the
    beds in; two rows for one bed must not consume two hues."""
    scale = charts.bed_color_scale(["BED-03", "BED-01", "BED-03", "BED-02"]).to_dict()
    assert scale["domain"] == ["BED-01", "BED-02", "BED-03"]
    assert scale["range"] == list(theme.SERIES_DISTINCT)


def test_a_fourth_bed_reaches_past_the_all_pairs_safe_subset() -> None:
    """Four simultaneous lines is already past what colour can carry, so the caller is meant
    to facet. The builder widens to the full ramp rather than repeating a hue, because two
    lines in the same colour is the one outcome with no reading at all."""
    scale = charts.bed_color_scale([f"BED-0{n}" for n in range(1, 5)]).to_dict()
    assert scale["range"] == list(theme.SERIES[:4])
    assert len(set(scale["range"])) == 4


def test_a_bed_can_change_colour_when_the_tracked_set_changes() -> None:
    """Pinned as behaviour because the docstring used to claim the opposite. Three hues over
    twelve beds cannot be stable - measured across every three-bed subset of a twelve-bed
    ward, swapping one member repaints a third of the survivors - which is precisely why
    :func:`charts.score_timeline` labels every line directly. Should this assertion ever
    start failing because the assignment became stable, the direct labels become a
    convenience; until then they are what carries identity."""
    before = charts.bed_color_scale(["BED-02", "BED-03", "BED-04"]).to_dict()
    after = charts.bed_color_scale(["BED-01", "BED-02", "BED-03"]).to_dict()
    was = dict(zip(before["domain"], before["range"], strict=True))
    now = dict(zip(after["domain"], after["range"], strict=True))
    assert was["BED-02"] != now["BED-02"]
    assert len(set(now.values())) == 3  # whatever moved, no two beds ever share a hue


def timeline_spec(beds: tuple[str, ...] = ("BED-01", "BED-02")) -> dict[str, Any]:
    rows = [
        {"bed": bed, "at": EPOCH + timedelta(minutes=index), "score": 30.0 + 10 * index}
        for bed in beds
        for index in range(4)
    ]
    chart = charts.score_timeline(rows, thresholds={"medium": 41.0, "high": 61.0, "critical": 81.0})
    return chart.to_dict()


def test_the_timeline_draws_thresholds_under_the_data_and_labels_over_it() -> None:
    """Draw order is the whole readability argument: reference rules behind, hover crosshair
    behind the line, the bed names on top where nothing can occlude them."""
    assert marks_of(timeline_spec()) == ["rule", "text", "rule", "line", "circle", "text"]


def test_every_escalation_threshold_is_drawn_and_named() -> None:
    """A composite score of 61 means nothing without the band it crosses. Names in the margin
    because a reader should not have to consult the Settings page to read this chart."""
    spec = timeline_spec()
    rules = rows_of(spec, spec["layer"][0])
    assert {row["y"] for row in rules} == {41.0, 61.0, 81.0}
    assert {row["name"] for row in rules} == {"Medium", "High", "Critical"}
    assert spec["layer"][1]["encoding"]["text"]["field"] == "name"


def test_the_timeline_shares_one_y_axis_across_every_bed() -> None:
    """Composite risk is one measure in one unit; per-bed scales would make two lines at the
    same height mean different numbers."""
    spec = timeline_spec()
    domains = [
        layer["encoding"]["y"]["scale"]["domain"]
        for layer in spec["layer"]
        if "scale" in layer["encoding"].get("y", {})
    ]
    assert domains and all(domain == [0, 100] for domain in domains)
    assert "resolve" not in spec


def test_every_line_is_named_at_its_own_end() -> None:
    """One text row per bed, positioned at that bed's last observation. This is the layer that
    makes the hue reassignment above harmless."""
    spec = timeline_spec(("BED-01", "BED-02", "BED-03"))
    labels = spec["layer"][5]
    assert labels["encoding"]["text"]["field"] == "bed"
    ends = rows_of(spec, labels)
    assert sorted(row["bed"] for row in ends) == ["BED-01", "BED-02", "BED-03"]
    assert {row["score"] for row in ends} == {60.0}  # the last point of each series


def test_one_bed_needs_no_legend_and_two_beds_do() -> None:
    """A legend for a single series is a box that repeats the title. Two series without one is
    a chart that cannot be read at all - so the rule is conditional, and both halves matter."""
    single = timeline_spec(("BED-01",))
    assert single["layer"][3]["encoding"]["color"].get("legend") is None
    pair = timeline_spec()
    assert pair["layer"][3]["encoding"]["color"]["legend"] == {"title": None}


def test_the_timeline_tooltip_reports_the_score_and_never_a_level() -> None:
    """History carries scores, not levels: a level also depends on clinical overrides that are
    not replayed here, so deriving one would put a number on screen the engine never produced."""
    titles = [tip["title"] for tip in timeline_spec()["layer"][4]["encoding"]["tooltip"]]
    assert titles == ["Bed", "Time", "Composite"]


def test_each_vital_sign_gets_its_own_panel_in_bedside_reading_order() -> None:
    """Five channels on one axis would need five hues - two more than survive an all-pairs
    colour-vision check - and a shared scale that none of them share. The panel order follows
    ``CHANNELS`` because that is the order a bedside chart is read in."""
    spec = charts.vitals_facets(history()).to_dict()
    expected = [f"{title} ({unit})" for _, title, unit, _, _ in charts.CHANNELS]
    assert spec["facet"]["field"] == "channel"
    assert spec["facet"]["sort"] == expected
    assert {row["channel"] for row in rows_of(spec)} == set(expected)
    assert spec["resolve"] == {"scale": {"y": "independent"}}


def test_a_channel_the_monitor_never_reported_gets_no_empty_panel() -> None:
    """An axis with no line on it reads as a flat trace at the bottom of the range. Absent is
    absent: the panel is not drawn, and the vitals grid says the channel was not measured."""
    spec = charts.vitals_facets(history(temperature=None, spo2=None)).to_dict()
    present = {row["channel"] for row in rows_of(spec)}
    assert not any("Temperature" in channel or "SpO" in channel for channel in present)
    assert spec["facet"]["sort"] == [
        f"{title} ({unit})"
        for _, title, unit, _, _ in charts.CHANNELS
        if f"{title} ({unit})" in present
    ]


def test_the_normal_band_follows_the_patients_oxygen_target() -> None:
    """Scale 2 is for chronic hypercapnic respiratory failure, where 88-92% *is* the target.
    Drawing the 96-100% band for that patient would mark correct management as abnormal."""
    scale_one = charts.vitals_facets(history()).to_dict()
    scale_two = charts.vitals_facets(history(), spo2_scale=2).to_dict()
    bands = {
        "one": {(r["low"], r["high"]) for r in rows_of(scale_one) if "SpO" in r["channel"]},
        "two": {(r["low"], r["high"]) for r in rows_of(scale_two) if "SpO" in r["channel"]},
    }
    assert bands["one"] == {(96.0, 100.0)}
    assert bands["two"] == {(88.0, 92.0)}


def test_the_band_is_one_rectangle_per_panel_not_one_per_observation() -> None:
    """The band travels in the same table as the readings, because Vega-Lite cannot facet a
    layer whose sub-layers carry different data. Aggregating it back to ``min``/``max`` is what
    stops six observations from stacking six translucent rectangles into a darker one."""
    band = charts.vitals_facets(history()).to_dict()["spec"]["layer"][0]
    assert band["encoding"]["y"]["aggregate"] == "min"
    assert band["encoding"]["y2"]["aggregate"] == "max"


def test_a_news2_red_score_gets_a_ring_not_a_different_hue() -> None:
    """A single parameter at 3 warrants review whatever the total is, so it has to be visible.
    A ring rather than a hue because red belongs to the status palette, and because a stroke
    survives greyscale printing where a fill change does not."""
    spec = charts.news2_breakdown(make_news2(9, red=True).components).to_dict()
    bars = spec["layer"][0]["encoding"]
    assert bars["stroke"]["condition"]["test"] == "datum.red"
    assert bars["stroke"]["condition"]["value"] == theme.STATUS["critical"]
    assert bars["stroke"]["value"] == theme.SURFACE  # the 2px gap every other bar keeps
    assert bars["color"]["scale"]["range"] == [theme.SEQUENTIAL[1], theme.SEQUENTIAL[4]]
    assert bars["color"]["legend"] is None


def test_news2_parameters_are_ordered_by_what_they_scored() -> None:
    """The reader's question is "what pushed this total up", so the answer goes first. A row
    scoring zero still appears - a parameter that was measured and normal is information."""
    rows = charts.news2_breakdown(make_news2(7).components).to_dict()
    scores = [row["score"] for row in rows_of(rows)]
    assert scores == sorted(scores, reverse=True)
    assert 0 in scores


def test_the_fusion_contributions_use_a_symmetric_diverging_scale() -> None:
    """Signed data, so the zero point has to sit in the middle of both the colour ramp and the
    axis. An asymmetric domain makes a -6 look larger than a +6."""
    spec = charts.factor_bars(make_assessment().factors).to_dict()
    assert marks_of(spec) == ["rule", "bar"]  # the zero line is drawn *behind* the bars
    encoding = spec["layer"][1]["encoding"]
    assert encoding["x"]["scale"]["domain"] == [-24.0, 24.0]
    assert encoding["color"]["scale"]["domain"] == [-24.0, 0, 24.0]
    assert encoding["color"]["scale"]["range"] == list(theme.DIVERGING)


def test_the_model_probabilities_only_show_classes_the_model_has() -> None:
    """A class the artefact never predicts must not appear as a zero bar, and a key that is
    not a risk level at all must not appear as a category."""
    spec = charts.probability_bars({"HIGH": 0.7, "LOW": 0.3, "NONSENSE": 0.9}).to_dict()
    rows = rows_of(spec)
    assert [row["level"] for row in rows] == ["LOW", "HIGH"]
    colour = spec["layer"][0]["encoding"]["color"]
    assert colour["scale"]["range"] == [theme.STATUS["good"], theme.STATUS["serious"]]
    assert colour["legend"] is None
    assert all(theme.level_glyph(row["level"]) in row["label"] for row in rows)


def test_alert_volume_is_ranked_and_capped() -> None:
    """The alarm-fatigue view. Sorted descending because the question is "what is firing most",
    and capped because a 30-row bar chart is a table with extra steps."""
    counts = {f"Kind {index:02d}": index for index in range(1, 15)}
    spec = charts.alert_kind_bars(counts).to_dict()
    rows = rows_of(spec)
    assert [row["count"] for row in rows] == list(range(14, 4, -1))
    assert len(rows) == 10
    assert spec["layer"][0]["mark"]["color"] == theme.SERIES[0]  # one series, so one hue
    assert "color" not in spec["layer"][0]["encoding"]  # ...and therefore no legend at all


def test_the_confusion_matrix_is_normalised_per_actual_class() -> None:
    """Raw counts flatter a model that always predicts the majority class. Row shares answer
    the question a clinician has - "of the patients who really were HIGH, how many did it
    catch" - and each row sums to 1 regardless of how imbalanced the classes are."""
    spec = charts.confusion_heatmap([[8, 2, 0], [1, 5, 1], [0, 0, 4]], ["LOW", "MED", "HIGH"])
    rows = rows_of(spec.to_dict())
    by_actual: dict[str, float] = {}
    for row in rows:
        by_actual[row["actual"]] = by_actual.get(row["actual"], 0.0) + row["share"]
    assert all(abs(total - 1.0) < 1e-9 for total in by_actual.values())
    assert next(r for r in rows if r["actual"] == "LOW" and r["predicted"] == "LOW")["share"] == 0.8


def test_a_class_with_no_observations_does_not_divide_by_zero() -> None:
    """An evaluation split can miss a rare class entirely. ``max(1, sum(row))`` keeps that row
    at zero instead of raising, so a sparse holdout still renders a readable matrix."""
    spec = charts.confusion_heatmap([[3, 0], [0, 0]], ["LOW", "CRITICAL"]).to_dict()
    empty = [row for row in rows_of(spec) if row["actual"] == "CRITICAL"]
    assert len(empty) == 2
    assert all(row["share"] == 0.0 and row["count"] == 0 for row in empty)


def test_every_confusion_cell_carries_its_number() -> None:
    """Colour is the reading aid; the number is the value. A heatmap where the reader has to
    interpolate against a gradient legend to recover a percentage is a quiz, not a result."""
    spec = charts.confusion_heatmap([[1, 0], [0, 1]], ["LOW", "HIGH"]).to_dict()
    assert marks_of(spec) == ["rect", "text"]
    assert spec["layer"][1]["encoding"]["text"]["field"] == "share"
    assert spec["layer"][0]["encoding"]["color"]["scale"]["range"] == list(theme.SEQUENTIAL)


CALIBRATION_BINS: tuple[dict[str, Any], ...] = (
    {
        "bin_lower": 0.0,
        "bin_upper": 0.25,
        "mean_predicted": 0.1,
        "observed_frequency": 0.08,
        "count": 400,
    },
    {
        "bin_lower": 0.25,
        "bin_upper": 0.5,
        "mean_predicted": 0.37,
        "observed_frequency": 0.44,
        "count": 30,
    },
)


def test_perfect_calibration_is_a_reference_line_not_a_second_series() -> None:
    """The diagonal is where the model *would* sit if it were perfectly calibrated - a
    reference, not an observation. Dashed and in the axis colour so it never reads as data,
    and drawn from its own two-point frame so it cannot be mistaken for a bin."""
    spec = charts.calibration_curve(CALIBRATION_BINS).to_dict()
    assert marks_of(spec) == ["line", "line", "circle"]
    diagonal = spec["layer"][0]
    assert diagonal["mark"]["strokeDash"] == [4, 4]
    assert diagonal["mark"]["color"] == theme.AXIS
    assert rows_of(spec, diagonal) == [{"x": 0, "y": 0}, {"x": 1, "y": 1}]


def test_a_calibration_bin_holding_few_samples_is_drawn_smaller() -> None:
    """A bin of 30 windows and a bin of 400 are not equally trustworthy, and a reliability
    diagram that draws them the same size invites the reader to over-read the sparse end."""
    spec = charts.calibration_curve(CALIBRATION_BINS).to_dict()
    points = spec["layer"][2]["encoding"]
    assert points["size"]["field"] == "count"
    assert points["size"]["legend"] is None  # the tooltip carries the exact number
    assert [row["count"] for row in rows_of(spec, spec["layer"][2])] == [400, 30]


def test_both_calibration_axes_are_probabilities_on_the_same_scale() -> None:
    """A reliability diagram is read by comparing a point to the diagonal, which is only
    meaningful when both axes run 0-100% over the same length."""
    line = charts.calibration_curve(CALIBRATION_BINS).to_dict()["layer"][1]["encoding"]
    assert line["x"]["scale"]["domain"] == [0, 1] == line["y"]["scale"]["domain"]
    assert line["x"]["axis"]["format"] == ".0%" == line["y"]["axis"]["format"]


def test_permutation_importance_carries_its_uncertainty() -> None:
    """Two features whose error bars overlap are not ranked, whatever order the bars are in.
    Without the whisker a reader takes the ordering as fact; ``low``/``high`` are the mean
    plus and minus one standard deviation, and the rule layer draws them."""
    rows = [
        {"feature": "spo2_min", "importance": 0.08, "std": 0.01},
        {"feature": "hr_slope", "importance": 0.07, "std": 0.02},
    ]
    spec = charts.importance_bars(rows).to_dict()
    assert marks_of(spec) == ["bar", "rule"]  # the whisker is drawn over the bar
    plotted = rows_of(spec)
    assert [(row["low"], row["high"]) for row in plotted] == [
        (pytest.approx(0.07), pytest.approx(0.09)),
        (pytest.approx(0.05), pytest.approx(0.09)),
    ]
    whisker = spec["layer"][1]["encoding"]
    assert whisker["x"]["field"] == "low"
    assert whisker["x2"]["field"] == "high"


def test_a_feature_with_no_reported_deviation_gets_a_zero_length_whisker() -> None:
    """``std`` is optional in the artefact's metrics payload. Missing means "not reported",
    which draws as no whisker rather than crashing the model page."""
    spec = charts.importance_bars([{"feature": "age", "importance": 0.05}]).to_dict()
    row = rows_of(spec)[0]
    assert row["low"] == row["high"] == pytest.approx(0.05)


def test_only_the_leading_features_are_drawn() -> None:
    """The artefact reports every feature it has. The long tail is noise, and 40 rows of it
    pushes the plot off the page."""
    rows = [{"feature": f"f{index}", "importance": 1.0 - index / 100} for index in range(40)]
    spec = charts.importance_bars(rows).to_dict()
    assert [row["feature"] for row in rows_of(spec)] == [f"f{index}" for index in range(12)]
    assert charts.importance_bars(rows, limit=3).to_dict()["layer"][0]["encoding"]["y"]["sort"] == [
        "f0",
        "f1",
        "f2",
    ]


# ------------------------------------------------------------------------------- components
# Pure functions first - the ones that build a string. Everything that renders through
# Streamlit follows, under ``AppTest``.


@pytest.mark.parametrize("level", [*theme.LEVEL_ORDER, "UNKNOWN"])
def test_a_badge_states_the_level_three_ways(level: str) -> None:
    """Colour, glyph and word, always all three. The status palette's tightest pair sits in
    the colour-vision band that is only legal *with* secondary encoding, so a badge that
    dropped the glyph or the word would make the palette itself non-compliant."""
    badge = ui.status_badge(level)
    assert theme.level_glyph(level) in badge
    assert level.title() in badge
    assert theme.level_color(level) in badge


def test_a_badge_accepts_the_enum_as_readily_as_the_string() -> None:
    """Call sites pass ``assessment.level`` (an enum) and ``alert.severity`` (sometimes a
    string off a database row). Both have to render, and identically."""
    assert ui.status_badge(RiskLevel.CRITICAL) == ui.status_badge("critical")


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (timedelta(seconds=0), "0 s ago"),
        (timedelta(seconds=42), "42 s ago"),
        (timedelta(seconds=59.6), "60 s ago"),
        (timedelta(minutes=4, seconds=12), "4 m 12 s ago"),
        (timedelta(minutes=59, seconds=59), "59 m 59 s ago"),
        (timedelta(hours=2, minutes=30), "2.5 h ago"),
    ],
)
def test_an_age_is_stated_in_the_unit_a_reader_needs(elapsed: timedelta, expected: str) -> None:
    """A two-second-old hypoxia alert and a nine-minute-old one call for different responses;
    ``14:31:07`` makes the reader do the subtraction. Seconds while seconds matter, then
    minutes, then hours once the exact second stops meaning anything."""
    assert ui.relative_age(EPOCH - elapsed, now=EPOCH) == expected


def test_an_event_with_no_timestamp_says_so() -> None:
    assert ui.relative_age(None) == "—"


def test_a_timestamp_from_the_future_reads_as_now_rather_than_negative() -> None:
    """Clock skew between a container and a database is ordinary. "-3 s ago" reads as a bug
    in the monitor, which is a worse failure than rounding a skewed clock to zero."""
    assert ui.relative_age(EPOCH + timedelta(seconds=90), now=EPOCH) == "0 s ago"


def test_a_naive_timestamp_is_read_as_utc() -> None:
    """SQLite hands back naive datetimes. Subtracting one from an aware ``now`` raises, so the
    alert list would crash on persisted history while working perfectly on live objects."""
    naive = datetime(2026, 9, 5, 11, 59, 30)  # deliberately tz-naive, as SQLite returns
    assert ui.relative_age(naive, now=EPOCH) == "30 s ago"


@pytest.mark.parametrize("attribute", ["heart_rate", "spo2", "temperature", "bp_systolic"])
def test_an_unmeasured_channel_shows_an_em_dash_not_a_zero(attribute: str) -> None:
    """**The one that matters.** A monitor displaying ``0 mmHg`` for "the cuff has not cycled
    yet" is worse than one admitting it does not know - zero is a catastrophic reading, and a
    reader who learns to ignore it will ignore a real one."""
    vitals = make_vitals(**{attribute: None})
    spec = next(row[3] for row in ui.VITAL_ROWS if row[0] == attribute)
    assert ui.format_vital(vitals, attribute, spec) == "—"
    assert ui.format_vital(make_vitals(), attribute, spec) != "—"


def test_a_derived_channel_is_absent_rather_than_wrong_when_its_inputs_are() -> None:
    """Shock index is heart rate over systolic. With no cuff reading there is no index, and
    the grid has to say so rather than print the numerator."""
    assert ui.format_vital(make_vitals(bp_systolic=None), "shock_index", "{:.2f}") == "—"
    assert ui.format_vital(make_vitals(), "shock_index", "{:.2f}") == "0.62"


def test_a_channel_the_type_does_not_have_at_all_is_a_dash() -> None:
    """``getattr(..., None)`` guards a stale row name surviving a rename. A typo in the grid
    must not take the whole bedside view down."""
    assert ui.format_vital(make_vitals(), "end_tidal_co2", "{:.0f}") == "—"


def test_every_vital_row_names_an_attribute_that_exists() -> None:
    """The other half of that guard: the grid is declared as data, so the declaration is what
    gets checked. A row pointing at nothing would render as eight em dashes and no error."""
    vitals = make_vitals()
    for attribute, label, _unit, spec in ui.VITAL_ROWS:
        assert hasattr(vitals, attribute), attribute
        assert label and spec.startswith("{:")
    assert ui.format_vital(vitals, "spo2", "{:.0f}") == "98"


def test_the_consciousness_line_states_acvpu_gcs_and_oxygen_together() -> None:
    """NEWS2 scores consciousness and supplemental oxygen separately, and a GCS of 15 on 60%
    FiO₂ is a different patient from a GCS of 15 on room air. Reading them as one line is
    what stops the oxygen flag being missed.

    The ACVPU letter *and* its word: the letter matches the paper chart, and ``V``/``P``/``U``
    mean nothing to a reader who does not already know the scale.
    """
    line = ui.consciousness_line(make_vitals(on_supplemental_oxygen=True, gcs=11.0))
    assert "ACVPU <b>A (Alert)</b>" in line
    assert "GCS <b>11</b>" in line  # an integer score, stored as a float
    assert "on supplemental O₂" in line
    assert "room air" in ui.consciousness_line(make_vitals())


@pytest.mark.parametrize("level", list(Consciousness))
def test_every_point_on_the_acvpu_scale_is_spelled_out(level: Consciousness) -> None:
    assert f"ACVPU <b>{level.value} ({level.label})</b>" in ui.consciousness_line(
        make_vitals(consciousness=level)
    )


def test_a_consciousness_read_off_a_database_row_degrades_to_the_letter() -> None:
    """Persisted vitals come back as plain strings, with no ``label`` to expand. The letter is
    still the charted value, so it is shown rather than dropped."""
    assert "ACVPU <b>V</b>" in ui.consciousness_line(make_vitals(consciousness="V"))


def test_an_unrecorded_consciousness_or_gcs_still_renders() -> None:
    """Both are optional in the schema and both are absent on a fresh admission."""
    line = ui.consciousness_line(make_vitals(consciousness=None, gcs=None))
    assert "ACVPU <b>—</b>" in line
    assert "GCS <b>—</b>" in line


# --------------------------------------------------------------- components, actually rendered
# Everything above tests a string. These run the component inside a real Streamlit script,
# which is the only way to catch the failure this module was written to prevent: a card that
# looks right and does nothing, because its widgets were never registered.


def _component_script(name, payload):
    """The script ``AppTest`` executes: import the module, call one component by name.

    Deliberately parameter-annotation-free. ``from_function`` re-executes *only* these source
    lines in an empty module, so a name used in an annotation would be resolved there rather
    than in this file - and the imports have to happen inside the body for the same reason.
    """
    from icu_monitor.ui import components

    getattr(components, name)(**payload)


def render(name: str, **payload: Any) -> AppTest:
    """Render one component, alone, in a script of its own.

    The payload travels as a live object, so a real ``BedSnapshot`` arrives as itself and a
    callback can record into a list this test still holds a reference to.
    """
    return AppTest.from_function(_component_script, default_timeout=30, args=(name, payload)).run()


def test_the_vitals_grid_registers_a_real_metric_for_every_channel() -> None:
    """**The regression guard for the whole module.** v1 wrapped this grid in a raw ``<div>``
    emitted through ``st.markdown``; Streamlit renders each element into its own DOM node, so
    the div closed before the widgets existed and the sanitiser dropped the rest. The card
    looked right and displayed nothing. Counting registered metrics is what proves otherwise."""
    app = render("vitals_grid", vitals=make_vitals())
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert labels == [label for _attr, label, _unit, _spec in ui.VITAL_ROWS]
    assert "75 bpm" in [metric.value for metric in app.metric]


def test_the_grid_prints_an_em_dash_for_a_channel_the_monitor_did_not_report() -> None:
    """The rendered counterpart of ``format_vital``: absent has to survive as far as the
    screen, unit and all, rather than being coerced to a plausible-looking zero."""
    app = render("vitals_grid", vitals=make_vitals(spo2=None, temperature=None))
    values = {metric.label: metric.value for metric in app.metric}
    assert values["SpO₂"] == "— %"
    assert values["Temperature"] == "— °C"
    assert values["Heart rate"] == "75 bpm"


def test_a_bed_card_opens_the_monitor_when_its_button_is_pressed() -> None:
    """The card is the ward's navigation. A button that renders but never fires its callback
    is the exact v1 failure - and it is invisible in a screenshot, so it is asserted by
    clicking and reading back what the callback recorded."""
    opened: list[str] = []
    app = render("bed_card", bed=make_bed("P007"), on_open=opened.append)
    assert not app.exception
    assert [button.label for button in app.button] == ["Open monitor"]

    app.button[0].click().run()
    assert opened == ["P007"]


def test_a_bed_card_shows_the_score_its_band_and_the_bedside_numbers() -> None:
    """The composite is the headline because it is the one number that orders the ward; the
    badge sits beside it so the number is never read without its band.

    Length of stay is measured against the wall clock, so the admission time is pinned to
    *now* rather than to ``EPOCH`` - otherwise this assertion would drift by an hour every
    hour and fail somebody else's morning.
    """
    admitted = datetime.now(timezone.utc) - timedelta(hours=30)
    bed = make_bed(patient=make_patient("P001", admitted_at=admitted))
    app = render("bed_card", bed=bed, on_open=lambda _pid: None)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {"Composite": "72", "NEWS2": "7"}
    text = "\n".join(str(element.value) for element in app.markdown)
    assert theme.level_glyph("HIGH") in text and "High" in text
    assert "Heart rate <b>75</b>" in text
    assert "Community-acquired pneumonia · 67y · LOS 30 h" in text


def test_a_bed_card_states_a_missing_news2_total_rather_than_zero() -> None:
    """NEWS2 needs a respiratory rate and a saturation. Without them there is no total, and a
    displayed 0 would read as the healthiest patient on the ward."""
    assessment = make_assessment(news2=None)
    app = render("bed_card", bed=make_bed(assessment=assessment), on_open=lambda _pid: None)
    assert {metric.label: metric.value for metric in app.metric}["NEWS2"] == "—"


def test_a_clinical_override_is_flagged_on_the_card() -> None:
    """An override is the reason the level does not follow the score, so it has to be on the
    card that shows the score - not one click away on the patient page."""
    assessment = make_assessment(overrides=("NEWS2 red score: SpO₂ 88%",))
    app = render("bed_card", bed=make_bed(assessment=assessment), on_open=lambda _pid: None)
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "⚑ NEWS2 red score: SpO₂ 88%" in text
    assert theme.STATUS["serious"] in text


def test_an_open_alert_can_be_acknowledged_from_its_card() -> None:
    """Acknowledging is the one write the alert list performs, so the callback is asserted by
    clicking it and reading back the id the handler received."""
    cleared: list[int] = []
    app = render("alert_card", alert=make_alert(alert_id=42), on_acknowledge=cleared.append)
    assert not app.exception
    assert [button.label for button in app.button] == ["Acknowledge"]

    app.button[0].click().run()
    assert cleared == [42]


def test_an_acknowledged_alert_says_so_instead_of_offering_the_button() -> None:
    """Two states, never both: an acknowledged alert stays on screen as history, and offering
    a second acknowledgement would only invite a double write."""
    alert = make_alert(acknowledged_at=EPOCH, acknowledged_by="rn.jordan")
    app = render("alert_card", alert=alert, on_acknowledge=lambda _id: None)
    assert app.button.values == []
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "✓ acknowledged" in text
    assert theme.STATUS["good"] in text
    assert "acknowledged by rn.jordan" in text


def test_an_alert_card_with_no_handler_is_read_only() -> None:
    """The patient page lists that bed's alerts for context. A button there would be a second,
    unaudited path to the same write, so the card renders without one."""
    app = render("alert_card", alert=make_alert(), on_acknowledge=None)
    assert app.button.values == []
    assert "SpO₂ 88% on room air" in "\n".join(str(el.value) for el in app.markdown)


def test_an_alert_card_leads_with_severity_kind_and_bed() -> None:
    """Triage order: how bad, what it is, whose it is. The severity badge is first because it
    is what decides whether the reader reads on."""
    app = render("alert_card", alert=make_alert(), on_acknowledge=lambda _id: None)
    text = "\n".join(str(element.value) for element in app.markdown)
    assert ui.status_badge(RiskLevel.HIGH, size="0.72rem") in text
    assert "P001 · Hypoxaemia" in text
    assert "raised " in text and " ago" in text


def test_an_alert_that_is_still_true_shows_both_times() -> None:
    """De-duplication means one alert covers a condition that has persisted for minutes. Both
    timestamps are needed: "raised 9 m ago" alone reads as stale, "still true" says it is not."""
    alert = make_alert(last_seen_at=EPOCH + timedelta(minutes=3))
    text = "\n".join(str(el.value) for el in render("alert_card", alert=alert).markdown)
    assert "raised " in text
    assert "still true " in text


def test_a_chart_panel_with_nothing_to_draw_says_what_is_missing() -> None:
    """Eight of the ten builders return ``None`` on empty input, so the panel is where "no
    data yet" is said. A blank bordered box reads as a rendering failure; naming the reason
    distinguishes "the ward just started" from "the chart is broken"."""
    app = render("chart_panel", title="Composite trend", chart=None, fallback="No history yet.")
    assert not app.exception
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "**Composite trend**" in text
    assert "No history yet." in text


def test_a_chart_panel_hands_a_real_chart_to_streamlit() -> None:
    """The other branch: an Altair object is passed through to ``st.altair_chart``, which is
    where a malformed encoding would surface."""
    chart = charts.probability_bars({"LOW": 0.3, "HIGH": 0.7})
    app = render("chart_panel", title="Model", chart=chart, note="Class probabilities.")
    assert not app.exception
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "Class probabilities." in text
    assert "No data yet." not in text


def test_a_kpi_row_registers_one_metric_per_headline() -> None:
    """The overview's top strip. Metrics, not markdown - the same lesson as the vitals grid."""
    app = render(
        "kpi_row",
        items=[
            ("Beds", 4, None),
            ("Open alerts", 2, "Unacknowledged"),
            ("Highest risk", "ICU-01 · 72", None),
        ],
    )
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Beds", "4"),
        ("Open alerts", "2"),
        ("Highest risk", "ICU-01 · 72"),
    ]


def test_an_empty_state_names_the_thing_that_is_missing() -> None:
    app = render("empty_state", message="No beds match “CRITICAL”.", icon="✓")
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "No beds match “CRITICAL”." in text
    assert "✓" in text


def test_a_definition_list_renders_every_pair() -> None:
    app = render("definition_list", rows={"Model": "hist_gradient_boosting", "Macro F1": "0.82"})
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "Model" in text and "hist_gradient_boosting" in text
    assert "Macro F1" in text and "0.82" in text


INJECTION = "<script>alert('x')</script>"


def test_a_patient_name_cannot_smuggle_markup_into_the_ward_board() -> None:
    """Every card here renders with ``unsafe_allow_html=True``, which turns the demographics
    fields into a template. Names and diagnoses arrive from a CSV in this project and from an
    HL7 feed in any real one, so they are escaped at the sink rather than trusted at the
    source. Streamlit's own sanitiser strips ``<script>``, but relying on it would leave
    ``<img onerror=…>`` and every other vector to somebody else's defaults."""
    patient = make_patient("P001", display_name=INJECTION, primary_diagnosis=INJECTION)
    app = render("bed_card", bed=make_bed(patient=patient), on_open=lambda _pid: None)
    text = "\n".join(str(element.value) for element in app.markdown)
    assert INJECTION not in text
    assert text.count("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;") == 2


def test_an_alert_message_is_escaped_as_well() -> None:
    """Alert text is assembled from vitals and thresholds today, but the field is free text in
    the schema and is echoed straight into the same HTML."""
    app = render("alert_card", alert=make_alert(message=INJECTION, acknowledged_by=INJECTION))
    text = "\n".join(str(element.value) for element in app.markdown)
    assert INJECTION not in text
    assert "&lt;script&gt;" in text


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty_state", {"message": INJECTION}),
        ("page_header", {"title": "Ward", "subtitle": INJECTION}),
        ("definition_list", {"rows": {INJECTION: INJECTION}}),
        ("chart_panel", {"title": "T", "chart": None, "fallback": INJECTION}),
    ],
)
def test_no_component_passes_untrusted_text_through_unescaped(
    name: str, payload: dict[str, Any]
) -> None:
    """The same rule applied to every remaining sink, so a new one cannot be added without
    this failing."""
    app = render(name, **payload)
    assert INJECTION not in "\n".join(str(element.value) for element in app.markdown)


def test_caption_is_documented_as_a_markup_sink_and_behaves_like_one() -> None:
    """``caption`` deliberately does *not* escape - its callers pass ``<b>`` runs built from
    already-escaped parts, which is why the escaping above happens in those callers. Pinned so
    the exception stays visible: anything reaching ``caption`` must arrive safe."""
    app = render("caption", text="Heart rate <b>75</b>")
    assert "<b>75</b>" in "\n".join(str(element.value) for element in app.markdown)


def test_the_header_status_slot_is_the_other_documented_exception() -> None:
    """``right`` carries a ``<br>`` from the overview to stack the tick above the clock, so it
    cannot be escaped. Only operator-controlled strings go there - tick counts, timestamps, the
    model version - while ``subtitle``, which is where patient counts and source labels land,
    is escaped."""
    app = render("page_header", title="Ward", subtitle=INJECTION, right="tick 12<br>08:00 UTC")
    text = "\n".join(str(element.value) for element in app.markdown)
    assert "tick 12<br>08:00 UTC" in text
    assert INJECTION not in text


# ------------------------------------------------------------------------------ the whole app
# Every test below runs ``main()`` end to end: engine, snapshot, sidebar and one view. That is
# also the deployability check - these configs have no camera, no detector and no database
# file, which is exactly what a fresh clone on somebody else's machine has.


@pytest.mark.parametrize("view", app_module.VIEWS)
def test_every_view_renders_on_a_bare_clone(tmp_path: Path, view: str) -> None:
    """**The end-to-end test this project exists to pass.** v1 opened a camera at import and
    called ``winsound`` on the first alert, so it ran on exactly one machine. Each view here
    is rendered with vision off, alerts silent and the database in memory, and an uncaught
    exception anywhere in the run - including inside a view - fails the test."""
    app = run_app(ui_settings(tmp_path), icu_view=view)
    assert not app.exception, [str(error.value) for error in app.exception]
    assert app.session_state["icu_view"] == view
    assert app.markdown  # something was actually drawn


@pytest.mark.parametrize("view", app_module.VIEWS)
def test_the_disclaimer_is_on_every_view(tmp_path: Path, view: str) -> None:
    """A risk score beside a patient name looks like a clinical instrument. The sidebar is
    always visible, so that is where the disclaimer goes - on every view, not just the one
    somebody remembers to read."""
    text = screen_text(run_app(ui_settings(tmp_path), icu_view=view))
    assert "<b>Not a medical device.</b>" in text
    assert "no output here should inform patient care" in text
    assert theme.STATUS["warning"] in text


def test_the_sidebar_summarises_the_ward_on_every_view(tmp_path: Path) -> None:
    """The counts a nurse checks first, kept out of the view so switching views never loses
    them: how many beds, how many need review, how many alerts are live, and which bed is
    worst. A tick number too - it is how "is this thing still running" gets answered."""
    text = screen_text(run_app(ui_settings(tmp_path, bed_count=6), icu_view="Settings"))
    for label in ("Beds", "Needing review", "Active alerts", "Highest risk", "Tick"):
        assert label in text
    assert ">6<" in text  # the bed count, rendered into the definition list


def test_the_ward_list_is_ordered_by_risk_and_not_by_bed_number(tmp_path: Path) -> None:
    """The whole reason for computing a composite score is to be able to sort by it. A list
    ordered by bed number is a filing system; this one answers "who needs me first".

    Asserted on the rendered ``Composite`` metrics rather than on the sort key, so it is a
    statement about what the reader sees.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6))
    assert not app.exception
    cards = [button.key for button in app.button if (button.key or "").startswith("overview_")]
    scores = [float(metric.value) for metric in app.metric if metric.label == "Composite"]
    assert len(cards) == 6
    assert len(scores) == 6
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("chosen", ["LOW", "MEDIUM", "HIGH", "CRITICAL", "Needs review"])
def test_the_overview_filter_only_ever_narrows_the_list(tmp_path: Path, chosen: str) -> None:
    """A filter that shows a bed it should not is a triage error. Whatever the ward looks like
    on the tick, the filtered list is a subset of the full one - and every card in it matches.

    Both runs share one engine, and ``tick_seconds`` is pinned so neither advances it: a subset
    assertion across two renders is only about the filter if the ward is identical in both.
    """
    cfg = ui_settings(tmp_path, bed_count=8, tick_seconds=30.0)
    everything = run_app(cfg)
    all_beds = {button.key for button in everything.button if button.key}

    filtered = run_app(cfg, icu_overview_filter=chosen)
    assert not filtered.exception
    shown = {button.key for button in filtered.button if button.key}
    assert shown <= all_beds

    levels = {str(element.value) for element in filtered.markdown}
    if chosen in theme.LEVEL_ORDER and shown:
        assert any(chosen.title() in text for text in levels)


def test_a_filter_that_matches_nothing_says_so_instead_of_showing_a_blank(
    tmp_path: Path,
) -> None:
    """A simulated ward rarely holds a CRITICAL bed on the first tick, and an empty panel reads
    as a broken filter. The empty state names the filter that produced it, with a tick rather
    than a warning glyph - no beds needing review is good news, not an error."""
    app = run_app(ui_settings(tmp_path, bed_count=2), icu_overview_filter="CRITICAL")
    assert not app.exception
    text = screen_text(app)
    if not any(button.key and button.key.startswith("overview_") for button in app.button):
        assert "No beds match “CRITICAL”." in text
        assert "✓" in text


def test_opening_a_bed_switches_to_the_patient_monitor_focused_on_it(tmp_path: Path) -> None:
    """The one navigation path in the app. Selecting the patient without switching the view
    would leave the reader on the overview wondering what their click did, so the callback
    writes both keys - and the assertion is that the *next* run lands on that patient's page."""
    app = run_app(ui_settings(tmp_path, bed_count=4))
    card = next(button for button in app.button if (button.key or "").startswith("overview_"))
    chosen = (card.key or "").removeprefix("overview_")

    card.click().run()

    assert not app.exception
    assert app.session_state["icu_view"] == "Patient monitor"
    assert app.session_state["icu_selected_patient"] == chosen
    assert "Patient monitor" in screen_text(app)


def test_the_patient_monitor_defaults_to_the_sickest_bed(tmp_path: Path) -> None:
    """ "Who needs me" rather than "who is in bed 1". With nothing selected the view opens on
    the worst bed, which is the same bed the overview sorts to the top.

    Two script runs have to agree about one ordering, so ``tick_seconds`` is pinned to the
    field's ceiling and the ward cannot move between them. At the default 0.25 s several ticks
    landed in the gap, and with two beds a point apart the top of the overview genuinely
    changed - so the assertion passed on an idle machine and failed inside the full suite, which
    is the worst way for a test to be wrong.
    """
    cfg = ui_settings(tmp_path, bed_count=5, tick_seconds=30.0)
    app = run_app(cfg, icu_view="Patient monitor")
    assert not app.exception
    chosen = app.session_state["icu_selected_patient"]
    assert chosen

    overview = run_app(cfg)
    first = next(b for b in overview.button if (b.key or "").startswith("overview_"))
    assert (first.key or "").removeprefix("overview_") == chosen


def sidebar_tick(app: AppTest) -> int:
    """The tick counter the sidebar prints, read back off the rendered definition list.

    Asserted through the screen rather than by calling ``app_state.snapshot()`` from the test,
    because that reaches into ``st.session_state`` from outside a script run - which is a
    different thing from what the dashboard does.
    """
    found = re.search(r"Tick</span><span[^>]*>(\d+)<", screen_text(app))
    assert found is not None, "the sidebar did not render a tick counter"
    return int(found.group(1))


def test_advancing_the_ward_by_hand_is_offered_only_when_live_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ways to advance the same ward is one too many: with Live on, the rerun loop is
    already ticking and a manual button would race it.

    The live branch ends in ``time.sleep`` then ``st.rerun()``, which under ``AppTest`` is an
    unbounded loop rather than a browser round trip, so both are recorded instead of performed.
    ``app.time`` is replaced wholesale rather than ``time.sleep`` patched globally - the test
    harness polls with ``time.sleep`` itself, and stubbing that hangs the runner.
    """
    paused = run_app(ui_settings(tmp_path))
    assert "Advance one tick" in [button.label for button in paused.button]

    slept: list[float] = []
    reran: list[int] = []
    monkeypatch.setattr(app_module, "time", SimpleNamespace(sleep=slept.append))
    monkeypatch.setattr(app_module.st, "rerun", lambda **_kwargs: reran.append(1))
    live = AppTest.from_string("from icu_monitor.ui.app import main\n\nmain()\n")
    live.session_state["icu_settings"] = ui_settings(tmp_path)
    live.session_state["icu_live"] = True
    live.run()

    assert not live.exception
    assert live.session_state["icu_live"] is True
    assert "Advance one tick" not in [button.label for button in live.button]
    assert reran == [1]
    assert slept and all(0.5 <= interval <= 5.0 for interval in slept)


def test_advancing_by_hand_moves_the_tick_on(tmp_path: Path) -> None:
    """The paused state still has to be usable, or "Live off" means "frozen". Pressing the
    button forces a tick even though ``tick_seconds`` has not elapsed - pinned at the maximum
    interval so nothing but the button could have advanced the ward."""
    app = run_app(ui_settings(tmp_path, tick_seconds=30.0))
    before = sidebar_tick(app)

    next(button for button in app.button if button.label == "Advance one tick").click().run()

    assert not app.exception
    assert sidebar_tick(app) == before + 1


@pytest.mark.parametrize("scope", ["Active", "Open", "All raised"])
def test_the_ledger_never_adds_active_and_open_together(tmp_path: Path, scope: str) -> None:
    """*Active* is "the condition is true right now" - the wall display. *Open* is "raised and
    nobody has acknowledged it" - the audit trail. They are different sets, and conflating
    them is how a monitor ends up either silent about a deteriorating patient or shouting
    about one who recovered ten minutes ago. Each scope is labelled with its own count."""
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Alerts", icu_alert_scope=scope)
    assert not app.exception
    text = screen_text(app)
    assert f"alert(s) · {scope.lower()}" in text
    assert {"Active now", "Open", "Raised (session)"} <= {m.label for m in app.metric}


def test_acknowledging_everything_reports_how_many_it_cleared(tmp_path: Path) -> None:
    """A bulk write has to say what it did. "Acknowledged 7 alert(s)" is auditable; a silently
    emptied list is indistinguishable from a filter that stopped matching."""
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Alerts", icu_alert_scope="Open")
    before = len([b for b in app.button if b.label == "Acknowledge"])
    assert before, "the simulated ward raised nothing to acknowledge"

    next(button for button in app.button if button.label == "Acknowledge all").click().run()

    assert not app.exception
    assert f"Acknowledged {before} alert(s)." in screen_text(app)
    assert not [button for button in app.button if button.label == "Acknowledge"]
    assert "Nothing matches this filter." in screen_text(app)


def test_acknowledging_one_alert_leaves_the_others_open(tmp_path: Path) -> None:
    """The per-card action is the normal path, and it has to be exactly one write: clearing a
    hypoxia alert must not clear the tachycardia alert next to it.

    Read under *All raised* rather than *Open*, because that is the scope where the cleared
    alert stays on screen - so both halves are observable at once: one fewer button, and that
    card now reading as acknowledged.
    """
    app = run_app(
        ui_settings(tmp_path, bed_count=6), icu_view="Alerts", icu_alert_scope="All raised"
    )
    open_before = [button.key for button in app.button if button.label == "Acknowledge"]
    assert len(open_before) > 1

    next(button for button in app.button if button.label == "Acknowledge").click().run()

    assert not app.exception
    open_after = [button.key for button in app.button if button.label == "Acknowledge"]
    assert len(open_after) == len(open_before) - 1
    assert "✓ acknowledged" in screen_text(app)
    assert "acknowledged by dashboard" in screen_text(app)


def test_a_filter_matching_nothing_says_which_kind_of_nothing(tmp_path: Path) -> None:
    """ "No conditions are currently true" is reassurance; "nothing matches this filter" is a
    hint to widen it. The same blank panel for both would be neither."""
    quiet = run_app(
        ui_settings(tmp_path),
        icu_view="Alerts",
        icu_alert_scope="Open",
        icu_alert_kinds=["Sensor failure"],
    )
    assert "Nothing matches this filter." in screen_text(quiet)
    assert "0 alert(s) · open" in screen_text(quiet)


def test_narrowing_to_one_bed_only_shows_that_beds_alerts(tmp_path: Path) -> None:
    """The bed filter is how a nurse checks one patient's history. An alert from another bed
    appearing here would be a charting error.

    Pinned ticks again: the bed list is read off the first run and the assertion is made against
    the second, so an alert firing between them would have been read as a filter leak.
    """
    cfg = ui_settings(tmp_path, bed_count=6, tick_seconds=30.0)
    everything = run_app(cfg, icu_view="Alerts", icu_alert_scope="All raised")
    assert not everything.exception
    beds = everything.selectbox[0].options
    assert beds[0] == "All beds"

    focused = run_app(
        cfg,
        icu_view="Alerts",
        icu_alert_scope="All raised",
        icu_alert_patient=beds[1],
    )
    assert not focused.exception
    text = screen_text(focused)
    for other in beds[2:]:
        assert f"{other} · " not in text


# ============================================================================ model insights

#: A depth-3 tree, registered under its own name so the real candidates stay untouched. Which
#: estimator won is not what this file is about; that a real ``joblib`` bundle on disk lights
#: up the populated view is.
TINY_TREE = "ui_tiny_tree"


def tiny_artefact(cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> str:
    """Train and persist a real artefact under ``cfg.project_root``; return its version.

    Nothing short of a real bundle exercises this view - ``available`` is
    ``load_model(...) is not None``, and the metrics, the calibration bins and the model card
    are all read back off disk. The registry cache is keyed on path plus mtime, so a bundle
    under a per-test ``tmp_path`` cannot leak into another test.
    """
    monkeypatch.setitem(
        pipeline_module.CANDIDATES,
        TINY_TREE,
        lambda seed=0: DecisionTreeClassifier(max_depth=3, random_state=seed),
    )
    result = train(
        config=cfg,
        frame=make_window_frame(patients=15, windows_per_patient=4),
        candidates=[TINY_TREE],
        n_splits=3,
    )
    return result.version


def test_the_model_view_says_what_still_works_when_there_is_no_artefact(tmp_path: Path) -> None:
    """A bare clone has no ``artifacts/model.joblib``, and this is the view most likely to be
    opened first by someone deciding whether the install worked.

    So the empty state carries three things: what is missing, what still runs without it, and
    the one command that changes that. A blank panel here reads as a broken install; a row of
    zeroed metrics would be worse, because zero is a measurement.
    """
    app = run_app(ui_settings(tmp_path), icu_view="Model insights")

    assert not app.exception
    text = screen_text(app)
    assert "No trained artefact is loaded." in text
    assert "Scoring still runs on NEWS2 and the vision signal" in text
    assert "python -m icu_monitor train" in text
    assert "◇" in text  # a lozenge, not a warning triangle: this is a normal state
    assert "no artefact" in text  # the header's version slot, rather than a blank
    assert not app.dataframe
    assert not app.get("vega_lite_chart")
    assert not app.metric  # no scores at all beats scores that mean nothing


def test_the_model_view_leads_with_the_imbalance_immune_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain accuracy on a three-class problem whose HIGH class is a small minority is
    technically true and practically misleading - a model that never predicts HIGH can still
    score well. So the headline row is macro F1, balanced accuracy, macro ROC-AUC and κ, each
    of which a majority-class guesser fails, and the version is named beside them so a number
    can always be traced to the artefact that produced it.
    """
    cfg = ui_settings(tmp_path, bed_count=3)
    version = tiny_artefact(cfg, monkeypatch)

    app = run_app(cfg, icu_view="Model insights")

    assert not app.exception
    headline = {metric.label for metric in app.metric}
    assert {
        "Macro F1",
        "Balanced accuracy",
        "ROC-AUC (macro)",
        "Cohen's κ",
        "Held-out patients",
    } <= headline
    text = screen_text(app)
    assert version in text
    assert TINY_TREE in text  # the algorithm, not just the version string
    # The interpretation travels with the number, as tooltip text on the metric itself - a
    # macro F1 of 0.9 means nothing to a reader who does not know what chance scores.
    tooltips = {metric.label: (metric.help or "") for metric in app.metric}
    assert "A coin flip scores ~0.33." in tooltips["Macro F1"]
    assert "immune to class imbalance" in tooltips["Balanced accuracy"]
    assert "Patient-disjoint from every training window." in tooltips["Held-out patients"]


def test_the_per_class_table_names_the_class_with_a_glyph_and_a_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HIGH is the class the system exists to catch and the hardest one, so its own precision
    and recall are on screen rather than folded into a macro average.

    Each row is labelled with the same glyph the charts use, which is what lets a reader carry
    a row across to a coloured mark. A table cell cannot carry colour at all, so the glyph is
    not decoration here - it is the only shared channel.
    """
    cfg = ui_settings(tmp_path, bed_count=3)
    tiny_artefact(cfg, monkeypatch)

    app = run_app(cfg, icu_view="Model insights")

    assert not app.exception
    table = app.dataframe[0].value
    assert set(table["Class"]) == {
        f"{theme.level_glyph(name)} {name}" for name in ("LOW", "MEDIUM", "HIGH")
    }
    assert {
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Avg precision",
        "Windows",
        "Prevalence",
    } <= set(table.columns)
    assert all(str(share).endswith("%") for share in table["Prevalence"])
    assert "its precision is the number to read before trusting" in screen_text(app)


def test_the_model_view_draws_the_three_diagnostics_a_scalar_cannot_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confusion, calibration, permutation importance. Which classes get mistaken *for what*,
    whether a predicted 0.8 happens 80 % of the time, and which features the score rests on -
    none of which any single summary metric answers."""
    cfg = ui_settings(tmp_path, bed_count=3)
    tiny_artefact(cfg, monkeypatch)

    app = run_app(cfg, icu_view="Model insights")

    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 3
    text = screen_text(app)
    assert "Confusion matrix (row-normalised)" in text
    assert "Raw counts would flatter a model that leans" in text
    assert "Calibration · P(" in text
    assert "Permutation importance · top features" in text
    assert "overlapping bars are not reliably ordered" in text


def test_the_model_card_opens_on_its_caveats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six sections, exactly one open, and it is the limitations.

    A model card whose caveats are one collapsed panel among six is a model card whose caveats
    do not get read - which is the entire failure mode the format exists to prevent.
    """
    cfg = ui_settings(tmp_path, bed_count=3)
    tiny_artefact(cfg, monkeypatch)

    app = run_app(cfg, icu_view="Model insights")

    assert not app.exception
    opened = {panel.label: panel.proto.expanded for panel in app.expander}
    assert {
        "Intended use and what is out of scope",
        "Training data and split",
        "Label definitions",
        "Candidates considered",
        "Ethical considerations",
        "Caveats and recommendations",
    } <= set(opened)
    assert opened["Caveats and recommendations"] is True
    assert sum(bool(flag) for flag in opened.values()) == 1
    assert "so it cannot drift from the artefact it describes" in screen_text(app)


def test_reloading_names_the_version_it_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training happens in another process - a terminal, a CI job, a container rebuild. The
    reload button is what makes that visible without restarting the dashboard, and it has to
    name what it loaded: ``reload_model`` returns the *model*, so interpolating its return
    value directly printed a Python repr at the reader.
    """
    cfg = ui_settings(tmp_path, bed_count=3)
    version = tiny_artefact(cfg, monkeypatch)
    app = run_app(cfg, icu_view="Model insights")

    next(button for button in app.button if button.label == "Reload artefact").click().run()

    assert not app.exception
    assert f"Loaded {version}." in screen_text(app)
    assert "RiskModel object at" not in screen_text(app)


# =================================================================================== settings


def widget(app: AppTest, kind: str, label: str) -> Any:
    """One widget, by the label the reader sees.

    None of the Settings widgets carry a ``key`` - they live in forms and are read by return
    value - so the label is the only handle, which is also the handle a person would use.
    """
    found = [element for element in getattr(app, kind) if element.label == label]
    assert found, f"no {kind} labelled {label!r}"
    return found[0]


def test_widget_bounds_are_read_off_the_schema_and_not_retyped() -> None:
    """The regression this exists for: the page offered a bed count up to 32 against a field
    that stops at 24, and listed a "none" detector that the config spells "off". A widget whose
    range is typed out by hand drifts from validation the first time either changes.

    ``cap`` only ever applies where the field declares no ceiling of its own, so it cannot
    contradict validation - there is nothing there to contradict.
    """
    assert settings_view._bounds("bed_count", 999) == (1.0, 24.0)  # the field's le wins
    assert settings_view._bounds("tick_seconds", 999.0) == (0.25, 30.0)
    assert settings_view._bounds("news2_medium_threshold", 20) == (1.0, 20.0)  # no le: cap wins
    assert settings_view._bounds("news2_medium_threshold", 11) == (1.0, 11.0)


def test_the_option_lists_are_the_config_literals_themselves() -> None:
    """Every dropdown is ``get_args`` of the ``Literal`` the field is typed with, so an option
    the schema would refuse cannot be offered."""
    assert get_args(VitalsSourceName) == settings_view.VITALS_SOURCES
    assert get_args(FrameSourceName) == settings_view.FRAME_SOURCES
    assert get_args(DetectorName) == settings_view.DETECTORS
    assert "off" in settings_view.DETECTORS
    assert "none" not in settings_view.DETECTORS  # what the page used to offer


@pytest.mark.parametrize(
    ("kind", "label", "expected"),
    [
        ("number_input", "Beds", (1.0, 24.0)),
        ("number_input", "Tick interval (s)", (0.25, 30.0)),
        ("number_input", "History window (observations)", (20.0, 2000.0)),
    ],
)
def test_the_rendered_widget_carries_those_bounds(
    tmp_path: Path, kind: str, label: str, expected: tuple[float, float]
) -> None:
    """The bounds have to reach the widget, not just the helper that computes them."""
    control = widget(run_app(ui_settings(tmp_path), icu_view="Settings"), kind, label)
    assert (control.min, control.max) == expected


def test_a_non_increasing_news2_pair_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    """MEDIUM at or above HIGH is not a smaller ward configuration, it is an incoherent one:
    every patient would be escalated to HIGH without passing through MEDIUM.

    The point of the assertion is the *second* half. A rejected form must leave the previous
    settings intact - half-applying a configuration gives a dashboard that draws one number and
    explains it with another.
    """
    app = run_app(ui_settings(tmp_path), icu_view="Settings")
    before = app.session_state["icu_settings"]

    widget(app, "number_input", "NEWS2 → MEDIUM").set_value(9)
    widget(app, "number_input", "NEWS2 → HIGH").set_value(5)
    widget(app, "button", "Apply thresholds").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [
        "The NEWS2 MEDIUM threshold must be below the HIGH threshold."
    ]
    assert app.session_state["icu_settings"] is before


def test_composite_bands_that_do_not_increase_are_refused(tmp_path: Path) -> None:
    """MEDIUM < HIGH < CRITICAL is what makes the bands bands. Checked separately from NEWS2
    because they fail for different reasons and a single "invalid thresholds" message would
    leave the reader guessing which of five numbers to fix."""
    app = run_app(ui_settings(tmp_path), icu_view="Settings")

    widget(app, "number_input", "Composite → MEDIUM").set_value(90.0)
    widget(app, "button", "Apply thresholds").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [
        "Composite thresholds must increase: MEDIUM < HIGH < CRITICAL."
    ]
    assert app.session_state["icu_settings"].composite_medium_threshold == 40.0


def test_applying_a_valid_threshold_rebuilds_the_ward_with_it(tmp_path: Path) -> None:
    """The claim the whole page rests on: a control that is read by the engine rather than
    displayed at the reader. The new value has to be in ``icu_settings`` - which is the object
    the engine is constructed from and the cache is keyed on - not only in the widget."""
    app = run_app(ui_settings(tmp_path), icu_view="Settings")

    widget(app, "number_input", "Composite → MEDIUM").set_value(30.0)
    widget(app, "button", "Apply thresholds").click().run()

    assert not app.exception
    assert not app.error
    assert app.session_state["icu_settings"].composite_medium_threshold == 30.0
    assert "Thresholds applied; the ward was rebuilt with them." in screen_text(app)


def test_weights_that_are_all_zero_are_refused(tmp_path: Path) -> None:
    """The weights are normalised over the available channels, so all-zero is a division by
    zero and a score with no inputs. Refused with the previous split left in place."""
    app = run_app(ui_settings(tmp_path), icu_view="Settings")

    for label in ("Model", "NEWS2", "Vision"):
        widget(app, "slider", label).set_value(0.0)
    widget(app, "button", "Apply weights").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == ["At least one weight must be above zero."]
    assert app.session_state["icu_settings"].weight_ml == 0.45


def test_the_weights_panel_shows_the_normalised_split_not_the_raw_numbers(tmp_path: Path) -> None:
    """0.45 / 0.40 / 0.15 are relative, and reading them as percentages only works because they
    happen to sum to 1. The panel shows what the fusion layer will actually use, which is the
    normalised share - so removing the camera visibly re-splits the remaining two rather than
    silently shrinking every score."""
    text = screen_text(run_app(ui_settings(tmp_path), icu_view="Settings"))
    assert "Normalised split: model 45% · NEWS2 40% · vision 15%" in text
    assert "so removing the camera does not silently shrink every score" in text


def test_rebuilding_the_ward_forgets_the_bed_that_no_longer_exists(tmp_path: Path) -> None:
    """A smaller ward can drop the selected bed, and a stale ``icu_selected_patient`` would
    leave the Patient Monitor pointed at a patient who is not there. The rebuild clears it and
    the view falls back to the worst remaining bed."""
    app = run_app(
        ui_settings(tmp_path, bed_count=4), icu_view="Settings", icu_selected_patient="P001"
    )

    widget(app, "number_input", "Beds").set_value(6)
    widget(app, "button", "Rebuild ward").click().run()

    assert not app.exception
    assert app.session_state["icu_settings"].bed_count == 6
    assert "icu_selected_patient" not in app.session_state
    assert "Ward rebuilt." in screen_text(app)


def test_purging_the_database_asks_first(tmp_path: Path) -> None:
    """The one irreversible action in the app. It is two clicks, the confirmation names what
    will be lost, and *Cancel* takes effect on the click that was pressed - flipping the flag
    from the script body left the panel on screen until something else happened to rerun the
    page, which reads as a button that does nothing."""
    app = run_app(ui_settings(tmp_path), icu_view="Settings")
    assert "Yes, purge" not in [button.label for button in app.button]

    widget(app, "button", "Purge all rows").click().run()

    warnings = [str(warning.value) for warning in app.warning]
    assert any("It cannot be undone." in text for text in warnings)
    assert any(
        "every stored patient, observation, assessment, and alert" in text for text in warnings
    )
    assert {"Yes, purge", "Cancel"} <= {button.label for button in app.button}

    widget(app, "button", "Cancel").click().run()

    assert not app.exception
    assert app.session_state["icu_confirm_purge"] is False
    assert "Yes, purge" not in [button.label for button in app.button]


def test_confirming_the_purge_empties_the_database_and_says_so(tmp_path: Path) -> None:
    """And the second click has to actually do it, once, and report it."""
    app = run_app(ui_settings(tmp_path), icu_view="Settings")
    widget(app, "button", "Purge all rows").click().run()

    widget(app, "button", "Yes, purge").click().run()

    assert not app.exception
    assert "Database emptied." in screen_text(app)
    assert app.session_state["icu_confirm_purge"] is False
    assert "Yes, purge" not in [button.label for button in app.button]
    # The stats block is rendered from the emptied tables.
    assert "0" in screen_text(app)


def test_the_settings_page_says_the_api_is_unauthenticated_when_it_is(tmp_path: Path) -> None:
    """An open ``/api/v1`` is a reasonable default for a local demo and the wrong one for
    anything reachable from a network. The dashboard is where an operator would look, so it says
    which of the two they are running - and says it as a warning, not a caption."""
    open_api = run_app(ui_settings(tmp_path), icu_view="Settings")
    warnings = "\n".join(str(warning.value) for warning in open_api.warning)
    assert "`ICU_API_KEY` is not set" in warnings
    assert "routes are unauthenticated" in warnings
    assert "set the variable before exposing the service beyond localhost" in warnings

    with_key = run_app(ui_settings(tmp_path, api_key="s3cret"), icu_view="Settings")
    text = screen_text(with_key)
    assert "`ICU_API_KEY` is set" in text
    assert "requires a matching" in text
    assert "s3cret" not in text  # the value is never echoed, only its presence
    assert "`ICU_API_KEY` is not set" not in "\n".join(str(w.value) for w in with_key.warning)


def test_the_system_panel_names_every_component_and_its_state(tmp_path: Path) -> None:
    """ "Is the camera on" and "is a model loaded" are the two questions asked of this project
    most often. Both are answered on one line each, with a glyph beside the colour so the state
    is readable without colour vision, and with the *reason* rather than a bare cross."""
    text = screen_text(run_app(ui_settings(tmp_path), icu_view="Settings"))
    for component in ("engine", "model", "vision", "database"):
        assert f">{component}<" in text
    assert "no trained artefact (NEWS2 still active)" in text
    assert theme.STATUS["good"] in text and theme.STATUS["warning"] in text
    assert "●" in text and "▲" in text


def test_resetting_drops_the_session_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch: whatever was fiddled with, one button puts the deployment's own
    configuration back. ``get_settings`` is redirected at ``tmp_path`` for the duration, because
    the point of the button is that it re-reads the environment - and the suite's first rule is
    that no test touches the real project.
    """
    monkeypatch.setattr(app_state, "get_settings", lambda: ui_settings(tmp_path, bed_count=3))
    app = run_app(ui_settings(tmp_path, bed_count=7), icu_view="Settings")
    assert app.session_state["icu_settings"].bed_count == 7

    widget(app, "button", "Reset all settings to environment defaults").click().run()

    assert not app.exception
    assert app.session_state["icu_settings"].bed_count == 3
    assert widget(app, "number_input", "Beds").value == 3


# ============================================================================== patient monitor
# The view the whole project argues for: a composite score is only actionable if the reader can
# take it apart. So most of these tests are about what sits *beside* the number.
#
# Nothing here asserts an exact score. The engine advances on the wall clock, so how many ticks
# have run by the time the page renders depends on how loaded the machine is - a suite that pins
# 80.0 fails on a slow CI runner and teaches its maintainer to delete tests. Shapes, orderings
# and cross-references between two things on the same screen are stable; single readings are not.

BED_OPTION = re.compile(
    r"^(?P<bed>[\w-]+) · (?P<name>.+) · (?P<glyph>\S+) (?P<word>\w+) (?P<score>\d+)$"
)


def bed_options(app: AppTest) -> list[re.Match[str]]:
    """The Bed dropdown's rendered options, parsed.

    ``format_func`` is applied by the time ``AppTest`` sees them, so ``options`` carries the
    labels a reader picks from while ``value`` carries the patient id underneath - which is
    what makes it possible to check the two against each other.
    """
    selector = widget(app, "selectbox", "Bed")
    matched = [BED_OPTION.match(str(option)) for option in selector.options]
    assert all(matched), f"unparseable bed labels: {selector.options}"
    return [match for match in matched if match is not None]


def test_the_monitor_opens_on_the_worst_bed_not_the_first(tmp_path: Path) -> None:
    """Whose bed is on screen when the page loads is a clinical decision, not a default.

    Checked against the dropdown rendered in the *same* run rather than a remembered patient
    id: the worst bed changes as the ward deteriorates, and a test that pins P002 would be
    asserting the simulator's seed rather than the rule.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Patient monitor")
    assert not app.exception
    worst = max(bed_options(app), key=lambda match: int(match["score"]))
    text = screen_text(app)
    assert worst["name"] in text  # the identity card names the same patient
    assert f">{worst['bed']} · " in text  # and the caption underneath it, the same bed
    assert app.session_state["icu_selected_patient"] == widget(app, "selectbox", "Bed").value


def test_the_bed_dropdown_carries_the_glyph_the_word_and_the_score(tmp_path: Path) -> None:
    """A dropdown has no room for a badge, so the risk band travels as a glyph and a word.

    This is the same rule the badges follow, applied where colour is not available at all: the
    reader picking a bed can see which one is worst without opening it, and without relying on
    a hue that a native ``<select>`` would not render anyway.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Patient monitor")
    options = bed_options(app)
    assert len(options) == 6
    for match in options:
        level = match["word"].upper()
        assert level in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}
        assert match["glyph"] == theme.level_glyph(level)
        assert 0 <= int(match["score"]) <= 100


def test_choosing_a_bed_moves_the_whole_view_to_it(tmp_path: Path) -> None:
    """The dropdown is this view's only navigation, and it has to move more than itself.

    Three things are asserted, because a half-moved view is the failure that actually happens:
    the identity card, the session key the rest of the app navigates by, and the bedside
    controls - whose widget keys are per-patient, so a control left pointing at the previous
    bed would inject an event into the wrong patient.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Patient monitor")
    before = widget(app, "selectbox", "Bed").value
    options = bed_options(app)
    index = next(
        i for i, match in enumerate(options) if f">{match['bed']} · " not in screen_text(app)
    )
    target = options[index]
    widget(app, "selectbox", "Bed").select_index(index).run()

    assert not app.exception
    chosen = app.session_state["icu_selected_patient"]
    assert chosen != before
    text = screen_text(app)
    assert f">{target['bed']} · " in text
    assert target["name"] in text
    assert widget(app, "selectbox", "Event").key == f"icu_event_{chosen}"
    assert widget(app, "button", "Inject").key == f"icu_inject_{chosen}"


def test_the_composite_is_never_shown_alone(tmp_path: Path) -> None:
    """The claim this view exists to make good on.

    Beside the one number are the four things that produced it: the observation trend, the
    NEWS2 parameter scores, the itemised fusion contributions, and the model's class
    probabilities. Each panel also carries a note saying how to read it, because a chart whose
    green band is unexplained is decoration.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Patient monitor")
    text = screen_text(app)
    assert re.search(r"\*\*Vital signs · last \d+ observations\*\*", text)
    for title in ("NEWS2 parameter scores", "Fusion contributions", "Model class probabilities"):
        assert f"**{title}**" in text
    assert "The green band is the NEWS2 zero-score range" in text
    assert "A red outline marks a red score" in text
    assert "These points reconcile to the composite score above" in text

    values = {metric.label: metric.value for metric in app.metric}
    assert re.fullmatch(r"\d+\.\d", values["Composite risk"]), values["Composite risk"]
    assert re.fullmatch(r"\d+/20", values["NEWS2"]), values["NEWS2"]
    # And when it was true: an absolute timestamp makes the reader do the arithmetic.
    assert re.search(r"assessed (?:\d+ s|\d+ m \d+ s|[\d.]+ h) ago", text)


def test_the_probability_panel_says_so_in_words_when_no_model_produced_it(tmp_path: Path) -> None:
    """A bare clone has no artefact, and the fusion layer still returns the class map - filled
    with zeros. Passing that through drew three 0% bars under a note saying nothing was loaded,
    which is a chart that encodes no information sitting where a sentence belongs. Only the
    three real diagnostics are drawn, and the fourth panel explains itself instead."""
    app = run_app(ui_settings(tmp_path, bed_count=4), icu_view="Patient monitor")
    text = screen_text(app)
    assert len(app.get("vega_lite_chart")) == 3
    assert "No trained artefact loaded — scoring runs on NEWS2 and vision alone." in text
    assert "No model prediction for this bed." in text


def test_a_trained_artefact_fills_in_the_fourth_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same rule: when a model *did* produce probabilities, they are drawn
    and the note names the artefact and its confidence, so the reader can tell which version of
    the model they are looking at."""
    cfg = ui_settings(tmp_path, bed_count=4)
    version = tiny_artefact(cfg, monkeypatch)
    app = run_app(cfg, icu_view="Patient monitor")
    text = screen_text(app)
    assert len(app.get("vega_lite_chart")) == 4
    assert f"{version} · confidence " in text
    assert re.search(rf"{re.escape(version)} · confidence \d+%", text)
    assert "No trained artefact loaded" not in text


def test_the_current_observation_reads_as_a_bedside_chart(tmp_path: Path) -> None:
    """Every channel, labelled, with its unit, plus the two things a number cannot carry.

    ACVPU travels as the letter *and* the word - ``A`` is the notation on the paper NEWS2 chart,
    while ``V``, ``P`` and ``U`` mean nothing to a reader who has not seen one - and whether the
    patient is on oxygen is stated in words, because it changes how the SpO₂ beside it scores.
    """
    app = run_app(ui_settings(tmp_path, bed_count=4), icu_view="Patient monitor")
    values = {metric.label: metric.value for metric in app.metric}
    for _attr, label, unit, _spec in ui.VITAL_ROWS:
        assert label in values, label
        if unit:
            assert values[label].endswith(f" {unit}")
        else:
            assert " " not in values[label]  # the shock index is a ratio, not a measurement

    text = screen_text(app)
    assert re.search(
        r"ACVPU <b>[ACVPU](?: \([A-Za-z ]+\))?</b> · GCS <b>(?:\d+|—)</b> · "
        r"(?:on supplemental O₂|room air)",
        text,
    ), "the consciousness line is missing or malformed"


def test_the_recommended_response_is_the_rcp_text_for_the_score_beside_it(tmp_path: Path) -> None:
    """The one sentence on this page that tells a reader what to *do*, cross-checked against the
    number it is derived from.

    Asserting the sentence is present would pass on a hard-coded string. Asserting it is the
    Royal College of Physicians response for the NEWS2 total displayed two inches away is what
    makes the pair trustworthy. Below 5 the red-score sentence is also admissible, because a
    single parameter at 3 mandates review on its own.
    """
    app = run_app(ui_settings(tmp_path, bed_count=6), icu_view="Patient monitor")
    text = screen_text(app)
    total = int({metric.label: metric.value for metric in app.metric}["NEWS2"].split("/")[0])
    banded = next(
        response for low, high, _tier, response in RESPONSE_BY_TOTAL if low <= total <= high
    )
    admissible = {banded} if total >= 5 else {banded, RED_SCORE_RESPONSE}
    shown = re.search(r"<b>Recommended response\.</b> (.+?)</div>", text)
    assert shown is not None, "no recommended response on the page"
    assert html.unescape(shown[1]) in admissible


def live_engine(cfg: Settings) -> MonitoringEngine:
    """The very engine the app under test is driving.

    ``get_app_state`` is a ``st.cache_resource`` keyed on the settings fingerprint and shared by
    the whole process, so asking for the same fingerprint from the test returns the same object
    the script run built. That is what makes "did the button change anything, or only say so"
    an answerable question rather than an act of faith in a toast.
    """
    return app_state.get_app_state(app_state.settings_fingerprint(cfg)).engine()


def test_injecting_an_event_names_the_event_and_the_bed(tmp_path: Path) -> None:
    """The demonstration path: a reader who cannot wait for a patient to deteriorate drives one.

    The toast names the event in the words the dropdown used and the bed it landed on - the
    engine returns the display label, not a sentence, and a bare "Desaturation episode" in a
    green box does not say which of the six beds now has one. The panel then reports it as
    running, on the next run rather than this one, because the active list is read before the
    button branch executes.
    """
    cfg = ui_settings(tmp_path, bed_count=4, tick_seconds=30.0)
    app = run_app(cfg, icu_view="Patient monitor")
    patient_id = app.session_state["icu_selected_patient"]
    events = widget(app, "selectbox", "Event")
    label = events.options[1]
    events.select_index(1)
    widget(app, "button", "Inject").click().run()

    assert not app.exception
    assert [message.value for message in app.success] == [f"{label} started for {patient_id}."]
    assert label in live_engine(cfg).active_events(patient_id)

    app.run()
    assert f"Running: {label}" in screen_text(app)


def test_an_event_the_provider_refuses_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused injection has to say so. ``inject_event`` returns ``None`` for an unknown slug
    or an unknown bed, and a control that reports nothing on failure trains the reader to
    distrust the ones that do."""
    monkeypatch.setattr(MonitoringEngine, "inject_event", lambda *_a, **_k: None)
    app = run_app(ui_settings(tmp_path, bed_count=4), icu_view="Patient monitor")
    widget(app, "button", "Inject").click().run()

    assert not app.exception
    assert "That event could not be started for this patient." in [
        warning.value for warning in app.warning
    ]
    assert not app.success


def test_the_trajectory_control_writes_through_to_the_engine(tmp_path: Path) -> None:
    """A control that reports success and changes nothing is the worst kind of control.

    So the engine is inspected directly rather than the toast believed. ``tick_seconds`` is
    lifted to 30 for this pair so the ward cannot advance mid-test: the simulator moves patients
    between states on its own, and a spontaneous transition would look exactly like a control
    that did not stick.
    """
    cfg = ui_settings(tmp_path, bed_count=4, tick_seconds=30.0)
    app = run_app(cfg, icu_view="Patient monitor")
    patient_id = app.session_state["icu_selected_patient"]
    engine = live_engine(cfg)
    before = engine.patient(patient_id)
    assert before is not None
    target = "recovering" if before.state.value != "recovering" else "stable"

    widget(app, "selectbox", "Clinical trajectory").set_value(target)
    widget(app, "button", "Apply").click().run()

    assert not app.exception
    assert f"{patient_id} set to {target}." in [message.value for message in app.success]
    record = engine.patient(patient_id)
    assert record is not None and record.state.value == target


def test_the_oxygen_controls_write_through_to_the_engine(tmp_path: Path) -> None:
    """Both halves of the oxygen state, because both change the score.

    Supplemental oxygen scores 2 on NEWS2 by itself, and scale 2 moves the SpO₂ target down to
    88-92% for chronic hypercapnic respiratory failure - a patient on the wrong scale is scored
    against the wrong band, so the radio has to reach the simulator and not just the widget.
    """
    cfg = ui_settings(tmp_path, bed_count=4, tick_seconds=30.0)
    app = run_app(cfg, icu_view="Patient monitor")
    patient_id = app.session_state["icu_selected_patient"]
    engine = live_engine(cfg)
    before = engine.patient(patient_id)
    assert before is not None
    wanted = not before.on_supplemental_oxygen

    widget(app, "toggle", "Supplemental O₂").set_value(wanted)
    widget(app, "radio", "SpO₂ target scale").set_value(2)
    widget(app, "button", "Update O₂").click().run()

    assert not app.exception
    assert "Oxygen settings updated." in [message.value for message in app.success]
    record = engine.patient(patient_id)
    assert record is not None
    assert record.on_supplemental_oxygen is wanted
    assert record.spo2_scale == 2


def test_a_provider_that_cannot_be_steered_says_which_one_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay providers offer no events, and the whole control panel goes with them.

    An empty card would leave the reader unable to tell a missing feature from an inapplicable
    one, so the panel states which provider is in charge and why nothing can be steered. The
    bedside adjustments go too - they would be refused for the same reason - which is why the
    sentence covers both rather than only the injection.
    """
    monkeypatch.setattr(MonitoringEngine, "available_events", lambda _self: {})
    app = run_app(ui_settings(tmp_path, bed_count=4), icu_view="Patient monitor")
    text = screen_text(app)

    assert not app.exception
    assert "**Inject a clinical event**" in text
    assert "Unavailable: Simulated ward (synthetic physiology) replays recorded data" in text
    assert "neither an injected event nor a bedside adjustment can steer its physiology" in text
    assert "**Bedside adjustments**" not in text
    assert not [button for button in app.button if button.label == "Inject"]


def test_the_camera_panel_is_present_and_honest_with_no_camera(tmp_path: Path) -> None:
    """A bare clone has no camera, and this panel is where that has to be said out loud.

    The panel is drawn either way - the reader is told the channel exists and is off, not left
    to guess whether the build has vision at all - and none of the detection read-outs are
    drawn, because a posture of "unknown" beside a motion index of 0.000 reads as a measurement.
    """
    app = run_app(ui_settings(tmp_path, bed_count=4), icu_view="Patient monitor")
    text = screen_text(app)

    assert "**Bedside camera**" in text
    assert "Vision disabled → Detection disabled" in text
    assert "Inactive — Vision disabled" in text
    for readout in ("Posture", "Motion index", "Fall suspected", "Latency"):
        assert readout not in text
    assert not [button for button in app.button if button.label == "Point camera here"]


def test_the_per_bed_alert_cards_drop_the_prefix_and_still_acknowledge(tmp_path: Path) -> None:
    """The ward's alert cards lead with the patient id; here that column is the whole page.

    ``tick_seconds`` is lifted to 30 so no live tick runs during the test, which leaves the
    back-dated warm-up as the only source of alerts - and that is seeded, so the bed this page
    opens on reliably has some. Acknowledging is checked against the manager rather than the
    card: the button is wired to a callback, and a callback that does not fire looks identical
    on screen until the next rerun.
    """
    cfg = ui_settings(tmp_path, bed_count=6, tick_seconds=30.0)
    app = run_app(cfg, icu_view="Patient monitor")
    patient_id = app.session_state["icu_selected_patient"]
    text = screen_text(app)
    cards = [button for button in app.button if str(button.key).startswith("patient_alert_")]

    assert "#### Alerts for this bed" in text
    assert cards, "the worst bed raised no alerts during a 12-tick warm-up"
    assert f"{patient_id} · " not in text  # the id is the page, not a prefix on every card
    assert any(kind.label in text for kind in AlertKind)

    alert_id = int(str(cards[0].key).rsplit("_", 1)[1])
    cards[0].click().run()

    assert not app.exception
    acknowledged = {
        alert.alert_id: alert.acknowledged_by
        for alert in live_engine(cfg).alerts.history
        if not alert.is_open
    }
    assert acknowledged.get(alert_id) == "dashboard"
    assert "✓ acknowledged" in screen_text(app)


# The identity card is rendered on its own below. The three cases that matter - an override, no
# override, and a NEWS2 that could not be scored - are properties of the bed rather than of the
# ward, and a seeded simulator does not hand them over on request.


def _view_script(module, name, payload):
    """The script ``AppTest`` executes: import one view module, call one function in it.

    Same two constraints as ``_component_script``, for the same reason - ``from_function``
    re-executes only these lines in a fresh module, so the import happens in the body and no
    parameter is annotated.
    """
    import importlib

    getattr(importlib.import_module(module), name)(**payload)


def render_part(module: str, name: str, **payload: Any) -> AppTest:
    """Render one view helper, alone, in a script of its own."""
    return AppTest.from_function(
        _view_script, default_timeout=30, args=(module, name, payload)
    ).run()


# Taken off the imported module rather than retyped, so a move or a rename is an import error
# here instead of a runtime "no module named" inside the AppTest script, where the traceback
# arrives as ``app.exception`` and reads like the view itself is broken.
PATIENT_VIEW = patient_view.__name__


def identity_bed(**assessment: Any) -> BedSnapshot:
    """A bed for the identity card, admitted 16 hours ago on the real clock.

    ``los_hours()`` reads ``datetime.now``, so the fixture's admission has to be relative to it -
    a fixed epoch would render a length of stay in the tens of thousands of hours.
    """
    patient = make_patient(
        "P007",
        bed="ICU-07",
        display_name="Ada <script>alert(1)</script> Lovelace",
        age=71,
        sex="M",
        primary_diagnosis="Septic shock",
        state=ClinicalState.DETERIORATING,
        admitted_at=datetime.now(timezone.utc) - timedelta(hours=16),
    )
    return make_bed(
        "P007", patient=patient, assessment=make_assessment(patient_id="P007", **assessment)
    )


def test_the_identity_card_answers_who_and_how_sick() -> None:
    """Name, band, bed, age, sex, diagnosis, length of stay, trajectory - and the score twice.

    The composite carries a decimal and NEWS2 carries its ceiling, because "16" alone invites
    the reader to compare it against a composite out of 100. Patient-supplied text is escaped:
    this card is drawn with ``unsafe_allow_html``, so a display name is the one place a stored
    string could otherwise become markup.
    """
    app = render_part(PATIENT_VIEW, "_identity", bed=identity_bed())
    assert not app.exception
    text = screen_text(app)

    assert "ICU-07 · 71y M · Septic shock · LOS 16 h · state Deteriorating" in text
    assert f"{theme.level_glyph('HIGH')} High" in text
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Composite risk"] == "72.5"
    assert values["NEWS2"] == "7/20"
    assert "Hourly observations; urgent review by a clinician." in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<script>" not in text


def test_an_overridden_score_is_flagged_with_every_reason() -> None:
    """An override is the one case where the arithmetic does not explain the number.

    The composite was lifted to a mandated floor, so the panel lists the reasons rather than
    leaving the reader to reconcile a sum that no longer adds up. Flag, capitals and colour, all
    three - the glyph and the words are what make the colour legal.
    """
    reasons = ("NEWS2 12 ≥ 7 → HIGH", "SpO₂ 84% ≤ 85 → CRITICAL")
    bed = identity_bed(level=RiskLevel.CRITICAL, composite_score=88.0, overrides=reasons)
    app = render_part(PATIENT_VIEW, "_identity", bed=bed)
    text = screen_text(app)

    assert "⚑ CLINICAL OVERRIDES APPLIED" in text
    for reason in reasons:
        assert f"<li style='margin:0.12rem 0'>{html.escape(reason)}</li>" in text
    assert theme.STATUS["serious"] in text
    assert f"{theme.level_glyph('CRITICAL')} Critical" in text


def test_a_bed_whose_news2_could_not_be_scored_says_so() -> None:
    """No NEWS2 means no total and no recommended response - not a zero and not a stale one.

    A dash where the reader expects a number is a smaller failure than a plausible number that
    was never measured, and the response line disappears with it rather than advising a course
    of action derived from nothing.
    """
    app = render_part(PATIENT_VIEW, "_identity", bed=identity_bed(news2=None))
    text = screen_text(app)

    assert {metric.label: metric.value for metric in app.metric}["NEWS2"] == "—"
    assert "Recommended response" not in text
    assert "ICU-07 · 71y M · Septic shock" in text  # the rest of the card still renders
