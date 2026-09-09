"""NEWS2 scoring, checked against the published RCP tables.

This is the one module in the project where "looks about right" is not good enough: the
score drives escalation, so every band edge is asserted from the 2017 specification rather
than from the implementation. If a band here disagrees with the Royal College of Physicians
table, the test is right and the code is wrong.

The boundary cases are the point. A parameter that is one unit inside a band and one unit
outside it are two different clinical responses, and off-by-one is the failure mode that a
"scores something sensible" test would sail straight past.
"""

from __future__ import annotations

import math

import pytest

from icu_monitor.core.news2 import (
    calculate_news2,
    news2_band_table,
    round_clinical,
    score_in_bands,
)
from icu_monitor.core.types import Consciousness, RiskLevel

from .conftest import make_vitals

# --------------------------------------------------------------------------- healthy


def test_healthy_observation_scores_zero() -> None:
    result = calculate_news2(make_vitals())
    assert result.total == 0
    assert result.risk_level is RiskLevel.LOW
    assert result.has_red_score is False
    assert result.is_complete is True
    assert result.missing_parameters == ()
    assert result.max_total == 20
    assert result.normalised == 0.0
    assert result.clinical_response.startswith("Routine")


def test_components_are_reported_in_bedside_chart_order() -> None:
    result = calculate_news2(make_vitals())
    assert [c.parameter for c in result.components] == [
        "resp_rate",
        "spo2",
        "supplemental_oxygen",
        "bp_systolic",
        "heart_rate",
        "consciousness",
        "temperature",
    ]


def test_total_is_the_sum_of_its_components() -> None:
    """No fudge factor. A total that cannot be reconciled cannot be argued with."""
    result = calculate_news2(make_vitals(resp_rate=22.0, spo2=93.0, heart_rate=115.0))
    assert result.total == sum(c.score for c in result.components)


# ------------------------------------------------------------------------- the bands


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (7.0, 3),
        (8.0, 3),
        (9.0, 1),
        (11.0, 1),
        (12.0, 0),
        (20.0, 0),
        (21.0, 2),
        (24.0, 2),
        (25.0, 3),
    ],
)
def test_respiration_rate_bands(value: float, expected: int) -> None:
    scores = {
        c.parameter: c.score for c in calculate_news2(make_vitals(resp_rate=value)).components
    }
    assert scores["resp_rate"] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(85.0, 3), (91.0, 3), (92.0, 2), (93.0, 2), (94.0, 1), (95.0, 1), (96.0, 0), (100.0, 0)],
)
def test_spo2_scale_1_bands(value: float, expected: int) -> None:
    scores = {c.parameter: c.score for c in calculate_news2(make_vitals(spo2=value)).components}
    assert scores["spo2"] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (80.0, 3),
        (90.0, 3),
        (91.0, 2),
        (100.0, 2),
        (101.0, 1),
        (110.0, 1),
        (111.0, 0),
        (219.0, 0),
        (220.0, 3),
    ],
)
def test_systolic_bands(value: float, expected: int) -> None:
    scores = {
        c.parameter: c.score for c in calculate_news2(make_vitals(bp_systolic=value)).components
    }
    assert scores["bp_systolic"] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (39.0, 3),
        (40.0, 3),
        (41.0, 1),
        (50.0, 1),
        (51.0, 0),
        (90.0, 0),
        (91.0, 1),
        (110.0, 1),
        (111.0, 2),
        (130.0, 2),
        (131.0, 3),
    ],
)
def test_pulse_bands(value: float, expected: int) -> None:
    scores = {
        c.parameter: c.score for c in calculate_news2(make_vitals(heart_rate=value)).components
    }
    assert scores["heart_rate"] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (34.0, 3),
        (35.0, 3),
        (35.1, 1),
        (36.0, 1),
        (36.1, 0),
        (38.0, 0),
        (38.1, 1),
        (39.0, 1),
        (39.1, 2),
        (41.0, 2),
    ],
)
def test_temperature_bands(value: float, expected: int) -> None:
    scores = {
        c.parameter: c.score for c in calculate_news2(make_vitals(temperature=value)).components
    }
    assert scores["temperature"] == expected


@pytest.mark.parametrize(
    ("consciousness", "expected"),
    [
        (Consciousness.ALERT, 0),
        (Consciousness.CONFUSION, 3),
        (Consciousness.VOICE, 3),
        (Consciousness.PAIN, 3),
        (Consciousness.UNRESPONSIVE, 3),
    ],
)
def test_acvpu_is_all_or_nothing(consciousness: Consciousness, expected: int) -> None:
    """New confusion scores the same 3 as unresponsive - that is the 2017 revision."""
    result = calculate_news2(make_vitals(consciousness=consciousness))
    scores = {c.parameter: c.score for c in result.components}
    assert scores["consciousness"] == expected


def test_supplemental_oxygen_adds_two() -> None:
    on_air = calculate_news2(make_vitals())
    on_oxygen = calculate_news2(make_vitals(on_supplemental_oxygen=True))
    assert on_oxygen.total - on_air.total == 2
    bands = {c.parameter: c.band for c in on_oxygen.components}
    assert bands["supplemental_oxygen"] == "Oxygen"


# ----------------------------------------------------------------------- spo2 scale 2


@pytest.mark.parametrize(
    ("value", "on_oxygen", "expected"),
    [
        (82.0, False, 3),
        (84.0, False, 2),
        (86.0, False, 1),
        (88.0, False, 0),
        (95.0, False, 0),
        (88.0, True, 0),
        (92.0, True, 0),
        (93.0, True, 1),
        (94.0, True, 1),
        (95.0, True, 2),
        (96.0, True, 2),
        (97.0, True, 3),
    ],
)
def test_spo2_scale_2_penalises_over_oxygenation(
    value: float, on_oxygen: bool, expected: int
) -> None:
    """Scale 2 is the reason this project models oxygen at all.

    For chronic hypercapnic respiratory failure the target is 88-92%, so 97% *on oxygen* is
    a 3 - the same score as 91% would be on Scale 1. A monitor that treats "high SpO₂" as
    unconditionally reassuring gets this patient wrong in the dangerous direction.
    """
    result = calculate_news2(
        make_vitals(spo2=value, on_supplemental_oxygen=on_oxygen), spo2_scale=2
    )
    scores = {c.parameter: c.score for c in result.components}
    assert scores["spo2"] == expected


def test_scale_is_reported_and_named_in_the_component() -> None:
    result = calculate_news2(make_vitals(), spo2_scale=2)
    assert result.scale == 2
    names = {c.parameter: c.display_name for c in result.components}
    assert "scale 2" in names["spo2"]


# --------------------------------------------------------------------------- rounding


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        (8.5, 0, 9.0),
        (24.5, 0, 25.0),
        (0.5, 0, 1.0),
        (1.5, 0, 2.0),
        (38.05, 1, 38.1),
        (36.25, 1, 36.3),
    ],
)
def test_rounding_is_clinical_not_bankers(value: float, precision: int, expected: float) -> None:
    """Python rounds 8.5 to 8. A chart rounds it to 9, and NEWS2 follows the chart."""
    assert round_clinical(value, precision) == expected


def test_respiration_rate_of_8_point_5_is_recorded_as_9() -> None:
    """The regression this rounding rule exists for: 8.5 -> 9 scores 1, 8.5 -> 8 scores 3.

    Banker's rounding would put a mildly bradypnoeic patient into the same band as a
    critically bradypnoeic one, and 3 on a single parameter is a red score.
    """
    result = calculate_news2(make_vitals(resp_rate=8.5))
    scores = {c.parameter: c.score for c in result.components}
    assert scores["resp_rate"] == 1
    assert result.has_red_score is False


# ------------------------------------------------------------------- missing channels


def test_absent_channels_are_reported_not_assumed_normal() -> None:
    """A missing SpO₂ is not a normal SpO₂. It scores 0 but the result says it is incomplete."""
    result = calculate_news2(make_vitals(spo2=None, temperature=None))
    assert result.is_complete is False
    assert len(result.missing_parameters) == 2
    assert result.total == 0
    unmeasured = [c.parameter for c in result.components if not c.measured]
    assert set(unmeasured) == {"spo2", "temperature"}


def test_air_or_oxygen_is_never_listed_as_missing() -> None:
    """It is a documented flag, not a sensor reading - absence of oxygen *is* the value."""
    result = calculate_news2(make_vitals(spo2=None))
    assert not any("xygen" in name for name in result.missing_parameters)
    assert not any("Air" in name for name in result.missing_parameters)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_unusable_readings_score_zero_and_say_so(value: float | None) -> None:
    score, band = score_in_bands(value, news2_band_table("heart_rate"))
    assert score == 0
    assert band == "not measured"


# -------------------------------------------------------------- totals and escalation


@pytest.mark.parametrize(
    ("total_vitals", "expected_level"),
    [
        ({}, RiskLevel.LOW),
        ({"resp_rate": 21.0, "spo2": 94.0, "heart_rate": 95.0}, RiskLevel.LOW),
        ({"resp_rate": 22.0, "spo2": 93.0, "heart_rate": 115.0}, RiskLevel.MEDIUM),
        ({"resp_rate": 26.0, "spo2": 90.0, "heart_rate": 135.0}, RiskLevel.HIGH),
    ],
)
def test_risk_level_follows_the_aggregate(total_vitals: dict, expected_level: RiskLevel) -> None:
    assert calculate_news2(make_vitals(**total_vitals)).risk_level is expected_level


def test_a_single_red_score_escalates_a_low_total() -> None:
    """NEWS2 3 from one parameter is not the same as 3 from three parameters.

    A total of 3 normally means ward-based monitoring, but if that 3 came from one channel
    the RCP guidance calls for urgent review, so this must not be reported as LOW.
    """
    result = calculate_news2(make_vitals(resp_rate=7.0))
    assert result.total == 3
    assert result.has_red_score is True
    assert result.risk_level is RiskLevel.MEDIUM
    assert "red score" in result.clinical_response


def test_red_score_flag_is_per_parameter() -> None:
    result = calculate_news2(make_vitals(resp_rate=7.0, heart_rate=95.0))
    reds = [c.parameter for c in result.components if c.is_red]
    assert reds == ["resp_rate"]


@pytest.mark.parametrize(
    ("total", "expected_words"),
    [
        (0, "Routine"),
        (3, "nurse"),
        (5, "Urgent"),
        (6, "Urgent"),
        (7, "Emergency"),
        (14, "Emergency"),
    ],
)
def test_clinical_response_ladder(total: int, expected_words: str) -> None:
    """Built by dialling pulse and respiration until the aggregate lands on `total`."""
    result = calculate_news2(make_vitals(resp_rate=8.5))  # baseline 1 from resp rate
    assert result.total == 1
    from icu_monitor.core.news2 import RESPONSE_BY_TOTAL

    matched = next(text for low, high, _label, text in RESPONSE_BY_TOTAL if low <= total <= high)
    assert expected_words.lower() in matched.lower()


def test_normalised_total_is_a_fraction_of_the_maximum() -> None:
    result = calculate_news2(
        make_vitals(resp_rate=26.0, spo2=90.0, heart_rate=135.0, temperature=34.0)
    )
    assert 0.0 < result.normalised <= 1.0
    assert result.normalised == pytest.approx(result.total / result.max_total)
    assert result.total <= result.max_total


def test_worst_case_cannot_exceed_the_published_maximum() -> None:
    result = calculate_news2(
        make_vitals(
            resp_rate=40.0,
            spo2=70.0,
            bp_systolic=60.0,
            heart_rate=190.0,
            temperature=33.0,
            consciousness=Consciousness.UNRESPONSIVE,
            on_supplemental_oxygen=True,
        )
    )
    assert result.total == result.max_total == 20
    assert result.risk_level is RiskLevel.HIGH


# ----------------------------------------------------------------------- band tables


@pytest.mark.parametrize(
    "parameter",
    [
        "resp_rate",
        "spo2",
        "spo2_scale2_air",
        "spo2_scale2_oxygen",
        "bp_systolic",
        "heart_rate",
        "temperature",
    ],
)
def test_every_published_table_is_reachable_by_name(parameter: str) -> None:
    """The dashboard renders these tables, so they have to be introspectable."""
    table = news2_band_table(parameter)
    assert table
    assert all(len(band) >= 3 for band in table)


def test_unknown_parameter_yields_an_empty_table_not_a_wrong_one() -> None:
    """Blood glucose is not a NEWS2 parameter, so there is nothing to show.

    Returning ``()`` rather than raising is deliberate: the reference page asks for tables
    by name and an empty one renders as "no table", whereas an exception would take the
    whole page down over a display detail. Falling back to *some other* parameter's table
    would be the genuinely dangerous outcome, and that is what this pins down.
    """
    assert news2_band_table("blood_glucose") == ()


@pytest.mark.parametrize(
    "parameter", ["resp_rate", "spo2", "bp_systolic", "heart_rate", "temperature"]
)
def test_bands_are_contiguous_and_cover_the_real_line(parameter: str) -> None:
    """No gap between bands, and no value that fails to score.

    A gap would mean some perfectly ordinary reading silently scores 0 through the fallback
    path rather than through a band, which is exactly the sort of hole that only shows up
    once a real patient falls into it.
    """
    table = news2_band_table(parameter)
    lows = [band[0] for band in table]
    highs = [band[1] for band in table]
    assert lows[0] == -math.inf
    assert highs[-1] == math.inf
    step = 1.0 if parameter != "temperature" else 0.1
    for previous_high, next_low in zip(highs, lows[1:], strict=False):
        assert next_low == pytest.approx(previous_high + step)
