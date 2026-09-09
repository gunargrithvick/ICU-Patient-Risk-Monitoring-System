"""Risk fusion: the composite score, its itemisation, and the hard overrides.

The invariant that matters most here is **reconciliation**: the itemised factors the
dashboard draws must add up to the composite number it prints beside them. A score that
cannot be taken apart is a score a clinician has to take on faith, and this project's whole
argument is that they should not have to. Several tests below do nothing but sum the factors
and compare.

The second theme is that overrides only ever escalate. Fusion is allowed to notice that a
patient is sicker than the weighted average suggests; it is never allowed to talk the score
down, because the failure mode of a monitor that reasons its way out of an alarm is the one
that kills people.
"""

from __future__ import annotations

import json

import pytest

from icu_monitor.config import Settings
from icu_monitor.core.fusion import (
    AGITATION_THRESHOLD,
    MLPrediction,
    band_for_score,
    fuse_risk,
    vision_severity,
)
from icu_monitor.core.news2 import calculate_news2
from icu_monitor.core.types import Consciousness, Posture, RiskLevel, VisionSignal

from .conftest import make_vitals


def assess(config: Settings, **kwargs: object):
    """``fuse_risk`` with the boilerplate filled in."""
    vitals = kwargs.pop("vitals", None) or make_vitals()
    scale = int(kwargs.pop("spo2_scale", 1))  # type: ignore[call-overload]
    news2 = kwargs.pop("news2", ...)
    if news2 is ...:
        news2 = calculate_news2(vitals, spo2_scale=scale)
    return fuse_risk(
        patient_id="P001",
        vitals=vitals,
        news2=news2,
        config=config,
        **kwargs,  # type: ignore[arg-type]
    )


def prediction(low: float, medium: float, high: float, **kwargs: object) -> MLPrediction:
    probabilities = {"LOW": low, "MEDIUM": medium, "HIGH": high}
    level = RiskLevel.coerce(max(probabilities, key=lambda k: probabilities[k]))
    values: dict[str, object] = {
        "level": level,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
        "available": True,
        "model_version": "test-model",
    }
    values.update(kwargs)
    return MLPrediction(**values)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- the bands


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, RiskLevel.LOW),
        (39.9, RiskLevel.LOW),
        (40.0, RiskLevel.MEDIUM),
        (64.9, RiskLevel.MEDIUM),
        (65.0, RiskLevel.HIGH),
        (84.9, RiskLevel.HIGH),
        (85.0, RiskLevel.CRITICAL),
        (100.0, RiskLevel.CRITICAL),
    ],
)
def test_band_edges(config: Settings, score: float, expected: RiskLevel) -> None:
    assert band_for_score(score, config) is expected


def test_bands_come_from_config_not_from_constants(tmp_path) -> None:
    """A ward that wants to escalate earlier changes settings, not source."""
    strict = Settings(
        project_root=tmp_path,
        database_url="sqlite://",
        composite_medium_threshold=10.0,
        composite_high_threshold=20.0,
        composite_critical_threshold=30.0,
    )
    assert band_for_score(15.0, strict) is RiskLevel.MEDIUM
    assert band_for_score(35.0, strict) is RiskLevel.CRITICAL


# --------------------------------------------------------------------- reconciliation


def factor_total(assessment) -> float:
    return sum(factor.points for factor in assessment.factors)


@pytest.mark.parametrize(
    "vitals_kwargs",
    [
        {},
        {"spo2": 93.0},
        {"resp_rate": 26.0, "spo2": 90.0, "heart_rate": 135.0},
        {"consciousness": Consciousness.VOICE, "gcs": 12.0},
        {"bp_systolic": 78.0},
        {"temperature": 33.5},
        {"spo2": None, "temperature": None},
    ],
)
def test_factors_always_reconcile_to_the_composite(config: Settings, vitals_kwargs: dict) -> None:
    """The dashboard draws these bars beside the number; they have to add up to it.

    An override lift is itself itemised as a factor, so this holds even when a hard rule has
    pushed the score above the weighted sum of its parts.
    """
    assessment = assess(config, vitals=make_vitals(**vitals_kwargs))
    assert factor_total(assessment) == pytest.approx(assessment.composite_score, abs=0.06)


def test_override_lift_is_itemised_rather_than_hidden(config: Settings) -> None:
    assessment = assess(
        config, vision=VisionSignal(available=True, patient_present=True, fall_suspected=True)
    )
    assert assessment.overrides
    sources = [f.source for f in assessment.factors]
    assert "override" in sources
    assert factor_total(assessment) == pytest.approx(assessment.composite_score, abs=0.06)


def test_top_factors_are_ordered_by_contribution(config: Settings) -> None:
    assessment = assess(config, vitals=make_vitals(resp_rate=26.0, spo2=88.0, heart_rate=132.0))
    points = [factor.points for factor in assessment.top_factors]
    assert points == sorted(points, reverse=True)


def test_assessment_is_json_serialisable(config: Settings) -> None:
    """The API returns this verbatim, so a stray enum in the payload is a 500."""
    assessment = assess(config, ml=prediction(0.2, 0.3, 0.5))
    text = json.dumps(assessment.as_dict(), default=str)
    assert "composite_score" in text


# ------------------------------------------------------------------- no data at all


def test_no_measured_channel_is_unknown_not_zero_risk(config: Settings) -> None:
    """An unplugged monitor must never read as a healthy patient.

    Scoring 0 and displaying LOW would be indistinguishable from a stable patient; UNKNOWN
    forces the reader to notice that nothing is being measured.
    """
    blank = make_vitals(
        heart_rate=None, spo2=None, bp_systolic=None, resp_rate=None, temperature=None
    )
    assessment = assess(config, vitals=blank)
    assert assessment.level is RiskLevel.UNKNOWN
    assert assessment.composite_score == 0.0


def test_a_single_channel_is_enough_to_score(config: Settings) -> None:
    partial = make_vitals(spo2=None, bp_systolic=None, resp_rate=None, temperature=None)
    assessment = assess(config, vitals=partial)
    assert assessment.level is not RiskLevel.UNKNOWN


# ------------------------------------------------------------------ the model channel


def test_ml_severity_is_a_probability_weighted_expectation() -> None:
    """MEDIUM counts half, HIGH counts full, and the denominator renormalises."""
    assert prediction(0.2, 0.3, 0.5).severity == pytest.approx(0.65)
    assert prediction(1.0, 0.0, 0.0).severity == pytest.approx(0.0)
    assert prediction(0.0, 0.0, 1.0).severity == pytest.approx(1.0)
    assert prediction(0.0, 1.0, 0.0).severity == pytest.approx(0.5)


def test_unnormalised_probabilities_are_renormalised() -> None:
    assert prediction(2.0, 3.0, 5.0).severity == pytest.approx(0.65)


def test_unavailable_model_contributes_nothing() -> None:
    assert MLPrediction().severity == 0.0
    assert MLPrediction().available is False


def test_missing_model_does_not_stop_scoring(config: Settings) -> None:
    """A bare clone has no artefact. It still has to monitor patients."""
    assessment = assess(config, vitals=make_vitals(spo2=90.0, resp_rate=25.0), ml=None)
    assert assessment.model_available is False
    assert assessment.composite_score > 0
    model_factors = [f for f in assessment.factors if f.source == "model"]
    assert len(model_factors) == 1
    assert model_factors[0].points == 0.0
    assert "unavailable" in model_factors[0].description


def test_an_offline_channel_surrenders_its_weight(tmp_path) -> None:
    """The bug this guards: a switched-off camera used to cap the whole scale.

    With ``w_vision`` spending 0.15 on a channel that always reports 0, the most a
    camera-less ward could score was 85 - so ``CRITICAL`` was unreachable on the default
    configuration, which ships with the camera off. Renormalising over the live channels
    means the same physiology reads the same whether or not a camera happens to be wired up.
    """
    cfg = Settings(project_root=tmp_path, database_url="sqlite://")
    dying = make_vitals(
        resp_rate=32.0,
        spo2=80.0,
        bp_systolic=70.0,
        heart_rate=160.0,
        temperature=33.0,
        consciousness=Consciousness.UNRESPONSIVE,
        gcs=5.0,
        on_supplemental_oxygen=True,
    )
    no_camera = assess(cfg, vitals=dying, ml=prediction(0.01, 0.04, 0.95))
    assert no_camera.level is RiskLevel.CRITICAL
    assert no_camera.composite_score > cfg.composite_critical_threshold

    news2_only = assess(cfg, vitals=dying)
    assert news2_only.composite_score == pytest.approx(100.0)
    assert news2_only.level is RiskLevel.CRITICAL


def test_an_offline_channel_says_where_its_weight_went(config: Settings) -> None:
    assessment = assess(config, vitals=make_vitals(spo2=90.0))
    offline = [
        f for f in assessment.factors if f.points == 0.0 and "redistributed" in f.description
    ]
    assert {f.source for f in offline} == {"model", "vision"}


def test_a_confident_high_prediction_raises_the_score(config: Settings) -> None:
    calm = assess(config, ml=prediction(0.9, 0.08, 0.02))
    alarming = assess(config, ml=prediction(0.05, 0.15, 0.80))
    assert alarming.composite_score > calm.composite_score
    assert alarming.model_available is True
    assert alarming.ml_level is RiskLevel.HIGH


def test_zero_weight_silences_a_channel(tmp_path) -> None:
    """A ward that does not trust the model can turn it off without editing code."""
    news2_only = Settings(
        project_root=tmp_path,
        database_url="sqlite://",
        weight_ml=0.0,
        weight_news2=1.0,
        weight_vision=0.0,
    )
    calm = assess(news2_only, ml=prediction(0.9, 0.08, 0.02))
    alarming = assess(news2_only, ml=prediction(0.02, 0.08, 0.90))
    assert calm.composite_score == pytest.approx(alarming.composite_score)


# ----------------------------------------------------------------- the vision channel


@pytest.mark.parametrize(
    ("kwargs", "expected_slug"),
    [
        ({"fall_suspected": True}, "fall"),
        ({"bed_exit_suspected": True}, "bed_exit"),
        ({"patient_present": False}, "absent"),
        ({"patient_present": True, "motion_index": 0.9}, "agitation"),
        ({"patient_present": True, "posture": Posture.RECUMBENT}, "recumbent_still"),
    ],
)
def test_vision_severity_ranks_the_events(kwargs: dict, expected_slug: str) -> None:
    signal = VisionSignal(available=True, **kwargs)
    severity, slug, description = vision_severity(signal)
    assert slug == expected_slug
    assert description
    assert 0.0 <= severity <= 1.0


def test_vision_events_are_ordered_worst_first() -> None:
    def severity_of(**kwargs: object) -> float:
        return vision_severity(VisionSignal(available=True, **kwargs))[0]  # type: ignore[arg-type]

    fall = severity_of(fall_suspected=True)
    exit_ = severity_of(bed_exit_suspected=True)
    absent = severity_of(patient_present=False)
    agitated = severity_of(patient_present=True, motion_index=0.9)
    assert fall > exit_ > absent > agitated > 0.0


def test_unavailable_camera_is_not_an_event() -> None:
    severity, _slug, _description = vision_severity(None)
    assert severity == 0.0
    assert vision_severity(VisionSignal(available=False))[0] == 0.0


def test_agitation_needs_motion_above_the_threshold() -> None:
    """Below the threshold a moving patient is just a patient moving."""
    below = VisionSignal(
        available=True, patient_present=True, motion_index=AGITATION_THRESHOLD - 0.01
    )
    above = VisionSignal(
        available=True, patient_present=True, motion_index=AGITATION_THRESHOLD + 0.01
    )
    assert vision_severity(below)[1] != "agitation"
    assert vision_severity(above)[1] == "agitation"


# ----------------------------------------------------------------------- the overrides


def test_a_suspected_fall_is_critical_whatever_the_vitals_say(config: Settings) -> None:
    """The one rule that outranks everything. A patient on the floor is not "low risk"."""
    assessment = assess(
        config,
        vitals=make_vitals(),
        vision=VisionSignal(available=True, patient_present=True, fall_suspected=True),
    )
    assert assessment.level is RiskLevel.CRITICAL
    assert assessment.composite_score >= config.composite_critical_threshold
    assert any("fall" in text.lower() for text in assessment.overrides)


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"spo2": 84.0}, "spo2"),
        ({"heart_rate": 39.0}, "bradycardia"),
        ({"heart_rate": 141.0}, "tachycardia"),
        ({"bp_systolic": 79.0}, "hypotension"),
        ({"temperature": 33.9}, "hypothermia"),
    ],
)
def test_extreme_single_values_force_at_least_high(
    config: Settings, kwargs: dict, label: str
) -> None:
    """Any one of these is a crash call regardless of what the weighted average thinks."""
    assessment = assess(config, vitals=make_vitals(**kwargs))
    assert assessment.level.rank >= RiskLevel.HIGH.rank
    assert assessment.composite_score >= config.composite_high_threshold
    assert assessment.overrides


def test_a_high_news2_lifts_the_composite_into_its_band(config: Settings) -> None:
    vitals = make_vitals(resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    news2 = calculate_news2(vitals)
    assert news2.total >= config.news2_high_threshold
    assessment = assess(config, vitals=vitals, news2=news2)
    assert assessment.composite_score >= config.composite_high_threshold
    assert assessment.level.rank >= RiskLevel.HIGH.rank


def test_the_override_floor_is_graded_not_a_single_step(config: Settings) -> None:
    """Two escalated patients should not be indistinguishable.

    A flat floor pins every NEWS2-7-or-above patient to exactly 65, which throws away the
    ordering the ward list depends on. The floor scales with the total instead, so a NEWS2 of
    13 still sorts above a NEWS2 of 7.
    """
    scores = []
    for kwargs in (
        {"resp_rate": 26.0, "spo2": 90.0, "heart_rate": 135.0},
        {"resp_rate": 26.0, "spo2": 90.0, "heart_rate": 135.0, "temperature": 34.5},
        {
            "resp_rate": 30.0,
            "spo2": 85.5,
            "heart_rate": 140.0,
            "temperature": 34.5,
            "consciousness": Consciousness.VOICE,
            "gcs": 11.0,
        },
    ):
        vitals = make_vitals(**kwargs)
        result = calculate_news2(vitals)
        scores.append((result.total, assess(config, vitals=vitals, news2=result).composite_score))

    totals = [total for total, _ in scores]
    composites = [composite for _, composite in scores]
    assert totals == sorted(totals)
    assert composites == sorted(composites)
    assert len(set(composites)) == len(composites)


def test_the_graded_floor_stops_short_of_the_next_band(config: Settings) -> None:
    """A NEWS2-driven HIGH floor must not silently manufacture a CRITICAL."""
    vitals = make_vitals(resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    assessment = assess(config, vitals=vitals)
    assert assessment.composite_score < config.composite_critical_threshold


def test_a_red_score_escalates_the_band_without_inventing_points(config: Settings) -> None:
    """The band moves to MEDIUM; the number stays honest.

    A red score says "look at this patient", not "this patient scores 40". Escalating the
    level without fabricating a floor keeps the composite interpretable.
    """
    vitals = make_vitals(resp_rate=7.0)
    news2 = calculate_news2(vitals)
    assert news2.has_red_score is True
    assert news2.total < config.news2_medium_threshold

    assessment = assess(config, vitals=vitals, news2=news2)
    assert assessment.level is RiskLevel.MEDIUM
    assert assessment.composite_score < config.composite_medium_threshold
    assert any("red score" in text.lower() for text in assessment.overrides)


def test_overrides_only_ever_escalate(config: Settings) -> None:
    """The score is already CRITICAL from the weighted channels; a lesser floor must not cut it."""
    vitals = make_vitals(
        resp_rate=32.0,
        spo2=80.0,
        bp_systolic=70.0,
        heart_rate=160.0,
        temperature=33.0,
        consciousness=Consciousness.UNRESPONSIVE,
        gcs=5.0,
        on_supplemental_oxygen=True,
    )
    with_model = assess(config, vitals=vitals, ml=prediction(0.01, 0.04, 0.95))
    assert with_model.level is RiskLevel.CRITICAL
    assert with_model.composite_score >= config.composite_critical_threshold


def test_the_binding_override_is_the_strongest_one(config: Settings) -> None:
    """Fall plus a high NEWS2: the fall's critical floor wins, and both are still reported."""
    vitals = make_vitals(resp_rate=26.0, spo2=90.0, heart_rate=135.0)
    assessment = assess(
        config,
        vitals=vitals,
        vision=VisionSignal(available=True, patient_present=True, fall_suspected=True),
    )
    assert assessment.level is RiskLevel.CRITICAL
    assert len(assessment.overrides) >= 2


# --------------------------------------------------------------------------- bounds


@pytest.mark.parametrize(
    "vitals_kwargs",
    [
        {},
        {"spo2": 70.0, "heart_rate": 200.0, "bp_systolic": 50.0, "temperature": 30.0},
        {"heart_rate": 300.0},
        {"temperature": 45.0},
    ],
)
def test_the_composite_stays_on_its_scale(config: Settings, vitals_kwargs: dict) -> None:
    assessment = assess(
        config,
        vitals=make_vitals(**vitals_kwargs),
        ml=prediction(0.0, 0.0, 1.0),
        vision=VisionSignal(available=True, patient_present=True, fall_suspected=True),
    )
    assert 0.0 <= assessment.composite_score <= 100.0


def test_a_healthy_patient_with_every_channel_available_stays_low(config: Settings) -> None:
    assessment = assess(
        config,
        ml=prediction(0.95, 0.04, 0.01),
        vision=VisionSignal(available=True, patient_present=True, posture=Posture.RECUMBENT),
    )
    assert assessment.level is RiskLevel.LOW
    assert assessment.overrides == ()
