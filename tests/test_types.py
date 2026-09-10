"""Domain types: the enumerations and value objects everything else is built from.

These are the smallest units in the system and the ones with no dependencies, so a defect
here surfaces as a confusing failure four layers up. Three themes are worth stating.

**The coercion boundary.** ``RiskLevel.coerce`` and ``ClinicalState.coerce`` are *tolerant*:
they read a stale value from an old database row and degrade to a benign default rather than
raising, because a monitor that crashes on one malformed history row is worse than one that
shows "No data" for it. ``Consciousness.parse`` is *strict*: it returns ``None``. The
difference is deliberate and it is a safety property - ``ALERT`` is worth zero NEWS2 points
and ``U`` is worth three, so guessing would understate exactly the patient this system
exists to escalate. Both halves are tested here, as is the asymmetry itself.

**Nothing divides by zero and nothing goes negative.** A monitor receives a systolic of 0
from a disconnected line and a detector emits an inverted box. Every derived property below
is checked against those inputs.

**Every enum member has its display strings.** The label and status-token tests iterate the
enum rather than listing members, so adding a level without teaching the UI how to draw it
fails here instead of raising ``KeyError`` in a Streamlit callback.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from icu_monitor.core.types import (
    ACVPU_TOKENS,
    ML_RISK_CLASSES,
    Alert,
    AlertKind,
    BedSnapshot,
    ClinicalState,
    Consciousness,
    Detection,
    Posture,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    VisionSignal,
    Vitals,
    utcnow,
)

from .conftest import EPOCH, make_patient, make_vitals

# ======================================================================================
# RiskLevel
# ======================================================================================


def test_the_ladder_is_strictly_ordered():
    """Ranks must be strictly increasing, because sorting the ward depends on it."""
    ladder = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    ranks = [level.rank for level in ladder]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_unknown_ranks_below_low():
    """ "No data" is not a mild reading - it must never sort above a scored patient."""
    assert RiskLevel.UNKNOWN.rank < RiskLevel.LOW.rank


@pytest.mark.parametrize("level", list(RiskLevel))
def test_every_level_has_display_strings(level: RiskLevel):
    """A new level without a label or status token breaks the UI, so fail here instead."""
    assert level.label
    assert level.status_token in {"muted", "good", "warning", "serious", "critical"}


def test_status_tokens_are_unique():
    """Two levels sharing a status role would draw identically on the wall display."""
    tokens = [level.status_token for level in RiskLevel]
    assert len(set(tokens)) == len(tokens)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.HIGH),
        (RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.HIGH),
        (RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.MEDIUM),
        (RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.CRITICAL),
    ],
)
def test_escalation_returns_the_more_severe_level(left, right, expected):
    assert left.escalate_to(right) is expected
    assert right.escalate_to(left) is expected


@pytest.mark.parametrize("other", [level for level in RiskLevel if level is not RiskLevel.UNKNOWN])
def test_unknown_always_loses_an_escalation(other: RiskLevel):
    """Absence of information must not out-rank information, in either argument position."""
    assert RiskLevel.UNKNOWN.escalate_to(other) is other
    assert other.escalate_to(RiskLevel.UNKNOWN) is other


def test_two_unknowns_stay_unknown():
    assert RiskLevel.UNKNOWN.escalate_to(RiskLevel.UNKNOWN) is RiskLevel.UNKNOWN


@pytest.mark.parametrize("level", list(RiskLevel))
def test_coerce_is_the_identity_on_a_real_level(level: RiskLevel):
    assert RiskLevel.coerce(level) is level


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HIGH", RiskLevel.HIGH),
        ("high", RiskLevel.HIGH),
        ("  Medium  ", RiskLevel.MEDIUM),
        ("MODERATE", RiskLevel.MEDIUM),
        ("moderate", RiskLevel.MEDIUM),
        ("SEVERE", RiskLevel.HIGH),
        ("NONE", RiskLevel.UNKNOWN),
        ("", RiskLevel.UNKNOWN),
        ("   ", RiskLevel.UNKNOWN),
    ],
)
def test_coerce_reads_the_spellings_that_arrive_from_stored_rows(raw: str, expected: RiskLevel):
    assert RiskLevel.coerce(raw) is expected


@pytest.mark.parametrize("raw", ["catastrophic", "9", None, 3, object(), [], {"level": "HIGH"}])
def test_coerce_degrades_rather_than_raising(raw: object):
    """A stored row is not a control surface: it must never be able to end the tick."""
    assert RiskLevel.coerce(raw) is RiskLevel.UNKNOWN


def test_coerce_never_invents_severity():
    """The tolerant default is the *least* severe level, so garbage cannot raise an alarm."""
    for raw in ("catastrophic", "", None, "unknown-to-us"):
        assert RiskLevel.coerce(raw).rank <= RiskLevel.UNKNOWN.rank


def test_the_ml_classes_are_the_first_three_levels_in_order():
    """The probability vector's column order is a contract with the saved estimator."""
    assert ML_RISK_CLASSES == ("LOW", "MEDIUM", "HIGH")
    assert RiskLevel.CRITICAL.value not in ML_RISK_CLASSES
    assert RiskLevel.UNKNOWN.value not in ML_RISK_CLASSES
    assert all(RiskLevel.coerce(name).value == name for name in ML_RISK_CLASSES)


def test_the_level_serialises_as_its_own_name():
    """``str`` subclassing is what lets a level go into JSON without a custom encoder."""
    assert json.dumps({"level": RiskLevel.HIGH}) == '{"level": "HIGH"}'


# ======================================================================================
# Consciousness - the strict half of the coercion boundary
# ======================================================================================


@pytest.mark.parametrize("level", list(Consciousness))
def test_parse_accepts_the_letter_the_name_and_the_label(level: Consciousness):
    """All three spellings a caller might send round-trip to the same member."""
    assert Consciousness.parse(level) is level
    assert Consciousness.parse(level.value) is level
    assert Consciousness.parse(level.name) is level
    assert Consciousness.parse(level.label) is level


@pytest.mark.parametrize("level", list(Consciousness))
def test_parse_ignores_case_and_surrounding_space(level: Consciousness):
    """Values arrive from query strings and hand-written JSON, not from a picker."""
    assert Consciousness.parse(f"  {level.name.lower()}  ") is level
    assert Consciousness.parse(level.name.title()) is level
    assert Consciousness.parse(level.label.upper()) is level


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("awake", Consciousness.ALERT),
        ("confused", Consciousness.CONFUSION),
        ("verbal", Consciousness.VOICE),
        ("painful", Consciousness.PAIN),
        ("unconscious", Consciousness.UNRESPONSIVE),
    ],
)
def test_parse_accepts_the_reasonable_english_spellings(alias: str, expected: Consciousness):
    assert Consciousness.parse(alias) is expected


@pytest.mark.parametrize("raw", ["drowsy", "obtunded", "GCS 9", "", "   ", None, 0, 7, object()])
def test_parse_refuses_rather_than_guessing(raw: object):
    """The whole point of this method: an unrecognised token is ``None``, never ``ALERT``.

    ``ALERT`` scores zero NEWS2 points and ``UNRESPONSIVE`` scores three. Defaulting an
    unparseable word to ``ALERT`` would erase three points from precisely the patient who
    needs escalating, and it would do so silently. The caller decides instead; the API
    turns this ``None`` into a 422 that names the accepted set.
    """
    assert Consciousness.parse(raw) is None


def test_the_two_coercion_styles_disagree_on_purpose():
    """A stored row degrades; a control surface refuses. Same input, different answers."""
    assert RiskLevel.coerce("nonsense") is RiskLevel.UNKNOWN
    assert Consciousness.parse("nonsense") is None


def test_the_rejection_message_can_name_every_accepted_letter_and_name():
    """``ACVPU_TOKENS`` is what a 422 body quotes, so it must stay in sync with the enum."""
    assert len(ACVPU_TOKENS) == 2 * len(Consciousness)
    for level in Consciousness:
        assert level.value in ACVPU_TOKENS
        assert level.name.lower() in ACVPU_TOKENS
    assert all(Consciousness.parse(token) is not None for token in ACVPU_TOKENS)


def test_the_tokens_are_in_acvpu_order():
    """A, C, V, P, U - the mnemonic order a nurse reads, not alphabetical."""
    assert ACVPU_TOKENS[::2] == ("A", "C", "V", "P", "U")


@pytest.mark.parametrize(
    ("gcs", "expected"),
    [
        (15, Consciousness.ALERT),
        (15.0, Consciousness.ALERT),
        (18, Consciousness.ALERT),
        (14, Consciousness.CONFUSION),
        (13, Consciousness.CONFUSION),
        (12.9, Consciousness.VOICE),
        (9, Consciousness.VOICE),
        (8.9, Consciousness.PAIN),
        (6, Consciousness.PAIN),
        (5.9, Consciousness.UNRESPONSIVE),
        (3, Consciousness.UNRESPONSIVE),
    ],
)
def test_gcs_maps_onto_the_acvpu_rungs_at_the_documented_boundaries(gcs, expected):
    """ICU datasets record GCS; NEWS2 wants ACVPU. Only a fully alert patient scores zero."""
    assert Consciousness.from_gcs(gcs) is expected


@pytest.mark.parametrize("absent", [None, float("nan")])
def test_a_missing_gcs_reads_as_alert(absent):
    """The retrospective default. It is optimistic, which is why the ETL is the only caller."""
    assert Consciousness.from_gcs(absent) is Consciousness.ALERT


def test_the_gcs_mapping_is_monotonic():
    """A lower GCS can never map to a *less* impaired rung."""
    rungs = [Consciousness.from_gcs(score) for score in range(15, 2, -1)]
    order = list(Consciousness)
    assert [order.index(rung) for rung in rungs] == sorted(order.index(rung) for rung in rungs)


# ======================================================================================
# ClinicalState, Posture, AlertKind
# ======================================================================================


@pytest.mark.parametrize("state", list(ClinicalState))
def test_a_state_round_trips_through_its_own_value(state: ClinicalState):
    assert ClinicalState.coerce(state) is state
    assert ClinicalState.coerce(state.value) is state
    assert ClinicalState.coerce(state.value.upper()) is state
    assert ClinicalState.coerce(f" {state.value} ") is state
    assert state.label == state.value.capitalize()


@pytest.mark.parametrize("raw", ["comatose", "", None, 4, object()])
def test_an_unknown_state_degrades_to_stable(raw: object):
    """Read from stored rows and query strings, so an old schema's value must not raise."""
    assert ClinicalState.coerce(raw) is ClinicalState.STABLE


@pytest.mark.parametrize("posture", list(Posture))
def test_every_posture_has_a_label(posture: Posture):
    assert posture.label == posture.value.capitalize()


@pytest.mark.parametrize("kind", list(AlertKind))
def test_every_alert_kind_has_a_human_label(kind: AlertKind):
    """The label is what a nurse reads on the wall, so a new kind must supply one."""
    assert kind.label
    assert kind.label != kind.value


def test_alert_kind_labels_are_unique():
    """Two kinds reading identically on screen would make the ledger ambiguous."""
    labels = [kind.label for kind in AlertKind]
    assert len(set(labels)) == len(labels)


def test_the_fourteen_rules_have_fourteen_kinds():
    """Documented in the README; asserted here so the two cannot drift apart."""
    assert len(AlertKind) == 14


def test_utcnow_is_timezone_aware_utc():
    """Everything compares timestamps, so a naive one anywhere would raise on subtraction."""
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
    assert (utcnow() - now).total_seconds() >= 0


# ======================================================================================
# Vitals
# ======================================================================================


def test_map_uses_the_standard_estimate():
    """(SBP + 2·DBP) / 3 - the formula the derived feature and the UI both quote."""
    vitals = make_vitals(bp_systolic=120.0, bp_diastolic=60.0)
    assert vitals.map_mmhg == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("systolic", "diastolic"),
    [(None, 75.0), (120.0, None), (None, None)],
)
def test_map_is_none_when_either_pressure_is_missing(systolic, diastolic):
    """Not imputed. Half a blood pressure is not a blood pressure."""
    assert make_vitals(bp_systolic=systolic, bp_diastolic=diastolic).map_mmhg is None


def test_shock_index_is_heart_rate_over_systolic():
    vitals = make_vitals(heart_rate=110.0, bp_systolic=100.0)
    assert vitals.shock_index == pytest.approx(1.1)


def test_a_disconnected_arterial_line_does_not_divide_by_zero():
    """A systolic of 0 is a sensor fault, not a pressure. It reads as unmeasured."""
    assert make_vitals(heart_rate=80.0, bp_systolic=0.0).shock_index is None
    assert make_vitals(heart_rate=None, bp_systolic=100.0).shock_index is None


def test_measured_channels_counts_the_five_physiology_channels():
    """Diastolic, GCS, and ACVPU are tracked separately from the vital-sign channels."""
    assert make_vitals().measured_channels == 5
    assert make_vitals(spo2=None, temperature=None).measured_channels == 3
    assert Vitals().measured_channels == 0


def test_a_zero_reading_still_counts_as_measured():
    """``0`` is a value a monitor can report; only ``None`` means "not measured"."""
    assert Vitals(heart_rate=0.0).measured_channels == 1


def test_the_serialised_observation_carries_the_derived_values_too():
    """The API and the UI both read this dict; a missing key is a KeyError in a view."""
    payload = make_vitals(bp_systolic=120.0, bp_diastolic=60.0, heart_rate=90.0).as_dict()
    assert payload["map"] == pytest.approx(80.0)
    assert payload["shock_index"] == pytest.approx(0.75)
    assert payload["consciousness"] == "A"
    assert payload["recorded_at"] == EPOCH.isoformat()
    assert json.dumps(payload)


def test_an_empty_observation_serialises_without_raising():
    """Every channel can drop at once - a disconnected monitor - and the tick continues."""
    payload = Vitals().as_dict()
    assert payload["map"] is None
    assert payload["shock_index"] is None
    assert payload["consciousness"] == "A"


# ======================================================================================
# Detection
# ======================================================================================


def test_the_box_geometry_is_what_the_analyser_reasons_over():
    box = Detection(x1=100, y1=200, x2=180, y2=440, confidence=0.87)
    assert (box.width, box.height) == (80, 240)
    assert box.area == 80 * 240
    assert box.aspect_ratio == pytest.approx(80 / 240)
    assert box.centroid == (140, 320)


def test_an_inverted_box_clamps_to_zero_instead_of_going_negative():
    """A detector can emit reversed corners. A negative area would invert posture logic."""
    box = Detection(x1=90, y1=90, x2=10, y2=10, confidence=0.5)
    assert box.width == 0
    assert box.height == 0
    assert box.area == 0


def test_a_zero_height_box_has_no_aspect_ratio_rather_than_an_exception():
    """Posture is inferred from the aspect ratio, so this path runs on every tick."""
    assert Detection(x1=0, y1=5, x2=40, y2=5, confidence=0.5).aspect_ratio == 0.0


def test_a_wide_box_is_recumbent_shaped_and_a_tall_one_is_not():
    """The ratio is width/height, and posture depends on which side of 1 it lands."""
    lying = Detection(x1=0, y1=0, x2=240, y2=80, confidence=0.9)
    standing = Detection(x1=0, y1=0, x2=80, y2=240, confidence=0.9)
    assert lying.aspect_ratio > 1.0 > standing.aspect_ratio


def test_the_serialised_box_is_rounded_for_display():
    payload = Detection(x1=1, y1=2, x2=3, y2=6, confidence=0.876543).as_dict()
    assert payload["confidence"] == 0.8765
    assert payload["aspect_ratio"] == 0.5
    assert payload["label"] == "patient"


# ======================================================================================
# VisionSignal
# ======================================================================================


def test_the_default_signal_is_unavailable_and_says_nothing_about_the_patient():
    """The important half: absent vision must not read as "patient present and fine"."""
    signal = VisionSignal()
    assert signal.available is False
    assert signal.patient_present is False
    assert signal.fall_suspected is False
    assert signal.person_count == 0
    assert signal.best_confidence == 0.0
    assert signal.posture is Posture.UNKNOWN


def test_best_confidence_is_the_maximum_over_the_boxes():
    signal = VisionSignal(
        available=True,
        detections=(
            Detection(x1=0, y1=0, x2=10, y2=20, confidence=0.4),
            Detection(x1=0, y1=0, x2=10, y2=20, confidence=0.91),
            Detection(x1=0, y1=0, x2=10, y2=20, confidence=0.7),
        ),
    )
    assert signal.person_count == 3
    assert signal.best_confidence == pytest.approx(0.91)


def test_the_serialised_signal_names_its_backend_and_source():
    """The UI states which detector is running, so these two fields are load-bearing."""
    payload = VisionSignal(backend="heuristic", source="synthetic", latency_ms=3.14159).as_dict()
    assert payload["backend"] == "heuristic"
    assert payload["source"] == "synthetic"
    assert payload["latency_ms"] == 3.14
    assert payload["posture"] == "unknown"
    assert json.dumps(payload)


# ======================================================================================
# Alert
# ======================================================================================


def make_alert(**overrides: object) -> Alert:
    values: dict[str, object] = {
        "patient_id": "P001",
        "kind": AlertKind.HYPOXIA,
        "severity": RiskLevel.HIGH,
        "message": "SpO₂ 89%",
        "created_at": EPOCH,
    }
    values.update(overrides)
    return Alert(**values)  # type: ignore[arg-type]


def test_an_unacknowledged_alert_is_open():
    assert make_alert().is_open is True


def test_the_dedupe_key_is_the_patient_and_the_kind():
    """Cooldown de-duplication keys on this, which is why it excludes the message text."""
    assert make_alert().dedupe_key == "P001:hypoxia"
    assert make_alert(message="SpO₂ 84%").dedupe_key == make_alert().dedupe_key
    assert make_alert(kind=AlertKind.BED_EXIT).dedupe_key != make_alert().dedupe_key


def test_acknowledgement_closes_it_and_records_who():
    alert = make_alert()
    alert.acknowledge(by="nurse-7")
    assert alert.is_open is False
    assert alert.acknowledged_by == "nurse-7"
    assert alert.acknowledged_at is not None


def test_acknowledgement_is_idempotent_and_the_first_signature_wins():
    """This is an audit fact. A second sign-off must not overwrite who actually saw it."""
    alert = make_alert()
    alert.acknowledge(by="nurse-7")
    first_at, first_by = alert.acknowledged_at, alert.acknowledged_by

    alert.acknowledge(by="someone-else")

    assert alert.acknowledged_at == first_at
    assert alert.acknowledged_by == first_by


def test_refresh_updates_the_live_text_without_moving_the_audit_timestamp():
    """``created_at`` is when the condition began. It is the fact the ledger exists to hold."""
    alert = make_alert()
    later = EPOCH + timedelta(minutes=4)

    alert.refresh("SpO₂ 84%", later)

    assert alert.message == "SpO₂ 84%"
    assert alert.last_seen_at == later
    assert alert.created_at == EPOCH


def test_refresh_does_not_reopen_an_acknowledged_alert():
    """A condition that is still true is not a new alert - that is the whole cooldown idea."""
    alert = make_alert()
    alert.acknowledge()
    alert.refresh("SpO₂ 84%", EPOCH + timedelta(minutes=1))
    assert alert.is_open is False


def test_duration_measures_from_first_detection_to_last_sighting():
    alert = make_alert()
    alert.refresh("still low", EPOCH + timedelta(minutes=7, seconds=30))
    assert alert.duration_seconds() == pytest.approx(450.0)


def test_the_last_sighting_wins_over_a_supplied_now():
    """Once the condition has stopped being seen, the duration must stop growing."""
    alert = make_alert()
    alert.refresh("still low", EPOCH + timedelta(minutes=2))
    assert alert.duration_seconds(now=EPOCH + timedelta(hours=5)) == pytest.approx(120.0)


def test_duration_never_goes_negative_on_a_clock_that_stepped_backwards():
    """Container clock skew is real, and a negative age would render as garbage."""
    alert = make_alert()
    assert alert.duration_seconds(now=EPOCH - timedelta(minutes=3)) == 0.0


def test_the_serialised_alert_carries_both_the_kind_and_its_label():
    payload = make_alert().as_dict()
    assert payload["kind"] == "hypoxia"
    assert payload["kind_label"] == "Hypoxaemia"
    assert payload["severity"] == "HIGH"
    assert payload["is_open"] is True
    assert payload["acknowledged_at"] is None
    assert payload["created_at"] == EPOCH.isoformat()
    assert json.dumps(payload)


# ======================================================================================
# Patient
# ======================================================================================


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Amara Osei", "AO"),
        ("john smith", "JS"),
        ("Cher", "C"),
        ("Maria de los Santos", "MD"),
    ],
)
def test_initials_take_the_first_two_words(name: str, expected: str):
    """Drawn on the bed card, so it must survive one-word and four-word names alike."""
    assert make_patient(display_name=name).initials == expected


def test_length_of_stay_is_measured_in_hours():
    patient = make_patient(admitted_at=EPOCH - timedelta(hours=30, minutes=30))
    assert patient.los_hours(now=EPOCH) == pytest.approx(30.5)


def test_a_naive_admission_timestamp_is_read_as_utc():
    """SQLite hands back naive datetimes. Subtracting one from an aware `now` would raise."""
    patient = make_patient(admitted_at=datetime(2026, 9, 5, 6, 0))
    assert patient.los_hours(now=EPOCH) == pytest.approx(6.0)


def test_length_of_stay_never_goes_negative_for_a_future_admission():
    assert make_patient(admitted_at=EPOCH + timedelta(hours=2)).los_hours(now=EPOCH) == 0.0


def test_the_serialised_patient_is_json_safe():
    payload = make_patient().as_dict()
    assert payload["state"] == "stable"
    assert payload["admitted_at"].endswith("+00:00")
    assert payload["los_hours"] >= 0
    assert json.dumps(payload)


# ======================================================================================
# RiskFactor, RiskAssessment, BedSnapshot
# ======================================================================================


def make_assessment(**overrides: object) -> RiskAssessment:
    values: dict[str, object] = {
        "patient_id": "P001",
        "level": RiskLevel.HIGH,
        "composite_score": 66.4,
        "ml_level": RiskLevel.MEDIUM,
        "ml_confidence": 0.61,
        "ml_probabilities": {"LOW": 0.14, "MEDIUM": 0.61, "HIGH": 0.25},
        "news2": None,
        "vision": None,
        "factors": (
            RiskFactor(source="news2", description="NEWS2 6", points=24.0, severity="serious"),
            RiskFactor(source="ml", description="Model MEDIUM", points=27.5),
            RiskFactor(source="vision", description="Motion", points=9.9),
        ),
        "assessed_at": EPOCH,
    }
    values.update(overrides)
    return RiskAssessment(**values)  # type: ignore[arg-type]


def test_the_factors_are_ordered_largest_contribution_first():
    """The patient view reads top-down, so the biggest reason has to be the first row."""
    points = [factor.points for factor in make_assessment().top_factors]
    assert points == sorted(points, reverse=True)


def test_a_factor_defaults_to_an_informational_severity():
    """Only a factor that claims a status role gets a coloured chip in the UI."""
    assert RiskFactor(source="ml", description="x", points=1.0).severity == "info"


def test_news2_total_is_none_when_the_score_could_not_be_computed():
    """A bare posted observation may have no NEWS2 at all; the property must not raise."""
    assert make_assessment().news2_total is None


def test_the_summary_names_the_level_the_score_and_any_override():
    summary = make_assessment(overrides=("Single red parameter",)).summary
    assert "High risk" in summary
    assert "66/100" in summary
    assert "override: Single red parameter" in summary


def test_the_summary_omits_the_override_clause_when_nothing_bound():
    assert "override" not in make_assessment().summary


def test_the_serialised_assessment_is_ordered_rounded_and_json_safe():
    """This dict is the API response body and the dashboard's input. Both need every key."""
    payload = make_assessment(overrides=("NEWS2 floor",)).as_dict()

    assert payload["level"] == "HIGH"
    assert payload["ml_level"] == "MEDIUM"
    assert payload["composite_score"] == 66.4
    assert payload["news2_total"] is None
    assert payload["news2_response"] is None
    assert payload["vision"] is None
    assert payload["overrides"] == ["NEWS2 floor"]
    assert payload["model_available"] is True
    assert [f["source"] for f in payload["factors"]] == ["ml", "news2", "vision"]
    assert json.dumps(payload)


def test_probabilities_are_rounded_but_still_sum_to_one():
    """Four decimal places, because the UI prints percentages and nothing needs more."""
    payload = make_assessment().as_dict()
    assert sum(payload["ml_probabilities"].values()) == pytest.approx(1.0)
    assert all(
        len(str(value).split(".")[-1]) <= 4 for value in payload["ml_probabilities"].values()
    )


def test_an_assessment_without_a_model_says_so_rather_than_scoring_zero():
    """The distinction the whole fusion layer rests on: unavailable is not "low risk"."""
    payload = make_assessment(
        ml_level=RiskLevel.UNKNOWN,
        ml_confidence=None,
        ml_probabilities={},
        model_available=False,
    ).as_dict()
    assert payload["model_available"] is False
    assert payload["ml_level"] == "UNKNOWN"
    assert payload["ml_confidence"] is None
    assert payload["ml_probabilities"] == {}


def test_a_bed_snapshot_exposes_the_level_it_will_be_sorted_by():
    snapshot = BedSnapshot(
        patient=make_patient(),
        vitals=make_vitals(),
        assessment=make_assessment(),
    )
    assert snapshot.level is RiskLevel.HIGH
    assert snapshot.new_alerts == ()

    payload = snapshot.as_dict()
    assert set(payload) == {"patient", "vitals", "assessment", "new_alerts"}
    assert payload["new_alerts"] == []
    assert json.dumps(payload)
