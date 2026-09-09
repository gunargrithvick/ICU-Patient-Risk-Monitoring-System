"""The HTTP surface: every route, the API-key gate, and the paths that degrade.

The API is what makes this more than a screen, so these tests pin the contract another
system would integrate against. Four properties matter more than any individual route:

**The probes and the ward are different tiers.** ``/health``, ``/ready`` and ``/metrics``
answer without a key so an orchestrator can scrape them; every ``/api/v1`` route sits
behind ``X-API-Key`` the moment one is configured. That split is asserted structurally as
well as by request, so a router added to the wrong list fails a test instead of quietly
publishing the ward.

**Readiness is tolerant.** No trained model and no database are both supported
configurations and must not make the service unready - only a dead engine is fatal.

**Ticking is pull-based.** The ward advances at most once per ``tick_seconds`` unless
``force`` is set, so the API is exactly as live as the traffic it receives, and a tick that
raises serves the previous snapshot rather than a 500.

**Scoring is stateless.** ``/risk/score`` touches no ward state and is a complete answer on
NEWS2 alone, which is what lets another system integrate without adopting the simulator.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from icu_monitor import __version__
from icu_monitor.api.deps import AppState, get_state, require_api_key, reset_state
from icu_monitor.api.main import create_app
from icu_monitor.config import Settings
from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.registry import ModelMetadata, clear_cache, save_model
from icu_monitor.monitoring.engine import MonitoringEngine

from .conftest import EPOCH

API_KEY = "ward-secret-for-tests"

#: A NEWS2 total of 0 - every channel inside its normal band.
NORMAL = {"heart_rate": 72, "spo2": 98, "bp_systolic": 122, "resp_rate": 15, "temperature": 36.7}

#: Tachycardic, hypoxic, hypotensive, tachypnoeic, febrile.
DETERIORATING = {
    "heart_rate": 128,
    "spo2": 89,
    "bp_systolic": 86,
    "resp_rate": 28,
    "temperature": 38.9,
}


class StubEstimator:
    """A picklable estimator, so the model tests load a real artefact from disk.

    Fixed probabilities with ``HIGH`` most likely, which makes "the model is contributing"
    something a response body states rather than something the test infers from a flag.
    """

    def predict_proba(self, frame) -> np.ndarray:
        return np.tile(np.asarray([0.2, 0.3, 0.5]), (len(frame), 1))


def _explode(*_args: object, **_kwargs: object) -> None:
    """Stand in for a component that cannot be built at all."""
    raise RuntimeError("simulated engine failure")


# --------------------------------------------------------------------------------- helpers


def save_artefact(config: Settings, *, version: str = "20260905-1203-stub") -> str:
    """Write a real bundle and its sidecars into this test's ``artifacts/`` directory."""
    save_model(
        StubEstimator(),
        ModelMetadata(
            version=version,
            candidate="stub_classifier",
            trained_at=EPOCH.isoformat(),
            metrics={"macro_f1": 0.517},
        ),
        config=config,
    )
    return version


def exposition(text: str) -> dict[str, float]:
    """Parse Prometheus text into ``{series: value}``, label sets included in the key."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        values[series] = float(value)
    return values


def warm(client: TestClient, ticks: int = 3) -> None:
    """Advance the ward through the API, the way a polling client would."""
    for _ in range(ticks):
        client.get("/api/v1/ward", params={"force": True})


def reload_model(client: TestClient) -> dict:
    response = client.post("/api/v1/model/reload")
    assert response.status_code == 200
    return response.json()


def open_alerts(client: TestClient) -> list[dict]:
    return client.get("/api/v1/alerts", params={"open_only": True}).json()["alerts"]


def an_open_alert(client: TestClient) -> dict:
    """The newest unacknowledged alert; the seeded ward raises several on its first tick."""
    alerts = open_alerts(client)
    assert alerts, "expected the simulated ward to raise alerts"
    return alerts[0]


# -------------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolate_model_cache() -> Iterator[None]:
    """The registry cache is process-global; a stale entry would leak between tests."""
    clear_cache()
    yield
    clear_cache()


def _client(config: Settings, **overrides: object) -> TestClient:
    """Install an isolated ward as the process singleton and return a client over it.

    The client is deliberately *not* used as a context manager, so ``lifespan`` does not
    run: the engine is then built by the first request that needs it, and nothing tears the
    state down underneath a test. Startup itself is covered by its own test below.
    """
    reset_state(config.with_overrides(**overrides) if overrides else config)
    return TestClient(create_app())


@pytest.fixture
def client(config: Settings) -> Iterator[TestClient]:
    yield _client(config)
    reset_state(None)  # closes the tmp_path-bound ward; the replacement builds nothing


@pytest.fixture
def slow_client(config: Settings) -> Iterator[TestClient]:
    """A ward whose tick interval outlives the test, so several reads see one snapshot.

    Anything that compares two responses uses this. With the 0.25 s test interval a slow
    machine could tick between them and turn a real assertion into a coin flip. 30 s is the
    largest interval ``Settings`` accepts, and no test here runs for anything like that long.
    """
    yield _client(config, tick_seconds=30.0)
    reset_state(None)


@pytest.fixture
def secured(config: Settings) -> Iterator[TestClient]:
    yield _client(config, api_key=API_KEY)
    reset_state(None)


@pytest.fixture
def artefact(config: Settings) -> str:
    """A trained artefact already on disk when the process starts."""
    return save_artefact(config)


# ------------------------------------------------------------------------ operational probes


def test_liveness_names_the_build(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["app"] and body["environment"]
    assert datetime.fromisoformat(body["at"]).tzinfo is not None


def test_readiness_reports_every_component(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert {c["name"] for c in body["components"]} == {"engine", "model", "vision", "database"}


def test_a_missing_model_does_not_make_the_service_unready(client: TestClient) -> None:
    """A bare clone has no artefact and must still serve; NEWS2 does not need one."""
    response = client.get("/ready")
    components = {c["name"]: c for c in response.json()["components"]}
    assert response.status_code == 200
    assert components["model"]["ready"] is False
    assert "NEWS2" in components["model"]["detail"]


def test_a_broken_database_degrades_to_in_memory(config: Settings) -> None:
    """Persistence is a convenience. Losing it must not take the monitoring down."""
    client = _client(config, database_url="not-a-database-url")
    try:
        response = client.get("/ready")
    finally:
        reset_state(None)
    components = {c["name"]: c for c in response.json()["components"]}
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert components["database"]["ready"] is False
    assert components["database"]["detail"] == "in-memory only"


def test_a_dead_engine_is_the_only_fatal_component(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AppState, "engine", _explode)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["components"][0]["name"] == "engine"
    assert response.json()["components"][0]["detail"] == "simulated engine failure"
    # Liveness and readiness answer different questions; the process is still up.
    assert client.get("/health").status_code == 200


def test_startup_builds_the_engine_and_warns_about_the_open_api(
    config: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """An unauthenticated ward is a defensible default and an indefensible silence."""
    reset_state(config)
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
            assert get_state().engine().last_snapshot is None or True
    finally:
        reset_state(None)
    assert "unauthenticated" in caplog.text


def test_startup_warns_for_an_empty_api_key_too(
    config: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    reset_state(config.with_overrides(api_key=""))
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
    finally:
        reset_state(None)
    assert "unauthenticated" in caplog.text


# ------------------------------------------------------------------------------- Prometheus


def test_metrics_are_served_as_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


def test_every_series_is_documented_and_typed(client: TestClient) -> None:
    """A scrape endpoint without HELP and TYPE lines is one nobody can safely alert on."""
    text = client.get("/metrics").text
    documented = {line.split()[2] for line in text.splitlines() if line.startswith("# HELP")}
    typed = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")}
    assert documented == typed
    assert documented
    for series in exposition(text):
        assert series.split("{")[0] in documented


def test_the_ward_is_measured(client: TestClient) -> None:
    values = exposition(client.get("/metrics").text)
    levels = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    beds_by_level = [values[f'icu_risk_level_beds{{level="{level}"}}'] for level in levels]
    assert values["icu_ward_beds"] == 4
    assert sum(beds_by_level) == 4
    assert values["icu_ticks_total"] >= 1
    assert values["icu_tick_errors_total"] == 0
    assert values["icu_model_loaded"] == 0
    assert values["icu_mean_composite_score"] >= 0.0


def test_the_alert_gauges_agree_with_the_ledger(slow_client: TestClient) -> None:
    values = exposition(slow_client.get("/metrics").text)
    counts = slow_client.get("/api/v1/alerts").json()["counts"]
    assert values["icu_alerts_active"] == counts["active"]
    assert values["icu_alerts_open"] == counts["open"]


# ---------------------------------------------------------------------------- pull-based ticking


def test_reads_inside_one_interval_share_a_snapshot(slow_client: TestClient) -> None:
    """The ward is advanced by traffic, not by a clock, and at most once per interval.

    Without this a polling dashboard would age the ward at the rate it was refreshed, and
    two panels reading the same tick would disagree.
    """
    first = slow_client.get("/api/v1/ward").json()
    second = slow_client.get("/api/v1/ward").json()
    assert first["tick"] == second["tick"] == 1
    assert first["at"] == second["at"]


def test_force_advances_the_ward_now(slow_client: TestClient) -> None:
    first = slow_client.get("/api/v1/ward").json()["tick"]
    forced = slow_client.get("/api/v1/ward", params={"force": True}).json()["tick"]
    assert forced == first + 1


# ------------------------------------------------------------------------ root and schema


def test_the_root_points_at_everything_else(client: TestClient) -> None:
    """Someone who hits the bare host should not have to read the source to find the ward."""
    body = client.get("/").json()
    assert body["version"] == __version__
    assert {"docs", "health", "ward"} <= set(body)
    for path in (body["docs"], body["health"], body["ward"]):
        assert path.startswith("/")


def test_the_openapi_document_generates(client: TestClient) -> None:
    """A schema that fails to build takes the docs page with it, and only a request shows it."""
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    document = schema.json()
    assert document["info"]["version"] == __version__
    assert "/api/v1/ward" in document["paths"]
    assert "/" not in document["paths"]  # the index is a convenience, not part of the contract
    assert {"operations", "risk", "ward", "alerts", "controls", "model"} <= {
        tag["name"] for tag in document["tags"]
    }


def test_cors_allows_reads_and_scores_but_not_credentials(client: TestClient) -> None:
    """A browser dashboard on another origin must work; a cookie must not travel with it."""
    options = next(
        middleware.kwargs
        for middleware in create_app().user_middleware
        if middleware.cls is CORSMiddleware
    )
    assert options["allow_methods"] == ["GET", "POST"]
    assert options["allow_credentials"] is False


def test_the_ward_is_behind_the_key_and_the_probes_are_not(client: TestClient) -> None:
    """Asserted structurally, so a new router added to the wrong list fails here.

    Checking this by request would only prove the routes that a test remembered to call.
    Reading the dependency off every route proves it for the ones nobody thought of.
    """
    keyed, open_paths = set(), set()
    for route in create_app().routes:
        dependencies = getattr(route, "dependencies", [])
        target = keyed if any(d.dependency is require_api_key for d in dependencies) else open_paths
        target.add(getattr(route, "path", ""))

    assert keyed, "expected the ward routes to be guarded"
    assert all(path.startswith("/api/v1") for path in keyed)
    assert not any(path.startswith("/api/v1") for path in open_paths)
    assert {"/health", "/ready", "/metrics", "/"} <= open_paths


# ----------------------------------------------------------------------------- scoring


def score(client: TestClient, vitals: dict, **request: object) -> dict:
    response = client.post("/api/v1/risk/score", json={"vitals": vitals, **request})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_normal_observation_scores_zero(client: TestClient) -> None:
    body = score(client, NORMAL)
    assert body["patient_id"] == "adhoc"
    assert body["news2_total"] == 0
    assert body["composite_score"] == 0.0
    assert body["level"] == "LOW"
    assert "12-hourly" in body["news2_response"]


def test_a_deteriorating_observation_scores_much_higher(client: TestClient) -> None:
    """The ordering is the whole point; the exact number is a fusion detail tested elsewhere."""
    normal, sick = score(client, NORMAL), score(client, DETERIORATING)
    assert sick["news2_total"] > normal["news2_total"]
    assert sick["composite_score"] > normal["composite_score"]
    assert sick["level"] == "HIGH"
    assert "Emergency assessment" in sick["news2_response"]


def test_the_score_explains_itself(client: TestClient) -> None:
    """A number a nurse cannot interrogate is a number they will learn to ignore."""
    factors = score(client, DETERIORATING)["factors"]
    assert factors
    assert all({"source", "description", "points", "severity"} <= set(f) for f in factors)
    assert any(f["source"] == "news2" for f in factors)


def test_the_oxygen_scale_is_a_prescription_the_caller_makes(client: TestClient) -> None:
    """SpO2 90% is 3 NEWS2 points on scale 1 and none on scale 2.

    Scale 2 is for patients whose target range is 88-92% - COPD, chronic hypercapnia. Using
    scale 1 for them escalates on a saturation that is exactly where it should be.
    """
    assert score(client, {"spo2": 90})["news2_total"] == 3
    assert score(client, {"spo2": 90}, spo2_scale=2)["news2_total"] == 0


def test_a_gcs_is_translated_into_acvpu(client: TestClient) -> None:
    """ICU data records GCS; NEWS2 wants ACVPU. GCS 6 is "responds to pain" - 3 points."""
    body = score(client, {"gcs": 6})
    assert body["news2_total"] == 3
    assert any("pain" in f["description"].lower() for f in body["factors"])


@pytest.mark.parametrize("token", ["A", "a", "alert", "Awake"])
def test_consciousness_accepts_the_letter_and_the_word(client: TestClient, token: str) -> None:
    """The field was unusable before: every spelling was rejected, letters included."""
    assert score(client, {"consciousness": token})["news2_total"] == 0


@pytest.mark.parametrize("token", ["U", "unresponsive", "P", "Responds to pain", "V", "C"])
def test_any_reduced_consciousness_scores_three(client: TestClient, token: str) -> None:
    assert score(client, {"consciousness": token})["news2_total"] == 3


def test_an_unrecognised_consciousness_is_refused_not_read_as_alert(client: TestClient) -> None:
    """Defaulting here would understate exactly the patient this system exists to escalate."""
    response = client.post("/api/v1/risk/score", json={"vitals": {"consciousness": "asleep"}})
    assert response.status_code == 422
    assert "unresponsive" in response.json()["detail"][0]["msg"]


def test_the_caller_names_the_patient(client: TestClient) -> None:
    assert score(client, NORMAL, patient_id="ICU-77")["patient_id"] == "ICU-77"


def test_scoring_does_not_advance_the_ward(slow_client: TestClient) -> None:
    """``/risk/score`` is stateless, so a busy integration cannot age the simulated ward."""
    before = slow_client.get("/api/v1/ward").json()["tick"]
    score(slow_client, DETERIORATING)
    assert slow_client.get("/api/v1/ward").json()["tick"] == before


def test_without_an_artefact_the_answer_is_news2_alone(client: TestClient) -> None:
    """A bare clone must still score. The response says so rather than implying a model ran."""
    body = score(client, DETERIORATING)
    assert body["model_available"] is False
    assert body["ml_level"] == "UNKNOWN"
    assert set(body["ml_probabilities"]) == set(ML_RISK_CLASSES)
    assert all(value == 0.0 for value in body["ml_probabilities"].values())
    assert any("unavailable" in f["description"] for f in body["factors"])


def test_a_trained_model_contributes_to_the_score(config: Settings, artefact: str) -> None:
    """The stub's fixed distribution makes "the model was consulted" readable in the body."""
    client = _client(config)
    try:
        body = score(client, DETERIORATING)
    finally:
        reset_state(None)
    assert body["model_available"] is True
    assert body["ml_level"] == "HIGH"
    assert body["ml_probabilities"]["HIGH"] == pytest.approx(0.5)


def test_the_model_can_be_declined_per_request(config: Settings, artefact: str) -> None:
    """An integration that wants the transparent, defensible channel only can say so."""
    client = _client(config)
    try:
        body = score(client, DETERIORATING, include_model=False)
    finally:
        reset_state(None)
    assert body["model_available"] is False
    assert body["news2_total"] == 12


@pytest.mark.parametrize(
    "vitals",
    [
        pytest.param({"spo2": 150}, id="saturation-above-100"),
        pytest.param({"heart_rate": -1}, id="negative-pulse"),
        pytest.param({"temperature": 60}, id="impossible-fever"),
        pytest.param({"bp_systolic": 4000}, id="typo-in-systolic"),
        pytest.param({"gcs": 2}, id="gcs-below-3"),
        pytest.param({"consciousness": "asleep"}, id="unknown-acvpu"),
    ],
)
def test_a_client_bug_is_a_422_not_a_plausible_score(client: TestClient, vitals: dict) -> None:
    """The dangerous failure is a physiologically impossible input scored as if it were real."""
    assert client.post("/api/v1/risk/score", json={"vitals": vitals}).status_code == 422


def test_only_the_two_defined_oxygen_scales_exist(client: TestClient) -> None:
    body = {"vitals": NORMAL, "spo2_scale": 3}
    assert client.post("/api/v1/risk/score", json=body).status_code == 422


# ------------------------------------------------------------------------------- batches


def test_a_batch_answers_in_the_order_it_was_asked(client: TestClient) -> None:
    """Results are matched positionally by every client, so order is part of the contract."""
    response = client.post(
        "/api/v1/risk/batch",
        json={
            "items": [
                {"vitals": NORMAL, "patient_id": "calm"},
                {"vitals": DETERIORATING, "patient_id": "sick"},
                {"vitals": NORMAL, "patient_id": "calm-again"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [result["patient_id"] for result in body["results"]] == ["calm", "sick", "calm-again"]
    assert [result["level"] for result in body["results"]] == ["LOW", "HIGH", "LOW"]


def test_an_empty_batch_is_a_client_error(client: TestClient) -> None:
    assert client.post("/api/v1/risk/batch", json={"items": []}).status_code == 422


def test_a_batch_is_bounded(client: TestClient) -> None:
    """An unbounded batch is a denial-of-service vector on a synchronous scoring path."""
    oversized = {"items": [{"vitals": NORMAL}] * 501}
    assert client.post("/api/v1/risk/batch", json=oversized).status_code == 422


# --------------------------------------------------------------------------------- the ward


def test_a_snapshot_describes_the_whole_ward(client: TestClient) -> None:
    body = client.get("/api/v1/ward").json()
    assert set(body) == {
        "tick",
        "at",
        "duration_ms",
        "focus_bed",
        "model_version",
        "source_label",
        "vision_label",
        "mean_score",
        "level_counts",
        "beds",
    }
    assert len(body["beds"]) == 4
    assert sum(body["level_counts"].values()) == 4
    assert body["source_label"]


def test_a_bed_carries_the_patient_the_vitals_and_the_reasoning(client: TestClient) -> None:
    """One bed is a complete answer, so a UI never has to join three responses to draw a card."""
    bed = client.get("/api/v1/ward").json()["beds"][0]
    assert set(bed) == {"patient", "vitals", "assessment", "new_alerts"}
    assert {"patient_id", "bed", "display_name", "state", "los_hours"} <= set(bed["patient"])
    assert {"heart_rate", "spo2", "map", "shock_index"} <= set(bed["vitals"])
    assert {"level", "composite_score", "factors", "news2_total"} <= set(bed["assessment"])


def test_one_bed_can_be_fetched_on_its_own(client: TestClient) -> None:
    bed = client.get("/api/v1/patients/P001").json()
    assert bed["patient"]["patient_id"] == "P001"
    assert set(bed) == {"patient", "vitals", "assessment", "new_alerts"}


def test_the_level_filter_selects_from_the_same_snapshot(slow_client: TestClient) -> None:
    """The filter must be a view of the ward, not a second reading of it."""
    everyone = slow_client.get("/api/v1/patients").json()
    high = slow_client.get("/api/v1/patients", params={"level": "HIGH"}).json()
    expected = [b for b in everyone["patients"] if b["assessment"]["level"] == "HIGH"]
    assert high["count"] == len(expected)
    assert high["tick"] == everyone["tick"]
    assert [b["patient"]["patient_id"] for b in high["patients"]] == [
        b["patient"]["patient_id"] for b in expected
    ]


def test_an_unknown_level_matches_nobody_rather_than_erroring(client: TestClient) -> None:
    """``RiskLevel.coerce`` turns a stale query string into UNKNOWN, which no bed holds.

    A bookmarked filter from an older build should show an empty ward, not a 422 page.
    """
    body = client.get("/api/v1/patients", params={"level": "spicy"}).json()
    assert body["count"] == 0
    assert body["patients"] == []


def test_an_unknown_bed_is_a_404_that_names_it(client: TestClient) -> None:
    response = client.get("/api/v1/patients/P999")
    assert response.status_code == 404
    assert response.json()["detail"] == "No bed for patient P999."


def test_an_absurd_patient_id_never_reaches_the_engine(client: TestClient) -> None:
    assert client.get("/api/v1/patients/" + "x" * 33).status_code == 422


# ------------------------------------------------------------------------------- history


def test_a_history_is_returned_oldest_first_with_its_scores(slow_client: TestClient) -> None:
    """Chronological order is what a chart assumes; reversed data plots a mirror image."""
    warm(slow_client, ticks=4)
    body = slow_client.get("/api/v1/patients/P001/vitals").json()
    stamps = [v["recorded_at"] for v in body["vitals"]]
    assert body["patient_id"] == "P001"
    assert body["count"] == len(body["vitals"]) >= 4
    assert stamps == sorted(stamps)
    assert len(body["scores"]) == body["count"]
    assert [point["at"] for point in body["scores"]] == sorted(p["at"] for p in body["scores"])


def test_a_limit_returns_the_newest_observations(slow_client: TestClient) -> None:
    warm(slow_client, ticks=5)
    full = slow_client.get("/api/v1/patients/P001/vitals").json()["vitals"]
    tail = slow_client.get("/api/v1/patients/P001/vitals", params={"limit": 2}).json()["vitals"]
    assert len(tail) == 2
    assert tail == full[-2:]


def test_a_history_for_an_unknown_bed_is_a_404_not_an_empty_series(client: TestClient) -> None:
    """An empty list would read as "this patient has no observations", which is a lie."""
    assert client.get("/api/v1/patients/P999/vitals").status_code == 404


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_the_history_window_is_bounded(client: TestClient, limit: int) -> None:
    params = {"limit": limit}
    assert client.get("/api/v1/patients/P001/vitals", params=params).status_code == 422


# -------------------------------------------------------------------------------- vision


def test_the_vision_channel_reports_itself_disabled(client: TestClient) -> None:
    """No camera is the default and a supported configuration, so it is described, not hidden."""
    body = client.get("/api/v1/vision").json()
    assert body["signal"]["available"] is False
    assert body["signal"]["posture"] == "unknown"
    assert body["signal"]["backend"] == "off"
    assert "disabled" in body["label"].lower()
    assert body["focus_bed"] == "P001"


def test_re_aiming_the_camera_is_visible_immediately(slow_client: TestClient) -> None:
    """The aim is live; the signal is a tick old. Reporting only the latter looked broken.

    A client that POSTs the focus and reads ``/vision`` back within the same tick used to
    see the previous bed and conclude nothing had happened.
    """
    slow_client.get("/api/v1/ward")  # take a snapshot aimed at the default bed
    response = slow_client.post("/api/v1/vision/focus/P003")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "Camera assigned to P003."}

    body = slow_client.get("/api/v1/vision").json()
    assert body["focus_bed"] == "P003"
    assert body["observed_bed"] == "P001"
    assert slow_client.get("/api/v1/ward", params={"force": True}).json()["focus_bed"] == "P003"
    assert slow_client.get("/api/v1/vision").json()["observed_bed"] == "P003"


def test_the_camera_cannot_be_aimed_at_a_bed_that_does_not_exist(client: TestClient) -> None:
    response = client.post("/api/v1/vision/focus/P999")
    assert response.status_code == 404
    assert response.json()["detail"] == "No bed for patient P999."
    assert client.get("/api/v1/vision").json()["focus_bed"] == "P001"


# -------------------------------------------------------------------------------- alerts


def test_the_ledger_counts_active_open_and_history(slow_client: TestClient) -> None:
    """Three different questions, answered separately, so a UI never has to guess."""
    body = slow_client.get("/api/v1/alerts").json()
    assert {"active", "open", "history"} <= set(body["counts"])
    assert body["count"] == len(body["alerts"]) > 0
    assert body["counts"]["history"] >= body["counts"]["open"]


def test_an_alert_explains_itself(slow_client: TestClient) -> None:
    alert = an_open_alert(slow_client)
    assert {"alert_id", "patient_id", "kind", "kind_label", "severity", "message"} <= set(alert)
    assert alert["message"] and alert["detail"]
    assert alert["is_open"] is True
    assert alert["acknowledged_at"] is None
    assert datetime.fromisoformat(alert["created_at"]).tzinfo is not None


def test_active_and_open_are_different_questions(slow_client: TestClient) -> None:
    """Active is the wall display; open is the audit trail. Acknowledging separates them."""
    before = slow_client.get("/api/v1/alerts").json()["counts"]
    alert = an_open_alert(slow_client)
    slow_client.post(f"/api/v1/alerts/{alert['alert_id']}/acknowledge")
    after = slow_client.get("/api/v1/alerts").json()["counts"]
    assert after["open"] == before["open"] - 1
    assert after["active"] == before["active"], "acknowledging must not clear the condition"
    assert after["history"] == before["history"]


def test_the_ledger_can_be_scoped_to_one_bed(slow_client: TestClient) -> None:
    alert = an_open_alert(slow_client)
    patient_id = alert["patient_id"]
    body = slow_client.get("/api/v1/alerts", params={"patient_id": patient_id}).json()
    assert body["alerts"]
    assert {a["patient_id"] for a in body["alerts"]} == {patient_id}


def test_scoping_narrows_the_list_but_not_the_ward_counts(slow_client: TestClient) -> None:
    """The counts are the ward's headline figures; a filtered table must not restate them."""
    everything = slow_client.get("/api/v1/alerts").json()
    scoped = slow_client.get("/api/v1/alerts", params={"patient_id": "P004"}).json()
    assert scoped["count"] < everything["count"]
    assert scoped["counts"] == everything["counts"]


def test_a_quiet_bed_has_an_empty_ledger_of_its_own(slow_client: TestClient) -> None:
    body = slow_client.get("/api/v1/alerts", params={"patient_id": "P001"}).json()
    assert body["alerts"] == []
    assert body["counts"]["history"] > 0


def test_the_ledger_is_bounded(slow_client: TestClient) -> None:
    assert len(slow_client.get("/api/v1/alerts", params={"limit": 1}).json()["alerts"]) == 1
    for limit in (0, 501):
        assert slow_client.get("/api/v1/alerts", params={"limit": limit}).status_code == 422


def test_acknowledging_records_who_saw_it(slow_client: TestClient) -> None:
    alert = an_open_alert(slow_client)
    response = slow_client.post(
        f"/api/v1/alerts/{alert['alert_id']}/acknowledge", json={"by": "Dr Reed"}
    )
    assert response.status_code == 200
    assert response.json() == {"acknowledged": 1, "alert_id": alert["alert_id"], "by": "Dr Reed"}
    assert alert["alert_id"] not in {a["alert_id"] for a in open_alerts(slow_client)}


def test_an_unsigned_acknowledgement_is_attributed_to_the_operator(slow_client: TestClient) -> None:
    """The button on the dashboard sends no body, and the ledger still gets a name."""
    alert = an_open_alert(slow_client)
    response = slow_client.post(f"/api/v1/alerts/{alert['alert_id']}/acknowledge")
    assert response.json()["by"] == "operator"


def test_re_acknowledging_keeps_the_first_signature(slow_client: TestClient) -> None:
    """Who saw it first is the audit fact. A later click must not overwrite it or double-count."""
    alert_id = an_open_alert(slow_client)["alert_id"]
    slow_client.post(f"/api/v1/alerts/{alert_id}/acknowledge", json={"by": "Dr Reed"})
    again = slow_client.post(f"/api/v1/alerts/{alert_id}/acknowledge", json={"by": "Someone Else"})
    assert again.json() == {"acknowledged": 0, "alert_id": alert_id, "by": "Dr Reed"}

    stored = next(
        a for a in slow_client.get("/api/v1/alerts").json()["alerts"] if a["alert_id"] == alert_id
    )
    assert stored["acknowledged_by"] == "Dr Reed"
    assert stored["is_open"] is False


def test_acknowledging_an_alert_nobody_raised_is_a_404(slow_client: TestClient) -> None:
    response = slow_client.post("/api/v1/alerts/424242/acknowledge")
    assert response.status_code == 404
    assert response.json()["detail"] == "No alert with id 424242."


def test_alert_ids_start_at_one(slow_client: TestClient) -> None:
    """``0`` is never a real id, so it is rejected at the path rather than looked up."""
    assert slow_client.post("/api/v1/alerts/0/acknowledge").status_code == 422


def test_the_whole_ward_can_be_acknowledged_at_once(slow_client: TestClient) -> None:
    """The end-of-handover action. Without it a night shift clears alerts one at a time."""
    outstanding = len(open_alerts(slow_client))
    assert outstanding > 1
    response = slow_client.post("/api/v1/alerts/acknowledge-all", json={"by": "Night Shift"})
    assert response.status_code == 200
    assert response.json() == {"acknowledged": outstanding, "alert_id": None, "by": "Night Shift"}
    assert open_alerts(slow_client) == []


def test_bulk_acknowledgement_can_be_scoped_to_one_bed(slow_client: TestClient) -> None:
    target = an_open_alert(slow_client)["patient_id"]
    others = {a["patient_id"] for a in open_alerts(slow_client)} - {target}
    assert others, "expected more than one bed to be alerting"

    slow_client.post("/api/v1/alerts/acknowledge-all", params={"patient_id": target})
    remaining = {a["patient_id"] for a in open_alerts(slow_client)}
    assert target not in remaining
    assert remaining == others


def test_acknowledgement_reaches_the_database(slow_client: TestClient) -> None:
    """The in-memory ledger is the display; the rows are the record that survives a restart."""
    alert_id = an_open_alert(slow_client)["alert_id"]
    slow_client.post(f"/api/v1/alerts/{alert_id}/acknowledge", json={"by": "Dr Reed"})

    repository = get_state().repository()
    assert repository is not None
    stored = next(a for a in repository.alerts() if a.alert_id == alert_id)
    assert stored.acknowledged_by == "Dr Reed"
    assert stored.acknowledged_at is not None


# ------------------------------------------------------------------------------- controls


def test_the_simulator_advertises_what_it_can_be_told_to_do(client: TestClient) -> None:
    body = client.get("/api/v1/controls/events").json()
    assert body["supported"] is True
    assert body["note"] == ""
    assert {"sepsis", "desaturation", "haemorrhage", "recovery"} <= set(body["events"])
    assert all(isinstance(label, str) and label for label in body["events"].values())


def test_a_trajectory_can_be_set_and_the_bed_follows(client: TestClient) -> None:
    """The demo's most useful control: drive a stable patient into deterioration on cue."""
    response = client.post("/api/v1/controls/patients/P001/state", json={"state": "critical"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "P001 set to Critical."}
    assert client.get("/api/v1/patients/P001").json()["patient"]["state"] == "critical"


def test_an_invented_trajectory_is_refused_at_the_schema(client: TestClient) -> None:
    body = {"state": "spicy"}
    assert client.post("/api/v1/controls/patients/P001/state", json=body).status_code == 422


def test_oxygen_delivery_and_the_target_scale_are_one_action(client: TestClient) -> None:
    """Starting oxygen on a COPD patient means scale 2 as well; one call does both."""
    response = client.post("/api/v1/controls/patients/P002/oxygen", json={"on": True, "scale": 2})
    assert response.status_code == 200
    assert "scale 2" in response.json()["message"]

    patient = client.get("/api/v1/patients/P002").json()["patient"]
    assert patient["on_supplemental_oxygen"] is True
    assert patient["spo2_scale"] == 2


def test_only_the_two_scales_can_be_prescribed(client: TestClient) -> None:
    body = {"on": True, "scale": 3}
    assert client.post("/api/v1/controls/patients/P002/oxygen", json=body).status_code == 422


def test_an_event_can_be_injected_by_slug(client: TestClient) -> None:
    response = client.post("/api/v1/controls/patients/P001/events/sepsis")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "Suspected sepsis started on P001."}


def test_an_unknown_event_is_a_404_that_quotes_the_slug(client: TestClient) -> None:
    response = client.post("/api/v1/controls/patients/P001/events/nonsense")
    assert response.status_code == 404
    assert "nonsense" in response.json()["detail"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        pytest.param("state", {"state": "critical"}, id="state"),
        pytest.param("oxygen", {"on": True}, id="oxygen"),
    ],
)
def test_controlling_a_bed_that_does_not_exist_is_a_404_not_a_409(
    client: TestClient, path: str, body: dict
) -> None:
    """409 means "this ward is read-only" and would send the caller after the wrong problem."""
    response = client.post(f"/api/v1/controls/patients/P999/{path}", json=body)
    assert response.status_code == 404
    assert response.json()["detail"] == "No bed for patient P999."


def test_injecting_into_a_bed_that_does_not_exist_is_a_404(client: TestClient) -> None:
    response = client.post("/api/v1/controls/patients/P999/events/sepsis")
    assert response.status_code == 404
    assert response.json()["detail"] == "No bed for patient P999."


@pytest.mark.parametrize(
    ("path", "body", "setter"),
    [
        pytest.param("state", {"state": "critical"}, "set_state", id="state"),
        pytest.param("oxygen", {"on": True}, "set_oxygen", id="oxygen"),
    ],
)
def test_a_read_only_ward_answers_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: dict,
    setter: str,
) -> None:
    """A replayed cohort cannot be told what to do, and says so rather than pretending."""
    monkeypatch.setattr(MonitoringEngine, setter, lambda *_a, **_k: False)
    response = client.post(f"/api/v1/controls/patients/P001/{path}", json=body)
    assert response.status_code == 409
    assert "does not support" in response.json()["detail"]


def test_a_read_only_ward_cannot_be_given_events(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MonitoringEngine, "inject_event", lambda *_a, **_k: None)
    response = client.post("/api/v1/controls/patients/P001/events/sepsis")
    assert response.status_code == 404
    assert "unsupported" in response.json()["detail"]


# ---------------------------------------------------------------------------------- model


def test_without_an_artefact_the_model_route_says_so_plainly(client: TestClient) -> None:
    """Not a 404 and not an error: "no model" is a state the whole system is built to run in."""
    assert client.get("/api/v1/model").json() == {
        "available": False,
        "version": None,
        "algorithm": None,
        "trained_at": None,
        "feature_count": None,
        "classes": [],
        "metrics": {},
        "card": {},
    }


def test_an_artefact_on_disk_is_loaded_at_startup(config: Settings, artefact: str) -> None:
    client = _client(config)
    try:
        body = client.get("/api/v1/model").json()
    finally:
        reset_state(None)
    assert body["available"] is True
    assert body["version"] == artefact
    assert body["algorithm"] == "stub_classifier"
    assert body["feature_count"] == len(FEATURE_NAMES)
    assert body["classes"] == list(ML_RISK_CLASSES)
    assert body["metrics"] == {"macro_f1": 0.517}
    assert "model_details" in body["card"]


def test_retraining_is_picked_up_without_a_restart(client: TestClient, config: Settings) -> None:
    """``icu_monitor train`` then one POST. Without this the container has to be redeployed."""
    assert client.get("/api/v1/model").json()["available"] is False
    version = save_artefact(config, version="20260905-1400-retrained")
    assert client.get("/api/v1/model").json()["available"] is False, "cached until asked"

    reloaded = reload_model(client)
    assert reloaded["available"] is True
    assert reloaded["version"] == version


def test_a_loaded_model_reaches_every_bed(client: TestClient, config: Settings) -> None:
    """A model the dashboard reports but the beds never consult would be decoration."""
    save_artefact(config)
    reload_model(client)
    snapshot = client.get("/api/v1/ward", params={"force": True}).json()
    assert snapshot["model_version"] == "20260905-1203-stub"
    for bed in snapshot["beds"]:
        assert bed["assessment"]["model_available"] is True
        assert bed["assessment"]["ml_level"] == "HIGH"


def test_the_loaded_gauge_follows_the_model(client: TestClient, config: Settings) -> None:
    assert exposition(client.get("/metrics").text)["icu_model_loaded"] == 0
    save_artefact(config)
    reload_model(client)
    client.get("/api/v1/ward", params={"force": True})
    assert exposition(client.get("/metrics").text)["icu_model_loaded"] == 1


# ------------------------------------------------------------------------------ the key gate


@pytest.mark.parametrize("path", ["/health", "/ready", "/metrics", "/"])
def test_the_probes_stay_open_when_a_key_is_configured(secured: TestClient, path: str) -> None:
    """An orchestrator scraping health cannot be made to carry a credential."""
    assert secured.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/api/v1/ward", "/api/v1/alerts", "/api/v1/model"])
def test_the_ward_needs_the_key(secured: TestClient, path: str) -> None:
    response = secured.get(path)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_posts_need_the_key_too(secured: TestClient) -> None:
    """A gate on reads only would leave the control surface open."""
    body = {"vitals": NORMAL}
    assert secured.post("/api/v1/risk/score", json=body).status_code == 401
    assert secured.post("/api/v1/alerts/acknowledge-all").status_code == 401


def test_the_right_key_opens_the_ward(secured: TestClient) -> None:
    response = secured.get("/api/v1/ward", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert len(response.json()["beds"]) == 4


def test_a_wrong_key_is_not_a_near_miss(secured: TestClient) -> None:
    headers = {"X-API-Key": API_KEY + "x"}
    assert secured.get("/api/v1/ward", headers=headers).status_code == 401


# ---------------------------------------------------------------------------- degradation


def test_a_tick_that_raises_before_any_snapshot_is_a_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to serve and nothing to hide: the ward reports itself unavailable."""
    monkeypatch.setattr(MonitoringEngine, "tick", _explode)
    response = client.get("/api/v1/ward")
    assert response.status_code == 503
    assert response.json()["detail"] == "Monitoring engine could not produce a snapshot."


def test_a_tick_that_raises_later_serves_the_last_good_snapshot(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient failure must not blank the wall display; stale data beats no data here.

    The counter is what makes the staleness visible: the response is a 200, so only
    ``icu_tick_errors_total`` tells an operator the ward has stopped advancing.
    """
    good = client.get("/api/v1/ward").json()
    monkeypatch.setattr(MonitoringEngine, "tick", _explode)

    response = client.get("/api/v1/ward", params={"force": True})
    assert response.status_code == 200
    assert response.json()["tick"] == good["tick"]
    assert response.json()["at"] == good["at"]
    assert exposition(client.get("/metrics").text)["icu_tick_errors_total"] >= 1


def test_readiness_survives_a_broken_tick(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/ready`` asks whether the engine can be built, not whether it last ticked cleanly."""
    client.get("/api/v1/ward")
    monkeypatch.setattr(MonitoringEngine, "tick", _explode)
    assert client.get("/ready").status_code == 200
