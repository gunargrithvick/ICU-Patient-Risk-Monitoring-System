"""The HTTP surface: FastAPI application and routes.

The API exists so the monitor is more than a screen. Three groups of endpoints:

*Operational* - ``/health`` and ``/ready`` for orchestrators, ``/metrics`` in Prometheus
text format so the ward can be scraped like any other service.

*Stateless scoring* - ``POST /api/v1/risk/score`` and ``/risk/batch`` score observations a
caller supplies. These touch no ward state, which makes them the honest integration point
for another system: same NEWS2 implementation, same fusion, same model.

*Ward state* - patients, vitals history, alerts, and the simulator's control surface, all
reading the identical :class:`~icu_monitor.monitoring.engine.WardSnapshot` the dashboard
renders. There is exactly one implementation of a tick in this project.

Every ``/api/v1`` route depends on :func:`~icu_monitor.api.deps.require_api_key`, which is
a no-op until ``ICU_API_KEY`` is set. The probes stay unauthenticated on purpose.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware

from icu_monitor import __version__
from icu_monitor.api import schemas as api
from icu_monitor.api.deps import AppState, StateDep, get_state, require_api_key
from icu_monitor.config import get_settings
from icu_monitor.core.fusion import MLPrediction, fuse_risk
from icu_monitor.core.news2 import calculate_news2
from icu_monitor.core.types import ClinicalState, RiskLevel, utcnow
from icu_monitor.logging_setup import configure_logging

logger = logging.getLogger(__name__)

TAGS = [
    {"name": "operations", "description": "Liveness, readiness, and Prometheus metrics."},
    {"name": "risk", "description": "Stateless NEWS2 + model + fusion scoring."},
    {"name": "ward", "description": "Live ward state: beds, vitals, and the camera."},
    {"name": "alerts", "description": "The de-duplicated alert ledger."},
    {"name": "controls", "description": "Simulator-only controls for demonstrating events."},
    {"name": "model", "description": "Artefact version, metrics, and model card."},
]


# --------------------------------------------------------------------------------------
# Operational routes (unauthenticated by design)
# --------------------------------------------------------------------------------------

ops = APIRouter(tags=["operations"])


@ops.get("/health", response_model=api.HealthResponse, summary="Liveness probe")
def health(state: StateDep) -> api.HealthResponse:
    """Cheap and dependency-free: answers "is the process up", nothing more."""
    cfg = state.config
    return api.HealthResponse(
        app=cfg.app_name,
        version=__version__,
        environment=cfg.environment,
        at=utcnow(),
    )


@ops.get("/ready", response_model=api.ReadyResponse, summary="Readiness probe")
def ready(state: StateDep, response: Response) -> api.ReadyResponse:
    """Readiness is per-component and *tolerant*.

    A missing model or database is reported but does not make the service unready - the
    system is designed to run on NEWS2 alone. Only a dead engine is fatal, because without
    it there is nothing to serve.
    """
    components = [api.ComponentStatus(**item) for item in state.readiness()]  # type: ignore[arg-type]
    engine_ok = next((c.ready for c in components if c.name == "engine"), False)
    if not engine_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return api.ReadyResponse(ready=engine_ok, components=components, at=utcnow())


@ops.get("/metrics", response_class=Response, summary="Prometheus exposition")
def metrics(state: StateDep) -> Response:
    """Hand-rolled Prometheus text format - a scrape endpoint is not worth a dependency."""
    snapshot = state.snapshot()
    counts = snapshot.level_counts()
    alerts = state.engine().alerts.counts()
    lines = [
        "# HELP icu_ward_beds Occupied beds being monitored.",
        "# TYPE icu_ward_beds gauge",
        f"icu_ward_beds {len(snapshot.beds)}",
        "# HELP icu_risk_level_beds Beds currently at each risk level.",
        "# TYPE icu_risk_level_beds gauge",
    ]
    lines += [
        f'icu_risk_level_beds{{level="{level.value}"}} {count}' for level, count in counts.items()
    ]
    lines += [
        "# HELP icu_mean_composite_score Mean composite risk score across the ward.",
        "# TYPE icu_mean_composite_score gauge",
        f"icu_mean_composite_score {snapshot.mean_score:.3f}",
        "# HELP icu_alerts_active Conditions currently true.",
        "# TYPE icu_alerts_active gauge",
        f"icu_alerts_active {alerts.get('active', 0)}",
        "# HELP icu_alerts_open Alerts raised and not yet acknowledged.",
        "# TYPE icu_alerts_open gauge",
        f"icu_alerts_open {alerts.get('open', 0)}",
        "# HELP icu_ticks_total Ticks executed since start.",
        "# TYPE icu_ticks_total counter",
        f"icu_ticks_total {snapshot.tick}",
        "# HELP icu_tick_duration_ms Duration of the most recent tick.",
        "# TYPE icu_tick_duration_ms gauge",
        f"icu_tick_duration_ms {snapshot.duration_ms:.3f}",
        "# HELP icu_tick_errors_total Ticks that raised.",
        "# TYPE icu_tick_errors_total counter",
        f"icu_tick_errors_total {state.tick_errors}",
        "# HELP icu_model_loaded Whether a trained artefact is in use.",
        "# TYPE icu_model_loaded gauge",
        f"icu_model_loaded {1 if snapshot.model_version else 0}",
        "",
    ]
    return Response("\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")


# --------------------------------------------------------------------------------------
# Stateless scoring
# --------------------------------------------------------------------------------------

risk = APIRouter(prefix="/api/v1/risk", tags=["risk"])


def _score(state: AppState, request: api.ScoreRequest) -> api.ScoreResponse:
    """Score one posted observation with no reference to ward state.

    The ML channel needs a *history* to build window features from, and a single posted
    observation is not one. Rather than fabricate a window, the model is asked only when
    the caller opts in and a model exists, and the response says plainly whether it
    contributed. A NEWS2-only score is a complete answer, not a degraded one.
    """
    vitals = request.vitals.to_domain()
    news2 = calculate_news2(vitals, spo2_scale=request.spo2_scale)

    prediction = MLPrediction.unavailable("not requested")
    model = state.model() if request.include_model else None
    if model is not None:
        synthetic = _adhoc_patient(request)
        try:
            prediction = model.predict_for_patient(synthetic, [vitals])
        except Exception as exc:
            logger.warning("Ad-hoc inference failed (%s).", exc)
            prediction = MLPrediction.unavailable(f"inference error: {exc}")

    assessment = fuse_risk(
        patient_id=request.patient_id,
        vitals=vitals,
        news2=news2,
        ml=prediction,
        vision=None,
        config=state.config,
    )
    return api.ScoreResponse(
        patient_id=assessment.patient_id,
        level=assessment.level.value,
        composite_score=round(assessment.composite_score, 2),
        news2_total=assessment.news2_total,
        news2_response=news2.clinical_response,
        factors=[factor.as_dict() for factor in assessment.top_factors],
        overrides=list(assessment.overrides),
        model_available=assessment.model_available,
        ml_level=assessment.ml_level.value,
        ml_probabilities={k: round(v, 4) for k, v in assessment.ml_probabilities.items()},
        assessed_at=assessment.assessed_at,
    )


def _adhoc_patient(request: api.ScoreRequest):
    from icu_monitor.core.types import Patient

    return Patient(
        patient_id=request.patient_id,
        bed="ADHOC",
        display_name="Ad-hoc request",
        age=request.age if request.age is not None else 65,
        sex="U",
        admitted_at=utcnow(),
        primary_diagnosis="",
        spo2_scale=request.spo2_scale,
    )


@risk.post("/score", response_model=api.ScoreResponse, summary="Score one observation")
def score(state: StateDep, request: api.ScoreRequest) -> api.ScoreResponse:
    return _score(state, request)


@risk.post("/batch", response_model=api.BatchScoreResponse, summary="Score many observations")
def score_batch(state: StateDep, request: api.BatchScoreRequest) -> api.BatchScoreResponse:
    results = [_score(state, item) for item in request.items]
    return api.BatchScoreResponse(count=len(results), results=results)


# --------------------------------------------------------------------------------------
# Ward state
# --------------------------------------------------------------------------------------

ward = APIRouter(prefix="/api/v1", tags=["ward"])

PatientId = Annotated[str, Path(min_length=1, max_length=32, description="e.g. P001")]


def _bed_or_404(state: AppState, patient_id: str):
    bed = state.snapshot().bed(patient_id)
    if bed is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"No bed for patient {patient_id}.")
    return bed


@ward.get("/ward", summary="The whole ward after the current tick")
def ward_snapshot(state: StateDep, force: bool = Query(False, description="Tick now")) -> dict:
    return state.snapshot(force=force).as_dict()


@ward.get("/patients", summary="Every monitored bed")
def list_patients(
    state: StateDep,
    level: Annotated[str | None, Query(description="Filter by risk level")] = None,
) -> dict[str, Any]:
    snapshot = state.snapshot()
    beds = snapshot.beds
    if level:
        wanted = RiskLevel.coerce(level)
        beds = tuple(bed for bed in beds if bed.level is wanted)
    return {
        "count": len(beds),
        "tick": snapshot.tick,
        "source_label": snapshot.source_label,
        "patients": [bed.as_dict() for bed in beds],
    }


@ward.get("/patients/{patient_id}", summary="One bed in full")
def get_patient(state: StateDep, patient_id: PatientId) -> dict[str, Any]:
    return _bed_or_404(state, patient_id).as_dict()


@ward.get("/patients/{patient_id}/vitals", summary="Recent observations for one bed")
def patient_vitals(
    state: StateDep,
    patient_id: PatientId,
    limit: Annotated[int, Query(ge=1, le=1000)] = 120,
) -> dict[str, Any]:
    _bed_or_404(state, patient_id)  # 404 before returning an empty series
    engine = state.engine()
    history = engine.history(patient_id)[-limit:]
    scores = engine.score_history(patient_id)[-limit:]
    return {
        "patient_id": patient_id,
        "count": len(history),
        "vitals": [v.as_dict() for v in history],
        "scores": [{"at": at.isoformat(), "composite_score": round(s, 2)} for at, s in scores],
    }


@ward.get("/vision", summary="The camera channel's current signal")
def vision(state: StateDep) -> dict[str, Any]:
    """Where the camera is aimed, and the signal from the bed it was aimed at last tick.

    Those are two different facts and the response says so. Re-aiming takes effect at once
    but the signal is a tick old, so reporting only the snapshot's bed would make
    ``POST /vision/focus/{id}`` look like it had silently failed until the ward advanced.
    """
    snapshot = state.snapshot()
    return {
        "focus_bed": state.engine().focus_bed,
        "observed_bed": snapshot.focus_bed,
        "label": snapshot.vision_label,
        "signal": snapshot.vision.as_dict(),
    }


@ward.post(
    "/vision/focus/{patient_id}", response_model=api.MessageResponse, summary="Aim the camera"
)
def set_focus(state: StateDep, patient_id: PatientId) -> api.MessageResponse:
    engine = state.engine()
    engine.focus_bed = patient_id
    if engine.focus_bed != patient_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"No bed for patient {patient_id}.")
    return api.MessageResponse(ok=True, message=f"Camera assigned to {patient_id}.")


# --------------------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------------------

alerts_router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@alerts_router.get("", summary="The alert ledger")
def list_alerts(
    state: StateDep,
    patient_id: Annotated[str | None, Query(max_length=32)] = None,
    open_only: Annotated[bool, Query(description="Only unacknowledged alerts")] = False,
    active_only: Annotated[bool, Query(description="Only conditions still true")] = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """``active`` and ``open`` are different questions - see the alerts module docstring.

    Active is the wall display: conditions true right now. Open is the audit trail:
    everything raised and not yet acknowledged, including conditions that have resolved.
    """
    state.snapshot()  # make sure the ledger reflects a current tick
    manager = state.engine().alerts
    if active_only:
        records = manager.active
    elif open_only:
        records = manager.open_alerts
    else:
        records = manager.history
    if patient_id:
        records = tuple(a for a in records if a.patient_id == patient_id)
    return {
        "counts": manager.counts(),
        "count": len(records[:limit]),
        "alerts": [a.as_dict() for a in records[:limit]],
    }


@alerts_router.post(
    "/acknowledge-all", response_model=api.AcknowledgeResponse, summary="Acknowledge in bulk"
)
def acknowledge_all(
    state: StateDep,
    request: api.AcknowledgeRequest | None = None,
    patient_id: Annotated[str | None, Query(max_length=32)] = None,
) -> api.AcknowledgeResponse:
    by = (request or api.AcknowledgeRequest()).by
    count = state.engine().alerts.acknowledge_all(patient_id=patient_id, by=by)
    repository = state.repository()
    if repository is not None:
        repository.acknowledge_all(patient_id=patient_id, by=by)
    return api.AcknowledgeResponse(acknowledged=count, by=by)


@alerts_router.post(
    "/{alert_id}/acknowledge", response_model=api.AcknowledgeResponse, summary="Acknowledge one"
)
def acknowledge(
    state: StateDep,
    alert_id: Annotated[int, Path(ge=1)],
    request: api.AcknowledgeRequest | None = None,
) -> api.AcknowledgeResponse:
    """Acknowledging records that a human saw the alert. It does not clear the condition.

    Idempotent, and the response says which of the two things happened: ``acknowledged``
    is 1 only when this call was the one that closed the alert, and ``by`` is whoever owns
    the signature afterwards - the first acknowledger, not necessarily this caller.
    """
    by = (request or api.AcknowledgeRequest()).by
    manager = state.engine().alerts
    was_open = any(a.alert_id == alert_id and a.is_open for a in manager.history)
    alert = manager.acknowledge(alert_id, by=by)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"No alert with id {alert_id}.")
    repository = state.repository()
    if repository is not None:
        repository.acknowledge(alert_id, by=by)
    return api.AcknowledgeResponse(
        acknowledged=1 if was_open else 0,
        alert_id=alert_id,
        by=alert.acknowledged_by or by,
    )


# --------------------------------------------------------------------------------------
# Controls (simulator only - the replay provider answers "unsupported")
# --------------------------------------------------------------------------------------

controls = APIRouter(prefix="/api/v1/controls", tags=["controls"])


@controls.get("/events", summary="Injectable clinical events")
def available_events(state: StateDep) -> dict[str, Any]:
    events = state.engine().available_events()
    return {
        "supported": bool(events),
        "events": events,
        "note": ""
        if events
        else "The active vitals source replays recorded data and is read-only.",
    }


@controls.post(
    "/patients/{patient_id}/state", response_model=api.MessageResponse, summary="Set trajectory"
)
def set_state(
    state: StateDep, patient_id: PatientId, request: api.StateRequest
) -> api.MessageResponse:
    """409 means the source is read-only; a bed that does not exist is a 404 first.

    The provider answers ``False`` to both, so the two are separated here - otherwise
    a typo in a bed id comes back as "this ward cannot be controlled", which sends the
    caller looking for the wrong problem.
    """
    _bed_or_404(state, patient_id)
    if not state.engine().set_state(patient_id, request.state):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="The active vitals source does not support changing patient state.",
        )
    return api.MessageResponse(
        ok=True, message=f"{patient_id} set to {ClinicalState.coerce(request.state).label}."
    )


@controls.post(
    "/patients/{patient_id}/oxygen", response_model=api.MessageResponse, summary="Set oxygen"
)
def set_oxygen(
    state: StateDep, patient_id: PatientId, request: api.OxygenRequest
) -> api.MessageResponse:
    _bed_or_404(state, patient_id)
    if not state.engine().set_oxygen(patient_id, on=request.on, scale=request.scale):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="The active vitals source does not support changing oxygen delivery.",
        )
    scale_note = f", SpO₂ target scale {request.scale}" if request.scale else ""
    return api.MessageResponse(
        ok=True,
        message=f"{patient_id} supplemental oxygen {'on' if request.on else 'off'}{scale_note}.",
    )


@controls.post(
    "/patients/{patient_id}/events/{slug}",
    response_model=api.MessageResponse,
    summary="Inject a clinical event",
)
def inject_event(state: StateDep, patient_id: PatientId, slug: str) -> api.MessageResponse:
    _bed_or_404(state, patient_id)
    label = state.engine().inject_event(patient_id, slug)
    if label is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Unknown event '{slug}' or unsupported by the active vitals source.",
        )
    return api.MessageResponse(ok=True, message=f"{label} started on {patient_id}.")


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

model_router = APIRouter(prefix="/api/v1/model", tags=["model"])


@model_router.get("", response_model=api.ModelInfoResponse, summary="Artefact, metrics, card")
def model_info(state: StateDep) -> api.ModelInfoResponse:
    return api.ModelInfoResponse(**state.metadata())  # type: ignore[arg-type]


@model_router.post("/reload", response_model=api.ModelInfoResponse, summary="Reload from disk")
def reload_model(state: StateDep) -> api.ModelInfoResponse:
    """Pick up a freshly trained artefact without restarting the service."""
    state.engine().reload_model()
    return api.ModelInfoResponse(**state.metadata())  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the engine eagerly so the first request is not the slow one."""
    configure_logging()
    state = get_state()
    if not state.config.api_key:
        logger.warning(
            "ICU_API_KEY is not set: /api/v1 routes are unauthenticated. Acceptable for a "
            "local demo; set a key before exposing this service on a network."
        )
    try:
        state.engine()
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("Engine failed to start (%s); /ready will report it.", exc)
    yield
    state.close()


def create_app() -> FastAPI:
    """Assemble the application. A factory, so tests can build isolated instances."""
    cfg = get_settings()
    app = FastAPI(
        title=f"{cfg.app_name} API",
        description=__doc__,
        version=__version__,
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(ops)
    guarded = [risk, ward, alerts_router, controls, model_router]
    for router in guarded:
        app.include_router(router, dependencies=[Depends(require_api_key)])

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "app": cfg.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "ward": "/api/v1/ward",
        }

    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
