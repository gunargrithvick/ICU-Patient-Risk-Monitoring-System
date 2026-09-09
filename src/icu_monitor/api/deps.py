"""Shared, lazily-built application state for the API.

One process holds one :class:`~icu_monitor.monitoring.engine.MonitoringEngine` and at most
one :class:`~icu_monitor.storage.repository.Repository`. Both are created on first use and
guarded by a lock, because ASGI serves requests from a thread pool and a half-built engine
is worse than a slow one.

Ticking is **pull-based**: :meth:`AppState.snapshot` advances the ward only when the last
snapshot is older than ``tick_seconds``. A background thread was the obvious alternative
and was rejected - it keeps generating synthetic patients (and database rows) in an idle
container, it complicates shutdown, and it makes tests depend on wall-clock timing. Pulling
means the API is exactly as live as the traffic it receives, and a test can drive it
deterministically.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from icu_monitor.config import Settings, get_settings
from icu_monitor.ml.registry import RiskModel, load_metrics, load_model_card
from icu_monitor.monitoring.engine import MonitoringEngine, WardSnapshot
from icu_monitor.storage.repository import Repository, build_repository

logger = logging.getLogger(__name__)


class AppState:
    """Owns the engine, the repository, and the tick clock for one process."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or get_settings()
        self._lock = threading.RLock()
        self._engine: MonitoringEngine | None = None
        self._repository: Repository | None = None
        self._repository_ready = False
        self._last_tick = 0.0
        self.started_at = time.time()
        self.request_count = 0
        self.tick_errors = 0

    # -- lazily-built components -------------------------------------------------------

    @property
    def config(self) -> Settings:
        return self._config

    def repository(self) -> Repository | None:
        with self._lock:
            if not self._repository_ready:
                self._repository = build_repository(self._config)
                self._repository_ready = True
            return self._repository

    def engine(self) -> MonitoringEngine:
        with self._lock:
            if self._engine is None:
                repository = self.repository()
                engine = MonitoringEngine(config=self._config, recorder=repository)
                # Resume the ward from the ledger before inventing a past. A restart should
                # pick up the trend charts, open alerts, and de-dup state it had, and only
                # fall back to synthetic warm-up when there is genuinely nothing to restore -
                # backfilling on top of real history would bury it under invented ticks.
                restored = engine.hydrate(repository) if repository is not None else False
                warmup = self._config.api_warmup_ticks
                if warmup > 0 and not restored:
                    engine.run(warmup, backfill=True)
                self._last_tick = time.monotonic()
                self._engine = engine
                logger.info(
                    "Engine ready: %d beds, model %s, source %s (%s)",
                    len(engine.patients),
                    engine.model_version or "none",
                    engine.provider.source_label,
                    "restored from storage" if restored else "fresh",
                )
            return self._engine

    def model(self) -> RiskModel | None:
        return self.engine().model

    # -- ticking -----------------------------------------------------------------------

    def snapshot(self, *, force: bool = False) -> WardSnapshot:
        """The current ward state, advancing it first if it has gone stale."""
        engine = self.engine()
        with self._lock:
            due = force or (time.monotonic() - self._last_tick) >= self._config.tick_seconds
            if due or engine.last_snapshot is None:
                try:
                    engine.tick()
                except Exception as exc:  # pragma: no cover - engine is defensive already
                    self.tick_errors += 1
                    logger.exception("Tick failed (%s).", exc)
                    if engine.last_snapshot is None:
                        raise HTTPException(
                            status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Monitoring engine could not produce a snapshot.",
                        ) from exc
                self._last_tick = time.monotonic()
            assert engine.last_snapshot is not None
            return engine.last_snapshot

    # -- readiness ---------------------------------------------------------------------

    def readiness(self) -> list[dict[str, object]]:
        """Component-by-component readiness, used by ``/ready``."""
        components: list[dict[str, object]] = []

        try:
            engine = self.engine()
            components.append(
                {
                    "name": "engine",
                    "ready": True,
                    "detail": f"{len(engine.patients)} beds · {engine.provider.source_label}",
                }
            )
            components.append(
                {
                    "name": "model",
                    "ready": engine.model is not None,
                    "detail": engine.model_version or "no trained artefact (NEWS2 still active)",
                }
            )
            components.append(
                {
                    "name": "vision",
                    "ready": engine.vision is not None,
                    "detail": engine.vision_label,
                }
            )
        except Exception as exc:
            components.append({"name": "engine", "ready": False, "detail": str(exc)})

        repository = self.repository()
        components.append(
            {
                "name": "database",
                "ready": repository is not None and repository.healthy(),
                "detail": "in-memory only" if repository is None else "connected",
            }
        )
        return components

    def metadata(self) -> dict[str, object]:
        """Model card + metrics, read from disk on demand so retraining shows up."""
        model = self.model()
        return {
            "available": model is not None,
            "version": model.version if model else None,
            "algorithm": model.metadata.candidate if model else None,
            "trained_at": model.metadata.trained_at if model else None,
            "feature_count": len(model.feature_names) if model else None,
            "classes": list(model.metadata.classes) if model else [],
            "metrics": (load_metrics(self._config) or {}),
            "card": (load_model_card(self._config) or {}),
        }

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            if self._repository is not None:
                self._repository.close()
                self._repository = None
            self._repository_ready = False


_STATE: AppState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> AppState:
    """The process-wide state singleton."""
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = AppState()
        return _STATE


def reset_state(config: Settings | None = None) -> AppState:
    """Replace the singleton - used by the test suite and by ``/api/v1/ward/reset``."""
    global _STATE
    with _STATE_LOCK:
        if _STATE is not None:
            _STATE.close()
        _STATE = AppState(config)
        return _STATE


# --------------------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------------------


def state_dependency() -> Iterator[AppState]:
    state = get_state()
    state.request_count += 1
    yield state


def require_api_key(
    state: Annotated[AppState, Depends(state_dependency)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce ``ICU_API_KEY`` when one is configured.

    With no key set the API is open, which is the right default for a local demo and the
    wrong one for anything reachable from a network. ``/health`` and ``/ready`` stay open
    either way so orchestrators can probe without a secret.
    """
    expected = state.config.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


StateDep = Annotated[AppState, Depends(state_dependency)]

__all__ = [
    "AppState",
    "StateDep",
    "get_state",
    "require_api_key",
    "reset_state",
    "state_dependency",
]
