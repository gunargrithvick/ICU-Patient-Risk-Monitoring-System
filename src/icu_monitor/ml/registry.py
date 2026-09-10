"""Model registry: persistence, versioning, and the inference wrapper.

The registry is the only place that touches the model file, and it exists to enforce
three properties that a bare ``joblib.load`` cannot:

1. **The artefact carries its own contract.** Feature names, class order, training
   date, dataset fingerprint, and metrics are saved alongside the estimator. Loading
   checks the saved feature list and class order against the running code; a mismatch
   means the code has moved on since training, and the artefact is refused rather than
   silently producing garbage.
2. **A missing model is a first-class state, not a crash.** The original project called
   ``joblib.load`` at import time with a relative path, so the app died on any machine
   whose working directory differed. Here the loader returns ``None`` and the UI shows
   "model unavailable" while NEWS2 and the vision channel keep working.
3. **Reloads are cheap but not stale.** The cache is keyed on the file's modification
   time, so retraining is picked up without restarting the process.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from icu_monitor import __version__
from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.fusion import MLPrediction
from icu_monitor.core.types import ML_RISK_CLASSES, Patient, RiskLevel, Vitals, utcnow
from icu_monitor.ml.features import FEATURE_NAMES, features_for_patient, features_to_frame

logger = logging.getLogger(__name__)

#: Bundle format version - bump when the saved structure changes incompatibly.
BUNDLE_FORMAT = 2


@dataclass(slots=True)
class ModelMetadata:
    """The artefact's self-description."""

    version: str
    candidate: str
    trained_at: str
    package_version: str = __version__
    bundle_format: int = BUNDLE_FORMAT
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    classes: list[str] = field(default_factory=lambda: list(ML_RISK_CLASSES))
    window_hours: int = 8
    dataset: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    candidates_considered: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def trained_on(self) -> str:
        try:
            moment = datetime.fromisoformat(self.trained_at)
            if moment.tzinfo is not None:
                moment = moment.astimezone(timezone.utc)
            return moment.strftime("%d %b %Y %H:%M UTC")
        except ValueError:  # pragma: no cover - defensive
            return self.trained_at


class RiskModel:
    """A fitted estimator plus its metadata, exposing the inference the app needs."""

    def __init__(self, estimator: Any, metadata: ModelMetadata) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self._feature_names = list(metadata.feature_names) or list(FEATURE_NAMES)
        self._classes = list(metadata.classes) or list(ML_RISK_CLASSES)

    # -- identity ----------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def schema_matches_code(self) -> tuple[bool, str]:
        """Whether the artefact was trained on the feature set this code builds."""
        if self._feature_names == list(FEATURE_NAMES):
            return True, "Feature schema matches the running code."
        expected, saved = set(FEATURE_NAMES), set(self._feature_names)
        return False, (
            f"Feature schema drift: {len(expected - saved)} new and "
            f"{len(saved - expected)} removed columns since training. Retrain with "
            "`python -m icu_monitor train`."
        )

    def contract_matches_code(self) -> tuple[bool, str]:
        """Validate both feature names and class order before inference."""
        schema_ok, schema_message = self.schema_matches_code()
        if not schema_ok:
            return False, schema_message
        expected = list(ML_RISK_CLASSES)
        if self._classes != expected:
            return False, (
                f"Model class order drift: saved {self._classes!r}, expected {expected!r}. "
                "Retrain with `python -m icu_monitor train`."
            )
        estimator_classes = getattr(self.estimator, "classes_", None)
        if estimator_classes is not None:
            try:
                numeric = [int(value) for value in estimator_classes]
            except (TypeError, ValueError):
                return False, "Model estimator exposes an unreadable classes_ contract."
            if numeric != list(range(len(expected))):
                return False, f"Model estimator class order is invalid: {numeric!r}."
        return True, "Feature schema and class order match the running code."

    # -- inference ---------------------------------------------------------------------

    def _align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reindex an incoming frame onto the *trained* column order."""
        aligned = frame.copy()
        for name in self._feature_names:
            if name not in aligned.columns:
                aligned[name] = float("nan")
        return aligned.loc[:, self._feature_names].astype(float)

    def predict_frame(self, frame: pd.DataFrame) -> list[MLPrediction]:
        """Predict for every row of a design matrix."""
        if frame.empty:
            return []
        aligned = self._align(frame)
        probabilities = np.asarray(self.estimator.predict_proba(aligned), dtype=float)
        if probabilities.ndim != 2 or not 1 <= probabilities.shape[1] <= len(self._classes):
            raise ValueError(
                f"Model returned {probabilities.shape} probabilities for "
                f"one to {len(self._classes)} classes."
            )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Model returned non-finite or negative probabilities.")
        predictions: list[MLPrediction] = []
        for row in probabilities:
            distribution = {
                name: float(row[index])
                for index, name in enumerate(self._classes)
                if index < len(row)
            }
            best = max(distribution, key=distribution.__getitem__)
            predictions.append(
                MLPrediction(
                    level=RiskLevel.coerce(best),
                    probabilities=distribution,
                    confidence=float(max(distribution.values())),
                    available=True,
                    model_version=self.version,
                )
            )
        return predictions

    def predict_one(self, features: dict[str, float] | pd.DataFrame) -> MLPrediction:
        """Predict for a single feature row."""
        frame = features if isinstance(features, pd.DataFrame) else features_to_frame([features])
        results = self.predict_frame(frame)
        return results[0] if results else MLPrediction.unavailable("empty feature row")

    def predict_for_patient(
        self,
        patient: Patient,
        history: list[Vitals],
        *,
        window_hours: float | None = None,
        weight_kg: float | None = None,
        height_cm: float | None = None,
        icu_type: int = 3,
    ) -> MLPrediction:
        """Predict from a live rolling buffer of bedside observations."""
        if not history:
            return MLPrediction.unavailable("no observations yet")
        frame = features_for_patient(
            patient,
            history,
            window_hours=window_hours if window_hours is not None else self.metadata.window_hours,
            weight_kg=weight_kg,
            height_cm=height_cm,
            icu_type=icu_type,
        )
        return self.predict_one(frame)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def save_model(
    estimator: Any,
    metadata: ModelMetadata,
    *,
    config: Settings | None = None,
) -> Path:
    """Write the estimator bundle and the human-readable sidecar files."""
    cfg = config or default_settings
    cfg.ensure_directories()
    _atomic_joblib_dump(
        {"format": BUNDLE_FORMAT, "estimator": estimator, "metadata": metadata.as_dict()},
        cfg.model_path,
    )
    _atomic_text_write(cfg.metrics_path, json.dumps(metadata.metrics, indent=2, default=str))
    _atomic_text_write(
        cfg.model_card_path,
        json.dumps(build_model_card(metadata, config=cfg), indent=2, default=str),
    )
    logger.info("Saved model %s to %s", metadata.version, cfg.model_path)
    return cfg.model_path


_CACHE: dict[str, tuple[float, RiskModel]] = {}


def load_model(*, config: Settings | None = None, refresh: bool = False) -> RiskModel | None:
    """Load the trained model, or ``None`` if there isn't one.

    Never raises for the ordinary "not trained yet" case - the app is expected to run
    with NEWS2 and vision only.
    """
    cfg = config or default_settings
    path = cfg.model_path
    if not path.exists():
        logger.info("No model artefact at %s; running without ML.", path)
        return None

    key = str(path)
    stamp = path.stat().st_mtime
    if not refresh:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        logger.warning("Could not load model at %s (%s); running without ML.", path, exc)
        return None

    if not isinstance(bundle, dict) or "estimator" not in bundle:
        logger.warning(
            "Model at %s is not in the expected bundle format (v%s); retrain to upgrade.",
            path,
            BUNDLE_FORMAT,
        )
        return None

    raw_metadata = dict(bundle.get("metadata") or {})
    known = set(ModelMetadata.__dataclass_fields__)
    metadata = ModelMetadata(
        **{
            **{
                "version": "unversioned",
                "candidate": "unknown",
                "trained_at": utcnow().isoformat(),
            },
            **{key_: value for key_, value in raw_metadata.items() if key_ in known},
        }
    )

    model = RiskModel(bundle["estimator"], metadata)
    matches, message = model.contract_matches_code()
    if not matches:
        logger.warning("Refusing model at %s: %s", path, message)
        return None
    _CACHE[key] = (stamp, model)
    return model


def _atomic_joblib_dump(bundle: dict[str, Any], destination: Path) -> None:
    """Write a complete model before replacing the live artefact."""
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(destination: Path, content: str) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def clear_cache() -> None:
    """Drop the in-process model cache (used by tests and after retraining)."""
    _CACHE.clear()


def _redact_dataset_paths(card: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-specific paths before a model card is persisted or exposed by API."""
    safe = deepcopy(card)
    training = safe.get("training_data")
    if not isinstance(training, dict):
        return safe
    etl = training.get("etl")
    if not isinstance(etl, dict):
        return safe
    paths = etl.get("paths")
    if not isinstance(paths, dict):
        return safe
    for name, value in paths.items():
        if isinstance(value, str):
            # DatasetSummary keeps absolute paths for local CLI diagnostics. A model card
            # may be returned over HTTP, so retain only the portable artifact name here.
            paths[name] = value.replace("\\", "/").rsplit("/", 1)[-1]
    return safe


# --------------------------------------------------------------------------------------
# Model card
# --------------------------------------------------------------------------------------


def build_model_card(metadata: ModelMetadata, *, config: Settings | None = None) -> dict[str, Any]:
    """Assemble a model card in the spirit of Mitchell et al. (2019).

    Written to ``artifacts/model_card.json`` and rendered on the Model Insights page, so
    the limitations travel with the model instead of living only in a README.
    """
    cfg = config or default_settings
    card = {
        "model_details": {
            "name": f"{cfg.app_name} bedside acuity classifier",
            "version": metadata.version,
            "package_version": metadata.package_version,
            "estimator": metadata.candidate,
            "trained_at": metadata.trained_at,
            "classes": metadata.classes,
            "n_features": len(metadata.feature_names),
            "window_hours": metadata.window_hours,
        },
        "intended_use": {
            "primary": (
                "Educational demonstration of an explainable ICU deterioration monitor. "
                "One of three channels fused with NEWS2 and a vision signal."
            ),
            "out_of_scope": [
                "Any clinical decision, triage, or patient management.",
                "Populations outside adult ICU.",
                "Deployment as a medical device - it is not certified as one.",
            ],
        },
        "training_data": metadata.dataset,
        "labels": metadata.labels,
        "metrics": metadata.metrics,
        "candidates_considered": metadata.candidates_considered,
        "ethical_considerations": {
            "population": (
                "PhysioNet/CinC Challenge 2012 set-a: adult ICU admissions from a single "
                "US hospital system, 2001-2008. Performance on other populations, other "
                "eras of practice, and paediatric patients is unknown."
            ),
            "fairness": (
                "Age and sex are model inputs. Race and socioeconomic data are absent "
                "from the source, so disparate performance across those groups cannot be "
                "measured here - absence of evidence is not evidence of fairness."
            ),
            "label_leakage": (
                "The MEDIUM class is partly defined by admission SOFA, which is itself "
                "derived from physiology feeding the features. Reported MEDIUM "
                "performance is therefore optimistic relative to a prospective task."
            ),
            "temporal_scope": (
                "A window's label describes the whole stay, so the model estimates "
                "'this patient is on a bad trajectory', not 'this patient is "
                "deteriorating in the next hour'."
            ),
        },
        "caveats_and_recommendations": [
            "Windows from one patient are correlated; all metrics use patient-disjoint "
            "splits (StratifiedGroupKFold on record_id). Any evaluation that ignores "
            "grouping will look far better and be wrong.",
            "Missing channels are left as NaN rather than imputed, so the model can "
            "learn from missingness patterns that may be institution-specific.",
            "NEWS2 is a published, validated score and is computed independently of "
            "this model; when the two disagree, NEWS2 is the defensible one.",
            metadata.notes
            or "Retrain with `python -m icu_monitor train` after any change to the feature module.",
        ],
    }
    return _redact_dataset_paths(card)


def load_model_card(config: Settings | None = None) -> dict[str, Any] | None:
    """Read the saved model card, if training has produced one."""
    cfg = config or default_settings
    if not cfg.model_card_path.exists():
        return None
    try:
        loaded = json.loads(cfg.model_card_path.read_text(encoding="utf-8"))
        return _redact_dataset_paths(loaded) if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        logger.warning("Could not read model card: %s", exc)
        return None


def load_metrics(config: Settings | None = None) -> dict[str, Any] | None:
    """Read the saved metrics file, if training has produced one."""
    cfg = config or default_settings
    if not cfg.metrics_path.exists():
        return None
    try:
        return json.loads(cfg.metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        logger.warning("Could not read metrics: %s", exc)
        return None


__all__ = [
    "BUNDLE_FORMAT",
    "ModelMetadata",
    "RiskModel",
    "build_model_card",
    "clear_cache",
    "load_metrics",
    "load_model",
    "load_model_card",
    "save_model",
]
