"""Training: pick a candidate honestly, then report what it actually does.

The procedure:

1. load the window table produced by the ETL;
2. hold out a **patient-disjoint** test set and do not touch it again until step 5;
3. cross-validate every candidate in
   :data:`~icu_monitor.ml.pipeline.CANDIDATES` on the training patients only,
   scoring macro-F1 so the 14 %-prevalence ``HIGH`` class actually counts;
4. refit the winner on all training patients;
5. evaluate once on the held-out patients and write the metrics, model card, and
   artefact.

Model selection never sees the test set, so the reported score is an estimate of
performance on unseen patients rather than a number that was optimised against.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.data.labels import label_definition
from icu_monitor.data.physionet import load_dataset_summary, load_windows
from icu_monitor.ml.evaluate import EvaluationReport, evaluate_model
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.pipeline import (
    CANDIDATES,
    Dataset,
    build_candidate,
    cv_folds,
    holdout_split,
    prepare_dataset,
)
from icu_monitor.ml.registry import ModelMetadata, clear_cache, save_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateResult:
    """Cross-validated score for one candidate estimator."""

    name: str
    macro_f1_mean: float
    macro_f1_std: float
    balanced_accuracy_mean: float
    folds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cv_macro_f1_mean": round(self.macro_f1_mean, 4),
            "cv_macro_f1_std": round(self.macro_f1_std, 4),
            "cv_balanced_accuracy_mean": round(self.balanced_accuracy_mean, 4),
            "cv_folds": self.folds,
        }


@dataclass(slots=True)
class TrainingResult:
    """Outcome of a full training run."""

    winner: str
    version: str
    report: EvaluationReport
    candidates: list[CandidateResult]
    dataset: dict[str, Any]

    @property
    def summary(self) -> str:
        return f"{self.winner} · {self.report.headline}"


def cross_validate_candidate(
    name: str,
    dataset: Dataset,
    *,
    config: Settings | None = None,
    n_splits: int = 5,
    progress: Callable[[str], None] | None = None,
) -> CandidateResult:
    """Patient-grouped cross-validation for one candidate."""
    cfg = config or default_settings
    say = progress or (lambda message: logger.info("%s", message))

    macro_scores: list[float] = []
    balanced_scores: list[float] = []
    for fold, (train_index, valid_index) in enumerate(
        cv_folds(dataset, n_splits=n_splits, seed=cfg.simulation_seed), start=1
    ):
        train, valid = dataset.subset(train_index), dataset.subset(valid_index)
        estimator = clone(build_candidate(name, config=cfg))
        estimator.fit(train.X, train.y)
        predicted = estimator.predict(valid.X)
        macro = float(f1_score(valid.y, predicted, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(valid.y, predicted))
        macro_scores.append(macro)
        balanced_scores.append(balanced)
        say(f"    fold {fold}: macro-F1 {macro:.3f}, balanced accuracy {balanced:.3f}")

    return CandidateResult(
        name=name,
        macro_f1_mean=float(np.mean(macro_scores)) if macro_scores else 0.0,
        macro_f1_std=float(np.std(macro_scores)) if macro_scores else 0.0,
        balanced_accuracy_mean=float(np.mean(balanced_scores)) if balanced_scores else 0.0,
        folds=len(macro_scores),
    )


#: What the card says when nothing on disk will vouch for the table.
UNRECORDED_SOURCE = "unrecorded (no dataset summary beside the processed table)"
SUPPLIED_SOURCE = "supplied in-process by the caller (not read from disk)"


def _provenance(cfg: Settings, *, supplied_frame: bool) -> dict[str, Any]:
    """Where the training rows came from, according to whatever wrote them.

    This used to be a constant naming PhysioNet, which is the one thing a provenance field
    must never be. ``etl --synthetic`` exists precisely so the project runs without the
    archive, and a model trained that way was getting a card asserting real patient data
    beside a near-perfect held-out score - the two claims that, together, are most likely to
    be believed and most certainly wrong.

    So the ETL now writes its own :class:`~icu_monitor.data.DatasetSummary` next to the table
    and this reads it back. Three answers are possible and the differences are all load-bearing:
    the recorded source, ``SUPPLIED_SOURCE`` for a frame handed in directly (no file speaks for
    it), and ``UNRECORDED_SOURCE`` for a table built before this existed. The last is not a
    failure - it is the honest answer, and better than a guess that happens to be plausible.
    """
    if supplied_frame:
        return {"source": SUPPLIED_SOURCE, "synthetic": False}
    recorded = load_dataset_summary(cfg)
    if recorded is None:
        return {"source": UNRECORDED_SOURCE, "synthetic": False}
    return {
        "source": str(recorded.get("source") or UNRECORDED_SOURCE),
        "synthetic": bool(recorded.get("synthetic", False)),
        "etl": recorded,
    }


def train(
    *,
    config: Settings | None = None,
    frame: pd.DataFrame | None = None,
    candidates: list[str] | None = None,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    with_importances: bool = True,
    progress: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Run the full training procedure and persist the winning model."""
    cfg = config or default_settings
    say = progress or (lambda message: logger.info("%s", message))

    windows = load_windows(cfg) if frame is None else frame
    dataset = prepare_dataset(windows)
    say(
        f"Loaded {len(dataset):,} windows from {dataset.n_patients:,} patients "
        f"· {len(FEATURE_NAMES)} features · classes {dataset.class_counts()}"
    )

    # Resolved here rather than at the end, so the caveat is printed beside the description of
    # the data and above the scores. Under a macro-F1 of 1.000 - which is what training on the
    # simulator produces - a caveat appended afterwards has already been disbelieved.
    provenance = _provenance(cfg, supplied_frame=frame is not None)
    if provenance.get("synthetic"):
        say(
            "NOTE: this table is synthetic. The scores below measure recovery of the "
            "simulator's own rules, not clinical performance."
        )

    train_set, test_set = holdout_split(
        dataset, test_fraction=test_fraction, seed=cfg.simulation_seed
    )
    say(
        f"Held out {len(test_set):,} windows from {test_set.n_patients:,} patients "
        f"(patient-disjoint); training on {train_set.n_patients:,} patients"
    )

    names = candidates or list(CANDIDATES)
    results: list[CandidateResult] = []
    for name in names:
        say(f"  cross-validating {name}")
        results.append(
            cross_validate_candidate(name, train_set, config=cfg, n_splits=n_splits, progress=say)
        )

    results.sort(key=lambda result: result.macro_f1_mean, reverse=True)
    best = results[0]
    say(f"Winner: {best.name} (CV macro-F1 {best.macro_f1_mean:.3f} ± {best.macro_f1_std:.3f})")

    estimator = clone(build_candidate(best.name, config=cfg))
    estimator.fit(train_set.X, train_set.y)

    say("Evaluating on the held-out patients")
    report = evaluate_model(
        estimator,
        test_set.X,
        test_set.y,
        test_set.groups,
        with_importances=with_importances,
        seed=cfg.simulation_seed,
    )
    say(f"Held-out: {report.headline}")

    trained_at = datetime.now(timezone.utc)
    version = f"{trained_at:%Y%m%d-%H%M}-{best.name}"
    dataset_summary = {
        **provenance,
        "windows_total": len(dataset),
        "patients_total": dataset.n_patients,
        "class_counts": dataset.class_counts(),
        "train_windows": len(train_set),
        "train_patients": train_set.n_patients,
        "test_windows": len(test_set),
        "test_patients": test_set.n_patients,
        "window_hours": cfg.window_hours,
        "window_stride_hours": cfg.window_stride_hours,
        "split": "StratifiedGroupKFold on record_id (patient-disjoint)",
    }

    metadata = ModelMetadata(
        version=version,
        candidate=best.name,
        trained_at=trained_at.isoformat(),
        feature_names=list(FEATURE_NAMES),
        window_hours=cfg.window_hours,
        dataset=dataset_summary,
        labels=label_definition(cfg).describe(),
        metrics=report.as_dict(),
        candidates_considered=[result.as_dict() for result in results],
        notes=(
            "Selection used cross-validation on training patients only; the held-out "
            "patients were scored exactly once."
        ),
    )
    save_model(estimator, metadata, config=cfg)
    clear_cache()
    say(
        f"Saved {version} → {cfg.model_path.name}, {cfg.metrics_path.name}, "
        f"{cfg.model_card_path.name}"
    )

    return TrainingResult(
        winner=best.name,
        version=version,
        report=report,
        candidates=results,
        dataset=dataset_summary,
    )


__all__ = [
    "SUPPLIED_SOURCE",
    "UNRECORDED_SOURCE",
    "CandidateResult",
    "TrainingResult",
    "cross_validate_candidate",
    "train",
]
