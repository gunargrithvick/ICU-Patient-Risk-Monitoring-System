"""Evaluation: the numbers that go in the model card.

Accuracy alone is misleading on this dataset - roughly 14 % of windows are ``HIGH``, so
a model that never predicts ``HIGH`` still scores in the eighties. Everything reported
here is therefore either class-balanced (macro-F1, balanced accuracy) or itemised per
class (precision/recall/support, confusion matrix), plus calibration, because a risk
gauge that reads "72 %" should mean something close to 72 % of such patients.

Discrimination is reported as one-vs-rest ROC-AUC and average precision. On an
imbalanced problem average precision is the more honest of the two - ROC-AUC can look
strong while the top of the ranked list is mostly false alarms.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.ml.features import FEATURE_NAMES

#: Number of equal-width bins used for the reliability curve.
CALIBRATION_BINS = 10

#: How many features to report in the importance table.
TOP_FEATURES = 15


@dataclass(slots=True)
class EvaluationReport:
    """Everything the model card and the Model Insights page need."""

    n_samples: int
    n_patients: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    log_loss: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    roc_auc: dict[str, float]
    average_precision: dict[str, float]
    calibration: dict[str, Any]
    importances: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "cohen_kappa": self.cohen_kappa,
            "log_loss": self.log_loss,
            "per_class": self.per_class,
            "confusion": self.confusion,
            "confusion_labels": list(ML_RISK_CLASSES),
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "calibration": self.calibration,
            "importances": self.importances,
        }

    @property
    def headline(self) -> str:
        return (
            f"macro-F1 {self.macro_f1:.3f} · balanced accuracy "
            f"{self.balanced_accuracy:.3f} · {self.n_samples:,} windows "
            f"from {self.n_patients:,} patients"
        )


def _safe_round(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits)) if np.isfinite(value) else float("nan")


def _one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((len(y_true), n_classes), dtype=float)
    matrix[np.arange(len(y_true)), y_true] = 1.0
    return matrix


def _discrimination(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """One-vs-rest ROC-AUC and average precision, per class."""
    n_classes = probabilities.shape[1]
    truth = _one_hot(y_true, n_classes)
    roc: dict[str, float] = {}
    ap: dict[str, float] = {}
    for index, name in enumerate(ML_RISK_CLASSES[:n_classes]):
        column = truth[:, index]
        # A class with no positives (or no negatives) has no defined AUC.
        if column.min() == column.max():
            roc[name] = float("nan")
            ap[name] = float("nan")
            continue
        roc[name] = _safe_round(roc_auc_score(column, probabilities[:, index]))
        ap[name] = _safe_round(average_precision_score(column, probabilities[:, index]))

    finite_roc = [value for value in roc.values() if np.isfinite(value)]
    finite_ap = [value for value in ap.values() if np.isfinite(value)]
    roc["macro"] = _safe_round(float(np.mean(finite_roc))) if finite_roc else float("nan")
    ap["macro"] = _safe_round(float(np.mean(finite_ap))) if finite_ap else float("nan")
    return roc, ap


def _calibration(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Reliability of the ``HIGH`` probability - the number the gauge leans on."""
    high_index = ML_RISK_CLASSES.index("HIGH")
    if probabilities.shape[1] <= high_index:
        return {"bins": [], "brier": float("nan"), "note": "HIGH class absent"}

    predicted = probabilities[:, high_index]
    observed = (y_true == high_index).astype(float)
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bins: list[dict[str, float]] = []
    for lower, upper in itertools.pairwise(edges):
        mask = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
        if not mask.any():
            continue
        bins.append(
            {
                "bin_lower": _safe_round(lower, 3),
                "bin_upper": _safe_round(upper, 3),
                "mean_predicted": _safe_round(float(predicted[mask].mean())),
                "observed_frequency": _safe_round(float(observed[mask].mean())),
                "count": int(mask.sum()),
            }
        )
    return {
        "target_class": "HIGH",
        "bins": bins,
        "brier": _safe_round(brier_score_loss(observed, predicted)),
        # Expected calibration error: mean |predicted - observed| weighted by bin size.
        "expected_calibration_error": _safe_round(
            float(
                sum(
                    row["count"] * abs(row["mean_predicted"] - row["observed_frequency"])
                    for row in bins
                )
                / max(1, len(predicted))
            )
        ),
    }


def evaluate_model(
    model: Any,
    X: Any,
    y_true: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    with_importances: bool = True,
    seed: int = 20260905,
) -> EvaluationReport:
    """Score a fitted model on held-out data."""
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    predictions = np.asarray(model.predict(X), dtype=int)
    n_classes = probabilities.shape[1]
    class_indices = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=class_indices, zero_division=0
    )
    per_class = {
        ML_RISK_CLASSES[index]: {
            "precision": _safe_round(precision[position]),
            "recall": _safe_round(recall[position]),
            "f1": _safe_round(f1[position]),
            "support": int(support[position]),
        }
        for position, index in enumerate(class_indices)
    }

    roc, ap = _discrimination(y_true, probabilities)

    importances: list[dict[str, float]] = []
    if with_importances:
        importances = _importances(model, X, y_true, seed=seed)

    return EvaluationReport(
        n_samples=len(y_true),
        n_patients=len(np.unique(groups)) if groups is not None else 0,
        accuracy=_safe_round(accuracy_score(y_true, predictions)),
        balanced_accuracy=_safe_round(balanced_accuracy_score(y_true, predictions)),
        macro_f1=_safe_round(f1_score(y_true, predictions, average="macro", zero_division=0)),
        weighted_f1=_safe_round(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        cohen_kappa=_safe_round(cohen_kappa_score(y_true, predictions)),
        log_loss=_safe_round(log_loss(y_true, probabilities, labels=class_indices)),
        per_class=per_class,
        confusion=confusion_matrix(y_true, predictions, labels=class_indices).tolist(),
        roc_auc=roc,
        average_precision=ap,
        calibration=_calibration(y_true, probabilities),
        importances=importances,
    )


def _importances(
    model: Any, X: Any, y_true: np.ndarray, *, seed: int, repeats: int = 4
) -> list[dict[str, float]]:
    """Permutation importance on held-out data.

    Permutation importance is used rather than a tree's internal split counts because
    it answers the question a reviewer actually has - *how much worse does the model
    get without this input* - and it is comparable across the two candidate estimators.
    """
    try:
        result = permutation_importance(
            model,
            X,
            y_true,
            scoring="f1_macro",
            n_repeats=repeats,
            random_state=seed,
            n_jobs=1,
        )
    except Exception:  # pragma: no cover - importance is a nice-to-have, never fatal
        return []

    names = list(getattr(X, "columns", FEATURE_NAMES))
    order = np.argsort(result.importances_mean)[::-1][:TOP_FEATURES]
    return [
        {
            "feature": str(names[index]),
            "importance": _safe_round(float(result.importances_mean[index])),
            "std": _safe_round(float(result.importances_std[index])),
        }
        for index in order
    ]


__all__ = [
    "CALIBRATION_BINS",
    "TOP_FEATURES",
    "EvaluationReport",
    "evaluate_model",
]
