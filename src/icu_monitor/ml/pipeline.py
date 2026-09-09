"""Model construction and leakage-free data splitting.

Two decisions in this file do most of the work of making the reported numbers real.

**Grouped splitting.** Each ICU stay contributes ~11 overlapping windows that share a
label and highly correlated physiology. A plain ``train_test_split`` puts some of a
patient's windows in train and the rest in test, so the model is scored on patients it
has already seen and the metrics come out inflated. Every split here groups by
``record_id`` via :class:`~sklearn.model_selection.StratifiedGroupKFold`, so a patient
is wholly in one side or the other.

**Integer-encoded classes in a fixed order.** scikit-learn sorts string labels
alphabetically, which would order the classes ``HIGH, LOW, MEDIUM`` and silently
misalign every ``predict_proba`` column against the UI's low→high gauge. Labels are
therefore encoded to ``0, 1, 2`` following
:data:`~icu_monitor.core.types.ML_RISK_CLASSES`, and decoded on the way out.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.ml.features import FEATURE_NAMES

#: Column carrying the patient grouping key.
GROUP_COLUMN = "record_id"

#: Column carrying the target.
TARGET_COLUMN = "acuity"

#: Label -> integer, in the canonical low→high order.
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(ML_RISK_CLASSES)}

#: Integer -> label.
INDEX_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_TO_INDEX.items()}


@dataclass(frozen=True, slots=True)
class Dataset:
    """A design matrix with the metadata needed for grouped evaluation."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_patients(self) -> int:
        return len(np.unique(self.groups))

    def class_counts(self) -> dict[str, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        counted = dict(zip(values.tolist(), counts.tolist(), strict=True))
        return {name: int(counted.get(name, 0)) for name in ML_RISK_CLASSES}

    def subset(self, index: np.ndarray) -> Dataset:
        return Dataset(
            X=self.X.iloc[index].reset_index(drop=True),
            y=self.y[index],
            groups=self.groups[index],
            labels=self.labels[index],
        )


def prepare_dataset(frame: pd.DataFrame) -> Dataset:
    """Validate the ETL output and split it into ``X``, ``y``, and groups."""
    for column in (GROUP_COLUMN, TARGET_COLUMN):
        if column not in frame.columns:
            raise ValueError(
                f"Training table is missing the required column '{column}'. "
                "Re-run the ETL (`python -m icu_monitor etl`)."
            )

    missing_features = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing_features:
        raise ValueError(
            f"Training table is missing {len(missing_features)} feature columns "
            f"(first few: {missing_features[:5]}). The ETL and the feature module have "
            "diverged - re-run the ETL."
        )

    usable = frame[frame[TARGET_COLUMN].isin(CLASS_TO_INDEX)].reset_index(drop=True)
    dropped = len(frame) - len(usable)
    if dropped:
        raise ValueError(
            f"{dropped} rows carry a label outside {list(ML_RISK_CLASSES)}. "
            "The label scheme and the dataset disagree."
        )
    if usable.empty:
        raise ValueError("Training table is empty.")

    labels = usable[TARGET_COLUMN].astype(str).to_numpy()
    return Dataset(
        X=usable.loc[:, list(FEATURE_NAMES)].astype(float),
        y=np.array([CLASS_TO_INDEX[label] for label in labels], dtype=int),
        groups=usable[GROUP_COLUMN].to_numpy(),
        labels=labels,
    )


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


def holdout_split(
    dataset: Dataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 20260905,
) -> tuple[Dataset, Dataset]:
    """Split off a patient-disjoint test set, keeping class proportions similar."""
    n_splits = max(2, min(10, round(1.0 / max(0.05, test_fraction))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(dataset.X, dataset.y, dataset.groups))
    train, test = dataset.subset(train_index), dataset.subset(test_index)

    overlap = set(train.groups.tolist()) & set(test.groups.tolist())
    if overlap:  # pragma: no cover - guards against a future splitter change
        raise AssertionError(f"{len(overlap)} patients leaked across the split.")
    return train, test


def cv_folds(
    dataset: Dataset,
    *,
    n_splits: int = 5,
    seed: int = 20260905,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield patient-disjoint, class-stratified cross-validation folds."""
    smallest_class = min(dataset.class_counts().values())
    effective = max(2, min(n_splits, smallest_class, dataset.n_patients))
    splitter = StratifiedGroupKFold(n_splits=effective, shuffle=True, random_state=seed)
    yield from splitter.split(dataset.X, dataset.y, dataset.groups)


# --------------------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------------------


def build_hist_gradient_boosting(seed: int = 20260905) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees with native missing-value support.

    ICU data is missing not-at-random - an arterial line that was never inserted is
    itself information - so the preferred estimator is one that routes ``NaN`` down its
    own branch instead of having it imputed to a fake "normal" value.
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=seed,
    )


def build_random_forest(seed: int = 20260905) -> Pipeline:
    """Calibrated random forest baseline.

    Forests cannot consume ``NaN``, so this candidate must impute - which is exactly
    why it is here. If it wins, the missingness pattern was not carrying much signal;
    if it loses, that is evidence for the boosted model's native handling.
    """
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("forest", CalibratedClassifierCV(forest, method="isotonic", cv=3)),
        ]
    )


#: Candidate name -> builder. Training evaluates every entry and keeps the best.
CANDIDATES: dict[str, Any] = {
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "random_forest": build_random_forest,
}


def build_candidate(name: str, *, config: Settings | None = None) -> Any:
    """Instantiate a candidate by name."""
    cfg = config or default_settings
    builder = CANDIDATES.get(name)
    if builder is None:
        raise KeyError(f"Unknown candidate '{name}'. Choose from {sorted(CANDIDATES)}.")
    return builder(cfg.simulation_seed)


def decode(indices: np.ndarray) -> np.ndarray:
    """Integer predictions back to ``LOW``/``MEDIUM``/``HIGH``."""
    return np.array([INDEX_TO_CLASS[int(value)] for value in indices], dtype=object)


__all__ = [
    "CANDIDATES",
    "CLASS_TO_INDEX",
    "GROUP_COLUMN",
    "INDEX_TO_CLASS",
    "TARGET_COLUMN",
    "Dataset",
    "build_candidate",
    "build_hist_gradient_boosting",
    "build_random_forest",
    "cv_folds",
    "decode",
    "holdout_split",
    "prepare_dataset",
]
