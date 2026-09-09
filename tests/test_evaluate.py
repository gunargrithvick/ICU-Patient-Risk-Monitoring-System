"""Evaluation: the numbers that go in the model card, and why they are these numbers.

Roughly one window in seven is ``HIGH``. That single fact decides everything in this module.
A model that never predicts ``HIGH`` at all - the failure mode that matters most, because the
``HIGH`` class is the entire point of an early-warning system - still scores in the eighties on
plain accuracy. So the report is built from metrics that cannot be fooled that way, and the
first test here is a model with exactly that flaw, checked to make sure the report exposes it.

Three further properties are pinned:

**A metric that is undefined is reported as undefined.** A class with no positives in the test
set has no ROC-AUC. Reporting ``0.0`` would read as "the model is useless at this class" and
reporting ``1.0`` as "perfect"; both are claims about a measurement that was never made, so it
is ``nan``, it is excluded from the macro average, and the macro average survives.

**Calibration is checked, not assumed.** A gauge that reads 72 % should be right about 72 % of
the time. The reliability bins and the Brier score are what make that auditable.

**Importance is a nice-to-have and never fatal.** It is the slowest part of training and the
only optional one, so a failure inside it returns an empty list rather than losing the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.ml import evaluate as evaluate_module
from icu_monitor.ml.evaluate import (
    CALIBRATION_BINS,
    TOP_FEATURES,
    EvaluationReport,
    evaluate_model,
)
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.pipeline import prepare_dataset

from .conftest import make_window_frame

N_CLASSES = len(ML_RISK_CLASSES)


class Fixed:
    """A fitted model that returns probabilities it was given, row for row.

    Evaluation is arithmetic on ``(y_true, probabilities)``. Supplying the probabilities
    directly is what lets a test state the exact confusion matrix it wants to see scored,
    instead of fitting an estimator and hoping it misclassifies the right rows.
    """

    def __init__(self, probabilities: Any) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features: Any) -> np.ndarray:
        return self.probabilities

    def predict(self, features: Any) -> np.ndarray:
        return self.probabilities.argmax(axis=1)


def one_hot(
    labels: list[int], *, confidence: float = 0.9, n_classes: int = N_CLASSES
) -> np.ndarray:
    """Probabilities that are confident but not certain, so log-loss stays finite."""
    spread = (1.0 - confidence) / (n_classes - 1)
    rows = np.full((len(labels), n_classes), spread, dtype=float)
    rows[np.arange(len(labels)), labels] = confidence
    return rows


def report_for(
    truth: list[int],
    predicted: list[int] | None = None,
    *,
    confidence: float = 0.9,
    n_classes: int = N_CLASSES,
    **kwargs: Any,
) -> EvaluationReport:
    """Score a model that predicts ``predicted`` (default: perfectly) against ``truth``."""
    labels = truth if predicted is None else predicted
    model = Fixed(one_hot(labels, confidence=confidence, n_classes=n_classes))
    return evaluate_model(
        model,
        np.zeros((len(truth), 1)),
        np.asarray(truth, dtype=int),
        with_importances=False,
        **kwargs,
    )


# ------------------------------------------------------------- the imbalance this project has


def test_a_model_that_never_predicts_high_is_exposed_by_the_headline_metrics() -> None:
    """The whole reason accuracy is not the headline. Six of seven windows are LOW, and a
    model that answers LOW to everything gets most of them right while missing every
    deterioration - which is the only thing anyone is watching for."""
    truth = [0] * 12 + [1] * 2 + [2] * 2
    report = report_for(truth, [0] * len(truth))

    assert report.accuracy == pytest.approx(0.75, abs=0.01)
    assert report.balanced_accuracy == pytest.approx(1 / 3, abs=0.01)
    assert report.macro_f1 < 0.4
    assert report.per_class["HIGH"]["recall"] == 0.0


def test_the_class_that_was_missed_still_reports_its_support() -> None:
    """Recall 0.0 with support 0 is a class that was not in the test set; recall 0.0 with
    support 2 is a model that failed. The card has to be able to tell them apart."""
    truth = [0] * 8 + [2] * 2
    report = report_for(truth, [0] * 10)
    assert report.per_class["HIGH"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 2,
    }


def test_kappa_discounts_the_agreement_that_chance_would_have_produced() -> None:
    truth = [0] * 12 + [1] * 2 + [2] * 2
    assert report_for(truth, [0] * len(truth)).cohen_kappa == pytest.approx(0.0, abs=0.01)


def test_a_perfect_model_scores_one_everywhere() -> None:
    truth = [0, 0, 1, 1, 2, 2]
    report = report_for(truth)
    assert report.accuracy == 1.0
    assert report.balanced_accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.weighted_f1 == 1.0


# ------------------------------------------------------------------------------- structure


def test_the_per_class_block_is_named_and_ordered_by_the_projects_classes() -> None:
    report = report_for([0, 1, 2, 0, 1, 2])
    assert list(report.per_class) == list(ML_RISK_CLASSES)


def test_the_confusion_matrix_ships_with_the_labels_of_its_own_axes() -> None:
    """A 3x3 grid of integers is unreadable without them, and a chart would guess."""
    payload = report_for([0, 1, 2]).as_dict()
    assert payload["confusion_labels"] == list(ML_RISK_CLASSES)
    assert np.shape(payload["confusion"]) == (N_CLASSES, N_CLASSES)


def test_the_confusion_matrix_reads_rows_as_truth_and_columns_as_prediction() -> None:
    """Transposed, it would blame the model for the opposite mistake to the one it made."""
    report = report_for([0, 0, 2, 2], [0, 0, 0, 0])
    high = ML_RISK_CLASSES.index("HIGH")
    low = ML_RISK_CLASSES.index("LOW")
    assert report.confusion[high][low] == 2
    assert report.confusion[low][high] == 0


def test_the_sample_and_patient_counts_are_both_reported() -> None:
    """Windows are not patients. 88 windows from 8 patients is a much smaller test set than
    88 suggests, and the card says both."""
    truth = [0, 1, 2, 0, 1, 2]
    groups = np.array([1, 1, 1, 2, 2, 2])
    model = Fixed(one_hot(truth))
    report = evaluate_model(
        model, np.zeros((6, 1)), np.asarray(truth), groups, with_importances=False
    )
    assert (report.n_samples, report.n_patients) == (6, 2)


def test_patients_are_zero_when_no_grouping_was_supplied() -> None:
    """Zero reads as "not recorded"; inventing ``n_samples`` would read as one window each."""
    assert report_for([0, 1, 2]).n_patients == 0


def test_the_headline_names_the_two_balanced_metrics_and_the_sample_size() -> None:
    report = report_for([0, 1, 2])
    assert "macro-F1" in report.headline
    assert "balanced accuracy" in report.headline
    assert "3 windows" in report.headline


def test_the_dict_carries_everything_the_model_card_renders() -> None:
    payload = report_for([0, 1, 2]).as_dict()
    assert set(payload) == {
        "n_samples",
        "n_patients",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "log_loss",
        "per_class",
        "confusion",
        "confusion_labels",
        "roc_auc",
        "average_precision",
        "calibration",
        "importances",
    }


def test_scores_are_rounded_for_a_readable_artefact() -> None:
    """The card is read by people and diffed by git; 17 decimal places help neither."""
    report = report_for([0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 2, 2])
    assert report.macro_f1 == round(report.macro_f1, 4)
    assert report.per_class["LOW"]["precision"] == round(report.per_class["LOW"]["precision"], 4)


# ------------------------------------------------------------------------- discrimination


def test_discrimination_is_reported_per_class_and_as_a_macro_average() -> None:
    report = report_for([0, 1, 2, 0, 1, 2])
    assert set(report.roc_auc) == {*ML_RISK_CLASSES, "macro"}
    assert set(report.average_precision) == {*ML_RISK_CLASSES, "macro"}


def test_a_class_with_no_positives_has_no_auc_rather_than_a_flattering_one() -> None:
    """0.0 would read as "useless at HIGH" and 1.0 as "perfect at HIGH". Neither was measured."""
    report = report_for([0, 0, 1, 1])
    assert np.isnan(report.roc_auc["HIGH"])
    assert np.isnan(report.average_precision["HIGH"])


def test_the_macro_average_ignores_the_classes_it_could_not_score() -> None:
    """Averaging in a nan would erase the two classes that *were* measured."""
    report = report_for([0, 0, 1, 1])
    assert np.isfinite(report.roc_auc["macro"])
    assert report.roc_auc["macro"] == pytest.approx(
        float(np.mean([report.roc_auc["LOW"], report.roc_auc["MEDIUM"]]))
    )


@pytest.mark.filterwarnings("ignore:.*single label.*:UserWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_a_test_set_of_one_class_leaves_the_macro_undefined() -> None:
    """Nothing was measurable, and saying so beats a number with no content.

    sklearn warns about the degenerate input, which is the point of the test - so the warning
    is silenced here rather than left to look like a defect in the suite output.
    """
    report = report_for([0, 0, 0])
    assert np.isnan(report.roc_auc["macro"])
    assert np.isnan(report.average_precision["macro"])


def test_a_model_with_fewer_columns_than_classes_is_scored_on_what_it_has() -> None:
    """A fold that never saw HIGH produces a two-column ``predict_proba``. Indexing a third
    column would raise, mid-training, after the expensive part."""
    model = Fixed(one_hot([0, 1, 0, 1], n_classes=2))
    report = evaluate_model(model, np.zeros((4, 1)), np.array([0, 1, 0, 1]), with_importances=False)
    assert set(report.roc_auc) == {"LOW", "MEDIUM", "macro"}
    assert report.calibration["note"] == "HIGH class absent"
    assert report.calibration["bins"] == []


# --------------------------------------------------------------------------- calibration


def test_calibration_is_measured_on_the_class_the_gauge_shows() -> None:
    report = report_for([0, 1, 2, 0, 1, 2])
    assert report.calibration["target_class"] == "HIGH"


def test_a_well_calibrated_model_has_a_small_brier_score() -> None:
    report = report_for([2, 2, 0, 0])
    assert report.calibration["brier"] < 0.05


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true:UserWarning")
def test_a_confidently_wrong_model_has_a_large_brier_score() -> None:
    """The number that catches a model whose gauge reads 90 % on patients who were fine."""
    report = report_for([0, 0, 0, 0], [2, 2, 2, 2])
    assert report.calibration["brier"] > 0.5


def test_the_reliability_curve_pairs_predicted_with_observed() -> None:
    report = report_for([2, 2, 0, 0])
    for row in report.calibration["bins"]:
        assert 0.0 <= row["mean_predicted"] <= 1.0
        assert 0.0 <= row["observed_frequency"] <= 1.0
        assert row["count"] >= 1


def test_empty_reliability_bins_are_omitted_rather_than_drawn_as_zero() -> None:
    """A bin nobody landed in is not a bin where the model reads 0 % - it is no data, and a
    chart drawing it at zero invents a dip in the curve."""
    report = report_for([2, 2, 0, 0])
    assert 0 < len(report.calibration["bins"]) < CALIBRATION_BINS


def test_every_prediction_lands_in_exactly_one_bin() -> None:
    """The top bin is closed at 1.0; a half-open one would silently drop p=1.0 predictions."""
    truth = [0, 1, 2, 2, 0, 1]
    report = report_for(truth, confidence=1.0)
    assert sum(row["count"] for row in report.calibration["bins"]) == len(truth)


@pytest.mark.filterwarnings("ignore:.*single label.*:UserWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_the_expected_calibration_error_is_zero_when_the_gauge_is_right() -> None:
    report = report_for([2, 2, 2, 2], confidence=1.0)
    assert report.calibration["expected_calibration_error"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------- importances


@pytest.fixture
def fitted() -> tuple[Any, Any, np.ndarray]:
    """A real estimator on real columns - permutation importance needs both."""
    dataset = prepare_dataset(make_window_frame(patients=9, windows_per_patient=2))
    estimator = DecisionTreeClassifier(max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    return estimator, dataset.X, dataset.y


def test_importances_are_skipped_when_not_asked_for(fitted: tuple[Any, Any, np.ndarray]) -> None:
    """It dominates training time, which is why ``--no-importances`` exists."""
    estimator, features, y = fitted
    assert evaluate_model(estimator, features, y, with_importances=False).importances == []


def test_importances_name_the_columns_they_measured(fitted: tuple[Any, Any, np.ndarray]) -> None:
    estimator, features, y = fitted
    rows = evaluate_model(estimator, features, y, with_importances=True, seed=1).importances
    assert rows
    assert all(row["feature"] in set(FEATURE_NAMES) for row in rows)
    assert all({"feature", "importance", "std"} == set(row) for row in rows)


def test_importances_are_the_top_few_in_descending_order(
    fitted: tuple[Any, Any, np.ndarray],
) -> None:
    """Fifty-one rows is a data dump; the card shows the ones that carried the model."""
    estimator, features, y = fitted
    rows = evaluate_model(estimator, features, y, with_importances=True, seed=1).importances
    assert len(rows) <= TOP_FEATURES
    values = [row["importance"] for row in rows]
    assert values == sorted(values, reverse=True)


def test_a_failure_inside_importance_costs_the_table_not_the_run(
    fitted: tuple[Any, Any, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last step of a run that has already fitted and scored everything. Losing all of
    it to an optional table would be the wrong trade."""

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("permutation failed")

    monkeypatch.setattr(evaluate_module, "permutation_importance", explode)
    estimator, features, y = fitted
    report = evaluate_model(estimator, features, y, with_importances=True)
    assert report.importances == []
    assert report.macro_f1 > 0.0


def test_importances_fall_back_to_the_feature_list_when_the_matrix_has_no_columns() -> None:
    """A caller may pass a bare ndarray; the names still have to come from somewhere."""
    dataset = prepare_dataset(make_window_frame(patients=9, windows_per_patient=2))
    estimator = DummyClassifier(strategy="prior").fit(dataset.X.to_numpy(), dataset.y)
    rows = evaluate_model(
        estimator, dataset.X.to_numpy(), dataset.y, with_importances=True, seed=1
    ).importances
    assert all(row["feature"] in set(FEATURE_NAMES) for row in rows)


def test_the_report_is_reproducible_for_the_same_seed(
    fitted: tuple[Any, Any, np.ndarray],
) -> None:
    estimator, features, y = fitted
    first = evaluate_model(estimator, features, y, with_importances=True, seed=5).as_dict()
    second = evaluate_model(estimator, features, y, with_importances=True, seed=5).as_dict()
    assert first == second
