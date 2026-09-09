"""Splitting and candidate construction - the two places a score becomes fiction.

Almost every inflated ML result in a paper or a portfolio comes from one of two mistakes, and
this module is a fence around both.

**A patient must not appear on both sides of a split.** Each ICU stay contributes ~11
overlapping windows that share a label and near-identical physiology, so a plain
``train_test_split`` scores the model on patients it has already memorised. Every split here
groups by ``record_id``; the tests assert disjointness directly, on the group arrays, rather
than trusting that the right splitter class was named.

**Class order is a contract, not a convention.** scikit-learn sorts string labels
alphabetically - ``HIGH, LOW, MEDIUM`` - which would silently transpose two columns of
``predict_proba`` and point the dashboard's low→high gauge at the wrong number. Labels are
therefore encoded to ``0, 1, 2`` in the project's own order, and these tests pin that order
against :data:`ML_RISK_CLASSES` rather than restating it as a literal.

The remaining tests are about *refusal*: a table missing a column, a label outside the scheme,
an unknown candidate name. Each one is a real thing that happens when the ETL and the model
drift apart, and each one is better as an exception with a remedy in it than as a silent
column of ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from icu_monitor.config import Settings
from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.pipeline import (
    CANDIDATES,
    CLASS_TO_INDEX,
    GROUP_COLUMN,
    INDEX_TO_CLASS,
    TARGET_COLUMN,
    build_candidate,
    build_hist_gradient_boosting,
    build_random_forest,
    cv_folds,
    decode,
    holdout_split,
    prepare_dataset,
)

from .conftest import make_window_frame

# --------------------------------------------------------------------------- the class order


def test_the_class_order_is_the_projects_order_not_the_alphabet() -> None:
    """Alphabetical would be HIGH, LOW, MEDIUM - a gauge pointing at the wrong column."""
    assert list(CLASS_TO_INDEX) == list(ML_RISK_CLASSES)
    assert list(CLASS_TO_INDEX) != sorted(CLASS_TO_INDEX)


def test_the_index_maps_are_inverses() -> None:
    inverted = {index: name for name, index in CLASS_TO_INDEX.items()}
    assert inverted == INDEX_TO_CLASS
    assert set(INDEX_TO_CLASS) == set(range(len(ML_RISK_CLASSES)))


def test_decoding_returns_the_label_the_ui_shows() -> None:
    assert list(decode(np.array([0, 2, 1]))) == [
        ML_RISK_CLASSES[0],
        ML_RISK_CLASSES[2],
        ML_RISK_CLASSES[1],
    ]


def test_decoding_survives_the_round_trip() -> None:
    labels = np.array(["HIGH", "LOW", "MEDIUM", "HIGH"], dtype=object)
    encoded = np.array([CLASS_TO_INDEX[label] for label in labels])
    assert list(decode(encoded)) == list(labels)


# ------------------------------------------------------------------------- prepare_dataset


def test_the_frame_becomes_a_design_matrix_of_exactly_the_models_inputs() -> None:
    dataset = prepare_dataset(make_window_frame(patients=6, windows_per_patient=2))
    assert list(dataset.X.columns) == list(FEATURE_NAMES)
    assert len(dataset) == 12
    assert dataset.n_patients == 6


def test_the_columns_the_etl_adds_do_not_reach_the_model() -> None:
    """``window_index`` and ``record_id`` are bookkeeping. A model that learned them would
    be learning the order the ETL happened to write rows in."""
    frame = make_window_frame(patients=3, windows_per_patient=2)
    dataset = prepare_dataset(frame)
    assert GROUP_COLUMN not in dataset.X.columns
    assert "window_index" not in dataset.X.columns


def test_the_target_is_encoded_and_the_original_label_is_kept() -> None:
    dataset = prepare_dataset(make_window_frame(patients=3, windows_per_patient=1))
    assert dataset.y.dtype == np.dtype(int)
    for index, label in zip(dataset.y, dataset.labels, strict=True):
        assert INDEX_TO_CLASS[int(index)] == label


def test_the_groups_are_the_record_ids() -> None:
    frame = make_window_frame(patients=4, windows_per_patient=3)
    dataset = prepare_dataset(frame)
    assert sorted(set(dataset.groups.tolist())) == sorted(set(frame[GROUP_COLUMN].tolist()))


def test_the_features_are_floats_whatever_the_csv_said() -> None:
    """A CSV round-trip can hand back object columns; a model cannot fit those."""
    frame = make_window_frame(patients=3, windows_per_patient=1)
    frame[FEATURE_NAMES[0]] = frame[FEATURE_NAMES[0]].astype(str)
    dataset = prepare_dataset(frame)
    assert dataset.X[FEATURE_NAMES[0]].dtype == np.dtype(float)


@pytest.mark.parametrize("column", [GROUP_COLUMN, TARGET_COLUMN])
def test_a_table_without_the_grouping_or_target_column_is_refused(column: str) -> None:
    """Both are structural. Neither can be inferred, so the ETL is named as the fix."""
    frame = make_window_frame(patients=3, windows_per_patient=1).drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        prepare_dataset(frame)


def test_the_refusal_names_the_command_that_rebuilds_the_table() -> None:
    frame = make_window_frame(patients=2, windows_per_patient=1).drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="etl"):
        prepare_dataset(frame)


def test_a_table_missing_feature_columns_says_how_many_and_which() -> None:
    """The count matters: one missing column is a typo, forty is a version mismatch."""
    frame = make_window_frame(patients=2, windows_per_patient=1).drop(
        columns=list(FEATURE_NAMES[:6])
    )
    with pytest.raises(ValueError, match="6 feature columns") as raised:
        prepare_dataset(frame)
    assert FEATURE_NAMES[0] in str(raised.value)


def test_a_label_outside_the_scheme_is_refused_rather_than_dropped() -> None:
    """Silently dropping rows would shrink the dataset by an amount nobody reported."""
    frame = make_window_frame(patients=3, windows_per_patient=2)
    frame.loc[0, TARGET_COLUMN] = "CRITICAL"
    with pytest.raises(ValueError, match="1 rows carry a label outside"):
        prepare_dataset(frame)


def test_an_empty_table_is_refused() -> None:
    frame = make_window_frame(patients=2, windows_per_patient=1).iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        prepare_dataset(frame)


def test_missing_feature_values_are_preserved_for_the_model_to_route() -> None:
    """An arterial line that was never inserted is information. Imputing it here would
    destroy that before either candidate got to decide what to do with it."""
    frame = make_window_frame(patients=3, windows_per_patient=2)
    frame.loc[0, FEATURE_NAMES[0]] = np.nan
    dataset = prepare_dataset(frame)
    assert dataset.X[FEATURE_NAMES[0]].isna().sum() == 1


# ----------------------------------------------------------------------------- Dataset


def test_the_dataset_counts_patients_not_rows() -> None:
    dataset = prepare_dataset(make_window_frame(patients=5, windows_per_patient=4))
    assert (len(dataset), dataset.n_patients) == (20, 5)


def test_the_class_counts_name_every_class_including_absent_ones() -> None:
    """A zero is a fact the model card needs. A missing key is a KeyError somewhere later."""
    dataset = prepare_dataset(make_window_frame(patients=1, windows_per_patient=3))
    counts = dataset.class_counts()
    assert list(counts) == list(ML_RISK_CLASSES)
    assert counts["LOW"] == 3
    assert counts["HIGH"] == 0


def test_a_subset_keeps_the_four_arrays_aligned() -> None:
    dataset = prepare_dataset(make_window_frame(patients=6, windows_per_patient=2))
    subset = dataset.subset(np.array([0, 5, 7]))
    assert len(subset) == 3
    assert list(subset.y) == [dataset.y[i] for i in (0, 5, 7)]
    assert list(subset.labels) == [dataset.labels[i] for i in (0, 5, 7)]
    assert list(subset.groups) == [dataset.groups[i] for i in (0, 5, 7)]


def test_a_subset_reindexes_so_positional_lookups_stay_valid() -> None:
    """``X.iloc[i]`` and ``y[i]`` must mean the same row after subsetting."""
    dataset = prepare_dataset(make_window_frame(patients=6, windows_per_patient=2))
    subset = dataset.subset(np.array([4, 9]))
    assert list(subset.X.index) == [0, 1]


def test_a_dataset_is_frozen() -> None:
    """Splits hand the same object to several folds; a mutable one would let a fold edit it."""
    dataset = prepare_dataset(make_window_frame(patients=2, windows_per_patient=1))
    with pytest.raises(AttributeError):
        dataset.y = np.array([0])  # type: ignore[misc]


# -------------------------------------------------------------------------- holdout_split


def test_the_holdout_is_patient_disjoint() -> None:
    """The one property the reported score depends on."""
    dataset = prepare_dataset(make_window_frame())
    train, test = holdout_split(dataset, test_fraction=0.2, seed=11)
    assert not set(train.groups.tolist()) & set(test.groups.tolist())


def test_every_window_lands_on_exactly_one_side() -> None:
    dataset = prepare_dataset(make_window_frame())
    train, test = holdout_split(dataset, test_fraction=0.2, seed=11)
    assert len(train) + len(test) == len(dataset)
    assert set(train.groups.tolist()) | set(test.groups.tolist()) == set(dataset.groups.tolist())


def test_a_patients_windows_are_never_divided() -> None:
    """The whole point: 4 windows of one stay go to one side, not 3 and 1."""
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    train, test = holdout_split(dataset, test_fraction=0.2, seed=3)
    for side in (train, test):
        for group in set(side.groups.tolist()):
            expected = int((dataset.groups == group).sum())
            assert int((side.groups == group).sum()) == expected


def test_the_holdout_keeps_all_three_classes_on_both_sides() -> None:
    """Stratification is not cosmetic: a test set with no HIGH cannot score HIGH recall."""
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    train, test = holdout_split(dataset, test_fraction=0.2, seed=5)
    assert all(count > 0 for count in train.class_counts().values())
    assert all(count > 0 for count in test.class_counts().values())


def test_the_split_is_reproducible_by_seed() -> None:
    dataset = prepare_dataset(make_window_frame())
    first = holdout_split(dataset, seed=99)[1].groups.tolist()
    second = holdout_split(dataset, seed=99)[1].groups.tolist()
    assert first == second


def test_a_different_seed_selects_different_patients() -> None:
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    assert set(holdout_split(dataset, seed=1)[1].groups.tolist()) != set(
        holdout_split(dataset, seed=8)[1].groups.tolist()
    )


@pytest.mark.parametrize("fraction", [0.5, 0.34, 0.2, 0.1])
def test_a_larger_test_fraction_holds_out_more(fraction: float) -> None:
    """The fraction becomes a fold count, so it is approximate by construction - but it has
    to at least move in the right direction."""
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    _, test = holdout_split(dataset, test_fraction=fraction, seed=4)
    assert 0 < len(test) < len(dataset)
    assert len(test) / len(dataset) <= fraction * 2.5


@pytest.mark.parametrize("fraction", [0.0, -1.0, 0.01])
def test_an_absurd_test_fraction_still_produces_a_usable_split(fraction: float) -> None:
    """Clamped rather than raising: a bad ``--test-fraction`` should not lose a training run."""
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    train, test = holdout_split(dataset, test_fraction=fraction, seed=4)
    assert len(train) > 0 and len(test) > 0


# ------------------------------------------------------------------------------ cv_folds


def test_every_fold_is_patient_disjoint() -> None:
    dataset = prepare_dataset(make_window_frame())
    for train_index, valid_index in cv_folds(dataset, n_splits=3, seed=2):
        train, valid = dataset.subset(train_index), dataset.subset(valid_index)
        assert not set(train.groups.tolist()) & set(valid.groups.tolist())


def test_the_folds_partition_the_dataset() -> None:
    dataset = prepare_dataset(make_window_frame())
    seen: list[int] = []
    for _, valid_index in cv_folds(dataset, n_splits=3, seed=2):
        seen.extend(valid_index.tolist())
    assert sorted(seen) == list(range(len(dataset)))


def test_the_requested_number_of_folds_is_produced_when_the_data_allows() -> None:
    dataset = prepare_dataset(make_window_frame(patients=15, windows_per_patient=4))
    assert len(list(cv_folds(dataset, n_splits=3, seed=2))) == 3


def test_the_fold_count_is_clamped_to_what_the_smallest_class_supports() -> None:
    """Ten folds over three HIGH windows is not ten folds; asking for it should not raise."""
    dataset = prepare_dataset(make_window_frame(patients=6, windows_per_patient=1))
    folds = list(cv_folds(dataset, n_splits=10, seed=2))
    assert 2 <= len(folds) <= 6


def test_at_least_two_folds_are_always_produced() -> None:
    """One fold is not cross-validation, and zero would make the score a silent 0.0."""
    dataset = prepare_dataset(make_window_frame(patients=3, windows_per_patient=2))
    assert len(list(cv_folds(dataset, n_splits=1, seed=2))) >= 2


def test_the_folds_are_reproducible_by_seed() -> None:
    dataset = prepare_dataset(make_window_frame())
    first = [valid.tolist() for _, valid in cv_folds(dataset, n_splits=3, seed=42)]
    second = [valid.tolist() for _, valid in cv_folds(dataset, n_splits=3, seed=42)]
    assert first == second


# --------------------------------------------------------------------------- the candidates


def test_both_candidates_are_registered() -> None:
    assert set(CANDIDATES) == {"hist_gradient_boosting", "random_forest"}


def test_the_boosted_candidate_handles_missing_values_natively() -> None:
    """It is the preferred model *because* of this, so it must not gain an imputer."""
    estimator = build_hist_gradient_boosting(seed=1)
    assert not isinstance(estimator, Pipeline)
    assert estimator.class_weight == "balanced"


def test_the_forest_candidate_imputes_because_it_has_to() -> None:
    """Forests cannot consume NaN. The imputer is what makes the comparison fair."""
    pipeline = build_random_forest(seed=1)
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["impute", "forest"]


def test_the_forest_keeps_empty_feature_columns() -> None:
    """A channel absent from every window still has to occupy its column, or the matrix
    handed to the estimator is narrower than the feature list the artefact promises."""
    pipeline = build_random_forest(seed=1)
    assert dict(pipeline.steps)["impute"].keep_empty_features is True


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_a_candidate_is_built_with_the_configured_seed(name: str, config: Settings) -> None:
    """Two runs of ``train`` on one table must produce the same model, or the metrics in the
    card describe a model nobody can rebuild."""
    estimator = build_candidate(name, config=config)
    params = estimator.get_params()
    seeds = [value for key, value in params.items() if key.endswith("random_state")]
    assert config.simulation_seed in seeds


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_a_candidate_can_fit_and_predict_the_projects_three_classes(
    name: str, config: Settings
) -> None:
    """A smoke test with teeth: it proves the class order survives fitting."""
    dataset = prepare_dataset(make_window_frame(patients=9, windows_per_patient=3))
    estimator = build_candidate(name, config=config)
    estimator.fit(dataset.X, dataset.y)
    probabilities = estimator.predict_proba(dataset.X)
    assert probabilities.shape == (len(dataset), len(ML_RISK_CLASSES))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert list(estimator.classes_) == list(range(len(ML_RISK_CLASSES)))


def test_an_unknown_candidate_lists_the_ones_that_exist() -> None:
    """``--candidates randomforest`` is a typo, and the fix is in the error message."""
    with pytest.raises(KeyError, match="random_forest"):
        build_candidate("randomforest")


def test_each_call_builds_a_fresh_estimator() -> None:
    """Training clones anyway, but a shared fitted estimator between folds would leak."""
    assert build_candidate("random_forest") is not build_candidate("random_forest")


def test_a_dataset_survives_a_csv_round_trip() -> None:
    """The ETL writes CSV, so this is the shape training actually receives on a machine
    without pyarrow."""
    frame = make_window_frame(patients=6, windows_per_patient=2)
    restored = pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)))
    assert len(prepare_dataset(restored)) == len(frame)
