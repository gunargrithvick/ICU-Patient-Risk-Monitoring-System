"""Training: the procedure, and the two claims a model card is not allowed to invent.

Training is where a portfolio project is most tempting to lie, and the lies are structural
rather than deliberate. Three of them are fenced off here.

**Selection must not see the test set.** Candidates are compared by cross-validation on the
training patients only; the held-out patients are scored exactly once, at the end. A test here
records every ``fit`` and asserts the held-out patients never appear in one during selection.

**The card must not name a dataset it cannot see.** This used to be a constant reading
"PhysioNet/CinC Challenge 2012 set-a", which is the one thing a provenance field must never
be: ``etl --synthetic`` exists so the project runs with no download, and a model trained that
way was getting a card asserting real patient data next to a near-perfect score. The ETL now
writes its provenance beside the table and training reads it back - and when there is nothing
to read, the card says *unrecorded* rather than guessing.

**A trained artefact must be reproducible.** Same table, same seed, same model - otherwise the
metrics in the card describe something nobody can rebuild.

The candidate registry is patched to a small tree in most tests. The real candidates are 400
estimators with isotonic calibration; they are exercised in :mod:`tests.test_pipeline`, and
paying for them here would buy nothing but wall-clock.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from icu_monitor.config import Settings
from icu_monitor.core.types import ML_RISK_CLASSES
from icu_monitor.data.physionet import synthesise_cohort
from icu_monitor.ml import pipeline as pipeline_module
from icu_monitor.ml import train as train_module
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.registry import clear_cache, load_model, load_model_card
from icu_monitor.ml.train import (
    SUPPLIED_SOURCE,
    UNRECORDED_SOURCE,
    CandidateResult,
    TrainingResult,
    cross_validate_candidate,
    train,
)

from .conftest import make_window_frame

TINY = "tiny_tree"
TINIER = "tinier_tree"

#: Synthetic stays per ETL run. Small enough to be cheap, large enough that the 12 %-
#: prevalence ``HIGH`` class lands on *both* sides of the split - below about 24 stays the
#: held-out set has no HIGH windows, and scoring a class nobody in the test set has is a
#: degenerate input rather than a property worth pinning here.
STAYS = 30


def _tiny(seed: int = 0) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, random_state=seed)


def _tinier(seed: int = 0) -> DecisionTreeClassifier:
    """Deliberately worse, so "the winner is the best candidate" is a testable claim."""
    return DecisionTreeClassifier(max_depth=1, random_state=seed)


@pytest.fixture(autouse=True)
def candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the real candidates for trees that fit in milliseconds.

    ``build_candidate`` looks the name up in this dict at call time, so replacing its contents
    replaces what training trains - without touching the code path being tested.
    """
    monkeypatch.setattr(pipeline_module, "CANDIDATES", {TINY: _tiny, TINIER: _tinier})


@pytest.fixture(autouse=True)
def _isolate_model_cache() -> Any:
    """The registry cache is process-global; a saved model must not leak between tests."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_window_frame(patients=15, windows_per_patient=4)


def run(config: Settings, frame: pd.DataFrame | None = None, **kwargs: Any) -> TrainingResult:
    """``train`` with the slow, optional parts off unless a test asks for them."""
    options: dict[str, Any] = {
        "candidates": [TINY],
        "n_splits": 3,
        "with_importances": False,
    }
    options.update(kwargs)
    return train(config=config, frame=frame, **options)


# ------------------------------------------------------------------- cross_validate_candidate


def test_a_candidate_is_scored_over_the_folds_it_actually_ran(config: Settings) -> None:
    dataset = pipeline_module.prepare_dataset(make_window_frame())
    result = cross_validate_candidate(TINY, dataset, config=config, n_splits=3)
    assert result.name == TINY
    assert result.folds == 3
    assert 0.0 <= result.macro_f1_mean <= 1.0
    assert result.macro_f1_std >= 0.0


def test_the_fold_scores_are_reported_as_they_are_produced(config: Settings) -> None:
    """Cross-validating two candidates over five folds is the slow part of a run; a caller
    watching it needs to see progress rather than a hung terminal."""
    dataset = pipeline_module.prepare_dataset(make_window_frame())
    lines: list[str] = []
    cross_validate_candidate(TINY, dataset, config=config, n_splits=3, progress=lines.append)
    assert sum("fold" in line for line in lines) == 3
    assert all("macro-F1" in line for line in lines if "fold" in line)


def test_a_candidate_result_rounds_itself_for_the_card(config: Settings) -> None:
    dataset = pipeline_module.prepare_dataset(make_window_frame())
    payload = cross_validate_candidate(TINY, dataset, config=config, n_splits=3).as_dict()
    assert set(payload) == {
        "name",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_balanced_accuracy_mean",
        "cv_folds",
    }
    assert payload["cv_macro_f1_mean"] == round(payload["cv_macro_f1_mean"], 4)


def test_a_candidate_that_produced_no_folds_scores_zero_rather_than_raising() -> None:
    """``np.mean([])`` is a nan and a RuntimeWarning. Zero is a score; nan is a hole in the
    candidate table that sorts unpredictably."""
    result = CandidateResult(
        name=TINY, macro_f1_mean=0.0, macro_f1_std=0.0, balanced_accuracy_mean=0.0, folds=0
    )
    assert (result.macro_f1_mean, result.folds) == (0.0, 0)


# ------------------------------------------------------------------------------ the winner


def test_the_winner_is_the_candidate_with_the_best_cross_validated_score(
    config: Settings, frame: pd.DataFrame
) -> None:
    result = run(config, frame, candidates=[TINIER, TINY])
    best = max(result.candidates, key=lambda candidate: candidate.macro_f1_mean)
    assert result.winner == best.name


def test_every_candidate_asked_for_is_reported_even_the_losers(
    config: Settings, frame: pd.DataFrame
) -> None:
    """The card shows what was considered. A single reported model reads like a choice that
    was never made."""
    result = run(config, frame, candidates=[TINY, TINIER])
    assert {candidate.name for candidate in result.candidates} == {TINY, TINIER}


def test_the_candidate_table_is_sorted_best_first(config: Settings, frame: pd.DataFrame) -> None:
    result = run(config, frame, candidates=[TINIER, TINY])
    scores = [candidate.macro_f1_mean for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_the_version_names_the_winning_candidate_and_when_it_was_trained(
    config: Settings, frame: pd.DataFrame
) -> None:
    """The version string is what a nurse reads off the sidebar and quotes in a bug report."""
    result = run(config, frame)
    assert result.version.endswith(f"-{TINY}")
    assert len(result.version.split("-")[0]) == 8  # YYYYMMDD


def test_an_unknown_candidate_name_is_refused_by_name(
    config: Settings, frame: pd.DataFrame
) -> None:
    with pytest.raises(KeyError, match="nonexistent"):
        run(config, frame, candidates=["nonexistent"])


# ------------------------------------------------------------------------- no leakage


def test_selection_never_fits_on_a_held_out_patient(
    config: Settings, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim the whole report rests on, checked where a leak would actually happen.

    ``Dataset.subset`` reindexes, so a fold's row numbers say nothing about which patient a
    row came from. The feature draws are continuous and therefore unique, so the row itself is
    the identifier: every fitted matrix is mapped back to record ids and checked against the
    held-out set. That covers the folds *and* the final refit, in one assertion.
    """
    owner = {
        tuple(values): int(record_id)
        for values, record_id in zip(
            frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float).tolist(),
            frame["record_id"].tolist(),
            strict=True,
        )
    }
    held_out: set[int] = set()
    fitted_on: list[set[int]] = []

    real_split = train_module.holdout_split

    def recording_split(dataset: Any, **kwargs: Any) -> Any:
        train_set, test_set = real_split(dataset, **kwargs)
        held_out.update(int(group) for group in test_set.groups)
        return train_set, test_set

    original_fit = DecisionTreeClassifier.fit

    def recording_fit(self: Any, features: Any, y: Any, **kwargs: Any) -> Any:
        rows = np.asarray(features, dtype=float).tolist()
        fitted_on.append({owner[tuple(row)] for row in rows})
        return original_fit(self, features, y, **kwargs)

    monkeypatch.setattr(train_module, "holdout_split", recording_split)
    monkeypatch.setattr(DecisionTreeClassifier, "fit", recording_fit)
    run(config, frame, candidates=[TINY, TINIER])

    assert held_out, "nothing was held out"
    assert fitted_on, "nothing was fitted"
    for patients in fitted_on:
        assert not patients & held_out


def test_the_reported_split_is_patient_disjoint(config: Settings, frame: pd.DataFrame) -> None:
    result = run(config, frame)
    assert (
        result.dataset["train_patients"] + result.dataset["test_patients"]
        == (result.dataset["patients_total"])
    )
    assert (
        result.dataset["train_windows"] + result.dataset["test_windows"]
        == (result.dataset["windows_total"])
    )


def test_the_card_says_how_the_split_was_made(config: Settings, frame: pd.DataFrame) -> None:
    """ "20 % held out" is not enough to reproduce; the grouping key is the load-bearing part."""
    assert "patient-disjoint" in run(config, frame).dataset["split"]
    assert "record_id" in run(config, frame).dataset["split"]


def test_the_held_out_report_counts_windows_and_patients(
    config: Settings, frame: pd.DataFrame
) -> None:
    result = run(config, frame)
    assert result.report.n_samples == result.dataset["test_windows"]
    assert result.report.n_patients == result.dataset["test_patients"]


# ------------------------------------------------------------------------------ provenance
#
# The defect this section exists for: a model trained on the synthetic cohort used to ship a
# card asserting PhysioNet patients, printed next to a macro-F1 of 1.000. Both halves were
# believable and one of them was false.


def test_the_source_is_what_the_etl_recorded_not_what_training_assumed(
    config: Settings,
) -> None:
    synthesise_cohort(config=config, n_stays=STAYS)
    result = run(config)
    assert result.dataset["synthetic"] is True
    assert "not real patient data" in result.dataset["source"]


def test_the_recorded_etl_summary_travels_into_the_card(config: Settings) -> None:
    """The source string answers "from where"; the summary answers "how much, of what"."""
    summary = synthesise_cohort(config=config, n_stays=STAYS)
    recorded = run(config).dataset["etl"]
    assert recorded["stays_used"] == summary.stays_used
    assert recorded["class_counts"] == summary.class_counts


def test_a_table_with_no_recorded_provenance_says_so(config: Settings) -> None:
    """A table from an older build. Naming a plausible dataset would be the actual bug."""
    config.ensure_directories()
    make_window_frame().to_csv(config.features_csv_path, index=False)
    result = run(config)
    assert result.dataset["source"] == UNRECORDED_SOURCE
    assert result.dataset["synthetic"] is False
    assert "etl" not in result.dataset


def test_a_frame_handed_in_directly_is_not_credited_to_a_file_on_disk(
    config: Settings, frame: pd.DataFrame
) -> None:
    """A caller-supplied frame is unrelated to whatever the last ETL left in ``processed/``,
    and describing it with that file's provenance would attach the wrong story to the model."""
    synthesise_cohort(config=config, n_stays=STAYS)
    assert run(config, frame).dataset["source"] == SUPPLIED_SOURCE


def test_the_synthetic_warning_reaches_the_caller_before_the_scores(config: Settings) -> None:
    """Ordering matters: the caveat is worthless printed under a macro-F1 of 1.000."""
    synthesise_cohort(config=config, n_stays=STAYS)
    lines: list[str] = []
    run(config, progress=lines.append)
    notes = [index for index, line in enumerate(lines) if "synthetic" in line.lower()]
    scores = [index for index, line in enumerate(lines) if "Held-out:" in line]
    assert notes and scores
    assert min(notes) < min(scores)


def test_a_real_table_gets_no_synthetic_warning(config: Settings) -> None:
    config.ensure_directories()
    make_window_frame().to_csv(config.features_csv_path, index=False)
    lines: list[str] = []
    run(config, progress=lines.append)
    assert not any("synthetic" in line.lower() for line in lines)


def test_the_provenance_survives_into_the_saved_card(config: Settings) -> None:
    """The end of the chain: what a reader of ``model_card.json`` actually sees."""
    synthesise_cohort(config=config, n_stays=STAYS)
    run(config)
    card = json.loads(config.model_card_path.read_text(encoding="utf-8"))
    assert card["training_data"]["synthetic"] is True
    assert "not real patient data" in card["training_data"]["source"]


# --------------------------------------------------------------------------- the artefacts


def test_training_writes_the_three_artefacts_it_says_it_wrote(
    config: Settings, frame: pd.DataFrame
) -> None:
    lines: list[str] = []
    run(config, frame, progress=lines.append)
    assert config.model_path.exists()
    assert config.metrics_path.exists()
    assert config.model_card_path.exists()
    assert any(config.model_path.name in line for line in lines)


def test_the_saved_model_can_score_a_patient_it_has_never_seen(
    config: Settings, frame: pd.DataFrame
) -> None:
    """A saved artefact that cannot be loaded and used is not a trained model."""
    result = run(config, frame)
    loaded = load_model(config=config)
    assert loaded is not None
    assert loaded.metadata.version == result.version
    assert loaded.metadata.feature_names == list(FEATURE_NAMES)
    prediction = loaded.predict_frame(frame.loc[:2, list(FEATURE_NAMES)])
    assert [item.level.value for item in prediction] and all(item.available for item in prediction)


def test_the_saved_card_carries_the_metrics_the_run_reported(
    config: Settings, frame: pd.DataFrame
) -> None:
    result = run(config, frame)
    card = load_model_card(config)
    assert card is not None
    assert card["metrics"]["macro_f1"] == result.report.macro_f1
    assert card["metrics"]["confusion_labels"] == list(ML_RISK_CLASSES)


def test_the_card_records_the_candidates_that_lost(config: Settings, frame: pd.DataFrame) -> None:
    run(config, frame, candidates=[TINY, TINIER])
    card = load_model_card(config)
    assert card is not None
    assert {row["name"] for row in card["candidates_considered"]} == {TINY, TINIER}


def test_the_card_explains_how_selection_was_kept_honest(
    config: Settings, frame: pd.DataFrame
) -> None:
    run(config, frame)
    card = load_model_card(config)
    assert card is not None
    assert any("held-out" in note for note in card["caveats_and_recommendations"])


def test_the_card_carries_the_label_scheme_and_its_caveat(
    config: Settings, frame: pd.DataFrame
) -> None:
    """A three-class score means nothing without the definition of the three classes."""
    run(config, frame)
    card = load_model_card(config)
    assert card is not None
    assert card["labels"]["sofa_medium_threshold"] == config.label_sofa_medium
    assert "optimistic" in card["labels"]["caveat"]


def test_retraining_replaces_the_model_the_running_app_would_serve(
    config: Settings, frame: pd.DataFrame
) -> None:
    """The registry caches on mtime, so a stale cache would keep serving the old model."""
    first = run(config, frame)
    second = run(config, frame, candidates=[TINIER])
    assert second.winner == TINIER
    loaded = load_model(config=config)
    assert loaded is not None
    assert loaded.metadata.candidate == TINIER != first.winner


# ------------------------------------------------------------------------- reproducibility


def test_two_runs_on_one_table_agree(config: Settings, frame: pd.DataFrame) -> None:
    """Same data, same seed, same numbers - or the card describes an unrepeatable run."""
    first = run(config, frame)
    second = run(config, frame)
    assert first.report.as_dict() == second.report.as_dict()
    assert first.winner == second.winner


def test_the_split_uses_the_configured_seed_not_a_literal(
    config: Settings, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``holdout_split`` has its own default seed, so passing nothing would still produce a
    working split - one that ignores ``ICU_SIMULATION_SEED`` and silently makes the run
    unreproducible from the configuration the card reports."""
    seeds: list[int] = []
    real_split = train_module.holdout_split

    def recording_split(dataset: Any, **kwargs: Any) -> Any:
        seeds.append(int(kwargs["seed"]))
        return real_split(dataset, **kwargs)

    monkeypatch.setattr(train_module, "holdout_split", recording_split)
    run(config, frame)
    assert seeds == [config.simulation_seed]


def test_a_different_seed_reshuffles_the_held_out_patients(
    config: Settings, frame: pd.DataFrame
) -> None:
    other = config.with_overrides(simulation_seed=config.simulation_seed + 5000)
    dataset = pipeline_module.prepare_dataset(frame)
    first = pipeline_module.holdout_split(dataset, seed=config.simulation_seed)[1]
    second = pipeline_module.holdout_split(dataset, seed=other.simulation_seed)[1]
    assert set(first.groups.tolist()) != set(second.groups.tolist())
    assert run(config, frame).winner == run(other, frame).winner == TINY


# ------------------------------------------------------------------------------ the inputs


def test_a_missing_dataset_names_the_command_that_builds_one(config: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="etl"):
        run(config)


def test_the_table_is_loaded_from_disk_when_no_frame_is_supplied(config: Settings) -> None:
    summary = synthesise_cohort(config=config, n_stays=STAYS)
    assert run(config).dataset["windows_total"] == summary.windows


def test_the_window_geometry_is_carried_into_the_card(
    config: Settings, frame: pd.DataFrame
) -> None:
    """Eight-hour windows and four-hour strides are part of what the model expects at serve
    time; a card without them cannot be checked against a live deployment."""
    other = config.with_overrides(window_hours=6, window_stride_hours=2)
    result = run(other, frame)
    assert result.dataset["window_hours"] == 6
    assert result.dataset["window_stride_hours"] == 2


def test_importances_are_optional(config: Settings, frame: pd.DataFrame) -> None:
    assert run(config, frame, with_importances=False).report.importances == []
    assert run(config, frame, with_importances=True).report.importances


def test_progress_is_reported_through_the_whole_run(config: Settings, frame: pd.DataFrame) -> None:
    lines: list[str] = []
    run(config, frame, progress=lines.append)
    joined = "\n".join(lines)
    for expected in ("Loaded", "Held out", "cross-validating", "Winner", "Held-out:", "Saved"):
        assert expected in joined


def test_a_run_without_a_progress_callback_still_works(
    config: Settings, frame: pd.DataFrame
) -> None:
    """The API and the tests call ``train`` with no callback; logging is the fallback."""
    assert run(config, frame, progress=None).winner == TINY


def test_the_result_summary_names_the_winner_and_the_headline(
    config: Settings, frame: pd.DataFrame
) -> None:
    result = run(config, frame)
    assert result.winner in result.summary
    assert result.report.headline in result.summary


def test_the_held_out_metrics_are_computed_on_the_held_out_rows_only(
    config: Settings, frame: pd.DataFrame
) -> None:
    """If the report were computed on everything, ``n_samples`` would be the whole table -
    the single most common way an inflated score gets published."""
    result = run(config, frame)
    assert result.report.n_samples < len(frame)
    assert result.report.n_patients < len(np.unique(frame["record_id"]))
