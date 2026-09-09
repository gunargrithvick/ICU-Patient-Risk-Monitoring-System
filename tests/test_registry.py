"""The model registry: the artefact's contract, and the missing-model path.

The original project called ``joblib.load`` on a relative path at import time, so the app
died on any machine whose working directory differed and gave no clue why. Three properties
replace that, and this module pins each one:

*The artefact carries its own contract.* Feature names and class order are saved beside the
estimator, ``_align`` reindexes incoming frames onto the *trained* order, and
:meth:`RiskModel.schema_matches_code` reports drift rather than letting a rescrambled matrix
through.

*A missing model is a state, not a crash.* No file, unreadable file, wrong format - all
return ``None`` so NEWS2 and the vision channel carry on.

*Reloads are cheap but not stale.* The cache is keyed on the file's mtime, so retraining is
picked up without a restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from icu_monitor import __version__
from icu_monitor.config import Settings
from icu_monitor.core.types import ML_RISK_CLASSES, RiskLevel
from icu_monitor.ml.features import FEATURE_NAMES
from icu_monitor.ml.registry import (
    BUNDLE_FORMAT,
    ModelMetadata,
    RiskModel,
    build_model_card,
    clear_cache,
    load_metrics,
    load_model,
    load_model_card,
    save_model,
)

from .conftest import EPOCH, make_patient, make_vitals


class Recording:
    """A stand-in estimator: fixed probabilities, and it remembers what it was handed.

    Defined at module scope so ``joblib`` can pickle it, which is what lets the save/load
    round-trip be a real round-trip rather than two assertions about a mock.
    """

    def __init__(self, probabilities: tuple[float, ...] = (0.2, 0.3, 0.5)) -> None:
        self.probabilities = list(probabilities)
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen = frame.copy()
        return np.tile(np.asarray(self.probabilities, dtype=float), (len(frame), 1))


def metadata(**overrides: object) -> ModelMetadata:
    values: dict[str, object] = {
        "version": "20260905-1203-test",
        "candidate": "hist_gradient_boosting",
        "trained_at": EPOCH.isoformat(),
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


def model(probabilities: tuple[float, ...] = (0.2, 0.3, 0.5), **overrides: object) -> RiskModel:
    return RiskModel(Recording(probabilities), metadata(**overrides))


@pytest.fixture(autouse=True)
def _isolate_cache() -> None:
    """The registry cache is process-global; no test may inherit another's model."""
    clear_cache()
    yield
    clear_cache()


# --------------------------------------------------------------------------- the contract


def test_metadata_defaults_to_the_running_codes_schema() -> None:
    saved = metadata()
    assert saved.feature_names == list(FEATURE_NAMES)
    assert saved.classes == list(ML_RISK_CLASSES)
    assert saved.bundle_format == BUNDLE_FORMAT
    assert saved.package_version == __version__


def test_metadata_renders_its_training_date_for_humans() -> None:
    assert metadata().trained_on == "05 Sep 2026 12:00 UTC"


def test_an_unparseable_training_date_is_shown_as_stored() -> None:
    """Better a strange-looking date on the model card than a traceback on page load."""
    assert metadata(trained_at="last Tuesday").trained_on == "last Tuesday"


def test_a_matching_schema_is_reported_as_matching() -> None:
    matches, message = model().schema_matches_code()
    assert matches is True
    assert "matches" in message


def test_schema_drift_is_named_and_counted() -> None:
    """Drift is the failure that produces plausible numbers from a rescrambled matrix."""
    drifted = model(feature_names=[*FEATURE_NAMES[:-2], "an_old_column"])
    matches, message = drifted.schema_matches_code()
    assert matches is False
    assert "drift" in message.lower()
    assert "2 new" in message and "1 removed" in message
    assert "icu_monitor train" in message


def test_the_feature_list_is_a_copy_not_the_models_own() -> None:
    instance = model()
    instance.feature_names.append("mutated")
    assert instance.feature_names == list(FEATURE_NAMES)


def test_an_artefact_with_no_saved_schema_falls_back_to_the_code() -> None:
    """Metadata predating the field is readable; it is not a reason to refuse the model."""
    instance = model(feature_names=[], classes=[])
    assert instance.feature_names == list(FEATURE_NAMES)
    assert instance.predict_one({"heart_rate_last": 90.0}).level in set(RiskLevel)


# ------------------------------------------------------------------------------ alignment


def test_prediction_reindexes_onto_the_trained_column_order() -> None:
    """The estimator must see the columns it was fitted on, in that order.

    Column order is positional to the estimator, so a frame built by a newer feature module
    has to be reindexed rather than passed through. Nothing here would raise if it were not -
    the numbers would just quietly be wrong.
    """
    instance = model(feature_names=["spo2_last", "heart_rate_last"])
    instance.predict_one({"heart_rate_last": 88.0, "spo2_last": 94.0})

    seen = instance.estimator.seen
    assert list(seen.columns) == ["spo2_last", "heart_rate_last"]
    assert seen.iloc[0].tolist() == [94.0, 88.0]


def test_a_column_the_artefact_wants_but_the_code_dropped_arrives_missing() -> None:
    instance = model(feature_names=["heart_rate_last", "a_retired_feature"])
    instance.predict_one({"heart_rate_last": 88.0})

    seen = instance.estimator.seen
    assert list(seen.columns) == ["heart_rate_last", "a_retired_feature"]
    assert pd.isna(seen.loc[0, "a_retired_feature"])


def test_columns_the_artefact_never_saw_are_dropped() -> None:
    instance = model(feature_names=["heart_rate_last"])
    instance.predict_frame(pd.DataFrame([{"heart_rate_last": 88.0, "brand_new_feature": 1.0}]))
    assert list(instance.estimator.seen.columns) == ["heart_rate_last"]


def test_alignment_does_not_mutate_the_callers_frame() -> None:
    instance = model(feature_names=["heart_rate_last", "spo2_last"])
    frame = pd.DataFrame([{"heart_rate_last": 88.0}])
    instance.predict_frame(frame)
    assert list(frame.columns) == ["heart_rate_last"]


# ----------------------------------------------------------------------------- inference


def test_the_predicted_level_is_the_most_probable_class() -> None:
    prediction = model((0.1, 0.7, 0.2)).predict_one({"heart_rate_last": 90.0})
    assert prediction.level is RiskLevel.MEDIUM
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.available is True
    assert prediction.model_version == "20260905-1203-test"


def test_the_distribution_is_labelled_by_class_name() -> None:
    """The dashboard prints these keys, and fusion reads ``HIGH`` off them by name."""
    prediction = model((0.5, 0.3, 0.2)).predict_one({"heart_rate_last": 90.0})
    assert prediction.probabilities == {
        "LOW": pytest.approx(0.5),
        "MEDIUM": pytest.approx(0.3),
        "HIGH": pytest.approx(0.2),
    }


def test_a_batch_predicts_one_result_per_row() -> None:
    frame = pd.DataFrame([{"heart_rate_last": value} for value in (70.0, 110.0, 140.0)])
    assert len(model().predict_frame(frame)) == 3


def test_an_empty_batch_predicts_nothing() -> None:
    assert model().predict_frame(pd.DataFrame(columns=list(FEATURE_NAMES))) == []


def test_an_empty_row_is_unavailable_not_an_exception() -> None:
    """``MLPrediction.unavailable`` carries the reason in ``model_version``, which is what
    the dashboard prints in place of a version string."""
    prediction = model().predict_one(pd.DataFrame(columns=list(FEATURE_NAMES)))
    assert prediction.available is False
    assert "empty" in prediction.model_version.lower()


def test_a_shorter_probability_vector_is_read_as_far_as_it_goes() -> None:
    """A binary artefact behind a three-class wrapper must not be indexed off the end."""
    prediction = model((0.4, 0.6)).predict_one({"heart_rate_last": 90.0})
    assert set(prediction.probabilities) == {"LOW", "MEDIUM"}
    assert prediction.level is RiskLevel.MEDIUM


# -------------------------------------------------------------------------- from a patient


def test_a_live_patient_is_scored_from_their_buffer() -> None:
    history = [make_vitals(at=EPOCH), make_vitals(at=EPOCH, heart_rate=120.0)]
    prediction = model().predict_for_patient(make_patient(), history)
    assert prediction.available is True
    assert prediction.model_version == "20260905-1203-test"


def test_a_patient_with_no_observations_is_not_guessed_at() -> None:
    """First tick after startup. An empty buffer is reported, never scored as a zero row."""
    prediction = model().predict_for_patient(make_patient(), [])
    assert prediction.available is False
    assert "no observations" in prediction.model_version


def test_the_window_comes_from_the_artefact_not_the_caller() -> None:
    """The model was fitted on an 8 h window; serving it a 24 h summary is train/serve skew.

    Heart rate rises with age here, so a wider window reaches further back into the higher
    readings and ``heart_rate_max`` moves. Nothing raises if the window is wrong - the row
    just summarises a different stretch of time than the one the model was fitted on.
    """
    from datetime import timedelta

    history = [
        make_vitals(at=EPOCH - timedelta(hours=ago), heart_rate=80.0 + ago)
        for ago in range(20, -1, -1)
    ]
    instance = model(window_hours=8)

    instance.predict_for_patient(make_patient(), history)
    assert instance.estimator.seen.loc[0, "heart_rate_max"] == pytest.approx(88.0)

    instance.predict_for_patient(make_patient(), history, window_hours=24.0)
    assert instance.estimator.seen.loc[0, "heart_rate_max"] == pytest.approx(100.0)


# ------------------------------------------------------------------------------- saving


def test_saving_writes_the_bundle_and_both_sidecars(config: Settings) -> None:
    path = save_model(Recording(), metadata(), config=config)
    assert path == config.model_path
    assert config.model_path.exists()
    assert json.loads(config.metrics_path.read_text(encoding="utf-8")) == {}
    assert "model_details" in json.loads(config.model_card_path.read_text(encoding="utf-8"))


def test_saving_creates_the_artefact_directory(config: Settings) -> None:
    """``save_model`` after a fresh clone must not fail on a missing ``artifacts/``."""
    assert not config.model_path.parent.exists()
    save_model(Recording(), metadata(), config=config)
    assert config.model_path.parent.is_dir()


def test_the_metrics_sidecar_is_the_metrics_verbatim(config: Settings) -> None:
    save_model(Recording(), metadata(metrics={"macro_f1": 0.517}), config=config)
    assert load_metrics(config) == {"macro_f1": 0.517}


# ------------------------------------------------------------------------------ loading


def test_a_saved_model_loads_and_predicts(config: Settings) -> None:
    save_model(Recording((0.1, 0.2, 0.7)), metadata(), config=config)
    loaded = load_model(config=config)
    assert loaded is not None
    assert loaded.version == "20260905-1203-test"
    assert loaded.predict_one({"heart_rate_last": 90.0}).level is RiskLevel.HIGH


def test_no_artefact_is_none_not_an_error(config: Settings) -> None:
    """The bare-clone path: nothing trained yet, and the ward still has to open."""
    assert load_model(config=config) is None


def test_an_unreadable_artefact_is_none(config: Settings, caplog) -> None:
    config.ensure_directories()
    config.model_path.write_bytes(b"not a joblib bundle")
    assert load_model(config=config) is None
    assert "Could not load model" in caplog.text


def test_an_artefact_in_the_wrong_shape_is_refused(config: Settings, caplog) -> None:
    """A bare pickled estimator from an older version has no metadata to trust."""
    config.ensure_directories()
    joblib.dump(Recording(), config.model_path)
    assert load_model(config=config) is None
    assert "bundle format" in caplog.text


def test_a_bundle_without_metadata_loads_with_placeholders(config: Settings) -> None:
    config.ensure_directories()
    joblib.dump({"format": BUNDLE_FORMAT, "estimator": Recording()}, config.model_path)
    loaded = load_model(config=config)
    assert loaded is not None
    assert loaded.version == "unversioned"
    assert loaded.metadata.candidate == "unknown"


def test_unknown_metadata_keys_are_ignored(config: Settings) -> None:
    """A newer artefact read by older code drops the fields it does not know about.

    The alternative is ``TypeError: unexpected keyword argument``, which would turn a
    forward-compatible file into a hard failure at startup.
    """
    config.ensure_directories()
    joblib.dump(
        {
            "format": BUNDLE_FORMAT,
            "estimator": Recording(),
            "metadata": {**metadata().as_dict(), "calibration_curve": [1, 2, 3]},
        },
        config.model_path,
    )
    loaded = load_model(config=config)
    assert loaded is not None
    assert loaded.version == "20260905-1203-test"


def test_drifted_schema_loads_with_a_warning_rather_than_refusing(config: Settings, caplog) -> None:
    """A stale artefact is still better than nothing, but it must say so in the log."""
    save_model(Recording(), metadata(feature_names=["heart_rate_last"]), config=config)
    assert load_model(config=config) is not None
    assert "drift" in caplog.text.lower()


# -------------------------------------------------------------------------------- caching


def test_a_second_load_is_the_same_object(config: Settings) -> None:
    save_model(Recording(), metadata(), config=config)
    assert load_model(config=config) is load_model(config=config)


def test_retraining_is_picked_up_without_a_restart(config: Settings) -> None:
    """``reload_model`` on the dashboard depends on this; so does a rebuilt container.

    The cache is keyed on the file's modification time rather than on its path, so a newly
    written artefact invalidates it on the next read.
    """
    save_model(Recording(), metadata(version="first"), config=config)
    first = load_model(config=config)
    stamp = config.model_path.stat().st_mtime

    save_model(Recording(), metadata(version="second"), config=config)
    import os

    os.utime(config.model_path, (stamp + 10, stamp + 10))

    second = load_model(config=config)
    assert first is not None and second is not None
    assert second.version == "second"


def test_refresh_reloads_from_disk(config: Settings) -> None:
    save_model(Recording(), metadata(), config=config)
    first = load_model(config=config)
    assert load_model(config=config, refresh=True) is not first


def test_clearing_the_cache_forces_a_reload(config: Settings) -> None:
    save_model(Recording(), metadata(), config=config)
    first = load_model(config=config)
    clear_cache()
    assert load_model(config=config) is not first


# ----------------------------------------------------------------------------- model card


def test_the_card_has_the_sections_a_reviewer_looks_for(config: Settings) -> None:
    card = build_model_card(metadata(), config=config)
    assert {
        "model_details",
        "intended_use",
        "training_data",
        "labels",
        "metrics",
        "ethical_considerations",
        "caveats_and_recommendations",
    } <= set(card)


def test_the_card_counts_the_features_it_was_trained_on(config: Settings) -> None:
    card = build_model_card(metadata(), config=config)
    assert card["model_details"]["n_features"] == len(FEATURE_NAMES)
    assert card["model_details"]["version"] == "20260905-1203-test"
    assert card["model_details"]["estimator"] == "hist_gradient_boosting"


def test_the_card_says_it_is_not_a_medical_device(config: Settings) -> None:
    """The most important sentence in the document, and the easiest one to omit."""
    out_of_scope = " ".join(
        build_model_card(metadata(), config=config)["intended_use"]["out_of_scope"]
    )
    assert "medical device" in out_of_scope
    assert "clinical decision" in out_of_scope.lower()


def test_the_card_carries_the_leakage_and_grouping_caveats(config: Settings) -> None:
    card = build_model_card(metadata(), config=config)
    leakage = card["ethical_considerations"]["label_leakage"].lower()
    assert "sofa" in leakage and "optimistic" in leakage
    assert any("patient-disjoint" in caveat for caveat in card["caveats_and_recommendations"])
    assert "race" in card["ethical_considerations"]["fairness"].lower()


def test_the_card_does_not_expose_machine_specific_dataset_paths(config: Settings) -> None:
    card = build_model_card(
        metadata(
            dataset={
                "source": "test",
                "etl": {"paths": {"csv": r"C:\\Users\\private\\data\\windows.csv"}},
            }
        ),
        config=config,
    )
    paths = card["training_data"]["etl"]["paths"]
    assert paths == {"csv": "windows.csv"}
    assert "private" not in json.dumps(card)


def test_training_notes_reach_the_card(config: Settings) -> None:
    card = build_model_card(metadata(notes="Trained on set-a only."), config=config)
    assert "Trained on set-a only." in card["caveats_and_recommendations"]


def test_without_notes_the_card_says_how_to_retrain(config: Settings) -> None:
    card = build_model_card(metadata(), config=config)
    assert any("icu_monitor train" in caveat for caveat in card["caveats_and_recommendations"])


# -------------------------------------------------------------------------- reading back


def test_the_card_and_metrics_are_absent_before_training(config: Settings) -> None:
    assert load_model_card(config) is None
    assert load_metrics(config) is None


def test_the_saved_card_reads_back(config: Settings) -> None:
    save_model(Recording(), metadata(metrics={"accuracy": 0.573}), config=config)
    card = load_model_card(config)
    assert card is not None
    assert card["metrics"] == {"accuracy": 0.573}


def test_a_corrupt_card_is_none_not_a_broken_page(config: Settings) -> None:
    """Model Insights degrades to "no card" rather than 500-ing on malformed JSON."""
    config.ensure_directories()
    config.model_card_path.write_text("{not json", encoding="utf-8")
    config.metrics_path.write_text("{not json", encoding="utf-8")
    assert load_model_card(config) is None
    assert load_metrics(config) is None


def test_paths_come_from_configuration(config: Settings, tmp_path: Path) -> None:
    """Nothing in the registry knows a relative path - the original project's fatal bug."""
    save_model(Recording(), metadata(), config=config)
    assert config.model_path.is_relative_to(tmp_path)
