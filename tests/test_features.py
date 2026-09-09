"""Feature engineering, and the train/serve parity that keeps it honest.

Train/serve skew is the classic way a working notebook becomes a broken product: the ETL
summarises a window one way, the live path summarises it another, and the model silently
receives inputs it was never fitted on. Both paths here call the same functions, so the
test that matters most is the one at the bottom - the same window, pushed through the
offline and the online entry points, has to produce the same row.

The other theme is that **missing stays missing**. The estimator handles ``NaN`` natively,
so nothing is imputed with a fake "normal" value; a channel that was never measured must
arrive as ``NaN`` rather than as a plausible-looking zero.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from icu_monitor.ml.features import (
    AGGREGATES,
    DERIVED_FEATURES,
    FEATURE_CHANNELS,
    FEATURE_NAMES,
    STATIC_FEATURES,
    build_feature_row,
    channels_from_history,
    features_for_patient,
    features_to_frame,
    static_from_patient,
    summarise_channel,
)

from .conftest import EPOCH, make_patient, make_vitals

# ------------------------------------------------------------------------ the schema


def test_the_design_matrix_has_the_shape_it_advertises() -> None:
    assert len(FEATURE_NAMES) == len(FEATURE_CHANNELS) * len(AGGREGATES) + len(
        STATIC_FEATURES
    ) + len(DERIVED_FEATURES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_column_order_is_channel_major() -> None:
    """A trained artefact is keyed to this order; reordering it silently rescrambles input."""
    expected = [
        f"{channel}_{aggregate}" for channel in FEATURE_CHANNELS for aggregate in AGGREGATES
    ]
    assert list(FEATURE_NAMES[: len(expected)]) == expected
    assert list(FEATURE_NAMES[-len(DERIVED_FEATURES) :]) == list(DERIVED_FEATURES)


# ------------------------------------------------------------- summarising one channel


def test_a_flat_channel_summarises_to_itself() -> None:
    summary = summarise_channel(np.array([0.0, 1.0, 2.0]), np.array([80.0, 80.0, 80.0]))
    assert summary["last"] == 80.0
    assert summary["mean"] == 80.0
    assert summary["min"] == summary["max"] == 80.0
    assert summary["std"] == 0.0
    assert summary["slope"] == pytest.approx(0.0)


def test_slope_is_in_units_per_hour() -> None:
    """A clinician reads "falling 2 % an hour" off a chart; the feature says the same thing."""
    hours = np.array([0.0, 1.0, 2.0, 3.0])
    summary = summarise_channel(hours, np.array([100.0, 98.0, 96.0, 94.0]))
    assert summary["slope"] == pytest.approx(-2.0)

    rising = summarise_channel(hours, np.array([60.0, 65.0, 70.0, 75.0]))
    assert rising["slope"] == pytest.approx(5.0)


def test_samples_are_ordered_by_time_not_by_position() -> None:
    """``last`` means latest. Source rows are not guaranteed to arrive sorted."""
    summary = summarise_channel(np.array([2.0, 0.0, 1.0]), np.array([70.0, 90.0, 80.0]))
    assert summary["last"] == 70.0
    assert summary["slope"] == pytest.approx(-10.0)


def test_an_empty_channel_is_all_missing() -> None:
    summary = summarise_channel(np.array([]), np.array([]))
    assert set(summary) == set(AGGREGATES)
    assert all(math.isnan(value) for value in summary.values())


def test_unusable_samples_are_dropped_not_scored() -> None:
    summary = summarise_channel(
        np.array([0.0, 1.0, 2.0, 3.0]), np.array([80.0, float("nan"), float("inf"), 90.0])
    )
    assert summary["mean"] == pytest.approx(85.0)
    assert summary["last"] == 90.0


def test_an_all_missing_channel_is_missing() -> None:
    summary = summarise_channel(np.array([0.0, 1.0]), np.array([float("nan"), float("nan")]))
    assert all(math.isnan(value) for value in summary.values())


def test_one_sample_has_no_variability_and_no_trend() -> None:
    """Zero std is a fact - nothing varied. An unknown slope is ``NaN``, not zero.

    Reporting slope 0 from a single point would tell the model "this patient is stable"
    on the strength of one reading, which is the same lie as imputing a normal value.
    """
    summary = summarise_channel(np.array([1.0]), np.array([120.0]))
    assert summary["std"] == 0.0
    assert math.isnan(summary["slope"])
    assert summary["last"] == summary["mean"] == 120.0


def test_simultaneous_samples_have_no_slope() -> None:
    summary = summarise_channel(np.array([2.0, 2.0]), np.array([80.0, 96.0]))
    assert math.isnan(summary["slope"])
    assert summary["mean"] == pytest.approx(88.0)


def test_variability_is_reported() -> None:
    summary = summarise_channel(np.array([0.0, 1.0]), np.array([70.0, 90.0]))
    assert summary["std"] == pytest.approx(10.0)
    assert summary["min"] == 70.0
    assert summary["max"] == 90.0


# --------------------------------------------------------------------------- one row


def channels(**series: tuple[list[float], list[float]]) -> dict:
    return dict(series)


def test_a_row_carries_every_declared_feature() -> None:
    row = build_feature_row(channels(heart_rate=([0.0, 1.0], [80.0, 90.0])))
    assert set(row) == set(FEATURE_NAMES)


def test_an_unmeasured_channel_is_missing_not_zero() -> None:
    """The whole reason the estimator is a histogram booster: ``NaN`` is a value it reads."""
    row = build_feature_row(channels(heart_rate=([0.0, 1.0], [80.0, 90.0])))
    assert row["heart_rate_last"] == 90.0
    assert math.isnan(row["spo2_last"])
    assert math.isnan(row["temperature_mean"])


def test_static_descriptors_are_carried_through() -> None:
    row = build_feature_row({}, {"age": 71, "sex_male": 1, "icu_type": 4})
    assert row["age"] == 71.0
    assert row["sex_male"] == 1.0
    assert row["icu_type"] == 4.0


def test_absent_statics_are_missing() -> None:
    row = build_feature_row({}, None)
    assert math.isnan(row["age"])
    assert math.isnan(row["weight_kg"])


def test_bmi_is_derived_when_it_can_be() -> None:
    row = build_feature_row({}, {"weight_kg": 80.0, "height_cm": 180.0})
    assert row["bmi"] == pytest.approx(80.0 / 1.8**2)


def test_a_supplied_bmi_wins_over_the_derivation() -> None:
    row = build_feature_row({}, {"weight_kg": 80.0, "height_cm": 180.0, "bmi": 25.0})
    assert row["bmi"] == 25.0


@pytest.mark.parametrize(
    "static",
    [
        {"weight_kg": 80.0, "height_cm": 0.0},
        {"weight_kg": 80.0, "height_cm": 18.0},  # metres recorded as centimetres
        {"weight_kg": 4.0, "height_cm": 180.0},
        {"weight_kg": None, "height_cm": 180.0},
    ],
)
def test_an_implausible_bmi_is_refused(static: dict) -> None:
    """Unit errors are endemic in the source file, and 2 469 kg/m² is not a feature."""
    assert math.isnan(build_feature_row({}, static)["bmi"])


def test_the_derived_features_are_the_clinical_ratios() -> None:
    row = build_feature_row(
        channels(
            heart_rate=([0.0], [120.0]),
            bp_systolic=([0.0], [90.0]),
            bp_diastolic=([0.0], [60.0]),
        )
    )
    assert row["shock_index"] == pytest.approx(120.0 / 90.0)
    assert row["pulse_pressure"] == pytest.approx(30.0)
    assert row["map_estimate"] == pytest.approx((90.0 + 120.0) / 3)


def test_derived_features_need_both_of_their_inputs() -> None:
    row = build_feature_row(channels(heart_rate=([0.0], [120.0])))
    assert math.isnan(row["shock_index"])
    assert math.isnan(row["pulse_pressure"])


def test_a_zero_pressure_does_not_divide() -> None:
    row = build_feature_row(channels(heart_rate=([0.0], [120.0]), bp_systolic=([0.0], [0.0])))
    assert math.isnan(row["shock_index"])


# ------------------------------------------------------------------------- the frame


def test_the_frame_uses_the_canonical_column_order() -> None:
    frame = features_to_frame([build_feature_row(channels(spo2=([0.0, 1.0], [95.0, 92.0])))])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 1
    # Floating point throughout: an object column would mean a value arrived as a string
    # and every downstream estimator would either refuse it or one-hot encode it.
    assert all(dtype.kind == "f" for dtype in frame.dtypes)


def test_an_empty_batch_still_has_the_schema() -> None:
    """``model.predict`` on an empty frame must fail loudly on rows, not on columns."""
    frame = features_to_frame([])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert frame.empty


def test_partial_rows_are_filled_and_extras_dropped() -> None:
    frame = features_to_frame([{"heart_rate_last": 88.0, "not_a_feature": 1.0}])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert frame.loc[0, "heart_rate_last"] == 88.0
    assert pd.isna(frame.loc[0, "spo2_last"])


def test_rows_stack_in_order() -> None:
    rows = [{"heart_rate_last": value} for value in (70.0, 80.0, 90.0)]
    frame = features_to_frame(rows)
    assert frame["heart_rate_last"].tolist() == [70.0, 80.0, 90.0]


# ------------------------------------------------------------------- the live path


def history(count: int, *, minutes: float = 5.0, **overrides: object) -> list:
    """``count`` observations spaced ``minutes`` apart, oldest first."""
    return [
        make_vitals(at=EPOCH + timedelta(minutes=minutes * index), **overrides)
        for index in range(count)
    ]


def test_history_is_measured_backwards_from_the_newest_observation() -> None:
    """Windows are relative: the model sees "two hours ago", never a wall-clock time."""
    result = channels_from_history(history(5, minutes=30.0))
    hours, values = result["heart_rate"]
    assert hours == pytest.approx([-2.0, -1.5, -1.0, -0.5, 0.0])
    assert len(values) == 5


def test_an_empty_history_yields_empty_channels() -> None:
    result = channels_from_history([])
    assert set(result) == set(FEATURE_CHANNELS)
    assert all(hours == [] and values == [] for hours, values in result.values())


def test_the_window_drops_older_observations() -> None:
    result = channels_from_history(history(13, minutes=30.0), window_hours=2.0)
    hours, _values = result["heart_rate"]
    assert len(hours) == 5
    assert min(hours) == pytest.approx(-2.0)


def test_a_dropped_out_channel_contributes_no_sample() -> None:
    """A missing SpO₂ shortens that channel's series; it does not pad it with a value."""
    observations = [
        make_vitals(at=EPOCH),
        make_vitals(at=EPOCH + timedelta(minutes=5), spo2=None),
        make_vitals(at=EPOCH + timedelta(minutes=10)),
    ]
    result = channels_from_history(observations)
    assert len(result["spo2"][1]) == 2
    assert len(result["heart_rate"][1]) == 3


def test_a_patients_statics_come_from_the_record() -> None:
    male = static_from_patient(make_patient(sex="M"))
    female = static_from_patient(make_patient(sex="F"))
    assert male["sex_male"] == 1.0
    assert female["sex_male"] == 0.0
    assert male["age"] == 67.0
    assert math.isnan(male["weight_kg"])


def test_a_live_patient_produces_one_scoreable_row() -> None:
    frame = features_for_patient(make_patient(), history(6))
    assert frame.shape == (1, len(FEATURE_NAMES))
    assert frame.loc[0, "heart_rate_last"] == 75.0
    assert frame.loc[0, "age"] == 67.0


def test_a_patient_with_no_history_still_produces_a_row() -> None:
    """The first tick after startup has nothing to summarise and must still score."""
    frame = features_for_patient(make_patient(), [])
    assert frame.shape == (1, len(FEATURE_NAMES))
    assert frame["age"].notna().all()


# ----------------------------------------------------------------- train/serve parity


def test_the_offline_and_online_paths_agree_on_the_same_window() -> None:
    """The invariant the whole module exists for.

    The ETL builds a row from arrays of ``(hours, values)``; the engine builds one from a
    buffer of observations. If those two disagree the model is fitted on one distribution
    and served another, and no amount of accuracy on the test split would reveal it.
    """
    observations = [
        make_vitals(
            at=EPOCH + timedelta(hours=index),
            heart_rate=80.0 + 4 * index,
            spo2=97.0 - index,
            bp_systolic=120.0 - 2 * index,
            bp_diastolic=70.0 - index,
        )
        for index in range(5)
    ]
    patient = make_patient()

    online = features_for_patient(patient, observations)
    offline = features_to_frame(
        [build_feature_row(channels_from_history(observations), static_from_patient(patient))]
    )
    pd.testing.assert_frame_equal(online, offline)


def test_a_deteriorating_window_reads_as_deteriorating() -> None:
    """Sanity on the direction of the trend features - a sign error here is invisible."""
    falling = [
        make_vitals(at=EPOCH + timedelta(minutes=30 * index), spo2=98.0 - 1.5 * index)
        for index in range(6)
    ]
    frame = features_for_patient(make_patient(), falling)
    assert frame.loc[0, "spo2_slope"] < 0
    assert frame.loc[0, "spo2_last"] < frame.loc[0, "spo2_max"]
