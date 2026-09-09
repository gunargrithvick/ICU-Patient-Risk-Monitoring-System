"""Acuity labelling: the fix for a target that made MEDIUM unreachable.

The original project labelled a stay ``HIGH`` on in-hospital death and ``LOW`` otherwise -
a two-class target behind a three-level dashboard, so the middle of the gauge was
decoration. These tests pin the three-class replacement: the cut points come from
configuration, the ordering is *died beats severe beats neither*, and the ``-1`` sentinel
the challenge file uses for "not recorded" is never read as a real zero.

The labelling caveat is asserted too. ``SOFA`` is derived from physiology that also feeds
the features, so the MEDIUM/LOW boundary shares information with the inputs; the model card
has to say so, and :meth:`LabelDefinition.describe` is where that text comes from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icu_monitor.config import Settings
from icu_monitor.core.types import RiskLevel
from icu_monitor.data.labels import (
    MISSING_SENTINEL,
    OUTCOME_COLUMNS,
    assign_acuity,
    class_distribution,
    label_definition,
    load_outcomes,
)


def outcomes(*rows: dict) -> pd.DataFrame:
    """An outcomes frame in the PhysioNet Challenge 2012 layout."""
    defaults = {
        "RecordID": 0,
        "SAPS-I": 14,
        "SOFA": 4,
        "Length_of_stay": 6,
        "Survival": -1,
        "In-hospital_death": 0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


# ------------------------------------------------------------------------- the ladder


def test_death_is_high(config: Settings) -> None:
    frame = outcomes({"RecordID": 1, "In-hospital_death": 1, "SOFA": 2, "Length_of_stay": 2})
    assert assign_acuity(frame, config=config).tolist() == [RiskLevel.HIGH.value]


def test_a_severe_survivor_is_medium(config: Settings) -> None:
    by_sofa = outcomes({"SOFA": config.label_sofa_medium, "Length_of_stay": 2})
    by_stay = outcomes({"SOFA": 1, "Length_of_stay": config.label_los_medium_days})
    assert assign_acuity(by_sofa, config=config).tolist() == [RiskLevel.MEDIUM.value]
    assert assign_acuity(by_stay, config=config).tolist() == [RiskLevel.MEDIUM.value]


def test_a_survivor_with_neither_marker_is_low(config: Settings) -> None:
    frame = outcomes({"SOFA": config.label_sofa_medium - 1, "Length_of_stay": 3})
    assert assign_acuity(frame, config=config).tolist() == [RiskLevel.LOW.value]


def test_death_outranks_severity(config: Settings) -> None:
    """A patient who died is not "medium because their stay was short"."""
    frame = outcomes({"In-hospital_death": 1, "SOFA": 20, "Length_of_stay": 90})
    assert assign_acuity(frame, config=config).tolist() == [RiskLevel.HIGH.value]


def test_the_cut_points_are_thresholds_not_ranges(config: Settings) -> None:
    """One below the cut is LOW, exactly on it is MEDIUM. Off-by-one here shifts the class."""
    frame = outcomes(
        {"RecordID": 1, "SOFA": config.label_sofa_medium - 1, "Length_of_stay": 1},
        {"RecordID": 2, "SOFA": config.label_sofa_medium, "Length_of_stay": 1},
    )
    assert assign_acuity(frame, config=config).tolist() == [
        RiskLevel.LOW.value,
        RiskLevel.MEDIUM.value,
    ]


def test_the_cut_points_come_from_configuration(config: Settings) -> None:
    """A different ward can relabel without editing source - and the model card records it."""
    strict = config.with_overrides(label_sofa_medium=3, label_los_medium_days=4)
    frame = outcomes({"SOFA": 4, "Length_of_stay": 2})
    assert assign_acuity(frame, config=config).tolist() == [RiskLevel.LOW.value]
    assert assign_acuity(frame, config=strict).tolist() == [RiskLevel.MEDIUM.value]


# ------------------------------------------------------------------- missing values


def test_an_unrecorded_marker_does_not_promote_a_stay(config: Settings) -> None:
    """``NaN >= 9`` is false, and it has to stay false rather than becoming a MEDIUM."""
    frame = outcomes({"SOFA": np.nan, "Length_of_stay": np.nan})
    assert assign_acuity(frame, config=config).tolist() == [RiskLevel.LOW.value]


def test_the_minus_one_sentinel_is_not_a_measurement(tmp_path, config: Settings) -> None:
    """``-1`` means "not recorded" in this file. Read as a number it is a very low SOFA.

    The distinction is invisible in the label - both give LOW - so it is asserted on the
    parsed column instead, where a future change to the reader would show up.
    """
    path = tmp_path / "Outcomes-a.txt"
    outcomes({"RecordID": 7, "SOFA": MISSING_SENTINEL, "Length_of_stay": 5}).to_csv(
        path, index=False
    )
    frame = load_outcomes(path, config=config)
    assert pd.isna(frame.loc[7, "SOFA"])


def test_a_missing_death_flag_is_read_as_survival(tmp_path, config: Settings) -> None:
    path = tmp_path / "Outcomes-a.txt"
    frame = outcomes({"RecordID": 3, "SOFA": 2, "Length_of_stay": 2})
    frame["In-hospital_death"] = np.nan
    frame.to_csv(path, index=False)
    assert load_outcomes(path, config=config).loc[3, "acuity"] == RiskLevel.LOW.value


# ---------------------------------------------------------------------- loading a file


def test_loading_attaches_the_label_and_indexes_by_record(tmp_path, config: Settings) -> None:
    path = tmp_path / "Outcomes-a.txt"
    outcomes(
        {"RecordID": 11, "In-hospital_death": 1},
        {"RecordID": 12, "SOFA": 12},
        {"RecordID": 13, "SOFA": 1, "Length_of_stay": 2},
    ).to_csv(path, index=False)

    frame = load_outcomes(path, config=config)
    assert frame.index.tolist() == [11, 12, 13]
    assert frame.loc[11, "acuity"] == RiskLevel.HIGH.value
    assert frame.loc[12, "acuity"] == RiskLevel.MEDIUM.value
    assert frame.loc[13, "acuity"] == RiskLevel.LOW.value
    # RecordID survives as a column as well as an index, so joins do not need a reset.
    assert frame.loc[11, "RecordID"] == 11


def test_the_wrong_file_is_rejected_with_the_expected_format(tmp_path, config: Settings) -> None:
    """Pointing the ETL at the wrong download should say so, not fail three steps later."""
    path = tmp_path / "Outcomes-a.txt"
    pd.DataFrame([{"RecordID": 1, "Died": 0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_outcomes(path, config=config)
    assert set(OUTCOME_COLUMNS) >= {"RecordID", "SOFA", "Length_of_stay", "In-hospital_death"}


# ------------------------------------------------------------------------ distribution


def test_the_distribution_is_reported_in_class_order() -> None:
    labels = pd.Series(
        [RiskLevel.HIGH.value, RiskLevel.LOW.value, RiskLevel.LOW.value, RiskLevel.MEDIUM.value]
    )
    assert list(class_distribution(labels)) == [
        RiskLevel.LOW.value,
        RiskLevel.MEDIUM.value,
        RiskLevel.HIGH.value,
    ]
    assert class_distribution(labels) == {"LOW": 2, "MEDIUM": 1, "HIGH": 1}


def test_an_absent_class_is_reported_as_zero_not_omitted() -> None:
    """A training run with no HIGH examples must be visible in the card, not silently absent."""
    labels = pd.Series([RiskLevel.LOW.value, RiskLevel.LOW.value])
    assert class_distribution(labels) == {"LOW": 2, "MEDIUM": 0, "HIGH": 0}


def test_an_empty_cohort_counts_zero_everywhere() -> None:
    assert class_distribution(pd.Series([], dtype=object)) == {"LOW": 0, "MEDIUM": 0, "HIGH": 0}


# ------------------------------------------------------------------- the definition


def test_the_definition_records_the_cut_points_that_were_used(config: Settings) -> None:
    definition = label_definition(config)
    assert definition.sofa_medium == config.label_sofa_medium
    assert definition.los_medium_days == config.label_los_medium_days


def test_the_definition_describes_all_three_classes(config: Settings) -> None:
    described = label_definition(config).describe()
    assert {"LOW", "MEDIUM", "HIGH"} <= set(described)
    assert str(config.label_sofa_medium) in described["MEDIUM"]
    assert str(config.label_los_medium_days) in described["MEDIUM"]


def test_the_definition_states_the_leakage_caveat(config: Settings) -> None:
    """The MEDIUM boundary shares information with the features, and the card must say so.

    Reporting MEDIUM performance without this sentence would overstate what the model
    learned, which is the kind of quiet dishonesty a model card exists to prevent.
    """
    caveat = label_definition(config).describe()["caveat"].lower()
    assert "sofa" in caveat
    assert "optimistic" in caveat
