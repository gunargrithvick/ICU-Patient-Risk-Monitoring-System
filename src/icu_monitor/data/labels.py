"""Acuity labelling for the PhysioNet Challenge 2012 cohort.

The problem with the original dataset
-------------------------------------
The first version of this project labelled a stay ``HIGH`` if the patient died in
hospital and ``LOW`` otherwise. That produces a **two**-class target, yet the
dashboard offered three risk levels - so ``MEDIUM`` was unreachable by
construction and the middle of the gauge was decoration.

The fix
-------
Build a genuine three-class *acuity* target from the outcome file, which carries
severity as well as mortality:

============ ==================================================================
``HIGH``     died in hospital
``MEDIUM``   survived, but with a severe or complicated course - admission SOFA
             at or above :attr:`~icu_monitor.config.Settings.label_sofa_medium`,
             or length of stay at or above
             :attr:`~icu_monitor.config.Settings.label_los_medium_days` days
``LOW``      survived with neither marker
============ ==================================================================

Both cut points are configuration, not magic numbers, and both are reported in
the model card so a reader can see exactly what the model was asked to learn.

Why these cut points
--------------------
Over the 4 000 stays in ``set-a`` the admission SOFA distribution has median 7 and
upper quartile 9, and length of stay has median 10 days and upper quartile 17. The
defaults - **SOFA ≥ 9** and **length of stay ≥ 14 days** - sit at recognised
severity marks rather than being tuned for balance: a SOFA at or above 9 indicates
substantial multi-organ dysfunction, and a stay of two weeks or more is the common
definition of a prolonged ICU admission. Together they yield roughly 44 % ``LOW`` /
42 % ``MEDIUM`` / 14 % ``HIGH``, which is a workable target without any class being
vanishingly rare.

Honest limitations
------------------
* **SOFA is partly derived from physiology that also feeds the features.** A stay
  labelled ``MEDIUM`` because of its SOFA score shares information with the
  cardiovascular, respiratory, and neurological channels the model reads. The
  ``MEDIUM``/``LOW`` boundary is therefore easier than a truly prospective task.
  ``HIGH`` (mortality) does not have this problem.
* The label describes the **whole stay**, while a feature window covers a few
  hours. A window early in a stay that ended badly still carries the ``HIGH``
  label, so the target is "this patient is on a bad trajectory", not "this
  patient is deteriorating right now".
* This is a teaching dataset. Nothing here is validated for clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import RiskLevel

#: Columns expected in ``Outcomes-a.txt``.
OUTCOME_COLUMNS: tuple[str, ...] = (
    "RecordID",
    "SAPS-I",
    "SOFA",
    "Length_of_stay",
    "Survival",
    "In-hospital_death",
)

#: The challenge file uses -1 for "not recorded" in several columns.
MISSING_SENTINEL = -1


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """A serialisable record of how labels were produced, for the model card."""

    sofa_medium: int
    los_medium_days: int

    def describe(self) -> dict[str, object]:
        return {
            "scheme": "three-class ICU acuity",
            "HIGH": "In-hospital death",
            "MEDIUM": (
                f"Survived with SOFA >= {self.sofa_medium} "
                f"or length of stay >= {self.los_medium_days} days"
            ),
            "LOW": "Survived with neither severity marker",
            "sofa_medium_threshold": self.sofa_medium,
            "los_medium_threshold_days": self.los_medium_days,
            "caveat": (
                "SOFA is derived from physiology that also feeds the features, so the "
                "MEDIUM/LOW boundary shares information with the inputs. Treat reported "
                "MEDIUM performance as optimistic."
            ),
        }


def load_outcomes(path, *, config: Settings | None = None) -> pd.DataFrame:
    """Read ``Outcomes-a.txt`` and attach the acuity label.

    Returns a frame indexed by ``RecordID`` with an added ``acuity`` column.
    """
    cfg = config or default_settings
    frame = pd.read_csv(path)

    missing = [column for column in OUTCOME_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: outcome file is missing required columns {missing}. "
            f"Expected the PhysioNet Challenge 2012 format {list(OUTCOME_COLUMNS)}."
        )

    frame = frame.copy()
    for column in ("SAPS-I", "SOFA", "Length_of_stay"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            MISSING_SENTINEL, np.nan
        )
    frame["In-hospital_death"] = (
        pd.to_numeric(frame["In-hospital_death"], errors="coerce").fillna(0).astype(int)
    )

    frame["acuity"] = assign_acuity(frame, config=cfg)
    return frame.set_index("RecordID", drop=False)


def assign_acuity(outcomes: pd.DataFrame, *, config: Settings | None = None) -> pd.Series:
    """Vectorised three-class acuity label for an outcomes frame."""
    cfg = config or default_settings

    died = outcomes["In-hospital_death"].astype(int) == 1
    sofa = pd.to_numeric(outcomes.get("SOFA"), errors="coerce")
    los = pd.to_numeric(outcomes.get("Length_of_stay"), errors="coerce")

    severe = (sofa >= cfg.label_sofa_medium).fillna(False) | (
        los >= cfg.label_los_medium_days
    ).fillna(False)

    labels = pd.Series(RiskLevel.LOW.value, index=outcomes.index, dtype=object)
    labels[severe] = RiskLevel.MEDIUM.value
    labels[died] = RiskLevel.HIGH.value
    return labels


def label_definition(config: Settings | None = None) -> LabelDefinition:
    cfg = config or default_settings
    return LabelDefinition(
        sofa_medium=cfg.label_sofa_medium,
        los_medium_days=cfg.label_los_medium_days,
    )


def class_distribution(labels: pd.Series) -> dict[str, int]:
    """Counts per class in the canonical class order."""
    counts = labels.value_counts().to_dict()
    return {
        level.value: int(counts.get(level.value, 0))
        for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
    }


__all__ = [
    "MISSING_SENTINEL",
    "OUTCOME_COLUMNS",
    "LabelDefinition",
    "assign_acuity",
    "class_distribution",
    "label_definition",
    "load_outcomes",
]
