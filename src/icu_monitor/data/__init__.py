"""Data access: raw PhysioNet extraction and acuity labelling.

:mod:`~icu_monitor.data.labels` defines *what the model is asked to predict*;
:mod:`~icu_monitor.data.physionet` defines *what it gets to look at*. Keeping them
apart makes both auditable - the label scheme and its caveats are readable without
wading through parsing code.
"""

from __future__ import annotations

from icu_monitor.data.labels import (
    LabelDefinition,
    assign_acuity,
    class_distribution,
    label_definition,
    load_outcomes,
)
from icu_monitor.data.physionet import (
    DatasetSummary,
    StayRecord,
    build_dataset,
    build_windows,
    iter_stays,
    load_dataset_summary,
    load_windows,
    parse_record,
    synthesise_cohort,
)

__all__ = [
    "DatasetSummary",
    "LabelDefinition",
    "StayRecord",
    "assign_acuity",
    "build_dataset",
    "build_windows",
    "class_distribution",
    "iter_stays",
    "label_definition",
    "load_dataset_summary",
    "load_outcomes",
    "load_windows",
    "parse_record",
    "synthesise_cohort",
]
