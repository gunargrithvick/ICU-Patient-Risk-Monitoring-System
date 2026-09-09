"""Shared fixtures.

Two rules the whole suite depends on:

**No test touches the real project.** Every fixture that can write is pointed at
``tmp_path`` and every database is ``sqlite://`` in memory, so running the suite can never
overwrite ``data/processed`` or the trained artefact in ``artifacts/``. Settings are built
by keyword rather than from the environment, which also makes the suite immune to whatever
``ICU_*`` variables or ``.env`` file happen to be present.

**Vision and the model are off unless a test is about them.** Both are optional at runtime,
so leaving them out of the default fixture keeps the suite fast *and* exercises the
degraded path that a bare clone actually runs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from icu_monitor.config import Settings  # noqa: E402
from icu_monitor.core.types import (  # noqa: E402
    ClinicalState,
    Consciousness,
    Patient,
    Vitals,
)
from icu_monitor.ml.features import FEATURE_NAMES  # noqa: E402

EPOCH = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def make_vitals(*, at: datetime | None = None, **overrides: object) -> Vitals:
    """A healthy observation - NEWS2 total 0 - with named channels overridden.

    Tests that care about one parameter say so and inherit normal values for the rest,
    which keeps a hypoxia test from accidentally also being a tachycardia test.
    """
    values: dict[str, object] = {
        "heart_rate": 75.0,
        "spo2": 98.0,
        "bp_systolic": 120.0,
        "bp_diastolic": 75.0,
        "resp_rate": 16.0,
        "temperature": 36.8,
        "consciousness": Consciousness.ALERT,
        "on_supplemental_oxygen": False,
        "gcs": 15.0,
        "recorded_at": at or EPOCH,
    }
    values.update(overrides)
    return Vitals(**values)  # type: ignore[arg-type]


def make_patient(patient_id: str = "P001", **overrides: object) -> Patient:
    values: dict[str, object] = {
        "patient_id": patient_id,
        "bed": f"ICU-{patient_id[-2:]}",
        "display_name": "Test Patient",
        "age": 67,
        "sex": "F",
        "admitted_at": EPOCH - timedelta(hours=30),
        "primary_diagnosis": "Community-acquired pneumonia",
        "state": ClinicalState.STABLE,
    }
    values.update(overrides)
    return Patient(**values)  # type: ignore[arg-type]


def make_window_frame(
    *,
    patients: int = 15,
    windows_per_patient: int = 4,
    classes: Sequence[str] = ("LOW", "MEDIUM", "HIGH"),
    seed: int = 7,
) -> pd.DataFrame:
    """A window table shaped like the ETL's output, small enough to train on in a test.

    Two properties earn their keep. **One label and one ``record_id`` per patient**, because
    that is what the real table looks like and what makes a leaking split detectable - a
    patient on both sides of a split is a visible fact, not a statistical suspicion. And
    **features drawn around a per-class centre**, so a model fitted here scores well above
    chance; on noise, a training test would only assert that the code ran.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(patients):
        acuity = classes[index % len(classes)]
        centre = float(classes.index(acuity)) * 4.0
        for window in range(windows_per_patient):
            row: dict[str, object] = {
                "record_id": 1000 + index,
                "acuity": acuity,
                "window_index": window,
            }
            for name in FEATURE_NAMES:
                row[name] = float(rng.normal(centre, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    """Settings isolated to ``tmp_path``, with vision and persistence off by default."""
    return Settings(
        project_root=tmp_path,
        database_url="sqlite://",
        bed_count=4,
        tick_seconds=0.25,
        simulation_seed=424242,
        frame_source="off",
        detector="off",
        api_warmup_ticks=0,
        audible_alerts=False,
        api_key=None,
    )


@pytest.fixture
def vitals() -> Vitals:
    return make_vitals()


@pytest.fixture
def patient() -> Patient:
    return make_patient()
