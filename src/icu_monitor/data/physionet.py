"""PhysioNet/CinC Challenge 2012 extraction.

The challenge publishes one text file per ICU stay in long format
(``Time,Parameter,Value``) covering the first 48 hours, plus an outcomes file.
This module turns that into a supervised learning table:

1. parse each stay into per-channel ``(hours, values)`` arrays;
2. drop physiologically implausible values as sensor artefact;
3. slice the stay into overlapping windows (default 8 h, 4 h stride);
4. summarise every window with :func:`icu_monitor.ml.features.build_feature_row`
   - the *same* function the live dashboard uses, so there is no train/serve skew;
5. attach the stay's three-class acuity label from :mod:`icu_monitor.data.labels`.

Windows inherit a stay-level label, so ``record_id`` is carried through and used as
the grouping key when splitting - a patient must never appear in both train and
test, or the reported scores are fiction.

If the raw archive is absent the module can synthesise a statistically similar
cohort (:func:`synthesise_cohort`), which keeps ``make bootstrap`` and the Docker
image working on a machine that has never downloaded the dataset.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.constants import PHYSIONET_PARAMETER_MAP, VITAL_SPECS
from icu_monitor.core.types import RiskLevel
from icu_monitor.data.labels import class_distribution, label_definition, load_outcomes
from icu_monitor.ml.features import FEATURE_NAMES, build_feature_row

logger = logging.getLogger(__name__)

#: Extra plausibility bounds for channels without a :class:`VitalSpec`.
EXTRA_BOUNDS: dict[str, tuple[float, float]] = {
    "gcs": (3.0, 15.0),
    "map": (20.0, 200.0),
    "urine": (0.0, 5000.0),
    "fio2": (0.21, 1.0),
    "mech_vent": (0.0, 1.0),
}

#: Channels required (any ``min_channels`` of them) for a window to be usable.
REQUIRED_CHANNELS: tuple[str, ...] = ("heart_rate", "spo2", "bp_systolic")

#: Stays sampled into the replay file, per acuity class.
REPLAY_PER_CLASS = 4


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class StayRecord:
    """One ICU stay parsed into arrays."""

    record_id: int
    static: dict[str, float] = field(default_factory=dict)
    channels: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        ends = [hours[-1] for hours, _ in self.channels.values() if hours]
        return max(ends) if ends else 0.0

    def channel_coverage(self) -> int:
        return sum(1 for key in REQUIRED_CHANNELS if self.channels.get(key, ([], []))[0])

    def slice_window(
        self, start_hours: float, end_hours: float
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Samples falling in ``[start_hours, end_hours)``, per channel."""
        window: dict[str, tuple[list[float], list[float]]] = {}
        for channel, (hours, values) in self.channels.items():
            picked_hours: list[float] = []
            picked_values: list[float] = []
            for hour, value in zip(hours, values, strict=True):
                if start_hours <= hour < end_hours:
                    picked_hours.append(hour)
                    picked_values.append(value)
            if picked_hours:
                window[channel] = (picked_hours, picked_values)
        return window


def _plausible(channel: str, value: float) -> bool:
    spec = VITAL_SPECS.get(channel)
    if spec is not None:
        return spec.is_plausible(value)
    bounds = EXTRA_BOUNDS.get(channel)
    if bounds is None:
        return True
    return bounds[0] <= value <= bounds[1]


def _parse_time(token: str) -> float | None:
    """``"HH:MM"`` -> hours as a float."""
    try:
        hours_text, minutes_text = token.split(":", 1)
        return int(hours_text) + int(minutes_text) / 60.0
    except (ValueError, AttributeError):
        return None


def parse_record(text: str, record_id: int | None = None) -> StayRecord:
    """Parse one challenge record file into a :class:`StayRecord`."""
    stay = StayRecord(record_id=record_id or 0)
    channels: dict[str, tuple[list[float], list[float]]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Time,"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        time_token, parameter, value_token = parts[0], parts[1], parts[2]

        if not value_token or value_token in {"-1", "NA", "?"}:
            continue
        try:
            value = float(value_token)
        except ValueError:
            continue

        if parameter == "RecordID":
            stay.record_id = int(value)
            continue
        if parameter == "Age":
            stay.static["age"] = value if 0 < value < 120 else float("nan")
            continue
        if parameter == "Gender":
            # Challenge encoding: 1 = male, 0 = female, -1 = unknown (already dropped).
            stay.static["sex_male"] = 1.0 if value == 1 else 0.0
            continue
        if parameter == "Height":
            stay.static["height_cm"] = value if 100 <= value <= 230 else float("nan")
            continue
        if parameter == "Weight":
            # Weight is re-recorded through the stay; keep the first plausible value.
            if 25 <= value <= 300 and "weight_kg" not in stay.static:
                stay.static["weight_kg"] = value
            continue
        if parameter == "ICUType":
            stay.static["icu_type"] = value
            continue

        channel = PHYSIONET_PARAMETER_MAP.get(parameter)
        if channel is None or not _plausible(channel, value):
            continue

        hours = _parse_time(time_token)
        if hours is None:
            continue

        bucket = channels.setdefault(channel, ([], []))
        bucket[0].append(hours)
        bucket[1].append(value)

    for hours, values in channels.values():
        order = sorted(range(len(hours)), key=hours.__getitem__)
        hours[:] = [hours[i] for i in order]
        values[:] = [values[i] for i in order]

    stay.channels = channels
    return stay


def iter_stays(archive: Path, *, limit: int | None = None) -> Iterator[StayRecord]:
    """Yield every stay in ``set-a.zip`` (or an extracted directory of the same)."""
    archive = Path(archive)
    if archive.is_dir():
        members: Sequence[Path] = sorted(archive.rglob("*.txt"))
        for index, path in enumerate(members):
            if limit is not None and index >= limit:
                return
            try:
                record_id = int(path.stem)
            except ValueError:
                continue
            yield parse_record(path.read_text(encoding="utf-8", errors="replace"), record_id)
        return

    if not archive.exists():
        raise FileNotFoundError(
            f"Raw record archive not found at {archive}. Download set-a.zip from "
            "https://physionet.org/content/challenge-2012/ or run with --synthetic."
        )

    with zipfile.ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.endswith(".txt") and not n.endswith("/")]
        names.sort()
        for index, name in enumerate(names):
            if limit is not None and index >= limit:
                return
            stem = Path(name).stem
            try:
                record_id = int(stem)
            except ValueError:
                continue
            with bundle.open(name) as handle:
                text = handle.read().decode("utf-8", errors="replace")
            yield parse_record(text, record_id)


# --------------------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------------------


def build_windows(
    stay: StayRecord,
    label: str,
    *,
    config: Settings | None = None,
    min_channels: int = 2,
) -> list[dict[str, Any]]:
    """Slice a stay into labelled feature rows."""
    cfg = config or default_settings
    window = float(cfg.window_hours)
    stride = float(cfg.window_stride_hours)
    horizon = max(window, stay.duration_hours)

    rows: list[dict[str, Any]] = []
    start = 0.0
    while start + window <= horizon + stride:
        sliced = stay.slice_window(start, start + window)
        present = sum(1 for key in REQUIRED_CHANNELS if key in sliced)
        if present >= min_channels:
            features = build_feature_row(sliced, stay.static)
            rows.append(
                {
                    "record_id": stay.record_id,
                    "window_start_h": round(start, 2),
                    "window_end_h": round(start + window, 2),
                    "n_samples": sum(len(values) for _, values in sliced.values()),
                    "acuity": label,
                    **features,
                }
            )
        start += stride
    return rows


# --------------------------------------------------------------------------------------
# Dataset build
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DatasetSummary:
    """What the ETL produced, for logging and the model card."""

    stays_seen: int
    stays_labelled: int
    stays_used: int
    windows: int
    class_counts: dict[str, int]
    features: int
    source: str
    paths: dict[str, str] = field(default_factory=dict)
    #: Set by :func:`synthesise_cohort`. A flag rather than a substring match on ``source``,
    #: because a downstream consumer deciding whether to caveat a score should not have to
    #: guess from prose it did not write.
    synthetic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stays_seen": self.stays_seen,
            "stays_labelled": self.stays_labelled,
            "stays_used": self.stays_used,
            "windows": self.windows,
            "class_counts": self.class_counts,
            "features": self.features,
            "source": self.source,
            "synthetic": self.synthetic,
            "paths": self.paths,
        }


def _write_table(frame: pd.DataFrame, cfg: Settings) -> dict[str, str]:
    """Persist the window table. Parquet when available, CSV always."""
    cfg.ensure_directories()
    written: dict[str, str] = {}
    try:
        frame.to_parquet(cfg.features_path, index=False)
        written["parquet"] = str(cfg.features_path)
    except Exception as exc:  # pragma: no cover - depends on optional pyarrow
        logger.info("Parquet unavailable (%s); writing CSV only.", exc)
    frame.to_csv(cfg.features_csv_path, index=False)
    written["csv"] = str(cfg.features_csv_path)
    return written


def _write_summary(summary: DatasetSummary, cfg: Settings) -> str:
    """Record how this table was built, next to the table itself.

    Training reads this back so the model card can state its actual provenance. Keeping it
    beside the data rather than in the process means a table built yesterday by
    ``etl --synthetic`` cannot be described by a run of ``train`` today as PhysioNet's.
    """
    payload = dict(summary.as_dict())
    payload["labels"] = label_definition(cfg).describe()
    payload["window_hours"] = cfg.window_hours
    payload["window_stride_hours"] = cfg.window_stride_hours
    cfg.dataset_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(cfg.dataset_summary_path)


def load_dataset_summary(config: Settings | None = None) -> dict[str, Any] | None:
    """The provenance written beside the processed table, or ``None`` if there is none.

    ``None`` is a real answer: a table from an older build, or one a reader dropped in by hand,
    has no recorded source, and saying so is better than naming one it might not have.
    """
    cfg = config or default_settings
    path = cfg.dataset_summary_path
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable dataset summary at %s (%s)", path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


def _build_replay(
    stays: dict[int, StayRecord],
    cohort: pd.DataFrame,
    cfg: Settings,
) -> str | None:
    """Write a long-format series for a few representative stays (replay source)."""
    chosen: list[int] = []
    for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH):
        subset = cohort[cohort["acuity"] == level.value]
        subset = subset.sort_values("coverage", ascending=False)
        chosen.extend(int(rid) for rid in subset["record_id"].head(REPLAY_PER_CLASS))

    frames: list[pd.DataFrame] = []
    for record_id in chosen:
        stay = stays.get(record_id)
        if stay is None:
            continue
        acuity = str(cohort.loc[cohort["record_id"] == record_id, "acuity"].iloc[0])
        for channel, (hours, values) in stay.channels.items():
            if channel not in VITAL_SPECS and channel != "gcs":
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "record_id": record_id,
                        "acuity": acuity,
                        "channel": channel,
                        "hours": hours,
                        "value": values,
                    }
                )
            )
    if not frames:
        return None
    replay = pd.concat(frames, ignore_index=True).sort_values(["record_id", "hours"])
    replay.to_csv(cfg.replay_path, index=False)
    return str(cfg.replay_path)


def build_dataset(
    *,
    config: Settings | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> DatasetSummary:
    """Run the full ETL and write the processed tables."""
    cfg = config or default_settings
    cfg.ensure_directories()
    say = progress or (lambda message: logger.info("%s", message))

    outcomes = load_outcomes(cfg.raw_outcomes, config=cfg)
    say(f"Loaded {len(outcomes):,} stay outcomes from {cfg.raw_outcomes.name}")

    label_by_record: dict[int, str] = {
        int(record_id): str(label)
        for record_id, label in zip(outcomes["RecordID"], outcomes["acuity"], strict=True)
    }

    window_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    kept_stays: dict[int, StayRecord] = {}
    seen = labelled = used = 0

    for stay in iter_stays(cfg.raw_records_zip, limit=limit):
        seen += 1
        if seen % 500 == 0:
            say(f"  parsed {seen:,} stays, {len(window_rows):,} windows so far")

        label = label_by_record.get(stay.record_id)
        if label is None:
            continue
        labelled += 1

        rows = build_windows(stay, label, config=cfg)
        if len(rows) < cfg.min_windows_per_record:
            continue

        used += 1
        window_rows.extend(rows)
        kept_stays[stay.record_id] = stay

        outcome = outcomes.loc[stay.record_id]
        cohort_rows.append(
            {
                "record_id": stay.record_id,
                "acuity": label,
                "died": int(outcome["In-hospital_death"]),
                "sofa": float(outcome["SOFA"]) if pd.notna(outcome["SOFA"]) else np.nan,
                "saps_i": float(outcome["SAPS-I"]) if pd.notna(outcome["SAPS-I"]) else np.nan,
                "length_of_stay": (
                    float(outcome["Length_of_stay"])
                    if pd.notna(outcome["Length_of_stay"])
                    else np.nan
                ),
                "windows": len(rows),
                "coverage": stay.channel_coverage(),
                "duration_h": round(stay.duration_hours, 2),
                **{
                    key: stay.static.get(key, np.nan)
                    for key in ("age", "sex_male", "icu_type", "weight_kg", "height_cm")
                },
            }
        )

    if not window_rows:
        raise RuntimeError(
            "ETL produced no usable windows. Check that the archive contains "
            "PhysioNet Challenge 2012 records and that record IDs match the outcome file."
        )

    windows = pd.DataFrame(window_rows)
    cohort = pd.DataFrame(cohort_rows)

    paths = _write_table(windows, cfg)
    cohort.to_csv(cfg.cohort_path, index=False)
    paths["cohort"] = str(cfg.cohort_path)
    replay_path = _build_replay(kept_stays, cohort, cfg)
    if replay_path:
        paths["replay"] = replay_path
    paths["summary"] = str(cfg.dataset_summary_path)

    summary = DatasetSummary(
        stays_seen=seen,
        stays_labelled=labelled,
        stays_used=used,
        windows=len(windows),
        class_counts=class_distribution(windows["acuity"]),
        features=len(FEATURE_NAMES),
        source=f"PhysioNet Challenge 2012 set-a ({cfg.raw_records_zip.name})",
        paths=paths,
    )
    _write_summary(summary, cfg)
    say(
        f"Wrote {summary.windows:,} windows from {summary.stays_used:,} stays "
        f"→ {cfg.features_csv_path.name}"
    )
    say(f"Class balance: {summary.class_counts}")
    say(f"Label definition: {label_definition(cfg).describe()['MEDIUM']}")
    return summary


# --------------------------------------------------------------------------------------
# Synthetic fallback
# --------------------------------------------------------------------------------------


def synthesise_cohort(
    *,
    config: Settings | None = None,
    n_stays: int = 900,
    progress: Callable[[str], None] | None = None,
) -> DatasetSummary:
    """Generate a stand-in cohort when the raw archive is unavailable.

    The generator reuses the live simulator, so the physiology is the same shape as
    the demo ward: each synthetic stay is assigned a trajectory, sampled hourly for
    48 h, and labelled from that trajectory. Useful for CI and for a first run on a
    machine that has not downloaded the 8 MB archive - but a model trained here has
    learned the simulator, not medicine, and the model card says so.
    """
    from icu_monitor.simulation.patient import PatientSimulator  # local: avoids cycle

    cfg = config or default_settings
    cfg.ensure_directories()
    say = progress or (lambda message: logger.info("%s", message))
    rng = np.random.default_rng(cfg.simulation_seed)

    state_to_label = {
        "stable": RiskLevel.LOW.value,
        "recovering": RiskLevel.LOW.value,
        "deteriorating": RiskLevel.MEDIUM.value,
        "critical": RiskLevel.HIGH.value,
    }
    states = ["stable", "recovering", "deteriorating", "critical"]
    weights = [0.45, 0.17, 0.26, 0.12]

    window_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []

    for index in range(n_stays):
        state = str(rng.choice(states, p=weights))
        label = state_to_label[state]
        simulator = PatientSimulator(
            seed=int(rng.integers(0, 2**31 - 1)),
            state=state,
            age=int(rng.integers(21, 92)),
        )
        stay = StayRecord(
            record_id=200_000 + index,
            static={
                "age": float(simulator.age),
                "sex_male": float(rng.integers(0, 2)),
                "icu_type": float(rng.integers(1, 5)),
                # `np.clip`, not `.clip()`: a scalar draw from `rng.normal` is a plain Python
                # float, and `float.clip` does not exist. This crashed `etl --synthetic` on its
                # first stay - the one command the README hands a reader who has not downloaded
                # the archive.
                "weight_kg": float(np.clip(rng.normal(78, 16), 42, 160)),
                "height_cm": float(np.clip(rng.normal(170, 10), 146, 198)),
            },
        )
        channels: dict[str, tuple[list[float], list[float]]] = {}
        for hour in range(48):
            observation = simulator.step(minutes=60)
            readings = {
                "heart_rate": observation.heart_rate,
                "spo2": observation.spo2,
                "bp_systolic": observation.bp_systolic,
                "bp_diastolic": observation.bp_diastolic,
                "resp_rate": observation.resp_rate,
                "temperature": observation.temperature,
                "gcs": observation.gcs,
            }
            for channel, value in readings.items():
                if value is None:
                    continue
                bucket = channels.setdefault(channel, ([], []))
                bucket[0].append(float(hour))
                bucket[1].append(float(value))
        stay.channels = channels

        rows = build_windows(stay, label, config=cfg)
        window_rows.extend(rows)
        cohort_rows.append(
            {
                "record_id": stay.record_id,
                "acuity": label,
                "died": int(label == RiskLevel.HIGH.value),
                "sofa": np.nan,
                "saps_i": np.nan,
                "length_of_stay": np.nan,
                "windows": len(rows),
                "coverage": stay.channel_coverage(),
                "duration_h": 48.0,
                **stay.static,
            }
        )
        if (index + 1) % 200 == 0:
            say(f"  synthesised {index + 1:,} stays")

    windows = pd.DataFrame(window_rows)
    cohort = pd.DataFrame(cohort_rows)
    paths = _write_table(windows, cfg)
    cohort.to_csv(cfg.cohort_path, index=False)
    paths["cohort"] = str(cfg.cohort_path)
    paths["summary"] = str(cfg.dataset_summary_path)

    summary = DatasetSummary(
        stays_seen=n_stays,
        stays_labelled=n_stays,
        stays_used=n_stays,
        windows=len(windows),
        class_counts=class_distribution(windows["acuity"]),
        features=len(FEATURE_NAMES),
        source="synthetic (simulator-generated; not real patient data)",
        paths=paths,
        synthetic=True,
    )
    _write_summary(summary, cfg)
    say(f"Wrote {summary.windows:,} synthetic windows → {cfg.features_csv_path.name}")
    say(f"Class balance: {summary.class_counts}")
    return summary


def load_windows(config: Settings | None = None) -> pd.DataFrame:
    """Read the processed window table, preferring Parquet."""
    cfg = config or default_settings
    if cfg.features_path.exists():
        try:
            return pd.read_parquet(cfg.features_path)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("Falling back to CSV (%s)", exc)
    if cfg.features_csv_path.exists():
        return pd.read_csv(cfg.features_csv_path)
    raise FileNotFoundError(
        f"No processed dataset found at {cfg.features_csv_path}. "
        "Run `python -m icu_monitor etl` (or `make bootstrap`) first."
    )


__all__ = [
    "EXTRA_BOUNDS",
    "REQUIRED_CHANNELS",
    "DatasetSummary",
    "StayRecord",
    "build_dataset",
    "build_windows",
    "iter_stays",
    "load_dataset_summary",
    "load_windows",
    "parse_record",
    "synthesise_cohort",
]
