"""Centralised, environment-driven configuration.

Every tunable in the system resolves through :data:`settings`. Values may be
overridden with ``ICU_``-prefixed environment variables or a ``.env`` file, which
is what makes the same codebase deployable to a laptop, a container, or a
hosted Streamlit runtime without edits.
"""

from __future__ import annotations

import os
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------------------
# Path anchors
# --------------------------------------------------------------------------------------
# src/icu_monitor/config.py -> src/icu_monitor -> src -> <project root>
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


def _default_root() -> Path:
    """Choose a writable data root for source checkouts and installed wheels.

    A source checkout has ``pyproject.toml`` beside ``src/`` and should keep the familiar
    repository-relative layout. In a non-editable wheel, the package lives in site-packages,
    which is not the application's data directory; use the launch directory instead. An
    explicit ``ICU_PROJECT_ROOT`` remains the authoritative choice for containers and hosts.
    """
    override = os.environ.get("ICU_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT
    return Path.cwd().resolve()


FrameSourceName = Literal["auto", "synthetic", "camera", "video", "off"]
DetectorName = Literal["auto", "yolo", "heuristic", "off"]
VitalsSourceName = Literal["simulator", "replay"]


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="ICU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Identity ----------------------------------------------------------------------
    app_name: str = "ICU Sentinel"
    app_tagline: str = "Patient deterioration monitoring & early warning"
    environment: Literal["local", "docker", "cloud"] = "local"
    debug: bool = False

    # -- Filesystem layout -------------------------------------------------------------
    project_root: Path = Field(default_factory=_default_root)
    data_dir: Path | None = None
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    artifacts_dir: Path | None = None

    # -- Database ----------------------------------------------------------------------
    # Unset (``None``) means "use the default file-based SQLite database", filled in by
    # ``model_post_init`` below - so persistence is ON by default and survives a restart.
    # Set ``ICU_DATABASE_URL`` to a Postgres URL to swap the backend, or to an empty string
    # to run with no persistence at all (``build_repository`` treats "" as an opt-out).
    database_url: str | None = None
    db_retention_rows: int = Field(default=20_000, ge=100)

    # -- Ward / simulation -------------------------------------------------------------
    bed_count: int = Field(default=6, ge=1, le=24)
    vitals_source: VitalsSourceName = "simulator"
    simulation_seed: int = 20260905
    tick_seconds: float = Field(default=2.0, ge=0.25, le=30.0)
    history_window: int = Field(default=180, ge=20)

    # -- Vision ------------------------------------------------------------------------
    frame_source: FrameSourceName = "auto"
    detector: DetectorName = "auto"
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    video_path: Path | None = None
    yolo_weights: str = "yolov8n.pt"
    person_confidence: float = Field(default=0.45, ge=0.05, le=0.95)
    fall_aspect_ratio: float = Field(default=1.6, ge=1.0, le=4.0)
    fall_persistence_frames: int = Field(default=3, ge=1, le=30)

    # -- Clinical thresholds -----------------------------------------------------------
    news2_medium_threshold: int = Field(default=5, ge=1)
    news2_high_threshold: int = Field(default=7, ge=2)
    composite_medium_threshold: float = Field(default=40.0, ge=1, le=99)
    composite_high_threshold: float = Field(default=65.0, ge=2, le=99)
    composite_critical_threshold: float = Field(default=85.0, ge=3, le=100)

    # Fusion weights (normalised at use site, so they need not sum to 1).
    weight_ml: float = Field(default=0.45, ge=0.0)
    weight_news2: float = Field(default=0.40, ge=0.0)
    weight_vision: float = Field(default=0.15, ge=0.0)

    # -- Labelling (dataset construction) ----------------------------------------------
    label_sofa_medium: int = Field(default=9, ge=1)
    label_los_medium_days: int = Field(default=14, ge=1)
    window_hours: int = Field(default=8, ge=1)
    window_stride_hours: int = Field(default=4, ge=1)
    min_windows_per_record: int = Field(default=1, ge=1)

    # -- Alerting ----------------------------------------------------------------------
    alert_cooldown_seconds: float = Field(default=20.0, ge=0.0)
    alert_max_open: int = Field(default=200, ge=10)
    audible_alerts: bool = True

    # -- API ---------------------------------------------------------------------------
    # Keep local runs private. Containers and managed hosts override this explicitly.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://localhost:8000"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Unset means the API is open. That is deliberate for a local demo and unacceptable
    # anywhere else, so set ICU_API_KEY before exposing the service beyond localhost;
    # every /api/v1 route then requires a matching X-API-Key header.
    api_key: str | None = None
    api_warmup_ticks: int = Field(default=30, ge=0, le=600)

    # ---------------------------------------------------------------------------------
    # Derived paths
    # ---------------------------------------------------------------------------------
    @field_validator("project_root", mode="after")
    @classmethod
    def _resolve_root(cls, value: Path) -> Path:
        return value.resolve()

    def model_post_init(self, _context: object) -> None:
        root = self.project_root
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", root / "data")
        if self.raw_data_dir is None:
            object.__setattr__(self, "raw_data_dir", self.data_dir / "raw")
        if self.processed_data_dir is None:
            object.__setattr__(self, "processed_data_dir", self.data_dir / "processed")
        if self.artifacts_dir is None:
            object.__setattr__(self, "artifacts_dir", root / "artifacts")
        if self.database_url is None:
            db_path = (self.data_dir / "icu_monitor.db").as_posix()
            object.__setattr__(self, "database_url", f"sqlite:///{db_path}")

    @model_validator(mode="after")
    def _validate_ordered_thresholds(self) -> Settings:
        """Reject threshold combinations that would make the risk ladder contradictory."""
        ladders = (
            (
                ("news2_medium_threshold", self.news2_medium_threshold),
                ("news2_high_threshold", self.news2_high_threshold),
            ),
            (
                ("composite_medium_threshold", self.composite_medium_threshold),
                ("composite_high_threshold", self.composite_high_threshold),
                ("composite_critical_threshold", self.composite_critical_threshold),
            ),
        )
        for ladder in ladders:
            for (left_name, left), (right_name, right) in pairwise(ladder):
                if left >= right:
                    raise ValueError(f"{left_name} must be lower than {right_name}")
        if self.window_stride_hours > self.window_hours:
            raise ValueError("window_stride_hours must not exceed window_hours")
        if self.environment == "cloud" and not self.api_key:
            raise ValueError("api_key must be set when environment is cloud")
        return self

    # -- Convenience accessors ---------------------------------------------------------
    def with_overrides(self, **changes: object) -> Settings:
        """A *validated* copy with ``changes`` applied.

        ``model_copy(update=...)`` is the obvious way to do this and it is wrong twice.
        It skips validation, so ``--beds 500`` would build a five-hundred-bed ward and
        ``--beds 0`` an empty one instead of being refused by the ``ge=1, le=24`` bound
        that the field already declares. And it skips :meth:`model_post_init`, so moving
        ``project_root`` would leave every derived path pointing into the old tree.

        Re-constructing runs both. Only the fields that were actually *supplied* travel
        across - by keyword, by an ``ICU_`` variable, or by ``.env``, which is exactly what
        ``model_fields_set`` records. The derived directories are left out, so they are
        re-derived from whichever root the copy ends up with. An operator who set
        ``ICU_DATA_DIR`` keeps it, one who did not gets paths that follow the root, and
        because the copy's ``model_fields_set`` is again only what was supplied, that stays
        true for a copy of a copy.

        Unknown names raise instead of being ignored. ``extra="ignore"`` is right for a stray
        variable in someone's shell and wrong here: an override that silently does nothing is
        how a dashboard ends up displaying one number and scoring with another.
        """
        unknown = set(changes) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
        data = {name: getattr(self, name) for name in self.model_fields_set}
        data.update(changes)
        return type(self)(**data)

    @property
    def features_path(self) -> Path:
        """Window-level training table produced by the ETL."""
        return self.processed_data_dir / "vitals_windows.parquet"

    @property
    def features_csv_path(self) -> Path:
        """CSV mirror of the training table (portable fallback)."""
        return self.processed_data_dir / "vitals_windows.csv"

    @property
    def cohort_path(self) -> Path:
        """One row per ICU stay: labels and stay-level summary."""
        return self.processed_data_dir / "cohort.csv"

    @property
    def replay_path(self) -> Path:
        """Long-format vitals time series used by the replay data source."""
        return self.processed_data_dir / "replay_series.csv"

    @property
    def dataset_summary_path(self) -> Path:
        """Provenance of the processed table: which ETL wrote it, from what, with what balance.

        Training reads this to describe its own inputs. Without it the model card can only
        assert where the data came from, and an assertion is wrong exactly when it matters -
        after ``etl --synthetic``, where the numbers are the simulator's and not medicine's.
        """
        return self.processed_data_dir / "dataset_summary.json"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "risk_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def model_card_path(self) -> Path:
        return self.artifacts_dir / "model_card.json"

    @property
    def raw_records_zip(self) -> Path:
        return self.raw_data_dir / "set-a.zip"

    @property
    def raw_outcomes(self) -> Path:
        return self.raw_data_dir / "Outcomes-a.txt"

    @property
    def fusion_weights(self) -> dict[str, float]:
        total = self.weight_ml + self.weight_news2 + self.weight_vision
        if total <= 0:
            return {"ml": 1 / 3, "news2": 1 / 3, "vision": 1 / 3}
        return {
            "ml": self.weight_ml / total,
            "news2": self.weight_news2 / total,
            "vision": self.weight_vision / total,
        }

    def ensure_directories(self) -> None:
        """Create every writable directory the app expects. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


settings = get_settings()

__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "settings"]
