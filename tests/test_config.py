"""What the settings object promises to everything downstream.

Three themes.

**A derived path stays derived.** Every module asks ``Settings`` where the data lives, so if
``project_root`` moves and ``data_dir`` does not follow, the ETL writes into one tree and the
model reads from another. That has to hold through a copy, and through a copy of a copy.

**A bound a field declares is a bound the whole system relies on.** Nothing downstream
re-checks ``bed_count``; the ward simply builds that many beds. So every constrained field is
pushed one step past its own edge here, and :meth:`Settings.with_overrides` is pushed with
it - that method exists precisely because ``model_copy(update=...)`` skips validation.

**Configuration comes from the environment; these tests must not.** Anything that reads
``ICU_`` variables runs in a scrubbed environment inside ``tmp_path``, because ``env_file`` is
resolved against the working directory, and a suite whose result depends on the developer's
own ``.env`` is not a test of anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import icu_monitor.config as config_module
from icu_monitor.config import (
    PROJECT_ROOT,
    DetectorName,
    FrameSourceName,
    Settings,
    VitalsSourceName,
    get_settings,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No ``ICU_`` variables, and a working directory with no ``.env`` in it.

    The directory is returned so a test that wants a ``.env`` can write one there.
    """
    for name in [n for n in os.environ if n.startswith("ICU_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ------------------------------------------------------------------------- derived paths


def test_the_four_directories_are_derived_from_the_root(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.raw_data_dir == tmp_path / "data" / "raw"
    assert cfg.processed_data_dir == tmp_path / "data" / "processed"
    assert cfg.artifacts_dir == tmp_path / "artifacts"


def test_the_database_lands_in_the_data_directory(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path)
    assert cfg.database_url is not None
    assert cfg.database_url.startswith("sqlite:///")
    assert cfg.database_url.endswith("/data/icu_monitor.db")


def test_the_database_url_is_a_url_not_a_windows_path(tmp_path: Path) -> None:
    """``as_posix()`` is load-bearing: a backslash in a SQLAlchemy URL is an escape."""
    cfg = Settings(project_root=tmp_path)
    assert cfg.database_url is not None
    assert "\\" not in cfg.database_url


def test_cloud_environment_requires_an_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="api_key must be set"):
        Settings(project_root=tmp_path, environment="cloud")
    assert (
        Settings(project_root=tmp_path, environment="cloud", api_key="secret").api_key == "secret"
    )


def test_an_explicit_data_directory_is_kept_and_its_children_follow_it(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path, data_dir=tmp_path / "mounted")
    assert cfg.data_dir == tmp_path / "mounted"
    assert cfg.raw_data_dir == tmp_path / "mounted" / "raw"
    assert cfg.processed_data_dir == tmp_path / "mounted" / "processed"
    # artifacts_dir hangs off the root, not off data_dir, so it is unmoved.
    assert cfg.artifacts_dir == tmp_path / "artifacts"


def test_an_explicit_database_url_is_not_overwritten(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path, database_url="sqlite://")
    assert cfg.database_url == "sqlite://"


def test_the_root_is_resolved_to_an_absolute_path(clean_env: Path) -> None:
    cfg = Settings(project_root=Path("."))
    assert cfg.project_root.is_absolute()
    assert cfg.project_root == clean_env.resolve()


ARTEFACT_PATHS = [
    ("features_path", "processed_data_dir"),
    ("features_csv_path", "processed_data_dir"),
    ("cohort_path", "processed_data_dir"),
    ("replay_path", "processed_data_dir"),
    ("model_path", "artifacts_dir"),
    ("metrics_path", "artifacts_dir"),
    ("model_card_path", "artifacts_dir"),
    ("raw_records_zip", "raw_data_dir"),
    ("raw_outcomes", "raw_data_dir"),
]


@pytest.mark.parametrize(("artefact", "directory"), ARTEFACT_PATHS)
def test_every_artefact_sits_in_the_directory_it_belongs_to(
    tmp_path: Path, artefact: str, directory: str
) -> None:
    cfg = Settings(project_root=tmp_path)
    assert getattr(cfg, artefact).parent == getattr(cfg, directory)


def test_ensure_directories_creates_every_writable_directory(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path / "fresh")
    cfg.ensure_directories()
    for directory in (cfg.data_dir, cfg.raw_data_dir, cfg.processed_data_dir, cfg.artifacts_dir):
        assert directory is not None
        assert directory.is_dir()


def test_ensure_directories_is_safe_to_call_twice(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path / "fresh")
    cfg.ensure_directories()
    (cfg.processed_data_dir / "keep.csv").write_text("x", encoding="utf-8")
    cfg.ensure_directories()  # exist_ok, and nothing already written is disturbed
    assert (cfg.processed_data_dir / "keep.csv").read_text(encoding="utf-8") == "x"


# -------------------------------------------------------------------------- field bounds

# One step past each declared edge. These are not arbitrary: every one of them is a value
# some part of the system would otherwise accept and then misbehave on - a ward with no beds,
# a tick loop that spins, a confidence threshold that admits every blob as a person.
PAST_THE_EDGE = [
    ("bed_count", 0),
    ("bed_count", 25),
    ("tick_seconds", 0.2),
    ("tick_seconds", 30.5),
    ("history_window", 19),
    ("db_retention_rows", 99),
    ("person_confidence", 0.04),
    ("person_confidence", 0.96),
    ("fall_aspect_ratio", 0.9),
    ("fall_aspect_ratio", 4.1),
    ("fall_persistence_frames", 0),
    ("fall_persistence_frames", 31),
    ("news2_medium_threshold", 0),
    ("news2_high_threshold", 1),
    ("composite_medium_threshold", 0.0),
    ("composite_medium_threshold", 99.5),
    ("composite_high_threshold", 1.0),
    ("composite_critical_threshold", 2.0),
    ("composite_critical_threshold", 100.5),
    ("weight_ml", -0.1),
    ("weight_news2", -0.1),
    ("weight_vision", -0.1),
    ("alert_cooldown_seconds", -1.0),
    ("alert_max_open", 9),
    ("api_warmup_ticks", -1),
    ("api_warmup_ticks", 601),
    ("label_sofa_medium", 0),
    ("label_los_medium_days", 0),
    ("window_hours", 0),
    ("window_stride_hours", 0),
    ("min_windows_per_record", 0),
]

# The edges themselves, which have to be *inside*. A bound written `gt` where `ge` was meant
# is invisible until someone configures exactly the documented default.
AT_THE_EDGE = [
    ("bed_count", 1),
    ("bed_count", 24),
    ("tick_seconds", 0.25),
    ("tick_seconds", 30.0),
    ("history_window", 20),
    ("db_retention_rows", 100),
    ("person_confidence", 0.05),
    ("person_confidence", 0.95),
    ("fall_aspect_ratio", 1.0),
    ("fall_aspect_ratio", 4.0),
    ("fall_persistence_frames", 1),
    ("fall_persistence_frames", 30),
    ("news2_medium_threshold", 1),
    ("news2_high_threshold", 2),
    ("composite_medium_threshold", 1.0),
    ("composite_high_threshold", 2.0),
    ("composite_critical_threshold", 100.0),
    ("weight_ml", 0.0),
    ("alert_cooldown_seconds", 0.0),
    ("alert_max_open", 10),
    ("api_warmup_ticks", 0),
    ("api_warmup_ticks", 600),
]


@pytest.mark.parametrize(("field", "value"), PAST_THE_EDGE)
def test_a_value_past_a_field_bound_is_refused(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, **{field: value})


@pytest.mark.parametrize(("field", "value"), AT_THE_EDGE)
def test_the_declared_edge_is_inside_the_bound(tmp_path: Path, field: str, value: object) -> None:
    related = {
        "news2_high_threshold": {"news2_medium_threshold": 1},
        "composite_high_threshold": {"composite_medium_threshold": 1.0},
    }
    cfg = Settings(project_root=tmp_path, **related.get(field, {}), **{field: value})
    assert getattr(cfg, field) == value


# ------------------------------------------------------------------------ closed choices

UNKNOWN_TOKENS = [
    ("environment", "prod"),
    ("vitals_source", "stream"),
    # "none" is the exact spelling the Settings page used to offer for these two fields, while
    # the config has always called it "off". Nothing caught it: the page built its new settings
    # with `model_copy`, which does not validate, so the vision channel silently fell back to
    # its default instead of switching off. The page now reads its option lists off these same
    # Literal aliases - see the drift test below - and this row keeps the token itself illegal.
    ("frame_source", "none"),
    ("detector", "none"),
]


@pytest.mark.parametrize(("field", "token"), UNKNOWN_TOKENS)
def test_an_unknown_token_is_refused(tmp_path: Path, field: str, token: str) -> None:
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, **{field: token})


def test_the_dashboard_offers_exactly_the_tokens_the_schema_accepts() -> None:
    """A drift guard, and the reason it lives here rather than with the UI tests.

    The claim is about the schema: whatever the Settings page puts in front of an operator has
    to be a value ``Settings`` will accept. Deriving both from the same ``Literal`` is what
    makes that true by construction; this asserts the derivation is still in place.
    """
    pytest.importorskip("streamlit")
    from icu_monitor.ui.views import settings as page

    assert get_args(VitalsSourceName) == page.VITALS_SOURCES
    assert get_args(FrameSourceName) == page.FRAME_SOURCES
    assert get_args(DetectorName) == page.DETECTORS


def test_the_dashboard_reads_its_numeric_limits_off_the_fields() -> None:
    pytest.importorskip("streamlit")
    from icu_monitor.ui.views import settings as page

    assert page._bounds("bed_count", 999) == (1.0, 24.0)
    assert page._bounds("tick_seconds", 999) == (0.25, 30.0)
    # A field with a floor and no ceiling takes the caller's cap for its top end.
    assert page._bounds("history_window", 2000) == (20.0, 2000.0)


def test_a_misspelled_setting_name_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    """``extra="ignore"`` - a stray ``ICU_`` variable in someone's shell must not stop the app.

    The override path makes the opposite choice on purpose; see
    :func:`test_an_unknown_override_name_is_refused`.
    """
    cfg = Settings(project_root=tmp_path, bed_kount=9)  # type: ignore[call-arg]
    assert cfg.bed_count == Settings.model_fields["bed_count"].default


# ------------------------------------------------------------------------ fusion weights


def test_the_weights_are_the_three_channels_and_nothing_else(tmp_path: Path) -> None:
    assert set(Settings(project_root=tmp_path).fusion_weights) == {"ml", "news2", "vision"}


def test_the_weights_are_normalised_to_one(tmp_path: Path) -> None:
    """Configured relatively, used absolutely: 2/1/1 is the same fusion as 0.5/0.25/0.25."""
    cfg = Settings(project_root=tmp_path, weight_ml=2.0, weight_news2=1.0, weight_vision=1.0)
    weights = cfg.fusion_weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["ml"] == pytest.approx(0.5)
    assert weights["news2"] == pytest.approx(0.25)
    assert weights["vision"] == pytest.approx(0.25)


def test_normalisation_preserves_the_ratio(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path, weight_ml=0.9, weight_news2=0.3, weight_vision=0.3)
    weights = cfg.fusion_weights
    assert weights["ml"] / weights["news2"] == pytest.approx(3.0)


def test_a_single_channel_takes_the_whole_weight(tmp_path: Path) -> None:
    cfg = Settings(project_root=tmp_path, weight_ml=0.7, weight_news2=0.0, weight_vision=0.0)
    assert cfg.fusion_weights == pytest.approx({"ml": 1.0, "news2": 0.0, "vision": 0.0})


def test_all_zero_weights_fall_back_to_equal_thirds(tmp_path: Path) -> None:
    """Zeroing every weight is a configuration mistake, not an instruction to score nothing.

    Dividing by the total would raise; returning zeros would flatten every patient to 0 and
    call it reassurance. Equal thirds is the reading that still escalates.
    """
    cfg = Settings(project_root=tmp_path, weight_ml=0.0, weight_news2=0.0, weight_vision=0.0)
    weights = cfg.fusion_weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(share == pytest.approx(1 / 3) for share in weights.values())


# -------------------------------------------------------------------------- environment


def test_a_prefixed_variable_is_read(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ICU_BED_COUNT", "9")
    assert Settings().bed_count == 9


def test_a_variable_is_validated_like_a_keyword(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment is not a trusted channel. ``ICU_BED_COUNT=500`` is refused too."""
    monkeypatch.setenv("ICU_BED_COUNT", "500")
    with pytest.raises(ValidationError):
        Settings()


def test_the_prefix_is_required(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unprefixed ``BED_COUNT`` belongs to some other program."""
    monkeypatch.setenv("BED_COUNT", "9")
    assert Settings().bed_count == Settings.model_fields["bed_count"].default


def test_variable_names_are_case_insensitive(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("icu_bed_count", "7")
    assert Settings().bed_count == 7


def test_a_dotenv_file_is_read_from_the_working_directory(clean_env: Path) -> None:
    (clean_env / ".env").write_text("ICU_BED_COUNT=11\nICU_APP_NAME=Ward B\n", encoding="utf-8")
    cfg = Settings()
    assert cfg.bed_count == 11
    assert cfg.app_name == "Ward B"


def test_a_variable_beats_the_dotenv_file(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployment wins over the checked-in default, which is the point of having both."""
    (clean_env / ".env").write_text("ICU_BED_COUNT=11\n", encoding="utf-8")
    monkeypatch.setenv("ICU_BED_COUNT", "3")
    assert Settings().bed_count == 3


def test_the_default_root_follows_the_package_not_the_working_directory(clean_env: Path) -> None:
    """``PROJECT_ROOT`` is anchored on ``config.py``'s own location.

    ``clean_env`` has already moved the working directory elsewhere. Anchoring on the package
    is why ``icu-monitor etl`` finds the same ``data/`` whichever directory it is invoked from.
    """
    assert Settings().project_root == PROJECT_ROOT
    assert clean_env.resolve() != PROJECT_ROOT


def test_an_installed_wheel_uses_the_launch_directory_for_default_data(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An installed package must not write its default SQLite file into site-packages."""
    installed_package_root = tmp_path / "venv"
    installed_package_root.mkdir()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", installed_package_root)
    assert Settings().project_root == clean_env.resolve()


def test_icu_project_root_relocates_every_derived_path(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setting a container needs: one variable moves the whole tree onto the volume."""
    mounted = (clean_env / "mounted").resolve()
    monkeypatch.setenv("ICU_PROJECT_ROOT", str(mounted))
    cfg = Settings()
    assert cfg.project_root == mounted
    assert cfg.data_dir == mounted / "data"
    assert cfg.artifacts_dir == mounted / "artifacts"
    assert cfg.database_url is not None
    assert cfg.database_url.endswith("/mounted/data/icu_monitor.db")


def test_get_settings_hands_out_one_instance(clean_env: Path) -> None:
    assert get_settings() is get_settings()


def test_get_settings_rereads_the_environment_once_its_cache_is_cleared(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ICU_BED_COUNT", "13")
    get_settings.cache_clear()
    try:
        assert get_settings().bed_count == 13
    finally:
        # Leave no configured singleton behind for whatever test runs next.
        get_settings.cache_clear()


# ------------------------------------------------------------------------- with_overrides


def test_an_override_is_validated_where_model_copy_is_not(tmp_path: Path) -> None:
    """The whole reason the method exists, asserted from both sides.

    The second half pins ``model_copy``'s behaviour deliberately. It is not a bug in pydantic -
    ``update`` is documented as unvalidated - but it means the obvious one-liner would have let
    ``icu-monitor tick --beds 500`` build a five-hundred-bed ward, and the Settings page write a
    bed count the schema forbids. Should this assertion ever start failing, the guard below it
    is redundant and can go; until then it is the difference between a refusal and a wrong ward.
    """
    base = Settings(project_root=tmp_path)
    with pytest.raises(ValidationError):
        base.with_overrides(bed_count=500)
    assert base.model_copy(update={"bed_count": 500}).bed_count == 500


@pytest.mark.parametrize(("field", "value"), [("bed_count", 0), ("tick_seconds", 0.0)])
def test_an_override_is_bound_by_the_same_edges(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path).with_overrides(**{field: value})


def test_an_override_of_a_closed_choice_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path).with_overrides(frame_source="none")


def test_an_override_runs_the_post_init_hook(tmp_path: Path) -> None:
    """``model_copy`` skips it, which would move the root and leave the paths behind."""
    moved = Settings(project_root=tmp_path / "a").with_overrides(project_root=tmp_path / "b")
    assert moved.data_dir == tmp_path / "b" / "data"
    assert moved.artifacts_dir == tmp_path / "b" / "artifacts"
    assert moved.database_url is not None
    assert moved.database_url.endswith("/b/data/icu_monitor.db")


def test_a_copy_of_a_copy_still_follows_the_root(tmp_path: Path) -> None:
    """Carrying the whole dump across would pin the derived paths on the first hop.

    Only the fields that were actually supplied travel, so ``data_dir`` is still absent from
    the copy's ``model_fields_set`` and is re-derived on the next hop as well.
    """
    first = Settings(project_root=tmp_path / "a").with_overrides(project_root=tmp_path / "b")
    second = first.with_overrides(project_root=tmp_path / "c")
    assert "data_dir" not in first.model_fields_set
    assert second.data_dir == tmp_path / "c" / "data"


def test_an_explicitly_configured_directory_survives_a_root_move(tmp_path: Path) -> None:
    """An operator who set ``ICU_DATA_DIR`` meant it, and keeps it however the root moves."""
    pinned = Settings(project_root=tmp_path / "a", data_dir=tmp_path / "volume")
    moved = pinned.with_overrides(project_root=tmp_path / "b")
    twice = moved.with_overrides(project_root=tmp_path / "c")
    assert moved.data_dir == tmp_path / "volume"
    assert twice.data_dir == tmp_path / "volume"
    assert twice.processed_data_dir == tmp_path / "volume" / "processed"


def test_untouched_settings_are_carried_across(tmp_path: Path) -> None:
    base = Settings(project_root=tmp_path, simulation_seed=99, api_key="secret", bed_count=4)
    copy = base.with_overrides(bed_count=8)
    assert copy.bed_count == 8
    assert copy.simulation_seed == 99
    assert copy.api_key == base.api_key
    assert copy.project_root == tmp_path


def test_the_original_is_left_alone(tmp_path: Path) -> None:
    base = Settings(project_root=tmp_path, bed_count=4)
    assert base.with_overrides(bed_count=8).bed_count == 8
    assert base.bed_count == 4


def test_an_unknown_override_name_is_refused(tmp_path: Path) -> None:
    """The opposite of ``extra="ignore"``, and on purpose.

    A stray variable in a shell should not stop the app. A misspelled *override* is a caller
    bug, and one that silently does nothing is how a dashboard ends up displaying one number
    while scoring with another.
    """
    with pytest.raises(ValueError, match="bed_kount"):
        Settings(project_root=tmp_path).with_overrides(bed_kount=8)


def test_a_subclass_copies_to_its_own_type(tmp_path: Path) -> None:
    class WardSettings(Settings):
        ward_name: str = "ICU-A"

    copy = WardSettings(project_root=tmp_path, ward_name="ICU-B").with_overrides(bed_count=8)
    assert isinstance(copy, WardSettings)
    assert copy.ward_name == "ICU-B"
    assert copy.bed_count == 8
