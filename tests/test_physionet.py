"""The ETL: a directory of text files becomes a supervised table, or says why it cannot.

This is the one module that reads data nobody in this repository wrote, so most of these tests
are about *refusing* things. Four themes.

**A sentinel is not a measurement.** The challenge files use ``-1`` for "not recorded", and a
``-1`` read as a number is a heart rate of minus one - a value that would then be summarised,
scaled, and learned from. The same goes for an age of 999 and a height of 3 cm.

**Artefact is dropped, not clipped.** A monitor that reports 900 bpm because a lead came loose
has told you nothing, and clamping it to 250 would invent a tachycardia that was never
observed. The plausibility bounds are the ones the live dashboard uses, so a value the ETL
keeps is a value the ward would also accept.

**A window is the unit of prediction, and it is honest about its own emptiness.** Windows are
built by the *same* function the dashboard calls, which is what makes train/serve skew
impossible rather than merely unlikely; a window without enough channels is skipped instead of
being imputed into existence.

**The synthetic path is real, and says it is synthetic.** It exists so a first run works with
no 8 MB download, and its ``source`` string is what stops a model card from implying the
numbers came from patients. That string is written to disk beside the table, because the
consumer that needs it - training, two commands later - has no other way to know.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from icu_monitor.config import Settings
from icu_monitor.core.types import RiskLevel
from icu_monitor.data.physionet import (
    EXTRA_BOUNDS,
    REQUIRED_CHANNELS,
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
from icu_monitor.ml.features import FEATURE_NAMES

HEADER = "Time,Parameter,Value"


def record(*lines: str, header: bool = True) -> str:
    """A challenge record file. ``header`` off is the "someone stripped it" case."""
    body = "\n".join(lines)
    return f"{HEADER}\n{body}\n" if header else f"{body}\n"


def series(parameter: str, *values: float, start: int = 0, step: int = 1) -> list[str]:
    """``parameter`` sampled hourly from ``start``, one line per value."""
    return [f"{start + i * step:02d}:00,{parameter},{value}" for i, value in enumerate(values)]


def stay_text(record_id: int, *, hours: int = 12, offset: float = 0.0) -> str:
    """A stay with all three required channels, long enough to slice into several windows."""
    lines = [f"00:00,RecordID,{record_id}", "00:00,Age,64", "00:00,Gender,1", "00:00,ICUType,3"]
    for hour in range(hours):
        lines.append(f"{hour:02d}:00,HR,{78 + offset + hour % 5}")
        lines.append(f"{hour:02d}:00,SaO2,{97 - hour % 3}")
        lines.append(f"{hour:02d}:00,NISysABP,{118 + offset - hour % 7}")
    return record(*lines)


def outcomes_text(*rows: tuple[int, int, int, int]) -> str:
    """``(record_id, sofa, length_of_stay, died)`` in the Outcomes-a.txt layout."""
    lines = ["RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"]
    lines.extend(f"{rid},14,{sofa},{los},-1,{died}" for rid, sofa, los, died in rows)
    return "\n".join(lines) + "\n"


@pytest.fixture
def archive(config: Settings) -> Path:
    """An extracted record directory at the path the ETL expects the zip to be.

    ``iter_stays`` accepts either, and a directory is what a reader who unzipped the download
    by hand actually has - so this doubles as coverage of that branch.
    """
    config.ensure_directories()
    path = config.raw_records_zip
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------- parsing


def test_a_record_becomes_per_channel_arrays() -> None:
    stay = parse_record(record(*series("HR", 80, 82, 84)), record_id=7)
    assert stay.record_id == 7
    hours, values = stay.channels["heart_rate"]
    assert hours == [0.0, 1.0, 2.0]
    assert values == [80.0, 82.0, 84.0]


def test_the_parameter_map_is_what_names_a_channel() -> None:
    """``HR`` is not a channel name anywhere else in the project; ``heart_rate`` is."""
    stay = parse_record(record("00:00,HR,80", "00:00,Unmapped,5"))
    assert set(stay.channels) == {"heart_rate"}


def test_invasive_and_cuff_pressures_are_the_same_channel() -> None:
    """A stay switched from an arterial line to a cuff mid-admission is still one series.

    Keeping ``SysABP`` and ``NISysABP`` apart would split one patient's blood pressure into two
    half-empty channels, and a window needing systolic pressure would see neither.
    """
    stay = parse_record(record("00:00,SysABP,120", "01:00,NISysABP,118"))
    assert stay.channels["bp_systolic"] == ([0.0, 1.0], [120.0, 118.0])


def test_the_header_line_is_not_data() -> None:
    assert parse_record(record("00:00,HR,80")).channels["heart_rate"][1] == [80.0]
    assert parse_record(record("00:00,HR,80", header=False)).channels["heart_rate"][1] == [80.0]


def test_minutes_become_a_fraction_of_an_hour() -> None:
    stay = parse_record(record("01:30,HR,80", "02:45,HR,82"))
    assert stay.channels["heart_rate"][0] == [1.5, 2.75]


def test_samples_are_sorted_even_when_the_file_is_not() -> None:
    """Every downstream slope and slice assumes time runs forwards."""
    stay = parse_record(record("05:00,HR,90", "01:00,HR,70", "03:00,HR,80"))
    hours, values = stay.channels["heart_rate"]
    assert hours == [1.0, 3.0, 5.0]
    assert values == [70.0, 80.0, 90.0]


MISSING_TOKENS = ["-1", "NA", "?", ""]


@pytest.mark.parametrize("token", MISSING_TOKENS)
def test_a_missing_value_token_is_not_a_measurement(token: str) -> None:
    """``-1`` is the one that matters: read as a number it is a heart rate of minus one."""
    stay = parse_record(record(f"00:00,HR,{token}", "01:00,HR,80"))
    assert stay.channels["heart_rate"] == ([1.0], [80.0])


def test_a_value_that_is_not_a_number_is_skipped() -> None:
    stay = parse_record(record("00:00,HR,eighty", "01:00,HR,80"))
    assert stay.channels["heart_rate"][1] == [80.0]


def test_a_short_line_is_skipped() -> None:
    stay = parse_record(record("00:00,HR", "01:00,HR,80"))
    assert stay.channels["heart_rate"][1] == [80.0]


def test_an_unparseable_timestamp_drops_the_sample() -> None:
    stay = parse_record(record("noon,HR,80", "01:00,HR,82"))
    assert stay.channels["heart_rate"] == ([1.0], [82.0])


IMPLAUSIBLE = [
    ("HR", 900.0),  # a loose lead, not a tachycardia
    ("HR", 2.0),
    ("SaO2", 5.0),  # a probe off the finger reads near zero
    ("SysABP", 1000.0),
    ("Temp", 0.0),
    ("RespRate", 400.0),
    ("GCS", 30.0),  # the scale stops at 15
    ("FiO2", 21.0),  # a percentage where a fraction belongs
]


@pytest.mark.parametrize(("parameter", "value"), IMPLAUSIBLE)
def test_artefact_is_dropped_rather_than_clipped(parameter: str, value: float) -> None:
    """Clamping 900 bpm to 250 would invent a tachycardia the monitor never observed."""
    stay = parse_record(record(f"00:00,{parameter},{value}"))
    assert stay.channels == {}


def test_a_channel_with_no_declared_bounds_is_kept() -> None:
    """``mech_vent`` is a flag, not a vital, and it has no ``VitalSpec`` to check against."""
    stay = parse_record(record("00:00,MechVent,1"))
    assert stay.channels["mech_vent"][1] == [1.0]
    assert "mech_vent" in EXTRA_BOUNDS


# ------------------------------------------------------------------------- static fields


def test_the_record_id_in_the_file_wins() -> None:
    """The filename is a hint; the file's own ``RecordID`` is the record."""
    assert parse_record(record("00:00,RecordID,132539"), record_id=1).record_id == 132539


def test_a_record_with_no_id_anywhere_is_zero_not_an_error() -> None:
    assert parse_record(record("00:00,HR,80")).record_id == 0


def test_the_record_id_is_not_also_a_channel() -> None:
    stay = parse_record(record("00:00,RecordID,7", "00:00,HR,80"))
    assert set(stay.channels) == {"heart_rate"}


STATIC_CASES = [
    ("Age", 64, "age", 64.0),
    ("Gender", 1, "sex_male", 1.0),
    ("Gender", 0, "sex_male", 0.0),
    ("Height", 175, "height_cm", 175.0),
    ("Weight", 82, "weight_kg", 82.0),
    ("ICUType", 3, "icu_type", 3.0),
]


@pytest.mark.parametrize(("parameter", "value", "key", "expected"), STATIC_CASES)
def test_a_static_descriptor_lands_in_static(
    parameter: str, value: float, key: str, expected: float
) -> None:
    assert parse_record(record(f"00:00,{parameter},{value}")).static[key] == expected


IMPLAUSIBLE_STATIC = [
    ("Age", 999, "age"),
    # `0 < value` - a zero age is falsy *and* impossible, and the guard has to catch it on the
    # second count rather than the first.
    ("Age", 0, "age"),
    ("Height", 3, "height_cm"),
    ("Height", 900, "height_cm"),
]


@pytest.mark.parametrize(("parameter", "value", "key"), IMPLAUSIBLE_STATIC)
def test_an_implausible_descriptor_becomes_missing_not_zero(
    parameter: str, value: float, key: str
) -> None:
    """``NaN`` is imputed downstream; ``0`` would be *learned from* as a real measurement."""
    assert pd.isna(parse_record(record(f"00:00,{parameter},{value}")).static[key])


def test_the_first_plausible_weight_is_the_admission_weight() -> None:
    """Weight is re-recorded through a stay, and later entries drift with fluid balance."""
    stay = parse_record(record("00:00,Weight,82", "12:00,Weight,88"))
    assert stay.static["weight_kg"] == 82.0


def test_an_implausible_weight_does_not_take_the_slot() -> None:
    stay = parse_record(record("00:00,Weight,-1", "00:00,Weight,3", "12:00,Weight,82"))
    assert stay.static["weight_kg"] == 82.0


# ----------------------------------------------------------------------------- StayRecord


def test_duration_is_the_last_observation_of_any_channel() -> None:
    stay = parse_record(record("00:00,HR,80", "11:30,SaO2,96"))
    assert stay.duration_hours == 11.5


def test_an_empty_stay_has_no_duration_rather_than_raising() -> None:
    """Reached by any record whose every line was artefact - and ``max(())`` raises."""
    assert StayRecord(record_id=1).duration_hours == 0.0
    assert parse_record(record("00:00,HR,900")).duration_hours == 0.0


def test_coverage_counts_the_channels_a_window_needs() -> None:
    """Used to pick the replay stays, so it counts the required three and nothing else."""
    stay = parse_record(record("00:00,HR,80", "00:00,SaO2,96", "00:00,Urine,120"))
    assert stay.channel_coverage() == 2
    assert set(REQUIRED_CHANNELS) == {"heart_rate", "spo2", "bp_systolic"}


def test_a_slice_is_half_open() -> None:
    """``[start, end)``: shared endpoints would put one sample in two consecutive windows."""
    stay = parse_record(record(*series("HR", 70, 71, 72, 73, 74)))
    assert stay.slice_window(1.0, 3.0)["heart_rate"] == ([1.0, 2.0], [71.0, 72.0])


def test_a_slice_omits_channels_with_nothing_in_it() -> None:
    """An empty ``([], [])`` entry would count as a present channel in ``build_windows``."""
    stay = parse_record(record("00:00,HR,80", "09:00,SaO2,96"))
    assert set(stay.slice_window(0.0, 8.0)) == {"heart_rate"}
    assert stay.slice_window(20.0, 28.0) == {}


# --------------------------------------------------------------------------- build_windows


def test_a_window_carries_the_label_and_its_own_bounds(config: Settings) -> None:
    rows = build_windows(parse_record(stay_text(1, hours=8)), "LOW", config=config)
    assert rows
    first = rows[0]
    assert first["record_id"] == 1
    assert first["acuity"] == "LOW"
    assert (first["window_start_h"], first["window_end_h"]) == (0.0, float(config.window_hours))
    assert first["n_samples"] > 0


def test_a_window_is_summarised_by_the_dashboards_own_feature_builder(config: Settings) -> None:
    """No train/serve skew: the columns here are exactly the model's input vector.

    If this ever diverges, the model is scored on one feature layout and served another - the
    failure that produces a confident, meaningless prediction rather than an error.
    """
    row = build_windows(parse_record(stay_text(1)), "LOW", config=config)[0]
    assert set(FEATURE_NAMES) <= set(row)
    assert len(FEATURE_NAMES) == 51


def test_windows_advance_by_the_configured_stride(config: Settings) -> None:
    rows = build_windows(parse_record(stay_text(1, hours=24)), "LOW", config=config)
    starts = [row["window_start_h"] for row in rows]
    assert starts == sorted(starts)
    assert starts[1] - starts[0] == float(config.window_stride_hours)


def test_windows_overlap_so_a_short_stay_still_yields_several(config: Settings) -> None:
    """8 h windows on a 4 h stride: consecutive windows share half their samples."""
    stay = parse_record(stay_text(1, hours=16))
    rows = build_windows(stay, "LOW", config=config)
    assert len(rows) > 1
    assert rows[0]["window_end_h"] > rows[1]["window_start_h"]


def test_a_stay_shorter_than_one_window_still_produces_one(config: Settings) -> None:
    """Otherwise a patient who died in their fourth hour contributes nothing to training."""
    rows = build_windows(parse_record(stay_text(1, hours=3)), "HIGH", config=config)
    assert len(rows) == 1
    assert rows[0]["window_end_h"] == float(config.window_hours)


def test_a_window_without_enough_channels_is_skipped(config: Settings) -> None:
    """Skipped, not imputed: a row of medians is not an observation of a patient."""
    lines = [f"{hour:02d}:00,HR,{80 + hour}" for hour in range(12)]
    assert build_windows(parse_record(record(*lines)), "LOW", config=config) == []


def test_the_channel_minimum_is_adjustable(config: Settings) -> None:
    lines = [f"{hour:02d}:00,HR,{80 + hour}" for hour in range(12)]
    stay = parse_record(record(*lines))
    assert build_windows(stay, "LOW", config=config, min_channels=1)


def test_the_window_geometry_comes_from_configuration(config: Settings) -> None:
    stay = parse_record(stay_text(1, hours=24))
    tight = config.with_overrides(window_hours=4, window_stride_hours=2)
    assert len(build_windows(stay, "LOW", config=tight)) > len(
        build_windows(stay, "LOW", config=config)
    )


# ------------------------------------------------------------------------------ iter_stays


def test_stays_are_read_from_an_extracted_directory(archive: Path) -> None:
    """What a reader who unzipped the download by hand has on disk."""
    for record_id in (140501, 132539):
        (archive / f"{record_id}.txt").write_text(stay_text(record_id), encoding="utf-8")
    assert [stay.record_id for stay in iter_stays(archive)] == [132539, 140501]  # sorted by name


def test_stays_are_read_from_the_zip(config: Settings) -> None:
    config.ensure_directories()
    with zipfile.ZipFile(config.raw_records_zip, "w") as bundle:
        bundle.writestr("set-a/132539.txt", stay_text(132539))
        bundle.writestr("set-a/140501.txt", stay_text(140501))
        bundle.writestr("set-a/README", "not a record")
    assert [stay.record_id for stay in iter_stays(config.raw_records_zip)] == [132539, 140501]


def test_a_file_that_is_not_a_record_id_is_skipped(archive: Path) -> None:
    (archive / "132539.txt").write_text(stay_text(132539), encoding="utf-8")
    (archive / "notes.txt").write_text("a stray file in the download", encoding="utf-8")
    assert [stay.record_id for stay in iter_stays(archive)] == [132539]


def test_the_limit_stops_reading_early(archive: Path) -> None:
    """``--limit`` exists so a reader can check the archive parses without a full pass."""
    for record_id in range(1, 6):
        (archive / f"{record_id}.txt").write_text(stay_text(record_id), encoding="utf-8")
    assert len(list(iter_stays(archive, limit=2))) == 2


def test_a_missing_archive_says_where_to_get_it(config: Settings) -> None:
    """The most likely first failure for a new reader, so the message is the documentation."""
    with pytest.raises(FileNotFoundError) as failure:
        list(iter_stays(config.raw_records_zip))
    message = str(failure.value)
    assert "physionet.org" in message
    assert "--synthetic" in message


# ---------------------------------------------------------------------------- build_dataset


@pytest.fixture
def cohort(config: Settings, archive: Path) -> Settings:
    """Five stays: three usable and labelled, one unlabelled, one with a single channel.

    The three counters in the summary are only distinguishable on a cohort where they differ,
    and "seen but unlabelled" and "labelled but unusable" are both ordinary in the real archive.
    """
    config.raw_outcomes.write_text(
        outcomes_text((1, 2, 3, 0), (2, 12, 5, 0), (3, 2, 2, 1), (5, 2, 2, 0)), encoding="utf-8"
    )
    for record_id in (1, 2, 3, 4):
        (archive / f"{record_id}.txt").write_text(stay_text(record_id, hours=16), encoding="utf-8")
    (archive / "5.txt").write_text(record(*series("HR", 80, 81, 82)), encoding="utf-8")
    return config


def test_the_etl_produces_one_labelled_row_per_window(cohort: Settings) -> None:
    summary = build_dataset(config=cohort)
    frame = load_windows(cohort)
    assert isinstance(summary, DatasetSummary)
    assert summary.windows == len(frame)
    assert sum(summary.class_counts.values()) == summary.windows
    assert set(frame["record_id"]) == {1, 2, 3}


def test_the_summary_separates_seen_labelled_and_used(cohort: Settings) -> None:
    """Three numbers, three different ways a stay can drop out. Collapsing them hides a bug."""
    summary = build_dataset(config=cohort)
    assert summary.stays_seen == 5
    assert summary.stays_labelled == 4  # record 4 has no outcome row
    assert summary.stays_used == 3  # record 5 has one channel, so no window qualifies


def test_the_label_travels_from_the_outcome_to_every_window(cohort: Settings) -> None:
    build_dataset(config=cohort)
    frame = load_windows(cohort)
    by_record = frame.groupby("record_id")["acuity"].unique()
    assert by_record[1].tolist() == [RiskLevel.LOW.value]
    assert by_record[2].tolist() == [RiskLevel.MEDIUM.value]  # SOFA 12 clears the medium cut
    assert by_record[3].tolist() == [RiskLevel.HIGH.value]  # died in hospital


def test_the_etl_writes_the_tables_it_reports(cohort: Settings) -> None:
    summary = build_dataset(config=cohort)
    assert Path(summary.paths["csv"]) == cohort.features_csv_path
    assert Path(summary.paths["cohort"]) == cohort.cohort_path
    for path in summary.paths.values():
        assert Path(path).exists()


def test_the_cohort_table_is_one_row_per_stay(cohort: Settings) -> None:
    """The model card's population description is built from this, not from the window table."""
    summary = build_dataset(config=cohort)
    frame = pd.read_csv(cohort.cohort_path)
    assert len(frame) == summary.stays_used
    assert set(frame["record_id"]) == {1, 2, 3}
    assert {"acuity", "died", "sofa", "windows", "coverage", "age"} <= set(frame.columns)


def test_the_etl_writes_a_replay_series(cohort: Settings) -> None:
    """``ICU_VITALS_SOURCE=replay`` drives the dashboard from real recorded physiology."""
    summary = build_dataset(config=cohort)
    replay = pd.read_csv(summary.paths["replay"])
    assert set(replay.columns) == {"record_id", "acuity", "channel", "hours", "value"}
    assert replay["hours"].is_monotonic_increasing or len(set(replay["record_id"])) > 1
    assert set(replay["channel"]) <= {"heart_rate", "spo2", "bp_systolic", "gcs"}


def test_the_feature_count_is_the_models_input_width(cohort: Settings) -> None:
    assert build_dataset(config=cohort).features == len(FEATURE_NAMES)


def test_the_source_names_the_archive_it_read(cohort: Settings) -> None:
    assert "PhysioNet" in build_dataset(config=cohort).source


def test_an_etl_that_produced_nothing_says_so(config: Settings, archive: Path) -> None:
    """Silence here would leave `train` to fail later on an empty file, three steps away."""
    config.raw_outcomes.write_text(outcomes_text((1, 2, 3, 0)), encoding="utf-8")
    (archive / "1.txt").write_text(record(*series("HR", 80, 81)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no usable windows"):
        build_dataset(config=config)


def test_the_etl_reports_progress_to_the_caller(cohort: Settings) -> None:
    """``icu-monitor etl`` prints these; a silent twenty-minute ETL looks like a hang."""
    lines: list[str] = []
    build_dataset(config=cohort, progress=lines.append)
    assert any("outcomes" in line for line in lines)
    assert any("Class balance" in line for line in lines)


def test_the_limit_reaches_the_reader(cohort: Settings) -> None:
    assert build_dataset(config=cohort, limit=2).stays_seen == 2


# ------------------------------------------------------------------------ synthetic cohort


def test_the_synthetic_cohort_is_a_usable_dataset(config: Settings) -> None:
    """The path a reader with no download takes, so it has to produce a trainable table."""
    summary = synthesise_cohort(config=config, n_stays=12)
    frame = load_windows(config)
    assert summary.windows == len(frame) > 0
    assert summary.features == len(FEATURE_NAMES)
    assert set(FEATURE_NAMES) <= set(frame.columns)
    assert (summary.stays_seen, summary.stays_labelled, summary.stays_used) == (12, 12, 12)


def test_the_synthetic_cohort_says_it_is_not_real_patients(config: Settings) -> None:
    """This string ends up in the model card. It is the difference between a demo and a claim."""
    source = synthesise_cohort(config=config, n_stays=4).source
    assert "synthetic" in source
    assert "not real patient data" in source


def test_the_synthetic_cohort_covers_all_three_classes(config: Settings) -> None:
    """A two-class stand-in would train a model that can never predict MEDIUM."""
    counts = synthesise_cohort(config=config, n_stays=40).class_counts
    assert set(counts) == {"LOW", "MEDIUM", "HIGH"}
    assert all(count > 0 for count in counts.values())


def test_the_synthetic_cohort_is_reproducible(config: Settings) -> None:
    first = synthesise_cohort(config=config, n_stays=8)
    second = synthesise_cohort(config=config, n_stays=8)
    assert (first.windows, first.class_counts) == (second.windows, second.class_counts)


def test_a_different_seed_gives_a_different_cohort(config: Settings) -> None:
    other = config.with_overrides(simulation_seed=config.simulation_seed + 1)
    assert (
        synthesise_cohort(config=config, n_stays=24).class_counts
        != synthesise_cohort(config=other, n_stays=24).class_counts
    )


def test_the_synthetic_cohort_reports_progress(config: Settings) -> None:
    lines: list[str] = []
    synthesise_cohort(config=config, n_stays=200, progress=lines.append)
    assert any("synthesised" in line for line in lines)


# ----------------------------------------------------------------------------- load_windows


def test_loading_prefers_parquet_and_falls_back_to_csv(config: Settings) -> None:
    """Parquet keeps dtypes and is ~10x smaller; CSV is what always gets written."""
    synthesise_cohort(config=config, n_stays=4)
    assert config.features_csv_path.exists()
    from_disk = load_windows(config)
    assert not from_disk.empty

    if config.features_path.exists():
        config.features_path.unlink()
    assert len(load_windows(config)) == len(from_disk)


def test_loading_a_dataset_that_was_never_built_says_which_command_builds_it(
    config: Settings,
) -> None:
    with pytest.raises(FileNotFoundError, match="etl"):
        load_windows(config)


# ------------------------------------------------------------------------------- provenance
#
# The table and the story of the table travel together. Training runs in a separate process,
# possibly days later, and cannot see which ETL wrote what it is loading - so the ETL leaves a
# note. Everything in this section is about that note being written, being readable, and being
# absent honestly rather than replaced by a guess.


def test_the_real_etl_records_where_the_rows_came_from(cohort: Settings) -> None:
    summary = build_dataset(config=cohort)
    recorded = load_dataset_summary(cohort)
    assert recorded is not None
    assert recorded["source"] == summary.source
    assert recorded["synthetic"] is False
    assert recorded["windows"] == summary.windows


def test_the_synthetic_etl_records_that_the_rows_are_synthetic(config: Settings) -> None:
    """The flag, not the prose. A consumer deciding whether to caveat a score reads this."""
    synthesise_cohort(config=config, n_stays=6)
    recorded = load_dataset_summary(config)
    assert recorded is not None
    assert recorded["synthetic"] is True
    assert "not real patient data" in recorded["source"]


def test_the_summary_is_one_of_the_paths_the_etl_reports(config: Settings) -> None:
    """``etl`` prints every path in ``paths``, so being listed is how an operator learns of it."""
    summary = synthesise_cohort(config=config, n_stays=4)
    assert summary.paths["summary"] == str(config.dataset_summary_path)
    assert config.dataset_summary_path.exists()


def test_the_recorded_summary_carries_the_label_scheme_and_window_geometry(
    config: Settings,
) -> None:
    """A source string alone cannot answer "8-hour windows of what, labelled how"."""
    synthesise_cohort(config=config, n_stays=4)
    recorded = load_dataset_summary(config)
    assert recorded is not None
    assert recorded["window_hours"] == config.window_hours
    assert recorded["window_stride_hours"] == config.window_stride_hours
    assert recorded["labels"]["sofa_medium_threshold"] == config.label_sofa_medium


def test_the_recorded_summary_is_the_summary_that_was_returned(config: Settings) -> None:
    summary = synthesise_cohort(config=config, n_stays=6)
    recorded = load_dataset_summary(config)
    assert recorded is not None
    assert {key: recorded[key] for key in summary.as_dict()} == summary.as_dict()


def test_no_recorded_summary_is_an_answer_not_an_error(config: Settings) -> None:
    """A table built by an older version has no note, and inventing one would be the bug."""
    config.ensure_directories()
    assert load_dataset_summary(config) is None


def test_an_unreadable_summary_is_treated_as_no_summary(config: Settings) -> None:
    """Truncated JSON is a broken note, not a reason to refuse to train."""
    config.ensure_directories()
    config.dataset_summary_path.write_text('{"source": "half a fi', encoding="utf-8")
    assert load_dataset_summary(config) is None


def test_a_summary_that_is_not_an_object_is_treated_as_no_summary(config: Settings) -> None:
    config.ensure_directories()
    config.dataset_summary_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_dataset_summary(config) is None


def test_the_recorded_summary_is_readable_by_anything_that_reads_json(config: Settings) -> None:
    """It is a deployment artefact: a reader with no Python should be able to open it."""
    synthesise_cohort(config=config, n_stays=4)
    text = config.dataset_summary_path.read_text(encoding="utf-8")
    assert isinstance(json.loads(text), dict)
    assert len(text.splitlines()) > 1, "written with indent=2, so a human can read the diff"


def test_whichever_etl_ran_last_owns_the_note(cohort: Settings) -> None:
    """A stale note is worse than none: it describes the table before last, convincingly."""
    synthesise_cohort(config=cohort, n_stays=4)
    assert (load_dataset_summary(cohort) or {})["synthetic"] is True

    real = build_dataset(config=cohort)
    recorded = load_dataset_summary(cohort)
    assert recorded is not None
    assert recorded["synthetic"] is False
    assert recorded["source"] == real.source
