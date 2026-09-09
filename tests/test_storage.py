"""Persistence: the layer that must never take the ward down.

Two contracts dominate this module. **Writes never raise into the caller** -
:meth:`Repository.record_tick` is what the engine holds as its ``Recorder``, and a monitor
that stops monitoring because a disk filled up is worse than one that stops recording. And
**retention is enforced on write**, because a demo left running over a weekend would
otherwise grow until the container died.

The rest is mapping. Everything above this layer speaks in domain objects, so the tests
here mostly push a ``Patient``/``Vitals``/``RiskAssessment``/``Alert`` through a real SQLite
database and read it back, including the ``NULL``s: missingness is clinical information and
the table must not launder it into a plausible zero.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest
from sqlalchemy import func, select, text

from icu_monitor.config import Settings
from icu_monitor.core.fusion import fuse_risk
from icu_monitor.core.news2 import calculate_news2
from icu_monitor.core.types import (
    Alert,
    AlertKind,
    ClinicalState,
    Consciousness,
    Patient,
    RiskAssessment,
    RiskLevel,
    Vitals,
    utcnow,
)
from icu_monitor.storage.database import (
    SCHEMA_VERSION,
    AlertRow,
    AssessmentRow,
    PatientRow,
    SchemaMeta,
    VitalsRow,
    build_engine,
    build_session_factory,
    create_all,
    migrate,
    session_scope,
)
from icu_monitor.storage.repository import TRIM_EVERY, Repository, build_repository

from .conftest import EPOCH, make_patient, make_vitals


def assess(config: Settings, patient: Patient, vitals: Vitals) -> RiskAssessment:
    return fuse_risk(
        patient_id=patient.patient_id,
        vitals=vitals,
        news2=calculate_news2(vitals, spo2_scale=patient.spo2_scale),
        config=config,
    )


def alarm(
    kind: AlertKind = AlertKind.HYPOXIA,
    *,
    patient_id: str = "P001",
    severity: RiskLevel = RiskLevel.HIGH,
    alert_id: int = 1,
    at=None,
    **overrides: object,
) -> Alert:
    values: dict[str, object] = {
        "patient_id": patient_id,
        "kind": kind,
        "severity": severity,
        "message": "SpO2 88 % - below the alerting threshold",
        "detail": "Sustained hypoxia; escalate per ward protocol.",
        "created_at": at or utcnow(),
        "alert_id": alert_id,
    }
    values.update(overrides)
    return Alert(**values)  # type: ignore[arg-type]


def record(
    repo: Repository,
    config: Settings,
    *,
    patient: Patient | None = None,
    at=EPOCH,
    alerts: tuple[Alert, ...] = (),
    **vitals_kwargs: object,
) -> tuple[Vitals, RiskAssessment]:
    """One bed's tick, persisted the way the engine persists it."""
    subject = patient or make_patient()
    observation = make_vitals(at=at, **vitals_kwargs)
    assessment = assess(config, subject, observation)
    repo.record_tick(subject, observation, assessment, alerts)
    return observation, assessment


def fill(repo: Repository, config: Settings, count: int) -> None:
    """``count`` ticks a minute apart, oldest first."""
    for index in range(count):
        record(repo, config, at=EPOCH + timedelta(minutes=index), heart_rate=70.0 + index % 20)


@pytest.fixture
def repo(config: Settings) -> Repository:
    instance = Repository(config=config)
    try:
        yield instance
    finally:
        instance.close()


# --------------------------------------------------------------------------- the plumbing


def test_an_in_memory_database_shares_one_connection(config: Settings) -> None:
    """``sqlite://`` with a per-session connection would give every session its own
    empty database, and every read would come back empty for reasons no log would explain."""
    from sqlalchemy.pool import StaticPool

    engine = build_engine(config)
    try:
        assert isinstance(engine.pool, StaticPool)
    finally:
        engine.dispose()


def test_sqlite_gets_foreign_keys_turned_on(config: Settings) -> None:
    """SQLite defaults them *off*, so an orphaned alert would insert happily."""
    engine = build_engine(config)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_a_file_database_runs_in_wal_mode(tmp_path) -> None:
    """Without WAL the dashboard's writer blocks the API's reader, which reads as a UI stall."""
    target = (tmp_path / "ward" / "icu.sqlite").as_posix()
    engine = build_engine(url=f"sqlite:///{target}")
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    finally:
        engine.dispose()


def test_a_file_database_creates_its_own_directory(tmp_path) -> None:
    """``ICU_DATABASE_URL=sqlite:///./data/icu.sqlite`` on a fresh clone has no ``data/``."""
    target = (tmp_path / "nested" / "deeper" / "icu.sqlite").as_posix()
    engine = build_engine(url=f"sqlite:///{target}")
    engine.dispose()
    assert (tmp_path / "nested" / "deeper").is_dir()


def test_creating_the_schema_twice_is_harmless(config: Settings) -> None:
    """Every start calls it, so it has to be idempotent rather than guarded by a flag."""
    engine = build_engine(config)
    try:
        create_all(engine)
        create_all(engine)
    finally:
        engine.dispose()


def test_a_failed_transaction_leaves_nothing_behind(config: Settings) -> None:
    engine = build_engine(config)
    create_all(engine)
    sessions = build_session_factory(engine)
    try:
        with pytest.raises(RuntimeError), session_scope(sessions) as session:
            session.add(PatientRow(patient_id="P001", bed="ICU-01"))
            raise RuntimeError("something went wrong mid-write")

        with session_scope(sessions) as session:
            assert session.scalar(select(func.count()).select_from(PatientRow)) == 0
    finally:
        engine.dispose()


def test_two_repositories_can_share_one_engine(config: Settings) -> None:
    """The dashboard and the API run in separate processes against the same file."""
    engine = build_engine(config)
    first, second = (
        Repository(config=config, engine=engine),
        Repository(config=config, engine=engine),
    )
    try:
        first.upsert_patient(make_patient("P001"))
        assert second.patient("P001") is not None
    finally:
        engine.dispose()


# ------------------------------------------------------------------------------- migration


def test_migration_backfills_legacy_alert_ids(config: Settings) -> None:
    """A row written before the row-PK-as-id fix has ``alert_id`` NULL; v1 adopts the PK.

    The per-process counter that older builds stored restarts at 1 in every process, so an
    acknowledgement could address more than one event. Copying the unique row PK over makes
    the historical rows consistent with what the repository writes now.
    """
    engine = build_engine(config)
    try:
        create_all(engine)
        sessions = build_session_factory(engine)
        with session_scope(sessions) as session:
            session.add(PatientRow(patient_id="P001", bed="ICU-01"))
            session.flush()
            session.add_all(
                [
                    AlertRow(
                        patient_id="P001",
                        kind="hypoxia",
                        severity="HIGH",
                        created_at=EPOCH,
                        alert_id=None,
                    ),
                    AlertRow(
                        patient_id="P001",
                        kind="tachycardia",
                        severity="HIGH",
                        created_at=EPOCH,
                        alert_id=None,
                    ),
                ]
            )

        assert migrate(engine) == SCHEMA_VERSION
        with session_scope(sessions) as session:
            rows = session.scalars(select(AlertRow).order_by(AlertRow.id)).all()
        assert len(rows) == 2
        assert all(row.alert_id == row.id for row in rows)
    finally:
        engine.dispose()


def test_migration_stamps_the_version_and_is_idempotent(config: Settings) -> None:
    """Every start migrates, so a second run has to be a no-op that still reports the version."""
    engine = build_engine(config)
    try:
        assert migrate(engine) == SCHEMA_VERSION
        sessions = build_session_factory(engine)
        with session_scope(sessions) as session:
            assert session.get(SchemaMeta, "version").value == str(SCHEMA_VERSION)
        assert migrate(engine) == SCHEMA_VERSION
    finally:
        engine.dispose()


def test_a_future_schema_is_left_untouched(config: Settings, caplog) -> None:
    """A database written by a newer build must be refused, loudly, not migrated backwards."""
    engine = build_engine(config)
    try:
        migrate(engine)
        sessions = build_session_factory(engine)
        with session_scope(sessions) as session:
            session.get(SchemaMeta, "version").value = "999"

        with caplog.at_level(logging.WARNING):
            assert migrate(engine) == 999
        assert "understands only" in caplog.text
    finally:
        engine.dispose()


# ------------------------------------------------------------------------------- patients


def test_a_patient_is_stored_and_read_back_whole(repo: Repository) -> None:
    repo.upsert_patient(
        make_patient(
            "P042",
            bed="ICU-42",
            age=81,
            sex="M",
            state=ClinicalState.DETERIORATING,
            spo2_scale=2,
            notes="COPD, target 88-92 %.",
        )
    )
    stored = repo.patient("P042")
    assert stored is not None
    assert (stored.bed, stored.age, stored.sex) == ("ICU-42", 81, "M")
    assert stored.state is ClinicalState.DETERIORATING
    assert stored.spo2_scale == 2
    assert "88-92" in stored.notes


def test_upserting_the_same_bed_updates_rather_than_duplicates(repo: Repository) -> None:
    repo.upsert_patient(make_patient("P001", state=ClinicalState.STABLE))
    repo.upsert_patient(make_patient("P001", state=ClinicalState.CRITICAL, notes="Escalated."))
    assert len(repo.patients()) == 1
    assert repo.patient("P001").state is ClinicalState.CRITICAL


def test_patients_are_listed_in_bed_order(repo: Repository) -> None:
    """The ward list is drawn in this order; sorting by insertion would shuffle the beds."""
    for patient_id, bed in (("P003", "ICU-03"), ("P001", "ICU-01"), ("P002", "ICU-02")):
        repo.upsert_patient(make_patient(patient_id, bed=bed))
    assert [patient.bed for patient in repo.patients()] == ["ICU-01", "ICU-02", "ICU-03"]


def test_syncing_reports_how_many_beds_it_wrote(repo: Repository) -> None:
    assert repo.sync_patients([make_patient("P001"), make_patient("P002")]) == 2
    assert len(repo.patients()) == 2


def test_an_unknown_patient_is_none(repo: Repository) -> None:
    assert repo.patient("P999") is None


# --------------------------------------------------------------------------- one tick


def test_a_tick_writes_the_observation_the_score_and_the_alerts(
    repo: Repository, config: Settings
) -> None:
    record(repo, config, alerts=(alarm(),))
    counts = repo.stats()
    assert counts["patients"] == 1
    assert counts["vitals"] == 1
    assert counts["assessments"] == 1
    assert counts["alerts"] == 1


def test_recording_creates_the_bed_it_was_never_told_about(
    repo: Repository, config: Settings
) -> None:
    """The engine may tick before anything called ``sync_patients``; the foreign key holds."""
    record(repo, config, patient=make_patient("P077", bed="ICU-77"))
    assert repo.patient("P077") is not None


def test_the_observation_survives_the_round_trip(repo: Repository, config: Settings) -> None:
    record(repo, config, heart_rate=132.0, spo2=88.0, temperature=39.1, gcs=13.0)
    stored = repo.recent_vitals("P001")[-1]
    assert stored.heart_rate == 132.0
    assert stored.spo2 == 88.0
    assert stored.temperature == pytest.approx(39.1)
    assert stored.gcs == 13.0
    assert stored.recorded_at == EPOCH


def test_a_channel_that_was_never_measured_stays_null(repo: Repository, config: Settings) -> None:
    """Missingness is clinical information. Writing 0 mmHg would be a fabricated reading."""
    record(repo, config, bp_systolic=None, bp_diastolic=None, temperature=None)
    stored = repo.recent_vitals("P001")[-1]
    assert stored.bp_systolic is None
    assert stored.bp_diastolic is None
    assert stored.temperature is None
    assert stored.heart_rate == 75.0


def test_how_many_channels_were_measured_is_stored(repo: Repository, config: Settings) -> None:
    """Derived on the domain object, but persisted, so a sensor-failure audit needs no replay."""
    record(repo, config, spo2=None, bp_systolic=None)
    with session_scope(build_session_factory(repo.engine)) as session:
        stored = session.scalars(select(VitalsRow)).one()
    assert stored.measured_channels == make_vitals(spo2=None, bp_systolic=None).measured_channels


def test_the_explanation_is_stored_with_the_score(repo: Repository, config: Settings) -> None:
    """A composite without its factors is a number nobody can defend six months later."""
    _vitals, assessment = record(repo, config, spo2=86.0, resp_rate=26.0, heart_rate=130.0)
    series = repo.score_series("P001")
    assert len(series) == 1
    at, score, level = series[0]
    assert at == EPOCH
    assert score == pytest.approx(assessment.composite_score)
    assert level == assessment.level.value

    with session_scope(build_session_factory(repo.engine)) as session:
        row = session.scalars(select(AssessmentRow)).one()
    assert len(row.factors) == len(assessment.factors)
    assert {"source", "description", "points", "severity"} == set(row.factors[0])
    assert row.news2_total == assessment.news2.total


def test_a_tick_with_no_model_records_that_fact(repo: Repository, config: Settings) -> None:
    """``model_available=False`` on the row is how the audit trail distinguishes "the model
    said LOW" from "there was no model"."""
    record(repo, config)
    with session_scope(build_session_factory(repo.engine)) as session:
        row = session.scalars(select(AssessmentRow)).one()
    assert row.model_available is False
    assert row.ml_level == RiskLevel.UNKNOWN.value
    assert row.ml_confidence is None


def test_timestamps_come_back_timezone_aware(repo: Repository, config: Settings) -> None:
    """SQLite stores no offset, so the mapper has to put UTC back on.

    Everything above this layer works in aware timestamps. A naive one escaping a read
    would raise ``can't subtract offset-naive and offset-aware datetimes`` in whichever
    caller first compared it with the present - a failure that surfaces nowhere near the
    row that caused it.
    """
    record(repo, config, alerts=(alarm(),))
    repo.acknowledge(repo.alerts()[0].alert_id)

    stored_vitals = repo.recent_vitals("P001")[-1]
    stored_alert = repo.alerts()[0]
    at, _score, _level = repo.score_series("P001")[0]

    assert stored_vitals.recorded_at == EPOCH
    assert at == EPOCH
    assert stored_alert.created_at.tzinfo is not None
    assert stored_alert.acknowledged_at.tzinfo is not None
    assert repo.patient("P001").admitted_at.tzinfo is not None
    # The comparison that would have raised.
    assert utcnow() - stored_alert.created_at >= timedelta(0)


def test_a_write_failure_is_counted_not_raised(repo: Repository, config: Settings) -> None:
    """The contract the engine depends on: persistence is best-effort, monitoring is not.

    Disposing the pool throws away the in-memory database, so the next write finds no
    tables - the closest thing to "the disk went away" that a test can stage.
    """
    repo.engine.dispose()
    record(repo, config)
    assert repo.write_errors == 1


# ------------------------------------------------------------------------ acknowledgement


def test_acknowledging_stamps_who_and_when(repo: Repository, config: Settings) -> None:
    record(repo, config, alerts=(alarm(),))
    stored_id = repo.alerts()[0].alert_id
    assert repo.acknowledge(stored_id, by="charge nurse") == 1

    stored = repo.alerts()[0]
    assert stored.acknowledged_at is not None
    assert stored.acknowledged_by == "charge nurse"


def test_acknowledging_twice_changes_nothing(repo: Repository, config: Settings) -> None:
    """The button is clickable twice and the ledger must record the first press only."""
    record(repo, config, alerts=(alarm(),))
    stored_id = repo.alerts()[0].alert_id
    assert repo.acknowledge(stored_id) == 1
    first = repo.alerts()[0].acknowledged_at
    assert repo.acknowledge(stored_id) == 0
    assert repo.alerts()[0].acknowledged_at == first


def test_acknowledging_something_unknown_touches_nothing(repo: Repository) -> None:
    assert repo.acknowledge(999) == 0


def test_acknowledging_all_clears_every_open_alert(repo: Repository, config: Settings) -> None:
    record(repo, config, alerts=(alarm(alert_id=1), alarm(AlertKind.TACHYCARDIA, alert_id=2)))
    assert repo.acknowledge_all() == 2
    assert repo.alerts(open_only=True) == []
    assert repo.stats()["open_alerts"] == 0


def test_acknowledging_one_bed_leaves_the_others_alone(repo: Repository, config: Settings) -> None:
    """ "Acknowledge all" on a patient page must not silence the rest of the ward."""
    record(repo, config, patient=make_patient("P001"), alerts=(alarm(alert_id=1),))
    record(
        repo,
        config,
        patient=make_patient("P002", bed="ICU-02"),
        alerts=(alarm(alert_id=2, patient_id="P002"),),
    )
    assert repo.acknowledge_all(patient_id="P001") == 1
    assert [a.patient_id for a in repo.alerts(open_only=True)] == ["P002"]


def test_acknowledging_all_with_nothing_open_is_zero(repo: Repository) -> None:
    assert repo.acknowledge_all() == 0


# ------------------------------------------------------------------------------- reading


def test_observations_come_back_oldest_first(repo: Repository, config: Settings) -> None:
    """The charts plot the list straight through, so newest-first would draw time backwards."""
    fill(repo, config, 5)
    stamps = [v.recorded_at for v in repo.recent_vitals("P001")]
    assert stamps == sorted(stamps)
    assert len(stamps) == 5


def test_the_newest_observations_are_the_ones_kept(repo: Repository, config: Settings) -> None:
    """``limit`` takes the newest rows and *then* reverses them - not the oldest three."""
    fill(repo, config, 10)
    kept = repo.recent_vitals("P001", limit=3)
    assert [v.recorded_at for v in kept] == [
        EPOCH + timedelta(minutes=7),
        EPOCH + timedelta(minutes=8),
        EPOCH + timedelta(minutes=9),
    ]


def test_a_zero_limit_still_returns_a_row(repo: Repository, config: Settings) -> None:
    """``LIMIT 0`` would silently blank the chart, so the floor is one."""
    fill(repo, config, 3)
    assert len(repo.recent_vitals("P001", limit=0)) == 1


def test_an_unknown_bed_reads_as_empty_not_as_an_error(repo: Repository) -> None:
    assert repo.recent_vitals("P999") == []
    assert repo.score_series("P999") == []
    assert repo.alerts(patient_id="P999") == []


def test_one_beds_history_excludes_the_others(repo: Repository, config: Settings) -> None:
    record(repo, config, patient=make_patient("P001"))
    record(repo, config, patient=make_patient("P002", bed="ICU-02"))
    record(
        repo, config, patient=make_patient("P002", bed="ICU-02"), at=EPOCH + timedelta(minutes=5)
    )
    assert len(repo.recent_vitals("P001")) == 1
    assert len(repo.recent_vitals("P002")) == 2


def test_the_score_series_is_oldest_first(repo: Repository, config: Settings) -> None:
    fill(repo, config, 4)
    stamps = [at for at, _score, _level in repo.score_series("P001")]
    assert stamps == sorted(stamps)


def test_alerts_come_back_newest_first(repo: Repository, config: Settings) -> None:
    """The ledger is read top-down, and the reader wants what just happened."""
    now = utcnow()
    record(
        repo,
        config,
        alerts=(
            alarm(alert_id=1, at=now - timedelta(minutes=10)),
            alarm(AlertKind.TACHYCARDIA, alert_id=2, at=now),
        ),
    )
    assert [a.alert_id for a in repo.alerts()] == [2, 1]


def test_open_only_hides_what_was_dealt_with(repo: Repository, config: Settings) -> None:
    record(repo, config, alerts=(alarm(alert_id=1), alarm(AlertKind.TACHYCARDIA, alert_id=2)))
    repo.acknowledge(1)
    assert [a.alert_id for a in repo.alerts(open_only=True)] == [2]
    assert len(repo.alerts()) == 2


def test_the_alert_survives_the_round_trip_whole(repo: Repository, config: Settings) -> None:
    record(repo, config, alerts=(alarm(AlertKind.BED_EXIT, severity=RiskLevel.MEDIUM),))
    stored = repo.alerts()[0]
    assert stored.kind is AlertKind.BED_EXIT
    assert stored.severity is RiskLevel.MEDIUM
    # The id is assigned by the database, not the caller: it is the row's primary key, which
    # is what makes it unique across the dashboard and API processes (issue: colliding ids).
    assert stored.alert_id is not None
    assert "hypoxia" in stored.detail.lower()


def test_alerts_are_counted_by_kind(repo: Repository, config: Settings) -> None:
    """The alert-mix chart on the analytics page is drawn from exactly this."""
    now = utcnow()
    record(
        repo,
        config,
        alerts=(
            alarm(alert_id=1, at=now),
            alarm(alert_id=2, at=now),
            alarm(AlertKind.TACHYCARDIA, alert_id=3, at=now),
        ),
    )
    assert repo.alert_counts_by_kind() == {"hypoxia": 2, "tachycardia": 1}


def test_the_count_window_excludes_older_alerts(repo: Repository, config: Settings) -> None:
    """ "Alerts in the last 24 h" has to mean the last 24 hours, not all time."""
    now = utcnow()
    record(
        repo,
        config,
        alerts=(alarm(alert_id=1, at=now), alarm(alert_id=2, at=now - timedelta(days=2))),
    )
    assert repo.alert_counts_by_kind(since_hours=24.0) == {"hypoxia": 1}
    assert repo.alert_counts_by_kind(since_hours=72.0) == {"hypoxia": 2}


def test_stats_counts_every_table(repo: Repository, config: Settings) -> None:
    fill(repo, config, 3)
    record(repo, config, at=EPOCH + timedelta(hours=1), alerts=(alarm(alert_id=1),))
    assert repo.stats() == {
        "patients": 1,
        "vitals": 4,
        "assessments": 4,
        "alerts": 1,
        "open_alerts": 1,
        "write_errors": 0,
    }


def test_health_is_a_real_round_trip(repo: Repository) -> None:
    """``/ready`` reports this, so it must query rather than check a flag."""
    assert repo.healthy() is True


def test_health_fails_when_the_database_is_gone(repo: Repository) -> None:
    repo.engine.dispose()
    assert repo.healthy() is False


# ----------------------------------------------------------------------------- retention


def test_nothing_is_trimmed_below_the_cap(repo: Repository, config: Settings) -> None:
    fill(repo, config, 20)
    assert repo.trim() == {"vitals": 0, "assessments": 0, "alerts": 0}
    assert repo.stats()["vitals"] == 20


def test_trimming_removes_the_oldest_rows_first(repo: Repository, config: Settings) -> None:
    """A capped table has to keep the recent past, which is the part anyone will look at."""
    fill(repo, config, 120)
    removed = repo.trim(keep=5)
    assert removed == {"vitals": 20, "assessments": 20, "alerts": 0}

    survivors = repo.recent_vitals("P001", limit=200)
    assert len(survivors) == 100
    assert survivors[0].recorded_at == EPOCH + timedelta(minutes=20)


def test_the_retention_floor_beats_a_silly_configuration(
    repo: Repository, config: Settings
) -> None:
    """``ICU_DB_RETENTION_ROWS=1`` must not leave the trend charts with a single point.

    The floor is 100 rows per table regardless of configuration, which is roughly the
    shortest history the dashboard can still draw something meaningful from.
    """
    fill(repo, config, 130)
    repo.trim(keep=1)
    assert repo.stats()["vitals"] == 100


def test_an_open_alert_is_never_trimmed(repo: Repository, config: Settings) -> None:
    """The ledger is an audit trail: retention deletes acknowledged alerts, never an open one.

    A demo running all weekend must not lose a critical event that nobody has attended to
    just because the vitals behind it aged past the cap.
    """
    for index in range(130):
        record(repo, config, at=EPOCH + timedelta(seconds=index), alerts=(alarm(),))
    repo.acknowledge_all()  # the first 130 are now acknowledged
    record(
        repo,
        config,
        at=EPOCH + timedelta(seconds=200),
        alerts=(alarm(AlertKind.TACHYCARDIA),),
    )

    removed = repo.trim(keep=1)
    # 131 rows, floor 100: 31 must go, and every doomed row is one that was acknowledged.
    assert removed["alerts"] == 31

    open_alerts = repo.alerts(open_only=True)
    assert [a.kind for a in open_alerts] == [AlertKind.TACHYCARDIA]
    stats = repo.stats()
    assert stats["alerts"] == 100
    assert stats["open_alerts"] == 1


def test_an_all_open_ledger_is_left_intact(repo: Repository, config: Settings) -> None:
    """With nothing acknowledged there is nothing eligible to delete, cap or no cap."""
    for index in range(130):
        record(repo, config, at=EPOCH + timedelta(seconds=index), alerts=(alarm(),))

    removed = repo.trim(keep=1)
    assert removed["alerts"] == 0
    assert repo.stats()["alerts"] == 130
    assert repo.stats()["open_alerts"] == 130


def test_retention_runs_itself_as_the_ward_ticks(config: Settings) -> None:
    """Left running all weekend, the ward must cap itself without anyone calling ``trim``.

    The sweep fires every ``TRIM_EVERY`` writes rather than on each one: counting is cheap
    and a DELETE per tick is not.
    """
    tight = config.with_overrides(db_retention_rows=100)
    repo = Repository(config=tight)
    try:
        fill(repo, tight, TRIM_EVERY)
        assert repo.stats()["vitals"] == 100
        assert repo.stats()["assessments"] == 100
    finally:
        repo.close()


def test_a_failed_sweep_does_not_propagate(repo: Repository, config: Settings) -> None:
    """Retention is housekeeping. It must not be the thing that kills a running ward."""
    fill(repo, config, 5)
    repo.engine.dispose()
    assert repo.trim(keep=1) == {"vitals": 0, "assessments": 0, "alerts": 0}


def test_purging_empties_everything(repo: Repository, config: Settings) -> None:
    fill(repo, config, 4)
    record(repo, config, at=EPOCH + timedelta(hours=2), alerts=(alarm(alert_id=1),))
    repo.purge()
    assert repo.stats() == {
        "patients": 0,
        "vitals": 0,
        "assessments": 0,
        "alerts": 0,
        "open_alerts": 0,
        "write_errors": 0,
    }


# ------------------------------------------------------------------------- reading back


def test_a_state_no_longer_in_the_code_degrades_to_stable(repo: Repository) -> None:
    """The one boundary where coercion is right.

    A control surface must refuse an unknown state - reporting success for a change it did
    not make is worse than an error. But a *stored* row was written by some earlier version
    of this app, and the ward still has to open, so reading falls back to the benign default.
    """
    repo.upsert_patient(make_patient("P001"))
    with session_scope(build_session_factory(repo.engine)) as session:
        session.get(PatientRow, "P001").state = "convalescing"

    assert repo.patient("P001").state is ClinicalState.STABLE


def test_a_row_with_holes_reads_as_a_usable_patient(repo: Repository) -> None:
    """Sparse rows come from a partial import, and the ward list must still render."""
    with session_scope(build_session_factory(repo.engine)) as session:
        session.add(PatientRow(patient_id="P001", bed="ICU-01"))

    stored = repo.patient("P001")
    assert stored is not None
    assert stored.display_name == ""
    assert stored.age == 0
    assert stored.sex == "U"
    assert stored.spo2_scale == 1
    assert stored.admitted_at is not None


def test_a_null_consciousness_reads_as_alert(repo: Repository, config: Settings) -> None:
    """``Vitals.consciousness`` is not optional, so the column's ``NULL`` maps to its default."""
    record(repo, config)
    with session_scope(build_session_factory(repo.engine)) as session:
        session.scalars(select(VitalsRow)).one().consciousness = None

    assert repo.recent_vitals("P001")[-1].consciousness is Consciousness.ALERT


def test_a_severity_the_code_no_longer_knows_reads_as_unknown(
    repo: Repository, config: Settings
) -> None:
    record(repo, config, alerts=(alarm(alert_id=1),))
    with session_scope(build_session_factory(repo.engine)) as session:
        session.scalars(select(AlertRow)).one().severity = "CATASTROPHIC"

    assert repo.alerts()[0].severity is RiskLevel.UNKNOWN


def test_a_legacy_severity_spelling_still_maps(repo: Repository, config: Settings) -> None:
    """``coerce`` knows the old vocabulary, so an early row is not degraded to "no data"."""
    record(repo, config, alerts=(alarm(alert_id=1),))
    with session_scope(build_session_factory(repo.engine)) as session:
        session.scalars(select(AlertRow)).one().severity = "SEVERE"

    assert repo.alerts()[0].severity is RiskLevel.HIGH


# --------------------------------------------------------------------------- degradation


def test_a_broken_database_url_degrades_to_no_persistence(tmp_path) -> None:
    """The whole point of ``build_repository`` returning ``None``: a bad ``ICU_DATABASE_URL``
    costs you the audit trail, not the monitor."""
    broken = Settings(project_root=tmp_path, database_url="not-a-url")
    assert build_repository(broken) is None


def test_a_working_url_builds_a_repository(config: Settings) -> None:
    instance = build_repository(config)
    assert isinstance(instance, Repository)
    instance.close()


def test_closing_releases_the_engine(config: Settings) -> None:
    instance = Repository(config=config)
    instance.close()
    assert instance.healthy() is False


# ---------------------------------------------------------------- ids across processes


def test_alert_ids_do_not_collide_across_processes(config: Settings, tmp_path) -> None:
    """Dashboard and API run as separate processes against one file.

    Each ``AlertManager`` mints its own ids starting at 1, so without a shared authority an
    acknowledgement of "alert 1" from one process would silence a different "alert 1"
    persisted by the other. The database's autoincrement PK is that authority: it never
    repeats, and ``record_tick`` writes it back onto the in-memory alert.
    """
    url = f"sqlite:///{(tmp_path / 'ward.db').as_posix()}"
    shared = config.with_overrides(database_url=url)

    first = Repository(config=shared)
    second = Repository(config=shared)
    try:
        dashboard_alert = alarm(patient_id="P001", alert_id=1)
        api_alert = alarm(AlertKind.TACHYCARDIA, patient_id="P001", alert_id=1)
        record(first, shared, patient=make_patient("P001"), at=EPOCH, alerts=(dashboard_alert,))
        record(
            second,
            shared,
            patient=make_patient("P001"),
            at=EPOCH + timedelta(seconds=1),
            alerts=(api_alert,),
        )

        # Both callers were handed the manager's id 1; the ledger gave them distinct PKs.
        assert dashboard_alert.alert_id == 1
        assert api_alert.alert_id == 2
        # Either process reads the same two, distinct ids off the shared file.
        assert sorted(a.alert_id for a in first.alerts()) == [1, 2]
        assert sorted(a.alert_id for a in second.alerts()) == [1, 2]
    finally:
        first.close()
        second.close()


# ------------------------------------------------------------------ persistence opt-out


def test_an_empty_database_url_opts_out_of_persistence(tmp_path) -> None:
    """An empty ``ICU_DATABASE_URL`` is a deliberate "run in memory" choice, not a failure."""
    settings = Settings(project_root=tmp_path, database_url="")
    assert build_repository(settings) is None


def test_a_whitespace_only_url_opts_out_too(tmp_path) -> None:
    """A value that is all spaces is empty in intent, so it takes the same opt-out path."""
    settings = Settings(project_root=tmp_path, database_url="   ")
    assert build_repository(settings) is None


def test_the_default_url_builds_a_file_backed_repository(tmp_path) -> None:
    """Unset means a SQLite file under ``data/`` - persistence is on and survives a restart."""
    settings = Settings(project_root=tmp_path, database_url=None)
    expected = f"sqlite:///{(tmp_path / 'data' / 'icu_monitor.db').as_posix()}"
    assert settings.database_url == expected

    instance = build_repository(settings)
    assert isinstance(instance, Repository)
    try:
        assert (tmp_path / "data" / "icu_monitor.db").exists()
    finally:
        instance.close()
