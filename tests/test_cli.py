"""What the command line promises an operator.

The CLI is the only interface with no browser in front of it, so it is also the one that has
to be honest without help. Three themes.

**A refusal names the flag the operator typed.** ``--beds 500`` is a typo, and the useful
answer is the field's own bound printed against ``--beds`` - not a traceback, not a
five-hundred-bed ward, and not ``--bed-count``, an option this command does not have. ``0`` is
part of that: it is a value to refuse, not an absence, which is why the guard tests ``is not
None`` rather than truthiness.

**A handler returns an exit code, and the code is the contract.** 0 for success, 2 for "you
asked for something impossible", 130 for an interruption. A shell script wiring ``etl`` to
``train`` reads those numbers and nothing else.

**Nothing here starts the real ward.** ``get_settings`` inside the CLI is redirected to the
``tmp_path`` configuration, and the two long-running handlers - ``etl`` and ``train`` - are
tested against their own return values rather than by running an ETL. Commands are dispatched
through their handler rather than through :func:`~icu_monitor.cli.main`, which reconfigures
process-wide logging; ``main`` has its own tests for exactly that.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from icu_monitor import __version__, cli
from icu_monitor.config import Settings
from icu_monitor.data import DatasetSummary
from icu_monitor.ml.train import CandidateResult, EvaluationReport, TrainingResult

SUBCOMMANDS = ("etl", "train", "serve", "dashboard", "tick", "info")


def run(*argv: str) -> int:
    """Parse ``argv`` and dispatch it, the way :func:`cli.main` does minus the logging setup."""
    args = cli.build_parser().parse_args(list(argv))
    return int(args.handler(args) or 0)


@pytest.fixture
def cli_config(config: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the CLI's ``get_settings`` at the isolated ward the ``config`` fixture built.

    ``cli.py`` imports the function by name at module scope, so patching it there covers every
    handler at once - and keeps ``tick`` and ``info`` off the developer's real ``data/``.
    """
    monkeypatch.setattr(cli, "get_settings", lambda: config)
    return config


def report(**overrides: Any) -> EvaluationReport:
    """A held-out report with the shape ``cmd_train`` prints, values chosen to be readable."""
    fields: dict[str, Any] = {
        "n_samples": 800,
        "n_patients": 200,
        "accuracy": 0.612,
        "balanced_accuracy": 0.548,
        "macro_f1": 0.531,
        "weighted_f1": 0.604,
        "cohen_kappa": 0.287,
        "log_loss": 0.842,
        "per_class": {
            "low": {"precision": 0.71, "recall": 0.83, "f1": 0.77, "support": 480},
            "medium": {"precision": 0.44, "recall": 0.31, "f1": 0.36, "support": 250},
            "high": {"precision": 0.52, "recall": 0.41, "f1": 0.46, "support": 70},
        },
        "confusion": [[400, 70, 10], [140, 78, 32], [26, 15, 29]],
        "roc_auc": {"macro": 0.708, "low": 0.75, "medium": 0.66, "high": 0.71},
        "average_precision": {"macro": 0.52},
        "calibration": {},
    }
    fields.update(overrides)
    return EvaluationReport(**fields)


def training_result(**overrides: Any) -> TrainingResult:
    fields: dict[str, Any] = {
        "winner": "hist_gradient_boosting",
        "version": "20260905-1203-hist_gradient_boosting",
        "report": report(),
        "candidates": [
            CandidateResult("random_forest", 0.498, 0.021, 0.511, 5),
            CandidateResult("hist_gradient_boosting", 0.527, 0.018, 0.544, 5),
        ],
        "dataset": {"windows": 4000},
    }
    fields.update(overrides)
    return TrainingResult(**fields)


def summary(**overrides: Any) -> DatasetSummary:
    fields: dict[str, Any] = {
        "stays_seen": 4000,
        "stays_labelled": 3900,
        "stays_used": 3810,
        "windows": 12000,
        "class_counts": {"low": 7000, "medium": 4000, "high": 1000},
        "features": 51,
        "source": "PhysioNet Challenge 2012 set-a",
        "paths": {"features": "data/processed/vitals_windows.parquet"},
    }
    fields.update(overrides)
    return DatasetSummary(**fields)


# ------------------------------------------------------------------------------- the parser


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_documented_subcommand_parses(name: str) -> None:
    assert callable(cli.build_parser().parse_args([name]).handler)


def test_each_subcommand_binds_its_own_handler() -> None:
    parser = cli.build_parser()
    handlers = {
        "etl": cli.cmd_etl,
        "train": cli.cmd_train,
        "serve": cli.cmd_serve,
        "dashboard": cli.cmd_dashboard,
        "tick": cli.cmd_tick,
        "info": cli.cmd_info,
    }
    assert set(handlers) == set(SUBCOMMANDS)
    for name, handler in handlers.items():
        assert parser.parse_args([name]).handler is handler


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args([])
    assert exit_info.value.code == 2


def test_an_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["deploy"])
    assert exit_info.value.code == 2


def test_the_version_flag_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


DEFAULTS = [
    ("tick", "ticks", 12),
    # `None`, not 0: `cmd_tick` distinguishes "no --beds" from "--beds 0", and a falsy default
    # would make the second look like the first.
    ("tick", "beds", None),
    ("tick", "json", False),
    ("train", "splits", 5),
    ("train", "test_fraction", 0.2),
    ("train", "no_importances", False),
    ("etl", "stays", 900),
    ("etl", "synthetic", False),
    ("etl", "limit", None),
    ("dashboard", "port", 8501),
    ("dashboard", "browser", False),
    # Unset, so `cmd_serve` can fall back to ICU_API_HOST / ICU_API_PORT.
    ("serve", "host", None),
    ("serve", "port", None),
    ("serve", "reload", False),
]


@pytest.mark.parametrize(("command", "option", "expected"), DEFAULTS)
def test_the_documented_default_is_the_parser_default(
    command: str, option: str, expected: object
) -> None:
    assert getattr(cli.build_parser().parse_args([command]), option) == expected


# --------------------------------------------------------------------------------- main()


def test_main_returns_the_handlers_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: None)
    monkeypatch.setattr(cli, "cmd_info", lambda _args: 3)
    assert cli.main(["info"]) == 3


def test_main_reads_a_handler_that_returns_nothing_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: None)
    monkeypatch.setattr(cli, "cmd_info", lambda _args: None)
    assert cli.main(["info"]) == 0


def test_main_reports_an_interruption_as_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C is how the ward is stopped, so it is an outcome rather than a crash."""

    def interrupted(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "configure_logging", lambda level=None: None)
    monkeypatch.setattr(cli, "cmd_tick", interrupted)
    assert cli.main(["tick"]) == 130
    assert "Interrupted" in capsys.readouterr().out


def test_the_verbose_flag_lowers_the_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``-v`` the level is left unset, so ``configure_logging`` keeps its own default."""
    seen: list[int | None] = []
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: seen.append(level))
    monkeypatch.setattr(cli, "cmd_info", lambda _args: 0)
    cli.main(["-v", "info"])
    cli.main(["info"])
    assert seen == [logging.DEBUG, None]


# ----------------------------------------------------------------------------------- tick

BAD_BED_COUNTS = [0, -3, 25, 500]


@pytest.mark.parametrize("beds", BAD_BED_COUNTS)
def test_tick_refuses_a_bed_count_the_field_forbids(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], beds: int
) -> None:
    """``0`` is the interesting one: it is falsy, and it used to be read as "not supplied"."""
    assert run("tick", "--beds", str(beds)) == 2
    out = capsys.readouterr().out
    assert "Invalid option:" in out
    assert "--beds" in out


def test_a_refusal_names_the_flag_not_the_field(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bound belongs to ``bed_count``; the operator typed ``--beds``."""
    run("tick", "--beds", "500")
    out = capsys.readouterr().out
    assert "--bed-count" not in out
    assert "less than or equal to 24" in out


def test_tick_builds_the_bed_count_it_was_given(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("tick", "--beds", "2", "--ticks", "1") == 0
    assert capsys.readouterr().out.count("BED-") == 2


def test_tick_prints_the_ward_at_the_requested_tick(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("tick", "--ticks", "3") == 0
    out = capsys.readouterr().out
    assert "Ward at tick 3" in out
    assert out.count("BED-") == cli_config.bed_count


class EngineSpy:
    """Wraps the engine ``cmd_tick`` builds, so a test can see the ward the printout came from.

    ``cmd_tick`` builds its own engine, which leaves two things otherwise unobservable: the
    snapshot it rendered, and whether it closed the engine afterwards. Recording both here is
    what lets the ordering test compare the printed rows against real scores instead of
    re-parsing them out of a padded, human-facing table.
    """

    def __init__(self) -> None:
        self.snapshots: list[Any] = []
        self.closed = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> EngineSpy:
        from icu_monitor import monitoring

        build = monitoring.build_engine

        def spy(**kwargs: Any) -> Any:
            engine = build(**kwargs)
            run_, close_ = engine.run, engine.close

            def run(*args: Any, **kw: Any) -> Any:
                snapshot = run_(*args, **kw)
                self.snapshots.append(snapshot)
                return snapshot

            def close() -> None:
                self.closed += 1
                close_()

            engine.run = run
            engine.close = close
            return engine

        monkeypatch.setattr(monitoring, "build_engine", spy)
        return self

    @property
    def snapshot(self) -> Any:
        assert len(self.snapshots) == 1, f"expected one run, saw {len(self.snapshots)}"
        return self.snapshots[0]


@pytest.fixture
def engine_spy(monkeypatch: pytest.MonkeyPatch) -> EngineSpy:
    return EngineSpy().install(monkeypatch)


def test_tick_orders_the_ward_by_score_worst_first(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], engine_spy: EngineSpy
) -> None:
    """The point of the printout: the bed that needs attention is the first row on screen."""
    assert run("tick", "--ticks", "4") == 0
    printed = [line.split()[0] for line in capsys.readouterr().out.splitlines() if "BED-" in line]
    expected = [
        bed.patient.bed
        for bed in sorted(
            engine_spy.snapshot.beds,
            key=lambda b: b.assessment.composite_score,
            reverse=True,
        )
    ]
    assert printed == expected


def test_tick_closes_the_engine_it_built(cli_config: Settings, engine_spy: EngineSpy) -> None:
    """A ``finally``, not a happy path: ``tick`` opens a database session and must give it back."""
    assert run("tick", "--ticks", "1") == 0
    assert engine_spy.closed == 1


def test_tick_json_is_the_snapshot_and_only_the_snapshot(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], engine_spy: EngineSpy
) -> None:
    """``--json`` is what a shell script reads, so a stray header line would corrupt it."""
    assert run("tick", "--ticks", "2", "--json") == 0
    out = capsys.readouterr().out
    # Parsing the *whole* stream is the assertion: one banner line and this raises.
    payload = json.loads(out)
    assert payload["tick"] == 2
    assert len(payload["beds"]) == cli_config.bed_count
    assert payload == json.loads(json.dumps(engine_spy.snapshot.as_dict(), default=str))
    assert "Ward at tick" not in out  # the human rule, and the machine mode returns before it


def test_tick_names_the_sources_it_resolved(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """A degraded deployment has to say which channels it is actually running on.

    The isolated ward has nothing trained, so the model line is the one that matters: printing
    an empty value there reads as a display bug, while naming the fallback is a diagnosis.
    """
    assert run("tick", "--ticks", "1") == 0
    out = capsys.readouterr().out
    assert "none (NEWS2 + vision only)" in out
    assert "source" in out
    assert "vision" in out
    assert "Active alerts" in out


# ----------------------------------------------------------------------------------- info


def test_info_prints_the_settings_it_resolved(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first command to run when a deployment misbehaves, so it must show real values."""
    assert run("info") == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert str(cli_config.bed_count) in out
    assert str(cli_config.project_root) in out
    assert cli_config.vitals_source in out


def test_info_reports_the_api_key_without_printing_it(
    config: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whether a key is set is operational information; the key itself is a secret."""
    secret = "s3cret-ward-key"
    monkeypatch.setattr(cli, "get_settings", lambda: config.with_overrides(api_key=secret))
    assert run("info") == 0
    out = capsys.readouterr().out
    assert "key set" in out
    assert secret not in out


def test_info_redacts_database_credentials(
    config: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    password = "do-not-print-this-password"
    database_url = f"postgresql+psycopg://icu:{password}@db.example/ward"
    monkeypatch.setattr(
        cli, "get_settings", lambda: config.with_overrides(database_url=database_url)
    )
    assert run("info") == 0
    out = capsys.readouterr().out
    assert password not in out
    assert "postgresql+psycopg://icu:***@db.example/ward" in out


def test_info_flags_a_missing_api_key(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_config.api_key is None
    run("info")
    assert "NOT SET" in capsys.readouterr().out


def test_info_reports_every_component_readiness(
    cli_config: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ready``/``degraded`` per component is the whole point; a short list hides a broken one."""
    from icu_monitor.api.deps import AppState

    state = AppState(cli_config)
    try:
        names = [component["name"] for component in state.readiness()]
    finally:
        state.close()

    assert run("info") == 0
    out = capsys.readouterr().out
    assert "Readiness" in out
    assert names
    for name in names:
        assert name in out


# ---------------------------------------------------------------------------------- serve


@pytest.fixture
def uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Record what ``cmd_serve`` would have started, without starting it."""
    import uvicorn

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    return calls


def test_serve_uses_the_configured_host_and_port(
    cli_config: Settings, uvicorn_run: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    assert run("serve") == 0
    (_args, kwargs) = uvicorn_run[0]
    assert kwargs["host"] == cli_config.api_host
    assert kwargs["port"] == cli_config.api_port


def test_serve_flags_override_the_configuration(
    cli_config: Settings, uvicorn_run: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    assert run("serve", "--host", "127.0.0.1", "--port", "9100") == 0
    (_args, kwargs) = uvicorn_run[0]
    assert (kwargs["host"], kwargs["port"]) == ("127.0.0.1", 9100)


def test_serve_passes_the_app_object_when_not_reloading(
    cli_config: Settings, uvicorn_run: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    """Handing uvicorn the object skips a second import of the whole application."""
    from icu_monitor.api.main import app

    assert run("serve") == 0
    (args, kwargs) = uvicorn_run[0]
    assert args[0] is app
    assert kwargs["log_config"] is None  # logging is already configured; uvicorn must not reset it


def test_serve_passes_an_import_string_when_reloading(
    cli_config: Settings, uvicorn_run: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    """``reload=True`` re-imports in a child process, which an object cannot survive."""
    assert run("serve", "--reload") == 0
    (args, kwargs) = uvicorn_run[0]
    assert args[0] == "icu_monitor.api.main:app"
    assert kwargs["reload"] is True


def test_serve_warns_when_the_api_is_unauthenticated(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], uvicorn_run: list[Any]
) -> None:
    """The default host is ``0.0.0.0``. An operator who has not set a key needs to know."""
    assert cli_config.api_key is None
    assert run("serve") == 0
    out = capsys.readouterr().out
    assert "ICU_API_KEY is not set" in out
    assert "unauthenticated" in out


def test_serve_is_quiet_once_a_key_is_set(
    config: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    uvicorn_run: list[Any],
) -> None:
    secret = "s3cret-ward-key"
    monkeypatch.setattr(cli, "get_settings", lambda: config.with_overrides(api_key=secret))
    assert run("serve") == 0
    out = capsys.readouterr().out
    assert "unauthenticated" not in out
    assert secret not in out


# ------------------------------------------------------------------------------ dashboard


@pytest.fixture
def streamlit_call(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the ``streamlit run`` command line instead of spawning Streamlit."""
    calls: list[dict[str, Any]] = []

    def call(command: list[str], env: dict[str, str] | None = None, **_: Any) -> int:
        calls.append({"command": command, "env": env or {}})
        return 0

    monkeypatch.setattr(cli.subprocess, "call", call)
    return calls


def test_dashboard_runs_the_packaged_script(
    cli_config: Settings, streamlit_call: list[dict[str, Any]]
) -> None:
    """The script inside the package, not the repository-root shim - a wheel has no shim."""
    assert run("dashboard") == 0
    command = streamlit_call[0]["command"]
    script = Path(command[4])
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert script.exists()
    assert script.name == "app.py"
    assert script.parent.name == "ui"


def test_dashboard_is_headless_unless_a_browser_is_asked_for(
    cli_config: Settings, streamlit_call: list[dict[str, Any]]
) -> None:
    """Headless is the container default; ``--browser`` is the developer's opt-in."""
    run("dashboard")
    run("dashboard", "--browser")
    headless = [
        call["command"][call["command"].index("--server.headless") + 1] for call in streamlit_call
    ]
    assert headless == ["true", "false"]


def test_dashboard_passes_the_port_through(
    cli_config: Settings, streamlit_call: list[dict[str, Any]]
) -> None:
    run("dashboard", "--port", "9001")
    command = streamlit_call[0]["command"]
    assert command[command.index("--server.port") + 1] == "9001"


def test_dashboard_never_phones_home(
    cli_config: Settings, streamlit_call: list[dict[str, Any]]
) -> None:
    """Streamlit's usage telemetry is on by default, and this is not the project's data to send."""
    run("dashboard")
    command = streamlit_call[0]["command"]
    assert command[command.index("--browser.gatherUsageStats") + 1] == "false"


def test_dashboard_makes_the_package_importable_to_the_child(
    cli_config: Settings, monkeypatch: pytest.MonkeyPatch, streamlit_call: list[dict[str, Any]]
) -> None:
    """``icu-monitor dashboard`` has to work from a clone, with no ``pip install -e .``.

    Streamlit owns its own process, so the child inherits nothing but the environment; without
    ``src`` on ``PYTHONPATH`` its very first ``import icu_monitor`` fails.
    """
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    run("dashboard")
    entries = streamlit_call[0]["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(Path(cli.__file__).resolve().parent.parent) in entries
    assert "/somewhere/else" in entries  # an existing path is prepended to, never replaced


def test_dashboard_returns_the_childs_exit_code(
    cli_config: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Streamlit that failed to start must not report success to the shell that called it."""
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **kw: 3)
    assert run("dashboard") == 3


# ------------------------------------------------------------------------------------ etl


@pytest.fixture
def etl_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Stub both ETL entry points. Neither reads the archive; both record how they were called.

    ``cmd_etl`` imports them from the ``icu_monitor.data`` package inside the handler, so the
    patch goes on the package - which is also where the real ETL would be found, and so is the
    one place that cannot drift from the import.
    """
    from icu_monitor import data

    calls: dict[str, list[dict[str, Any]]] = {"build": [], "synthesise": []}

    def record(key: str, result: Any) -> Any:
        def stub(**kwargs: Any) -> Any:
            calls[key].append(kwargs)
            return result() if callable(result) else result

        return stub

    monkeypatch.setattr(data, "build_dataset", record("build", summary))
    monkeypatch.setattr(data, "synthesise_cohort", record("synthesise", summary))
    return calls


def test_etl_prints_what_it_built(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], etl_calls: dict[str, Any]
) -> None:
    assert run("etl") == 0
    out = capsys.readouterr().out
    assert "PhysioNet Challenge 2012 set-a" in out
    assert "12,000" in out  # windows, thousands-separated because six digits are unreadable
    assert "58.3%" in out  # 7000 low of 12000 windows
    assert "vitals_windows.parquet" in out


def test_etl_passes_the_limit_through(cli_config: Settings, etl_calls: dict[str, Any]) -> None:
    """``--limit`` is the "does my download parse at all" check, so it must reach the reader."""
    assert run("etl", "--limit", "25") == 0
    assert etl_calls["build"][0]["limit"] == 25
    assert etl_calls["build"][0]["config"] is cli_config


def test_etl_reads_the_archive_by_default(cli_config: Settings, etl_calls: dict[str, Any]) -> None:
    run("etl")
    assert etl_calls["build"][0]["limit"] is None
    assert etl_calls["synthesise"] == []


def test_etl_synthetic_uses_the_simulator_instead(
    cli_config: Settings, etl_calls: dict[str, Any]
) -> None:
    """The no-download path, and the one the README tells a first-time reader to run."""
    assert run("etl", "--synthetic", "--stays", "5") == 0
    assert etl_calls["synthesise"][0]["n_stays"] == 5
    assert etl_calls["build"] == []


def test_etl_points_a_missing_archive_at_the_synthetic_path(
    cli_config: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that only says "file not found" leaves the reader stuck at step one."""
    from icu_monitor import data

    def missing(**_: Any) -> Any:
        raise FileNotFoundError("set-a not found under data/raw")

    monkeypatch.setattr(data, "build_dataset", missing)
    assert run("etl") == 2
    out = capsys.readouterr().out
    assert "set-a not found under data/raw" in out
    assert "etl --synthetic" in out


def test_etl_survives_a_cohort_with_no_windows(
    cli_config: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--limit 1`` can legitimately yield nothing, and a share of nothing is not a crash."""
    from icu_monitor import data

    empty = summary(windows=0, class_counts={"low": 0, "medium": 0, "high": 0}, paths={})
    monkeypatch.setattr(data, "build_dataset", lambda **_: empty)
    assert run("etl") == 0
    assert "0.0%" in capsys.readouterr().out


# ---------------------------------------------------------------------------------- train


@pytest.fixture
def train_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from icu_monitor.ml import train as train_module

    calls: list[dict[str, Any]] = []

    def stub(**kwargs: Any) -> TrainingResult:
        calls.append(kwargs)
        return training_result()

    monkeypatch.setattr(train_module, "train", stub)
    return calls


def test_train_marks_the_winner_in_the_candidate_table(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], train_calls: list[Any]
) -> None:
    """Best first, with the chosen model starred: the table is a decision, not a log."""
    assert run("train") == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if "macro-F1" in line]
    # The name is whatever sits before the score, which keeps this off the star column's width.
    assert [row.split("macro-F1")[0].split()[-1] for row in rows] == [
        "hist_gradient_boosting",
        "random_forest",
    ]
    assert rows[0].lstrip().startswith("*")
    assert not rows[1].lstrip().startswith("*")


def test_train_prints_the_held_out_metrics(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], train_calls: list[Any]
) -> None:
    """Held-out numbers, not cross-validation ones: the version and what it actually scored."""
    assert run("train") == 0
    out = capsys.readouterr().out
    assert "20260905-1203-hist_gradient_boosting" in out
    for value in ("0.531", "0.548", "0.612", "0.287", "0.708"):
        assert value in out
    for label in ("low", "medium", "high"):
        assert label in out


def test_train_passes_every_flag_through(cli_config: Settings, train_calls: list[Any]) -> None:
    assert (
        run(
            "train",
            "--candidates",
            "random_forest, logistic_regression",
            "--splits",
            "3",
            "--test-fraction",
            "0.3",
            "--no-importances",
        )
        == 0
    )
    call = train_calls[0]
    # Split *and stripped*: `--candidates "a, b"` is what a shell user types.
    assert call["candidates"] == ["random_forest", "logistic_regression"]
    assert call["n_splits"] == 3
    assert call["test_fraction"] == 0.3
    assert call["with_importances"] is False


def test_train_defaults_to_every_candidate_and_full_importances(
    cli_config: Settings, train_calls: list[Any]
) -> None:
    """``None`` means "the whole roster"; an empty list would silently train nothing."""
    run("train")
    assert train_calls[0]["candidates"] is None
    assert train_calls[0]["with_importances"] is True


def test_train_refuses_a_missing_dataset(
    cli_config: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from icu_monitor.ml import train as train_module

    def missing(**_: Any) -> TrainingResult:
        raise FileNotFoundError("run `icu-monitor etl` first")

    monkeypatch.setattr(train_module, "train", missing)
    assert run("train") == 2
    assert "run `icu-monitor etl` first" in capsys.readouterr().out


def test_train_says_the_numbers_are_modest_on_purpose(
    cli_config: Settings, capsys: pytest.CaptureFixture[str], train_calls: list[Any]
) -> None:
    """A macro-F1 of 0.53 printed without context invites exactly the wrong conclusion.

    Six hours of vitals do not determine an ICU outcome, so this note is part of the output
    rather than a line in the README nobody reads at 2 a.m.
    """
    run("train")
    out = capsys.readouterr().out
    assert "modest by design" in out
    assert "model card" in out
