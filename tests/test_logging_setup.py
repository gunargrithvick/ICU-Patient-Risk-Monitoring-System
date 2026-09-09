"""Logging setup: the two failures this module exists to prevent.

**A crash on the first alert.** The original project printed ``SpO₂ 88%`` to a Windows console
whose default encoding is ``cp1252``, and ``UnicodeEncodeError`` took the process down with it.
Every unit here goes through :func:`force_utf8_streams`, including the cases where a stream
cannot be reconfigured - a host that swapped in its own object, a closed handle - because an
entry point must not fail over its own logging.

**Silently swallowing everybody else's handlers.** ``dictConfig`` *replaces* the root handler
list. Under pytest that discards ``caplog``; under uvicorn or Streamlit it discards the parent's
handler, and in a container it discards the log aggregator. The whole suite would go quiet in a
way that looks like passing tests, so the restore is asserted directly.

The global state this module owns - ``_CONFIGURED``, the root handlers, the quiet loggers'
levels - is snapshotted and put back around every test, since a leak here would corrupt every
later test file rather than fail this one.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from icu_monitor import logging_setup
from icu_monitor.logging_setup import (
    QUIET_LOGGERS,
    build_config,
    configure_logging,
    force_utf8_streams,
)


@pytest.fixture(autouse=True)
def isolated_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Snapshot every piece of global state this module writes to, and restore it."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    quiet = {name: logging.getLogger(name).level for name, _ in QUIET_LOGGERS}

    monkeypatch.setattr(logging_setup, "_CONFIGURED", False)
    monkeypatch.delenv("ICU_LOG_LEVEL", raising=False)
    yield

    root.handlers[:] = handlers
    root.setLevel(level)
    for name, saved in quiet.items():
        logging.getLogger(name).setLevel(saved)


class FakeStream:
    """A stream that records how it was reconfigured - or refuses, like a real one can."""

    def __init__(self, encoding: str = "cp1252", *, refuse: Exception | None = None) -> None:
        self.encoding = encoding
        self.refuse = refuse
        self.calls: list[dict[str, Any]] = []

    def reconfigure(self, **kwargs: Any) -> None:
        if self.refuse is not None:
            raise self.refuse
        self.calls.append(kwargs)
        # ``getattr`` because one test deletes ``encoding`` to model a stream that reports none.
        self.encoding = str(kwargs.get("encoding", getattr(self, "encoding", "")))


class BareStream:
    """A stream object with no ``reconfigure`` at all - Streamlit and Jupyter both supply one."""

    encoding = "cp1252"


def patch_streams(monkeypatch: pytest.MonkeyPatch, out: Any, err: Any) -> None:
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)


# ------------------------------------------------------------------------- force_utf8_streams


def test_a_windows_console_is_switched_to_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cp1252`` cannot encode ``SpO₂``, ``≥`` or ``°C``, all of which appear in ordinary log
    lines. ``errors="replace"`` is part of the contract: a stream that still cannot carry a
    character must print a placeholder, never raise."""
    out, err = FakeStream("cp1252"), FakeStream("cp1252")
    patch_streams(monkeypatch, out, err)

    force_utf8_streams()

    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


@pytest.mark.parametrize("encoding", ["utf-8", "UTF-8", "utf8", "UTF8"])
def test_a_stream_already_carrying_utf8_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    """Reconfiguring a working stream is a chance to break it for no gain, and the spelling of
    the encoding varies by platform - so all four spellings have to be recognised."""
    out, err = FakeStream(encoding), FakeStream(encoding)
    patch_streams(monkeypatch, out, err)

    force_utf8_streams()

    assert out.calls == []
    assert err.calls == []


def test_a_stream_that_cannot_be_reconfigured_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notebooks and Streamlit replace ``sys.stdout`` with their own capture object. Not being
    able to set its encoding is normal, not an error."""
    patch_streams(monkeypatch, BareStream(), FakeStream("cp1252"))
    force_utf8_streams()  # must not raise on the bare stream
    assert sys.stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


@pytest.mark.parametrize("refusal", [ValueError("detached buffer"), OSError("handle is closed")])
def test_a_stream_that_refuses_does_not_fail_the_entry_point(
    monkeypatch: pytest.MonkeyPatch, refusal: Exception
) -> None:
    """A closed handle raises from inside ``reconfigure``. Nothing about logging setup is worth
    aborting the dashboard or the API over."""
    patch_streams(monkeypatch, FakeStream(refuse=refusal), FakeStream(refuse=refusal))
    force_utf8_streams()


def test_a_stream_with_no_encoding_attribute_is_still_reconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``getattr(stream, "encoding", "")`` guards a stream that reports nothing; the safe
    reading of 'unknown encoding' is 'not UTF-8 yet'."""
    out = FakeStream("cp1252")
    del out.encoding
    patch_streams(monkeypatch, out, FakeStream("utf-8"))

    force_utf8_streams()

    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]


# ------------------------------------------------------------------------------- build_config


def test_logs_go_to_stderr_so_stdout_stays_machine_readable() -> None:
    """``icu vitals --json`` pipes structured output on stdout. A log line landing in the middle
    of it makes the JSON unparseable, so the console handler is pinned to stderr."""
    config = build_config("INFO")
    assert config["handlers"]["console"]["stream"] == "ext://sys.stderr"
    assert config["handlers"]["console"]["class"] == "logging.StreamHandler"


def test_the_config_names_the_logger_in_every_line() -> None:
    """The format carries the module name because 'which subsystem said this' is the first
    question asked of a monitoring tool's log."""
    fmt = build_config("DEBUG")["formatters"]["plain"]["format"]
    assert "%(name)" in fmt
    assert "%(levelname)" in fmt
    assert "%(asctime)" in fmt
    assert build_config("DEBUG")["root"]["level"] == "DEBUG"


def test_existing_loggers_are_not_disabled() -> None:
    """``disable_existing_loggers`` defaults to *true* in ``dictConfig`` and would silence every
    module-level logger created at import time - which, in this package, is all of them."""
    assert build_config("INFO")["disable_existing_loggers"] is False


# --------------------------------------------------------------------------- configure_logging


def test_configuring_installs_one_console_handler_at_the_requested_level() -> None:
    configure_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert any(isinstance(handler, logging.StreamHandler) for handler in root.handlers)


def test_configuring_twice_is_a_no_op_unless_forced() -> None:
    """Every entry point calls this - the CLI, the API, the dashboard, and the dashboard again
    on each rerun. Reconfiguring on a Streamlit rerun would stack a new handler every few
    seconds and duplicate every line."""
    configure_logging("ERROR")
    before = list(logging.getLogger().handlers)

    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.ERROR
    assert logging.getLogger().handlers == before

    configure_logging("DEBUG", force=True)
    assert logging.getLogger().level == logging.DEBUG


def test_the_environment_overrides_the_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container is made verbose with ``ICU_LOG_LEVEL=DEBUG`` and a restart, without a rebuild
    and without editing the code that picked ``INFO``."""
    monkeypatch.setenv("ICU_LOG_LEVEL", "debug")
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.parametrize("level", ["", "verbose", "TRACE", "17", "info-ish"])
def test_an_unusable_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    """A typo in an environment variable must not turn logging off or crash the entry point."""
    monkeypatch.setenv("ICU_LOG_LEVEL", level)
    configure_logging()
    assert logging.getLogger().level == logging.INFO


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (logging.DEBUG, logging.DEBUG),
        (logging.CRITICAL, logging.CRITICAL),
        ("warning", logging.WARNING),
        (None, logging.INFO),
    ],
)
def test_a_level_may_be_given_as_a_number_a_name_or_not_at_all(
    value: str | int | None, expected: int
) -> None:
    configure_logging(value)
    assert logging.getLogger().level == expected


def test_a_numeric_level_with_no_name_falls_back_to_info() -> None:
    """``logging.getLevelName(17)`` returns ``"Level 17"``, which is not a level at all."""
    configure_logging(17)
    assert logging.getLogger().level == logging.INFO


def test_chatty_libraries_are_turned_down() -> None:
    """At DEBUG, watchdog and matplotlib bury the application's own output. A monitoring tool
    whose logs cannot be read is not monitoring anything."""
    configure_logging("DEBUG")
    for name, expected in QUIET_LOGGERS:
        assert logging.getLogger(name).level == expected
    assert logging.getLogger("icu_monitor").getEffectiveLevel() == logging.DEBUG


def test_handlers_installed_by_the_host_survive_configuration() -> None:
    """**The regression guard.** ``dictConfig`` replaces the root handler list. pytest's
    ``caplog``, a uvicorn parent and a container's log aggregator all attach at the root before
    an entry point runs, and dropping them makes their output vanish with no error anywhere."""
    root = logging.getLogger()
    foreign = logging.StreamHandler()
    root.addHandler(foreign)

    configure_logging("INFO")

    assert foreign in root.handlers
    assert len(root.handlers) >= 2  # the host's handler *and* this module's console


def test_the_hosts_handler_is_not_added_twice() -> None:
    """Restoration is by identity, so a handler ``dictConfig`` happened to keep is not doubled."""
    root = logging.getLogger()
    foreign = logging.StreamHandler()
    root.addHandler(foreign)

    configure_logging("INFO", force=True)
    configure_logging("INFO", force=True)

    assert root.handlers.count(foreign) == 1


def test_a_forced_reconfigure_does_not_print_every_line_twice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Restoring 'everybody else's' handlers has to exclude the console handler this module
    installed last time. Counted at the stream, because a duplicated handler is invisible in
    the handler list and obvious in the output."""
    for _ in range(3):
        configure_logging("INFO", force=True)
    logging.getLogger("icu_monitor.test").info("exactly one copy of this line")
    assert capsys.readouterr().err.count("exactly one copy of this line") == 1


def test_a_log_record_survives_the_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    """The end-to-end assertion behind all of the above: configure, log a line containing the
    characters that used to crash the process, and read it back.

    A single ``%`` on purpose: ``logging`` only applies ``%``-formatting when args are passed,
    so ``%%`` would arrive doubled.
    """
    configure_logging("INFO", force=True)
    with caplog.at_level(logging.INFO):
        logging.getLogger("icu_monitor.test").info("SpO₂ 88% ≥ threshold at 38.4°C")
    assert "SpO₂ 88% ≥ threshold at 38.4°C" in caplog.text
