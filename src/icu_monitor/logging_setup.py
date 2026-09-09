"""Logging configuration, applied once per process.

Two things here are not boilerplate.

**UTF-8 on Windows.** The default console encoding is ``cp1252``, and this codebase prints
``SpO₂``, ``≥``, and ``°C`` in ordinary log lines. Without reconfiguring the streams the
first alert message raises ``UnicodeEncodeError`` and takes the process with it - a real
bug in the original project, not a theoretical one.

**Third-party noise is turned down explicitly.** ``watchdog`` and ``matplotlib`` at DEBUG
bury the application's own output, and a monitoring tool whose logs cannot be read is not
monitored.
"""

from __future__ import annotations

import contextlib
import logging
import logging.config
import os
import sys
from typing import Any

#: Libraries that are chatty at DEBUG and rarely interesting.
QUIET_LOGGERS: tuple[tuple[str, int], ...] = (
    ("watchdog", logging.WARNING),
    ("matplotlib", logging.WARNING),
    ("PIL", logging.WARNING),
    ("urllib3", logging.WARNING),
    ("asyncio", logging.WARNING),
    ("multipart", logging.WARNING),
    ("streamlit.runtime.scriptrunner_utils", logging.WARNING),
    ("sqlalchemy.engine", logging.WARNING),
)

_CONFIGURED = False

#: The handlers *this module* installed, so a forced reconfigure replaces them instead of
#: restoring them alongside their own replacements and doubling every line.
_OWNED: list[logging.Handler] = []


def force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 where the platform default cannot carry the output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding.replace("-", "") in {"utf8", "utf8mb4"}:
            continue
        # A stream can refuse: a closed handle, or a host that has swapped in its own
        # non-reconfigurable object. Nothing here is worth failing an entry point over.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def build_config(level: str) -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s %(levelname)-8s %(name)-34s %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {"level": level, "handlers": ["console"]},
    }


def configure_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure logging once. Safe to call from every entry point.

    ``ICU_LOG_LEVEL`` wins over the argument so a container can be made verbose without a
    rebuild. Reading it from the environment directly rather than from ``Settings`` keeps
    this module importable before configuration is resolved.

    ``force=True`` re-applies the configuration - used when a level changes at runtime. The
    console handler installed last time is *not* treated as somebody else's and so is not
    restored beside its own replacement, which would print every line twice.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    force_utf8_streams()
    owned = set(map(id, _OWNED))
    inherited = [handler for handler in logging.getLogger().handlers if id(handler) not in owned]
    resolved = os.environ.get("ICU_LOG_LEVEL") or level or "INFO"
    if isinstance(resolved, int):
        resolved = logging.getLevelName(resolved)
    resolved = str(resolved).upper()
    if resolved not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        resolved = "INFO"

    logging.config.dictConfig(build_config(resolved))
    _OWNED[:] = list(logging.getLogger().handlers)
    _restore_foreign_handlers(inherited)
    for name, quiet_level in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(quiet_level)
    _CONFIGURED = True


def _restore_foreign_handlers(inherited: list[logging.Handler]) -> None:
    """Put back root handlers installed by whatever is hosting this process.

    ``dictConfig`` *replaces* the root handler list, which is right for the console
    handler this module owns and wrong for everyone else's: pytest's ``caplog``, a
    Streamlit or uvicorn parent, or an aggregator in a container all attach at the root
    before an entry point gets here, and silently dropping them makes their output vanish.
    Re-adding by identity is safe because the fresh console handler is a new object.
    """
    root = logging.getLogger()
    current = set(map(id, root.handlers))
    for handler in inherited:
        if id(handler) not in current:
            root.addHandler(handler)


__all__ = ["QUIET_LOGGERS", "build_config", "configure_logging", "force_utf8_streams"]
