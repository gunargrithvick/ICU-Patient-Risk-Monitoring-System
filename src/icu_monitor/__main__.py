"""``python -m icu_monitor`` - the same command line as the ``icu-monitor`` script.

Kept as a module entry point as well as a console script so a checkout that has not been
``pip install``ed still has one supported way in: ``PYTHONPATH=src python -m icu_monitor …``.
"""

from __future__ import annotations

from icu_monitor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
