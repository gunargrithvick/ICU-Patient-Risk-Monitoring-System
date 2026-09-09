"""``streamlit run app.py`` - the entry point Streamlit Community Cloud looks for.

The real dashboard lives in :mod:`icu_monitor.ui.app`. This file exists because hosted
Streamlit runs a script at the repository root and cannot be told to install the project
first, so the ``src`` layout has to be put on ``sys.path`` by hand. When the package *is*
installed (``pip install -e .``) the insert is a no-op and ``icu-monitor dashboard`` is the
better entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from icu_monitor.ui.app import main  # noqa: E402  (path setup must precede the import)

main()
