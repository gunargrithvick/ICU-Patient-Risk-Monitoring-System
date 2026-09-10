"""``streamlit run app.py`` - the entry point Streamlit Community Cloud looks for.

The real dashboard lives in :mod:`icu_monitor.ui.app`. This repository-root shim is the
entry point used by Streamlit hosting, while the ``src`` path fallback also keeps a direct
checkout runnable before installation. Hosted installs use ``requirements.txt``; when the
package is already installed (``python -m pip install -e .``), the path insert is a no-op and
``python -m icu_monitor dashboard`` is the equivalent command-line entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from icu_monitor.ui.app import main  # noqa: E402  (path setup must precede the import)

main()
