"""Static assets shipped inside the package.

This module exists so ``importlib.resources.files("icu_monitor.ui.assets")`` resolves and so
``setuptools`` finds the directory as a package and honours the ``package-data`` entry for
``ui/assets/*.css``. A stylesheet loaded from a path relative to ``__file__`` works from a
source checkout and breaks inside a wheel or a zipapp; reading it as a package resource works
in both.
"""

from __future__ import annotations

__all__: list[str] = []
