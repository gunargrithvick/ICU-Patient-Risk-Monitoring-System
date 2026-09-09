"""Visual tokens for the dashboard, and the CSS that applies them.

**Dark-only, deliberately.** A ward display sits in a dim room for twelve-hour shifts, and
a light theme is not a colour flip: every step would have to be re-chosen against a white
surface and re-validated. Two of the four risk colours cannot clear 3:1 contrast on white
while staying distinguishable from each other, so a light mode here would either lie about
its accessibility or quietly drop a risk level. One validated theme beats two guessed ones.

**The palette was computed, not eyeballed.** Each set below was run through a
colour-vision-deficiency validator against the real surface (``#141a21``); the recorded
figures are perceptual OKLab ΔE ×100 under simulated protanopia/deuteranopia/tritanopia.

* ``SERIES`` - 5 categorical hues. Adjacent pairs pass (worst 8.4 protan, amber↔green) and
  all pairs are ≥ 15 apart for normal vision when compared adjacently.
* ``SERIES_DISTINCT`` - the first 3, the only subset that survives an *all-pairs* check
  (worst 9.4 deutan). Any chart where every series is visible at once and must be told
  apart from every other uses this. More than three simultaneous lines gets faceted into
  small multiples instead of gaining a sixth hue.
* ``STATUS`` - the four risk levels. Ordinal, so it spans lightness on purpose and fails a
  categorical lightness-band check by design; all four clear 3:1 on the surface and adjacent
  steps are ≥ 16.8 apart for normal vision. Its worst all-pairs CVD figure (6.9 protan,
  HIGH↔LOW) sits in the band that is only legal with secondary encoding - which is why
  :func:`status_badge` in ``components`` always emits a glyph *and* the level word, and why
  no risk level is ever communicated by fill alone.
"""

from __future__ import annotations

from importlib import resources

# -- surfaces ---------------------------------------------------------------------------

PAGE_BG = "#0b0f14"
SURFACE = "#141a21"
SURFACE_RAISED = "#1b2430"
BORDER = "#232c36"
BORDER_STRONG = "#31404f"

# -- ink --------------------------------------------------------------------------------

INK = "#f5f7fa"
INK_SECONDARY = "#c2cad6"
INK_MUTED = "#8b95a3"

# -- categorical ------------------------------------------------------------------------

SERIES: tuple[str, ...] = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")
SERIES_DISTINCT: tuple[str, ...] = SERIES[:3]

# -- ordinal risk status ----------------------------------------------------------------

STATUS: dict[str, str] = {
    "good": "#2fa36b",
    "warning": "#f0c22a",
    "serious": "#ef7c2b",
    "critical": "#c22b34",
    "muted": INK_MUTED,
}

#: Level name -> (colour, glyph). The glyph is the secondary encoding that makes the
#: status palette legal for colour-blind readers; never drop it.
LEVEL_STYLE: dict[str, tuple[str, str]] = {
    "LOW": (STATUS["good"], "●"),
    "MEDIUM": (STATUS["warning"], "▲"),
    "HIGH": (STATUS["serious"], "◆"),
    "CRITICAL": (STATUS["critical"], "■"),
    "UNKNOWN": (STATUS["muted"], "○"),
}

LEVEL_ORDER: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

#: Single hue, light -> dark, for magnitude. Never a rainbow.
SEQUENTIAL: tuple[str, ...] = ("#dbe9fb", "#a8cbf4", "#71a8ec", "#3987e5", "#1f5fa8", "#123a68")

#: Two poles plus a neutral midpoint, for signed change. Cool = improving, warm = worsening.
DIVERGING: tuple[str, str, str] = ("#3987e5", "#5b6673", "#d95926")

GRID = "#232c36"
AXIS = "#3a4653"


def level_color(level: str) -> str:
    """The fill for a risk level, tolerant of enum values or plain strings."""
    return LEVEL_STYLE.get(str(level).upper(), LEVEL_STYLE["UNKNOWN"])[0]


def level_glyph(level: str) -> str:
    """The shape that carries the level when colour cannot."""
    return LEVEL_STYLE.get(str(level).upper(), LEVEL_STYLE["UNKNOWN"])[1]


def load_css() -> str:
    """Read the stylesheet shipped inside the package."""
    try:
        return (
            resources.files("icu_monitor.ui.assets")
            .joinpath("dashboard.css")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - packaging guard
        return ""


__all__ = [
    "AXIS",
    "BORDER",
    "BORDER_STRONG",
    "DIVERGING",
    "GRID",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "LEVEL_ORDER",
    "LEVEL_STYLE",
    "PAGE_BG",
    "SEQUENTIAL",
    "SERIES",
    "SERIES_DISTINCT",
    "STATUS",
    "SURFACE",
    "SURFACE_RAISED",
    "level_color",
    "level_glyph",
    "load_css",
]
