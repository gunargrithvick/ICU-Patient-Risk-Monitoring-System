"""Frame sources: where video comes from, and what to do when there is none.

The original project called ``cv2.VideoCapture(0, cv2.CAP_DSHOW)`` at import time. That
is a DirectShow call, so the module only imported on Windows; it grabbed the default
webcam, so it only worked on a machine with one attached; and it ran at import, so
merely importing the package could hang. The app could not be containerised, could not
run in CI, and could not run on a hosted Streamlit instance.

Here a source is anything satisfying :class:`FrameSource`. Four ship:

``SyntheticSource``
    Renders a ward scene in pure NumPy - no OpenCV, no camera, no files. It runs a
    scripted clinical scenario (settled → agitation → bed exit → fall → recovery) so the
    vision channel and its alerts are demonstrable on any machine, including CI.
``CameraSource``
    A real webcam, opened lazily with a platform-appropriate backend.
``VideoFileSource``
    A recorded clip, looped - the closest thing to a repeatable field test.
``NullSource``
    Explicitly off. The dashboard shows the channel as offline; nothing breaks.

:func:`build_frame_source` resolves configuration to an instance. ``auto`` may use the
synthetic ward when hardware is absent; an explicit camera or video request stays offline
when it cannot be honoured so a real feed is never confused with generated footage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import utcnow

logger = logging.getLogger(__name__)

#: Scene palette (RGB). Deliberately low-saturation so overlay boxes stay legible.
#:
#: The gown is a deep blue against pale sheets on purpose. Background subtraction works
#: on luminance, and an earlier palette put the gown at grey 148 against sheets at 159 -
#: an 11-level difference that no threshold can separate from sensor noise, so a settled
#: patient was invisible to the detector. Real ward linen and gowns do contrast; the
#: scene should not be gentler than reality or it tests nothing.
#:
#: The same rule applies to the *bed frame*, not just the sheet: a recumbent patient
#: overhangs the mattress onto it, and the frame used to sit at luminance 95 against a
#: gown at 101 - so the lower half of a lying patient fell below the detector's threshold
#: and the box collapsed to a thin band along their shoulders. Every surface the patient
#: can be drawn across now clears :data:`~icu_monitor.vision.detector.FOREGROUND_THRESHOLD`,
#: which ``test_vision.py`` asserts directly against that constant.
SCENE_COLOURS: dict[str, tuple[int, int, int]] = {
    "floor": (28, 34, 42),
    "wall": (38, 46, 56),
    "bed": (56, 64, 76),
    "bed_rail": (120, 132, 148),
    "sheet": (150, 160, 174),
    "pillow": (196, 203, 214),
    "patient": (176, 146, 128),
    "gown": (74, 106, 148),
    "monitor": (20, 26, 32),
    "trace": (57, 135, 229),
    "drip": (168, 178, 192),
}

#: The scripted scenario, as ``(label, duration_frames)`` pairs.
#:
#: It opens on an **empty bay**, and that is not staging. Background subtraction cannot see
#: the interior of an object that was already there when the model was seeded - the patient
#: *is* the background, and no amount of adaptation recovers what was never observed. A
#: script that begins with the patient in bed therefore hands the detector the one input it
#: provably cannot solve, and the demo it produces is of a detector failing rather than of a
#: ward being monitored. Twenty-four empty frames cost under a second and remove the problem
#: entirely. They also mean the default cycle exercises
#: :attr:`~icu_monitor.core.types.AlertKind.PATIENT_ABSENT`, which no scene reached before.
SCENARIO_SCRIPT: tuple[tuple[str, int], ...] = (
    ("absent", 24),
    ("settled", 90),
    ("agitation", 60),
    ("sitting", 45),
    ("bed_exit", 55),
    ("fall", 70),
    ("recovery", 50),
    ("settled", 80),
)


@dataclass(slots=True)
class Frame:
    """One captured image plus provenance."""

    image: np.ndarray
    index: int
    captured_at: datetime
    source: str = "unknown"
    scene: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])


@runtime_checkable
class FrameSource(Protocol):
    """Anything that can hand out frames."""

    @property
    def description(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------------------
# NumPy drawing primitives (no OpenCV dependency)
# --------------------------------------------------------------------------------------


def _fill_rect(
    image: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    colour: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    height, width = image.shape[:2]
    xa, xb = int(max(0, min(x0, x1))), int(min(width, max(x0, x1)))
    ya, yb = int(max(0, min(y0, y1))), int(min(height, max(y0, y1)))
    if xb <= xa or yb <= ya:
        return
    patch = image[ya:yb, xa:xb].astype(np.float32)
    tint = np.array(colour, dtype=np.float32)
    image[ya:yb, xa:xb] = (patch * (1.0 - alpha) + tint * alpha).astype(np.uint8)


def _fill_ellipse(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    colour: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    height, width = image.shape[:2]
    x0, x1 = int(max(0, cx - rx)), int(min(width, cx + rx + 1))
    y0, y1 = int(max(0, cy - ry)), int(min(height, cy + ry + 1))
    if x1 <= x0 or y1 <= y0 or rx <= 0 or ry <= 0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    if not mask.any():
        return
    region = image[y0:y1, x0:x1].astype(np.float32)
    tint = np.array(colour, dtype=np.float32)
    region[mask] = region[mask] * (1.0 - alpha) + tint * alpha
    image[y0:y1, x0:x1] = region.astype(np.uint8)


def _vertical_gradient(
    height: int, width: int, top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)
    bottom_arr = np.array(bottom, dtype=np.float32)
    blend = top_arr[None, :] * (1 - ramp) + bottom_arr[None, :] * ramp
    return np.repeat(blend[:, None, :], width, axis=1).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Synthetic ward
# --------------------------------------------------------------------------------------


class SyntheticSource:
    """A rendered ward bay that runs a scripted clinical scenario.

    This is not decoration - it is what makes the vision channel testable. The scene
    steps through settled, agitated, sitting, out-of-bed, fallen, and recovering states,
    so posture classification, the fall-persistence rule, and the vision override each
    get exercised on every run, on any machine, with no hardware.
    """

    def __init__(
        self,
        *,
        width: int = 960,
        height: int = 540,
        scenario: str = "cycle",
        seed: int = 7,
        loop: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scenario = scenario
        self.loop = loop
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._script = self._build_script(scenario)
        self._backdrop = self._render_backdrop()
        self._drift = np.zeros(2, dtype=float)

    # -- scripting ---------------------------------------------------------------------

    @staticmethod
    def _build_script(scenario: str) -> list[str]:
        if scenario == "cycle":
            script: list[str] = []
            for label, frames in SCENARIO_SCRIPT:
                script.extend([label] * frames)
            return script
        return [scenario] * 240

    @property
    def scene(self) -> str:
        if not self._script:
            return "settled"
        if self.loop:
            return self._script[self._index % len(self._script)]
        return self._script[min(self._index, len(self._script) - 1)]

    # -- rendering ---------------------------------------------------------------------

    def _render_backdrop(self) -> np.ndarray:
        image = _vertical_gradient(
            self.height, self.width, SCENE_COLOURS["wall"], SCENE_COLOURS["floor"]
        )
        bed_top = int(self.height * 0.46)
        bed_bottom = int(self.height * 0.86)
        bed_left = int(self.width * 0.14)
        bed_right = int(self.width * 0.84)

        # Bed frame, mattress, rails, pillow.
        _fill_rect(image, bed_left, bed_top, bed_right, bed_bottom, SCENE_COLOURS["bed"])
        _fill_rect(
            image,
            bed_left + 8,
            bed_top + 10,
            bed_right - 8,
            bed_bottom - 34,
            SCENE_COLOURS["sheet"],
        )
        _fill_rect(image, bed_left, bed_top - 6, bed_right, bed_top + 4, SCENE_COLOURS["bed_rail"])
        _fill_rect(
            image,
            bed_left,
            bed_bottom - 34,
            bed_right,
            bed_bottom - 26,
            SCENE_COLOURS["bed_rail"],
        )
        _fill_ellipse(image, bed_left + 74, bed_top + 54, 62, 34, SCENE_COLOURS["pillow"])

        # Monitor on the wall, with a static trace so the scene reads as clinical.
        mx0, my0 = int(self.width * 0.70), int(self.height * 0.08)
        mx1, my1 = int(self.width * 0.95), int(self.height * 0.34)
        _fill_rect(image, mx0, my0, mx1, my1, SCENE_COLOURS["monitor"])
        _fill_rect(image, mx0 + 4, my0 + 4, mx1 - 4, my1 - 4, (12, 16, 20))
        trace_y = (my0 + my1) // 2
        for step in range(mx0 + 10, mx1 - 10):
            phase = (step - mx0) / 18.0
            offset = int(16 * math.sin(phase) * math.exp(-(((phase % 6.283) - 1.2) ** 2)))
            _fill_rect(
                image,
                step,
                trace_y + offset,
                step + 2,
                trace_y + offset + 2,
                SCENE_COLOURS["trace"],
            )

        # IV pole.
        _fill_rect(
            image,
            bed_left - 34,
            self.height * 0.20,
            bed_left - 28,
            bed_bottom,
            SCENE_COLOURS["drip"],
        )
        _fill_rect(
            image,
            bed_left - 48,
            self.height * 0.22,
            bed_left - 14,
            self.height * 0.30,
            SCENE_COLOURS["drip"],
            alpha=0.7,
        )
        return image

    def _patient_box(self, scene: str) -> tuple[float, float, float, float] | None:
        """Bounding box ``(x0, y0, x1, y1)`` for the patient in this scene."""
        bed_top = self.height * 0.46
        bed_bottom = self.height * 0.86
        bed_left = self.width * 0.14
        bed_right = self.width * 0.84
        breath = math.sin(self._index * 0.22) * 3.0

        if scene == "absent":
            return None
        if scene in {"settled", "agitation", "recovery"}:
            # Recumbent: wide and shallow, lying along the bed.
            width = (bed_right - bed_left) * 0.72
            height = (bed_bottom - bed_top) * 0.42
            x0 = bed_left + 46 + self._drift[0]
            y0 = bed_top + 62 + breath + self._drift[1]
            return x0, y0, x0 + width, y0 + height
        if scene == "sitting":
            width = (bed_right - bed_left) * 0.24
            height = (bed_bottom - bed_top) * 0.92
            x0 = bed_left + 90 + self._drift[0]
            y0 = bed_top - 46 + breath * 0.5 + self._drift[1]
            return x0, y0, x0 + width, y0 + height
        if scene == "bed_exit":
            width = (bed_right - bed_left) * 0.20
            height = (bed_bottom - bed_top) * 1.05
            x0 = bed_right + 12 + self._drift[0]
            y0 = bed_top - 30 + self._drift[1]
            return x0, y0, x0 + width, y0 + height
        if scene == "fall":
            # On the floor beside the bed: wide, shallow, and low in frame.
            width = (bed_right - bed_left) * 0.46
            height = (bed_bottom - bed_top) * 0.22
            x0 = bed_right - 40 + self._drift[0]
            y0 = self.height * 0.88 + self._drift[1]
            return x0, y0, x0 + width, min(self.height - 4, y0 + height)
        return None

    def _advance_drift(self, scene: str) -> None:
        """Per-frame movement: breathing-scale for settled, large for agitation."""
        magnitude = {
            "settled": 0.4,
            "recovery": 0.9,
            "agitation": 6.5,
            "sitting": 2.0,
            "bed_exit": 3.4,
            "fall": 2.2,
            "absent": 0.0,
        }.get(scene, 0.6)
        self._drift += self._rng.normal(0.0, magnitude, size=2)
        # Mean-revert so the patient does not wander out of the bay.
        self._drift *= 0.86

    def read(self) -> Frame | None:
        scene = self.scene
        if not self.loop and self._index >= len(self._script):
            return None

        self._advance_drift(scene)
        image = self._backdrop.copy()
        box = self._patient_box(scene)
        if box is not None:
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
            _fill_ellipse(image, cx, cy, rx, ry, SCENE_COLOURS["gown"], alpha=0.96)
            # A head blob at the appropriate end gives the box a realistic profile.
            head_r = min(rx, ry) * 0.55
            if rx >= ry:  # lying or fallen: head to the left
                _fill_ellipse(
                    image, x0 + head_r, cy, head_r, head_r, SCENE_COLOURS["patient"], alpha=0.98
                )
            else:  # upright: head at the top
                _fill_ellipse(
                    image, cx, y0 + head_r, head_r, head_r, SCENE_COLOURS["patient"], alpha=0.98
                )

        # Sensor noise keeps background subtraction honest rather than trivially clean.
        noise = self._rng.normal(0.0, 3.2, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        frame = Frame(
            image=image,
            index=self._index,
            captured_at=utcnow(),
            source="synthetic",
            scene=scene,
        )
        self._index += 1
        return frame

    @property
    def description(self) -> str:
        return f"Synthetic ward bay ({self.width}×{self.height}, scenario '{self.scenario}')"

    @property
    def available(self) -> bool:
        return True

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# OpenCV-backed sources
# --------------------------------------------------------------------------------------


def _load_cv2() -> Any | None:
    """Import OpenCV on demand. Absence is expected, not exceptional."""
    try:
        import cv2
    except ImportError:
        return None
    return cv2


class _OpenCVSource:
    """Shared plumbing for camera and file capture."""

    def __init__(self, label: str) -> None:
        self._cv2 = _load_cv2()
        self._capture: Any | None = None
        self._index = 0
        self._label = label
        self._failures = 0

    def _to_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._cv2 is None:  # pragma: no cover - only reached without OpenCV
            return frame_bgr
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    @property
    def available(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def description(self) -> str:
        return self._label

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class CameraSource(_OpenCVSource):
    """A physical camera. Opened lazily so import never touches hardware."""

    def __init__(
        self,
        *,
        index: int = 0,
        width: int = 960,
        height: int = 540,
    ) -> None:
        super().__init__(f"Camera {index}")
        self.camera_index = int(index)
        self.width = int(width)
        self.height = int(height)
        if self._cv2 is None:
            logger.warning("OpenCV is not installed; camera source unavailable.")
            return
        self._capture = self._open()

    def _open(self) -> Any | None:
        cv2 = self._cv2
        if cv2 is None:  # pragma: no cover - guarded by caller
            return None
        # CAP_DSHOW is a Windows-only backend; probing it on Linux/macOS just fails.
        backends = [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_ANY", 0)]
        for backend in [b for b in backends if b is not None]:
            try:
                capture = cv2.VideoCapture(self.camera_index, backend)
            except Exception:  # pragma: no cover - backend-specific
                continue
            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return capture
            if capture is not None:
                capture.release()
        logger.warning("Could not open camera %s.", self.camera_index)
        return None

    def read(self) -> Frame | None:
        if not self.available:
            return None
        ok, frame_bgr = self._capture.read()  # type: ignore[union-attr]
        if not ok or frame_bgr is None:
            self._failures += 1
            if self._failures > 30:
                logger.warning("Camera %s stopped delivering frames.", self.camera_index)
                self.close()
            return None
        self._failures = 0
        frame = Frame(
            image=self._to_rgb(frame_bgr),
            index=self._index,
            captured_at=utcnow(),
            source="camera",
        )
        self._index += 1
        return frame


class VideoFileSource(_OpenCVSource):
    """A recorded clip, looped so a short file still drives a long session."""

    def __init__(self, path: Path | str, *, loop: bool = True) -> None:
        self.path = Path(path)
        super().__init__(f"Video file {self.path.name}")
        self.loop = bool(loop)
        if self._cv2 is None:
            logger.warning("OpenCV is not installed; video source unavailable.")
            return
        if not self.path.exists():
            logger.warning("Video file %s does not exist.", self.path)
            return
        self._capture = self._cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():  # pragma: no cover - codec-specific
            logger.warning("OpenCV could not decode %s.", self.path)
            self._capture = None

    def read(self) -> Frame | None:
        if not self.available:
            return None
        ok, frame_bgr = self._capture.read()  # type: ignore[union-attr]
        if not ok or frame_bgr is None:
            if not self.loop:
                return None
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[union-attr]
            ok, frame_bgr = self._capture.read()  # type: ignore[union-attr]
            if not ok or frame_bgr is None:  # pragma: no cover - empty file
                return None
        frame = Frame(
            image=self._to_rgb(frame_bgr),
            index=self._index,
            captured_at=utcnow(),
            source="video",
        )
        self._index += 1
        return frame


class NullSource:
    """Vision explicitly disabled."""

    @property
    def description(self) -> str:
        return "Vision disabled"

    @property
    def available(self) -> bool:
        return False

    def read(self) -> Frame | None:
        return None

    def close(self) -> None:
        return None


class UnavailableSource:
    """A requested source that could not be opened.

    Explicit configuration must fail visibly. Silently replacing a missing camera or
    unreadable clip with synthetic footage can make an operator believe they are watching
    a real bedside feed.
    """

    def __init__(self, description: str) -> None:
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    @property
    def available(self) -> bool:
        return False

    def read(self) -> Frame | None:
        return None

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_frame_source(config: Settings | None = None) -> FrameSource:
    """Resolve ``ICU_FRAME_SOURCE`` to a working source, degrading gracefully."""
    cfg = config or default_settings
    choice = cfg.frame_source

    if choice == "off":
        return NullSource()

    if choice in {"camera", "auto"}:
        camera = CameraSource(
            index=cfg.camera_index, width=cfg.camera_width, height=cfg.camera_height
        )
        if camera.available:
            return camera
        camera.close()
        if choice == "camera":
            logger.warning("Camera requested but unavailable; vision is disabled.")
            return UnavailableSource(f"Camera {cfg.camera_index} unavailable")

    if choice in {"video", "auto"} and cfg.video_path is not None:
        video = VideoFileSource(cfg.video_path)
        if video.available:
            return video
        video.close()
        if choice == "video":
            logger.warning("Video requested but unreadable; vision is disabled.")
            return UnavailableSource(f"Video source unavailable: {cfg.video_path}")

    if choice == "video":
        logger.warning("Video requested without a video_path; vision is disabled.")
        return UnavailableSource("Video source requested but no video_path was configured")

    return SyntheticSource(
        width=cfg.camera_width, height=cfg.camera_height, seed=cfg.simulation_seed % 10_000
    )


__all__ = [
    "SCENARIO_SCRIPT",
    "SCENE_COLOURS",
    "CameraSource",
    "Frame",
    "FrameSource",
    "NullSource",
    "SyntheticSource",
    "UnavailableSource",
    "VideoFileSource",
    "build_frame_source",
]
