"""Turning boxes into clinical meaning - posture, falls, bed exit, and motion.

A bounding box on its own says nothing. This module is the interpretation layer, and it
exists mainly to fix the single worst bug in the original project::

    if width / height > 2.2:
        fall_detected = True          # one frame, no context, no bed

That fires on a patient who is simply lying down, and it fires on a single noisy frame.
Three things are needed to make the signal usable:

**Geometry, not just aspect ratio.** Lying *in bed* is normal; lying *on the floor* is a
fall. Posture is combined with the box's position relative to a bed region of interest,
so ``RECUMBENT`` inside the bed is unremarkable while ``RECUMBENT`` outside it is
``COLLAPSED``.

**Persistence.** A fall must be seen for
:attr:`~icu_monitor.config.Settings.fall_persistence_frames` consecutive frames before it
is reported, and absence likewise. One bad frame can no longer page the nurse.

**Motion, measured.** The motion index is the mean absolute frame-to-frame difference
inside the patient's box, normalised to ``[0, 1]``, which distinguishes a settled
patient from an agitated one - the input to the agitation term in risk fusion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import Detection, Posture, VisionSignal
from icu_monitor.vision.detector import Detector, build_detector, timed_detect
from icu_monitor.vision.sources import Frame, FrameSource, build_frame_source

logger = logging.getLogger(__name__)

#: Bed region of interest as fractions of frame width/height ``(x0, y0, x1, y1)``.
#: Matches the synthetic ward bay; override for a real camera placement.
DEFAULT_BED_REGION: tuple[float, float, float, float] = (0.13, 0.40, 0.85, 0.90)

#: Aspect ratio at or below which a box reads as standing rather than sitting.
UPRIGHT_ASPECT_RATIO = 0.62

#: Fraction of the box that must sit inside the bed ROI to count as "in bed".
IN_BED_OVERLAP = 0.45

#: Frame difference (0-255) that counts as fully saturated motion.
MOTION_SCALE = 22.0

#: Consecutive empty frames before the patient is declared absent.
ABSENCE_FRAMES = 8


@dataclass(frozen=True, slots=True)
class BedRegion:
    """A rectangular region of interest in normalised frame coordinates."""

    x0: float = DEFAULT_BED_REGION[0]
    y0: float = DEFAULT_BED_REGION[1]
    x1: float = DEFAULT_BED_REGION[2]
    y1: float = DEFAULT_BED_REGION[3]

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x0 * width),
            int(self.y0 * height),
            int(self.x1 * width),
            int(self.y1 * height),
        )

    def overlap_fraction(self, detection: Detection, width: int, height: int) -> float:
        """Fraction of the detection's area that lies inside the bed."""
        bx0, by0, bx1, by1 = self.pixels(width, height)
        ix0, iy0 = max(detection.x1, bx0), max(detection.y1, by0)
        ix1, iy1 = min(detection.x2, bx1), min(detection.y2, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return ((ix1 - ix0) * (iy1 - iy0)) / max(1, detection.area)


def classify_posture(detection: Detection, *, fall_aspect_ratio: float) -> Posture:
    """Coarse posture from the box's shape alone (position is applied separately)."""
    ratio = detection.aspect_ratio
    if ratio <= 0:
        return Posture.UNKNOWN
    if ratio >= fall_aspect_ratio:
        return Posture.RECUMBENT
    if ratio <= UPRIGHT_ASPECT_RATIO:
        return Posture.UPRIGHT
    return Posture.SEATED


class VisionAnalyzer:
    """Stateful interpretation of a detection stream for one bed."""

    def __init__(
        self,
        *,
        config: Settings | None = None,
        bed_region: BedRegion | None = None,
    ) -> None:
        self._config = config or default_settings
        self.bed_region = bed_region or BedRegion()
        self._previous_grey: np.ndarray | None = None
        self._collapse_streak = 0
        self._exit_streak = 0
        self._absent_streak = 0
        self._motion_history: deque[float] = deque(maxlen=12)

    # -- motion ------------------------------------------------------------------------

    def _motion_index(self, frame: Frame, detection: Detection | None) -> float:
        image = frame.image
        grey = (
            image.astype(np.float32)
            if image.ndim == 2
            else image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        )
        previous, self._previous_grey = self._previous_grey, grey
        if previous is None or previous.shape != grey.shape:
            return 0.0

        difference = np.abs(grey - previous)
        if detection is not None:
            # Measure motion where the patient is, not across the whole room.
            y0 = max(0, detection.y1)
            y1 = min(grey.shape[0], detection.y2)
            x0 = max(0, detection.x1)
            x1 = min(grey.shape[1], detection.x2)
            if y1 > y0 and x1 > x0:
                difference = difference[y0:y1, x0:x1]
        instant = float(np.clip(difference.mean() / MOTION_SCALE, 0.0, 1.0))
        self._motion_history.append(instant)
        # Smooth: agitation is sustained movement, not one twitch.
        return float(np.mean(self._motion_history))

    # -- main entry point --------------------------------------------------------------

    def analyse(
        self,
        frame: Frame | None,
        detections: list[Detection],
        *,
        backend: str,
        latency_ms: float = 0.0,
    ) -> VisionSignal:
        """Fold one frame's detections into a :class:`VisionSignal`."""
        if frame is None:
            return VisionSignal(
                available=False,
                backend=backend,
                source="off",
                note="No frame available from the video source",
            )

        width, height = frame.size
        best = max(detections, key=lambda item: item.area, default=None)
        motion = self._motion_index(frame, best)

        if best is None:
            self._absent_streak += 1
            self._collapse_streak = 0
            self._exit_streak = 0
            absent = self._absent_streak >= ABSENCE_FRAMES
            return VisionSignal(
                available=True,
                patient_present=not absent,
                detections=(),
                posture=Posture.UNKNOWN,
                motion_index=motion,
                backend=backend,
                source=frame.source,
                latency_ms=latency_ms,
                note=(
                    "No person detected in frame"
                    if absent
                    else f"Detection lost ({self._absent_streak}/{ABSENCE_FRAMES} frames)"
                ),
            )

        self._absent_streak = 0
        posture = classify_posture(best, fall_aspect_ratio=self._config.fall_aspect_ratio)
        overlap = self.bed_region.overlap_fraction(best, width, height)
        in_bed = overlap >= IN_BED_OVERLAP

        # A recumbent patient outside the bed has collapsed; upright outside it has
        # climbed out. Both need to persist before they are believed.
        collapsed_now = posture is Posture.RECUMBENT and not in_bed
        exited_now = posture in {Posture.UPRIGHT, Posture.SEATED} and not in_bed

        self._collapse_streak = self._collapse_streak + 1 if collapsed_now else 0
        self._exit_streak = self._exit_streak + 1 if exited_now else 0

        needed = self._config.fall_persistence_frames
        fall = self._collapse_streak >= needed
        bed_exit = self._exit_streak >= needed

        if fall:
            posture = Posture.COLLAPSED

        notes: list[str] = [
            f"box {best.width}×{best.height} px, aspect {best.aspect_ratio:.2f}",
            f"{overlap:.0%} inside bed" if overlap else "outside bed",
        ]
        if collapsed_now and not fall:
            notes.append(f"possible fall ({self._collapse_streak}/{needed} frames)")
        if exited_now and not bed_exit:
            notes.append(f"possible bed exit ({self._exit_streak}/{needed} frames)")
        if frame.scene:
            notes.append(f"scene '{frame.scene}'")

        return VisionSignal(
            available=True,
            patient_present=True,
            detections=tuple(detections[:4]),
            posture=posture,
            fall_suspected=fall,
            bed_exit_suspected=bed_exit,
            motion_index=motion,
            backend=backend,
            source=frame.source,
            latency_ms=latency_ms,
            note="; ".join(notes),
        )

    def reset(self) -> None:
        self._previous_grey = None
        self._collapse_streak = 0
        self._exit_streak = 0
        self._absent_streak = 0
        self._motion_history.clear()


# --------------------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------------------


class VisionPipeline:
    """Source → detector → analyzer, with the last annotated frame kept for display."""

    def __init__(
        self,
        *,
        config: Settings | None = None,
        source: FrameSource | None = None,
        detector: Detector | None = None,
        bed_region: BedRegion | None = None,
    ) -> None:
        self._config = config or default_settings
        self.source = source or build_frame_source(self._config)
        self.detector = detector or build_detector(self._config)
        self.analyzer = VisionAnalyzer(config=self._config, bed_region=bed_region)
        self.last_frame: Frame | None = None
        self.last_signal: VisionSignal = VisionSignal(
            backend=self.detector.name, source="pending", note="Vision starting up"
        )

    @property
    def description(self) -> str:
        return f"{self.source.description} → {self.detector.description}"

    @property
    def enabled(self) -> bool:
        return self.source.available and self.detector.available

    def step(self) -> VisionSignal:
        """Capture, detect, interpret. Never raises - vision is a best-effort channel.

        A camera that errors mid-read, or a detector that trips over a malformed frame, must
        cost the dashboard its *video* panel and nothing else: the vitals, the risk model and
        the alerts all keep running. The failure is reported in the signal the UI already
        renders, so it shows up as a message in place of the frame rather than as a gap.
        """
        if not self.source.available:
            self.last_signal = VisionSignal(
                available=False,
                backend=self.detector.name,
                source="off",
                note=self.source.description,
            )
            return self.last_signal

        try:
            frame = self.source.read()
        except Exception as exc:
            logger.warning("Frame capture failed (%s); vision unavailable this tick.", exc)
            self.last_signal = VisionSignal(
                available=False,
                backend=self.detector.name,
                source="off",
                note=f"Frame capture failed: {exc}",
            )
            return self.last_signal

        if frame is None:
            self.last_signal = VisionSignal(
                available=False,
                backend=self.detector.name,
                source="off",
                note="Video source returned no frame",
            )
            return self.last_signal

        try:
            detections, latency = timed_detect(self.detector, frame)
            self.last_frame = frame
            self.last_signal = self.analyzer.analyse(
                frame, detections, backend=self.detector.name, latency_ms=latency
            )
        except Exception as exc:
            logger.warning("Vision analysis failed (%s); reporting the channel as down.", exc)
            self.last_signal = VisionSignal(
                available=False,
                backend=self.detector.name,
                source=frame.source,
                note=f"Vision analysis failed: {exc}",
            )
        return self.last_signal

    def annotated_frame(self) -> np.ndarray | None:
        """The most recent frame with the bed ROI and detections drawn on it."""
        if self.last_frame is None:
            return None
        return annotate(
            self.last_frame,
            self.last_signal,
            bed_region=self.analyzer.bed_region,
        )

    def close(self) -> None:
        try:
            self.source.close()
        except Exception as exc:  # pragma: no cover - device teardown is platform-specific
            logger.warning("Releasing the frame source failed (%s); continuing.", exc)


# --------------------------------------------------------------------------------------
# Annotation (pure NumPy so it works without OpenCV)
# --------------------------------------------------------------------------------------

#: Overlay colours (RGB) mirroring the dashboard's status palette.
OVERLAY_COLOURS: dict[str, tuple[int, int, int]] = {
    "bed": (139, 149, 163),
    "ok": (12, 163, 12),
    "warning": (250, 178, 25),
    "critical": (208, 59, 59),
}


def _stroke_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    height, width = image.shape[:2]
    x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        return
    tint = np.array(colour, dtype=np.uint8)
    thickness = max(1, int(thickness))
    image[y0 : min(height, y0 + thickness), x0:x1] = tint
    image[max(0, y1 - thickness) : y1, x0:x1] = tint
    image[y0:y1, x0 : min(width, x0 + thickness)] = tint
    image[y0:y1, max(0, x1 - thickness) : x1] = tint


def _dashed_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    colour: tuple[int, int, int],
    dash: int = 14,
) -> None:
    """A dashed outline - used for the bed ROI so it reads as a reference, not a finding."""
    for x in range(x0, x1, dash * 2):
        _stroke_rect(image, x, y0, min(x + dash, x1), y0 + 2, colour, 2)
        _stroke_rect(image, x, y1 - 2, min(x + dash, x1), y1, colour, 2)
    for y in range(y0, y1, dash * 2):
        _stroke_rect(image, x0, y, x0 + 2, min(y + dash, y1), colour, 2)
        _stroke_rect(image, x1 - 2, y, x1, min(y + dash, y1), colour, 2)


def annotate(
    frame: Frame,
    signal: VisionSignal,
    *,
    bed_region: BedRegion | None = None,
) -> np.ndarray:
    """Draw the bed ROI and detection boxes onto a copy of the frame.

    Colour follows the finding, and the dashboard pairs the image with a text label -
    per the accessibility rule that colour alone never carries meaning.
    """
    image = np.array(frame.image, dtype=np.uint8, copy=True)
    height, width = image.shape[:2]

    region = bed_region or BedRegion()
    bx0, by0, bx1, by1 = region.pixels(width, height)
    _dashed_rect(image, bx0, by0, bx1, by1, OVERLAY_COLOURS["bed"])

    if signal.fall_suspected:
        colour = OVERLAY_COLOURS["critical"]
    elif signal.bed_exit_suspected or not signal.patient_present:
        colour = OVERLAY_COLOURS["warning"]
    else:
        colour = OVERLAY_COLOURS["ok"]

    for index, detection in enumerate(signal.detections):
        _stroke_rect(
            image,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            colour,
            thickness=3 if index == 0 else 1,
        )
    return image


__all__ = [
    "ABSENCE_FRAMES",
    "DEFAULT_BED_REGION",
    "IN_BED_OVERLAP",
    "MOTION_SCALE",
    "OVERLAY_COLOURS",
    "UPRIGHT_ASPECT_RATIO",
    "BedRegion",
    "VisionAnalyzer",
    "VisionPipeline",
    "annotate",
    "classify_posture",
]
