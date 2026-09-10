"""Patient detection: a real model when available, a working fallback when not.

Two detectors implement the same :class:`Detector` protocol.

``YoloDetector``
    Ultralytics YOLO, filtered to the ``person`` class. Imported lazily and only when
    weights are actually present, because ``ultralytics`` pulls in Torch - roughly
    800 MB - and the app must remain installable and runnable without it.

``HeuristicDetector``
    Classical background subtraction: an exponentially-weighted background model, a
    difference threshold, and the largest above-threshold blob, with short-term box
    persistence so a momentarily still patient is not "lost". No Torch, no OpenCV, no
    downloads - it is pure NumPy, so it runs in CI and in a slim container.

The heuristic is not as good as YOLO and the UI says which one is running. That is the
point: the channel degrades visibly instead of pretending, and the composite risk score
records the backend it came from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from icu_monitor.config import Settings
from icu_monitor.config import settings as default_settings
from icu_monitor.core.types import Detection
from icu_monitor.vision.sources import Frame

logger = logging.getLogger(__name__)

#: Ultralytics/COCO class index for "person".
PERSON_CLASS_ID = 0

#: Pixels differing from the background model by more than this count as foreground.
FOREGROUND_THRESHOLD = 26.0

#: How fast the background model forgets. Lower = longer memory.
BACKGROUND_ALPHA = 0.035

#: A blob smaller than this fraction of the frame is noise, not a patient.
MIN_BLOB_AREA_FRACTION = 0.012

#: Frames a stale box is re-reported before the patient is declared absent.
BOX_PERSISTENCE_FRAMES = 12

#: Frames the background model settles for before any detection is reported.
WARMUP_FRAMES = 6

#: Width in pixels of the ring sampled when repairing the background behind a patient.
REPAIR_RING = 14


@runtime_checkable
class Detector(Protocol):
    """Anything that can find people in a frame."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def detect(self, frame: Frame) -> list[Detection]: ...


# --------------------------------------------------------------------------------------
# Heuristic detector
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Blob:
    x1: int
    y1: int
    x2: int
    y2: int
    coverage: float


class HeuristicDetector:
    """Background-subtraction detector: no model weights, no downloads.

    The background is an exponential moving average of past frames, so anything that
    moves - or that has appeared since the model settled - stands out. Three refinements
    make it work on a *stationary* patient, which is the case plain background
    subtraction gets wrong, and a fourth keeps the machinery from outliving its subject.

    **The background behind the patient is repaired.** When a box is acquired, the
    foreground pixels inside it are overwritten with a per-row median of the ring around
    it: an estimate of the linen behind the patient. From then on the silhouette differs
    from the model and is detected every frame, moving or not.

    **Only foreground is repaired.** This guard is the whole difference between the repair
    working and the detector latching up. Filling the entire box interior also overwrites
    the parts of the *scene* the box happens to span - the pillow, the bed rail, the frame -
    with a value they never had, which invents a difference where there was none. Those
    pixels then read as patient forever, so the box freezes over a region the patient has
    long since left and every posture downstream is read off a ghost. Repairing only what
    was already foreground cannot invent anything.

    **A repaired pixel is an estimate, so it is allowed to be corrected until the scene
    agrees with it.** Adaptation is frozen inside the tracked box - otherwise the EMA would
    slowly re-absorb a patient who settles - but repaired pixels *outside* it keep adapting,
    and they keep that right until they actually match, not merely until they have been
    blended once. Retiring the flag after a single blend leaves a wrong estimate 3.5% of the
    way corrected and then freezes it under the ordinary "do not adapt a difference" rule -
    which is how a patient who sits up leaves a phantom of their old pose lying in the bed,
    wide enough to dominate the projections and drag the box off the real patient.

    **An estimate is discarded outright once the scene disproves it.** Adaptation is the slow
    correction; this is the fast one, and it is the only one that works when the estimate is
    *inside* the tracked box - where adaptation is deliberately frozen. Each repaired pixel
    remembers the value it held before the guess, which is the last thing actually observed
    there. A frame showing that value again is proof the patient has moved off it, so the
    observation is restored and the flag dropped. Without it a discharged patient leaves a
    silhouette that is detected as a patient, whose own box then protects it from correction:
    the bay reads as occupied for the rest of the session. See
    :meth:`_revert_disproved_repairs`.
    """

    def __init__(
        self,
        *,
        threshold: float = FOREGROUND_THRESHOLD,
        alpha: float = BACKGROUND_ALPHA,
        min_area_fraction: float = MIN_BLOB_AREA_FRACTION,
        persistence: int = BOX_PERSISTENCE_FRAMES,
        warmup_frames: int = WARMUP_FRAMES,
    ) -> None:
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.min_area_fraction = float(min_area_fraction)
        self.persistence = int(persistence)
        self.warmup_frames = int(warmup_frames)
        self._background: np.ndarray | None = None
        self._repaired: np.ndarray | None = None
        self._original: np.ndarray | None = None
        self._frames = 0
        self._last: Detection | None = None
        self._stale_frames = 0

    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def description(self) -> str:
        return "Background-subtraction blob detector (NumPy, no model weights)"

    @property
    def available(self) -> bool:
        return True

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _grey(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float32)
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return image[..., :3].astype(np.float32) @ weights

    @staticmethod
    def _clip_box(box: Detection, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[0], shape[1]
        return (
            max(0, min(box.x1, width)),
            max(0, min(box.y1, height)),
            max(0, min(box.x2, width)),
            max(0, min(box.y2, height)),
        )

    def _repair_background(self, box: Detection, mask: np.ndarray) -> None:
        """Estimate the linen behind a newly-found patient and write it into the model.

        Each row of the box is filled with the median of that row's pixels just left and
        right of the box. The ward scene - like most fixed camera views of a bed - is
        banded horizontally, so a row's neighbours are the best available guess at what
        the patient is covering.

        Two pixels are deliberately left alone. Anything **not currently foreground** is
        part of the scene, not of the patient, and writing a row estimate over it would
        manufacture a permanent difference (see the class docstring). Anything **already
        repaired** keeps its earlier estimate, so the fill converges as the box settles
        instead of being rewritten every frame.

        The value each repaired pixel held *before* the estimate replaced it is kept in
        ``_original``. That is the observation the estimate is standing in for, and it is
        what makes the guess reversible - see :meth:`_revert_disproved_repairs`.
        """
        if self._background is None or self._repaired is None or self._original is None:
            return
        x0, y0, x1, y1 = self._clip_box(box, self._background.shape)
        if x1 <= x0 or y1 <= y0:
            return
        target = mask[y0:y1, x0:x1] & ~self._repaired[y0:y1, x0:x1]
        if not target.any():
            return

        width = self._background.shape[1]
        left = self._background[y0:y1, max(0, x0 - REPAIR_RING) : x0]
        right = self._background[y0:y1, x1 : min(width, x1 + REPAIR_RING)]
        ring = [side for side in (left, right) if side.size]
        if not ring:  # pragma: no cover - only when the box spans the full width
            return
        fill = np.median(np.concatenate(ring, axis=1), axis=1)[:, None]

        patch = self._background[y0:y1, x0:x1]
        keep = self._original[y0:y1, x0:x1]
        self._original[y0:y1, x0:x1] = np.where(target, patch, keep)
        self._background[y0:y1, x0:x1] = np.where(target, np.broadcast_to(fill, patch.shape), patch)
        self._repaired[y0:y1, x0:x1] |= target

    def _revert_disproved_repairs(self, grey: np.ndarray) -> None:
        """Undo a repair the moment the scene shows what it was standing in for.

        A repaired pixel says "the patient is covering this; the linen behind them probably
        looks like their neighbours". The frame that disproves that is the one showing the
        value the pixel had *before* the guess - the last thing actually observed there. If
        the patient has moved away, that is exactly what the camera now sees, so the estimate
        is discarded and the observation restored.

        Without this, a wrong estimate can keep itself alive. A patient wheeled out of the bay
        leaves their silhouette flagged as a difference; that difference is detected as a
        patient; the box drawn around it freezes adaptation over precisely the pixels that
        would have corrected it. The bed then reads as occupied for the rest of the session and
        ``PATIENT_ABSENT`` can never fire again. Reverting to the disproved value breaks the
        cycle in a single frame, and it cannot fire while the patient is really there: their
        gown is what made the pixel foreground in the first place, so it does not match.
        """
        if self._background is None or self._repaired is None or self._original is None:
            return
        if not self._repaired.any():
            return
        disproved = self._repaired & (np.abs(grey - self._original) < self.threshold)
        if not disproved.any():
            return
        self._background = np.where(disproved, self._original, self._background)
        self._repaired &= ~disproved

    def _adapt(self, grey: np.ndarray, difference: np.ndarray) -> None:
        """Move the background toward the current frame, except where the patient is."""
        if self._background is None:
            return
        # A pixel the model already agrees with. This is both the ordinary adaptation set
        # and the retirement condition for a repair: an estimate stops being an estimate
        # once the scene confirms it.
        matched = difference < self.threshold
        adapt = matched.copy()
        if self._repaired is not None:
            # A repaired pixel holds an estimate rather than an observation, so it is
            # always eligible to be corrected - otherwise a wrong guess at the linen
            # outlives the patient who prompted it. It keeps that eligibility until it
            # matches; retiring the flag merely because one blend happened would strand
            # the estimate 3.5% of the way corrected and freeze it there forever.
            adapt |= self._repaired
        if self._last is not None:
            x0, y0, x1, y1 = self._clip_box(self._last, grey.shape)
            adapt[y0:y1, x0:x1] = False
        blended = self._background * (1 - self.alpha) + grey * self.alpha
        self._background = np.where(adapt, blended, self._background)
        if self._repaired is not None:
            self._repaired &= ~matched

    @staticmethod
    def _largest_blob(mask: np.ndarray) -> _Blob | None:
        """Bounding box of the dominant foreground region.

        Rather than full connected-component labelling, this takes the row and column
        projections of the mask and keeps the widest contiguous run above a fraction of
        the peak. For a single subject in a fixed camera - which is the entire use case
        here - that is equivalent, and it is an order of magnitude less code.
        """
        if not mask.any():
            return None
        rows = mask.sum(axis=1).astype(np.float32)
        cols = mask.sum(axis=0).astype(np.float32)

        def dominant_run(projection: np.ndarray) -> tuple[int, int] | None:
            peak = float(projection.max())
            if peak <= 0:
                return None
            active = projection >= max(1.0, peak * 0.22)
            best: tuple[int, int] | None = None
            start: int | None = None
            for index, flag in enumerate(active):
                if flag and start is None:
                    start = index
                elif not flag and start is not None:
                    if best is None or index - start > best[1] - best[0]:
                        best = (start, index)
                    start = None
            if start is not None and (best is None or len(active) - start > best[1] - best[0]):
                best = (start, len(active))
            return best

        row_run = dominant_run(rows)
        col_run = dominant_run(cols)
        if row_run is None or col_run is None:
            return None
        y1, y2 = row_run
        x1, x2 = col_run
        area = max(1, (x2 - x1) * (y2 - y1))
        coverage = float(mask[y1:y2, x1:x2].sum()) / area
        return _Blob(x1=x1, y1=y1, x2=x2, y2=y2, coverage=coverage)

    # -- detection ---------------------------------------------------------------------

    def detect(self, frame: Frame) -> list[Detection]:
        grey = self._grey(frame.image)
        if self._background is None or self._background.shape != grey.shape:
            self.reset()
            self._background = grey.copy()
            self._repaired = np.zeros(grey.shape, dtype=bool)
            self._original = grey.copy()
            return []

        # Discard any repair this frame disproves, before it is compared against.
        self._revert_disproved_repairs(grey)

        difference = np.abs(grey - self._background)
        mask = difference >= self.threshold
        # Adapt before deciding, but never inside the patient's box.
        self._adapt(grey, difference)

        self._frames += 1
        if self._frames < self.warmup_frames:
            return []

        detection = self._detection_from(mask, difference)
        if detection is not None:
            self._last = detection
            self._stale_frames = 0
            self._repair_background(detection, mask)
            return [detection]

        # Hold the last known box briefly - a patient can be momentarily obscured by
        # staff or bedding, and reporting "absent" for that would be a false alarm.
        if self._last is not None and self._stale_frames < self.persistence:
            self._stale_frames += 1
            decayed = Detection(
                x1=self._last.x1,
                y1=self._last.y1,
                x2=self._last.x2,
                y2=self._last.y2,
                confidence=max(0.05, self._last.confidence * 0.88),
                label="patient",
            )
            self._last = decayed
            return [decayed]

        self._last = None
        return []

    def _detection_from(self, mask: np.ndarray, difference: np.ndarray) -> Detection | None:
        """Promote the dominant foreground blob to a detection, if it is big enough."""
        blob = self._largest_blob(mask)
        if blob is None:
            return None
        height, width = mask.shape
        area = (blob.x2 - blob.x1) * (blob.y2 - blob.y1)
        if area / float(height * width) < self.min_area_fraction or blob.coverage < 0.12:
            return None
        strength = float(difference[mask].mean()) if mask.any() else 0.0
        confidence = float(np.clip(0.35 + 0.45 * blob.coverage + strength / 220.0, 0.05, 0.97))
        return Detection(
            x1=int(blob.x1),
            y1=int(blob.y1),
            x2=int(blob.x2),
            y2=int(blob.y2),
            confidence=confidence,
            label="patient",
        )

    def reset(self) -> None:
        self._background = None
        self._repaired = None
        self._original = None
        self._frames = 0
        self._last = None
        self._stale_frames = 0


# --------------------------------------------------------------------------------------
# YOLO detector
# --------------------------------------------------------------------------------------


class YoloDetector:
    """Ultralytics YOLO restricted to the ``person`` class."""

    def __init__(self, *, weights: str = "yolov8n.pt", confidence: float = 0.45) -> None:
        self.weights = weights
        self.confidence = float(confidence)
        self._model: Any | None = None
        self._error: str = ""
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._error = f"ultralytics not installed ({exc.__class__.__name__})"
            logger.info("YOLO unavailable: %s", self._error)
            return
        weight_path = Path(self.weights).expanduser()
        if not weight_path.is_file():
            self._error = f"could not load {self.weights}: weights file not found"
            logger.warning("YOLO unavailable: %s", self._error)
            return
        try:
            self._model = YOLO(self.weights)
        except Exception as exc:  # pragma: no cover - depends on weights availability
            self._error = f"could not load {self.weights}: {exc}"
            logger.warning("YOLO unavailable: %s", self._error)

    @property
    def name(self) -> str:
        return "yolo"

    @property
    def description(self) -> str:
        if self._model is None:
            return f"YOLO unavailable - {self._error}"
        return f"Ultralytics YOLO ({self.weights}), person class only"

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: Frame) -> list[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                frame.image, conf=self.confidence, classes=[PERSON_CLASS_ID], verbose=False
            )
        except Exception as exc:  # pragma: no cover - runtime/model specific
            logger.warning("YOLO inference failed (%s); disabling this backend.", exc)
            self._model = None
            self._error = str(exc)
            return []

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                coordinates = np.asarray(box.xyxy, dtype=float).reshape(-1)[:4]
                confidence = float(np.asarray(box.conf, dtype=float).reshape(-1)[0])
                x1, y1, x2, y2 = (round(value) for value in coordinates)
                detections.append(
                    Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, label="patient")
                )
        detections.sort(key=lambda item: item.area, reverse=True)
        return detections


class NullDetector:
    """Detection disabled."""

    @property
    def name(self) -> str:
        return "off"

    @property
    def description(self) -> str:
        return "Detection disabled"

    @property
    def available(self) -> bool:
        return False

    def detect(self, frame: Frame) -> list[Detection]:
        return []


class UnavailableDetector(NullDetector):
    """A detector explicitly requested by the operator but unavailable at runtime."""

    def __init__(self, description: str) -> None:
        self._description = description

    @property
    def description(self) -> str:
        return self._description


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_detector(config: Settings | None = None) -> Detector:
    """Resolve ``ICU_DETECTOR`` to a working detector, degrading gracefully."""
    cfg = config or default_settings
    choice = cfg.detector

    if choice == "off":
        return NullDetector()

    if choice in {"yolo", "auto"}:
        yolo = YoloDetector(weights=cfg.yolo_weights, confidence=cfg.person_confidence)
        if yolo.available:
            return yolo
        if choice == "yolo":
            logger.warning("YOLO requested but unavailable; detection is disabled.")
            return UnavailableDetector(yolo.description)

    return HeuristicDetector()


def timed_detect(detector: Detector, frame: Frame) -> tuple[list[Detection], float]:
    """Run a detector and report its latency in milliseconds."""
    started = time.perf_counter()
    detections = detector.detect(frame)
    return detections, (time.perf_counter() - started) * 1000.0


__all__ = [
    "BACKGROUND_ALPHA",
    "BOX_PERSISTENCE_FRAMES",
    "FOREGROUND_THRESHOLD",
    "MIN_BLOB_AREA_FRACTION",
    "PERSON_CLASS_ID",
    "REPAIR_RING",
    "WARMUP_FRAMES",
    "Detector",
    "HeuristicDetector",
    "NullDetector",
    "UnavailableDetector",
    "YoloDetector",
    "build_detector",
    "timed_detect",
]
