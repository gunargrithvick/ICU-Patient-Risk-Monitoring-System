"""Computer vision: frame sources, person detection, and clinical interpretation.

The layer is three swappable stages behind one class. :class:`VisionPipeline` wires a
:class:`~icu_monitor.vision.sources.FrameSource` to a
:class:`~icu_monitor.vision.detector.Detector` and a
:class:`~icu_monitor.vision.analyzer.VisionAnalyzer`, and every stage has a working
no-hardware, no-weights implementation - so the vision channel runs in CI, in a slim
container, and on a hosted Streamlit instance, degrading visibly rather than crashing.
"""

from __future__ import annotations

from icu_monitor.vision.analyzer import (
    ABSENCE_FRAMES,
    DEFAULT_BED_REGION,
    IN_BED_OVERLAP,
    MOTION_SCALE,
    UPRIGHT_ASPECT_RATIO,
    BedRegion,
    VisionAnalyzer,
    VisionPipeline,
    annotate,
    classify_posture,
)
from icu_monitor.vision.detector import (
    Detector,
    HeuristicDetector,
    NullDetector,
    UnavailableDetector,
    YoloDetector,
    build_detector,
    timed_detect,
)
from icu_monitor.vision.sources import (
    SCENARIO_SCRIPT,
    CameraSource,
    Frame,
    FrameSource,
    NullSource,
    SyntheticSource,
    UnavailableSource,
    VideoFileSource,
    build_frame_source,
)

__all__ = [
    "ABSENCE_FRAMES",
    "DEFAULT_BED_REGION",
    "IN_BED_OVERLAP",
    "MOTION_SCALE",
    "SCENARIO_SCRIPT",
    "UPRIGHT_ASPECT_RATIO",
    "BedRegion",
    "CameraSource",
    "Detector",
    "Frame",
    "FrameSource",
    "HeuristicDetector",
    "NullDetector",
    "NullSource",
    "SyntheticSource",
    "UnavailableDetector",
    "UnavailableSource",
    "VideoFileSource",
    "VisionAnalyzer",
    "VisionPipeline",
    "YoloDetector",
    "annotate",
    "build_detector",
    "build_frame_source",
    "classify_posture",
    "timed_detect",
]
