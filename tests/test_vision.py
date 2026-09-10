"""The vision channel: no camera, no OpenCV, no Torch - and no crying wolf.

Three properties are worth a test file this size.

**The channel must work on a machine with none of its optional dependencies.** This one has
neither ``cv2`` nor ``ultralytics``, so a test that merely *runs* here proves only that the
degraded path exists. Every hardware-backed path is therefore exercised twice: once with
absence **forced** (``sys.modules["ultralytics"] = None``, ``_load_cv2`` patched to ``None``)
so the fallback is asserted rather than assumed, and once with a **fake injected** so the
real code path - the backend probe, the BGR→RGB conversion, the loop-on-EOF rewind - is
executed on a machine that cannot install the dependency at all.

**A false fall is worse than no fall detector.** The original project had
``if width / height > 2.2: fall_detected = True``, which fires on a patient who is lying
down comfortably in bed. That exact case is pinned here, as is the persistence rule that
stops one noisy frame from paging a nurse, and the whole 474-frame script is run end to end
with the alarm count asserted: exactly one fall episode, exactly one bed-exit episode, and
nothing else - on three frame sizes, because a demo that only behaves at one resolution is a
coincidence rather than a detector.

**The scene and the detector are one contract, not two modules.** The synthetic ward has to
present the patient at a luminance separation the detector's threshold can actually resolve,
on *every* surface the patient is drawn across - sheet, pillow, mattress, rail, floor, wall.
A recumbent patient overhangs the mattress onto the bed frame, and while the frame sat six
luminance levels from the gown the detector saw only a thin band along the shoulders, read it
as an extreme aspect ratio, and reported falls on a sleeping patient. The contrast test here
asserts the gap directly against :data:`FOREGROUND_THRESHOLD`, in the other module, so
lightening a colour in the palette fails the vision suite rather than the demo.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from icu_monitor.config import Settings
from icu_monitor.core.types import Detection, Posture, VisionSignal
from icu_monitor.vision import sources as sources_module
from icu_monitor.vision.analyzer import (
    ABSENCE_FRAMES,
    DEFAULT_BED_REGION,
    IN_BED_OVERLAP,
    MOTION_SCALE,
    OVERLAY_COLOURS,
    UPRIGHT_ASPECT_RATIO,
    BedRegion,
    VisionAnalyzer,
    VisionPipeline,
    annotate,
    classify_posture,
)
from icu_monitor.vision.detector import (
    BACKGROUND_ALPHA,
    BOX_PERSISTENCE_FRAMES,
    FOREGROUND_THRESHOLD,
    MIN_BLOB_AREA_FRACTION,
    PERSON_CLASS_ID,
    WARMUP_FRAMES,
    Detector,
    HeuristicDetector,
    NullDetector,
    YoloDetector,
    build_detector,
    timed_detect,
)
from icu_monitor.vision.sources import (
    SCENARIO_SCRIPT,
    SCENE_COLOURS,
    CameraSource,
    Frame,
    FrameSource,
    NullSource,
    SyntheticSource,
    VideoFileSource,
    build_frame_source,
)

#: The luminance weights every module in the channel agrees on (Rec. 601).
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

#: Alpha the ward renderer paints the gown with, so the pixel the detector sees is a blend.
GOWN_ALPHA = 0.96

EPOCH = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)

#: Frame sizes the scripted scenario is asserted at. A detector that only behaves at the
#: shipped default is a coincidence; the blob geometry has to survive a change of scale.
FRAME_SIZES = ((480, 270), (640, 360), (960, 540))


def luminance(colour: tuple[int, int, int]) -> float:
    """Greyscale value of an RGB triple, by the same weights the detector uses."""
    return float(np.array(colour, dtype=np.float32) @ LUMA)


def vision_settings(**overrides: Any) -> Settings:
    """Settings with the vision channel on, isolated from the environment."""
    values: dict[str, Any] = {
        "frame_source": "synthetic",
        "detector": "heuristic",
        "camera_width": 480,
        "camera_height": 270,
        "database_url": "sqlite://",
        "api_key": None,
    }
    values.update(overrides)
    return Settings(**values)


def make_frame(
    image: np.ndarray, *, index: int = 0, source: str = "test", scene: str = ""
) -> Frame:
    return Frame(image=image, index=index, captured_at=EPOCH, source=source, scene=scene)


def flat_image(width: int = 120, height: int = 80, value: int = 40) -> np.ndarray:
    """A featureless frame - the background model's easiest possible input."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def with_block(
    image: np.ndarray,
    *,
    rows: tuple[int, int] = (20, 60),
    cols: tuple[int, int] = (30, 90),
    value: int = 200,
) -> np.ndarray:
    """A copy of ``image`` with a bright rectangle pasted in - a synthetic 'patient'."""
    out = image.copy()
    out[rows[0] : rows[1], cols[0] : cols[1]] = value
    return out


def settle(detector: HeuristicDetector, image: np.ndarray, frames: int = WARMUP_FRAMES) -> None:
    """Seed and warm the background model on an unchanging frame."""
    for index in range(frames + 1):
        detector.detect(make_frame(image.copy(), index=index))


def detect_block(detector: HeuristicDetector, image: np.ndarray, frames: int = 1) -> list:
    """Feed the same frame ``frames`` times and return the last detection list."""
    found: list = []
    for offset in range(frames):
        found = detector.detect(make_frame(image.copy(), index=100 + offset))
    return found


def make_detection(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, label="patient")


# ------------------------------------------------------------------- fakes for absent deps


class FakeCapture:
    """The slice of ``cv2.VideoCapture`` the sources actually call."""

    def __init__(
        self, *, frames: int = 3, opened: bool = True, size: tuple[int, int] = (8, 6)
    ) -> None:
        self.frames = int(frames)
        self._opened = bool(opened)
        self.size = size
        self.reads = 0
        self.rewinds = 0
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:  # OpenCV's camelCase is the contract, not a style choice
        return self._opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.reads >= self.frames:
            return False, None
        self.reads += 1
        width, height = self.size
        # A BGR frame whose channels are distinguishable, so the RGB swap is observable.
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = 10  # blue
        frame[..., 1] = 20  # green
        frame[..., 2] = 30  # red
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        if prop == FakeCv2.CAP_PROP_POS_FRAMES and value == 0:
            self.reads = 0
            self.rewinds += 1
        return True

    def release(self) -> None:
        self.released = True


class FakeCv2:
    """A stand-in for the ``cv2`` module: enough surface for both OpenCV-backed sources."""

    CAP_DSHOW = 700
    CAP_ANY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2RGB = 4

    def __init__(
        self,
        *,
        capture: FakeCapture | None = None,
        opens: bool = True,
        camera_opens: bool = True,
    ) -> None:
        self.capture = capture if capture is not None else FakeCapture()
        self.opens = bool(opens)
        #: Fails the camera without failing file capture, so the factory's *order* of
        #: preference is testable and not just its two endpoints.
        self.camera_opens = bool(camera_opens)
        self.backends_tried: list[int] = []
        self.paths_opened: list[str] = []

    def VideoCapture(self, target: int | str, backend: int | None = None) -> FakeCapture:
        if isinstance(target, str):
            self.paths_opened.append(target)
            opens = self.opens
        else:
            self.backends_tried.append(backend if backend is not None else self.CAP_ANY)
            opens = self.opens and self.camera_opens
        if not opens:
            return FakeCapture(opened=False)
        return self.capture

    def cvtColor(self, image: np.ndarray, code: int) -> np.ndarray:
        assert code == self.COLOR_BGR2RGB
        return image[..., ::-1].copy()


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> FakeCv2:
    """Install a fake OpenCV so the camera and video paths execute without the real one."""
    stub = FakeCv2()
    monkeypatch.setattr(sources_module, "_load_cv2", lambda: stub)
    return stub


@pytest.fixture
def no_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force OpenCV's absence, whether or not this machine happens to have it."""
    monkeypatch.setattr(sources_module, "_load_cv2", lambda: None)


class FakeYoloBox:
    """One Ultralytics box: tensors in, ``xyxy`` and ``conf`` out."""

    def __init__(self, xyxy: tuple[float, float, float, float], conf: float) -> None:
        self.xyxy = np.array([xyxy], dtype=float)
        self.conf = np.array([conf], dtype=float)


class FakeYoloResult:
    def __init__(self, boxes: list[FakeYoloBox] | None) -> None:
        self.boxes = boxes


class FakeYoloModel:
    """Records what it was asked, so the class filter and confidence can be asserted."""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.calls: list[dict[str, Any]] = []
        self.boxes: list[FakeYoloBox] = []
        self.raises: Exception | None = None
        self.boxes_attribute = True

    def predict(self, image: np.ndarray, **kwargs: Any) -> list[FakeYoloResult]:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return [FakeYoloResult(list(self.boxes) if self.boxes_attribute else None)]


def install_fake_yolo(monkeypatch: pytest.MonkeyPatch) -> list[FakeYoloModel]:
    """Inject a fake ``ultralytics`` so the YOLO path runs where Torch is not installed.

    Returns the list the fake appends each constructed model to, so a test can reach the
    instance a :class:`YoloDetector` built for itself and script its next answer.
    """
    built: list[FakeYoloModel] = []

    def factory(weights: str) -> FakeYoloModel:
        model = FakeYoloModel(weights)
        built.append(model)
        return model

    monkeypatch.setitem(sys.modules, "ultralytics", type("ultralytics", (), {"YOLO": factory}))
    # The production adapter intentionally refuses to let Ultralytics download a missing
    # model. The fake does not need a real binary, so make the path probe succeed here.
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: path.suffix == ".pt" or real_is_file(path))
    return built


@pytest.fixture
def no_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Ultralytics' absence. ``None`` in ``sys.modules`` raises a real ImportError."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)


# ------------------------------------------------------------------ the scripted run helper


@dataclass(frozen=True)
class ScriptRun:
    """Per-scene tally of one complete pass through :data:`SCENARIO_SCRIPT`."""

    frames: int
    total: Counter[str]
    boxed: Counter[str]
    falls: Counter[str]
    exits: Counter[str]
    unseen: Counter[str]
    postures: dict[str, Counter[str]]
    fall_episodes: int
    exit_episodes: int
    #: Frames the *original* project's ``width / height > 2.2`` rule would have alarmed on.
    naive_alarms: int
    motion: dict[str, list[float]]

    def median_motion(self, scene: str) -> float:
        return float(np.median(self.motion[scene]))


def run_script(width: int, height: int) -> ScriptRun:
    """Drive source → detector → analyzer over the whole script exactly once."""
    source = SyntheticSource(width=width, height=height, loop=False)
    detector = HeuristicDetector()
    analyzer = VisionAnalyzer(config=vision_settings())

    total: Counter[str] = Counter()
    boxed: Counter[str] = Counter()
    falls: Counter[str] = Counter()
    exits: Counter[str] = Counter()
    unseen: Counter[str] = Counter()
    postures: dict[str, Counter[str]] = defaultdict(Counter)
    motion: dict[str, list[float]] = defaultdict(list)
    fall_episodes = exit_episodes = frames = naive_alarms = 0
    was_falling = was_exiting = False

    while (frame := source.read()) is not None:
        frames += 1
        detections, latency = timed_detect(detector, frame)
        signal = analyzer.analyse(frame, detections, backend=detector.name, latency_ms=latency)
        total[frame.scene] += 1
        boxed[frame.scene] += bool(signal.person_count)
        unseen[frame.scene] += not signal.patient_present
        postures[frame.scene][signal.posture.value] += 1
        motion[frame.scene].append(signal.motion_index)
        falls[frame.scene] += signal.fall_suspected
        exits[frame.scene] += signal.bed_exit_suspected
        fall_episodes += signal.fall_suspected and not was_falling
        exit_episodes += signal.bed_exit_suspected and not was_exiting
        was_falling, was_exiting = signal.fall_suspected, signal.bed_exit_suspected
        if detections:
            widest = max(detections, key=lambda box: box.area)
            naive_alarms += widest.aspect_ratio > 2.2

    return ScriptRun(
        frames=frames,
        total=total,
        boxed=boxed,
        falls=falls,
        exits=exits,
        unseen=unseen,
        postures=postures,
        fall_episodes=fall_episodes,
        exit_episodes=exit_episodes,
        naive_alarms=naive_alarms,
        motion=motion,
    )


@pytest.fixture(scope="module")
def script_runs() -> dict[tuple[int, int], ScriptRun]:
    """One scripted pass per frame size, shared - each is ~500 frames of NumPy."""
    return {size: run_script(*size) for size in FRAME_SIZES}


# ------------------------------------------------------------------------------------ Frame


def test_a_frames_size_is_width_then_height_not_the_array_shape() -> None:
    """``shape`` is (rows, cols); ``size`` is (x, y). Transposing them puts the bed ROI on
    its side, which is silently plausible on a square frame and wrong on every other."""
    frame = make_frame(np.zeros((270, 480, 3), dtype=np.uint8))
    assert frame.size == (480, 270)


def test_a_frame_carries_its_provenance() -> None:
    """The dashboard names the backend that produced each finding, so it has to travel."""
    frame = make_frame(flat_image(), index=12, source="synthetic", scene="fall")
    assert (frame.index, frame.source, frame.scene) == (12, "synthetic", "fall")
    assert frame.captured_at == EPOCH


def test_a_greyscale_frame_still_reports_a_size() -> None:
    """A mono camera hands back two dimensions. Nothing downstream may assume three."""
    assert make_frame(np.zeros((60, 80), dtype=np.uint8)).size == (80, 60)


def test_the_shipped_sources_satisfy_the_protocol() -> None:
    """``FrameSource`` is runtime-checkable so the factory's return type is verifiable."""
    for source in (SyntheticSource(width=64, height=48), NullSource()):
        assert isinstance(source, FrameSource)


def test_the_shipped_detectors_satisfy_the_protocol() -> None:
    for detector in (HeuristicDetector(), NullDetector()):
        assert isinstance(detector, Detector)


# ------------------------------------------------------- the scene/detector contrast contract


@pytest.mark.parametrize("surface", ["sheet", "pillow", "bed", "bed_rail", "floor", "wall"])
def test_the_patient_contrasts_with_every_surface_they_are_drawn_across(surface: str) -> None:
    """The test that would have caught the defect this file was written after.

    A recumbent patient overhangs the mattress onto the *bed frame*, and the frame used to
    sit at luminance 95 against a gown at 101 - a six-level gap under a threshold of 26. The
    detector saw a thin band along the shoulders, read the extreme aspect ratio as a
    collapse, and reported falls on a sleeping patient. Asserting the gap against
    :data:`FOREGROUND_THRESHOLD` in the *other* module means a palette tweak fails here
    rather than in the demo.
    """
    gown = luminance(SCENE_COLOURS["gown"])
    background = luminance(SCENE_COLOURS[surface])
    # The renderer paints the gown at alpha 0.96, so this is the pixel the detector sees.
    painted = GOWN_ALPHA * gown + (1.0 - GOWN_ALPHA) * background
    assert abs(painted - background) >= FOREGROUND_THRESHOLD


def test_the_head_contrasts_with_the_gown_so_the_box_has_a_profile() -> None:
    """The head blob is what stops a recumbent patient reading as a featureless bar."""
    assert abs(luminance(SCENE_COLOURS["patient"]) - luminance(SCENE_COLOURS["gown"])) >= 20.0


def test_every_scene_colour_is_a_legal_rgb_triple() -> None:
    for name, colour in SCENE_COLOURS.items():
        assert len(colour) == 3, name
        assert all(0 <= channel <= 255 for channel in colour), name


# --------------------------------------------------------------------------- SCENARIO_SCRIPT


def test_the_script_opens_on_an_empty_bay() -> None:
    """Not staging - a hard limit of the method. Background subtraction cannot see the
    interior of an object that was already there when the model was seeded, because the
    patient *is* the background. A script starting in bed hands the detector the one input
    it provably cannot solve, so the demo shows a detector failing rather than a ward."""
    assert SCENARIO_SCRIPT[0][0] == "absent"
    assert SCENARIO_SCRIPT[0][1] >= WARMUP_FRAMES + ABSENCE_FRAMES


def test_the_script_visits_every_state_the_analyzer_can_report() -> None:
    """Each of these drives a different branch: absence, agitation, posture, exit, fall."""
    scenes = {name for name, _ in SCENARIO_SCRIPT}
    assert scenes == {"absent", "settled", "agitation", "sitting", "bed_exit", "fall", "recovery"}


def test_every_scene_lasts_long_enough_to_clear_the_persistence_rule() -> None:
    """A scene shorter than the fall/exit debounce could never raise its own alarm."""
    needed = vision_settings().fall_persistence_frames
    for name, duration in SCENARIO_SCRIPT:
        assert duration > needed, name


def test_the_script_ends_settled_so_a_looping_demo_does_not_jump_scene() -> None:
    """The dashboard loops this forever; ending mid-crisis would restart in an empty bay
    one frame after a fall, which reads as a glitch rather than a scenario."""
    assert SCENARIO_SCRIPT[-1][0] == "settled"


def test_the_scripts_length_is_the_sum_of_its_scenes() -> None:
    source = SyntheticSource(width=64, height=48)
    assert len(source._script) == sum(duration for _, duration in SCENARIO_SCRIPT)


# --------------------------------------------------------------------------- SyntheticSource


def test_the_synthetic_ward_needs_no_hardware_and_says_so() -> None:
    source = SyntheticSource(width=320, height=180)
    assert source.available is True
    assert "320×180" in source.description
    assert "cycle" in source.description
    assert source.close() is None


def test_a_frame_comes_back_at_the_requested_size_and_is_rgb() -> None:
    frame = SyntheticSource(width=320, height=180).read()
    assert frame is not None
    assert frame.image.shape == (180, 320, 3)
    assert frame.image.dtype == np.uint8
    assert frame.size == (320, 180)


def test_a_frame_is_labelled_with_the_scene_that_produced_it() -> None:
    """The scene name is what makes a failure legible: 'no box during bed_exit' localises a
    bug that 'no box at frame 213' does not."""
    source = SyntheticSource(width=64, height=48)
    frame = source.read()
    assert frame is not None
    assert frame.source == "synthetic"
    assert frame.scene == SCENARIO_SCRIPT[0][0]


def test_the_frame_index_counts_up() -> None:
    source = SyntheticSource(width=64, height=48)
    assert [source.read().index for _ in range(4)] == [0, 1, 2, 3]  # type: ignore[union-attr]


def test_the_scenes_arrive_in_the_scripted_order() -> None:
    source = SyntheticSource(width=64, height=48)
    seen: list[str] = []
    for _ in range(sum(duration for _, duration in SCENARIO_SCRIPT)):
        frame = source.read()
        assert frame is not None
        if not seen or seen[-1] != frame.scene:
            seen.append(frame.scene)
    assert seen == [name for name, _ in SCENARIO_SCRIPT]


def test_a_named_scenario_holds_that_one_scene() -> None:
    """``ICU_FRAME_SOURCE`` aside, a fixed scenario is how a UI test pins one appearance."""
    source = SyntheticSource(width=64, height=48, scenario="fall")
    assert {source.read().scene for _ in range(30)} == {"fall"}  # type: ignore[union-attr]


def test_a_looping_source_wraps_back_to_the_first_scene() -> None:
    source = SyntheticSource(width=64, height=48, loop=True)
    length = len(source._script)
    first = source.read()
    for _ in range(length - 1):
        source.read()
    wrapped = source.read()
    assert first is not None and wrapped is not None
    assert wrapped.scene == first.scene
    assert wrapped.index == length


def test_a_non_looping_source_ends_rather_than_repeating() -> None:
    """``loop=False`` is what lets a test assert on a finite, complete scenario."""
    source = SyntheticSource(width=64, height=48, loop=False)
    for _ in range(len(source._script)):
        assert source.read() is not None
    assert source.read() is None
    assert source.read() is None


def test_the_last_scene_is_held_rather_than_indexed_past_the_end() -> None:
    """``scene`` is read before the exhaustion check, so it must not raise at the boundary."""
    source = SyntheticSource(width=64, height=48, loop=False)
    for _ in range(len(source._script)):
        source.read()
    assert source.scene == SCENARIO_SCRIPT[-1][0]


def test_the_same_seed_renders_the_same_frames() -> None:
    """The suite asserts alarm counts over the script; that is only meaningful if the pixels
    are reproducible."""
    first = SyntheticSource(width=96, height=64, seed=5)
    second = SyntheticSource(width=96, height=64, seed=5)
    for _ in range(8):
        left, right = first.read(), second.read()
        assert left is not None and right is not None
        assert np.array_equal(left.image, right.image)


def test_a_different_seed_renders_different_frames() -> None:
    first = SyntheticSource(width=96, height=64, seed=1)
    second = SyntheticSource(width=96, height=64, seed=2)
    assert not np.array_equal(first.read().image, second.read().image)  # type: ignore[union-attr]


def test_the_scene_carries_sensor_noise_so_subtraction_is_not_trivial() -> None:
    """Two frames of an *empty* bay must still differ. A perfectly static backdrop would let
    any threshold succeed, and the demo would prove nothing about a real camera."""
    source = SyntheticSource(width=96, height=64, scenario="absent")
    first, second = source.read(), source.read()
    assert first is not None and second is not None
    assert not np.array_equal(first.image, second.image)
    # Noise, not motion: the difference is small and spread over the whole frame.
    difference = np.abs(first.image.astype(float) - second.image.astype(float))
    assert difference.mean() < FOREGROUND_THRESHOLD / 2


@pytest.mark.parametrize(("width", "height"), FRAME_SIZES)
def test_seeding_on_an_occupied_bed_is_the_case_no_detector_can_solve(
    width: int, height: int
) -> None:
    """Why :data:`SCENARIO_SCRIPT` opens empty, stated as a measurement rather than a comment.

    Frame differencing compares against a model of what was there before. A patient present
    when that model was seeded *is* the background, so their interior never differs and no
    amount of adaptation recovers what was never observed - only the breathing edge shows,
    and the box collapses onto a sliver whose aspect ratio reads as a collapse.

    Same scene, same detector, two seedings: from an empty bay the box recovers ~97% of the
    drawn silhouette; from an occupied one, under a quarter of it. The 24 empty frames the
    script opens with cost under a second and are the entire difference.
    """
    drawn_height = (0.86 - 0.46) * height * 0.42

    occupied = SyntheticSource(width=width, height=height, scenario="settled")
    seeded_occupied = HeuristicDetector()
    sliver: Detection | None = None
    for _ in range(40):
        found = seeded_occupied.detect(occupied.read())  # type: ignore[arg-type]
        if found:
            sliver = found[0]
    assert sliver is not None, "the breathing edge should still be found - just not the patient"
    assert (sliver.y2 - sliver.y1) < 0.45 * drawn_height

    empty_bay = SyntheticSource(width=width, height=height, loop=False)
    seeded_empty = HeuristicDetector()
    heights: list[int] = []
    for _ in range(SCENARIO_SCRIPT[0][1] + 40):
        frame = empty_bay.read()
        assert frame is not None
        found = seeded_empty.detect(frame)
        if found and frame.scene == "settled":
            heights.append(found[0].y2 - found[0].y1)
    assert heights and min(heights) > 0.85 * drawn_height


# ------------------------------------------------------------------------- _load_cv2 / camera


def test_opencv_is_imported_on_demand_and_absence_is_a_return_value() -> None:
    """v1 imported ``cv2`` at module scope, so the whole package failed to import without it.
    Here the import is a function call whose failure mode is ``None``, not a traceback."""
    loaded = sources_module._load_cv2()
    assert loaded is None or hasattr(loaded, "VideoCapture")


def test_a_camera_without_opencv_is_unavailable_rather_than_an_error(
    no_cv2: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        camera = CameraSource(index=0)
    assert camera.available is False
    assert camera.read() is None
    assert "OpenCV is not installed" in caplog.text
    camera.close()


def test_the_camera_probes_the_windows_backend_first_then_the_portable_one(
    fake_cv2: FakeCv2,
) -> None:
    """``CAP_DSHOW`` is Windows-only. v1 hard-wired it, so the module could not open a camera
    on Linux or macOS at all; it is now the first *preference*, with ``CAP_ANY`` behind it."""
    CameraSource(index=2).close()
    assert fake_cv2.backends_tried == [FakeCv2.CAP_DSHOW]

    fake_cv2.opens = False
    CameraSource(index=2).close()
    assert fake_cv2.backends_tried[-2:] == [FakeCv2.CAP_DSHOW, FakeCv2.CAP_ANY]


def test_the_camera_asks_for_the_configured_capture_size(fake_cv2: FakeCv2) -> None:
    CameraSource(index=0, width=1280, height=720).close()
    assert fake_cv2.capture.properties[FakeCv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert fake_cv2.capture.properties[FakeCv2.CAP_PROP_FRAME_HEIGHT] == 720


def test_a_camera_frame_is_converted_from_bgr_to_rgb(fake_cv2: FakeCv2) -> None:
    """OpenCV hands back BGR. Skipping the swap tints the whole dashboard and, worse, changes
    the luminance the detector thresholds on."""
    camera = CameraSource(index=0)
    frame = camera.read()
    assert frame is not None
    assert tuple(frame.image[0, 0]) == (30, 20, 10)  # R, G, B - the fake's channels, swapped
    assert frame.source == "camera"
    camera.close()


def test_the_camera_frame_index_counts_up(fake_cv2: FakeCv2) -> None:
    fake_cv2.capture.frames = 3
    camera = CameraSource(index=0)
    assert [camera.read().index for _ in range(3)] == [0, 1, 2]  # type: ignore[union-attr]
    camera.close()


def test_a_camera_that_stops_delivering_frames_is_eventually_given_up_on(
    fake_cv2: FakeCv2, caplog: pytest.LogCaptureFixture
) -> None:
    """A USB camera that is unplugged mid-session returns ``(False, None)`` forever. Retrying
    it every tick for the rest of the run is how the dashboard ends up stuttering; the source
    tolerates a burst of dropped frames, then releases the handle."""
    fake_cv2.capture.frames = 0
    camera = CameraSource(index=0)
    for _ in range(30):
        assert camera.read() is None
    assert camera.available is True

    # One good frame clears the count: a dropped burst is not a dead camera.
    fake_cv2.capture.frames = fake_cv2.capture.reads + 1
    assert camera.read() is not None
    fake_cv2.capture.frames = 0
    for _ in range(30):
        assert camera.read() is None
    assert camera.available is True

    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        assert camera.read() is None
    assert camera.available is False
    assert fake_cv2.capture.released is True
    assert "stopped delivering frames" in caplog.text


def test_closing_a_camera_releases_the_handle(fake_cv2: FakeCv2) -> None:
    """Streamlit re-runs the script on every interaction. A source that never releases its
    capture leaks a device handle per re-run until the camera can no longer be opened."""
    camera = CameraSource(index=0)
    camera.close()
    assert fake_cv2.capture.released is True
    assert camera.available is False
    assert camera.close() is None  # idempotent


# ---------------------------------------------------------------------------- VideoFileSource


def test_a_video_source_without_opencv_is_unavailable(
    no_cv2: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clip = tmp_path / "ward.mp4"
    clip.write_bytes(b"not really a video")
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        video = VideoFileSource(clip)
    assert video.available is False
    assert video.read() is None
    assert "OpenCV is not installed" in caplog.text


def test_a_missing_clip_is_reported_not_opened(
    fake_cv2: FakeCv2, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        video = VideoFileSource(tmp_path / "nothing-here.mp4")
    assert video.available is False
    assert fake_cv2.paths_opened == []
    assert "does not exist" in caplog.text


def test_a_video_source_names_the_clip_it_is_playing(fake_cv2: FakeCv2, tmp_path: Path) -> None:
    """The dashboard prints this, so a demo recorded from a file is distinguishable from a
    live camera at a glance."""
    clip = tmp_path / "bay-3-overnight.mp4"
    clip.write_bytes(b"stub")
    video = VideoFileSource(clip)
    assert video.available is True
    assert video.description == "Video file bay-3-overnight.mp4"
    assert fake_cv2.paths_opened == [str(clip)]
    frame = video.read()
    assert frame is not None
    assert frame.source == "video"
    assert tuple(frame.image[0, 0]) == (30, 20, 10)
    video.close()


def test_a_short_clip_rewinds_instead_of_ending(fake_cv2: FakeCv2, tmp_path: Path) -> None:
    """A ten-second clip has to drive an hour-long demo, and the rewind must be invisible
    downstream: the frame index keeps counting, so nothing reads the loop as a restart."""
    clip = tmp_path / "loop.mp4"
    clip.write_bytes(b"stub")
    fake_cv2.capture.frames = 2
    video = VideoFileSource(clip, loop=True)
    indices = [frame.index for _ in range(5) if (frame := video.read()) is not None]
    assert indices == [0, 1, 2, 3, 4]
    assert fake_cv2.capture.rewinds == 2
    video.close()


def test_a_clip_played_once_ends_at_the_last_frame(fake_cv2: FakeCv2, tmp_path: Path) -> None:
    clip = tmp_path / "once.mp4"
    clip.write_bytes(b"stub")
    fake_cv2.capture.frames = 2
    video = VideoFileSource(clip, loop=False)
    assert video.read() is not None
    assert video.read() is not None
    assert video.read() is None
    assert fake_cv2.capture.rewinds == 0
    video.close()


# ------------------------------------------------------------- NullSource / build_frame_source


def test_the_null_source_is_off_and_says_so() -> None:
    """``ICU_FRAME_SOURCE=off`` is a supported deployment, not a broken one: the dashboard
    shows the channel as disabled and every other panel keeps working."""
    null = NullSource()
    assert (null.available, null.read(), null.description) == (False, None, "Vision disabled")
    assert null.close() is None


def test_off_builds_the_null_source() -> None:
    assert isinstance(build_frame_source(vision_settings(frame_source="off")), NullSource)


def test_synthetic_is_built_at_the_configured_size_and_seed() -> None:
    """The camera dimensions double as the synthetic frame size, so one setting controls the
    resolution whichever backend ends up running."""
    config = vision_settings(frame_source="synthetic", camera_width=320, camera_height=180)
    source = build_frame_source(config)
    assert isinstance(source, SyntheticSource)
    assert source.width == 320
    assert source.height == 180
    assert source.read().size == (320, 180)  # type: ignore[union-attr]


def test_a_camera_request_that_cannot_be_honoured_degrades_to_the_ward(
    no_cv2: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The headline deployment fix. v1 opened the webcam at import; here a camera-configured
    deployment on a machine with no camera still starts, with a warning."""
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        source = build_frame_source(vision_settings(frame_source="camera"))
    assert source.available is False
    assert "unavailable" in source.description.lower()
    assert "Camera requested but unavailable" in caplog.text


def test_a_camera_request_uses_the_camera_when_there_is_one(fake_cv2: FakeCv2) -> None:
    source = build_frame_source(vision_settings(frame_source="camera", camera_index=1))
    assert isinstance(source, CameraSource)
    assert source.camera_index == 1
    source.close()


def test_auto_prefers_the_camera_then_a_clip_then_the_ward(
    fake_cv2: FakeCv2, tmp_path: Path
) -> None:
    """One setting, three deployments: a ward PC with a camera, a demo machine with a
    recording, and CI with neither - and ``auto`` is the shipped default, so this ordering is
    what almost every run actually takes."""
    clip = tmp_path / "auto.mp4"
    clip.write_bytes(b"stub")
    config = vision_settings(frame_source="auto", video_path=clip)

    with_camera = build_frame_source(config)
    assert isinstance(with_camera, CameraSource)
    with_camera.close()

    fake_cv2.camera_opens = False
    fake_cv2.capture = FakeCapture()  # the closed camera released the shared one
    with_clip = build_frame_source(config)
    assert isinstance(with_clip, VideoFileSource)
    assert with_clip.path == clip
    with_clip.close()

    fake_cv2.opens = False
    assert isinstance(build_frame_source(config), SyntheticSource)


def test_auto_skips_the_clip_when_none_is_configured(no_cv2: None) -> None:
    config = vision_settings(frame_source="auto", video_path=None)
    assert isinstance(build_frame_source(config), SyntheticSource)


def test_a_video_request_that_cannot_be_honoured_degrades_to_the_ward(
    fake_cv2: FakeCv2, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clip = tmp_path / "unreadable.mp4"
    clip.write_bytes(b"stub")
    fake_cv2.opens = False
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.sources"):
        source = build_frame_source(vision_settings(frame_source="video", video_path=clip))
    assert source.available is False
    assert "video" in source.description.lower()
    assert "Video requested but unreadable" in caplog.text


def test_a_video_request_without_a_path_is_reported_as_unavailable(no_cv2: None) -> None:
    """An explicit video request must not quietly become generated footage."""
    source = build_frame_source(vision_settings(frame_source="video", video_path=None))
    assert source.available is False
    assert "no video_path" in source.description


# -------------------------------------------------------------------------- HeuristicDetector


def test_the_heuristic_detector_needs_no_weights_and_says_which_it_is() -> None:
    """The dashboard prints this. A demo running the fallback must not look like YOLO."""
    detector = HeuristicDetector()
    assert detector.available is True
    assert detector.name == "heuristic"
    assert "NumPy" in detector.description


def test_the_first_frame_only_seeds_the_model() -> None:
    """There is nothing to subtract from yet, so any answer would be invented."""
    detector = HeuristicDetector()
    assert detector.detect(make_frame(with_block(flat_image()))) == []


def test_nothing_is_reported_until_the_background_has_settled() -> None:
    """A model seeded one frame ago is mostly noise. Reporting from it means the dashboard
    opens with a box somewhere arbitrary, which is worse than opening with none."""
    detector = HeuristicDetector()
    detector.detect(make_frame(flat_image()))
    patient = with_block(flat_image())
    during_warmup = [
        len(detector.detect(make_frame(patient.copy()))) for _ in range(WARMUP_FRAMES - 1)
    ]
    assert during_warmup == [0] * (WARMUP_FRAMES - 1)
    assert len(detector.detect(make_frame(patient.copy()))) == 1


def test_a_settled_model_boxes_the_patient_exactly() -> None:
    """The box is what every posture decision downstream is read off, so 'roughly right' is
    not a passing grade: the analyzer's aspect-ratio thresholds are within a few percent."""
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)
    found = detect_block(detector, with_block(background))
    assert len(found) == 1
    box = found[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (30, 20, 90, 60)
    assert box.label == "patient"
    assert box.confidence == pytest.approx(0.97, abs=0.01)


def test_a_greyscale_camera_is_detected_from_just_as_well() -> None:
    """``_grey`` has two paths; a mono feed must not fall down the colour one."""
    detector = HeuristicDetector()
    background = np.full((80, 120), 40, dtype=np.uint8)
    settle(detector, background)
    patient = background.copy()
    patient[20:60, 30:90] = 200
    found = detect_block(detector, patient)
    assert [(found[0].x1, found[0].y1, found[0].x2, found[0].y2)] == [(30, 20, 90, 60)]


def test_a_speck_of_noise_is_not_a_patient() -> None:
    """``MIN_BLOB_AREA_FRACTION`` is what stops a moving curtain or a flickering monitor from
    being boxed and posture-classified."""
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)
    speck = with_block(background, rows=(10, 20), cols=(10, 20))
    assert MIN_BLOB_AREA_FRACTION * 120 * 80 > (20 - 10) * (20 - 10)  # the premise, stated
    assert detect_block(detector, speck) == []


def test_a_momentarily_obscured_patient_is_held_not_lost() -> None:
    """Staff step in front of the bed; bedding is rearranged. Reporting 'no patient' for that
    would raise a false absence alert every time a nurse leans over, so the last box is
    re-reported with decaying confidence - and then given up on."""
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)
    found = detect_block(detector, with_block(background))
    original = found[0]

    held = []
    for index in range(BOX_PERSISTENCE_FRAMES + 2):
        held.append(detector.detect(make_frame(background.copy(), index=300 + index)))

    assert all(len(frame) == 1 for frame in held[:BOX_PERSISTENCE_FRAMES])
    assert all(frame == [] for frame in held[BOX_PERSISTENCE_FRAMES:])
    # Same box, decaying confidence: the UI can fade it rather than blinking it away.
    first_held = held[0][0]
    assert (first_held.x1, first_held.y1, first_held.x2, first_held.y2) == (
        original.x1,
        original.y1,
        original.x2,
        original.y2,
    )
    confidences = [frame[0].confidence for frame in held[:BOX_PERSISTENCE_FRAMES]]
    assert confidences[0] == pytest.approx(original.confidence * 0.88)
    assert confidences == sorted(confidences, reverse=True)


def test_a_change_of_frame_size_reseeds_rather_than_crashes() -> None:
    """The camera can be reconfigured mid-session, and the synthetic source is sized from
    settings. Differencing frames of different shapes is a broadcast error."""
    detector = HeuristicDetector()
    settle(detector, flat_image())
    assert detector.detect(make_frame(flat_image(width=60, height=40))) == []
    settle(detector, flat_image(width=60, height=40))
    found = detect_block(
        detector, with_block(flat_image(width=60, height=40), rows=(8, 30), cols=(10, 50))
    )
    assert len(found) == 1


def test_reset_forgets_everything() -> None:
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)
    assert detect_block(detector, with_block(background))
    detector.reset()
    assert detector.detect(make_frame(with_block(background))) == []


def test_a_patient_who_moves_leaves_no_ghost_behind_them() -> None:
    """A patient who sits up must not leave a phantom of their old pose lying in the bed.

    The background behind a patient is *estimated* so that a stationary patient stays
    visible, and an estimate that outlives its subject is a second patient as far as the blob
    search is concerned - and a wide, shallow one, which reads as a collapse. Here the box
    must follow the patient with nothing left at the old position."""
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)

    lying = with_block(background, rows=(48, 68), cols=(14, 104))
    sitting = with_block(background, rows=(12, 62), cols=(46, 74))
    assert detect_block(detector, lying, frames=20)[0].y1 == 48

    found = detect_block(detector, sitting, frames=30)
    assert len(found) == 1
    assert (found[0].x1, found[0].y1, found[0].x2, found[0].y2) == (46, 12, 74, 62)


def test_a_patient_who_leaves_empties_the_bay_again() -> None:
    """A discharge must not leave the bed reading as occupied for the rest of the session.

    This is the failure the estimate-revert rule exists for, and it is self-sustaining
    without it: the departed patient's silhouette is still flagged as a difference, that
    difference is detected as a patient, and the box drawn around it is precisely where
    adaptation is frozen - so nothing can ever correct it. ``PATIENT_ABSENT`` then never
    fires again. The scene here has vertical structure the per-row repair estimate cannot
    guess, which is what makes the estimate wrong in the first place."""
    detector = HeuristicDetector()
    empty = flat_image()
    empty[:, 52:64] = 170  # an IV pole: structure running *across* the patient's rows
    settle(detector, empty)
    occupied = with_block(empty, rows=(26, 58), cols=(18, 102))
    assert detect_block(detector, occupied, frames=25)[0].y1 == 26

    seen = [bool(detector.detect(make_frame(empty.copy(), index=400 + i))) for i in range(40)]
    # Only the deliberate persistence hold, then an empty bay - and it stays empty.
    assert sum(seen) == BOX_PERSISTENCE_FRAMES
    assert not any(seen[BOX_PERSISTENCE_FRAMES:])


def test_a_gradual_lighting_change_is_absorbed_rather_than_alarmed_on() -> None:
    """Dawn through the window, or a dimmer being turned up. The model's memory is set by
    ``BACKGROUND_ALPHA``, and it tracks a ramp with a steady-state lag of drift/alpha - which
    has to stay under the threshold or every sunrise boxes the whole frame."""
    drift = 0.5
    assert drift / BACKGROUND_ALPHA < FOREGROUND_THRESHOLD  # the premise, stated
    detector = HeuristicDetector()
    level = 40.0
    detector.detect(make_frame(flat_image(value=int(level))))
    for index in range(200):
        level += drift
        frame = flat_image(value=int(min(255, level)))
        assert detector.detect(make_frame(frame, index=index)) == []


# ------------------------------------------------------------------------------- YoloDetector


def test_yolo_without_ultralytics_is_unavailable_and_says_why(no_ultralytics: None) -> None:
    """Ultralytics pulls in Torch - some 800 MB - so the package must stay installable and
    runnable without it. Absence is reported in the description the dashboard shows, and it
    is the same string a genuinely missing module produces: forcing it changes nothing."""
    detector = YoloDetector()
    assert detector.available is False
    assert detector.name == "yolo"
    assert "ultralytics not installed (ModuleNotFoundError)" in detector.description
    assert detector.detect(make_frame(flat_image())) == []


def test_yolo_that_cannot_load_its_weights_is_unavailable_and_says_which(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The likeliest real failure: ``ICU_DETECTOR=yolo`` set without the weights downloaded."""

    def explode(weights: str) -> None:
        raise FileNotFoundError(weights)

    monkeypatch.setitem(sys.modules, "ultralytics", type("ultralytics", (), {"YOLO": explode}))
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.detector"):
        detector = YoloDetector(weights="yolov8x.pt")
    assert detector.available is False
    assert "could not load yolov8x.pt" in detector.description
    assert "YOLO unavailable" in caplog.text


def test_yolo_asks_only_for_people_and_only_above_the_configured_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing the class filter to the model rather than filtering afterwards is what keeps a
    ward full of chairs, drip stands and visitors' bags out of the patient channel."""
    built = install_fake_yolo(monkeypatch)
    detector = YoloDetector(weights="yolov8n.pt", confidence=0.5)
    assert detector.available is True
    assert "yolov8n.pt" in detector.description

    detector.detect(make_frame(flat_image()))
    assert len(built) == 1
    assert built[0].weights == "yolov8n.pt"
    assert built[0].calls == [{"conf": 0.5, "classes": [PERSON_CLASS_ID], "verbose": False}]


def test_yolo_box_coordinates_become_whole_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ultralytics returns floats; a box index has to be an integer, and rounding rather than
    truncating keeps the box centred on the person instead of shifted up and left."""
    built = install_fake_yolo(monkeypatch)
    detector = YoloDetector()
    built[0].boxes = [FakeYoloBox((10.4, 20.6, 60.2, 90.8), 0.81)]
    found = detector.detect(make_frame(flat_image()))
    assert [(box.x1, box.y1, box.x2, box.y2) for box in found] == [(10, 21, 60, 91)]
    assert found[0].confidence == pytest.approx(0.81)
    assert found[0].label == "patient"


def test_the_biggest_person_comes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analyzer reads posture off ``detections[0]``. In a ward frame that has to be the
    patient in the bed, not the visitor standing at the door - so nearest, meaning largest."""
    built = install_fake_yolo(monkeypatch)
    detector = YoloDetector()
    built[0].boxes = [
        FakeYoloBox((0, 0, 10, 10), 0.9),  # a distant figure: 100 px
        FakeYoloBox((20, 20, 120, 90), 0.6),  # the patient: 7,000 px
        FakeYoloBox((5, 5, 45, 45), 0.7),  # someone mid-frame: 1,600 px
    ]
    found = detector.detect(make_frame(flat_image()))
    assert [box.area for box in found] == sorted((box.area for box in found), reverse=True)
    assert (found[0].x1, found[0].y1, found[0].x2, found[0].y2) == (20, 20, 120, 90)


def test_a_result_without_boxes_is_skipped_not_unpacked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some Ultralytics tasks return results with no ``boxes`` attribute at all."""
    built = install_fake_yolo(monkeypatch)
    detector = YoloDetector()
    built[0].boxes_attribute = False
    assert detector.detect(make_frame(flat_image())) == []


def test_a_frame_with_nobody_in_it_is_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_yolo(monkeypatch)
    assert YoloDetector().detect(make_frame(flat_image())) == []


def test_an_inference_failure_disables_the_backend_instead_of_crashing_the_tick(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CUDA out-of-memory mid-session is a real event. One frame's exception must not take the
    dashboard down, and retrying a broken model every tick would make it unusable - so the
    channel degrades to unavailable and the UI says so."""
    built = install_fake_yolo(monkeypatch)
    detector = YoloDetector()
    built[0].raises = RuntimeError("CUDA out of memory")
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.detector"):
        assert detector.detect(make_frame(flat_image())) == []
    assert detector.available is False
    assert "CUDA out of memory" in detector.description
    assert "disabling this backend" in caplog.text
    assert detector.detect(make_frame(flat_image())) == []
    assert len(built[0].calls) == 1  # not retried


# ------------------------------------------------------- NullDetector / build_detector / timing


def test_the_null_detector_is_off_and_says_so() -> None:
    null = NullDetector()
    assert (null.name, null.available, null.description) == ("off", False, "Detection disabled")
    assert null.detect(make_frame(flat_image())) == []


def test_detection_can_be_turned_off_entirely() -> None:
    assert isinstance(build_detector(vision_settings(detector="off")), NullDetector)


def test_the_heuristic_can_be_asked_for_by_name() -> None:
    """A ward that has YOLO installed may still want the cheap detector on a shared box."""
    assert isinstance(build_detector(vision_settings(detector="heuristic")), HeuristicDetector)


def test_yolo_requested_but_unavailable_degrades_loudly(
    no_ultralytics: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Loudly, because the operator asked for something they did not get - unlike ``auto``,
    where the fallback is the documented behaviour and a warning would be noise."""
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.detector"):
        detector = build_detector(vision_settings(detector="yolo"))
    assert detector.available is False
    assert "unavailable" in detector.description.lower()
    assert "YOLO requested but unavailable" in caplog.text


def test_auto_falls_back_to_the_heuristic_quietly(
    no_ultralytics: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="icu_monitor.vision.detector"):
        detector = build_detector(vision_settings(detector="auto"))
    assert isinstance(detector, HeuristicDetector)
    assert "YOLO requested but unavailable" not in caplog.text


@pytest.mark.parametrize("choice", ["yolo", "auto"])
def test_yolo_is_used_when_it_loads(choice: str, monkeypatch: pytest.MonkeyPatch) -> None:
    built = install_fake_yolo(monkeypatch)
    config = vision_settings(detector=choice, yolo_weights="yolov8s.pt", person_confidence=0.6)
    detector = build_detector(config)
    assert isinstance(detector, YoloDetector)
    assert built[0].weights == "yolov8s.pt"
    detector.detect(make_frame(flat_image()))
    assert built[0].calls[0]["conf"] == pytest.approx(0.6)


def test_timing_a_detection_reports_a_latency_without_changing_the_answer() -> None:
    """The dashboard shows this per tick; it is how a user notices the vision channel is what
    made the app sluggish, rather than guessing."""
    detector = HeuristicDetector()
    background = flat_image()
    settle(detector, background)
    frame = make_frame(with_block(background), index=99)
    detections, latency = timed_detect(detector, frame)
    assert len(detections) == 1
    assert latency >= 0.0
    assert latency < 5_000.0


# ---------------------------------------------------------------------------------- BedRegion


def test_the_bed_region_converts_fractions_to_pixels() -> None:
    """The ROI is stored as fractions so one calibration survives a change of resolution -
    which is also why the scripted run is asserted at three frame sizes."""
    assert BedRegion().pixels(100, 100) == (13, 40, 85, 90)
    assert BedRegion().pixels(480, 270) == (62, 108, 408, 243)


def test_the_default_bed_region_is_the_documented_one() -> None:
    region = BedRegion()
    assert (region.x0, region.y0, region.x1, region.y1) == DEFAULT_BED_REGION


def test_a_box_inside_the_bed_overlaps_completely() -> None:
    assert BedRegion().overlap_fraction(make_detection(20, 45, 70, 85), 100, 100) == 1.0


def test_a_box_clear_of_the_bed_overlaps_not_at_all() -> None:
    assert BedRegion().overlap_fraction(make_detection(0, 0, 10, 10), 100, 100) == 0.0
    assert BedRegion().overlap_fraction(make_detection(88, 45, 98, 85), 100, 100) == 0.0


def test_a_box_hanging_off_the_bed_overlaps_partly() -> None:
    """The measure is the fraction of the *patient* inside the bed, not of the bed covered:
    a patient half out of bed is halfway to an exit however big the bed is."""
    assert BedRegion().overlap_fraction(make_detection(13, 30, 53, 70), 100, 100) == 0.75


# ----------------------------------------------------------------------------- posture from a box


def test_a_box_with_no_height_has_no_posture() -> None:
    """A degenerate box divides by zero. ``UNKNOWN`` is the honest answer, not a fall."""
    assert (
        classify_posture(make_detection(10, 10, 50, 10), fall_aspect_ratio=1.6) is Posture.UNKNOWN
    )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (80, 50, Posture.RECUMBENT),  # ratio exactly 1.60 - the boundary is inclusive
        (160, 50, Posture.RECUMBENT),
        (79, 50, Posture.SEATED),  # 1.58 - just inside
        (50, 50, Posture.SEATED),
        (32, 50, Posture.SEATED),  # 0.64 - just above upright
        (31, 50, Posture.UPRIGHT),  # exactly 0.62 - inclusive the other way
        (20, 100, Posture.UPRIGHT),
    ],
)
def test_posture_follows_the_boxs_shape(width: int, height: int, expected: Posture) -> None:
    detection = make_detection(10, 10, 10 + width, 10 + height)
    assert classify_posture(detection, fall_aspect_ratio=1.6) is expected


def test_the_upright_threshold_is_the_one_the_module_publishes() -> None:
    """Stated so the parametrised boundaries above cannot silently stop being boundaries."""
    at_threshold = make_detection(0, 0, int(UPRIGHT_ASPECT_RATIO * 100), 100)
    assert classify_posture(at_threshold, fall_aspect_ratio=1.6) is Posture.UPRIGHT


# ------------------------------------------------------------------------------ VisionAnalyzer

#: The analyzer tests all run at this size, where the bed ROI lands on (62, 108, 408, 243).
ANALYSER_SIZE = (480, 270)

#: A patient lying comfortably in bed: aspect ratio 3.0, wholly inside the ROI.
IN_BED_RECUMBENT = make_detection(100, 130, 340, 210)

#: The same shape on the floor below the bed - nothing but the *position* differs.
ON_FLOOR_RECUMBENT = make_detection(100, 246, 340, 266)

#: Standing beside the bed, clear of the ROI: aspect ratio 0.21.
BESIDE_BED_UPRIGHT = make_detection(420, 100, 450, 240)

#: Perched on the edge of the chair beside the bed: aspect ratio 0.75.
BESIDE_BED_SEATED = make_detection(420, 100, 450, 140)


def analyser_frame(index: int = 0, *, value: int = 40, scene: str = "") -> Frame:
    """A featureless frame at :data:`ANALYSER_SIZE` - motion is added by varying ``value``."""
    width, height = ANALYSER_SIZE
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return make_frame(image, index=index, source="synthetic", scene=scene)


def analyse_repeatedly(
    analyzer: VisionAnalyzer, detection: Detection | None, frames: int
) -> list[Any]:
    """Feed one unchanging detection ``frames`` times and return every signal."""
    boxes = [] if detection is None else [detection]
    return [
        analyzer.analyse(analyser_frame(index), list(boxes), backend="heuristic")
        for index in range(frames)
    ]


def test_a_patient_lying_comfortably_in_bed_is_never_a_fall() -> None:
    """**The original project's worst bug, pinned.** v1 read only the aspect ratio::

        if width / height > 2.2: fall_detected = True

    which fires on every sleeping patient, forever. Position is what distinguishes lying
    down from falling down, so this box - ratio 3.0, wholly inside the bed - must stay
    ``RECUMBENT`` no matter how long it is held.
    """
    analyzer = VisionAnalyzer(config=vision_settings())
    signals = analyse_repeatedly(analyzer, IN_BED_RECUMBENT, 60)

    assert all(signal.posture is Posture.RECUMBENT for signal in signals)
    assert not any(signal.fall_suspected for signal in signals)
    assert not any(signal.bed_exit_suspected for signal in signals)
    assert all(signal.patient_present for signal in signals)
    assert "100% inside bed" in signals[-1].note


def test_a_fall_must_persist_before_it_is_believed() -> None:
    """The same shape *outside* the bed is a fall - but only once it has been seen for
    ``fall_persistence_frames`` in a row. The frames before that say 'possible', which is
    what the dashboard shows and what stops one noisy frame from paging a nurse."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=3))
    signals = analyse_repeatedly(analyzer, ON_FLOOR_RECUMBENT, 4)

    assert [signal.fall_suspected for signal in signals] == [False, False, True, True]
    assert "possible fall (1/3 frames)" in signals[0].note
    assert "possible fall (2/3 frames)" in signals[1].note
    assert "possible fall" not in signals[2].note  # no longer possible; now reported
    assert signals[0].posture is Posture.RECUMBENT
    assert signals[2].posture is Posture.COLLAPSED
    assert "outside bed" in signals[2].note


def test_one_clean_frame_resets_the_fall_streak() -> None:
    """Persistence has to mean *consecutive*. Two floor frames either side of one in-bed
    frame is not a fall, or the debounce is decorative."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=3))
    sequence = [ON_FLOOR_RECUMBENT, ON_FLOOR_RECUMBENT, IN_BED_RECUMBENT, ON_FLOOR_RECUMBENT]
    suspected = [
        analyzer.analyse(analyser_frame(index), [box], backend="heuristic").fall_suspected
        for index, box in enumerate(sequence)
    ]
    assert suspected == [False, False, False, False]


def test_a_longer_persistence_setting_is_honoured() -> None:
    """A noisy camera can be made less trigger-happy from configuration alone."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=8))
    signals = analyse_repeatedly(analyzer, ON_FLOOR_RECUMBENT, 8)
    assert [signal.fall_suspected for signal in signals].count(True) == 1
    assert signals[-1].fall_suspected


def test_standing_beside_the_bed_is_a_bed_exit_not_a_fall() -> None:
    """An upright patient outside the ROI has climbed out - a fall risk to escalate, not a
    fall to alarm on. The posture stays ``UPRIGHT``: only a fall rewrites it."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=3))
    signals = analyse_repeatedly(analyzer, BESIDE_BED_UPRIGHT, 4)

    assert [signal.bed_exit_suspected for signal in signals] == [False, False, True, True]
    assert "possible bed exit (2/3 frames)" in signals[1].note
    assert not any(signal.fall_suspected for signal in signals)
    assert all(signal.posture is Posture.UPRIGHT for signal in signals)


def test_sitting_up_outside_the_bed_is_also_a_bed_exit() -> None:
    """Sitting on the edge of a chair is a patient out of bed just as much as standing is."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=2))
    signals = analyse_repeatedly(analyzer, BESIDE_BED_SEATED, 2)
    assert signals[-1].bed_exit_suspected
    assert signals[-1].posture is Posture.SEATED


def test_the_in_bed_threshold_is_inclusive() -> None:
    """A box overlapping the ROI by exactly :data:`IN_BED_OVERLAP` counts as in bed, and one
    pixel row less does not. The boundary is asserted because it is the difference between a
    shrug and an alarm."""
    width, height = 100, 100
    frame = make_frame(np.full((height, width, 3), 40, dtype=np.uint8))
    at_threshold = make_detection(13, 18, 63, 58)  # 900 of 2 000 px inside = 0.45
    just_below = make_detection(13, 17, 63, 57)  # 850 of 2 000 px inside = 0.425
    assert BedRegion().overlap_fraction(at_threshold, width, height) == IN_BED_OVERLAP
    assert BedRegion().overlap_fraction(just_below, width, height) < IN_BED_OVERLAP

    inside = VisionAnalyzer(config=vision_settings(fall_persistence_frames=1))
    signal = inside.analyse(frame, [at_threshold], backend="heuristic")
    assert not signal.bed_exit_suspected
    assert "45% inside bed" in signal.note

    outside = VisionAnalyzer(config=vision_settings(fall_persistence_frames=1))
    assert outside.analyse(frame, [just_below], backend="heuristic").bed_exit_suspected


def test_absence_is_declared_only_after_the_countdown() -> None:
    """A detector that drops one frame has not lost the patient. The signal says how far
    through the countdown it is, so a nurse can tell 'blinked' from 'gone'."""
    analyzer = VisionAnalyzer(config=vision_settings())
    signals = analyse_repeatedly(analyzer, None, ABSENCE_FRAMES + 2)

    present = [signal.patient_present for signal in signals]
    assert present[: ABSENCE_FRAMES - 1] == [True] * (ABSENCE_FRAMES - 1)
    assert present[ABSENCE_FRAMES - 1 :] == [False] * 3
    assert f"Detection lost (1/{ABSENCE_FRAMES} frames)" in signals[0].note
    assert signals[-1].note == "No person detected in frame"
    assert all(signal.available for signal in signals)
    assert all(signal.posture is Posture.UNKNOWN for signal in signals)
    assert all(signal.detections == () for signal in signals)


def test_a_returning_patient_clears_the_absence_countdown() -> None:
    analyzer = VisionAnalyzer(config=vision_settings())
    analyse_repeatedly(analyzer, None, ABSENCE_FRAMES + 4)
    back = analyzer.analyse(analyser_frame(99), [IN_BED_RECUMBENT], backend="heuristic")
    assert back.patient_present
    gone_again = analyzer.analyse(analyser_frame(100), [], backend="heuristic")
    assert gone_again.patient_present
    assert f"Detection lost (1/{ABSENCE_FRAMES} frames)" in gone_again.note


def patched_frame(index: int, *, patch: tuple[slice, slice], value: int) -> Frame:
    """A frame at :data:`ANALYSER_SIZE` with one rectangle raised - localised motion."""
    width, height = ANALYSER_SIZE
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[patch[0], patch[1]] = value
    return make_frame(image, index=index, source="synthetic")


def test_the_first_frame_has_no_motion_to_measure() -> None:
    """Motion is a frame *difference*; there is nothing to subtract from on frame one, and
    guessing would make the agitation term jump at startup."""
    analyzer = VisionAnalyzer(config=vision_settings())
    signal = analyzer.analyse(analyser_frame(0), [IN_BED_RECUMBENT], backend="heuristic")
    assert signal.motion_index == 0.0


def test_a_settled_patient_registers_no_motion() -> None:
    analyzer = VisionAnalyzer(config=vision_settings())
    signals = analyse_repeatedly(analyzer, IN_BED_RECUMBENT, 20)
    assert all(signal.motion_index == 0.0 for signal in signals)


def test_a_thrashing_patient_saturates_the_motion_index() -> None:
    """``MOTION_SCALE`` is well below full swing, so genuine agitation pins the index at 1.0
    rather than living in the bottom of the range where the risk model cannot see it."""
    assert MOTION_SCALE < 255.0
    analyzer = VisionAnalyzer(config=vision_settings())
    box = (slice(130, 210), slice(100, 340))
    readings = [
        analyzer.analyse(
            patched_frame(index, patch=box, value=255 if index % 2 else 40),
            [IN_BED_RECUMBENT],
            backend="heuristic",
        ).motion_index
        for index in range(6)
    ]
    assert readings[0] == 0.0
    assert readings[-1] == pytest.approx(1.0)


def test_motion_is_measured_where_the_patient_is() -> None:
    """A nurse walking past the door is not the patient fidgeting. Restricting the
    difference to the box is what keeps ward traffic out of the agitation term."""
    analyzer = VisionAnalyzer(config=vision_settings())
    elsewhere = (slice(0, 60), slice(0, 60))  # clear of the patient's box
    readings = [
        analyzer.analyse(
            patched_frame(index, patch=elsewhere, value=255 if index % 2 else 40),
            [IN_BED_RECUMBENT],
            backend="heuristic",
        ).motion_index
        for index in range(6)
    ]
    assert readings == [0.0] * 6


def test_no_frame_at_all_is_reported_as_an_unavailable_channel() -> None:
    """Vision going dark must never look like a patient who has vanished."""
    analyzer = VisionAnalyzer(config=vision_settings())
    signal = analyzer.analyse(None, [], backend="heuristic")
    assert not signal.available
    assert signal.source == "off"
    assert signal.note == "No frame available from the video source"
    assert not signal.fall_suspected
    assert not signal.bed_exit_suspected


def test_the_signal_carries_the_scene_and_a_readable_geometry_note() -> None:
    """The note is what the dashboard prints beside the video, so it has to be legible on
    its own: box size, aspect ratio, position relative to the bed, and the scripted scene."""
    analyzer = VisionAnalyzer(config=vision_settings())
    frame = analyser_frame(0, scene="agitation")
    signal = analyzer.analyse(frame, [IN_BED_RECUMBENT], backend="heuristic", latency_ms=4.5)

    assert "box 240×80 px, aspect 3.00" in signal.note
    assert "scene 'agitation'" in signal.note
    assert signal.backend == "heuristic"
    assert signal.source == "synthetic"
    assert signal.latency_ms == 4.5


def test_only_the_first_few_boxes_are_carried_forward() -> None:
    """The signal is serialised into every snapshot; an unbounded list of boxes from a busy
    frame would bloat the history for no clinical gain. The detector hands them over
    largest-first, so truncating keeps the patient and drops the visitors."""
    analyzer = VisionAnalyzer(config=vision_settings())
    extras = [make_detection(10 + i, 120, 20 + i, 140) for i in range(9)]
    signal = analyzer.analyse(analyser_frame(0), [IN_BED_RECUMBENT, *extras], backend="heuristic")
    assert len(signal.detections) == 4
    assert signal.detections[0] == IN_BED_RECUMBENT
    assert signal.posture is Posture.RECUMBENT  # the biggest box drives the posture


def test_the_biggest_box_drives_the_posture_whatever_its_position_in_the_list() -> None:
    """Posture is taken from the largest box by area, not from whichever arrived first - a
    visitor's boxy silhouette must not be mistaken for the patient."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=1))
    visitor = make_detection(430, 120, 460, 200)  # upright, outside the bed, but small
    signal = analyzer.analyse(analyser_frame(0), [visitor, IN_BED_RECUMBENT], backend="heuristic")
    assert signal.posture is Posture.RECUMBENT
    assert not signal.bed_exit_suspected


def test_resetting_the_analyzer_forgets_every_streak() -> None:
    """A bed reassigned to a new patient must not inherit the last one's fall streak."""
    analyzer = VisionAnalyzer(config=vision_settings(fall_persistence_frames=3))
    analyse_repeatedly(analyzer, ON_FLOOR_RECUMBENT, 2)
    analyzer.reset()
    after = analyzer.analyse(analyser_frame(0), [ON_FLOOR_RECUMBENT], backend="heuristic")
    assert not after.fall_suspected
    assert "possible fall (1/3 frames)" in after.note
    assert after.motion_index == 0.0


def test_a_custom_bed_region_moves_the_whole_judgement_with_it() -> None:
    """Camera placement varies bed to bed. Recalibrating the ROI is the only change a new
    mounting should need - the same box reads as in bed or out of it accordingly."""
    default_view = VisionAnalyzer(config=vision_settings(fall_persistence_frames=1))
    assert default_view.analyse(
        analyser_frame(0), [BESIDE_BED_UPRIGHT], backend="heuristic"
    ).bed_exit_suspected

    shifted = BedRegion(x0=0.85, y0=0.35, x1=0.95, y1=0.90)  # the bed is where the patient is
    recalibrated = VisionAnalyzer(
        config=vision_settings(fall_persistence_frames=1), bed_region=shifted
    )
    signal = recalibrated.analyse(analyser_frame(0), [BESIDE_BED_UPRIGHT], backend="heuristic")
    assert not signal.bed_exit_suspected
    assert "100% inside bed" in signal.note
    assert recalibrated.bed_region is shifted


# ------------------------------------------------------------------------------ VisionPipeline


class ScriptedSource:
    """A frame source under the test's control, including its failures."""

    def __init__(self, *, frames: list[Frame] | None = None, available: bool = True) -> None:
        self.queue = list(frames or [])
        self.available = available
        self.description = "Scripted source"
        self.closed = 0
        self.raise_on_read: Exception | None = None

    def read(self) -> Frame | None:
        if self.raise_on_read is not None:
            raise self.raise_on_read
        return self.queue.pop(0) if self.queue else None

    def close(self) -> None:
        self.closed += 1


class ScriptedDetector:
    """A detector under the test's control, including its failures."""

    def __init__(self, *, boxes: list[Detection] | None = None) -> None:
        self.boxes = list(boxes or [])
        self.name = "scripted"
        self.description = "Scripted detector"
        self.available = True
        self.raise_on_detect: Exception | None = None

    def detect(self, frame: Frame) -> list[Detection]:
        if self.raise_on_detect is not None:
            raise self.raise_on_detect
        return list(self.boxes)


def scripted_pipeline(
    *,
    frames: list[Frame] | None = None,
    boxes: list[Detection] | None = None,
    available: bool = True,
) -> tuple[VisionPipeline, ScriptedSource, ScriptedDetector]:
    source = ScriptedSource(frames=frames, available=available)
    detector = ScriptedDetector(boxes=boxes)
    pipeline = VisionPipeline(config=vision_settings(), source=source, detector=detector)
    return pipeline, source, detector


def test_a_disabled_channel_says_so_and_draws_nothing() -> None:
    """With vision off the dashboard must still render - a missing panel, not a stack trace."""
    pipeline = VisionPipeline(config=vision_settings(frame_source="off", detector="off"))
    assert not pipeline.enabled
    assert pipeline.description == "Vision disabled → Detection disabled"
    assert pipeline.annotated_frame() is None

    signal = pipeline.step()
    assert not signal.available
    assert signal.source == "off"
    assert signal.note == "Vision disabled"
    assert pipeline.last_signal is signal


def test_the_pipeline_reports_a_starting_signal_before_its_first_step() -> None:
    """The UI reads ``last_signal`` on its first paint, which happens before any tick."""
    pipeline, _, _ = scripted_pipeline()
    assert pipeline.last_signal.note == "Vision starting up"
    assert pipeline.last_signal.source == "pending"
    assert pipeline.last_signal.backend == "scripted"


def test_one_step_runs_source_detector_and_analyzer_together() -> None:
    pipeline, _, _ = scripted_pipeline(
        frames=[analyser_frame(0, scene="settled")], boxes=[IN_BED_RECUMBENT]
    )
    signal = pipeline.step()
    assert signal.available
    assert signal.patient_present
    assert signal.posture is Posture.RECUMBENT
    assert signal.backend == "scripted"
    assert signal.latency_ms >= 0.0
    assert pipeline.last_frame is not None
    assert pipeline.description == "Scripted source → Scripted detector"


def test_a_source_that_runs_dry_is_reported_not_raised() -> None:
    pipeline, _, _ = scripted_pipeline(frames=[])
    signal = pipeline.step()
    assert not signal.available
    assert signal.note == "Video source returned no frame"


def test_a_camera_that_throws_mid_read_costs_the_video_panel_and_nothing_else(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A USB camera yanked out of its port raises from inside ``read()``. ``step()`` promises
    never to raise, because the vitals, the risk model and the alerting all keep running on a
    ward whose camera has failed - the dashboard shows a message where the frame was."""
    pipeline, source, _ = scripted_pipeline(frames=[analyser_frame(0)])
    source.raise_on_read = OSError("device disconnected")

    with caplog.at_level(logging.WARNING):
        signal = pipeline.step()

    assert not signal.available
    assert signal.note == "Frame capture failed: device disconnected"
    assert signal.source == "off"
    assert pipeline.last_signal is signal
    assert "Frame capture failed" in caplog.text


def test_a_detector_that_throws_is_contained_the_same_way(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed frame that trips the detector must not take the ward down either. The
    frame's own provenance is preserved in the signal so the failure can be traced."""
    pipeline, _, detector = scripted_pipeline(frames=[analyser_frame(0)])
    detector.raise_on_detect = ValueError("bad tensor shape")

    with caplog.at_level(logging.WARNING):
        signal = pipeline.step()

    assert not signal.available
    assert signal.note == "Vision analysis failed: bad tensor shape"
    assert signal.source == "synthetic"
    assert "Vision analysis failed" in caplog.text


def test_a_recovering_camera_is_picked_back_up() -> None:
    """The failure is per-tick, not terminal: the channel must come back on its own once the
    device does, without the operator restarting the app."""
    pipeline, source, _ = scripted_pipeline(frames=[analyser_frame(0)], boxes=[IN_BED_RECUMBENT])
    source.raise_on_read = OSError("device busy")
    assert not pipeline.step().available

    source.raise_on_read = None
    assert pipeline.step().available


def test_an_unavailable_source_is_never_read_from() -> None:
    """The source's own description is surfaced, because that is the sentence explaining
    *why* there is no video - 'no camera found', 'clip missing', and so on."""
    pipeline, source, _ = scripted_pipeline(frames=[analyser_frame(0)], available=False)
    source.raise_on_read = AssertionError("read() must not be called")
    signal = pipeline.step()
    assert not signal.available
    assert signal.note == "Scripted source"
    assert len(source.queue) == 1


def test_closing_the_pipeline_releases_the_source() -> None:
    pipeline, source, _ = scripted_pipeline()
    pipeline.close()
    assert source.closed == 1


def test_a_source_that_throws_on_close_does_not_break_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Teardown is platform-specific and the app is already exiting; a device that refuses to
    release must not turn a clean shutdown into a traceback."""

    class StubbornSource(ScriptedSource):
        def close(self) -> None:
            raise OSError("device still in use")

    pipeline = VisionPipeline(
        config=vision_settings(), source=StubbornSource(), detector=ScriptedDetector()
    )
    with caplog.at_level(logging.WARNING):
        pipeline.close()
    assert "Releasing the frame source failed" in caplog.text


def test_the_annotated_frame_is_only_available_once_one_has_been_captured() -> None:
    pipeline, _, _ = scripted_pipeline(frames=[analyser_frame(0)], boxes=[IN_BED_RECUMBENT])
    assert pipeline.annotated_frame() is None
    pipeline.step()
    annotated = pipeline.annotated_frame()
    assert annotated is not None
    assert annotated.shape == (ANALYSER_SIZE[1], ANALYSER_SIZE[0], 3)


def test_the_annotated_frame_uses_the_pipelines_own_bed_region() -> None:
    """The overlay and the judgement must be drawn from one calibration, or the operator sees
    a box outside a bed the analyzer thinks it is inside."""
    region = BedRegion(x0=0.02, y0=0.02, x1=0.30, y1=0.30)
    pipeline = VisionPipeline(
        config=vision_settings(),
        source=ScriptedSource(frames=[analyser_frame(0)]),
        detector=ScriptedDetector(),
        bed_region=region,
    )
    assert pipeline.analyzer.bed_region is region
    pipeline.step()
    annotated = pipeline.annotated_frame()
    assert annotated is not None
    x0, y0, x1, _ = region.pixels(*ANALYSER_SIZE)
    assert tuple(annotated[y0, x0]) == OVERLAY_COLOURS["bed"]
    assert tuple(annotated[y0, x1 - 1]) == OVERLAY_COLOURS["bed"]
    # ...and nothing is drawn at the default ROI this calibration replaced.
    default_x0, default_y0 = BedRegion().pixels(*ANALYSER_SIZE)[:2]
    assert tuple(annotated[default_y0, default_x0]) != OVERLAY_COLOURS["bed"]


# ------------------------------------------------------------------------------------ annotate


def signal_for(
    detections: tuple[Detection, ...],
    *,
    fall: bool = False,
    exit_: bool = False,
    present: bool = True,
) -> Any:
    return VisionSignal(
        available=True,
        patient_present=present,
        detections=detections,
        posture=Posture.COLLAPSED if fall else Posture.RECUMBENT,
        fall_suspected=fall,
        bed_exit_suspected=exit_,
        backend="scripted",
        source="synthetic",
    )


def test_annotating_never_touches_the_frame_it_was_given() -> None:
    """The frame is also handed to the motion estimator and cached for the next tick. Drawing
    on it in place would make the overlay part of the next frame's background."""
    frame = analyser_frame(0)
    before = frame.image.copy()
    annotate(frame, signal_for((IN_BED_RECUMBENT,)))
    assert np.array_equal(frame.image, before)


def test_the_bed_outline_is_dashed_so_it_reads_as_a_reference() -> None:
    """A solid rectangle would look like a finding. The ROI is context, so it is drawn as
    context - and the gaps are what prove it."""
    annotated = annotate(analyser_frame(0), signal_for(()))
    x0, y0, x1, _ = BedRegion().pixels(*ANALYSER_SIZE)
    top_edge = annotated[y0, x0:x1]
    painted = np.all(top_edge == np.array(OVERLAY_COLOURS["bed"]), axis=-1)
    assert painted.any()  # some of the edge is drawn...
    assert not painted.all()  # ...and some of it is deliberately not


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"fall": True}, "critical"),
        ({"exit_": True}, "warning"),
        ({"present": False}, "warning"),
        ({"fall": True, "exit_": True}, "critical"),  # the worse finding wins
        ({}, "ok"),
    ],
)
def test_the_box_colour_follows_the_finding(kwargs: dict[str, bool], expected: str) -> None:
    """Colour is a second channel, never the only one - the dashboard prints the finding as
    text beside the frame. It still has to agree with the text."""
    annotated = annotate(analyser_frame(0), signal_for((IN_BED_RECUMBENT,), **kwargs))
    edge = annotated[IN_BED_RECUMBENT.y1, IN_BED_RECUMBENT.x1 + 5]
    assert tuple(edge) == OVERLAY_COLOURS[expected]


# --------------------------------------------------------------- the whole script, end to end

#: What each scene of :data:`SCENARIO_SCRIPT` must be read as, exactly, at every frame size:
#: ``(frames, boxed, falls, exits, unseen, postures)``. Asserting the whole table rather than
#: a summary is what localises a regression - a change that starts calling the sitting scene
#: 'upright' is a bad threshold, not a bad frame.
SCRIPT_EXPECTATIONS: tuple[tuple[str, int, int, int, int, int, dict[str, int]], ...] = (
    ("absent", 24, 0, 0, 0, 17, {"unknown": 24}),
    ("settled", 170, 170, 0, 0, 0, {"recumbent": 170}),
    ("agitation", 60, 60, 0, 0, 0, {"recumbent": 60}),
    ("sitting", 45, 45, 0, 0, 0, {"seated": 45}),
    ("bed_exit", 55, 55, 0, 53, 0, {"upright": 55}),
    ("fall", 70, 70, 68, 0, 0, {"collapsed": 68, "recumbent": 2}),
    ("recovery", 50, 50, 0, 0, 0, {"recumbent": 50}),
)


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_the_script_is_run_to_its_end(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    run = script_runs[size]
    assert run.frames == sum(count for _, count in SCENARIO_SCRIPT) == 474
    assert set(run.total) == {scene for scene, _ in SCENARIO_SCRIPT}


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_the_nurse_is_paged_exactly_once_for_the_fall_and_once_for_the_exit(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """**The headline claim of the whole channel.** 474 frames of a patient settling, fidgeting,
    sitting up, climbing out, falling and being helped back produce exactly two escalations -
    and the count, not the accuracy of any one frame, is what makes a fall detector usable.

    Asserted at three resolutions because a detector that only behaves at the shipped default
    is a coincidence: the blob geometry has to survive a change of scale."""
    run = script_runs[size]
    assert run.fall_episodes == 1
    assert run.exit_episodes == 1


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_the_rule_this_replaced_would_have_cried_wolf_350_times(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """The measured cost of the original project's heuristic, on the same frames::

        if width / height > 2.2: fall_detected = True

    350 of the 474 frames have a largest-box aspect ratio above 2.2 - every settled frame,
    every agitated one, the whole recovery - because a patient asleep in bed is wider than they
    are tall. An alarm that fires through 74% of a quiet night is one the ward switches off,
    which is how a fall detector ends up costing lives rather than saving them."""
    assert script_runs[size].naive_alarms == 350
    assert script_runs[size].fall_episodes == 1


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
@pytest.mark.parametrize(
    ("scene", "frames", "boxed", "falls", "exits", "unseen", "postures"),
    SCRIPT_EXPECTATIONS,
    ids=[row[0] for row in SCRIPT_EXPECTATIONS],
)
def test_every_scene_is_read_the_way_it_was_staged(
    script_runs: dict[tuple[int, int], ScriptRun],
    size: tuple[int, int],
    scene: str,
    frames: int,
    boxed: int,
    falls: int,
    exits: int,
    unseen: int,
    postures: dict[str, int],
) -> None:
    """:data:`SCRIPT_EXPECTATIONS`, asserted exactly and identically at all three frame sizes.

    Two rows carry most of the weight. ``settled`` is the case the original project got wrong:
    170 frames of a patient lying down, every one of them boxed, none of them a fall. ``fall``
    is the case it got right by accident: 68 of 70 frames alarmed, the missing two being the
    persistence rule proving the finding before reporting it."""
    run = script_runs[size]
    assert run.total[scene] == frames
    assert run.boxed[scene] == boxed
    assert run.falls[scene] == falls
    assert run.exits[scene] == exits
    assert run.unseen[scene] == unseen
    assert dict(run.postures[scene]) == postures


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_the_patient_is_found_in_every_frame_they_are_present_in(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """Detection coverage, not just alarm accuracy: a detector that finds the patient in 70% of
    frames cannot support a persistence rule, however good its posture calls are. 450 of the
    474 frames have a patient in them, and all 450 are boxed."""
    run = script_runs[size]
    occupied = [scene for scene in run.total if scene != "absent"]
    assert sum(run.total[scene] for scene in occupied) == 450
    assert all(run.boxed[scene] == run.total[scene] for scene in occupied)
    assert all(run.unseen[scene] == 0 for scene in occupied)


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_the_empty_bay_at_the_start_is_recognised_as_empty(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """The script opens on an empty bay - both a clinical case (an unoccupied bed) and a
    technical necessity: a background model seeded on a frame the patient is already in can
    never see the patient, only their edges. The countdown is why 17 of the 24 frames read as
    absent rather than all of them - a dropped detection is not a discharge."""
    run = script_runs[size]
    assert run.boxed["absent"] == 0
    assert run.unseen["absent"] == run.total["absent"] - (ABSENCE_FRAMES - 1) == 17


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_nothing_alarms_through_the_scenes_staged_to_be_unremarkable(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """349 of the 474 frames should pass without a word: the empty bay, the settled patient, the
    agitated one, the patient sitting up *in* bed, and the recovery afterwards. Only the 125
    frames of the bed-exit and fall scenes are meant to escalate at all."""
    run = script_runs[size]
    quiet = ("absent", "settled", "agitation", "sitting", "recovery")
    assert sum(run.total[scene] for scene in quiet) == 349
    for scene in quiet:
        assert run.falls[scene] == 0, scene
        assert run.exits[scene] == 0, scene
    assert {scene for scene, count in run.falls.items() if count} == {"fall"}
    assert {scene for scene, count in run.exits.items() if count} == {"bed_exit"}


@pytest.mark.parametrize("size", FRAME_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_agitation_registers_more_motion_than_settling(
    script_runs: dict[tuple[int, int], ScriptRun], size: tuple[int, int]
) -> None:
    """The motion index feeds the agitation term in risk fusion, so it has to separate the two
    scenes staged to differ in exactly that - compared as medians, so one thrash in an otherwise
    settled scene cannot close the gap."""
    run = script_runs[size]
    settled = run.median_motion("settled")
    agitated = run.median_motion("agitation")
    assert 0.0 <= settled < agitated <= 1.0
    assert agitated > settled * 1.5
