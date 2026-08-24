"""Turning uploaded media into observations the embedding branches can use.

Enrollment previously accepted only video, because the pipeline's natural input
is a tracked person moving through frames. Uploading photographs is a different
shape: each image is independent, there is nothing to track, and the person has
to be located in each one separately.

Both paths end at the same place -- a list of `TrackObservation` -- so
everything downstream is unchanged.

What each signal actually needs
-------------------------------
This is the part worth getting right in a UI, because the three modalities do
NOT have the same requirements and a profile can look complete while being
useless:

* **Face** -- photos or video. Needs the face visible and roughly frontal, and
  enough pixels (about 112px of face height). Several angles beat one good one.
* **Appearance (re-ID)** -- photos or video. Needs the whole body in frame.
* **Gait** -- **video only, of the person walking.** A photograph contains no
  gait information whatsoever, and neither does video of someone standing
  still. It needs a couple of seconds of walking, ideally viewed from the side.

`assess_readiness` reports which of these an upload actually satisfies, so the
interface can say "this will match on face and appearance but not gait" rather
than quietly storing a partial profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.track_buffer import TrackBufferStore
from app.core.types import Modality, TrackObservation
from app.detection.yolo_detector import YOLOPersonDetector
from app.pipeline import DetectionTrackingPipeline

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image file, or return None if it is not one.

    Not `cv2.imread`: importing ultralytics replaces it with a wrapper that
    dereferences the decoded array before checking it, so an undecodable file
    raises AttributeError from deep inside a library instead of returning None.
    An upload is untrusted input -- "this file is not an image" is a normal
    outcome to report, not a crash. Reading the bytes ourselves also sidesteps
    imread's inability to open paths that are not ASCII.
    """
    try:
        buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"


@dataclass
class MediaSummary:
    """What a set of uploads yielded, and what it did not."""

    images: int = 0
    videos: int = 0
    observations: int = 0
    frames_seen: int = 0
    tracks_found: int = 0
    #: Files that produced nothing, with the reason.
    rejected: list[tuple[str, str]] = field(default_factory=list)
    #: True when at least one video contributed, so gait is even conceivable.
    has_motion_source: bool = False

    #: Observations that came from video rather than photographs. Gait must be
    #: judged on this, never on `observations`: counting photographs toward a
    #: frame requirement made 20 photos plus a 5-frame video report "gait
    #: ready: 25 continuous frames", which is exactly the mistake the readiness
    #: check exists to prevent.
    video_observations: int = 0
    #: Longest run of consecutive frames from a SINGLE video. Gait reads a
    #: sequence, and two clips concatenated are not one -- the cadence and GEI
    #: would be computed across the join between separate recordings.
    longest_video_run: int = 0
    #: Index range of that run within the returned observations, so callers can
    #: hand gait a genuinely contiguous sequence.
    gait_segment: tuple[int, int] = (0, 0)


def observations_from_images(
    paths: list[Path], settings: Settings, detector: YOLOPersonDetector | None = None
) -> tuple[list[TrackObservation], MediaSummary]:
    """One observation per photograph, using the largest person detected.

    No tracking: each photograph is an independent sighting. The largest
    detection is taken because an enrollment photo is of one person, and if
    someone else is in the background they are smaller and not the subject.
    """
    summary = MediaSummary()
    detector = detector or YOLOPersonDetector(settings)
    observations: list[TrackObservation] = []

    for index, path in enumerate(paths):
        image = read_image(path)
        if image is None:
            summary.rejected.append((path.name, "could not be decoded as an image"))
            continue

        summary.images += 1
        summary.frames_seen += 1

        detections = detector.detect(image)
        if not detections:
            summary.rejected.append(
                (path.name, "no person detected -- is the whole body in frame?")
            )
            continue

        largest = max(detections, key=lambda d: d.height)
        crop = largest.crop(image)
        if crop.size == 0:
            summary.rejected.append((path.name, "person box was empty after cropping"))
            continue

        observations.append(
            TrackObservation(
                frame_index=index,
                # Photographs have no real timeline. Spacing them a second
                # apart keeps timestamps monotonic without implying they are
                # consecutive frames -- which matters because gait would read
                # a fake cadence out of them otherwise.
                timestamp_s=float(index),
                crop=_downscale(crop, settings.track_buffer.store_height),
                box_height=largest.height,
                detection_confidence=largest.confidence,
            )
        )

    summary.observations = len(observations)
    return observations, summary


#: Enrolment keeps far more frames per track than live matching, but not an
#: unbounded number. The original code lifted the cap to 100,000, which with a
#: 500MB upload allowance is tens of thousands of 256px crops in RAM -- exactly
#: the exhaustion `track_buffer.py` exists to prevent, disabled on the one path
#: that takes untrusted input. 600 frames is 24 seconds at 25fps, ample for a
#: reference, and bounded.
ENROLMENT_MAX_OBSERVATIONS = 600
ENROLMENT_MAX_TRACKS = 20


def observations_from_video(
    path: Path, settings: Settings, pipeline: DetectionTrackingPipeline | None = None
) -> tuple[list[TrackObservation], MediaSummary]:
    """Observations for the dominant track in one video.

    Uses the longest-lived track, on the assumption that enrollment footage is
    of one person. If several people are present this reports it in the summary
    rather than silently blending them, because a reference vector built from
    two people produces confident wrong matches forever after.
    """
    summary = MediaSummary(videos=1, has_motion_source=True)

    # Build an explicit buffer config rather than mutating `settings`. The
    # settings object is a process-wide singleton (`get_settings` is cached),
    # and `TrackBufferStore` holds a live reference to it -- so temporarily
    # rewriting it here changed the eviction bounds of any scan running
    # concurrently, then changed them back underneath it.
    buffer_config = settings.track_buffer.model_copy(
        update={
            "max_observations": ENROLMENT_MAX_OBSERVATIONS,
            "max_tracks": ENROLMENT_MAX_TRACKS,
        }
    )

    pipeline = pipeline or DetectionTrackingPipeline(settings)
    store = TrackBufferStore(settings, config=buffer_config)
    for result, frame in pipeline.stream(path):
        store.update(result, frame)
        summary.frames_seen += 1

    buffers = sorted(store, key=lambda b: len(b), reverse=True)
    summary.tracks_found = len(buffers)

    if not buffers:
        summary.rejected.append(
            (path.name, "no person was tracked -- check the footage shows a person")
        )
        return [], summary

    observations = list(buffers[0])
    summary.observations = len(observations)
    summary.video_observations = len(observations)
    summary.longest_video_run = len(observations)
    summary.gait_segment = (0, len(observations))
    return observations, summary


def observations_from_uploads(
    paths: list[Path], settings: Settings
) -> tuple[list[TrackObservation], MediaSummary]:
    """Handle a mixed batch of photos and videos.

    Video observations come first, then photos, and the frame indices are
    renumbered so they stay monotonic. Ordering matters to gait, which reads a
    sequence; photos appended at the end carry timestamps far past the video's,
    so gait's cadence detection sees them as a gap rather than as motion.
    """
    combined = MediaSummary()
    videos = [p for p in paths if kind_of(p) == "video"]
    images = [p for p in paths if kind_of(p) == "image"]

    for path in paths:
        if kind_of(path) == "unknown":
            combined.rejected.append((path.name, f"unsupported file type {path.suffix}"))

    observations: list[TrackObservation] = []

    for path in videos:
        video_observations, summary = observations_from_video(path, settings)

        # Renumber onto the end of what came before. Each video's frame indices
        # start at 0, so concatenating them raw produced [0,1,2,3,0,1,2,3] --
        # not monotonic, despite this function's docstring promising it was.
        #
        # Frame indices are counters and may be renumbered freely. Timestamps
        # are NOT: they are the only record of how fast the footage was shot,
        # and gait recovers cadence from them (LOG-06). This used to assign
        # `float(offset + index)` to both -- one second per frame, whatever the
        # camera actually did. Gait needs at least 3 samples across a 0.28s
        # half cycle; at the resulting 1Hz it got 0.28, so every enrolment
        # through this function refused gait with "frames too far apart to
        # measure cadence". Not some enrolments -- every one, for everybody,
        # which is why no watchlist entry has ever held a gait template.
        #
        # So: shift each clip's real timestamps to sit after the previous
        # clip's, and leave the spacing within a clip exactly as recorded.
        offset = (observations[-1].frame_index + 1) if observations else 0
        time_offset = (observations[-1].timestamp_s + 1.0) if observations else 0.0
        clip_start_s = video_observations[0].timestamp_s if video_observations else 0.0
        start = len(observations)
        for index, observation in enumerate(video_observations):
            observation.frame_index = offset + index
            observation.timestamp_s = time_offset + (
                observation.timestamp_s - clip_start_s
            )
        observations.extend(video_observations)

        # Track the longest single-video run, and where it sits. Gait reads a
        # sequence; a run spanning two recordings, potentially at different
        # frame rates and of different people, is not one.
        if len(video_observations) > combined.longest_video_run:
            combined.longest_video_run = len(video_observations)
            combined.gait_segment = (start, start + len(video_observations))

        combined.videos += summary.videos
        combined.frames_seen += summary.frames_seen
        combined.video_observations += len(video_observations)
        combined.tracks_found = max(combined.tracks_found, summary.tracks_found)
        combined.rejected.extend(summary.rejected)
        combined.has_motion_source = combined.has_motion_source or summary.has_motion_source

    if images:
        image_observations, summary = observations_from_images(images, settings)
        offset = (observations[-1].frame_index + 1) if observations else 0
        # A second apart is arbitrary, and honestly so: photographs are
        # independent sightings with no rate between them. What matters is that
        # they land after the video and stay ordered. Gait never reads them --
        # `gait_segment` covers one video only -- so no cadence is derived from
        # this spacing. Offset in seconds rather than by frame index, so a long
        # upload cannot leave a photo timestamped before the footage it follows.
        time_offset = (observations[-1].timestamp_s + 1.0) if observations else 0.0
        for index, observation in enumerate(image_observations):
            observation.frame_index = offset + index
            observation.timestamp_s = time_offset + float(index)
        observations.extend(image_observations)
        combined.images += summary.images
        combined.frames_seen += summary.frames_seen
        combined.rejected.extend(summary.rejected)

    combined.observations = len(observations)
    return observations, combined


def gait_observations(
    observations: list[TrackObservation], summary: MediaSummary
) -> list[TrackObservation]:
    """The subset of `observations` gait may legitimately read.

    One contiguous run from a single video. Photographs are excluded because
    they carry no gait, and separate videos are not spliced together because
    the cadence would be computed across the join.
    """
    start, end = summary.gait_segment
    return observations[start:end] if end > start else []


@dataclass
class ModalityReadiness:
    """Whether an upload can support one modality, and what is missing."""

    modality: Modality
    ready: bool
    reason: str
    requirement: str


def assess_readiness(
    summary: MediaSummary, settings: Settings
) -> list[ModalityReadiness]:
    """What this upload can and cannot support, before any embedding runs.

    Cheap and predictive: it looks at what was uploaded, not at what the models
    produced. The point is to tell someone *while they are still at the
    computer* that they need a walking video, rather than after enrollment
    quietly stores a face-only profile.
    """
    checks: list[ModalityReadiness] = []
    minimum_gait = settings.gait.min_frames

    checks.append(
        ModalityReadiness(
            modality=Modality.FACE,
            ready=summary.observations > 0,
            reason=(
                f"{summary.observations} usable image(s) of the person"
                if summary.observations
                else "no person was found in any file"
            ),
            requirement=(
                "Photos or video with the face visible and roughly frontal. "
                "Several angles are better than one. The face needs enough "
                "pixels -- a distant or blurred face will not enroll."
            ),
        )
    )

    # Judged on the longest single-video run, never on total observations.
    # Counting photographs here reported "gait ready" for an upload of 20
    # photos and a 5-frame clip -- the precise failure this check exists to
    # catch, in the check built to catch it.
    usable_gait_frames = summary.longest_video_run
    gait_ready = summary.has_motion_source and usable_gait_frames >= minimum_gait

    if not summary.has_motion_source:
        gait_reason = "photos only -- a still image contains no gait information"
    elif usable_gait_frames < minimum_gait:
        gait_reason = (
            f"the longest single clip gave {usable_gait_frames} tracked frames; "
            f"gait needs at least {minimum_gait} of continuous walking"
        )
        if summary.images:
            gait_reason += " (photographs do not count toward this)"
    else:
        gait_reason = (
            f"{usable_gait_frames} continuous frames from one clip"
        )

    checks.append(
        ModalityReadiness(
            modality=Modality.GAIT,
            ready=gait_ready,
            reason=gait_reason,
            requirement=(
                "VIDEO ONLY, of the person WALKING -- roughly two seconds or "
                "more, whole body in frame, ideally viewed from the side. "
                "Standing still or turning on the spot produces no gait signal."
            ),
        )
    )

    checks.append(
        ModalityReadiness(
            modality=Modality.REID,
            ready=summary.observations > 0,
            reason=(
                f"{summary.observations} usable image(s) of the person"
                if summary.observations
                else "no person was found in any file"
            ),
            requirement=(
                "Photos or video with the whole body visible. Note that this "
                "signal largely describes clothing, so it goes stale -- "
                "re-enroll if the reference needs to stay current."
            ),
        )
    )

    return checks


def _downscale(crop: np.ndarray, target_height: int) -> np.ndarray:
    """Match the buffering path, so photo and video crops are the same size."""
    height = crop.shape[0]
    if height <= target_height:
        return crop.copy()
    scale = target_height / height
    width = max(1, int(round(crop.shape[1] * scale)))
    return cv2.resize(crop, (width, target_height), interpolation=cv2.INTER_AREA)
