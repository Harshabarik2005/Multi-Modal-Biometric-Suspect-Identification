"""Phase 1 pipeline: read video -> detect people -> track them across frames.

This is the spine every later phase plugs into. Phases 2-4 will consume the
per-track crops this produces (`Track.crop(frame)`) to build face, gait and
re-ID embeddings; Phase 6 fuses them. For now it only proves detection and
tracking work end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.types import FrameResult
from app.core.video import VideoReader
from app.detection.yolo_detector import YOLOPersonDetector
from app.tracking.deepsort_tracker import DeepSortTracker

logger = get_logger(__name__)


@dataclass
class TrackStats:
    """Lifetime summary of one track, for the end-of-run report."""

    track_id: int
    frame_count: int = 0
    first_frame: int = 0
    last_frame: int = 0
    max_box_height: float = 0.0

    @property
    def span(self) -> int:
        """Frames between first and last sighting, gaps included."""
        return self.last_frame - self.first_frame + 1


@dataclass
class PipelineReport:
    """What a full run produced. Printed by the CLI, asserted on in tests."""

    source: str
    frames_processed: int = 0
    total_detections: int = 0
    tracks: dict[int, TrackStats] = field(default_factory=dict)
    annotated_path: Path | None = None

    @property
    def unique_track_ids(self) -> int:
        return len(self.tracks)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Source              : {self.source}",
            f"Frames processed    : {self.frames_processed}",
            f"Person detections   : {self.total_detections}",
            f"Unique track IDs    : {self.unique_track_ids}",
        ]
        if self.annotated_path:
            lines.append(f"Annotated video     : {self.annotated_path}")
        if self.tracks:
            lines.append("")
            header = (
                f"{'track_id':>9}  {'frames':>7}  {'first':>7}  "
                f"{'last':>7}  {'span':>6}  {'max_h':>6}"
            )
            lines.append(header)
            for stats in sorted(self.tracks.values(), key=lambda s: s.track_id):
                lines.append(
                    f"{stats.track_id:>9}  {stats.frame_count:>7}  "
                    f"{stats.first_frame:>7}  {stats.last_frame:>7}  "
                    f"{stats.span:>6}  {stats.max_box_height:>6.0f}"
                )
        return lines


# Distinct, high-contrast BGR colors cycled per track ID.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _annotate(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Draw track boxes + IDs onto a copy of the frame."""
    canvas = frame.copy()
    for track in result.tracks:
        x1, y1, x2, y2 = (int(v) for v in track.xyxy)
        color = _color_for(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track.track_id}"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        # Labels sit above the box, but a person entering at the top of the
        # frame has no room there -- flip inside the box rather than drawing
        # off-canvas, where OpenCV would silently clip the ID away.
        band_h = text_h + 8
        if y1 - band_h >= 0:
            band_top, text_baseline = y1 - band_h, y1 - 5
        else:
            band_top, text_baseline = y1, y1 + text_h + 3
        band_left = max(0, min(x1, canvas.shape[1] - text_w - 6))

        cv2.rectangle(
            canvas,
            (band_left, band_top),
            (band_left + text_w + 6, band_top + band_h),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (band_left + 3, text_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"frame {result.frame_index}  tracks {len(result.tracks)}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


class DetectionTrackingPipeline:
    """Detector + tracker wired to a video source.

    Models load once at construction, so reuse one instance across videos --
    `run` resets the tracker each time so IDs restart at 1.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.device = self.settings.resolve_device()
        logger.info("Pipeline device: %s", self.device)

        self.detector = YOLOPersonDetector(self.settings, device=self.device)
        self.tracker = DeepSortTracker(self.settings, device=self.device)
        #: Frame rate of the source currently being streamed. Set by `stream`.
        self._source_fps = 30.0

    def stream(
        self, source: str | int | Path
    ) -> Iterator[tuple[FrameResult, np.ndarray]]:
        """Yield `(FrameResult, frame)` for each processed frame.

        Use this when a later phase needs the pixels as well as the boxes.
        """
        video_cfg = self.settings.video
        self.tracker.reset()

        with VideoReader(
            source,
            frame_stride=video_cfg.frame_stride,
            max_frames=video_cfg.max_frames,
        ) as reader:
            # Kept so the annotated writer can use the source's real frame
            # rate. VideoReader already substitutes 30 for a file that reports
            # a nonsense one, so this is never zero.
            self._source_fps = reader.meta.fps
            logger.info(
                "Opened %s (%dx%d @ %.2f fps, %s frames)",
                source,
                reader.meta.width,
                reader.meta.height,
                reader.meta.fps,
                reader.meta.frame_count or "unknown",
            )
            for frame_index, timestamp_s, frame in reader:
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                yield (
                    FrameResult(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        detections=detections,
                        tracks=tracks,
                    ),
                    frame,
                )

    def run(self, source: str | int | Path) -> PipelineReport:
        """Process a whole source and return aggregate stats."""
        report = PipelineReport(source=str(source))
        progress_every = self.settings.logging.progress_every
        writer: cv2.VideoWriter | None = None

        try:
            for result, frame in self.stream(source):
                report.frames_processed += 1
                report.total_detections += len(result.detections)

                for track in result.tracks:
                    stats = report.tracks.get(track.track_id)
                    if stats is None:
                        stats = TrackStats(
                            track_id=track.track_id,
                            first_frame=result.frame_index,
                        )
                        report.tracks[track.track_id] = stats
                        logger.info(
                            "New track %d at frame %d",
                            track.track_id,
                            result.frame_index,
                        )
                    stats.frame_count += 1
                    stats.last_frame = result.frame_index
                    stats.max_box_height = max(stats.max_box_height, track.height)

                if self.settings.video.save_annotated:
                    if writer is None:
                        writer = self._open_writer(source, frame.shape)
                        report.annotated_path = self._annotated_path(source)
                    writer.write(_annotate(frame, result))

                if report.frames_processed % progress_every == 0:
                    logger.info(
                        "frame %d | detections %d | active tracks %d | ids seen %d",
                        result.frame_index,
                        len(result.detections),
                        len(result.tracks),
                        report.unique_track_ids,
                    )
        finally:
            if writer is not None:
                writer.release()

        return report

    def _annotated_path(self, source: str | int | Path) -> Path:
        out_dir = self.settings.paths.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
        )
        return out_dir / f"{stem}_tracked.mp4"

    def _open_writer(
        self, source: str | int | Path, frame_shape: tuple[int, ...]
    ) -> cv2.VideoWriter:
        path = self._annotated_path(source)
        height, width = frame_shape[:2]

        # The source's own frame rate, divided by the stride, so the annotated
        # video plays at the speed the footage was shot at.
        #
        # This used to assume 30 fps regardless of what the file said, which
        # the reader had already measured. 25 fps footage played 20% fast and
        # 60 fps at half speed -- and because frame numbers are burnt into the
        # annotation, the timestamps stopped agreeing with where they appear in
        # the clip. On a recording that has to stand up as evidence, that is
        # not a cosmetic difference.
        fps = self._source_fps / max(1, self.settings.video.frame_stride)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        logger.info("Writing annotated video to %s", path)
        return writer
