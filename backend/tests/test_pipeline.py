"""Pipeline tests.

The report-aggregation tests are pure logic and always run. The end-to-end
test needs YOLO weights plus the generated clip, so it is marked `slow` and
only runs with `pytest --run-slow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.types import FrameResult, Track
from app.pipeline import PipelineReport, TrackStats, _annotate


class TestTrackStats:
    def test_span_counts_both_endpoints(self) -> None:
        stats = TrackStats(track_id=1, first_frame=10, last_frame=19)
        assert stats.span == 10


class TestPipelineReport:
    def test_empty_report(self) -> None:
        report = PipelineReport(source="x.mp4")
        assert report.unique_track_ids == 0
        assert "Frames processed    : 0" in "\n".join(report.summary_lines())

    def test_summary_lists_every_track(self) -> None:
        report = PipelineReport(source="x.mp4", frames_processed=100)
        report.tracks[2] = TrackStats(2, frame_count=40, first_frame=5, last_frame=60)
        report.tracks[1] = TrackStats(1, frame_count=90, first_frame=0, last_frame=99)

        text = "\n".join(report.summary_lines())
        assert report.unique_track_ids == 2
        # Sorted by track_id, so 1 must appear before 2.
        assert text.index("\n        1 ") < text.index("\n        2 ")


class TestAnnotate:
    def test_does_not_mutate_the_source_frame(self, blank_frame) -> None:
        original = blank_frame.copy()
        result = FrameResult(
            frame_index=3,
            timestamp_s=0.1,
            tracks=[Track(track_id=1, x1=50, y1=50, x2=150, y2=250)],
        )
        canvas = _annotate(blank_frame, result)
        assert (blank_frame == original).all()
        assert canvas.shape == blank_frame.shape
        assert not (canvas == blank_frame).all(), "nothing was drawn"

    def test_handles_a_frame_with_no_tracks(self, blank_frame) -> None:
        result = FrameResult(frame_index=0, timestamp_s=0.0)
        assert _annotate(blank_frame, result).shape == blank_frame.shape

    def test_label_stays_on_canvas_for_a_track_at_the_top_edge(
        self, blank_frame
    ) -> None:
        """A person entering at the top of frame must still show their ID.

        The label normally sits above the box; at y1=0 there is no room, so it
        has to flip inside the box. Drawn off-canvas, OpenCV clips it silently
        and the track renders with no visible ID at all.
        """
        result = FrameResult(
            frame_index=0,
            timestamp_s=0.0,
            tracks=[Track(track_id=4, x1=100, y1=0, x2=200, y2=300)],
        )
        canvas = _annotate(blank_frame, result)

        # The label band must be painted inside the box, just below its top.
        band = canvas[0:30, 100:200]
        assert (band != 128).any(), "label band was clipped off-canvas"

    def test_label_stays_on_canvas_for_a_track_at_the_right_edge(
        self, blank_frame
    ) -> None:
        width = blank_frame.shape[1]
        result = FrameResult(
            frame_index=0,
            timestamp_s=0.0,
            tracks=[Track(track_id=5, x1=width - 10, y1=100, x2=width, y2=300)],
        )
        # Must not raise, and must draw something visible near the edge.
        canvas = _annotate(blank_frame, result)
        assert (canvas[80:100, width - 80 : width] != 128).any()


@pytest.mark.slow
class TestEndToEnd:
    def test_detects_and_tracks_people(self, synthetic_people_video: Path) -> None:
        from app.pipeline import DetectionTrackingPipeline

        settings = get_settings()
        settings.video.max_frames = 60
        settings.video.save_annotated = False

        report = DetectionTrackingPipeline(settings).run(synthetic_people_video)

        assert report.frames_processed == 60
        assert report.total_detections > 0, "YOLO found no people in the clip"
        assert report.unique_track_ids > 0, "DeepSORT confirmed no tracks"
        # A confirmed track must survive at least n_init frames.
        longest = max(stats.frame_count for stats in report.tracks.values())
        assert longest >= settings.tracking.n_init


class TestAnnotatedVideoTiming:
    """LOG-07: the writer invented a frame rate the reader had already measured."""

    def test_the_writer_uses_the_source_frame_rate(self, tmp_path, monkeypatch) -> None:
        import cv2
        import numpy as np

        from app.core.config import get_settings
        from app.pipeline import DetectionTrackingPipeline

        # 25 fps: the case that played 20% fast under the old fixed 30.0.
        source = tmp_path / "25fps.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48)
        )
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

        settings = get_settings()
        monkeypatch.setattr(settings.paths, "output_dir", tmp_path)

        pipeline = DetectionTrackingPipeline.__new__(DetectionTrackingPipeline)
        pipeline.settings = settings
        pipeline._source_fps = 25.0

        opened = {}

        class FakeWriter:
            def isOpened(self):
                return True

        def capture(path, fourcc, fps, size):
            opened["fps"] = fps
            return FakeWriter()

        monkeypatch.setattr(cv2, "VideoWriter", capture)
        monkeypatch.setattr(settings.video, "frame_stride", 1)
        pipeline._open_writer(source, (48, 64, 3))
        assert opened["fps"] == 25.0

        # Stride still divides it, so the output plays in real time.
        monkeypatch.setattr(settings.video, "frame_stride", 5)
        pipeline._open_writer(source, (48, 64, 3))
        assert opened["fps"] == 5.0
