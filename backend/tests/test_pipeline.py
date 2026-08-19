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
