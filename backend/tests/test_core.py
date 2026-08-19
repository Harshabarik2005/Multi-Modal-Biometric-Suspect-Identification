"""Tests for config loading, shared types, and the video reader.

None of these need model weights, so they run on every `pytest` invocation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.core.types import Detection, FrameResult, Track
from app.core.video import VideoReader


class TestConfig:
    def test_defaults_load(self) -> None:
        settings = get_settings()
        assert settings.project_name
        assert settings.detection.person_class_id == 0
        assert 0.0 <= settings.detection.conf_threshold <= 1.0

    def test_paths_are_absolute_after_load(self) -> None:
        settings = get_settings()
        assert settings.paths.data_dir.is_absolute()
        assert settings.paths.models_dir.is_absolute()

    def test_yaml_overrides_defaults(self, tmp_path: Path) -> None:
        config = tmp_path / "custom.yaml"
        config.write_text(
            "detection:\n  conf_threshold: 0.9\n  model: yolov8s.pt\n",
            encoding="utf-8",
        )
        settings = get_settings(config)
        assert settings.detection.conf_threshold == 0.9
        assert settings.detection.model == "yolov8s.pt"

    def test_env_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRS_DETECTION__CONF_THRESHOLD", "0.75")
        get_settings.cache_clear()
        try:
            assert get_settings().detection.conf_threshold == 0.75
        finally:
            get_settings.cache_clear()

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError):
            Settings(detection={"conf_threshold": 1.5})

    def test_resolve_device_returns_concrete_value(self) -> None:
        assert get_settings().resolve_device() in {"cuda", "cpu"}


class TestDetection:
    def test_geometry(self) -> None:
        det = Detection(x1=10, y1=20, x2=50, y2=120, confidence=0.9)
        assert det.width == 40
        assert det.height == 100
        assert det.xyxy == (10, 20, 50, 120)
        assert det.ltwh == (10, 20, 40, 100)

    def test_crop(self, blank_frame: np.ndarray) -> None:
        det = Detection(x1=100, y1=50, x2=200, y2=250, confidence=0.8)
        crop = det.crop(blank_frame)
        assert crop.shape == (200, 100, 3)

    def test_crop_clamps_to_frame_bounds(self, blank_frame: np.ndarray) -> None:
        det = Detection(x1=-50, y1=-50, x2=900, y2=900, confidence=0.8)
        crop = det.crop(blank_frame)
        assert crop.shape == blank_frame.shape

    def test_crop_of_degenerate_box_is_empty(self, blank_frame: np.ndarray) -> None:
        det = Detection(x1=700, y1=500, x2=800, y2=600, confidence=0.8)
        assert det.crop(blank_frame).size == 0


class TestTrack:
    def test_geometry_and_crop(self, blank_frame: np.ndarray) -> None:
        track = Track(track_id=7, x1=0, y1=0, x2=64, y2=128)
        assert track.width == 64
        assert track.height == 128
        assert track.crop(blank_frame).shape == (128, 64, 3)


class TestFrameResult:
    def test_defaults_are_independent_lists(self) -> None:
        first = FrameResult(frame_index=0, timestamp_s=0.0)
        second = FrameResult(frame_index=1, timestamp_s=0.04)
        first.detections.append(Detection(0, 0, 1, 1, 0.5))
        assert second.detections == []


class TestVideoReader:
    def test_reads_every_frame(self, tiny_video: Path) -> None:
        with VideoReader(tiny_video) as reader:
            frames = list(reader)
        assert len(frames) == 10
        assert [index for index, _, _ in frames] == list(range(10))

    def test_metadata(self, tiny_video: Path) -> None:
        with VideoReader(tiny_video) as reader:
            assert reader.meta.width == 160
            assert reader.meta.height == 120
            assert reader.meta.fps > 0

    def test_stride_skips_frames_but_keeps_true_indices(
        self, tiny_video: Path
    ) -> None:
        with VideoReader(tiny_video, frame_stride=3) as reader:
            indices = [index for index, _, _ in reader]
        assert indices == [0, 3, 6, 9]

    def test_max_frames_caps_output(self, tiny_video: Path) -> None:
        with VideoReader(tiny_video, max_frames=4) as reader:
            assert len(list(reader)) == 4

    def test_timestamps_follow_source_fps(self, tiny_video: Path) -> None:
        with VideoReader(tiny_video, frame_stride=2) as reader:
            fps = reader.meta.fps
            stamps = [ts for _, ts, _ in reader]
        assert stamps[0] == pytest.approx(0.0)
        assert stamps[1] == pytest.approx(2 / fps)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nope.mp4")

    def test_invalid_stride_raises(self, tiny_video: Path) -> None:
        with pytest.raises(ValueError):
            VideoReader(tiny_video, frame_stride=0)
