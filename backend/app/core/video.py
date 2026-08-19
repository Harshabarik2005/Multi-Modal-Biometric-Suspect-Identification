"""Video source reading.

`VideoReader` handles both files and live camera indices, applies the
configured frame stride, and reports the true frame index / timestamp of every
frame it yields (so a stride of 5 still gives correct wall-clock times).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int  # 0 when unknown (live cameras, some streams)


class VideoReader:
    """Iterates frames from a video file, camera index, or stream URL.

    Yields `(frame_index, timestamp_seconds, frame_bgr)` where `frame_index`
    counts frames in the *source*, not frames emitted.
    """

    def __init__(
        self,
        source: str | int | Path,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        self.source = source if isinstance(source, int) else str(source)
        self.frame_stride = frame_stride
        self.max_frames = max_frames

        if isinstance(self.source, str) and not self.source.startswith(
            ("rtsp://", "http://", "https://")
        ):
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Video source not found: {self.source}")

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.meta = VideoMeta(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Some containers report 0 or NaN; fall back to a sane default.
            fps=float(fps) if fps and fps > 0 else 30.0,
            frame_count=max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        frame_index = 0
        emitted = 0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                if frame_index % self.frame_stride == 0:
                    yield frame_index, frame_index / self.meta.fps, frame
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break

                frame_index += 1
        finally:
            self.release()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
