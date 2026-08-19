"""Shared data structures passed between pipeline stages.

Keeping these in one place means the detector, tracker and (later) the
embedding branches agree on a single box convention: `xyxy` in absolute pixel
coordinates of the source frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Detection:
    """One person detected in one frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def ltwh(self) -> tuple[float, float, float, float]:
        """Left-top-width-height, the format deep-sort-realtime expects."""
        return self.x1, self.y1, self.width, self.height

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Clamped crop of this box out of `frame` (BGR)."""
        h, w = frame.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]


@dataclass(slots=True)
class Track:
    """A person followed across frames, identified by a stable `track_id`."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    # Frames this track has existed without a matching detection.
    time_since_update: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return Detection(self.x1, self.y1, self.x2, self.y2, self.confidence).crop(frame)


@dataclass(slots=True)
class FrameResult:
    """Everything the Phase-1 pipeline knows about a single processed frame."""

    frame_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
