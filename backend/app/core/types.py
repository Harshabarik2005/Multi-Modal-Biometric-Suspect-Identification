"""Shared data structures passed between pipeline stages.

Keeping these in one place means the detector, tracker and (later) the
embedding branches agree on a single box convention: `xyxy` in absolute pixel
coordinates of the source frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


# ---------------------------------------------------------------------------
# Modality embeddings (Phases 2-4) and their fusion (Phases 5-6)
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """The three signals this system fuses."""

    FACE = "face"
    GAIT = "gait"
    REID = "reid"


@dataclass(slots=True)
class ModalityEmbedding:
    """One modality's read on one person.

    `vector` is None when the modality had nothing to say -- no face visible,
    too few frames for a gait cycle. That is the *normal* case in this system,
    not an error: the whole premise is that face frequently fails and the other
    branches carry the identification. Callers must check `has_signal` rather
    than assuming a vector exists.

    `quality` in [0, 1] is how much this observation should be trusted. It is
    not a match score -- it says "this is a good look at the person", not "this
    is the person". The Phase-6 attention head consumes it, so a branch that
    returns a constant quality silently disables the adaptive weighting that is
    the point of the project.
    """

    modality: Modality
    vector: np.ndarray | None = None
    quality: float = 0.0
    # How many frames actually contributed. Distinguishes a confident read of
    # 40 frames from a lucky single frame at the same quality.
    frames_used: int = 0
    # Free-form per-branch detail, surfaced in the explainability view:
    # face stores yaw/pitch, gait stores cycles detected, etc.
    detail: dict[str, float] = field(default_factory=dict)
    # Which model produced this vector, e.g. "osnet_x1_0/msmt17".
    #
    # An embedding is only meaningful inside the space its model defines.
    # Cosine similarity between vectors from two different models is not a
    # weak signal, it is noise -- and it is noise that looks exactly like a
    # score. Recording the model is what lets a stored reference and a live
    # probe be checked for having come from the same one (DES-01). Empty
    # means unknown, which is the case for anything enrolled before this
    # existed.
    model_id: str = ""

    @property
    def has_signal(self) -> bool:
        return self.vector is not None and self.vector.size > 0

    @classmethod
    def empty(cls, modality: Modality, reason: str = "") -> "ModalityEmbedding":
        """A clean 'nothing to report' result."""
        return cls(
            modality=modality,
            vector=None,
            quality=0.0,
            frames_used=0,
            detail={"no_signal": 1.0} if reason else {},
        )

    def similarity(self, other: "ModalityEmbedding") -> float | None:
        """Cosine similarity against another embedding of the same modality.

        Returns None when either side has no signal, so "could not compare"
        never silently collapses into "compared and got zero" -- those mean
        very different things to the fusion stage.
        """
        if self.modality is not other.modality:
            raise ValueError(
                f"Cannot compare {self.modality} with {other.modality}"
            )
        if not self.has_signal or not other.has_signal:
            return None
        return cosine_similarity(self.vector, other.vector)


@dataclass(slots=True)
class TrackObservation:
    """One frame's look at one tracked person.

    The unit the embedding branches consume. Holds the crop rather than the
    whole frame: a busy scene has many tracks, and keeping full frames per
    track exhausts memory fast.
    """

    frame_index: int
    timestamp_s: float
    crop: np.ndarray
    box_height: float
    detection_confidence: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe against zero norms."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length. A zero vector is returned unchanged."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm
