"""Per-track observation buffering.

The build plan calls for a "per-track frame buffer" feeding the embedding
branches. Written naively that is an unbounded dict of lists of crops, which is
the fastest way to exhaust memory on this machine: a busy camera holds twenty
tracks, each accumulating a full-resolution crop every frame, for as long as
the feed runs.

So this buffer is bounded in three directions at once:

* **Per track** -- a ring of the most recent `max_observations` frames.
* **Per crop** -- crops are downscaled to `store_height` on the way in.
* **Across tracks** -- at most `max_tracks` are held; the least recently seen
  is evicted first.

Keeping the *most recent* frames rather than the first is deliberate: a person
walking towards a camera gets larger and clearer, so recent frames are usually
the better ones, and a track that has just been re-identified after occlusion
should be judged on what it looks like now.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Iterator

import cv2
import numpy as np

from app.core.config import Settings, TrackBufferSettings, get_settings
from app.core.logging import get_logger
from app.core.types import FrameResult, TrackObservation

logger = get_logger(__name__)


class TrackBuffer:
    """Rolling window of observations for a single tracked person."""

    def __init__(self, track_id: int, max_observations: int) -> None:
        self.track_id = track_id
        self.observations: deque[TrackObservation] = deque(maxlen=max_observations)
        self.first_frame: int | None = None
        self.last_frame: int = -1
        #: Frame index at which this track was last matched, so the pipeline
        #: can rate-limit re-matching without re-deriving it.
        self.last_matched_frame: int | None = None
        #: Total observations ever added, including those aged out of the ring.
        self.total_seen: int = 0

    def add(self, observation: TrackObservation) -> None:
        if self.first_frame is None:
            self.first_frame = observation.frame_index
        self.last_frame = observation.frame_index
        self.total_seen += 1
        self.observations.append(observation)

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[TrackObservation]:
        return iter(self.observations)

    @property
    def is_empty(self) -> bool:
        return not self.observations

    def best(self, n: int) -> list[TrackObservation]:
        """The `n` largest observations, as a cheap proxy for the clearest.

        Box height is not a quality score -- the branches compute their own --
        but it is a reliable way to pick which frames are worth running an
        expensive model on when you cannot afford to run it on all of them.
        """
        return sorted(self.observations, key=lambda o: o.box_height, reverse=True)[:n]


class TrackBufferStore:
    """All live track buffers, bounded in total size."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: TrackBufferSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = config or self.settings.track_buffer
        # OrderedDict as an LRU: most recently updated moves to the end.
        self._buffers: OrderedDict[int, TrackBuffer] = OrderedDict()

    # -- ingestion ---------------------------------------------------------

    def update(self, result: FrameResult, frame: np.ndarray) -> list[int]:
        """Buffer this frame's tracks. Returns the track IDs that were updated."""
        updated: list[int] = []

        for track in result.tracks:
            if track.height < self.cfg.min_box_height:
                # Too small to yield a usable face or body crop; skip rather
                # than spend memory on something no branch can read.
                continue

            crop = track.crop(frame)
            if crop.size == 0:
                continue

            observation = TrackObservation(
                frame_index=result.frame_index,
                timestamp_s=result.timestamp_s,
                crop=self._downscale(crop),
                box_height=track.height,
                detection_confidence=track.confidence,
            )
            self._buffer_for(track.track_id).add(observation)
            self._buffers.move_to_end(track.track_id)
            updated.append(track.track_id)

        self._evict()
        return updated

    def _downscale(self, crop: np.ndarray) -> np.ndarray:
        """Shrink tall crops to `store_height`, preserving aspect ratio.

        Crops are never *upscaled* -- inventing pixels would only inflate
        memory and mislead the quality scores downstream.
        """
        height = crop.shape[0]
        if height <= self.cfg.store_height:
            return crop.copy()

        scale = self.cfg.store_height / height
        width = max(1, int(round(crop.shape[1] * scale)))
        return cv2.resize(
            crop, (width, self.cfg.store_height), interpolation=cv2.INTER_AREA
        )

    def _buffer_for(self, track_id: int) -> TrackBuffer:
        buffer = self._buffers.get(track_id)
        if buffer is None:
            buffer = TrackBuffer(track_id, self.cfg.max_observations)
            self._buffers[track_id] = buffer
        return buffer

    def _evict(self) -> None:
        while len(self._buffers) > self.cfg.max_tracks:
            track_id, _ = self._buffers.popitem(last=False)
            logger.debug(
                "Evicted buffer for track %d (max_tracks=%d reached)",
                track_id,
                self.cfg.max_tracks,
            )

    # -- access ------------------------------------------------------------

    def get(self, track_id: int) -> TrackBuffer | None:
        return self._buffers.get(track_id)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._buffers

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Iterator[TrackBuffer]:
        return iter(self._buffers.values())

    def ready(self, min_observations: int) -> list[TrackBuffer]:
        """Buffers holding enough observations to be worth embedding."""
        return [b for b in self._buffers.values() if len(b) >= min_observations]

    def drop(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)

    def reset(self) -> None:
        self._buffers.clear()

    def memory_estimate_mb(self) -> float:
        """Rough resident size of the buffered crops, for logging."""
        total = sum(
            obs.crop.nbytes for buffer in self._buffers.values() for obs in buffer
        )
        return total / (1024 * 1024)
