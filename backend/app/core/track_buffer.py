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

Gait also keeps a second ring of the same observations, thinned towards
`gait_sample_hz` a second so it spans a whole stride whatever the frame rate.

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

    def __init__(
        self,
        track_id: int,
        max_observations: int,
        gait_max_observations: int | None = None,
        gait_sample_interval_s: float = 0.0,
    ) -> None:
        self.track_id = track_id
        self.observations: deque[TrackObservation] = deque(maxlen=max_observations)
        self.first_frame: int | None = None
        self.last_frame: int = -1
        #: Frame index at which this track was last matched, so the pipeline
        #: can rate-limit re-matching without re-deriving it.
        self.last_matched_frame: int | None = None
        #: Total observations ever added, including those aged out of the ring.
        self.total_seen: int = 0
        #: Gait's own history: the same observation objects, thinned in time.
        #:
        #: Gait needs a whole stride and `observations` is a frame count, 1.07s
        #: at 60fps. Face and appearance keep that ring exactly as it was,
        #: because they choose crops from it by box height and a longer window
        #: changes which crops win (`gait_sample_hz` has the measurements).
        self.gait_observations: deque[TrackObservation] = deque(
            maxlen=gait_max_observations or max_observations
        )
        #: Spacing gait aims for between kept observations, in seconds.
        #: 0 keeps every frame.
        self.gait_sample_interval_s = gait_sample_interval_s
        #: The source's frame period, learned as the shortest gap seen. Dropped
        #: frames only ever lengthen a gap, so the shortest one is the period
        #: even when the tracker misses frames.
        self._frame_period_s: float = 0.0
        self._last_timestamp_s: float = -1.0
        self._gait_countdown: int = 0

    def add(self, observation: TrackObservation) -> None:
        if self.first_frame is None:
            self.first_frame = observation.frame_index
        self.last_frame = observation.frame_index
        self.total_seen += 1
        self.observations.append(observation)
        self._note_frame_period(observation.timestamp_s)
        if self._gait_countdown <= 0:
            self.gait_observations.append(observation)
            self._gait_countdown = self.gait_stride() - 1
        else:
            self._gait_countdown -= 1

    def gait_stride(self) -> int:
        """How many frames pass between the ones gait keeps.

        A whole number of frames, never a wall-clock slot. Keeping "one every
        0.05s" out of a 25fps stream keeps four frames in five, whose gaps run
        0.04, 0.04, 0.04, 0.08 -- and `resample_cadence` lays its grid on the
        MEDIAN gap, so one grid point in five has no real sample near it.
        Measured on evenly spaced frames with a perfect segmenter, cadence
        coverage falls from 1.00 to 0.80 at 25fps and to 0.67 at 29.97fps
        against a floor of 0.60, so a track that walks behind something for
        0.4s is then refused for "too many gaps" on footage that used to carry
        gait. An integer stride keeps the spacing even at every frame rate: 1
        at 25 and 29.97fps, which is what gait saw before this ring existed,
        and 3 at 60fps, which is what makes a stride fit.

        1 until a second frame has arrived, since the period is unknown until
        then, and 1 whenever the clock does not advance at all -- thinning by
        time is impossible there, and keeping one frame per track would be a
        far worse failure than not thinning.
        """
        if self.gait_sample_interval_s <= 0.0 or self._frame_period_s <= 0.0:
            return 1
        return max(1, int(round(self.gait_sample_interval_s / self._frame_period_s)))

    def _note_frame_period(self, timestamp_s: float) -> None:
        """Learn the source's frame period from the shortest gap seen."""
        if timestamp_s >= 0.0 and self._last_timestamp_s >= 0.0:
            gap = timestamp_s - self._last_timestamp_s
            if gap > 0.0 and (
                self._frame_period_s <= 0.0 or gap < self._frame_period_s
            ):
                self._frame_period_s = gap
        self._last_timestamp_s = max(self._last_timestamp_s, timestamp_s)

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

            # Whether the body is cut off can only be seen against the frame,
            # and the frame is not kept. A box running into an edge is one the
            # detector wanted to draw larger, so the person continues past it
            # and the crop holds part of a body presented as a whole one.
            #
            # All four edges, not just top and bottom. Vertical clipping breaks
            # gait's height normalisation; horizontal clipping is worse still,
            # because the cadence signal IS the width of the silhouette's lower
            # third, so a body half out of shot contributes a truncated width
            # exactly where the walk is being read. Measured on a clip of
            # someone crossing the frame, 16% of frames were part-way out of
            # shot at entry and exit, and excluding them moved periodicity from
            # 0.166 to 0.388 and area stability from 0.145 to 0.092.
            #
            # Within a small margin of an edge rather than exactly on it: a box
            # on a body leaving the frame stops a pixel or two short of it
            # (`edge_margin_fraction`).
            frame_height, frame_width = frame.shape[:2]
            margin_x = self.cfg.edge_margin_fraction * frame_width
            margin_y = self.cfg.edge_margin_fraction * frame_height
            observation = TrackObservation(
                frame_index=result.frame_index,
                timestamp_s=result.timestamp_s,
                crop=self._downscale(crop),
                box_height=track.height,
                detection_confidence=track.confidence,
                at_frame_edge=bool(
                    track.x1 <= margin_x
                    or track.y1 <= margin_y
                    or track.x2 >= float(frame_width) - margin_x
                    or track.y2 >= float(frame_height) - margin_y
                ),
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
            hz = self.cfg.gait_sample_hz
            buffer = TrackBuffer(
                track_id,
                self.cfg.max_observations,
                gait_max_observations=self.cfg.gait_max_observations,
                gait_sample_interval_s=(1.0 / hz) if hz > 0 else 0.0,
            )
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
        # Gait keeps observations the main ring has already let go, and both
        # rings hold the same objects, so each crop is counted once.
        crops = {
            id(obs): obs
            for buffer in self._buffers.values()
            for obs in (*buffer.observations, *buffer.gait_observations)
        }
        total = sum(obs.crop.nbytes for obs in crops.values())
        return total / (1024 * 1024)
