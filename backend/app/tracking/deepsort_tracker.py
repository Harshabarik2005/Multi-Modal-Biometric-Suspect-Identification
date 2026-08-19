"""DeepSORT multi-object tracker.

Wraps `deep-sort-realtime` so the pipeline gets stable `track_id`s across
frames. The appearance embedder configured here is used *only* for
frame-to-frame association -- the identity embeddings that actually decide a
watchlist match come from the face / gait / re-ID branches in later phases.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings, TrackingSettings, get_settings
from app.core.logging import get_logger
from app.core.types import Detection, Track

logger = get_logger(__name__)


class DeepSortTracker:
    """Associates per-frame detections into persistent tracks."""

    def __init__(
        self,
        settings: Settings | None = None,
        tracking: TrackingSettings | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = tracking or self.settings.tracking
        device = device or self.settings.resolve_device()
        embedder_gpu = self.cfg.embedder_gpu and device.startswith("cuda")

        from deep_sort_realtime.deepsort_tracker import DeepSort

        logger.info(
            "Initialising DeepSORT (embedder=%s, gpu=%s, max_age=%d, n_init=%d)",
            self.cfg.embedder,
            embedder_gpu,
            self.cfg.max_age,
            self.cfg.n_init,
        )
        self.tracker = DeepSort(
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            max_cosine_distance=self.cfg.max_cosine_distance,
            nn_budget=self.cfg.nn_budget,
            embedder=self.cfg.embedder,
            embedder_gpu=embedder_gpu,
            half=embedder_gpu,
            bgr=True,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Feed one frame's detections in, get the confirmed tracks back.

        Only confirmed tracks (survived `n_init` frames) are returned, so a
        one-frame false positive never reaches the matching stage.
        """
        # deep-sort-realtime wants [([left, top, w, h], confidence, class), ...]
        raw = [
            (list(det.ltwh), det.confidence, det.class_id) for det in detections
        ]

        tracks = self.tracker.update_tracks(raw, frame=frame)

        confirmed: list[Track] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            confirmed.append(
                Track(
                    track_id=int(track.track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(track.det_conf or 0.0),
                    time_since_update=int(track.time_since_update),
                )
            )
        return confirmed

    def reset(self) -> None:
        """Clear all tracks. Call between videos so IDs restart at 1."""
        self.tracker.delete_all_tracks()
