"""Person silhouette extraction and normalisation for the gait branch.

Gait recognition does not work on raw pixels. It works on binary silhouettes,
normalised so that the person's size and position in frame stop mattering and
only their *shape over time* remains. Two people filmed at different distances
must produce the same silhouette sequence if they walk the same way.

Normalisation follows the convention every gait model uses (GaitSet, GaitGL,
GaitBase all expect it), which is what keeps the door open to swapping a
learned encoder in later:

1. Take the binary mask and crop to its tight bounding box.
2. Scale so the person's height is exactly `silhouette_height` (64).
3. Horizontally centre on the silhouette's **centre of mass**, not the centre
   of the bounding box. An outstretched arm or a swinging bag shifts the box
   but barely moves the mass, so centre-of-mass keeps the torso in a stable
   column while the legs swing around it. Centring on the box instead makes
   the whole body jitter left and right, which is exactly the signal gait
   recognition is trying to read.
4. Paste into a fixed `silhouette_height` x `silhouette_width` canvas (64x44).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.core.config import GaitSettings, Settings, get_settings
from app.core.logging import get_logger
from app.core.types import TrackObservation

logger = get_logger(__name__)


@dataclass(slots=True)
class Silhouette:
    """One normalised binary silhouette plus how much to trust it."""

    frame_index: int
    #: (silhouette_height, silhouette_width) float32 in [0, 1].
    image: np.ndarray
    #: Fraction of the source crop the raw mask covered. Very low means the
    #: segmenter found little, very high means it probably grabbed background.
    coverage: float
    #: True when the mask touches the frame edge, so the body is likely cut off.
    clipped: bool
    #: When this frame was captured, in seconds from the start of the source.
    #:
    #: Cadence is a rate, so it can only be recovered in real time. Frame
    #: counts are not real time: `extract` drops frames whose mask fails, and
    #: `video.frame_stride` skips frames before that, so consecutive
    #: silhouettes can be any distance apart (LOG-06). Defaults to -1, meaning
    #: unknown, which makes the analysis fall back to assuming uniform
    #: sampling.
    timestamp_s: float = -1.0


class SilhouetteExtractor:
    """Cuts normalised person silhouettes out of track crops with YOLOv8-seg."""

    def __init__(
        self,
        settings: Settings | None = None,
        gait: GaitSettings | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = gait or self.settings.gait
        self.device = device or self.settings.resolve_device()

        weights = self._resolve_weights(self.cfg.seg_model)
        logger.info("Loading segmentation weights %s on %s", weights, self.device)

        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.model.to(self.device)

    def _resolve_weights(self, model: str):
        """Same models_dir discipline as the detector: never litter the cwd."""
        from pathlib import Path

        candidate = Path(model)
        if candidate.is_file():
            return candidate

        models_dir = self.settings.paths.models_dir
        local = models_dir / candidate.name
        if local.is_file():
            return local

        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            return Path(attempt_download_asset(local))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not pre-fetch %s into %s (%s); letting Ultralytics resolve it",
                candidate.name, models_dir, exc,
            )
            return model

    # -- extraction --------------------------------------------------------

    def masks_for(self, crop: np.ndarray) -> np.ndarray | None:
        """Largest person mask in a crop, as a bool array the crop's size."""
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None

        try:
            results = self.model.predict(
                crop,
                conf=self.cfg.seg_conf,
                classes=[0],
                device=self.device,
                verbose=False,
                retina_masks=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad crop must not stop a run
            logger.debug("Segmentation failed on a crop: %s", exc)
            return None

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None

        masks = results[0].masks.data.cpu().numpy()
        # The crop is centred on one tracked person; if a neighbour bleeds into
        # the box, the tracked person is the larger mask.
        mask = masks[int(np.argmax(masks.sum(axis=(1, 2))))]

        if mask.shape != crop.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask > 0.5

    def normalise(self, mask: np.ndarray) -> np.ndarray | None:
        """Scale and centre a binary mask onto the standard gait canvas."""
        height, width = self.cfg.silhouette_height, self.cfg.silhouette_width

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None

        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
        cropped = mask[top:bottom, left:right].astype(np.float32)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            return None

        # Scale to the target height, preserving aspect ratio.
        scale = height / cropped.shape[0]
        scaled_width = max(1, int(round(cropped.shape[1] * scale)))
        resized = cv2.resize(
            cropped, (scaled_width, height), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros((height, width), dtype=np.float32)

        # Horizontal centre of mass, so swinging limbs do not drag the whole
        # body sideways between frames.
        column_mass = resized.sum(axis=0)
        total = float(column_mass.sum())
        if total <= 0:
            return None
        centre = float((column_mass * np.arange(scaled_width)).sum() / total)

        # Place `resized` so its centre of mass lands on the canvas centre.
        start = int(round(width / 2.0 - centre))
        src_left = max(0, -start)
        src_right = min(scaled_width, width - start)
        if src_right <= src_left:
            return None
        dst_left = max(0, start)
        canvas[:, dst_left : dst_left + (src_right - src_left)] = resized[
            :, src_left:src_right
        ]
        return np.clip(canvas, 0.0, 1.0)

    def extract(
        self, observations: Sequence[TrackObservation]
    ) -> list[Silhouette]:
        """Normalised silhouettes for a track's observations, poor ones dropped."""
        silhouettes: list[Silhouette] = []

        for observation in observations:
            mask = self.masks_for(observation.crop)
            if mask is None:
                continue

            coverage = float(mask.mean())
            if coverage < self.cfg.min_silhouette_coverage:
                # Barely any foreground -- the segmenter did not really find a
                # person, and a GEI built from this would be noise.
                continue

            normalised = self.normalise(mask)
            if normalised is None:
                continue

            # A mask touching the top or bottom edge means the body is cut off,
            # so its height normalisation is wrong and its shape misleading.
            clipped = bool(mask[0].any() or mask[-1].any())

            silhouettes.append(
                Silhouette(
                    frame_index=observation.frame_index,
                    image=normalised,
                    coverage=coverage,
                    clipped=clipped,
                    timestamp_s=observation.timestamp_s,
                )
            )

        return silhouettes
