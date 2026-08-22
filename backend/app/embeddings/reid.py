"""Re-ID embedding branch (Phase 4) -- OSNet whole-body appearance.

Encodes what a person looks like overall: build, clothing, gross shape. It is
the modality that works when the face is invisible and the person is not
walking, which between them covers a great deal of real CCTV.

Per-frame, so it subclasses `PerFrameBranch` and inherits quality-weighted
aggregation from the face branch's machinery for free.

Two things worth understanding about this modality
--------------------------------------------------
**It is not the DeepSORT embedder.** DeepSORT already runs a small appearance
model to associate boxes between adjacent frames. That one answers "is this the
same blob as last frame"; this one answers "is this the person on the
watchlist". Different jobs, different requirements, and conflating them is a
natural mistake because both get called "re-ID".

**It goes stale.** Face and gait are properties of a person. Re-ID is mostly a
property of their clothes. A match across two hours is strong evidence; the
same score across two weeks means almost nothing, because it is probably
matching a common jacket. `trust_at()` implements that decay, so a stored
reference loses weight with age rather than silently asserting a stale
identity. That is Phase 11's "time-aware trust" brought forward, because the
alternative is a system that confidently misidentifies people by their coat.

The model definition is vendored under `vendor/` (MIT) rather than depended on;
see that directory's README for why. Weights download on first use.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.core.config import ReIDSettings, Settings, get_settings
from app.core.logging import get_logger
from app.core.types import Modality, ModalityEmbedding, TrackObservation, l2_normalize
from app.embeddings.base import PerFrameBranch

logger = get_logger(__name__)

# Google Drive ids from torchreid's `osnet.py` pretrained_urls, verified live.
PRETRAINED_URLS = {
    "osnet_x1_0": "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    "osnet_x0_75": "https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    "osnet_x0_5": "https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    "osnet_x0_25": "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
    "osnet_ibn_x1_0": "https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l",
}

# ImageNet statistics, which is what OSNet was trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDEmbedder(PerFrameBranch):
    """Whole-body appearance embeddings from OSNet."""

    modality = Modality.REID
    embedding_dim = 512

    def __init__(
        self,
        settings: Settings | None = None,
        reid: ReIDSettings | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = reid or self.settings.reid
        self.device = device or self.settings.resolve_device()
        self.min_observations = self.cfg.min_frames

        import torch

        from app.embeddings.vendor import osnet as osnet_module

        builder = getattr(osnet_module, self.cfg.model, None)
        if builder is None:
            raise ValueError(
                f"Unknown OSNet variant {self.cfg.model!r}. Available: "
                + ", ".join(sorted(PRETRAINED_URLS))
            )

        self.torch = torch
        self.model = builder(num_classes=1000, pretrained=False)
        self._load_weights()
        self.model.eval().to(self.device)
        # OSNet returns features rather than logits in eval mode.
        logger.info("OSNet %s ready on %s", self.cfg.model, self.device)

    # -- weights -----------------------------------------------------------

    def _weights_path(self) -> Path:
        return self.settings.paths.models_dir / f"{self.cfg.model}_imagenet.pth"

    def _load_weights(self) -> None:
        path = self._weights_path()
        if not path.is_file():
            self._download_weights(path)

        # weights_only=True. These checkpoints are downloaded over the network
        # from Google Drive with no signature and no checksum, and pickle
        # deserialisation executes arbitrary code -- so anything able to
        # substitute the Drive object, or write into models_dir, would get code
        # execution as the server user. These are plain state dicts, so the
        # restricted loader reads them unchanged.
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)

        # Checkpoints saved from DataParallel carry a "module." prefix.
        cleaned = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }

        model_state = self.model.state_dict()
        matched = {
            key: value
            for key, value in cleaned.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        self.model.load_state_dict(matched, strict=False)

        if not matched:
            raise RuntimeError(
                f"No layers matched when loading {path}. The checkpoint does not "
                f"correspond to {self.cfg.model}."
            )
        skipped = len(model_state) - len(matched)
        logger.info(
            "Loaded %d/%d OSNet layers from %s%s",
            len(matched), len(model_state), path.name,
            # The classifier is trained for a different label set and is
            # unused: we take features, not class predictions.
            f" ({skipped} skipped, expected for the classifier head)" if skipped else "",
        )

    def _download_weights(self, path: Path) -> None:
        url = PRETRAINED_URLS.get(self.cfg.model)
        if url is None:
            raise FileNotFoundError(
                f"No weights at {path} and no known download URL for "
                f"{self.cfg.model!r}. Fetch them manually into {path.parent}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading OSNet weights for %s (~11MB)", self.cfg.model)
        try:
            import gdown

            gdown.download(url, str(path), quiet=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not download OSNet weights: {exc}. These are hosted on "
                "Google Drive and can hit quota limits. Download manually from "
                f"{url} and save to {path}."
            ) from exc

        if not path.is_file():
            raise RuntimeError(
                f"Weight download reported success but {path} is missing. Google "
                "Drive quota errors can look like this; try again later or "
                "fetch manually."
            )

        # A Drive quota page is HTML and passes is_file() happily, then fails
        # deep inside the loader with something unhelpful. Torch checkpoints
        # are zip archives ("PK") or legacy pickles (0x80); HTML is neither.
        head = path.read_bytes()[:2]
        if head[:1] not in (b"P", b"\x80"):
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"What downloaded to {path.name} is not a torch checkpoint -- "
                "it starts with "
                f"{head!r}, which usually means Google Drive returned a quota "
                f"or login page. Download it manually from {url}."
            )

    # -- preprocessing -----------------------------------------------------

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop -> normalised CHW float array at OSNet's input size."""
        resized = cv2.resize(
            crop,
            (self.cfg.input_width, self.cfg.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - _MEAN) / _STD
        return np.transpose(normalised, (2, 0, 1))

    # -- quality -----------------------------------------------------------

    def _quality(self, observation: TrackObservation) -> tuple[float, dict]:
        """How much to trust this body crop, in [0, 1].

        Re-ID degrades differently from face. It does not care about pose --
        a back view is perfectly usable, which is the point of the modality --
        but it cares a great deal about how much of the body is present and how
        many pixels it has.
        """
        crop = observation.crop
        height, width = crop.shape[:2]

        resolution = min(1.0, height / float(self.cfg.ideal_box_height))

        # A standing person is roughly 2.5x taller than wide. A box far from
        # that ratio usually means a partial body or a merged detection, and
        # its appearance vector will describe something other than one person.
        aspect = height / max(1.0, float(width))
        ratio = min(aspect / self.cfg.ideal_aspect, self.cfg.ideal_aspect / aspect)

        detection = float(np.clip(observation.detection_confidence, 0.0, 1.0))
        # Detection confidence is often 0 for tracker-predicted boxes; treat
        # that as neutral rather than as evidence of a bad crop.
        if detection <= 0.0:
            detection = 0.5

        quality = resolution * ratio * detection
        return float(np.clip(quality, 0.0, 1.0)), {
            "resolution": resolution,
            "aspect": aspect,
            "aspect_ratio_score": ratio,
            "detection_confidence": detection,
            "box_height": float(observation.box_height),
        }

    # -- embedding ---------------------------------------------------------

    def embed_frame(self, observation: TrackObservation) -> ModalityEmbedding:
        crop = observation.crop
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
            return ModalityEmbedding.empty(self.modality, reason="empty crop")

        quality, detail = self._quality(observation)
        if quality < self.cfg.min_quality:
            return ModalityEmbedding.empty(self.modality, reason="below min_quality")

        batch = self.torch.from_numpy(self.preprocess(crop)[None, ...]).to(self.device)
        with self.torch.no_grad():
            features = self.model(batch)
        vector = features.squeeze(0).float().cpu().numpy()

        return ModalityEmbedding(
            modality=self.modality,
            vector=l2_normalize(vector),
            quality=quality,
            frames_used=1,
            detail=detail,
        )

    def embed_batch(self, observations: list[TrackObservation]) -> list[ModalityEmbedding]:
        """Embed several observations in one forward pass."""
        usable: list[tuple[int, TrackObservation, float, dict]] = []
        results: list[ModalityEmbedding] = [
            ModalityEmbedding.empty(self.modality) for _ in observations
        ]

        for index, observation in enumerate(observations):
            crop = observation.crop
            if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
                continue
            quality, detail = self._quality(observation)
            if quality < self.cfg.min_quality:
                continue
            usable.append((index, observation, quality, detail))

        if not usable:
            return results

        batch = np.stack([self.preprocess(o.crop) for _, o, _, _ in usable])
        tensor = self.torch.from_numpy(batch).to(self.device)
        with self.torch.no_grad():
            features = self.model(tensor).float().cpu().numpy()

        for (index, _, quality, detail), vector in zip(usable, features):
            results[index] = ModalityEmbedding(
                modality=self.modality,
                vector=l2_normalize(vector),
                quality=quality,
                frames_used=1,
                detail=detail,
            )
        return results

    def embed(self, observations) -> ModalityEmbedding:
        """Batched override of the per-frame loop in `PerFrameBranch`."""
        per_frame = self.embed_batch(list(observations))
        usable = [e for e in per_frame if e.has_signal]
        if len(usable) < self.min_observations:
            return ModalityEmbedding.empty(
                self.modality, reason="too few usable observations"
            )
        return self.aggregate(usable)


def trust_at(days_elapsed: float, half_life_days: float) -> float:
    """How far a re-ID reference should still be trusted after `days_elapsed`.

    Exponential decay: 1.0 the day it was enrolled, 0.5 after one half-life.

    Face and gait describe a person. Re-ID largely describes their clothing, so
    a strong score weeks after enrollment is far more likely to mean "similar
    jacket" than "same person". Decaying the weight is what stops the system
    confidently misidentifying someone by their coat.

    Returns a multiplier for the modality's fusion weight, never a match score.
    """
    if half_life_days <= 0:
        return 1.0
    return float(math.pow(0.5, max(0.0, days_elapsed) / half_life_days))
