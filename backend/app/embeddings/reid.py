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

# Which checkpoint to load (DES-01).
#
# The branch used to load whatever `pretrained_urls` in torchreid's osnet.py
# points at. Those are the *ImageNet* checkpoints -- the filename torchreid
# writes is literally `<model>_imagenet.pth` -- so the re-ID branch was running
# generic visual features and calling them re-identification. That is a
# plausible explanation for the impostor score this project measured: two
# strangers at 0.755 is what generic ImageNet features do to two photographs of
# a person-shaped thing. A model actually trained to separate identities pushes
# strangers far lower.
#
# Keyed by (variant, training set) so the two cannot be confused, and so the
# cached file for one is never silently reused for the other.
#
# Ids taken from torchreid's model zoo, fetched and checked rather than
# remembered: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
_DRIVE = "https://drive.google.com/uc?id="

PRETRAINED_URLS = {
    # ImageNet classification. NOT re-identification -- see IMAGENET_WARNING.
    ("osnet_x1_0", "imagenet"): _DRIVE + "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    ("osnet_x0_75", "imagenet"): _DRIVE + "1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    ("osnet_x0_5", "imagenet"): _DRIVE + "16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    ("osnet_x0_25", "imagenet"): _DRIVE + "1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
    ("osnet_ibn_x1_0", "imagenet"): _DRIVE + "1sr90V6irlYYDd4_4ISU2iruoRG8J__6l",

    # Trained for person re-identification. These are what this branch is for.
    ("osnet_x1_0", "msmt17"): _DRIVE + "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    ("osnet_x1_0", "market1501"): _DRIVE + "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    ("osnet_x1_0", "dukemtmcreid"): _DRIVE + "1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",

    # Multi-source domain generalisation: trained on MSMT17 + DukeMTMC + CUHK03
    # together, specifically to transfer to cameras it has never seen. That is
    # exactly this system's situation, which is why it is worth having.
    ("osnet_ibn_x1_0", "multi_source"): _DRIVE + "14sH6yZwuNHPTElVoEZ26zozOOZIej5Mf",
}

#: Checkpoints that are not re-ID models. Loading one is legitimate -- it is
#: the only option for the smaller variants -- but it must never be silent.
NON_REID_WEIGHTS = {"imagenet"}

#: How much of the network must load before a checkpoint is believed.
#: The classifier head is trained for a different label set and is discarded,
#: so a genuine checkpoint still matches nearly everything else. A checkpoint
#: matching only a handful of layers means the wrong file: the model would run,
#: return numbers, and compare untrained features against each other.
_MIN_MATCHED_FRACTION = 0.90


IMAGENET_WARNING = (
    "OSNet %s is running %s weights, which are NOT trained for person "
    "re-identification -- they are generic ImageNet classification features. "
    "Two strangers will score far higher than they should, and every number "
    "calibrated against re-ID weights (reid_impostor, reid_genuine, "
    "reid_threshold, the fusion weights) is wrong for this model. Use it to "
    "get the pipeline running, not to identify anybody."
)


def available_weights(model: str) -> list[str]:
    return sorted(w for (m, w) in PRETRAINED_URLS if m == model)

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
            variants = sorted({m for m, _ in PRETRAINED_URLS})
            raise ValueError(
                f"Unknown OSNet variant {self.cfg.model!r}. Available: "
                + ", ".join(variants)
            )

        self.weights = self.cfg.weights
        if (self.cfg.model, self.weights) not in PRETRAINED_URLS:
            options = available_weights(self.cfg.model)
            raise ValueError(
                f"No {self.weights!r} checkpoint for {self.cfg.model!r}. "
                + (
                    f"Available for this variant: {', '.join(options)}."
                    if options
                    else "This variant has no known checkpoints."
                )
                + " Only osnet_x1_0 and osnet_ibn_x1_0 have re-ID-trained "
                "weights published; the smaller variants are ImageNet only."
            )

        self.torch = torch
        # num_classes only sizes the classifier head, which is discarded: this
        # branch takes features, never class predictions.
        self.model = builder(num_classes=1000, pretrained=False)
        self._load_weights()
        self.model.eval().to(self.device)
        # OSNet returns features rather than logits in eval mode.
        logger.info(
            "OSNet %s ready on %s (%s weights)",
            self.cfg.model, self.device, self.weights,
        )
        if self.weights in NON_REID_WEIGHTS:
            logger.warning(IMAGENET_WARNING, self.cfg.model, self.weights)

    @property
    def model_id(self) -> str:
        # Both halves matter: osnet_x1_0/imagenet and osnet_x1_0/msmt17 are
        # the same architecture holding completely different weights, and
        # vectors from one mean nothing against the other.
        return f"{self.cfg.model}/{self.weights}"

    # -- weights -----------------------------------------------------------

    def _weights_path(self) -> Path:
        # The training set is in the filename. Without it, switching weights
        # would find the old file already on disk and load it happily -- the
        # quietest possible way to keep running the wrong model.
        return self.settings.paths.models_dir / f"{self.cfg.model}_{self.weights}.pth"

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
        # Refuse a checkpoint that only half fits BEFORE loading it. The
        # classifier head is trained for a different label set and is expected
        # to be skipped, but everything else should match: a checkpoint that
        # matches a handful of layers is the wrong file, and loading it leaves
        # most of the network at its random initialisation. The model would
        # still run and still return confident-looking cosine similarities --
        # between untrained features.
        classifier = {k for k in model_state if k.startswith("classifier.")}
        backbone = set(model_state) - classifier
        matched_backbone = len(backbone & set(matched))
        fraction = matched_backbone / len(backbone) if backbone else 0.0

        if fraction < _MIN_MATCHED_FRACTION:
            raise RuntimeError(
                f"Only {matched_backbone}/{len(backbone)} layers "
                f"({fraction:.0%}) matched when loading {path.name}. That is "
                f"not a {self.cfg.model} checkpoint -- loading it would leave "
                "most of the network randomly initialised while still "
                "returning plausible-looking similarity scores. Delete the "
                "file and let it download again."
            )

        self.model.load_state_dict(matched, strict=False)

        skipped = len(model_state) - len(matched)
        logger.info(
            "Loaded %d/%d OSNet layers from %s%s",
            len(matched), len(model_state), path.name,
            f" ({skipped} skipped, expected for the classifier head)" if skipped else "",
        )

    def _download_weights(self, path: Path) -> None:
        url = PRETRAINED_URLS.get((self.cfg.model, self.weights))
        if url is None:
            raise FileNotFoundError(
                f"No weights at {path} and no known download URL for "
                f"{self.cfg.model!r} trained on {self.weights!r}. Fetch them "
                f"manually into {path.parent}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading OSNet %s weights trained on %s (~11MB)",
            self.cfg.model, self.weights,
        )
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
            model_id=self.model_id,
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
                model_id=self.model_id,
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
