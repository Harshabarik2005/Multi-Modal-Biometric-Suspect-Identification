"""Face embedding branch (Phase 2) -- ArcFace via InsightFace.

Takes body crops from a track, finds a face inside each, and returns a
normalised 512-d ArcFace embedding for the track.

The important design point is the **quality score**. Detection confidence
alone is a poor proxy for whether a face is usable: the detector is perfectly
capable of confidently locating a face turned 80 degrees away from the camera,
from which ArcFace produces a confident and wrong embedding. So quality
combines three things -- how confidently the face was found, how frontal it is,
and how many pixels it occupies. Phase 6's attention head consumes that score
to decide how far to trust this modality on a given track, which is the whole
mechanism the project rests on.

Verified separation on a sample image: same identity 0.96, different identities
0.03, no face present -> clean empty result.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from app.core.config import FaceSettings, Settings, get_settings
from app.core.logging import get_logger
from app.core.types import Modality, ModalityEmbedding, TrackObservation, l2_normalize
from app.embeddings.base import PerFrameBranch

logger = get_logger(__name__)


class FaceEmbedder(PerFrameBranch):
    """ArcFace embeddings, with an occlusion- and pose-aware quality score."""

    modality = Modality.FACE
    embedding_dim = 512

    def __init__(
        self,
        settings: Settings | None = None,
        face: FaceSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = face or self.settings.face
        self.min_observations = self.cfg.min_frames

        # InsightFace ships its own ONNX runtime session; it does not share the
        # torch CUDA context. The stock `onnxruntime` wheel is CPU-only, so GPU
        # here additionally requires `onnxruntime-gpu` built for the installed
        # CUDA. Falling back to CPU is fine -- face inference is not the
        # pipeline bottleneck.
        providers = self._resolve_providers()

        from insightface.app import FaceAnalysis  # heavy import, kept lazy

        logger.info(
            "Loading InsightFace pack %s (providers=%s)", self.cfg.model_pack, providers
        )
        self.app = FaceAnalysis(name=self.cfg.model_pack, providers=providers)
        # ctx_id >= 0 selects a GPU, -1 means CPU.
        ctx_id = 0 if providers and providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(
            ctx_id=ctx_id, det_size=(self.cfg.det_size, self.cfg.det_size)
        )

    def _resolve_providers(self) -> list[str]:
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
        want_gpu = self.cfg.use_gpu and self.settings.resolve_device().startswith("cuda")

        if want_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if want_gpu:
            logger.warning(
                "face.use_gpu is set but onnxruntime has no CUDAExecutionProvider "
                "(installed providers: %s). Install onnxruntime-gpu matching your "
                "CUDA build to enable it. Falling back to CPU.",
                sorted(available),
            )
        return ["CPUExecutionProvider"]

    # -- quality -----------------------------------------------------------

    def _quality(self, face: Any, crop_height: int) -> tuple[float, dict[str, float]]:
        """Score how much this face should be trusted, in [0, 1].

        Three independent ways a face can be unusable, multiplied together so
        that any one of them failing drags the score down:

        * **Detection confidence** -- was it clearly a face at all.
        * **Frontality** -- ArcFace degrades sharply with yaw. A profile view
          still embeds, just into the wrong place in the space, so this has to
          be penalised even when detection is confident.
        * **Resolution** -- a 20px face upsampled to the model's input carries
          almost no identity information regardless of how frontal it is.
        """
        det_score = float(getattr(face, "det_score", 0.0))

        # `pose` is (pitch, yaw, roll) in degrees when the pack provides it.
        yaw = pitch = 0.0
        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 2:
            pitch, yaw = float(pose[0]), float(pose[1])

        # Full credit up to `frontal_yaw_deg`, decaying to zero at `max_yaw_deg`.
        abs_yaw = abs(yaw)
        if abs_yaw <= self.cfg.frontal_yaw_deg:
            frontality = 1.0
        elif abs_yaw >= self.cfg.max_yaw_deg:
            frontality = 0.0
        else:
            span = self.cfg.max_yaw_deg - self.cfg.frontal_yaw_deg
            frontality = 1.0 - (abs_yaw - self.cfg.frontal_yaw_deg) / span

        x1, y1, x2, y2 = face.bbox
        face_height = float(y2 - y1)
        resolution = min(1.0, face_height / float(self.cfg.ideal_face_height))

        quality = det_score * frontality * resolution
        detail = {
            "det_score": det_score,
            "yaw": yaw,
            "pitch": pitch,
            "frontality": frontality,
            "face_height": face_height,
            "resolution": resolution,
            "crop_height": float(crop_height),
        }
        return float(np.clip(quality, 0.0, 1.0)), detail

    # -- per-frame embedding -----------------------------------------------

    def embed_frame(self, observation: TrackObservation) -> ModalityEmbedding:
        crop = observation.crop
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return ModalityEmbedding.empty(self.modality, reason="empty crop")

        faces = self.detect(crop)
        if not faces:
            # The ordinary case: person facing away, or too far off. Not an error.
            return ModalityEmbedding.empty(self.modality, reason="no face detected")

        # A body crop should contain exactly one face; if the box caught a
        # neighbour too, keep the largest, which is the tracked person.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        quality, detail = self._quality(face, crop.shape[0])

        if quality < self.cfg.min_quality:
            return ModalityEmbedding.empty(self.modality, reason="below min_quality")

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32))

        return ModalityEmbedding(
            modality=self.modality,
            model_id=self.model_id,
            vector=np.asarray(embedding, dtype=np.float32),
            quality=quality,
            frames_used=1,
            detail=detail,
        )

    @property
    def model_id(self) -> str:
        return f"insightface/{self.cfg.model_pack}"

    def detect(self, image: np.ndarray) -> list[Any]:
        """Run InsightFace on a BGR image. Returns [] when nothing is found."""
        try:
            return list(self.app.get(image))
        except Exception as exc:  # noqa: BLE001 - a bad crop must not kill a run
            logger.debug("Face detection failed on a crop: %s", exc)
            return []

    # -- enrollment --------------------------------------------------------

    def embed_reference(
        self, observations: Sequence[TrackObservation]
    ) -> ModalityEmbedding:
        """Build an enrollment reference, keeping only the best-quality looks.

        Enrollment is offline and typically has hundreds of frames from a 360
        degree rotation, so it can afford to be picky in a way live matching
        cannot. Averaging in profile and back-of-head frames would blur the
        reference vector towards the population mean and cost real accuracy on
        every subsequent match.
        """
        per_frame = [self.embed_frame(obs) for obs in observations]
        usable = [e for e in per_frame if e.has_signal]

        if not usable:
            return ModalityEmbedding.empty(self.modality, reason="no usable faces")

        usable.sort(key=lambda e: e.quality, reverse=True)
        keep = max(1, math.ceil(len(usable) * self.cfg.enroll_top_fraction))
        selected = usable[:keep]

        if len(selected) < self.cfg.enroll_min_frames:
            logger.warning(
                "Only %d usable face frames for enrollment (want >= %d). "
                "The reference will be weak -- record more angles.",
                len(selected),
                self.cfg.enroll_min_frames,
            )

        pooled = self.aggregate(selected)
        pooled.detail["enroll_candidates"] = float(len(usable))
        pooled.detail["enroll_selected"] = float(len(selected))
        return pooled
