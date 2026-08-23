"""The contract every embedding branch implements.

The three modalities do not have the same shape in time, and the interface
refuses to pretend otherwise:

* **Face** and **re-ID** are per-frame. You can embed a single crop, and a
  track's embedding is an aggregate over its frames.
* **Gait** is not. A single frame of a person walking carries no gait
  information at all -- you need a sequence spanning at least one step cycle.

So the shared entry point is `embed(observations)`: every branch takes a
*sequence* of looks at one tracked person and returns one `ModalityEmbedding`.
`PerFrameBranch` then implements that aggregation once, for the two branches
that genuinely work frame by frame, while the gait branch implements `embed`
directly over the whole sequence.

Forcing gait through a per-frame interface would have meant faking it -- and a
fake gait signal is worse than none, because fusion would weight it as real.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from app.core.logging import get_logger
from app.core.types import (
    Modality,
    ModalityEmbedding,
    TrackObservation,
    l2_normalize,
)

logger = get_logger(__name__)


class EmbeddingBranch(ABC):
    """One modality's encoder: a sequence of looks at a person -> one vector."""

    #: Which signal this branch produces.
    modality: Modality
    #: Length of the vectors it returns. Used to validate gallery entries.
    embedding_dim: int
    #: Below this many usable observations the branch reports no signal rather
    #: than guessing. Gait needs far more than face.
    min_observations: int = 1

    @property
    def model_id(self) -> str:
        """Which model this branch is running, e.g. "osnet_x1_0/msmt17".

        Stamped onto every embedding so a stored reference and a live probe
        can be checked for having come from the same space before their cosine
        similarity is believed (DES-01). Branches that do not override this
        return "", which reads as "unknown" and is never treated as a match.
        """
        return ""

    @abstractmethod
    def embed(self, observations: Sequence[TrackObservation]) -> ModalityEmbedding:
        """Encode one track's observations. Never raises on 'no signal'.

        A branch that cannot produce a vector -- no face visible, too few
        frames for a gait cycle -- returns `ModalityEmbedding.empty(...)`.
        Callers rely on that: absence of signal is ordinary here.
        """

    def embed_reference(
        self, observations: Sequence[TrackObservation]
    ) -> ModalityEmbedding:
        """Encode enrollment footage into a stored reference vector.

        Defaults to the live path. Override where enrollment should differ --
        the face branch, for instance, can afford stricter quality filtering
        offline than it can on a live feed.
        """
        return self.embed(observations)


class PerFrameBranch(EmbeddingBranch):
    """Base for modalities that embed each frame independently.

    Subclasses implement `embed_frame`; this class handles aggregation across
    a track, weighting each frame by its own quality so that one clear look
    counts for more than a dozen poor ones.
    """

    @abstractmethod
    def embed_frame(self, observation: TrackObservation) -> ModalityEmbedding:
        """Encode a single observation, or return `empty` if it carries none."""

    def embed(self, observations: Sequence[TrackObservation]) -> ModalityEmbedding:
        per_frame = [self.embed_frame(obs) for obs in observations]
        usable = [e for e in per_frame if e.has_signal]

        if len(usable) < self.min_observations:
            return ModalityEmbedding.empty(
                self.modality, reason="too few usable observations"
            )
        return self.aggregate(usable)

    def aggregate(self, embeddings: Sequence[ModalityEmbedding]) -> ModalityEmbedding:
        """Quality-weighted mean of per-frame embeddings, re-normalised.

        Weighting by quality is what lets a track survive a bad stretch: ten
        frames of a turned-away head barely move the average, one clear frontal
        frame dominates it. A plain mean would let the bad frames outvote the
        good one purely by being more numerous.
        """
        vectors = np.stack([e.vector for e in embeddings])
        weights = np.array([e.quality for e in embeddings], dtype=np.float32)

        if float(weights.sum()) <= 0.0:
            # Every frame scored zero quality; fall back to an unweighted mean
            # rather than dividing by zero, and report the low confidence.
            weights = np.ones_like(weights)

        pooled = np.average(vectors, axis=0, weights=weights)
        qualities = [e.quality for e in embeddings]

        return ModalityEmbedding(
            modality=self.modality,
            model_id=self.model_id,
            vector=l2_normalize(pooled),
            # Track-level quality is the best single look, not the mean: one
            # unambiguous frame is enough to trust the identification, and
            # averaging would punish it for the frames around it.
            quality=float(max(qualities)),
            frames_used=len(embeddings),
            detail={
                "mean_frame_quality": float(np.mean(qualities)),
                "max_frame_quality": float(max(qualities)),
            },
        )
