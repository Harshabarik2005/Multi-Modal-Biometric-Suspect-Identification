"""Keyless attention fusion (Phase 6) -- the core contribution.

A small trainable head that learns how far to trust each modality, replacing
the fixed rules of Phase 5.

What "keyless" means
--------------------
Standard attention scores a *query* against *keys*. There is no query here --
nothing is being looked up. Each modality's own projected embedding, together
with how good a look it got, is scored directly by a learned function, and
those scores are softmaxed into weights alpha (face), beta (gait), gamma
(re-ID). Hence keyless: the values score themselves.

Where this diverges from the reference paper
--------------------------------------------
* **Three modalities, not two.** The paper fuses face and gait; re-ID is added
  here, and it is the one that goes stale, so its trust decays with age.
* **Open-set metric learning, not closed-set softmax.** The paper classifies
  over 20 known people. A watchlist grows, and almost everyone who walks past a
  camera is on nobody's list, so this trains with a triplet loss on cosine
  distance and matches by similarity against a gallery.
* **Different embedding dimensions.** Face is 512-d, re-ID 512-d, gait 812-d
  with the classical descriptor. They cannot be summed directly, so each is
  projected into a shared space first. The paper's modalities were already
  dimensionally aligned.

The specific failure this has to fix
------------------------------------
Measured in Phase 5: `QualityWeightedFusion` gave re-ID *more* weight than face
(0.54 vs 0.46) on the test clip, because the re-ID crop scored higher quality
even though face is far more discriminative. Fixed rules weight by **how good a
look you got**, not by **how much that modality is worth**. The attention head
should learn that a mediocre face beats an excellent jacket.

Refusing to run untrained
-------------------------
An untrained head is strictly worse than the fixed rules it replaces -- random
projections destroy the carefully calibrated similarities from Phase 5. So
`AttentionFusion` will not produce weights until it has been trained or had
weights loaded, and `is_trained` is False until then. The matching CLI checks
that and falls back to the Phase-5 strategies rather than silently emitting
noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.core.logging import get_logger
from app.core.types import Modality

logger = get_logger(__name__)

#: Order is fixed so saved weights stay loadable.
MODALITY_ORDER: tuple[Modality, ...] = (Modality.FACE, Modality.GAIT, Modality.REID)


@dataclass
class ModalityBatch:
    """One modality's side of a training or inference batch.

    `present` marks which rows actually have this modality. A row where the
    face was never visible must not contribute to the softmax at all -- absence
    is not a low score, it is no score, and conflating them is the mistake this
    whole codebase is built to avoid.
    """

    embeddings: torch.Tensor  # (batch, dim)
    qualities: torch.Tensor  # (batch,)
    present: torch.Tensor  # (batch,) bool


class KeylessAttentionFusion(nn.Module):
    """Projects each modality to a shared space and learns to weight them."""

    def __init__(
        self,
        dims: dict[Modality, int],
        shared_dim: int = 128,
        attention_hidden: int = 64,
        dropout: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # Seed BEFORE building the layers. Seeding only inside train() leaves
        # weight initialisation to the ambient RNG state, which made otherwise
        # identical runs differ substantially -- the same configuration was
        # measured at both 1.78x and 4.37x train/validation overfitting purely
        # because the model was constructed at a different point in the
        # program. Reproducibility has to start at initialisation.
        if seed is not None:
            torch.manual_seed(seed)
        self.dims = dict(dims)
        self.shared_dim = shared_dim
        self.modalities = tuple(m for m in MODALITY_ORDER if m in self.dims)

        # Each modality gets its own projection: they are different sizes and
        # carry different kinds of information, so a shared projection would be
        # forcing an alignment that does not exist.
        self.projections = nn.ModuleDict(
            {
                modality.value: nn.Sequential(
                    nn.Linear(self.dims[modality], shared_dim),
                    nn.LayerNorm(shared_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(shared_dim, shared_dim),
                )
                for modality in self.modalities
            }
        )

        # The keyless scorer. Input is the projected embedding plus that
        # observation's quality, so the head can learn interactions the fixed
        # rules cannot express -- "a low-quality face still beats a
        # high-quality jacket" is exactly such an interaction.
        self.scorer = nn.Sequential(
            nn.Linear(shared_dim + 1, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # A learned per-modality bias, which is where a prior like "face is
        # generally more trustworthy" can live independently of any one
        # observation's quality.
        self.modality_bias = nn.Parameter(torch.zeros(len(self.modalities)))

        self._trained = False

    # -- state -------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """False until trained or loaded. Callers must not use it before then."""
        return self._trained

    def mark_trained(self) -> None:
        self._trained = True

    # -- forward -----------------------------------------------------------

    def forward(
        self, batch: dict[Modality, ModalityBatch]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse a batch. Returns `(fused, weights)`.

        `fused` is (batch, shared_dim), L2-normalised so cosine similarity is
        just a dot product. `weights` is (batch, n_modalities), each row
        summing to 1 over the modalities actually present.
        """
        any_input = next(iter(batch.values()))
        size = any_input.embeddings.shape[0]
        device = any_input.embeddings.device

        projected: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for index, modality in enumerate(self.modalities):
            item = batch.get(modality)
            if item is None:
                projected.append(torch.zeros(size, self.shared_dim, device=device))
                scores.append(torch.full((size,), float("-inf"), device=device))
                masks.append(torch.zeros(size, dtype=torch.bool, device=device))
                continue

            vectors = self.projections[modality.value](item.embeddings)
            projected.append(vectors)

            score = self.scorer(
                torch.cat([vectors, item.qualities.unsqueeze(1)], dim=1)
            ).squeeze(1)
            score = score + self.modality_bias[index]

            # Absent modalities are masked out of the softmax entirely rather
            # than given a low score, so they neither contribute nor dilute.
            score = score.masked_fill(~item.present, float("-inf"))
            scores.append(score)
            masks.append(item.present)

        stacked_scores = torch.stack(scores, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        # A row with no modality at all would softmax over all -inf and produce
        # NaN. Give those rows uniform weights; the caller filters them out by
        # the mask, but NaNs would poison the gradients of the whole batch.
        empty_rows = ~stacked_masks.any(dim=1)
        stacked_scores = torch.where(
            empty_rows.unsqueeze(1),
            torch.zeros_like(stacked_scores),
            stacked_scores,
        )

        weights = torch.softmax(stacked_scores, dim=1)
        weights = torch.where(
            empty_rows.unsqueeze(1), torch.zeros_like(weights), weights
        )

        stacked_projected = torch.stack(projected, dim=1)
        fused = (stacked_projected * weights.unsqueeze(2)).sum(dim=1)
        return nn.functional.normalize(fused, p=2, dim=1), weights

    # -- inference helpers -------------------------------------------------

    @torch.no_grad()
    def fuse_one(
        self, embeddings: dict[Modality, np.ndarray], qualities: dict[Modality, float]
    ) -> tuple[np.ndarray, dict[Modality, float]]:
        """Fuse a single observation. Returns the vector and its weights."""
        if not self.is_trained:
            raise RuntimeError(
                "This attention head has not been trained. Using it would "
                "produce noise that is strictly worse than the Phase-5 fixed "
                "rules. Train it (scripts/train_fusion.py) or load weights."
            )

        self.eval()
        device = next(self.parameters()).device
        batch: dict[Modality, ModalityBatch] = {}

        for modality in self.modalities:
            vector = embeddings.get(modality)
            present = vector is not None
            batch[modality] = ModalityBatch(
                embeddings=torch.tensor(
                    np.asarray(
                        vector if present else np.zeros(self.dims[modality]),
                        dtype=np.float32,
                    )[None, :],
                    device=device,
                ),
                qualities=torch.tensor(
                    [float(qualities.get(modality, 0.0))], device=device
                ),
                present=torch.tensor([present], dtype=torch.bool, device=device),
            )

        fused, weights = self.forward(batch)
        return (
            fused.squeeze(0).cpu().numpy(),
            {m: float(weights[0, i]) for i, m in enumerate(self.modalities)},
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "dims": {m.value: d for m, d in self.dims.items()},
                "shared_dim": self.shared_dim,
                "trained": self._trained,
            },
            path,
        )
        logger.info("Saved attention head to %s", path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "KeylessAttentionFusion":
        # weights_only=True: a head loaded from disk is data, not code. The
        # payload is a state dict plus plain ints and strings, all of which the
        # restricted loader handles.
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            dims={Modality(k): v for k, v in payload["dims"].items()},
            shared_dim=payload["shared_dim"],
        )
        model.load_state_dict(payload["state_dict"])
        model._trained = bool(payload.get("trained", True))
        model.to(device)
        logger.info("Loaded attention head from %s", path)
        return model


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Triplet margin loss on cosine distance.

    Cosine rather than Euclidean because matching is done by cosine similarity
    against the gallery -- training on a different metric than you evaluate on
    optimises the wrong geometry.

    Triplet rather than an ArcFace-style margin softmax because this is an
    open-set problem. A softmax over enrolled identities learns a fixed roster,
    and the watchlist grows.
    """
    positive_distance = 1.0 - (anchor * positive).sum(dim=1)
    negative_distance = 1.0 - (anchor * negative).sum(dim=1)
    return torch.clamp(positive_distance - negative_distance + margin, min=0.0).mean()
