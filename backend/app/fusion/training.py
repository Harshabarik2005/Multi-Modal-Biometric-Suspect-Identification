"""Training the keyless attention head (Phase 6).

Open-set metric learning: the head never learns "who is this", it learns "how
much should each modality be trusted so that same-person pairs end up close and
different-person pairs end up far apart". That is what lets the watchlist grow
without retraining.

Training data is a stream of triplets -- an anchor observation, another
observation of the *same* person, and one of a *different* person -- each
carrying whatever modalities happened to be available, with their qualities.
Crucially the modality availability must vary across the training set: a head
that only ever saw all three present will not learn what to do when the face is
missing, which is the case that matters most in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from app.core.logging import get_logger
from app.core.types import Modality
from app.fusion.attention import (
    KeylessAttentionFusion,
    ModalityBatch,
    triplet_loss,
)

logger = get_logger(__name__)


@dataclass
class Observation:
    """One person seen once: whichever modalities fired, and how well."""

    embeddings: dict[Modality, np.ndarray] = field(default_factory=dict)
    qualities: dict[Modality, float] = field(default_factory=dict)

    def has(self, modality: Modality) -> bool:
        return modality in self.embeddings


@dataclass
class Triplet:
    anchor: Observation
    positive: Observation  # same person as anchor
    negative: Observation  # a different person


def to_batch(
    observations: list[Observation],
    dims: dict[Modality, int],
    modalities: tuple[Modality, ...],
    device: str = "cpu",
) -> dict[Modality, ModalityBatch]:
    """Pack observations into per-modality tensors with presence masks."""
    batch: dict[Modality, ModalityBatch] = {}

    for modality in modalities:
        dim = dims[modality]
        vectors = np.zeros((len(observations), dim), dtype=np.float32)
        qualities = np.zeros(len(observations), dtype=np.float32)
        present = np.zeros(len(observations), dtype=bool)

        for index, observation in enumerate(observations):
            if not observation.has(modality):
                continue
            vectors[index] = observation.embeddings[modality]
            qualities[index] = observation.qualities.get(modality, 0.0)
            present[index] = True

        batch[modality] = ModalityBatch(
            embeddings=torch.from_numpy(vectors).to(device),
            qualities=torch.from_numpy(qualities).to(device),
            present=torch.from_numpy(present).to(device),
        )
    return batch


@dataclass
class TrainingReport:
    epochs: int = 0
    epochs_run: int = 0
    final_loss: float = 0.0
    best_validation_loss: float = float("inf")
    losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    stopped_early: bool = False

    #: Mean attention weight per modality across ALL observations. Misleading
    #: on its own: a modality that is often absent scores low here simply
    #: because absent rows contribute a weight of zero.
    mean_weights: dict[Modality, float] = field(default_factory=dict)
    #: Mean attention weight ONLY over observations where the modality was
    #: actually present. This is the number that answers "how much does the
    #: head trust this modality when it has one?", and it is the one to read.
    mean_weights_when_present: dict[Modality, float] = field(default_factory=dict)
    #: Fraction of observations carrying each modality.
    availability: dict[Modality, float] = field(default_factory=dict)

    #: Separation (same-person minus different-person cosine) on train and
    #: validation. A large gap between them is overfitting, and an overfitted
    #: head is worse in the field than the fixed rules it replaces.
    train_separation: float = 0.0
    validation_separation: float = 0.0

    @property
    def overfitting_ratio(self) -> float:
        """How much better it does on data it trained on. 1.0 is ideal."""
        if abs(self.validation_separation) < 1e-6:
            return float("inf")
        return self.train_separation / self.validation_separation

    def summary_lines(self) -> list[str]:
        lines = [
            f"epochs run     : {self.epochs_run}/{self.epochs}"
            + ("  (early stopped)" if self.stopped_early else ""),
            f"final loss     : {self.final_loss:.4f}",
        ]
        if self.losses:
            lines.append(f"first loss     : {self.losses[0]:.4f}")
        if self.validation_losses:
            lines.append(f"best val loss  : {self.best_validation_loss:.4f}")

        lines.append("")
        lines.append(f"separation train : {self.train_separation:+.4f}")
        lines.append(f"separation val   : {self.validation_separation:+.4f}")
        if self.validation_separation > 0:
            ratio = self.overfitting_ratio
            verdict = (
                "healthy" if ratio < 1.5
                else "some overfitting" if ratio < 3.0
                else "SEVERE overfitting - do not deploy this head"
            )
            lines.append(f"train/val ratio  : {ratio:.2f}x  ({verdict})")

        if self.mean_weights_when_present:
            lines.append("")
            lines.append("attention weight when the modality is present:")
            for modality, weight in sorted(
                self.mean_weights_when_present.items(), key=lambda kv: -kv[1]
            ):
                available = self.availability.get(modality, 0.0)
                overall = self.mean_weights.get(modality, 0.0)
                lines.append(
                    f"  {modality.value:<6} {weight:.3f}   "
                    f"(available {available:.0%}, overall {overall:.3f})"
                )
        return lines


def _epoch_loss(
    model: KeylessAttentionFusion,
    triplets: list[Triplet],
    batch_size: int,
    margin: float,
    device: str,
) -> float:
    """Mean triplet loss over `triplets` without updating anything."""
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(triplets), batch_size):
            chunk = triplets[start : start + batch_size]
            if len(chunk) < 2:
                continue
            anchor, _ = model(
                to_batch([t.anchor for t in chunk], model.dims, model.modalities, device)
            )
            positive, _ = model(
                to_batch([t.positive for t in chunk], model.dims, model.modalities, device)
            )
            negative, _ = model(
                to_batch([t.negative for t in chunk], model.dims, model.modalities, device)
            )
            losses.append(
                float(triplet_loss(anchor, positive, negative, margin=margin).item())
            )
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def train(
    model: KeylessAttentionFusion,
    triplets: list[Triplet],
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    margin: float = 0.3,
    weight_decay: float = 1e-3,
    validation_fraction: float = 0.2,
    patience: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> TrainingReport:
    """Train the head on triplets, holding out a validation split.

    The validation split is not optional garnish. Measured on synthetic data,
    an unregularised run reached a training separation of 0.76 against 0.13 on
    held-out triplets -- a head that had memorised its triplets and learned
    little that transfers. Deployed, that is worse than the Phase-5 fixed rules,
    because it discards their careful calibration in exchange for noise.

    So this holds out `validation_fraction`, applies weight decay, and stops
    when validation loss has not improved for `patience` epochs, restoring the
    best weights seen.
    """
    if not triplets:
        raise ValueError("No triplets supplied; there is nothing to learn from.")

    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)

    shuffled = [triplets[i] for i in generator.permutation(len(triplets))]
    split = int(len(shuffled) * (1.0 - validation_fraction))
    split = max(1, min(split, len(shuffled) - 1)) if len(shuffled) > 1 else 1
    training, validation = shuffled[:split], shuffled[split:]

    model.to(device).train()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    report = TrainingReport(epochs=epochs)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_validation = float("inf")
    since_improvement = 0

    for epoch in range(epochs):
        order = generator.permutation(len(training))
        epoch_losses = []

        for start in range(0, len(order), batch_size):
            chunk = [training[i] for i in order[start : start + batch_size]]
            if len(chunk) < 2:
                continue

            anchor, _ = model(
                to_batch([t.anchor for t in chunk], model.dims, model.modalities, device)
            )
            positive, _ = model(
                to_batch([t.positive for t in chunk], model.dims, model.modalities, device)
            )
            negative, _ = model(
                to_batch([t.negative for t in chunk], model.dims, model.modalities, device)
            )

            loss = triplet_loss(anchor, positive, negative, margin=margin)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.item()))

        if not epoch_losses:
            continue

        mean_loss = float(np.mean(epoch_losses))
        report.losses.append(mean_loss)
        report.epochs_run = epoch + 1

        validation_loss = (
            _epoch_loss(model, validation, batch_size, margin, device)
            if validation
            else mean_loss
        )
        report.validation_losses.append(validation_loss)

        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1

        if epoch % max(1, epochs // 6) == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %d/%d  train %.4f  val %.4f",
                epoch + 1, epochs, mean_loss, validation_loss,
            )

        if since_improvement >= patience:
            logger.info(
                "Early stop at epoch %d: validation loss has not improved for "
                "%d epochs.", epoch + 1, patience,
            )
            report.stopped_early = True
            break

    # Restore the best weights rather than the last, which are usually worse.
    model.load_state_dict(best_state)
    report.best_validation_loss = best_validation
    report.final_loss = report.losses[-1] if report.losses else 0.0
    model.mark_trained()

    report.train_separation = evaluate_separation(model, training, device)["separation"]
    if validation:
        report.validation_separation = evaluate_separation(
            model, validation, device
        )["separation"]

    # What the head actually learned to trust. The present-only figure is the
    # meaningful one: averaging over rows where a modality was absent just
    # measures how often it was missing.
    model.eval()
    with torch.no_grad():
        batch = to_batch(
            [t.anchor for t in shuffled], model.dims, model.modalities, device
        )
        _, weights = model(batch)
        weights = weights.cpu().numpy()

    for index, modality in enumerate(model.modalities):
        present = batch[modality].present.cpu().numpy()
        report.mean_weights[modality] = float(weights[:, index].mean())
        report.availability[modality] = float(present.mean())
        report.mean_weights_when_present[modality] = (
            float(weights[present, index].mean()) if present.any() else 0.0
        )
    return report


# ---------------------------------------------------------------------------
# Synthetic data, for proving the machinery works without real footage
# ---------------------------------------------------------------------------


def synthetic_triplets(
    count: int = 600,
    dims: dict[Modality, int] | None = None,
    face_noise: float = 0.15,
    gait_noise: float = 0.55,
    reid_noise: float = 0.45,
    face_available: float = 0.6,
    gait_available: float = 0.5,
    seed: int = 0,
) -> list[Triplet]:
    """Build triplets where the modalities differ in how informative they are.

    Each person gets a latent identity vector per modality. An observation is
    that vector plus noise, so a modality with *low* noise carries identity
    reliably and one with *high* noise mostly does not.

    The defaults deliberately reproduce the situation measured in Phase 5:
    face is the most informative modality but is often missing, while re-ID is
    nearly always present and much noisier. A correct attention head should
    learn to lean on face when it is there despite its lower availability --
    which is exactly what the fixed rules failed to do.

    Qualities are correlated with the actual noise on each observation, so the
    head has a genuine signal to learn from rather than a constant.
    """
    dims = dims or {Modality.FACE: 64, Modality.GAIT: 64, Modality.REID: 64}
    generator = np.random.default_rng(seed)
    noises = {
        Modality.FACE: face_noise,
        Modality.GAIT: gait_noise,
        Modality.REID: reid_noise,
    }
    availability = {
        Modality.FACE: face_available,
        Modality.GAIT: gait_available,
        Modality.REID: 0.95,
    }

    def identity() -> dict[Modality, np.ndarray]:
        return {
            modality: _unit(generator.normal(size=dims[modality]))
            for modality in dims
        }

    def observe(latent: dict[Modality, np.ndarray]) -> Observation:
        observation = Observation()
        for modality, vector in latent.items():
            if generator.random() > availability[modality]:
                continue
            # Per-observation noise level, so quality is informative.
            scale = noises[modality] * generator.uniform(0.5, 1.5)
            noisy = _unit(vector + generator.normal(scale=scale, size=vector.shape))
            observation.embeddings[modality] = noisy.astype(np.float32)
            # Higher noise -> lower quality, plus a little measurement error.
            observation.qualities[modality] = float(
                np.clip(1.0 - scale + generator.normal(scale=0.05), 0.0, 1.0)
            )
        return observation

    triplets: list[Triplet] = []
    while len(triplets) < count:
        person_a, person_b = identity(), identity()
        anchor, positive, negative = observe(person_a), observe(person_a), observe(person_b)
        # A triplet where some side has nothing at all teaches nothing.
        if not anchor.embeddings or not positive.embeddings or not negative.embeddings:
            continue
        triplets.append(Triplet(anchor, positive, negative))
    return triplets


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def evaluate_separation(
    model: KeylessAttentionFusion,
    triplets: list[Triplet],
    device: str = "cpu",
) -> dict[str, float]:
    """Mean same-person and different-person similarity after fusion.

    The number that matters is the separation. A head that pushes both up
    equally has learned nothing useful.
    """
    model.eval()
    with torch.no_grad():
        anchor, _ = model(
            to_batch([t.anchor for t in triplets], model.dims, model.modalities, device)
        )
        positive, _ = model(
            to_batch([t.positive for t in triplets], model.dims, model.modalities, device)
        )
        negative, _ = model(
            to_batch([t.negative for t in triplets], model.dims, model.modalities, device)
        )
        same = float((anchor * positive).sum(dim=1).mean())
        different = float((anchor * negative).sum(dim=1).mean())

    return {
        "same_person": same,
        "different_person": different,
        "separation": same - different,
    }
