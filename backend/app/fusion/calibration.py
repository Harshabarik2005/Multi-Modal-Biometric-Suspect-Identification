"""Putting the three modalities on a comparable scale.

Raw cosine similarities from the three branches are **not** comparable, and
averaging them directly is the single easiest way to build a fusion stage that
looks reasonable and is wrong. Measured on real crops and synthetic gait:

| modality | different people | same person | gap   |
|----------|------------------|-------------|-------|
| face     | 0.03             | 0.95        | 0.92  |
| re-ID    | 0.755            | 0.980       | 0.225 |
| gait     | 0.52 (centred)   | 1.00        | 0.48  |

A raw mean would let re-ID and gait dominate every fused score purely by living
in a higher numeric range -- a face score of 0.60, which is a strong
identification, would be dragged down by a re-ID score of 0.70, which is
nothing at all.

`ModalityCalibration` maps each modality's raw similarity onto a common [0, 1]
scale anchored on two reference points: the similarity a *different* person
typically scores, and the similarity the *same* person typically scores. A
calibrated 0.0 means "indistinguishable from a stranger", 1.0 means "as good as
a genuine match gets", and 0.5 means "halfway between" for every modality alike.

This is a linear rescale, not a probability. It does not claim a calibrated 0.7
means a 70% chance of identity. Real probability calibration needs labelled
pairs, which is Phase 10's job; these anchors are measurements from a handful of
clips and should be re-derived from the TAR@FAR curve on real footage.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.types import Modality

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModalityCalibration:
    """Maps one modality's raw cosine similarity onto a common [0, 1] scale."""

    modality: Modality
    #: Similarity two *different* people typically score. Maps to 0.0.
    impostor_anchor: float
    #: Similarity the *same* person typically scores. Maps to 1.0.
    genuine_anchor: float
    #: Why this modality may not vote yet, or "" when it may.
    #:
    #: For anchors that have never been measured on the real distribution. A
    #: score calibrated against them is not evidence, but it is still a number,
    #: and fusion would weigh it as one. The modality is still compared, and
    #: the reviewer is told it was not counted and why; it just carries no
    #: weight in the score.
    withheld_reason: str = ""

    def __post_init__(self) -> None:
        if self.genuine_anchor <= self.impostor_anchor:
            raise ValueError(
                f"{self.modality.value}: genuine_anchor "
                f"({self.genuine_anchor}) must exceed impostor_anchor "
                f"({self.impostor_anchor}); otherwise higher similarity would "
                "mean less likely to be the same person."
            )

    @property
    def separation(self) -> float:
        """How much room this modality has between stranger and self.

        A modality with a tiny separation cannot discriminate, whatever its raw
        similarities look like.
        """
        return self.genuine_anchor - self.impostor_anchor

    def calibrate(self, similarity: float) -> float:
        """Raw similarity -> [0, 1], clipped at both ends.

        Clipping is deliberate. Scores beyond the anchors carry no extra
        information -- a face similarity of 0.99 is not meaningfully better
        evidence than 0.95 -- and letting them run past 1.0 would let one
        modality outvote the rest through sheer numeric size.
        """
        scaled = (similarity - self.impostor_anchor) / self.separation
        return float(min(1.0, max(0.0, scaled)))


def default_calibrations(settings) -> dict[Modality, ModalityCalibration]:
    """Build the calibration set from config."""
    cfg = settings.fusion

    # Anchors belong to a specific checkpoint. Measured on one model they say
    # nothing about another, so a mismatch is reported here -- at the point
    # they are turned into a scale -- rather than left to be discovered in a
    # match score that looks fine (DES-01).
    mismatch = settings.reid_calibration_mismatch()
    if mismatch:
        logger.warning("%s", mismatch)

    gait_withheld = (
        ""
        if cfg.gait_anchors_validated
        else "gait scoring has only been calibrated on synthetic walkers, "
        "not on real footage"
    )

    return {
        Modality.FACE: ModalityCalibration(
            Modality.FACE, cfg.face_impostor, cfg.face_genuine
        ),
        Modality.GAIT: ModalityCalibration(
            Modality.GAIT,
            cfg.gait_impostor,
            cfg.gait_genuine,
            withheld_reason=gait_withheld,
        ),
        Modality.REID: ModalityCalibration(
            Modality.REID, cfg.reid_impostor, cfg.reid_genuine
        ),
    }
