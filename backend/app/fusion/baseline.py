"""Baseline fusion (Phase 5) -- the number the real contribution has to beat.

Three fixed-rule strategies, mirroring the "average fusion" row of the
reference paper's Table I. Deliberately unlearned: no trainable parameters,
no attention. Phase 6 replaces the weighting with a keyless-attention head and
has to demonstrate it is actually better than these.

That means these must be implemented *honestly*. A baseline that has been
quietly hobbled makes the contribution look good and proves nothing, so each of
these is the strongest version of its idea:

* `SingleBestFusion`   -- trust only the most reliable modality available.
* `AverageFusion`      -- the paper's baseline: equal weight to every modality
                          that has something to say.
* `QualityWeightedFusion` -- weight each modality by its own quality score and
                          re-ID by how stale its reference is.

All three operate on *calibrated* scores (see `calibration.py`). Fusing raw
cosine similarities would let whichever modality happens to produce larger
numbers dominate, regardless of how much it actually knows.

Missing modalities are excluded, not zeroed. A face that could not be seen is
not evidence against a match, and scoring it 0 would actively penalise exactly
the situation this project exists to handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.types import Modality
from app.fusion.calibration import ModalityCalibration

logger = get_logger(__name__)


@dataclass
class FusionInput:
    """One modality's contribution to one candidate comparison."""

    modality: Modality
    #: Raw cosine similarity, or None when the modality could not compare.
    similarity: float | None
    #: How good a look this modality got, in [0, 1].
    quality: float = 0.0
    #: Extra multiplier on this modality's weight, e.g. re-ID staleness decay.
    trust: float = 1.0

    @property
    def available(self) -> bool:
        return self.similarity is not None


@dataclass
class FusionResult:
    """A fused score plus everything needed to explain it.

    The weights are part of the output contract, not an internal detail. The
    dashboard surfaces them so a reviewer can see whether a match was driven by
    a clear face or mostly by a jacket -- which are very different grounds for
    acting on an identification.
    """

    score: float
    weights: dict[Modality, float] = field(default_factory=dict)
    calibrated: dict[Modality, float] = field(default_factory=dict)
    strategy: str = ""

    @property
    def contributing(self) -> list[Modality]:
        return [m for m, w in self.weights.items() if w > 0.0]

    def explain(self) -> str:
        if not self.weights:
            return "no modality could be compared"
        parts = [
            f"{m.value}={self.calibrated.get(m, 0.0):.2f}(w{self.weights[m]:.2f})"
            for m in sorted(self.contributing, key=lambda m: m.value)
        ]
        return f"{self.score:.3f} = " + " + ".join(parts)


class FusionStrategy(ABC):
    """Combines per-modality similarities into one comparable score."""

    name: str

    def __init__(self, calibrations: dict[Modality, ModalityCalibration]) -> None:
        self.calibrations = calibrations

    def _calibrate(self, inputs: list[FusionInput]) -> dict[Modality, float]:
        """Calibrated score per available modality."""
        result: dict[Modality, float] = {}
        for item in inputs:
            if not item.available:
                continue
            calibration = self.calibrations.get(item.modality)
            if calibration is None:
                logger.warning(
                    "No calibration for %s; excluding it from fusion rather "
                    "than fusing an uncalibrated score.",
                    item.modality.value,
                )
                continue
            result[item.modality] = calibration.calibrate(item.similarity)
        return result

    @abstractmethod
    def fuse(self, inputs: list[FusionInput]) -> FusionResult:
        """Combine. Returns score 0.0 with empty weights when nothing compares."""

    def _empty(self) -> FusionResult:
        return FusionResult(score=0.0, strategy=self.name)


class SingleBestFusion(FusionStrategy):
    """Trust one modality: the most reliable one that has a signal.

    The simplest possible baseline, and the behaviour the matching CLI had
    before fusion existed. Its weakness is the point: it throws away
    corroboration. Two modalities each moderately agreeing should be worth more
    than either alone, and this strategy cannot express that.

    Priority follows measured discriminative power -- face, then gait, then
    re-ID -- rather than which happens to score highest, so a weak modality
    cannot win by being generous.
    """

    name = "single_best"
    PRIORITY = (Modality.FACE, Modality.GAIT, Modality.REID)

    def fuse(self, inputs: list[FusionInput]) -> FusionResult:
        calibrated = self._calibrate(inputs)
        if not calibrated:
            return self._empty()

        for modality in self.PRIORITY:
            if modality in calibrated:
                return FusionResult(
                    score=calibrated[modality],
                    weights={modality: 1.0},
                    calibrated=calibrated,
                    strategy=self.name,
                )
        return self._empty()


class AverageFusion(FusionStrategy):
    """Equal weight to every modality that has something to say.

    This is the reference paper's "average fusion" baseline. It corroborates,
    unlike `SingleBestFusion`, but it cannot tell a clear frontal face from a
    glancing one -- every available modality counts the same regardless of how
    good a look it got. That blindness is exactly what the attention head in
    Phase 6 is supposed to fix.
    """

    name = "average"

    def fuse(self, inputs: list[FusionInput]) -> FusionResult:
        calibrated = self._calibrate(inputs)
        if not calibrated:
            return self._empty()

        weight = 1.0 / len(calibrated)
        return FusionResult(
            score=sum(calibrated.values()) / len(calibrated),
            weights={m: weight for m in calibrated},
            calibrated=calibrated,
            strategy=self.name,
        )


class QualityWeightedFusion(FusionStrategy):
    """Weight each modality by how good a look it got, and how fresh it is.

    The strongest fixed rule available without learning anything, and therefore
    the honest bar for Phase 6. It already does something the paper's baseline
    cannot: a clear frontal face outvotes a glancing one.

    What it still cannot do -- and what keyless attention is for -- is learn
    that a *particular combination* of qualities should be trusted differently.
    It applies one fixed formula regardless of context.
    """

    name = "quality_weighted"

    def fuse(self, inputs: list[FusionInput]) -> FusionResult:
        calibrated = self._calibrate(inputs)
        if not calibrated:
            return self._empty()

        raw_weights: dict[Modality, float] = {}
        for item in inputs:
            if item.modality not in calibrated:
                continue
            # Quality of the look, times how far the reference can still be
            # trusted. A stale re-ID reference is downweighted even when the
            # crop itself was excellent.
            raw_weights[item.modality] = max(0.0, item.quality) * max(0.0, item.trust)

        total = sum(raw_weights.values())
        if total <= 0.0:
            # Every available modality scored zero quality. Fall back to an
            # equal-weight average rather than dividing by zero -- the
            # similarities are still real, we just have no basis to rank them.
            return AverageFusion(self.calibrations).fuse(inputs)

        weights = {m: w / total for m, w in raw_weights.items()}
        return FusionResult(
            score=sum(calibrated[m] * w for m, w in weights.items()),
            weights=weights,
            calibrated=calibrated,
            strategy=self.name,
        )


STRATEGIES: dict[str, type[FusionStrategy]] = {
    SingleBestFusion.name: SingleBestFusion,
    AverageFusion.name: AverageFusion,
    QualityWeightedFusion.name: QualityWeightedFusion,
}


def build_strategy(
    name: str, calibrations: dict[Modality, ModalityCalibration]
) -> FusionStrategy:
    strategy = STRATEGIES.get(name)
    if strategy is None:
        raise ValueError(
            f"Unknown fusion strategy {name!r}. Available: "
            + ", ".join(sorted(STRATEGIES))
        )
    return strategy(calibrations)
