"""Ablation harness (Phase 10).

Mirrors Table I of the reference paper: each modality alone, then naive /
average fusion, then quality-weighted fusion, then keyless attention -- all on
identical splits. This is the table that shows whether the attention head earns
its complexity, and it is the central claim of the project.

Two things this harness enforces, because an ablation that gets either wrong
proves nothing:

**Identical splits across every row** -- and, more subtly, identical
*coverage*. Every configuration is offered the same pairs, but a single-modality
row can only score the pairs where that modality is present on both sides. Face
is often hidden, so "face only" ends up graded on the subset where a face was
visible: exactly the easy cases. Measured on synthetic data, face-only scored
0.978 AUC over 6,903 scorable pairs while fusion scored 0.697 over 18,949 -- and
reading that as "fusion is worse" would be flatly wrong, because the two numbers
describe different populations.

So `run_ablation` reports both views. The *native* table shows each row on the
pairs it can actually score, which is the honest picture of how useful each
configuration is in deployment. The *common-subset* table restricts every row to
pairs where all modalities are present, which is the only apples-to-apples AUC
comparison. Neither alone is sufficient: the first conflates coverage with
quality, and the second measures a population the system does not actually face.

**Honest baselines.** The single-modality and fixed-fusion rows use the real
implementations from phases 2-5, not weakened stand-ins. If attention wins, it
has to win against the strongest fixed rule available -- which measurement in
Phase 5 showed is `quality_weighted`, not the paper's `average`.

The result object reports which row won and by how much, so "attention is
better" is a number rather than an assertion.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.types import Modality  # noqa: E402
from app.fusion.baseline import FusionInput, build_strategy  # noqa: E402
from app.fusion.calibration import ModalityCalibration  # noqa: E402
from eval.metrics import (  # noqa: E402
    DEFAULT_FARS,
    VerificationReport,
    evaluate_verification,
)


@dataclass
class Observation:
    """One sighting: whichever modalities fired, their vectors and qualities."""

    label: str
    embeddings: dict[Modality, np.ndarray] = field(default_factory=dict)
    qualities: dict[Modality, float] = field(default_factory=dict)
    group: str = ""

    def has(self, modality: Modality) -> bool:
        return modality in self.embeddings


@dataclass
class Pair:
    """Two sightings and whether they are the same person."""

    a: Observation
    b: Observation
    same: bool

    @property
    def group(self) -> str:
        return self.a.group or self.b.group


@dataclass
class AblationRow:
    name: str
    report: VerificationReport | None = None
    comparable_pairs: int = 0
    skipped_pairs: int = 0
    note: str = ""

    @property
    def coverage(self) -> float:
        """Fraction of offered pairs this configuration could actually score.

        A modality that is rarely available has low coverage, and its AUC
        describes only the pairs it happened to get.
        """
        total = self.comparable_pairs + self.skipped_pairs
        return self.comparable_pairs / total if total else 0.0

    @property
    def auc(self) -> float:
        return self.report.auc if self.report else 0.0

    def tar(self, far: float) -> float:
        return self.report.tar_at_far.get(far, 0.0) if self.report else 0.0


@dataclass
class AblationResult:
    rows: list[AblationRow] = field(default_factory=list)
    headline_far: float = 0.001

    def best(self) -> AblationRow | None:
        scored = [r for r in self.rows if r.report is not None]
        return max(scored, key=lambda r: r.auc) if scored else None

    def row(self, name: str) -> AblationRow | None:
        return next((r for r in self.rows if r.name == name), None)

    def table(self, note_coverage: bool = True) -> str:
        fars = sorted(DEFAULT_FARS, reverse=True)
        header = f"{'configuration':<24} {'AUC':>7} {'EER':>7}"
        header += "".join(f"  TAR@{f:<8g}" for f in fars)
        header += f" {'pairs':>7} {'cover':>6}"
        lines = [header, "-" * len(header)]

        for row in self.rows:
            if row.report is None:
                lines.append(f"{row.name:<24} {'-':>7} {'-':>7}   {row.note}")
                continue
            line = (
                f"{row.name:<24} {row.report.auc:>7.4f} {row.report.eer:>7.4f}"
                + "".join(f"  {row.tar(f):>11.4f}" for f in fars)
                + f" {row.comparable_pairs:>7} {row.coverage:>5.0%}"
            )
            lines.append(line)

        if note_coverage:
            spread = [r.coverage for r in self.rows if r.report is not None]
            if spread and (max(spread) - min(spread)) > 0.1:
                lines.append("")
                lines.append(
                    "Coverage differs between rows, so these AUCs are NOT directly\n"
                    "comparable: a row that only scores the pairs where its modality\n"
                    "was visible is being graded on the easy cases. See the\n"
                    "common-subset table below for the like-for-like comparison."
                )

        winner = self.best()
        if winner:
            lines.append("")
            lines.append(f"Best by AUC: {winner.name} ({winner.auc:.4f})")

            attention = self.row("attention")
            quality = self.row("fusion:quality_weighted")
            if attention and quality and attention.report and quality.report:
                delta = attention.auc - quality.auc
                verdict = (
                    "attention beats the strongest fixed rule"
                    if delta > 0
                    else "attention does NOT beat the strongest fixed rule"
                )
                lines.append(f"attention - quality_weighted = {delta:+.4f}  ({verdict})")
        return "\n".join(lines)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)


def score_single_modality(pairs: list[Pair], modality: Modality):
    """Genuine/impostor scores using one modality alone.

    Pairs where either side lacks the modality are *skipped*, not scored zero.
    Scoring them zero would measure availability rather than discriminative
    power, and would make face look terrible purely for being often hidden.
    """
    genuine, impostor, skipped = [], [], 0
    for pair in pairs:
        if not (pair.a.has(modality) and pair.b.has(modality)):
            skipped += 1
            continue
        score = _cosine(pair.a.embeddings[modality], pair.b.embeddings[modality])
        (genuine if pair.same else impostor).append(score)
    return np.array(genuine), np.array(impostor), skipped


def score_fusion(
    pairs: list[Pair],
    strategy_name: str,
    calibrations: dict[Modality, ModalityCalibration],
):
    """Genuine/impostor scores using one of the Phase-5 fixed rules."""
    strategy = build_strategy(strategy_name, calibrations)
    genuine, impostor, skipped = [], [], 0

    for pair in pairs:
        inputs = []
        for modality in calibrations:
            if pair.a.has(modality) and pair.b.has(modality):
                similarity = _cosine(
                    pair.a.embeddings[modality], pair.b.embeddings[modality]
                )
                quality = min(
                    pair.a.qualities.get(modality, 0.0),
                    pair.b.qualities.get(modality, 0.0),
                )
            else:
                similarity, quality = None, 0.0
            inputs.append(FusionInput(modality, similarity, quality))

        result = strategy.fuse(inputs)
        if not result.weights:
            skipped += 1
            continue
        (genuine if pair.same else impostor).append(result.score)
    return np.array(genuine), np.array(impostor), skipped


def score_attention(pairs: list[Pair], model):
    """Genuine/impostor scores using the trained attention head."""
    if not getattr(model, "is_trained", False):
        raise RuntimeError(
            "The attention head is untrained. Including it in an ablation "
            "would compare the fixed rules against noise."
        )

    genuine, impostor, skipped = [], [], 0
    for pair in pairs:
        try:
            a_vector, _ = model.fuse_one(pair.a.embeddings, pair.a.qualities)
            b_vector, _ = model.fuse_one(pair.b.embeddings, pair.b.qualities)
        except Exception:  # noqa: BLE001 - dimension mismatch, empty inputs
            skipped += 1
            continue
        (genuine if pair.same else impostor).append(_cosine(a_vector, b_vector))
    return np.array(genuine), np.array(impostor), skipped


def common_subset(
    pairs: list[Pair], modalities: tuple[Modality, ...] = tuple(Modality)
) -> list[Pair]:
    """Pairs where every modality is present on both sides.

    The only population on which cross-row AUCs are comparable. It is also
    an unrepresentatively easy one -- in real footage a face is often missing,
    which is the entire premise of the project -- so it answers "which fusion
    rule is best when everything is available", not "how well does this work".
    """
    return [
        pair
        for pair in pairs
        if all(pair.a.has(m) and pair.b.has(m) for m in modalities)
    ]


def run_ablation(
    pairs: list[Pair],
    calibrations: dict[Modality, ModalityCalibration],
    attention_model=None,
    fars: tuple[float, ...] = DEFAULT_FARS,
) -> AblationResult:
    """Run every configuration over identical pairs."""
    result = AblationResult()

    def add(name: str, scored) -> None:
        genuine, impostor, skipped = scored
        row = AblationRow(
            name=name,
            comparable_pairs=int(genuine.size + impostor.size),
            skipped_pairs=int(skipped),
        )
        if genuine.size == 0 or impostor.size == 0:
            row.note = (
                f"not scorable: {genuine.size} genuine, {impostor.size} impostor"
            )
        else:
            row.report = evaluate_verification(genuine, impostor, fars=fars)
        result.rows.append(row)

    for modality in (Modality.FACE, Modality.GAIT, Modality.REID):
        add(f"{modality.value} only", score_single_modality(pairs, modality))

    for strategy in ("single_best", "average", "quality_weighted"):
        add(f"fusion:{strategy}", score_fusion(pairs, strategy, calibrations))

    if attention_model is not None:
        add("attention", score_attention(pairs, attention_model))
    else:
        result.rows.append(
            AblationRow(
                name="attention",
                note="no trained head supplied -- see docs/phase-notes.md",
            )
        )

    return result


def build_pairs(
    observations: list[Observation], max_pairs: int | None = None, seed: int = 0
) -> list[Pair]:
    """All pairwise comparisons, optionally subsampled.

    Subsampling keeps the genuine/impostor ratio of the full set rather than
    sampling uniformly, because impostor pairs vastly outnumber genuine ones
    and a uniform sample would leave too few genuine pairs to measure TAR.
    """
    genuine_pairs, impostor_pairs = [], []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a, b = observations[i], observations[j]
            pair = Pair(a, b, a.label == b.label)
            (genuine_pairs if pair.same else impostor_pairs).append(pair)

    if max_pairs is None or len(genuine_pairs) + len(impostor_pairs) <= max_pairs:
        return genuine_pairs + impostor_pairs

    generator = np.random.default_rng(seed)
    total = len(genuine_pairs) + len(impostor_pairs)
    keep_genuine = max(1, int(max_pairs * len(genuine_pairs) / total))
    keep_impostor = max(1, max_pairs - keep_genuine)

    def sample(items, count):
        if count >= len(items):
            return items
        return [items[i] for i in generator.choice(len(items), count, replace=False)]

    return sample(genuine_pairs, keep_genuine) + sample(impostor_pairs, keep_impostor)
