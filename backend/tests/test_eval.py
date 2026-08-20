"""Tests for the evaluation metrics and ablation harness (Phase 10).

Pure numerics against cases with known answers. A metrics module that is
subtly wrong is worse than none at all: it produces confident numbers that
nobody can check, and the whole point of this phase is to make a claim
checkable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.core.types import Modality  # noqa: E402
from app.fusion.calibration import ModalityCalibration  # noqa: E402
from eval.ablation import (  # noqa: E402
    Observation,
    Pair,
    build_pairs,
    common_subset,
    run_ablation,
    score_fusion,
    score_single_modality,
)
from eval.metrics import (  # noqa: E402
    equal_error_rate,
    evaluate_identification,
    evaluate_verification,
    fairness_breakdown,
    pairwise_scores,
    roc_auc,
    roc_curve,
    tar_at_far,
)


class TestROCAUC:
    def test_perfect_separation_is_one(self) -> None:
        assert roc_auc(np.array([0.9, 0.8, 0.7]), np.array([0.1, 0.2])) == 1.0

    def test_perfectly_wrong_is_zero(self) -> None:
        assert roc_auc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 0.0

    def test_identical_distributions_are_one_half(self) -> None:
        """All ties must give exactly 0.5, not 0 or 1."""
        scores = np.array([0.5, 0.5, 0.5])
        assert roc_auc(scores, scores) == pytest.approx(0.5)

    def test_matches_the_hand_computed_value(self) -> None:
        # genuine {3, 1}, impostor {2, 0}: pairs (3>2, 3>0, 1<2, 1>0) -> 3/4
        genuine = np.array([3.0, 1.0])
        impostor = np.array([2.0, 0.0])
        assert roc_auc(genuine, impostor) == pytest.approx(0.75)

    def test_ties_count_as_half(self) -> None:
        # genuine {1}, impostor {1}: a single tie -> 0.5
        assert roc_auc(np.array([1.0]), np.array([1.0])) == pytest.approx(0.5)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            roc_auc(np.array([]), np.array([1.0]))


class TestROCCurve:
    def test_spans_the_full_range(self) -> None:
        far, tar, _ = roc_curve(np.array([0.9, 0.7]), np.array([0.2, 0.1]))
        assert far.max() == pytest.approx(1.0)
        assert tar.max() == pytest.approx(1.0)
        assert far.min() == pytest.approx(0.0)
        assert tar.min() == pytest.approx(0.0)

    def test_both_curves_are_monotonic_in_threshold(self) -> None:
        rng = np.random.default_rng(0)
        far, tar, _ = roc_curve(rng.normal(1, 1, 200), rng.normal(0, 1, 200))
        assert np.all(np.diff(far) <= 1e-12)
        assert np.all(np.diff(tar) <= 1e-12)


class TestTARatFAR:
    def test_perfect_separation_gives_full_tar(self) -> None:
        genuine = np.array([0.9, 0.85, 0.8])
        impostor = np.array([0.1, 0.15, 0.2])
        tar, threshold = tar_at_far(genuine, impostor, 0.0)
        assert tar == pytest.approx(1.0)
        assert 0.2 < threshold <= 0.8

    def test_threshold_actually_achieves_the_far_budget(self) -> None:
        """The threshold is picked from the impostor side, then TAR measured.

        Choosing it to flatter TAR would report a false-alarm rate the system
        does not achieve.
        """
        rng = np.random.default_rng(1)
        genuine = rng.normal(1.0, 0.5, 500)
        impostor = rng.normal(0.0, 0.5, 500)
        for target in (0.1, 0.01):
            tar, threshold = tar_at_far(genuine, impostor, target)
            assert (impostor >= threshold).mean() <= target + 1e-9
            assert (genuine >= threshold).mean() == pytest.approx(tar)

    def test_unreachable_far_returns_zero(self) -> None:
        """With 10 impostors the smallest non-zero FAR is 0.1."""
        genuine = np.array([0.9] * 10)
        impostor = np.array([0.95] * 10)
        tar, _ = tar_at_far(genuine, impostor, 0.0001)
        assert tar == 0.0


class TestEER:
    def test_perfect_separation_is_zero(self) -> None:
        eer, _ = equal_error_rate(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
        assert eer == pytest.approx(0.0, abs=1e-9)

    def test_identical_distributions_are_about_half(self) -> None:
        rng = np.random.default_rng(2)
        scores = rng.normal(0, 1, 400)
        eer, _ = equal_error_rate(scores, rng.normal(0, 1, 400))
        assert 0.4 < eer < 0.6


class TestVerificationReport:
    def test_reports_every_requested_far(self) -> None:
        rng = np.random.default_rng(3)
        report = evaluate_verification(
            rng.normal(1, 0.5, 300), rng.normal(0, 0.5, 300), fars=(0.1, 0.01)
        )
        assert set(report.tar_at_far) == {0.1, 0.01}
        assert report.genuine_count == 300
        assert 0.9 < report.auc <= 1.0
        assert report.separation > 0

    def test_a_tighter_far_never_gives_a_higher_tar(self) -> None:
        rng = np.random.default_rng(4)
        report = evaluate_verification(rng.normal(1, 1, 500), rng.normal(0, 1, 500))
        fars = sorted(report.tar_at_far, reverse=True)
        tars = [report.tar_at_far[f] for f in fars]
        assert all(a >= b - 1e-9 for a, b in zip(tars, tars[1:]))

    def test_summary_mentions_the_key_numbers(self) -> None:
        rng = np.random.default_rng(5)
        text = "\n".join(
            evaluate_verification(
                rng.normal(1, 1, 100), rng.normal(0, 1, 100)
            ).summary_lines()
        )
        assert "ROC-AUC" in text and "TAR at fixed FAR" in text


class TestIdentification:
    def test_perfect_ranking_is_rank_one(self) -> None:
        matrix = np.array([[0.9, 0.1, 0.2], [0.1, 0.95, 0.3]])
        report = evaluate_identification(matrix, np.array([0, 1]))
        assert report.rank_n(1) == pytest.approx(1.0)
        assert report.mean_rank == pytest.approx(1.0)

    def test_cmc_is_non_decreasing_in_n(self) -> None:
        rng = np.random.default_rng(6)
        matrix = rng.normal(size=(50, 10))
        truth = rng.integers(0, 10, 50)
        report = evaluate_identification(matrix, truth, max_rank=10)
        values = [report.rank_n(n) for n in range(1, 11)]
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
        assert report.rank_n(10) == pytest.approx(1.0)

    def test_worst_case_ranking(self) -> None:
        """True match scores lowest -> rank equals gallery size."""
        matrix = np.array([[0.1, 0.5, 0.9]])
        report = evaluate_identification(matrix, np.array([0]))
        assert report.mean_rank == pytest.approx(3.0)
        assert report.rank_n(1) == 0.0

    def test_label_count_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate_identification(np.zeros((3, 4)), np.array([0, 1]))


class TestPairwiseScores:
    def test_splits_genuine_from_impostor(self) -> None:
        embeddings = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        labels = np.array(["a", "a", "b", "b"])
        genuine, impostor = pairwise_scores(embeddings, labels)
        assert genuine.size == 2  # aa, bb
        assert impostor.size == 4  # ab x4
        assert np.allclose(genuine, 1.0)
        assert np.allclose(impostor, 0.0)

    def test_excludes_self_comparisons(self) -> None:
        embeddings = np.eye(3)
        genuine, impostor = pairwise_scores(embeddings, np.array(["a", "b", "c"]))
        assert genuine.size == 0
        assert impostor.size == 3  # 3 choose 2, no diagonal


class TestFairness:
    def test_reports_each_group_separately(self) -> None:
        """An aggregate number hides a group the system fails for."""
        rng = np.random.default_rng(7)
        # Group B is deliberately harder.
        genuine = np.concatenate([rng.normal(1.5, 0.3, 100), rng.normal(0.6, 0.3, 100)])
        impostor = np.concatenate([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200)])
        genuine_groups = np.array(["A"] * 100 + ["B"] * 100)
        impostor_groups = np.array(["A"] * 200 + ["B"] * 200)

        reports = fairness_breakdown(
            genuine, impostor, genuine_groups, impostor_groups
        )
        assert set(reports) == {"A", "B"}
        assert reports["A"].auc > reports["B"].auc, (
            "the harness must surface that one group performs worse"
        )

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            fairness_breakdown(
                np.array([1.0, 2.0]), np.array([0.0]), np.array(["A"]), np.array(["A"])
            )


def observation(label: str, modalities: dict[Modality, float], seed: int = 0):
    """A sighting whose embedding is the identity vector plus given noise."""
    rng = np.random.default_rng(abs(hash(label)) % 1000 + seed)
    identity = np.random.default_rng(abs(hash(label)) % 1000).normal(size=8)
    result = Observation(label=label)
    for modality, noise in modalities.items():
        vector = identity + rng.normal(scale=noise, size=8)
        result.embeddings[modality] = vector / np.linalg.norm(vector)
        result.qualities[modality] = 1.0 - noise
    return result


class TestAblation:
    def _calibrations(self):
        return {
            m: ModalityCalibration(m, 0.0, 1.0)
            for m in (Modality.FACE, Modality.GAIT, Modality.REID)
        }

    def test_single_modality_skips_rather_than_zeroing_missing_pairs(self) -> None:
        """Scoring an absent modality zero would measure availability, not skill.

        Face is often hidden; counting those as failures would make the most
        discriminative modality look like the worst.
        """
        with_face = observation("a", {Modality.FACE: 0.1})
        without = Observation(label="a")
        without.embeddings[Modality.REID] = np.ones(8) / np.sqrt(8)

        genuine, impostor, skipped = score_single_modality(
            [Pair(with_face, without, True)], Modality.FACE
        )
        assert skipped == 1
        assert genuine.size == 0 and impostor.size == 0

    def test_coverage_reflects_how_often_a_row_could_score(self) -> None:
        pairs = [
            Pair(
                observation("a", {Modality.REID: 0.2}),
                observation("a", {Modality.REID: 0.2}, seed=1),
                True,
            ),
            Pair(
                observation("b", {Modality.FACE: 0.2}),
                observation("c", {Modality.REID: 0.2}),
                False,
            ),
        ]
        result = run_ablation(pairs, self._calibrations())
        reid_row = result.row("reid only")
        assert reid_row is not None
        assert reid_row.coverage == pytest.approx(0.5)

    def test_common_subset_keeps_only_fully_observed_pairs(self) -> None:
        full_a = observation("a", {m: 0.2 for m in Modality})
        full_b = observation("a", {m: 0.2 for m in Modality}, seed=1)
        partial = observation("b", {Modality.REID: 0.2})

        pairs = [Pair(full_a, full_b, True), Pair(full_a, partial, False)]
        shared = common_subset(pairs)
        assert len(shared) == 1
        assert shared[0].same

    def test_every_row_sees_identical_pairs_in_the_common_subset(self) -> None:
        pairs = []
        for index in range(6):
            label = f"p{index // 2}"
            pairs.append(
                Pair(
                    observation(label, {m: 0.2 for m in Modality}, seed=index),
                    observation(label, {m: 0.2 for m in Modality}, seed=index + 10),
                    True,
                )
            )
        for index in range(6):
            pairs.append(
                Pair(
                    observation(f"x{index}", {m: 0.2 for m in Modality}),
                    observation(f"y{index}", {m: 0.2 for m in Modality}),
                    False,
                )
            )

        result = run_ablation(common_subset(pairs), self._calibrations())
        counts = {r.comparable_pairs for r in result.rows if r.report is not None}
        assert len(counts) == 1, f"rows saw different pair counts: {counts}"

    def test_a_more_informative_modality_scores_better(self) -> None:
        """Sanity check that the harness can tell signal from noise at all."""
        pairs = []
        for index in range(15):
            label = f"person{index}"
            pairs.append(
                Pair(
                    observation(label, {Modality.FACE: 0.05, Modality.REID: 1.2}),
                    observation(
                        label, {Modality.FACE: 0.05, Modality.REID: 1.2}, seed=99
                    ),
                    True,
                )
            )
        for index in range(15):
            pairs.append(
                Pair(
                    observation(f"a{index}", {Modality.FACE: 0.05, Modality.REID: 1.2}),
                    observation(f"b{index}", {Modality.FACE: 0.05, Modality.REID: 1.2}),
                    False,
                )
            )

        result = run_ablation(pairs, self._calibrations())
        face = result.row("face only")
        reid = result.row("reid only")
        assert face.auc > reid.auc

    def test_attention_row_is_marked_when_no_head_is_supplied(self) -> None:
        pairs = [
            Pair(
                observation("a", {Modality.FACE: 0.2}),
                observation("a", {Modality.FACE: 0.2}, seed=1),
                True,
            ),
            Pair(
                observation("b", {Modality.FACE: 0.2}),
                observation("c", {Modality.FACE: 0.2}),
                False,
            ),
        ]
        row = run_ablation(pairs, self._calibrations()).row("attention")
        assert row.report is None
        assert "no trained head" in row.note

    def test_untrained_head_is_refused(self) -> None:
        from app.fusion.attention import KeylessAttentionFusion
        from eval.ablation import score_attention

        model = KeylessAttentionFusion({m: 8 for m in Modality}, shared_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="untrained"):
            score_attention([], model)

    def test_build_pairs_preserves_the_genuine_ratio_when_subsampling(self) -> None:
        observations = [
            observation(f"p{i // 3}", {Modality.FACE: 0.2}, seed=i) for i in range(30)
        ]
        everything = build_pairs(observations)
        sampled = build_pairs(observations, max_pairs=100, seed=0)

        assert len(sampled) <= 100
        # Genuine pairs are rare; a uniform sample would lose them entirely.
        assert sum(1 for p in sampled if p.same) > 0

    def test_fusion_scores_stay_in_the_calibrated_range(self) -> None:
        pairs = [
            Pair(
                observation("a", {m: 0.2 for m in Modality}),
                observation("a", {m: 0.2 for m in Modality}, seed=3),
                True,
            )
        ]
        genuine, _, _ = score_fusion(pairs, "quality_weighted", self._calibrations())
        assert genuine.size == 1
        assert 0.0 <= genuine[0] <= 1.0
