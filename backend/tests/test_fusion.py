"""Tests for calibration and baseline fusion (Phase 5).

All pure logic -- no model weights, no video.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.core.types import Modality, ModalityEmbedding, l2_normalize
from app.fusion.baseline import (
    AverageFusion,
    FusionInput,
    QualityWeightedFusion,
    SingleBestFusion,
    build_strategy,
)
from app.fusion.calibration import ModalityCalibration, default_calibrations
from app.matching.gallery import Gallery, PersonRecord

CALIBRATIONS = {
    Modality.FACE: ModalityCalibration(Modality.FACE, 0.03, 0.95),
    Modality.GAIT: ModalityCalibration(Modality.GAIT, 0.52, 0.95),
    Modality.REID: ModalityCalibration(Modality.REID, 0.755, 0.980),
}


class TestCalibration:
    def test_anchors_map_to_zero_and_one(self) -> None:
        for calibration in CALIBRATIONS.values():
            assert calibration.calibrate(calibration.impostor_anchor) == pytest.approx(0.0)
            assert calibration.calibrate(calibration.genuine_anchor) == pytest.approx(1.0)

    def test_scores_are_clipped_at_both_ends(self) -> None:
        calibration = CALIBRATIONS[Modality.FACE]
        assert calibration.calibrate(-1.0) == 0.0
        assert calibration.calibrate(1.0) == 1.0

    def test_puts_different_modalities_on_a_comparable_scale(self) -> None:
        """The whole point: equal evidence must calibrate to equal scores.

        A face similarity of 0.03 and a re-ID similarity of 0.755 both mean
        "indistinguishable from a stranger", despite being wildly different
        numbers. Fusing them raw would let re-ID dominate every score.
        """
        face = CALIBRATIONS[Modality.FACE].calibrate(0.03)
        reid = CALIBRATIONS[Modality.REID].calibrate(0.755)
        assert face == pytest.approx(reid, abs=1e-6) == pytest.approx(0.0)

    def test_the_impostor_that_would_have_matched_raw_calibrates_near_zero(self) -> None:
        """Measured: a real impostor scored 0.771 by re-ID.

        Raw, that is a high-looking number. Calibrated it is ~0.07, which is
        what stops it being read as a match.
        """
        assert CALIBRATIONS[Modality.REID].calibrate(0.771) < 0.10

    def test_separation_reflects_discriminative_power(self) -> None:
        assert CALIBRATIONS[Modality.FACE].separation > CALIBRATIONS[Modality.GAIT].separation
        assert CALIBRATIONS[Modality.GAIT].separation > CALIBRATIONS[Modality.REID].separation

    def test_inverted_anchors_are_rejected(self) -> None:
        """Genuine below impostor would invert the meaning of similarity."""
        with pytest.raises(ValueError, match="must exceed"):
            ModalityCalibration(Modality.FACE, 0.9, 0.1)

    def test_defaults_load_from_config(self) -> None:
        calibrations = default_calibrations(get_settings())
        assert set(calibrations) == set(Modality)


def face(sim, quality=0.8, trust=1.0):
    return FusionInput(Modality.FACE, sim, quality, trust)


def gait(sim, quality=0.5, trust=1.0):
    return FusionInput(Modality.GAIT, sim, quality, trust)


def reid(sim, quality=0.6, trust=1.0):
    return FusionInput(Modality.REID, sim, quality, trust)


class TestSingleBest:
    def _strategy(self) -> SingleBestFusion:
        return SingleBestFusion(CALIBRATIONS)

    def test_uses_face_when_available(self) -> None:
        result = self._strategy().fuse([face(0.95), reid(0.98)])
        assert result.weights == {Modality.FACE: 1.0}
        assert result.score == pytest.approx(1.0)

    def test_falls_back_to_gait_then_reid(self) -> None:
        strategy = self._strategy()
        assert strategy.fuse([face(None), gait(0.95), reid(0.98)]).weights == {
            Modality.GAIT: 1.0
        }
        assert strategy.fuse([face(None), gait(None), reid(0.98)]).weights == {
            Modality.REID: 1.0
        }

    def test_priority_is_reliability_not_score(self) -> None:
        """A weak modality must not win by scoring generously.

        Re-ID at its ceiling calibrates to 1.0; a mediocre face calibrates
        lower. Face still wins, because it is the more trustworthy signal.
        """
        result = self._strategy().fuse([face(0.5), reid(0.99)])
        assert result.weights == {Modality.FACE: 1.0}
        assert result.score < 1.0

    def test_nothing_available_scores_zero(self) -> None:
        result = self._strategy().fuse([face(None), gait(None), reid(None)])
        assert result.score == 0.0
        assert result.weights == {}


class TestAverageFusion:
    def test_equal_weights_across_available_modalities(self) -> None:
        result = AverageFusion(CALIBRATIONS).fuse([face(0.95), reid(0.980)])
        assert result.weights == {Modality.FACE: 0.5, Modality.REID: 0.5}
        assert result.score == pytest.approx(1.0)

    def test_missing_modalities_are_excluded_not_zeroed(self) -> None:
        """A face that could not be seen is not evidence against a match.

        Scoring it 0 would penalise exactly the situation this project exists
        to handle -- someone whose face is hidden.
        """
        strategy = AverageFusion(CALIBRATIONS)
        excluded = strategy.fuse([face(None), reid(0.980)])
        zeroed = strategy.fuse([face(0.03), reid(0.980)])
        assert excluded.score == pytest.approx(1.0)
        assert zeroed.score == pytest.approx(0.5)
        assert excluded.score > zeroed.score

    def test_corroboration_beats_a_single_modality(self) -> None:
        strategy = AverageFusion(CALIBRATIONS)
        alone = strategy.fuse([face(0.49), gait(None), reid(None)])
        both = strategy.fuse([face(0.49), reid(0.98)])
        assert both.score > alone.score


class TestQualityWeightedFusion:
    def test_a_clearer_look_carries_more_weight(self) -> None:
        strategy = QualityWeightedFusion(CALIBRATIONS)
        result = strategy.fuse([face(0.95, quality=0.9), reid(0.755, quality=0.1)])
        assert result.weights[Modality.FACE] > result.weights[Modality.REID]
        # Dominated by the good face, so close to its calibrated 1.0.
        assert result.score > 0.85

    def test_weights_sum_to_one(self) -> None:
        result = QualityWeightedFusion(CALIBRATIONS).fuse(
            [face(0.8, 0.7), gait(0.8, 0.3), reid(0.9, 0.5)]
        )
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_stale_reid_reference_is_downweighted(self) -> None:
        """Re-ID mostly encodes clothing, so an old reference means less."""
        strategy = QualityWeightedFusion(CALIBRATIONS)
        fresh = strategy.fuse([face(0.5, 0.5), reid(0.98, 0.5, trust=1.0)])
        stale = strategy.fuse([face(0.5, 0.5), reid(0.98, 0.5, trust=0.1)])
        assert stale.weights[Modality.REID] < fresh.weights[Modality.REID]
        assert stale.score < fresh.score

    def test_all_zero_quality_falls_back_to_average(self) -> None:
        """No basis to rank them is not a reason to divide by zero."""
        result = QualityWeightedFusion(CALIBRATIONS).fuse(
            [face(0.95, quality=0.0), reid(0.980, quality=0.0)]
        )
        assert result.score == pytest.approx(1.0)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_nothing_available_scores_zero(self) -> None:
        assert QualityWeightedFusion(CALIBRATIONS).fuse([face(None)]).score == 0.0


class TestFusionResult:
    def test_explain_names_each_contributing_modality(self) -> None:
        result = AverageFusion(CALIBRATIONS).fuse([face(0.95), reid(0.98)])
        text = result.explain()
        assert "face=" in text and "reid=" in text and "w0.50" in text

    def test_explain_when_nothing_compared(self) -> None:
        assert "no modality" in AverageFusion(CALIBRATIONS).fuse([face(None)]).explain()


class TestBuildStrategy:
    def test_builds_each_known_strategy(self) -> None:
        for name in ("single_best", "average", "quality_weighted"):
            assert build_strategy(name, CALIBRATIONS).name == name

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            build_strategy("magic", CALIBRATIONS)


class TestGaitPopulationCentring:
    """Gait descriptors are dominated by the shared 'generic human' shape.

    Measured on synthetic walkers: raw cosine between different walking styles
    was 0.986 against 1.000 for the same style, a separation of 0.014. Removing
    the population mean took that to 0.484.
    """

    def _person(self, pid: str, vector) -> PersonRecord:
        return PersonRecord(
            pid,
            pid.title(),
            {
                Modality.GAIT: ModalityEmbedding(
                    Modality.GAIT, l2_normalize(np.array(vector, dtype=np.float32)), 0.7
                )
            },
        )

    def _gallery(self, n: int) -> Gallery:
        # Vectors sharing a large common component, like real GEI descriptors.
        common = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        people = []
        for i in range(n):
            distinct = np.zeros(4, dtype=np.float32)
            distinct[i % 4] = 0.35
            people.append(self._person(f"p{i}", common + distinct))
        return Gallery(people)

    def test_no_mean_when_too_few_references(self) -> None:
        assert self._gallery(2).gait_population_mean(3) is None

    def test_mean_available_once_enough_references_exist(self) -> None:
        mean = self._gallery(4).gait_population_mean(3)
        assert mean is not None
        assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-5)

    def test_gait_comparison_is_refused_without_a_mean(self) -> None:
        """Refusing beats returning a misleadingly high number.

        Uncentred gait similarities sit above 0.93 for everyone, so a returned
        score would look like a confident match on no evidence.
        """
        gallery = self._gallery(2)
        probe = ModalityEmbedding(
            Modality.GAIT, l2_normalize(np.array([1.0, 1.35, 1.0, 1.0])), 0.7
        )
        candidates = gallery.rank({Modality.GAIT: probe}, gait_min_references=3)
        assert all(c.scores[Modality.GAIT].similarity is None for c in candidates)

    def test_centring_separates_people_that_raw_similarity_cannot(self) -> None:
        gallery = self._gallery(4)
        mean = gallery.gait_population_mean(3)
        assert mean is not None

        common = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        a = l2_normalize(common + np.array([0.35, 0, 0, 0], dtype=np.float32))
        b = l2_normalize(common + np.array([0, 0.35, 0, 0], dtype=np.float32))

        from app.core.types import cosine_similarity

        raw = cosine_similarity(a, b)
        centred = cosine_similarity(
            Gallery.remove_population_component(a, mean),
            Gallery.remove_population_component(b, mean),
        )
        assert raw > 0.9, "raw similarity is dominated by the shared component"
        assert centred < raw, f"centring must increase separation ({centred} !< {raw})"

    def test_centred_output_is_unit_length(self) -> None:
        mean = self._gallery(4).gait_population_mean(3)
        vector = l2_normalize(np.array([1.0, 1.3, 1.0, 1.0], dtype=np.float32))
        out = Gallery.remove_population_component(vector, mean)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


class TestGalleryFusionIntegration:
    def _gallery(self) -> Gallery:
        def person(pid, face_vec, reid_vec):
            return PersonRecord(
                pid,
                pid.title(),
                {
                    Modality.FACE: ModalityEmbedding(
                        Modality.FACE, l2_normalize(np.array(face_vec, dtype=np.float32)), 0.8
                    ),
                    Modality.REID: ModalityEmbedding(
                        Modality.REID, l2_normalize(np.array(reid_vec, dtype=np.float32)), 0.6
                    ),
                },
            )

        return Gallery([person("alice", [1, 0, 0], [1, 0, 0]),
                        person("bob", [0, 1, 0], [0, 1, 0])])

    def _probes(self):
        return {
            Modality.FACE: ModalityEmbedding(
                Modality.FACE, l2_normalize(np.array([1.0, 0.05, 0.0])), 0.9
            ),
            Modality.REID: ModalityEmbedding(
                Modality.REID, l2_normalize(np.array([1.0, 0.1, 0.0])), 0.5
            ),
        }

    def test_ranks_the_right_person_first_under_every_strategy(self) -> None:
        gallery = self._gallery()
        for name in ("single_best", "average", "quality_weighted"):
            strategy = build_strategy(name, CALIBRATIONS)
            ranked = gallery.rank(self._probes(), strategy=strategy)
            assert ranked[0].person.person_id == "alice", name
            assert ranked[0].fused_similarity > ranked[1].fused_similarity, name

    def test_fusion_breakdown_is_attached_for_the_audit_log(self) -> None:
        ranked = self._gallery().rank(
            self._probes(), strategy=build_strategy("average", CALIBRATIONS)
        )
        assert ranked[0].fusion is not None
        assert ranked[0].fusion.strategy == "average"
        assert set(ranked[0].weights) == {Modality.FACE, Modality.REID}

    def test_ranking_without_a_strategy_leaves_scores_unfused(self) -> None:
        ranked = self._gallery().rank(self._probes())
        assert ranked[0].fusion is None
        assert ranked[0].scores[Modality.FACE].similarity is not None

    def test_scores_stay_within_the_calibrated_range(self) -> None:
        for name in ("single_best", "average", "quality_weighted"):
            ranked = self._gallery().rank(
                self._probes(), strategy=build_strategy(name, CALIBRATIONS)
            )
            for candidate in ranked:
                assert 0.0 <= candidate.fused_similarity <= 1.0


class TestTheAuditRecordsTheRuleThatRan:
    """LOG-12: a fallback recorded the strategy that was asked for.

    QualityWeightedFusion falls back to a plain average when every modality
    scored zero quality. The audit trail recorded "quality_weighted" anyway, so
    it named a rule that was never applied -- in the one record a reviewer has
    for reconstructing how a decision was reached.
    """

    def test_the_fallback_names_itself(self) -> None:
        from app.core.types import Modality
        from app.fusion.baseline import FusionInput, QualityWeightedFusion
        from app.fusion.calibration import ModalityCalibration

        calibrations = {
            Modality.FACE: ModalityCalibration(Modality.FACE, 0.03, 0.95),
            Modality.REID: ModalityCalibration(Modality.REID, 0.726, 0.881),
        }
        strategy = QualityWeightedFusion(calibrations)

        # Every modality at zero quality: the weighting has nothing to work
        # with and an equal-weight average is used instead.
        result = strategy.fuse(
            [
                FusionInput(modality=Modality.FACE, similarity=0.9, quality=0.0),
                FusionInput(modality=Modality.REID, similarity=0.8, quality=0.0),
            ]
        )
        assert result.strategy == "average", (
            "the record would claim a rule that never ran"
        )

    def test_the_normal_path_still_names_itself(self) -> None:
        from app.core.types import Modality
        from app.fusion.baseline import FusionInput, QualityWeightedFusion
        from app.fusion.calibration import ModalityCalibration

        strategy = QualityWeightedFusion(
            {Modality.FACE: ModalityCalibration(Modality.FACE, 0.03, 0.95)}
        )
        result = strategy.fuse(
            [FusionInput(modality=Modality.FACE, similarity=0.9, quality=0.8)]
        )
        assert result.strategy == "quality_weighted"
