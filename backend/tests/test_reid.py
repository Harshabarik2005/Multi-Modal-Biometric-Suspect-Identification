"""Tests for the re-ID branch.

Preprocessing, quality scoring and trust decay are pure logic and run always.
Anything needing OSNet weights is marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.core.types import Modality, TrackObservation
from app.embeddings.reid import ReIDEmbedder, trust_at


def body_crop(height: int = 240, width: int = 96, colour=(120, 90, 70)) -> np.ndarray:
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, :] = colour
    crop[height // 3 :, :] = [max(0, c - 40) for c in colour]
    return crop


def observation(
    height: int = 240, width: int = 96, confidence: float = 0.9
) -> TrackObservation:
    return TrackObservation(
        frame_index=0,
        timestamp_s=0.0,
        crop=body_crop(height, width),
        box_height=float(height),
        detection_confidence=confidence,
    )


class TestTrustDecay:
    """Re-ID mostly encodes clothing, so a stored reference goes stale."""

    def test_full_trust_on_the_day_of_enrollment(self) -> None:
        assert trust_at(0.0, 3.0) == pytest.approx(1.0)

    def test_half_trust_after_one_half_life(self) -> None:
        assert trust_at(3.0, 3.0) == pytest.approx(0.5)

    def test_quarter_trust_after_two_half_lives(self) -> None:
        assert trust_at(6.0, 3.0) == pytest.approx(0.25)

    def test_decays_toward_zero_but_never_negative(self) -> None:
        assert 0.0 < trust_at(365.0, 3.0) < 0.001

    def test_decay_is_monotonic(self) -> None:
        values = [trust_at(d, 3.0) for d in range(0, 30, 3)]
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_zero_half_life_disables_decay(self) -> None:
        assert trust_at(1000.0, 0.0) == 1.0

    def test_negative_elapsed_time_is_clamped(self) -> None:
        """A clock skew must not manufacture MORE than full trust."""
        assert trust_at(-5.0, 3.0) == pytest.approx(1.0)


class TestQualityScore:
    def _embedder(self) -> ReIDEmbedder:
        # Bypass __init__ so no weights load; only the scoring maths is tested.
        embedder = ReIDEmbedder.__new__(ReIDEmbedder)
        embedder.cfg = get_settings().reid
        return embedder

    def test_taller_boxes_score_higher(self) -> None:
        embedder = self._embedder()
        small, _ = embedder._quality(observation(height=80, width=32))
        large, _ = embedder._quality(observation(height=240, width=96))
        assert large > small

    def test_resolution_saturates_at_the_ideal_height(self) -> None:
        embedder = self._embedder()
        _, detail = embedder._quality(observation(height=400, width=160))
        assert detail["resolution"] == pytest.approx(1.0)

    def test_person_shaped_boxes_beat_odd_ones(self) -> None:
        """A box far from human proportions is a partial or merged body."""
        embedder = self._embedder()
        normal, _ = embedder._quality(observation(height=250, width=100))
        squat, _ = embedder._quality(observation(height=250, width=250))
        assert normal > squat

    def test_zero_detection_confidence_is_neutral_not_disqualifying(self) -> None:
        """Tracker-predicted boxes report 0.0 confidence; that is not evidence
        the crop is bad, so it must not zero the whole quality score."""
        embedder = self._embedder()
        quality, detail = embedder._quality(observation(confidence=0.0))
        assert detail["detection_confidence"] == pytest.approx(0.5)
        assert quality > 0.0

    def test_quality_stays_in_range(self) -> None:
        embedder = self._embedder()
        for height, width, confidence in (
            (10, 200, 0.0), (1000, 10, 1.0), (240, 96, 0.5), (5, 5, 0.9)
        ):
            quality, _ = embedder._quality(observation(height, width, confidence))
            assert 0.0 <= quality <= 1.0


class TestPreprocessing:
    def _embedder(self) -> ReIDEmbedder:
        embedder = ReIDEmbedder.__new__(ReIDEmbedder)
        embedder.cfg = get_settings().reid
        return embedder

    def test_output_shape_matches_osnet_input(self) -> None:
        embedder = self._embedder()
        out = embedder.preprocess(body_crop(300, 140))
        assert out.shape == (3, embedder.cfg.input_height, embedder.cfg.input_width)

    def test_any_input_size_is_accepted(self) -> None:
        embedder = self._embedder()
        for height, width in ((50, 20), (600, 300), (128, 128)):
            assert embedder.preprocess(body_crop(height, width)).shape == (
                3, embedder.cfg.input_height, embedder.cfg.input_width
            )

    def test_normalisation_moves_values_off_the_raw_0_1_range(self) -> None:
        """ImageNet normalisation must actually be applied."""
        embedder = self._embedder()
        out = embedder.preprocess(body_crop())
        assert out.min() < 0.0 or out.max() > 1.0


@pytest.mark.slow
class TestReIDEmbedderEndToEnd:
    @pytest.fixture(scope="class")
    def embedder(self):
        return ReIDEmbedder(get_settings())

    def test_produces_a_unit_length_512d_vector(self, embedder) -> None:
        result = embedder.embed_frame(observation())
        assert result.has_signal
        assert result.vector.size == embedder.embedding_dim == 512
        assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-4)
        assert result.modality is Modality.REID

    def test_degenerate_crop_yields_no_signal(self, embedder) -> None:
        tiny = TrackObservation(
            frame_index=0,
            timestamp_s=0.0,
            crop=np.zeros((4, 2, 3), dtype=np.uint8),
            box_height=4.0,
            detection_confidence=0.9,
        )
        assert not embedder.embed_frame(tiny).has_signal

    def test_identical_crops_score_one(self, embedder) -> None:
        a = embedder.embed_frame(observation())
        b = embedder.embed_frame(observation())
        assert a.similarity(b) == pytest.approx(1.0, abs=1e-4)

    def test_batch_matches_per_frame_results(self, embedder) -> None:
        """`embed_batch` is an optimisation; it must not change the answer."""
        observations = [observation(240, 96), observation(200, 80)]
        batched = embedder.embed_batch(observations)
        for obs, batch_result in zip(observations, batched):
            single = embedder.embed_frame(obs)
            assert single.has_signal == batch_result.has_signal
            if single.has_signal:
                assert single.similarity(batch_result) == pytest.approx(1.0, abs=1e-3)

    def test_batch_handles_a_mix_of_usable_and_unusable_crops(self, embedder) -> None:
        observations = [
            observation(240, 96),
            TrackObservation(1, 0.0, np.zeros((2, 2, 3), np.uint8), 2.0, 0.9),
            observation(220, 88),
        ]
        results = embedder.embed_batch(observations)
        assert len(results) == 3
        assert results[0].has_signal and results[2].has_signal
        assert not results[1].has_signal

    def test_too_few_observations_yields_no_signal(self, embedder) -> None:
        assert not embedder.embed([observation()]).has_signal
