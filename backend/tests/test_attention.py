"""Tests for the keyless attention head (Phase 6).

Runs on CPU with small synthetic tensors, so it needs no weights and no video.
The training tests are kept short deliberately -- they check that the machinery
learns *the right thing*, not that it reaches any particular accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.types import Modality
from app.fusion.attention import (
    KeylessAttentionFusion,
    ModalityBatch,
    triplet_loss,
)
from app.fusion.training import (
    Observation,
    evaluate_separation,
    synthetic_triplets,
    to_batch,
    train,
)

DIMS = {Modality.FACE: 16, Modality.GAIT: 24, Modality.REID: 16}


def model(seed: int = 0) -> KeylessAttentionFusion:
    return KeylessAttentionFusion(DIMS, shared_dim=32, seed=seed)


def batch_of(rows: list[dict[Modality, bool]]) -> dict[Modality, ModalityBatch]:
    """Build a batch from a presence pattern, with random embeddings."""
    generator = np.random.default_rng(0)
    observations = []
    for row in rows:
        observation = Observation()
        for modality, present in row.items():
            if present:
                observation.embeddings[modality] = generator.normal(
                    size=DIMS[modality]
                ).astype(np.float32)
                observation.qualities[modality] = 0.7
        observations.append(observation)
    return to_batch(observations, DIMS, tuple(DIMS))


class TestArchitecture:
    def test_output_is_unit_length(self) -> None:
        net = model()
        fused, _ = net(batch_of([{m: True for m in DIMS}] * 4))
        assert fused.shape == (4, 32)
        assert torch.allclose(
            fused.norm(dim=1), torch.ones(4), atol=1e-5
        ), "fused vectors must be L2-normalised for cosine matching"

    def test_weights_sum_to_one_over_present_modalities(self) -> None:
        net = model()
        _, weights = net(batch_of([{m: True for m in DIMS}] * 3))
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_absent_modalities_get_exactly_zero_weight(self) -> None:
        """Absence must not dilute the softmax.

        A missing face is not a low-scoring face. If absent modalities took a
        share of the weight, every observation with a hidden face would have
        its real signals diluted by nothing.
        """
        net = model()
        _, weights = net(
            batch_of([{Modality.FACE: False, Modality.GAIT: True, Modality.REID: True}])
        )
        index = net.modalities.index(Modality.FACE)
        assert weights[0, index].item() == pytest.approx(0.0, abs=1e-6)
        assert weights[0].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_single_modality_takes_all_the_weight(self) -> None:
        net = model()
        _, weights = net(
            batch_of([{Modality.FACE: True, Modality.GAIT: False, Modality.REID: False}])
        )
        index = net.modalities.index(Modality.FACE)
        assert weights[0, index].item() == pytest.approx(1.0, abs=1e-5)

    def test_a_row_with_nothing_does_not_produce_nan(self) -> None:
        """All -inf would softmax to NaN and poison the whole batch's gradients."""
        net = model()
        fused, weights = net(
            batch_of([
                {m: False for m in DIMS},
                {m: True for m in DIMS},
            ])
        )
        assert not torch.isnan(fused).any()
        assert not torch.isnan(weights).any()
        assert weights[0].sum().item() == pytest.approx(0.0, abs=1e-6)
        # The valid row alongside it must still be handled normally.
        assert weights[1].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_handles_modalities_of_different_dimensions(self) -> None:
        """Face 512, re-ID 512, gait 812 -- they cannot be summed directly."""
        net = model()
        assert net.dims[Modality.GAIT] != net.dims[Modality.FACE]
        fused, _ = net(batch_of([{m: True for m in DIMS}]))
        assert fused.shape[1] == net.shared_dim


class TestTrainedGate:
    def test_untrained_head_refuses_to_fuse(self) -> None:
        """An untrained head is worse than the fixed rules it replaces.

        Random projections destroy the calibrated similarities from Phase 5,
        so producing output before training would be actively harmful.
        """
        net = model()
        assert not net.is_trained
        with pytest.raises(RuntimeError, match="has not been trained"):
            net.fuse_one(
                {Modality.FACE: np.zeros(DIMS[Modality.FACE], dtype=np.float32)},
                {Modality.FACE: 0.9},
            )

    def test_training_marks_it_trained(self) -> None:
        net = model()
        train(net, synthetic_triplets(60, DIMS, seed=1), epochs=2, seed=0)
        assert net.is_trained

    def test_fuse_one_works_once_trained(self) -> None:
        net = model()
        train(net, synthetic_triplets(60, DIMS, seed=1), epochs=2, seed=0)
        vector, weights = net.fuse_one(
            {
                Modality.FACE: np.zeros(DIMS[Modality.FACE], dtype=np.float32),
                Modality.REID: np.zeros(DIMS[Modality.REID], dtype=np.float32),
            },
            {Modality.FACE: 0.9, Modality.REID: 0.4},
        )
        assert vector.shape == (net.shared_dim,)
        assert weights[Modality.GAIT] == pytest.approx(0.0, abs=1e-6)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)


class TestTripletLoss:
    def test_zero_when_the_margin_is_already_satisfied(self) -> None:
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[-1.0, 0.0]])
        assert triplet_loss(anchor, positive, negative, margin=0.3).item() == 0.0

    def test_positive_when_the_negative_is_closer(self) -> None:
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[-1.0, 0.0]])
        negative = torch.tensor([[1.0, 0.0]])
        assert triplet_loss(anchor, positive, negative, margin=0.3).item() > 0.0

    def test_is_differentiable(self) -> None:
        anchor = torch.randn(4, 8, requires_grad=True)
        loss = triplet_loss(anchor, torch.randn(4, 8), torch.randn(4, 8))
        loss.backward()
        assert anchor.grad is not None


class TestLearning:
    def test_training_reduces_loss(self) -> None:
        net = model()
        report = train(net, synthetic_triplets(300, DIMS, seed=2), epochs=25, seed=0)
        assert report.losses[-1] < report.losses[0]

    def test_learns_to_favour_the_more_informative_modality(self) -> None:
        """The whole point of Phase 6, and the fix for Phase 5's blind spot.

        The synthetic data gives face far less noise than re-ID but makes
        re-ID much more often available. The fixed quality-weighted rule was
        measured giving re-ID MORE weight than face (0.54 vs 0.46) on real
        data, because it weights by how good a look you got rather than by how
        much the modality is worth. The attention head has to reverse that.

        Uses the ADEQUATE configuration deliberately. Measured across six
        seeds, 64-d modalities with a 128-d shared space and 1500 triplets got
        this right 6/6 times; 16-d modalities with a 64-d shared space and 800
        triplets got it right only 2/6 -- a coin flip. Under-resourced, the
        head does not learn the weighting at all, which is a real deployment
        constraint rather than a quirk of this test.

        Compared present-only: averaging over rows where a modality is absent
        just measures how often it was missing.
        """
        dims = {Modality.FACE: 64, Modality.GAIT: 64, Modality.REID: 64}
        net = KeylessAttentionFusion(dims, shared_dim=128, seed=0)
        report = train(net, synthetic_triplets(1500, dims, seed=100), epochs=80, seed=0)

        face = report.mean_weights_when_present[Modality.FACE]
        reid = report.mean_weights_when_present[Modality.REID]
        assert face > reid, (
            f"face ({face:.3f}) should outweigh re-ID ({reid:.3f}) when both "
            "are present, since face carries far more identity information"
        )

    @pytest.mark.slow
    def test_modality_ranking_is_robust_across_seeds(self) -> None:
        """One passing seed proves nothing; this checks it is not luck.

        Guards against a future change that makes the head learn the right
        answer only occasionally -- which is exactly what an under-resourced
        configuration does.
        """
        dims = {Modality.FACE: 64, Modality.GAIT: 64, Modality.REID: 64}
        wins = 0
        for seed in range(3):
            net = KeylessAttentionFusion(dims, shared_dim=128, seed=seed)
            report = train(
                net, synthetic_triplets(1500, dims, seed=100 + seed),
                epochs=80, seed=seed,
            )
            weights = report.mean_weights_when_present
            wins += weights[Modality.FACE] > weights[Modality.REID]
        assert wins == 3, f"face outweighed re-ID in only {wins}/3 seeds"

    def test_reports_overfitting_rather_than_hiding_it(self) -> None:
        net = model()
        report = train(net, synthetic_triplets(200, DIMS, seed=4), epochs=20, seed=0)
        assert report.train_separation != 0.0
        assert report.validation_separation != 0.0
        assert report.overfitting_ratio > 0
        assert "separation val" in "\n".join(report.summary_lines())

    def test_availability_is_reported_alongside_weights(self) -> None:
        net = model()
        report = train(net, synthetic_triplets(200, DIMS, seed=5), epochs=5, seed=0)
        for modality in net.modalities:
            assert 0.0 <= report.availability[modality] <= 1.0

    def test_is_reproducible_given_the_same_seeds(self) -> None:
        """Seeding only training left initialisation to the ambient RNG.

        The same configuration measured 1.78x and 4.37x overfitting purely
        because the model was constructed at a different point in the program.
        """
        triplets = synthetic_triplets(200, DIMS, seed=6)
        results = []
        for _ in range(2):
            net = KeylessAttentionFusion(DIMS, shared_dim=32, seed=7)
            report = train(net, triplets, epochs=10, seed=0)
            results.append(report.final_loss)
        assert results[0] == pytest.approx(results[1], abs=1e-9)

    def test_rejects_an_empty_training_set(self) -> None:
        with pytest.raises(ValueError, match="nothing to learn"):
            train(model(), [], epochs=1)


class TestPersistence:
    def test_round_trip_preserves_behaviour(self, tmp_path) -> None:
        net = model()
        train(net, synthetic_triplets(80, DIMS, seed=8), epochs=3, seed=0)

        path = tmp_path / "attention.pt"
        net.save(path)
        loaded = KeylessAttentionFusion.load(path)

        assert loaded.is_trained
        assert loaded.dims == net.dims

        batch = batch_of([{m: True for m in DIMS}] * 2)
        net.eval()
        loaded.eval()
        with torch.no_grad():
            original, _ = net(batch)
            restored, _ = loaded(batch)
        assert torch.allclose(original, restored, atol=1e-6)

    def test_loaded_head_counts_as_trained(self, tmp_path) -> None:
        net = model()
        train(net, synthetic_triplets(60, DIMS, seed=9), epochs=2, seed=0)
        path = tmp_path / "a.pt"
        net.save(path)
        assert KeylessAttentionFusion.load(path).is_trained


class TestSyntheticData:
    def test_availability_varies_across_the_set(self) -> None:
        """A head that only saw all three present cannot learn what to do
        when the face is missing -- the case that matters most here."""
        triplets = synthetic_triplets(300, DIMS, seed=10)
        has_face = sum(1 for t in triplets if t.anchor.has(Modality.FACE))
        assert 0 < has_face < len(triplets)

    def test_every_observation_carries_something(self) -> None:
        for triplet in synthetic_triplets(200, DIMS, seed=11):
            assert triplet.anchor.embeddings
            assert triplet.positive.embeddings
            assert triplet.negative.embeddings

    def test_evaluate_separation_reports_all_three_numbers(self) -> None:
        net = model()
        result = evaluate_separation(net, synthetic_triplets(50, DIMS, seed=12))
        assert set(result) == {"same_person", "different_person", "separation"}
        assert result["separation"] == pytest.approx(
            result["same_person"] - result["different_person"], abs=1e-6
        )


class TestAttentionRanking:
    """`Gallery.rank_attention` compares in the fused space, not per modality."""

    def _gallery(self, dims):
        from app.core.types import ModalityEmbedding, l2_normalize
        from app.matching.gallery import PersonRecord
        from app.matching.gallery import Gallery

        generator = np.random.default_rng(0)
        self.alice = generator.normal(size=dims[Modality.FACE])
        self.bob = generator.normal(size=dims[Modality.FACE])

        def record(pid, latent):
            return PersonRecord(
                pid,
                pid.title(),
                {
                    Modality.FACE: ModalityEmbedding(
                        Modality.FACE, l2_normalize(latent.astype(np.float32)), 0.8
                    )
                },
            )

        return Gallery([record("alice", self.alice), record("bob", self.bob)])

    def _probe(self, dims):
        from app.core.types import ModalityEmbedding, l2_normalize

        noisy = self.alice + 0.1 * np.random.default_rng(7).normal(size=self.alice.shape)
        return {
            Modality.FACE: ModalityEmbedding(
                Modality.FACE, l2_normalize(noisy.astype(np.float32)), 0.9
            )
        }

    def test_untrained_head_is_refused(self) -> None:
        dims = {Modality.FACE: 32, Modality.GAIT: 32, Modality.REID: 32}
        gallery = self._gallery(dims)
        net = KeylessAttentionFusion(dims, shared_dim=32, seed=0)
        with pytest.raises(RuntimeError, match="untrained"):
            gallery.rank_attention(self._probe(dims), net)

    def test_dimension_mismatch_is_refused_with_a_clear_message(self) -> None:
        """A head trained on synthetic 64-d data cannot fuse real 512-d ones."""
        from app.core.types import ModalityEmbedding, l2_normalize

        dims = {Modality.FACE: 32, Modality.GAIT: 32, Modality.REID: 32}
        gallery = self._gallery(dims)
        net = KeylessAttentionFusion(dims, shared_dim=32, seed=0)
        train(net, synthetic_triplets(80, dims, seed=1), epochs=2, seed=0)

        wrong = {
            Modality.FACE: ModalityEmbedding(
                Modality.FACE, l2_normalize(np.ones(512, dtype=np.float32)), 0.9
            )
        }
        with pytest.raises(ValueError, match="expects 32-d"):
            gallery.rank_attention(wrong, net)

    def test_ranks_the_right_person_first(self) -> None:
        dims = {Modality.FACE: 32, Modality.GAIT: 32, Modality.REID: 32}
        gallery = self._gallery(dims)
        net = KeylessAttentionFusion(dims, shared_dim=64, seed=0)
        train(net, synthetic_triplets(400, dims, seed=2), epochs=15, seed=0)

        ranked = gallery.rank_attention(self._probe(dims), net)
        assert ranked[0].person.person_id == "alice"
        assert ranked[0].fused_similarity > ranked[1].fused_similarity

    def test_records_the_attention_weights_it_used(self) -> None:
        dims = {Modality.FACE: 32, Modality.GAIT: 32, Modality.REID: 32}
        gallery = self._gallery(dims)
        net = KeylessAttentionFusion(dims, shared_dim=64, seed=0)
        train(net, synthetic_triplets(200, dims, seed=3), epochs=5, seed=0)

        ranked = gallery.rank_attention(self._probe(dims), net)
        # Only face was supplied, so it must carry all the weight.
        assert ranked[0].weights == {Modality.FACE: pytest.approx(1.0, abs=1e-5)}
