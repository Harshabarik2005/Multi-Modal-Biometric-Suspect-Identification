"""Tests for the modality-embedding contract, buffering, and the gallery.

Everything here runs without model weights: the branch logic is exercised
through a fake embedder so aggregation, buffering and matching are tested
independently of whether ArcFace is installed. The real ArcFace test lives in
`test_face_branch.py` and is marked slow.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.core.config import get_settings
from app.core.track_buffer import TrackBufferStore
from app.core.types import (
    FrameResult,
    Modality,
    ModalityEmbedding,
    Track,
    TrackObservation,
    cosine_similarity,
    l2_normalize,
)
from app.embeddings.base import PerFrameBranch
from app.matching.gallery import Gallery, GalleryStore, PersonRecord


def make_observation(
    frame_index: int = 0, height: int = 200, width: int = 100, value: int = 128
) -> TrackObservation:
    return TrackObservation(
        frame_index=frame_index,
        timestamp_s=frame_index / 25.0,
        crop=np.full((height, width, 3), value, dtype=np.uint8),
        box_height=float(height),
        detection_confidence=0.9,
    )


class FakeBranch(PerFrameBranch):
    """A per-frame branch whose output is scripted by the test."""

    modality = Modality.FACE
    embedding_dim = 4

    def __init__(self, scripted: list[tuple[np.ndarray | None, float]]) -> None:
        self.scripted = scripted
        self.calls = 0
        self.min_observations = 1

    def embed_frame(self, observation: TrackObservation) -> ModalityEmbedding:
        vector, quality = self.scripted[self.calls % len(self.scripted)]
        self.calls += 1
        if vector is None:
            return ModalityEmbedding.empty(self.modality, reason="scripted")
        return ModalityEmbedding(
            modality=self.modality,
            vector=np.asarray(vector, dtype=np.float32),
            quality=quality,
            frames_used=1,
        )


class TestVectorHelpers:
    def test_cosine_of_identical_vectors_is_one(self) -> None:
        v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_of_orthogonal_vectors_is_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_handles_zero_vector_without_dividing_by_zero(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity(np.zeros(3), np.zeros(4))

    def test_l2_normalize_gives_unit_length(self) -> None:
        v = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_l2_normalize_leaves_zero_vector_alone(self) -> None:
        v = l2_normalize(np.zeros(3, dtype=np.float32))
        assert np.allclose(v, 0.0)


class TestModalityEmbedding:
    def test_empty_has_no_signal(self) -> None:
        e = ModalityEmbedding.empty(Modality.GAIT)
        assert not e.has_signal
        assert e.quality == 0.0

    def test_similarity_against_no_signal_is_none_not_zero(self) -> None:
        """'Could not compare' and 'compared, got zero' are different facts.

        Collapsing them would let fusion treat a missing modality as active
        evidence of a mismatch.
        """
        real = ModalityEmbedding(
            Modality.FACE, np.array([1.0, 0.0], dtype=np.float32), 0.9
        )
        assert real.similarity(ModalityEmbedding.empty(Modality.FACE)) is None

    def test_similarity_rejects_cross_modality_comparison(self) -> None:
        face = ModalityEmbedding(Modality.FACE, np.array([1.0, 0.0]), 0.9)
        gait = ModalityEmbedding(Modality.GAIT, np.array([1.0, 0.0]), 0.9)
        with pytest.raises(ValueError):
            face.similarity(gait)


class TestPerFrameAggregation:
    def test_reports_no_signal_when_every_frame_fails(self) -> None:
        branch = FakeBranch([(None, 0.0)])
        result = branch.embed([make_observation(i) for i in range(5)])
        assert not result.has_signal

    def test_ignores_frames_that_carry_no_signal(self) -> None:
        branch = FakeBranch([(np.array([1, 0, 0, 0]), 0.8), (None, 0.0)])
        result = branch.embed([make_observation(i) for i in range(6)])
        assert result.has_signal
        assert result.frames_used == 3

    def test_quality_weighting_favours_the_clear_frame(self) -> None:
        """One good look must outweigh several poor ones.

        Otherwise a track that is mostly turned away outvotes its one clean
        frontal frame purely by weight of numbers.
        """
        good = np.array([1.0, 0.0, 0.0, 0.0])
        bad = np.array([0.0, 1.0, 0.0, 0.0])
        branch = FakeBranch([(good, 0.95), (bad, 0.05), (bad, 0.05), (bad, 0.05)])
        result = branch.embed([make_observation(i) for i in range(4)])

        assert cosine_similarity(result.vector, good) > cosine_similarity(
            result.vector, bad
        )

    def test_track_quality_is_the_best_frame_not_the_mean(self) -> None:
        branch = FakeBranch(
            [(np.array([1, 0, 0, 0]), 0.9), (np.array([1, 0, 0, 0]), 0.1)]
        )
        result = branch.embed([make_observation(i) for i in range(2)])
        assert result.quality == pytest.approx(0.9)
        assert result.detail["mean_frame_quality"] == pytest.approx(0.5)

    def test_output_is_unit_length(self) -> None:
        branch = FakeBranch([(np.array([3.0, 4.0, 0.0, 0.0]), 0.5)])
        result = branch.embed([make_observation(0)])
        assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-6)

    def test_all_zero_quality_does_not_divide_by_zero(self) -> None:
        branch = FakeBranch([(np.array([1.0, 0, 0, 0]), 0.0)])
        result = branch.embed([make_observation(i) for i in range(3)])
        assert result.has_signal

    def test_min_observations_is_respected(self) -> None:
        branch = FakeBranch([(np.array([1, 0, 0, 0]), 0.8)])
        branch.min_observations = 5
        assert not branch.embed([make_observation(i) for i in range(3)]).has_signal
        assert branch.embed([make_observation(i) for i in range(5)]).has_signal


class TestTrackBuffer:
    def _frame_result(self, frame_index: int, tracks: list[Track]) -> FrameResult:
        return FrameResult(
            frame_index=frame_index, timestamp_s=frame_index / 25.0, tracks=tracks
        )

    def test_buffers_observations_per_track(self) -> None:
        store = TrackBufferStore(get_settings())
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        for i in range(5):
            store.update(
                self._frame_result(i, [Track(1, 10, 10, 110, 210)]), frame
            )
        assert len(store) == 1
        assert len(store.get(1)) == 5

    def test_observation_cap_is_enforced(self) -> None:
        """Unbounded buffers are how a long run exhausts memory."""
        settings = get_settings()
        settings.track_buffer.max_observations = 8
        store = TrackBufferStore(settings)
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        for i in range(50):
            store.update(self._frame_result(i, [Track(1, 10, 10, 110, 210)]), frame)

        buffer = store.get(1)
        assert len(buffer) == 8
        assert buffer.total_seen == 50
        # It keeps the most RECENT frames, not the first.
        assert buffer.observations[-1].frame_index == 49
        settings.track_buffer.max_observations = 64

    def test_track_cap_evicts_least_recently_seen(self) -> None:
        settings = get_settings()
        settings.track_buffer.max_tracks = 3
        store = TrackBufferStore(settings)
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        for tid in range(6):
            store.update(
                self._frame_result(tid, [Track(tid, 10, 10, 110, 210)]), frame
            )
        assert len(store) == 3
        assert 0 not in store and 5 in store
        settings.track_buffer.max_tracks = 50

    def test_small_boxes_are_skipped(self) -> None:
        settings = get_settings()
        settings.track_buffer.min_box_height = 100
        store = TrackBufferStore(settings)
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        store.update(
            self._frame_result(0, [Track(1, 10, 10, 60, 60)]), frame
        )  # 50px tall
        assert len(store) == 0
        settings.track_buffer.min_box_height = 60

    def test_tall_crops_are_downscaled_but_short_ones_are_not(self) -> None:
        settings = get_settings()
        settings.track_buffer.store_height = 128
        store = TrackBufferStore(settings)
        frame = np.full((600, 640, 3), 100, dtype=np.uint8)
        store.update(self._frame_result(0, [Track(1, 0, 0, 200, 400)]), frame)
        assert store.get(1).observations[0].crop.shape[0] == 128

        store.reset()
        store.update(self._frame_result(0, [Track(2, 0, 0, 50, 100)]), frame)
        assert store.get(2).observations[0].crop.shape[0] == 100
        settings.track_buffer.store_height = 256


class TestGallery:
    def _person(self, pid: str, vector: list[float]) -> PersonRecord:
        return PersonRecord(
            person_id=pid,
            display_name=pid.title(),
            embeddings={
                Modality.FACE: ModalityEmbedding(
                    Modality.FACE,
                    l2_normalize(np.array(vector, dtype=np.float32)),
                    0.8,
                )
            },
        )

    def test_ranks_the_right_person_first(self) -> None:
        gallery = Gallery(
            [self._person("alice", [1, 0, 0]), self._person("bob", [0, 1, 0])]
        )
        probe = ModalityEmbedding(
            Modality.FACE, l2_normalize(np.array([0.9, 0.1, 0.0])), 0.9
        )
        ranked = gallery.rank_single(Modality.FACE, probe)
        assert ranked[0].person.person_id == "alice"
        assert ranked[0].fused_similarity > ranked[1].fused_similarity

    def test_probe_with_no_signal_ranks_everyone_at_the_floor(self) -> None:
        gallery = Gallery([self._person("alice", [1, 0, 0])])
        ranked = gallery.rank_single(
            Modality.FACE, ModalityEmbedding.empty(Modality.FACE)
        )
        assert ranked[0].fused_similarity == -1.0

    def test_explain_names_each_modality(self) -> None:
        gallery = Gallery([self._person("alice", [1, 0, 0])])
        probe = ModalityEmbedding(
            Modality.FACE, l2_normalize(np.array([1.0, 0, 0])), 0.9
        )
        assert "face=" in gallery.rank_single(Modality.FACE, probe)[0].explain()

    def test_empty_gallery_ranks_nothing(self) -> None:
        probe = ModalityEmbedding(Modality.FACE, np.array([1.0, 0, 0]), 0.9)
        assert Gallery().rank_single(Modality.FACE, probe) == []


class TestGalleryStore:
    def _store(self, tmp_path: Path) -> GalleryStore:
        settings = get_settings()
        store = GalleryStore(settings)
        store.root = tmp_path / "enrollment"
        return store

    def _person(self) -> PersonRecord:
        return PersonRecord(
            person_id="ravi",
            display_name="Ravi Kumar",
            embeddings={
                Modality.FACE: ModalityEmbedding(
                    Modality.FACE,
                    l2_normalize(np.arange(512, dtype=np.float32)),
                    0.77,
                    frames_used=42,
                    detail={"yaw": 3.5},
                )
            },
        )

    def test_round_trip_preserves_the_vector(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        original = self._person()
        store.save(original)
        loaded = store.load_person("ravi")

        assert loaded.display_name == "Ravi Kumar"
        embedding = loaded.embeddings[Modality.FACE]
        assert embedding.quality == pytest.approx(0.77)
        assert embedding.frames_used == 42
        assert np.allclose(
            embedding.vector, original.embeddings[Modality.FACE].vector
        )

    def test_refuses_to_save_a_person_with_no_embedding(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        empty = PersonRecord(person_id="nobody", display_name="Nobody")
        with pytest.raises(ValueError):
            store.save(empty)

    def test_encrypts_when_a_key_is_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Biometric templates must not be readable straight off disk."""
        from cryptography.fernet import Fernet

        monkeypatch.setenv("FRS_TEMPLATE_ENCRYPTION_KEY", Fernet.generate_key().decode())
        store = self._store(tmp_path)
        store.save(self._person())

        blob = (store.person_dir("ravi") / "templates.npz").read_bytes()
        assert blob.startswith(b"FRSENC1:")
        # numpy's zip magic must not be sitting there in the clear.
        assert not blob.startswith(b"PK")
        assert np.allclose(
            store.load_person("ravi").embeddings[Modality.FACE].vector,
            self._person().embeddings[Modality.FACE].vector,
        )

    def test_encrypted_record_needs_the_key_to_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cryptography.fernet import Fernet

        monkeypatch.setenv("FRS_TEMPLATE_ENCRYPTION_KEY", Fernet.generate_key().decode())
        store = self._store(tmp_path)
        store.save(self._person())

        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY")
        with pytest.raises(RuntimeError, match="encrypted"):
            store.load_person("ravi")

    def test_plaintext_still_loads_when_no_key_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        store = self._store(tmp_path)
        store.save(self._person())
        assert store.load_person("ravi").person_id == "ravi"

    def test_load_gallery_skips_a_corrupt_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unreadable record must not take the whole watchlist down."""
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        store = self._store(tmp_path)
        store.save(self._person())

        broken = store.root / "broken"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "metadata.json").write_text("{not json", encoding="utf-8")
        (broken / "templates.npz").write_bytes(b"garbage")

        gallery = store.load_gallery()
        assert len(gallery) == 1
        assert gallery.get("ravi") is not None

    def test_missing_person_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            self._store(tmp_path).load_person("ghost")
