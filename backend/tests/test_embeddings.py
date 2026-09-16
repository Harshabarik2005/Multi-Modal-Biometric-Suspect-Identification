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

    def test_empty_keeps_the_reason_it_was_given(self) -> None:
        """`empty()` took a `reason` and used it only as a boolean.

        Every branch writes a precise diagnosis of which gate refused and what
        it measured, and all of it was discarded at the door -- leaving callers
        to invent a generic message. Someone whose enrolment stored no gait
        could not find out whether the footage was too short, too coarsely
        sampled, or simply not a walk, which is the one moment the answer is
        worth having.
        """
        refused = ModalityEmbedding.empty(
            Modality.GAIT, reason="frames too far apart to measure cadence"
        )
        assert refused.reason == "frames too far apart to measure cadence"
        assert not refused.has_signal

    def test_a_real_embedding_carries_no_reason(self) -> None:
        assert ModalityEmbedding(
            Modality.FACE, np.array([1.0, 0.0], dtype=np.float32), 0.9
        ).reason == ""

    def test_empty_without_a_reason_is_still_fine(self) -> None:
        assert ModalityEmbedding.empty(Modality.GAIT).reason == ""


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

    def _paced(
        self,
        fps: float,
        seconds: float,
        hz: float,
        start_s: float = 0.0,
        store: TrackBufferStore | None = None,
        gait_max_observations: int = 64,
    ) -> TrackBufferStore:
        settings = get_settings()
        store = store or TrackBufferStore(
            settings,
            config=settings.track_buffer.model_copy(
                update={
                    "gait_sample_hz": hz,
                    "gait_max_observations": gait_max_observations,
                }
            ),
        )
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        for i in range(int(round(fps * seconds))):
            store.update(
                FrameResult(
                    frame_index=i,
                    timestamp_s=start_s + i / fps,
                    tracks=[Track(1, 10, 10, 110, 210)],
                ),
                frame,
            )
        return store

    def test_gaits_history_covers_a_stride_on_any_camera(self) -> None:
        """64 frames is 1.07s at 60fps, shorter than one stride. Gait's ring
        covers seconds whatever the camera, and never unevenly."""
        for fps in (60.0, 30.0, 25.0):
            gait = list(self._paced(fps, seconds=10.0, hz=20.0).get(1).gait_observations)
            span = gait[-1].timestamp_s - gait[0].timestamp_s
            gaps = [b.timestamp_s - a.timestamp_s for a, b in zip(gait, gait[1:])]
            assert len(gait) == 64
            assert span >= 2.5, (fps, span)
            assert max(gaps) - min(gaps) < 1e-6, (fps, sorted(set(gaps))[:4])

    def test_gaits_ring_reads_cadence_as_well_as_the_main_one(self) -> None:
        """Cadence is resampled onto a grid built from the MEDIAN gap, so a ring
        whose frames are unevenly spaced loses coverage before a single mask has
        failed. At 25 and 29.97fps a 0.05s slot would keep four frames in five."""
        from types import SimpleNamespace

        from app.embeddings.gait import resample_cadence

        def coverage(observations) -> float:
            silhouettes = [
                SimpleNamespace(timestamp_s=o.timestamp_s, frame_index=o.frame_index)
                for o in observations
            ]
            values = np.arange(len(silhouettes), dtype=np.float32)
            return resample_cadence(silhouettes, values, 25.0).coverage

        for fps in (25.0, 29.97, 60.0):
            buffer = self._paced(fps, seconds=6.0, hz=20.0).get(1)
            assert coverage(buffer.gait_observations) >= coverage(buffer.observations), fps

    def test_face_and_appearance_still_see_every_frame(self) -> None:
        """They pick crops from the main ring by box height, and a longer window
        changes which crops win. Thinning belongs to gait alone."""
        buffer = self._paced(60.0, seconds=2.0, hz=20.0).get(1)
        frames = [o.frame_index for o in buffer.observations]
        assert len(frames) == 64
        assert frames == list(range(frames[0], frames[0] + 64))
        # Every third frame of 120, plus the first, which is kept before a
        # second frame has revealed the frame period.
        assert len(buffer.gait_observations) == 41

    def test_the_memory_estimate_counts_each_crop_once(self) -> None:
        """The rings overlap and gait holds crops the main ring has dropped."""
        store = self._paced(60.0, seconds=2.0, hz=20.0)
        buffer = store.get(1)
        held = {
            id(obs): obs
            for obs in (*buffer.observations, *buffer.gait_observations)
        }
        assert len(held) > len(buffer.observations), "gait should hold older crops"
        expected = sum(o.crop.nbytes for o in held.values()) / (1024 * 1024)
        assert abs(store.memory_estimate_mb() - expected) < 1e-9

    def test_the_rings_share_observations_rather_than_copying(self) -> None:
        buffer = self._paced(60.0, seconds=1.0, hz=20.0).get(1)
        assert all(
            any(kept is seen for seen in buffer.observations)
            for kept in buffer.gait_observations
        )

    def test_gait_skips_whole_frames_not_wall_clock_slots(self) -> None:
        """Every Nth frame, so the gaps stay equal. 25 and 29.97fps are already
        slower than 20 a second, so nothing is skipped there at all."""
        for fps, stride in ((60.0, 3), (30.0, 2), (29.97, 1), (25.0, 1)):
            buffer = self._paced(
                fps, seconds=4.0, hz=20.0, gait_max_observations=1000
            ).get(1)
            assert buffer.gait_stride() == stride, fps
            kept = [o.frame_index for o in buffer.gait_observations]
            # From the second kept frame on: the first is kept before the
            # period is known.
            gaps = {b - a for a, b in zip(kept[1:], kept[2:])}
            assert gaps == {stride}, (fps, sorted(gaps))

    def test_gait_thinning_can_be_turned_off(self) -> None:
        store = self._paced(60.0, seconds=1.0, hz=0.0, gait_max_observations=1000)
        assert len(store.get(1).gait_observations) == 60

    def test_a_clock_that_does_not_advance_keeps_every_frame(self) -> None:
        """Pacing needs time. Keeping only the first frame would be far worse
        than not pacing at all."""
        store = TrackBufferStore(get_settings())
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        for i in range(10):
            store.update(
                FrameResult(
                    frame_index=i, timestamp_s=0.0, tracks=[Track(1, 10, 10, 110, 210)]
                ),
                frame,
            )
        assert len(store.get(1).gait_observations) == 10

    def test_a_gap_does_not_cause_a_burst_of_kept_frames(self) -> None:
        """A track that reappears is thinned as before, not kept frame after
        frame while a wall-clock schedule catches up."""
        store = self._paced(60.0, seconds=1.0, hz=20.0, gait_max_observations=1000)
        self._paced(60.0, seconds=1.0, hz=20.0, start_s=10.0, store=store)
        kept = list(store.get(1).gait_observations)
        after = [o.frame_index for o in kept if o.timestamp_s >= 10.0]
        assert {b - a for a, b in zip(after, after[1:])} == {3}
        assert 38 <= len(kept) <= 42


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
        """Records written before encryption was required must stay readable.

        Refusing to *read* them would strand data someone has to migrate
        rather than protect anything.
        """
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", "1")
        store = self._store(tmp_path)
        store.save(self._person())
        assert store.load_person("ravi").person_id == "ravi"

    def test_load_gallery_skips_a_corrupt_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unreadable record must not take the whole watchlist down."""
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", "1")
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
