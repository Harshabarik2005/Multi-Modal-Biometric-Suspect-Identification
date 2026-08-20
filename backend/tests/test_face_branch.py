"""End-to-end tests for the real ArcFace branch (Phase 2).

Marked slow: needs InsightFace, its ~280MB model pack, and the generated
fixtures. Run with `pytest --run-slow`.

Generate the fixtures first:
    python scripts/make_face_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import get_settings
from app.core.track_buffer import TrackBufferStore
from app.core.types import Modality, TrackObservation

REPO_ROOT = Path(__file__).resolve().parents[2]
ENROLL_CLIP = REPO_ROOT / "data" / "test_videos" / "enroll_subject_a.mp4"
PROBE_CLIP = REPO_ROOT / "data" / "test_videos" / "probe_two_subjects.mp4"


@pytest.fixture(scope="module")
def fixtures_present() -> None:
    if not ENROLL_CLIP.exists() or not PROBE_CLIP.exists():
        pytest.skip("Run `python scripts/make_face_fixtures.py` to generate fixtures.")


@pytest.fixture(scope="module")
def embedder():
    from app.embeddings.face import FaceEmbedder

    return FaceEmbedder(get_settings())


def buffers_for(clip: Path, max_frames: int = 12) -> TrackBufferStore:
    from app.pipeline import DetectionTrackingPipeline

    settings = get_settings()
    previous = settings.video.max_frames
    settings.video.max_frames = max_frames
    try:
        pipeline = DetectionTrackingPipeline(settings)
        store = TrackBufferStore(settings)
        for result, frame in pipeline.stream(clip):
            store.update(result, frame)
        return store
    finally:
        settings.video.max_frames = previous


@pytest.mark.slow
class TestFaceEmbedder:
    def test_blank_image_yields_no_signal(self, embedder) -> None:
        """No face present is the normal case here, not an error path."""
        observation = TrackObservation(
            frame_index=0,
            timestamp_s=0.0,
            crop=np.full((256, 128, 3), 128, dtype=np.uint8),
            box_height=256.0,
            detection_confidence=0.9,
        )
        result = embedder.embed_frame(observation)
        assert not result.has_signal
        assert result.quality == 0.0

    def test_degenerate_crop_yields_no_signal(self, embedder) -> None:
        observation = TrackObservation(
            frame_index=0,
            timestamp_s=0.0,
            crop=np.empty((0, 0, 3), dtype=np.uint8),
            box_height=0.0,
            detection_confidence=0.1,
        )
        assert not embedder.embed_frame(observation).has_signal

    def test_embedding_is_512d_and_unit_length(
        self, embedder, fixtures_present
    ) -> None:
        store = buffers_for(ENROLL_CLIP)
        buffer = next(iter(store))
        result = embedder.embed(list(buffer))

        assert result.has_signal
        assert result.vector.size == embedder.embedding_dim == 512
        assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-4)
        assert 0.0 < result.quality <= 1.0


@pytest.mark.slow
class TestQualityScore:
    def test_profile_face_is_scored_down_despite_confident_detection(
        self, embedder, fixtures_present
    ) -> None:
        """A confidently-detected profile must not be trusted.

        The impostor in the probe clip is detected clearly (larger face than
        the enrolled subject) but is turned ~67 degrees away. ArcFace still
        returns an embedding for it -- a confident, unreliable one. The quality
        score has to catch that, because detection confidence alone will not.
        """
        store = buffers_for(PROBE_CLIP)
        found_profile = False

        for buffer in store:
            for observation in list(buffer)[:4]:
                faces = embedder.detect(observation.crop)
                if not faces:
                    continue
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                quality, detail = embedder._quality(face, observation.crop.shape[0])
                if abs(detail["yaw"]) > embedder.cfg.max_yaw_deg:
                    found_profile = True
                    assert detail["det_score"] > 0.4, "expected a confident detection"
                    assert detail["frontality"] == 0.0
                    assert quality == 0.0, "extreme yaw must zero the quality score"

        if not found_profile:
            pytest.skip("No extreme-yaw face in the fixture to exercise this")


@pytest.mark.slow
class TestOpenSetMatching:
    def test_enrolled_subject_matches_and_impostor_does_not(
        self, fixtures_present, tmp_path: Path
    ) -> None:
        """The end-to-end Phase-2 claim, with a real impostor present.

        Quality gating is disabled for this test so it measures the EMBEDDING's
        discrimination rather than the gate's. Both matter, but they are
        different guarantees and a test that conflates them proves neither.
        """
        from app.embeddings.face import FaceEmbedder
        from app.matching.gallery import GalleryStore, PersonRecord

        settings = get_settings()
        saved = (settings.face.min_quality, settings.face.max_yaw_deg)
        settings.face.min_quality = 0.0
        settings.face.max_yaw_deg = 90.0
        try:
            embedder = FaceEmbedder(settings)

            enroll_store = buffers_for(ENROLL_CLIP)
            assert len(enroll_store) == 1, "enrollment clip must hold exactly one person"
            reference = embedder.embed_reference(list(next(iter(enroll_store))))
            assert reference.has_signal

            store = GalleryStore(settings)
            store.root = tmp_path / "enrollment"
            store.save(
                PersonRecord(
                    person_id="subject_a",
                    display_name="Test Subject A",
                    embeddings={Modality.FACE: reference},
                )
            )
            gallery = store.load_gallery()

            probes = buffers_for(PROBE_CLIP)
            assert len(probes) >= 2, "probe clip must hold the subject and an impostor"

            similarities = []
            for buffer in probes:
                embedding = embedder.embed(list(buffer))
                if not embedding.has_signal:
                    continue
                best = gallery.rank_single(Modality.FACE, embedding)[0]
                similarities.append(best.fused_similarity)

            assert len(similarities) >= 2, "expected both tracks to embed"
            best, worst = max(similarities), min(similarities)

            assert best > 0.6, f"enrolled subject should match strongly, got {best:.3f}"
            assert worst < 0.3, f"impostor should not match, got {worst:.3f}"
            assert best - worst > 0.4, "separation between subject and impostor too small"
        finally:
            settings.face.min_quality, settings.face.max_yaw_deg = saved
