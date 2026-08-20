"""Tests for disguise augmentation (Phase 11).

Pure image manipulation, no model weights. The slow test that measures the
face branch's actual behaviour under disguise lives at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings.disguise import (
    Disguise,
    all_disguises,
    apply,
    apply_to_region,
)


def face_image(height: int = 120, width: int = 100) -> np.ndarray:
    """A crude face: a light oval on a dark ground, with darker eye patches."""
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    centre = (width // 2, height // 2)
    axes = (int(width * 0.36), int(height * 0.44))
    import cv2

    cv2.ellipse(image, centre, axes, 0, 0, 360, (190, 170, 160), -1)
    cv2.circle(image, (int(width * 0.36), int(height * 0.38)), 5, (60, 60, 70), -1)
    cv2.circle(image, (int(width * 0.64), int(height * 0.38)), 5, (60, 60, 70), -1)
    return image


class TestDisguises:
    def test_every_disguise_changes_the_image(self) -> None:
        original = face_image()
        for disguise in Disguise:
            result = apply(original, disguise)
            assert result.image.shape == original.shape
            assert not np.array_equal(result.image, original), disguise

    def test_the_original_is_not_mutated(self) -> None:
        original = face_image()
        untouched = original.copy()
        for disguise in Disguise:
            apply(original, disguise)
        assert np.array_equal(original, untouched)

    def test_mask_covers_the_lower_face_and_leaves_the_eyes(self) -> None:
        """A mask that also covered the eyes would not be testing a mask."""
        original = face_image()
        masked = apply(original, Disguise.MASK).image

        eyes = slice(int(120 * 0.30), int(120 * 0.45))
        mouth = slice(int(120 * 0.70), int(120 * 0.95))
        assert np.array_equal(masked[eyes], original[eyes]), "eyes must be untouched"
        assert not np.array_equal(masked[mouth], original[mouth])

    def test_sunglasses_cover_the_eyes_and_leave_the_mouth(self) -> None:
        original = face_image()
        shaded = apply(original, Disguise.SUNGLASSES).image

        eyes = slice(int(120 * 0.30), int(120 * 0.45))
        mouth = slice(int(120 * 0.80), int(120 * 0.95))
        assert not np.array_equal(shaded[eyes], original[eyes])
        assert np.array_equal(shaded[mouth], original[mouth]), "mouth must be untouched"

    def test_combined_disguise_covers_more_than_either_alone(self) -> None:
        original = face_image()
        mask = apply(original, Disguise.MASK)
        glasses = apply(original, Disguise.SUNGLASSES)
        both = apply(original, Disguise.MASK_AND_SUNGLASSES)

        assert both.occluded_fraction > mask.occluded_fraction
        assert both.occluded_fraction > glasses.occluded_fraction

    def test_blur_occludes_nothing_but_still_degrades(self) -> None:
        """Blur is distance, not disguise -- occlusion is the wrong measure."""
        original = face_image()
        blurred = apply(original, Disguise.BLUR)
        assert blurred.occluded_fraction == 0.0
        assert not np.array_equal(blurred.image, original)

    def test_occluded_fraction_stays_in_range(self) -> None:
        for disguise in Disguise:
            assert 0.0 <= apply(face_image(), disguise).occluded_fraction <= 1.0

    def test_empty_image_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            apply(np.empty((0, 0, 3), dtype=np.uint8), Disguise.MASK)

    def test_all_disguises_returns_one_of_each(self) -> None:
        results = all_disguises(face_image())
        assert set(results) == set(Disguise)


class TestApplyToRegion:
    """Disguises are face-proportioned; body crops need the region form.

    Applying them to a body crop directly put the "mask" around the knees,
    which made an early robustness run report similarity 1.000 for a masked
    face -- a broken test that looked like impressive invariance.
    """

    def test_only_the_face_region_changes(self) -> None:
        body = np.full((400, 160, 3), 90, dtype=np.uint8)
        body[40:160, 30:130] = face_image()

        result = apply_to_region(body, Disguise.MASK, (30, 40, 130, 160))
        # Well below the face box: untouched.
        assert np.array_equal(result.image[250:], body[250:])
        assert not np.array_equal(result.image[40:160], body[40:160])

    def test_reports_occlusion_relative_to_the_face(self) -> None:
        """Covering half a face is the meaningful number, not half a body.

        Compared loosely: `apply_to_region` adds a 15% margin around the box,
        so it disguises a slightly larger region and integer rounding of the
        mask boundary lands a fraction of a percent differently. What matters
        is that the figure describes the face, not the body crop — against the
        body it would read as roughly a tenth rather than a third.
        """
        body = np.full((400, 160, 3), 90, dtype=np.uint8)
        body[40:160, 30:130] = face_image()

        whole = apply(face_image(), Disguise.MASK)
        region = apply_to_region(body, Disguise.MASK, (30, 40, 130, 160))
        assert region.occluded_fraction == pytest.approx(
            whole.occluded_fraction, abs=0.02
        )
        # And decisively not the body-relative figure.
        assert region.occluded_fraction > 0.2

    def test_region_is_clamped_to_the_image(self) -> None:
        image = np.full((100, 100, 3), 90, dtype=np.uint8)
        result = apply_to_region(image, Disguise.MASK, (-20, -20, 120, 120))
        assert result.image.shape == image.shape

    def test_a_tiny_region_is_rejected(self) -> None:
        image = np.full((100, 100, 3), 90, dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            apply_to_region(image, Disguise.MASK, (50, 50, 52, 52))


@pytest.mark.slow
class TestFaceBranchUnderDisguise:
    def test_quality_falls_when_the_face_is_heavily_occluded(self) -> None:
        """The behaviour fusion depends on.

        A branch that returns a confident embedding for a mostly-covered face
        is worse than one that returns nothing, because fusion weights by
        quality and would trust it.
        """
        from app.core.config import get_settings
        from app.core.types import TrackObservation
        from app.embeddings.face import FaceEmbedder

        settings = get_settings()
        embedder = FaceEmbedder(settings)

        # A synthetic face the detector can actually find is not guaranteed,
        # so use the real fixture if it is available.
        from pathlib import Path

        clip = (
            Path(__file__).resolve().parents[2]
            / "data" / "test_videos" / "enroll_subject_a.mp4"
        )
        if not clip.exists():
            pytest.skip("Run scripts/make_face_fixtures.py first.")

        import cv2

        capture = cv2.VideoCapture(str(clip))
        ok, frame = capture.read()
        capture.release()
        assert ok

        faces = embedder.detect(frame)
        if not faces:
            pytest.skip("No face detected in the fixture frame.")
        box = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ).bbox

        clean = embedder.embed_frame(
            TrackObservation(0, 0.0, frame, float(frame.shape[0]), 0.9)
        )
        if not clean.has_signal:
            pytest.skip("Clean frame produced no face embedding.")

        heavy = apply_to_region(frame, Disguise.MASK_AND_SUNGLASSES, box)
        occluded = embedder.embed_frame(
            TrackObservation(0, 0.0, heavy.image, float(frame.shape[0]), 0.9)
        )

        if not occluded.has_signal:
            return  # No signal at all is the safest possible outcome.

        assert occluded.quality < clean.quality, (
            "quality must fall when the face is covered, or fusion will trust "
            "an embedding it should not"
        )
