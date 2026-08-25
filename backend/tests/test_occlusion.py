"""Tests for occlusion-aware face quality (Phase 11).

The failure this exists to remove, measured on a real enrolment crop before
the change:

    disguise              face covered   similarity   quality
    (none)                          0%       1.0000     0.596
    mask                           36%       0.6277     0.642
    mask + sunglasses              52%       0.2480     0.610

Covering a face made the branch MORE confident, because an opaque shape is a
clean high-contrast region the detector is surer about, while yaw and pixel
count do not move. Quality is what fusion weights by, so the branch that had
stopped identifying anyone was handed the largest share of the vote.

Most of these run on synthetic images and are always on. The ones needing
InsightFace are marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.embeddings.occlusion import (
    FEATURE_DIM,
    OCCLUDED_THRESHOLD,
    FaceRegion,
    OcclusionDetector,
    OcclusionMap,
    ite_features,
    region_boxes,
)

#: A plausible five-point set: eyes level and apart, nose between and below,
#: mouth below that. Matches InsightFace's documented `kps` order.
KPS = np.array(
    [[40.0, 45.0], [72.0, 45.0], [56.0, 62.0], [44.0, 80.0], [68.0, 80.0]],
    dtype=np.float32,
)
BBOX = (24.0, 20.0, 88.0, 104.0)


class TestRegionGeometry:
    def test_every_region_is_located(self) -> None:
        boxes = region_boxes(BBOX, KPS)
        assert set(boxes) == set(FaceRegion)

    def test_regions_sit_where_their_names_say(self) -> None:
        """A region called 'mouth' that lands on the forehead would produce a
        confident, wrong occlusion map and there is nothing downstream to catch
        it."""
        boxes = region_boxes(BBOX, KPS)
        eye_y = (KPS[0][1] + KPS[1][1]) / 2
        mouth_y = (KPS[3][1] + KPS[4][1]) / 2

        assert boxes[FaceRegion.FOREHEAD].y2 <= eye_y
        assert boxes[FaceRegion.CHIN].y1 >= mouth_y
        assert boxes[FaceRegion.EYE_LEFT].x2 < boxes[FaceRegion.EYE_RIGHT].x1
        assert boxes[FaceRegion.CHEEK_LEFT].x1 < boxes[FaceRegion.EYE_LEFT].x1
        assert boxes[FaceRegion.CHEEK_RIGHT].x2 > boxes[FaceRegion.EYE_RIGHT].x2
        # The nose sits between the eye and mouth lines, not beside them.
        nose = boxes[FaceRegion.NOSE]
        assert eye_y < (nose.y1 + nose.y2) / 2 < mouth_y

    def test_regions_scale_with_the_face(self) -> None:
        """Sizes are in inter-ocular distance, so a face twice as far away
        gets regions half the size rather than a fixed pixel box that would
        swallow the whole head."""
        near = region_boxes(BBOX, KPS)
        far = region_boxes(
            tuple(v / 2 for v in BBOX), KPS / 2
        )
        for region in FaceRegion:
            assert far[region].width < near[region].width
            assert far[region].height < near[region].height

    def test_a_degenerate_eye_line_does_not_divide_by_zero(self) -> None:
        """Both eyes project to nearly the same point on a near-profile face."""
        collapsed = KPS.copy()
        collapsed[1] = collapsed[0]
        boxes = region_boxes(BBOX, collapsed)
        assert set(boxes) == set(FaceRegion)
        assert all(b.width > 0 for b in boxes.values())

    def test_too_few_keypoints_is_refused(self) -> None:
        with pytest.raises(ValueError, match="five keypoints"):
            region_boxes(BBOX, KPS[:3])

    def test_a_tiny_region_is_not_usable(self) -> None:
        """A 3x2 patch has no texture to measure, and a classifier asked about
        one answers from noise."""
        boxes = region_boxes((0.0, 0.0, 12.0, 16.0), KPS / 8.0)
        assert any(not box.usable for box in boxes.values())


class TestITEFeatures:
    def test_dimension_is_fixed(self) -> None:
        patch = np.full((20, 20, 3), 128, dtype=np.uint8)
        assert ite_features(patch).shape == (FEATURE_DIM,)

    def test_an_empty_patch_is_all_zeros_not_a_crash(self) -> None:
        assert ite_features(np.zeros((0, 0, 3), np.uint8)).shape == (FEATURE_DIM,)
        assert not ite_features(np.zeros((0, 0, 3), np.uint8)).any()

    def test_flat_and_textured_patches_differ(self) -> None:
        """The whole premise: an opaque covering has near-zero local variation
        and skin never does."""
        rng = np.random.default_rng(0)
        flat = np.full((24, 24, 3), 150, dtype=np.uint8)
        textured = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
        assert not np.allclose(ite_features(flat), ite_features(textured))

    def test_features_are_scale_normalised(self) -> None:
        """A big region and a small one of the same material must look alike,
        or region size would leak into the occlusion decision."""
        small = np.full((12, 12, 3), 90, dtype=np.uint8)
        large = np.full((48, 48, 3), 90, dtype=np.uint8)
        assert ite_features(small) == pytest.approx(ite_features(large), abs=0.05)


class TestOcclusionMap:
    def test_a_clear_face_leaves_everything_visible(self) -> None:
        area = {region: 100.0 for region in FaceRegion}
        visible = {region: True for region in FaceRegion}
        assert OcclusionMap(visible=visible, areas=area).visible_identity == 1.0

    def test_covering_regions_reduces_visible_identity(self) -> None:
        area = {region: 100.0 for region in FaceRegion}
        visible = {region: True for region in FaceRegion}
        visible[FaceRegion.MOUTH] = False
        visible[FaceRegion.NOSE] = False
        got = OcclusionMap(visible=visible, areas=area).visible_identity
        assert got == pytest.approx(6 / 8)

    def test_bigger_regions_count_for_more(self) -> None:
        """Weighting is by area when no measured weights exist, so covering a
        large region has to cost more than covering a small one."""
        areas = {region: 10.0 for region in FaceRegion}
        areas[FaceRegion.FOREHEAD] = 200.0

        hide_big = {region: True for region in FaceRegion}
        hide_big[FaceRegion.FOREHEAD] = False
        hide_small = {region: True for region in FaceRegion}
        hide_small[FaceRegion.CHIN] = False

        assert (
            OcclusionMap(visible=hide_big, areas=areas).visible_identity
            < OcclusionMap(visible=hide_small, areas=areas).visible_identity
        )

    def test_regions_off_the_edge_of_frame_do_not_look_like_a_disguise(self) -> None:
        """A face at the edge of a shot loses regions to the frame boundary.
        Counting those as covered would report a crop as a disguise."""
        located = [FaceRegion.EYE_LEFT, FaceRegion.EYE_RIGHT, FaceRegion.NOSE]
        result = OcclusionMap(
            visible={region: True for region in located},
            areas={region: 100.0 for region in located},
            unknown={r for r in FaceRegion if r not in located},
        )
        assert result.visible_identity == 1.0

    def test_describe_names_what_is_covered(self) -> None:
        """A reviewer looking at a weak score needs to know why; 0.31 is not
        an explanation."""
        visible = {region: True for region in FaceRegion}
        visible[FaceRegion.MOUTH] = False
        text = OcclusionMap(
            visible=visible, areas={r: 100.0 for r in FaceRegion}
        ).describe()
        assert "mouth" in text
        assert "%" in text

    def test_describe_says_so_when_nothing_is_covered(self) -> None:
        result = OcclusionMap(visible={r: True for r in FaceRegion})
        assert "whole face visible" in result.describe()


class TestDetector:
    def test_an_untrained_detector_penalises_nothing(self) -> None:
        """Failing closed would mark every face as disguised, which looks
        exactly like the model working and is far harder to spot."""
        detector = OcclusionDetector({"weights": [], "mean": [], "std": []})
        assert not detector.trained

        image = np.full((120, 100, 3), 128, dtype=np.uint8)
        result = detector.detect(image, BBOX, KPS)
        assert result.visible_identity == 1.0
        assert result.unknown == set(FaceRegion)

    def test_the_shipped_coefficients_are_present_and_the_right_shape(self) -> None:
        """A silently untrained detector would disable the whole feature."""
        detector = OcclusionDetector()
        assert detector.trained, (
            "occlusion.py carries no fitted coefficients -- run "
            "scripts/train_occlusion.py and paste its output"
        )

    def test_a_flat_covering_scores_higher_than_skin_texture(self) -> None:
        rng = np.random.default_rng(1)
        detector = OcclusionDetector()
        flat = np.full((30, 30, 3), 150, dtype=np.uint8)
        # Skin is not uniform: fine variation at low amplitude.
        skin = np.clip(
            rng.normal(150, 14, (30, 30, 3)), 0, 255
        ).astype(np.uint8)
        assert detector.probability(flat) > detector.probability(skin)

    def test_probability_stays_in_range(self) -> None:
        detector = OcclusionDetector()
        for value in (0, 128, 255):
            patch = np.full((20, 20, 3), value, dtype=np.uint8)
            assert 0.0 <= detector.probability(patch) <= 1.0

    def test_threshold_is_above_a_coin_flip(self) -> None:
        """The two errors are not symmetric, but a detector that cries disguise
        at every shadow gets switched off."""
        assert 0.5 < OCCLUDED_THRESHOLD < 1.0


@pytest.mark.slow
class TestOnRealFaces:
    """End to end, through InsightFace. Needs the model pack downloaded."""

    @staticmethod
    def _face_image():
        cv2 = pytest.importorskip("cv2")
        from pathlib import Path

        for candidate in (
            Path(__file__).resolve().parents[2] / "data" / "test_videos",
        ):
            for path in sorted(candidate.glob("*.jpg")) + sorted(
                candidate.glob("*.jpeg")
            ):
                image = cv2.imread(str(path))
                if image is not None:
                    return image
        pytest.skip("No face fixture available")

    def test_a_masked_face_scores_lower_than_a_clear_one(self) -> None:
        """The acceptance test. Before this change the masked face scored
        HIGHER: 0.642 against 0.596."""
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("insightface")

        from app.core.types import TrackObservation
        from app.embeddings.disguise import Disguise, apply_to_region
        from app.embeddings.face import FaceEmbedder

        image = self._face_image()
        embedder = FaceEmbedder(get_settings())
        faces = embedder.detect(image)
        if not faces:
            pytest.skip("No face detected in the fixture")

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        box = tuple(float(v) for v in face.bbox)

        def quality_of(picture):
            observation = TrackObservation(
                frame_index=0,
                timestamp_s=0.0,
                crop=picture,
                box_height=float(picture.shape[0]),
                detection_confidence=0.95,
            )
            return embedder.embed_frame(observation)

        clean = quality_of(image)
        if not clean.has_signal:
            pytest.skip("Fixture face is not usable")

        masked = quality_of(
            apply_to_region(image.copy(), Disguise.MASK, box).image
        )
        # Either refused outright, or kept with a markedly lower score. Both
        # are correct; silently scoring it higher is not.
        assert not masked.has_signal or masked.quality < clean.quality
