"""Which parts of a face can actually be seen (Phase 11).

The face branch scores quality as `detection confidence x frontality x
resolution`. None of those notice occlusion, and measured on a real enrolment
crop that produces the following:

    disguise              face covered   similarity   quality
    (none)                          0%       1.0000     0.596
    mask                           36%       0.6277     0.642
    mask + sunglasses              52%       0.2480     0.610

A mask *raises* the score. Half the face covered, identity gone -- 0.248 is
stranger territory -- and the branch is more confident than it was on the clean
photograph. The reason is mechanical: an opaque shape is a clean, high-contrast
region, so the detector is surer it found a face; yaw and pixel count are
unchanged. Confidence is what fusion weights by, so the branch that has stopped
working gets handed the vote.

`disguise.py` has warned about this since it was written -- "a branch that
returns a confident embedding for a masked face is worse than one that returns
nothing, because fusion will trust it". It provided the instrument to see it.
This module is the part that does something about it.

Approach
--------
From Dhamecha et al., *Recognizing Disguised Faces: Human and Machine
Evaluation* (PLoS ONE 2014), whose hypothesis is:

    "The facial part or patches which are under the effect of disguise are the
    least useful for face recognition, and may also provide misleading
    details."

Their Anavrta framework tessellates the face, classifies each patch biometric
(usable) or non-biometric (disguised) from an Intensity and Texture Encoder
(ITE) feature, and matches only on patches clean in both gallery and probe.

Two deliberate departures:

* **Regions, not a fixed grid.** They used a 5x5 tessellation because they had
  no landmarks. InsightFace gives five stable keypoints for free, so regions
  follow the face instead of the image, and stay aligned across pose and scale.

* **Quality, not matching.** Their patch AND requires a patch-based matcher;
  ArcFace reads the whole face at once and cannot be handed a face with holes
  in it. So the occlusion map drives the *quality* score -- which already
  decides how much fusion trusts this branch -- rather than the embedding.
  The claim "half this face is covered" is exactly what quality is supposed to
  express and currently cannot.

What this is not
----------------
The classifier is trained on synthetic occlusions: flat shapes drawn by
`disguise.py`. It is therefore very good at recognising flat drawn shapes. How
far that carries to a real cloth mask under real lighting is a question this
project cannot answer without real disguise footage, and no number produced
here should be read as evidence that it does. What is being contributed is the
architecture -- a face branch that knows which parts of a face it can see --
together with an honest measurement of the failure it removes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FaceRegion(str, Enum):
    """The parts a face is judged in.

    Chosen so each can be located from the five keypoints every detection
    carries, and so each corresponds to something a real accessory covers: a
    mask takes the mouth, nose and chin; sunglasses take both eyes; a hood or
    cap takes the forehead.
    """

    FOREHEAD = "forehead"
    EYE_LEFT = "eye_left"
    EYE_RIGHT = "eye_right"
    CHEEK_LEFT = "cheek_left"
    CHEEK_RIGHT = "cheek_right"
    NOSE = "nose"
    MOUTH = "mouth"
    CHIN = "chin"


#: Optional per-region identity weights, or None to weight by area.
#:
#: `scripts/measure_regions.py` derives these by covering one region at a time
#: and recording how far the embedding moves. It is the right way to set them,
#: and it needs a face set this project does not have: measured on the one
#: identity available here, the eyes came out *least* important -- which is
#: false for any deep encoder, and happened because that subject wears glasses,
#: so the region was already partly occluded before the experiment covered it.
#: Freezing that would teach the system that hiding the eyes costs nothing.
#:
#: So the default is None, and weight falls back to the share of face area each
#: region occupies. That assumes only that covering more of a face loses more
#: of it, which is weak but true, and carries no false precision. Run the
#: script on a proper multi-identity set and paste the result here.
REGION_IDENTITY_WEIGHT: dict["FaceRegion", float] | None = None


@dataclass(frozen=True)
class RegionBox:
    """An axis-aligned region of a face, in image coordinates."""

    region: FaceRegion
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def usable(self) -> bool:
        """Big enough to say anything about.

        A 3x2 patch has no texture to measure, and a classifier asked about one
        answers from noise. Regions this small are reported as unknown rather
        than guessed at.
        """
        return self.width >= 8 and self.height >= 8

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return image[
            max(0, self.y1) : min(h, self.y2), max(0, self.x1) : min(w, self.x2)
        ]


def region_boxes(
    bbox: tuple[float, float, float, float],
    keypoints: np.ndarray,
) -> dict[FaceRegion, RegionBox]:
    """Locate each region from the face box and its five keypoints.

    `keypoints` is InsightFace's `kps`, in its documented order: left eye,
    right eye, nose tip, left mouth corner, right mouth corner. "Left" is the
    image's left, not the subject's -- which does not matter here, because the
    regions are symmetric and only ever compared with themselves.

    The five-point set is used rather than the 106-point one on purpose. `kps`
    comes from the detector itself and is always present; `landmark_2d_106`
    needs a second model that a lighter model pack may not ship. Sizes are
    expressed in inter-ocular distance, the standard unit of face scale, so
    regions track the face through zoom and distance without a threshold in
    pixels anywhere.
    """
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 5:
        raise ValueError(
            f"Need five keypoints to locate face regions, got {points.shape[0]}."
        )

    eye_left, eye_right, nose, mouth_left, mouth_right = points[:5]
    x1, y1, x2, y2 = (float(v) for v in bbox)

    # Inter-ocular distance. Falls back to a fraction of box width when the
    # eyes coincide, which happens on a near-profile face where both project
    # to nearly the same point.
    iod = float(np.hypot(*(eye_right - eye_left)))
    if iod < 1.0:
        iod = max(1.0, (x2 - x1) * 0.45)

    eye_y = float((eye_left[1] + eye_right[1]) / 2.0)
    mouth_y = float((mouth_left[1] + mouth_right[1]) / 2.0)
    mid_x = float((eye_left[0] + eye_right[0]) / 2.0)

    def box(region: FaceRegion, cx, cy, half_w, half_h) -> RegionBox:
        return RegionBox(
            region,
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        )

    eye_half_w, eye_half_h = iod * 0.38, iod * 0.28
    # The gap between the eye line and the mouth line sets the vertical scale
    # of everything between them, so a long face does not get a nose region
    # sized for a short one.
    mid_span = max(iod * 0.4, mouth_y - eye_y)

    boxes = [
        # Above the eyes, stopping at the box top: hair and cap live here and
        # the region should not run off into the background.
        RegionBox(
            FaceRegion.FOREHEAD,
            int(round(mid_x - iod * 0.75)),
            int(round(max(y1, eye_y - iod * 1.05))),
            int(round(mid_x + iod * 0.75)),
            int(round(eye_y - iod * 0.30)),
        ),
        box(FaceRegion.EYE_LEFT, eye_left[0], eye_left[1], eye_half_w, eye_half_h),
        box(FaceRegion.EYE_RIGHT, eye_right[0], eye_right[1], eye_half_w, eye_half_h),
        box(
            FaceRegion.NOSE,
            nose[0],
            (eye_y + mouth_y) / 2.0,
            iod * 0.26,
            mid_span * 0.42,
        ),
        box(
            FaceRegion.MOUTH,
            (mouth_left[0] + mouth_right[0]) / 2.0,
            mouth_y,
            iod * 0.46,
            iod * 0.26,
        ),
        # Lateral, between the eye and mouth lines, outside the nose.
        box(
            FaceRegion.CHEEK_LEFT,
            eye_left[0] - iod * 0.24,
            (eye_y + mouth_y) / 2.0,
            iod * 0.30,
            mid_span * 0.34,
        ),
        box(
            FaceRegion.CHEEK_RIGHT,
            eye_right[0] + iod * 0.24,
            (eye_y + mouth_y) / 2.0,
            iod * 0.30,
            mid_span * 0.34,
        ),
        RegionBox(
            FaceRegion.CHIN,
            int(round(mid_x - iod * 0.42)),
            int(round(mouth_y + iod * 0.22)),
            int(round(mid_x + iod * 0.42)),
            int(round(min(y2, mouth_y + iod * 0.85))),
        ),
    ]
    return {b.region: b for b in boxes}


# -- features ---------------------------------------------------------------

#: Bins in the coarse intensity histogram.
#:
#: Sixteen, where the paper used 256. A 256-bin histogram of a synthetic
#: occlusion is a near-delta at whatever grey `disguise.py` happens to paint,
#: and a classifier given that learns the colour rather than the concept --
#: which would score beautifully in testing and detect nothing real. Coarse
#: bins keep the shape of the distribution and throw away its exact position.
INTENSITY_BINS = 16

#: Uniform LBP with 8 neighbours yields 10 rotation-invariant classes.
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2


def ite_features(patch: np.ndarray) -> np.ndarray:
    """Intensity and Texture Encoder for one region (paper's Eq. 1).

    A concatenation of what the region looks like and how it is textured. The
    paper's reasoning holds and is worth restating: some occlusions are
    distinguishable by texture -- hair, a knitted scarf -- and others by
    intensity, such as dark sunglasses against skin. Neither alone separates
    both, and their concatenation does.

    Returned normalised, so a large region and a small one are comparable.
    """
    if patch is None or patch.size == 0:
        return np.zeros(INTENSITY_BINS + LBP_BINS + 3, dtype=np.float32)

    grey = patch if patch.ndim == 2 else _to_grey(patch)
    grey = np.asarray(grey, dtype=np.uint8)

    intensity, _ = np.histogram(
        grey, bins=INTENSITY_BINS, range=(0, 256), density=False
    )
    intensity = intensity.astype(np.float32) / max(1, grey.size)

    # Border pixels are dropped before histogramming. LBP has no neighbours
    # there, so skimage computes them from padding, and the resulting codes are
    # an artefact of the padding rather than of the patch. On a 12x12 region
    # that artefact is 31% of the pixels; on a 48x48 one it is 8% -- so leaving
    # it in made the texture histogram a function of region SIZE, and a
    # classifier fitted on that can learn "small region" instead of "covered".
    lbp = _uniform_lbp(grey)
    if lbp.shape[0] > 2 * LBP_RADIUS and lbp.shape[1] > 2 * LBP_RADIUS:
        lbp = lbp[LBP_RADIUS:-LBP_RADIUS, LBP_RADIUS:-LBP_RADIUS]
    texture, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=False)
    texture = texture.astype(np.float32) / max(1, lbp.size)

    # Three scalars the histograms cannot express. Flatness is what actually
    # separates an accessory from skin: an opaque shape has near-zero local
    # variation wherever it is not an edge, and skin never does.
    values = grey.astype(np.float32)
    local_std = float(np.mean(np.abs(values - _box_blur(values))))
    summary = np.array(
        [
            float(values.std()) / 128.0,
            local_std / 64.0,
            float(np.mean(_edge_density(grey))),
        ],
        dtype=np.float32,
    )

    return np.concatenate([intensity, texture, summary]).astype(np.float32)


FEATURE_DIM = INTENSITY_BINS + LBP_BINS + 3


def _to_grey(patch: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


def _box_blur(values: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.blur(values, (3, 3))


def _edge_density(grey: np.ndarray) -> np.ndarray:
    import cv2

    edges = cv2.Canny(grey, 60, 160)
    return (edges > 0).astype(np.float32)


def _uniform_lbp(grey: np.ndarray) -> np.ndarray:
    """Uniform-pattern LBP, via scikit-image.

    `uniform` rather than the paper's basic 256-class operator: uniform
    patterns collapse the 256 codes into 10 by rotation, which is both far
    fewer parameters to fit and invariant to the face being slightly rotated in
    plane. With patches this small, 256 bins on a few hundred pixels is mostly
    empty anyway.
    """
    from skimage.feature import local_binary_pattern

    return local_binary_pattern(grey, LBP_POINTS, LBP_RADIUS, method="uniform")


# -- occlusion map ----------------------------------------------------------


@dataclass
class OcclusionMap:
    """Which regions of one face are visible, and what that leaves.

    `visible_identity` is the number the face branch needs: the share of the
    face still on view. Weighted by REGION_IDENTITY_WEIGHT when one has been
    measured, and by region area otherwise -- see the note on that constant for
    why measured weights are not shipped by default.
    """

    visible: dict[FaceRegion, bool] = field(default_factory=dict)
    #: Per-region probability that the region is covered, in [0, 1].
    scores: dict[FaceRegion, float] = field(default_factory=dict)
    #: Regions too small or too far off-frame to judge. Treated as visible, so
    #: a tightly cropped photograph is not reported as a disguise.
    unknown: set[FaceRegion] = field(default_factory=set)
    #: Pixel area of each region, used for weighting when no measured weights
    #: exist. Carried on the map rather than recomputed, because the boxes are
    #: derived from this face and are not reconstructable from the result.
    areas: dict[FaceRegion, float] = field(default_factory=dict)

    def _weights(self) -> dict[FaceRegion, float]:
        if REGION_IDENTITY_WEIGHT:
            return REGION_IDENTITY_WEIGHT
        return self.areas or {region: 1.0 for region in FaceRegion}

    @property
    def visible_identity(self) -> float:
        """Weighted share of the face still visible, in [0, 1]."""
        weights = self._weights()
        # Judged over the regions actually located on this face. Including a
        # region that fell outside the frame would count it as lost identity
        # when nothing was hidden -- a face at the edge of a shot would then
        # look disguised.
        considered = [r for r in FaceRegion if r in weights]
        total = sum(weights.get(r, 0.0) for r in considered)
        if total <= 0:
            return 1.0
        seen = sum(
            weights.get(region, 0.0)
            for region in considered
            if self.visible.get(region, True)
        )
        return float(np.clip(seen / total, 0.0, 1.0))

    @property
    def occluded_regions(self) -> list[FaceRegion]:
        return [r for r in FaceRegion if not self.visible.get(r, True)]

    def describe(self) -> str:
        """One line for the explainability view.

        A reviewer looking at a weak face score needs to know *why* it is weak.
        "mouth and nose covered" is actionable; 0.31 is not.
        """
        covered = self.occluded_regions
        if not covered:
            return "whole face visible"
        names = ", ".join(r.value.replace("_", " ") for r in covered)
        return f"{names} covered ({self.visible_identity:.0%} of the face usable)"


# -- detector ---------------------------------------------------------------

#: Logistic regression over ITE features, fitted offline and frozen here.
#:
#: Inline rather than a checkpoint file for the same reason the rest of this
#: project avoids them where it can: a weights file is one more thing to ship,
#: to version, and to get out of step with the code that reads it. Thirty
#: coefficients are small enough to live in the source, where they are diffable
#: and reviewable.
#:
#: Regenerate with `python scripts/train_occlusion.py`, which prints a block
#: ready to paste over this one. scikit-learn is deliberately not a dependency
#: -- see requirements.txt -- so the fit is a few lines of gradient descent in
#: that script.
OCCLUSION_MODEL: dict[str, list[float]] = {
    "weights": [
        -0.235769, 0.0593878, -0.0997707, -0.234576, -0.148839, 0.0753261,
        -0.312844, -0.461502, -0.383929, 0.0983703, -0.230337, -0.0606139,
        0.0702654, 0.200127, 0.0836713, -0.000468577, 0.734623, -0.0038069,
        -0.365206, -1.16793, -1.34187, -0.57374, -0.738149, -0.0379149,
        0.517381, 0.504824, 0.0592027, 0.746505, 0.340888, 4.20549,
    ],
    "mean": [
        0.00300474, 0.13242, 0.0781848, 0.0143879, 0.0205074, 0.150089,
        0.0310004, 0.0310093, 0.0306026, 0.275771, 0.0201865, 0.0130647,
        0.0679012, 0.00483069, 0.126796, 0.000244777, 0.0251246, 0.022252,
        0.0112833, 0.0189245, 0.0401069, 0.065904, 0.014182, 0.0225658,
        0.73828, 0.0413764, 0.155769, 0.149698, 0.10716,
    ],
    "std": [
        0.0360517, 0.330093, 0.175163, 0.030839, 0.0425149, 0.324933,
        0.0634241, 0.063081, 0.0636211, 0.421267, 0.0509624, 0.0319668,
        0.15602, 0.0124638, 0.330075, 0.00109177, 0.0564215, 0.0411636,
        0.0204143, 0.040805, 0.101868, 0.113859, 0.026396, 0.0418496,
        0.37517, 0.0847933, 0.225058, 0.240391, 0.153026,
    ],
}

#: Probability above which a region is called covered.
#:
#: Above 0.5 on purpose. The two errors are not symmetric: calling a visible
#: region covered costs a little quality on a face that was fine, while calling
#: a covered region visible restores exactly the failure this module exists to
#: remove. But a detector that cries disguise at every shadow gets switched off,
#: so this is not pushed further down.
OCCLUDED_THRESHOLD = 0.65


class OcclusionDetector:
    """Reports which regions of a face are covered.

    Stateless and cheap -- histograms and an LBP over eight small patches --
    so it can run on every face the branch embeds rather than on a sample.
    """

    def __init__(self, model: dict[str, list[float]] | None = None) -> None:
        model = model or OCCLUSION_MODEL
        self.weights = np.asarray(model.get("weights", []), dtype=np.float64)
        self.mean = np.asarray(model.get("mean", []), dtype=np.float64)
        self.std = np.asarray(model.get("std", []), dtype=np.float64)

    @property
    def trained(self) -> bool:
        """Whether there are usable coefficients.

        Checked rather than assumed: an untrained detector must report "I do
        not know" and leave quality alone, not silently score every face as
        fully visible, which would look identical to working.
        """
        return (
            self.weights.size == FEATURE_DIM + 1
            and self.mean.size == FEATURE_DIM
            and self.std.size == FEATURE_DIM
        )

    def probability(self, patch: np.ndarray) -> float:
        """Probability that this patch is covered."""
        if not self.trained:
            return 0.0
        x = (ite_features(patch).astype(np.float64) - self.mean) / np.maximum(
            self.std, 1e-8
        )
        z = float(np.dot(np.append(x, 1.0), self.weights))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))

    def detect(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        keypoints: np.ndarray,
    ) -> OcclusionMap:
        """Occlusion map for one detected face."""
        result = OcclusionMap()
        if not self.trained:
            # Every region unknown, so `visible_identity` is 1.0 and quality is
            # unchanged. An undertrained model must not quietly penalise faces.
            result.unknown = set(FaceRegion)
            return result

        height, width = image.shape[:2]
        for region, box in region_boxes(bbox, keypoints).items():
            patch = box.crop(image)
            # `usable` is about the box; the crop can still come back empty or
            # clipped to nothing when the face runs off the edge of the frame.
            if not box.usable or patch.size == 0 or min(patch.shape[:2]) < 8:
                result.unknown.add(region)
                result.visible[region] = True
                continue

            probability = self.probability(patch)
            result.scores[region] = probability
            result.visible[region] = probability < OCCLUDED_THRESHOLD
            result.areas[region] = float(patch.shape[0] * patch.shape[1])

        del height, width
        return result
