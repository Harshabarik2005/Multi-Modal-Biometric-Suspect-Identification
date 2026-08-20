"""Disguise augmentation (Phase 11) -- masks, sunglasses, hoods.

The build plan claims an "occlusion-aware face branch". This module is what
makes that claim testable: it synthesises the common occlusions onto face crops
so the face branch's behaviour under them can be *measured* rather than
asserted.

What this is for, and what it is not
------------------------------------
It is a **test instrument** first. Given a clean enrollment image, it produces
the same face wearing a surgical mask or sunglasses, so you can measure how far
the embedding moves and whether the quality score notices. A branch that
returns a confident embedding for a masked face is worse than one that returns
nothing, because fusion will trust it.

It can also generate **training** augmentation, but a synthetic mask is a flat
shape pasted on a photograph, not a real mask under real lighting. Training on
these alone teaches robustness to *this drawing*, not to masks. Treat improved
numbers on synthetic disguises as evidence the plumbing works, not as evidence
of real disguise robustness.

The occlusions are deliberately crude and opaque. A subtle, realistic mask
would be a weaker test: if the branch copes with a hard-edged block covering
the whole lower face, it will cope with a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Disguise(str, Enum):
    """The occlusions worth testing, chosen for what they remove."""

    #: Covers nose and mouth. Leaves the eyes, which carry a lot of identity.
    MASK = "mask"
    #: Covers the eyes. Removes the most informative region for most encoders.
    SUNGLASSES = "sunglasses"
    #: Covers the hairline, forehead and face outline.
    HOOD = "hood"
    #: Mask plus sunglasses: very little face left.
    MASK_AND_SUNGLASSES = "mask_and_sunglasses"
    #: Heavy blur, standing in for distance or motion rather than disguise.
    BLUR = "blur"


@dataclass
class DisguiseResult:
    image: np.ndarray
    disguise: Disguise
    #: Fraction of the face region that was covered. Useful for reporting how
    #: hard a given test actually was.
    occluded_fraction: float


def _region(shape: tuple[int, ...], top: float, bottom: float) -> tuple[int, int]:
    height = shape[0]
    return int(height * top), int(height * bottom)


def apply_mask(image: np.ndarray, colour: tuple[int, int, int] = (210, 210, 215)):
    """A surgical mask: the lower ~45% of the face, with ear loops.

    Placed by proportion rather than by detected landmarks so it works on any
    crop, including ones where the detector would fail -- which is precisely
    the case worth testing.
    """
    out = image.copy()
    height, width = out.shape[:2]
    top, bottom = _region(out.shape, 0.55, 1.0)

    # A rounded trapezoid, roughly mask-shaped.
    points = np.array(
        [
            [int(width * 0.10), top],
            [int(width * 0.90), top],
            [int(width * 0.82), bottom - 1],
            [int(width * 0.18), bottom - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [points], colour)
    # Ear loops, so the shape does not read as a plain rectangle.
    cv2.line(out, (int(width * 0.10), top), (0, int(top * 0.85)), colour, 2)
    cv2.line(out, (int(width * 0.90), top), (width, int(top * 0.85)), colour, 2)
    return out, (bottom - top) / height * 0.8


def apply_sunglasses(image: np.ndarray, colour: tuple[int, int, int] = (25, 25, 30)):
    """Dark glasses across the eye band."""
    out = image.copy()
    height, width = out.shape[:2]
    top, bottom = _region(out.shape, 0.28, 0.48)

    lens_width = int(width * 0.34)
    for x in (int(width * 0.08), width - int(width * 0.08) - lens_width):
        cv2.rectangle(out, (x, top), (x + lens_width, bottom), colour, -1)
    # Bridge.
    cv2.rectangle(
        out,
        (int(width * 0.42), top + (bottom - top) // 3),
        (int(width * 0.58), top + (bottom - top) // 2),
        colour,
        -1,
    )
    return out, (bottom - top) / height * 0.76


def apply_hood(image: np.ndarray, colour: tuple[int, int, int] = (40, 42, 48)):
    """A hood: the top band plus the sides, leaving a central oval visible."""
    out = image.copy()
    height, width = out.shape[:2]

    cv2.rectangle(out, (0, 0), (width, int(height * 0.22)), colour, -1)
    side = int(width * 0.16)
    cv2.rectangle(out, (0, 0), (side, height), colour, -1)
    cv2.rectangle(out, (width - side, 0), (width, height), colour, -1)
    return out, 0.22 + (2 * side / width) * 0.78


def apply_blur(image: np.ndarray, strength: int = 9):
    """Heavy blur. Stands in for distance or motion, not disguise.

    Included because it is the most common real reason a face becomes
    unusable, and a quality score that only reacts to occlusion would miss it.
    """
    kernel = max(3, strength | 1)
    return cv2.GaussianBlur(image, (kernel, kernel), 0), 0.0


def apply(image: np.ndarray, disguise: Disguise) -> DisguiseResult:
    """Apply one disguise to a BGR image."""
    if image is None or image.size == 0:
        raise ValueError("Cannot disguise an empty image.")

    if disguise is Disguise.MASK:
        out, fraction = apply_mask(image)
    elif disguise is Disguise.SUNGLASSES:
        out, fraction = apply_sunglasses(image)
    elif disguise is Disguise.HOOD:
        out, fraction = apply_hood(image)
    elif disguise is Disguise.MASK_AND_SUNGLASSES:
        masked, mask_fraction = apply_mask(image)
        out, glasses_fraction = apply_sunglasses(masked)
        fraction = min(1.0, mask_fraction + glasses_fraction)
    elif disguise is Disguise.BLUR:
        out, fraction = apply_blur(image)
    else:
        raise ValueError(f"Unknown disguise: {disguise}")

    return DisguiseResult(
        image=out, disguise=disguise, occluded_fraction=float(fraction)
    )


def apply_to_region(
    image: np.ndarray,
    disguise: Disguise,
    box: tuple[float, float, float, float],
    margin: float = 0.15,
) -> DisguiseResult:
    """Apply a disguise to a face region inside a larger image.

    Necessary because the pipeline works in *body* crops, while these
    occlusions are proportioned to a *face*. Applied to a body crop directly, a
    "mask" lands somewhere around the knees -- which was exactly the bug that
    made an early robustness run report similarity 1.000 for a masked face, and
    look like impressive invariance rather than a broken test.

    `box` is the face box in the image's own coordinates. A small margin is
    added because face detectors crop tightly and a mask extends past the jaw.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1

    x1 = max(0, int(x1 - box_width * margin))
    y1 = max(0, int(y1 - box_height * margin))
    x2 = min(width, int(x2 + box_width * margin))
    y2 = min(height, int(y2 + box_height * margin))

    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f"Face region is too small to disguise: {box}")

    face = image[y1:y2, x1:x2]
    disguised = apply(face, disguise)

    out = image.copy()
    out[y1:y2, x1:x2] = disguised.image
    # Report occlusion relative to the FACE, not the whole image: covering half
    # a face is the meaningful number, not what fraction of a body crop it is.
    return DisguiseResult(
        image=out,
        disguise=disguise,
        occluded_fraction=disguised.occluded_fraction,
    )


def all_disguises(image: np.ndarray) -> dict[Disguise, DisguiseResult]:
    """Every disguise applied to one image, for a robustness sweep."""
    return {disguise: apply(image, disguise) for disguise in Disguise}
