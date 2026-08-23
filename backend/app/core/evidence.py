"""Evidence images: the crop a reviewer actually looks at (DES-02).

The architecture rests on a human confirming every identification, and the
review card showed a score, some weight bars and a caution line. No crop, no
frame, no clip. A reviewer could see *how* the system reached its conclusion
and had no way at all to judge *whether* it was right -- which turns the
guardrail into a formality, and a formality that produces an audit trail
saying a human checked.

A crop is not a template. A template is a biometric vector that identifies a
person across footage they have never appeared in; a crop is a picture of one
moment, which is what somebody needs in order to say "no, that is not him".
They are stored with the same encryption because a picture of an identified
person is still personal data, and they are bounded in size and number for the
same reason -- this is evidence for one decision, not a photo library.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Tall enough to recognise a face on, small enough that thousands of them do
#: not turn the database into an image store. ~10-20KB each in practice.
MAX_HEIGHT = 320
JPEG_QUALITY = 80


def encode_crop(crop: np.ndarray | None, max_height: int = MAX_HEIGHT) -> bytes | None:
    """JPEG-encode a person crop for review. None when there is nothing usable.

    Returns None rather than raising: a decision with no image is worse than
    one with an image, but far better than a scan that dies because one frame
    was malformed.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if crop.ndim != 3 or crop.shape[2] != 3:
        return None

    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return None

    if height > max_height:
        scale = max_height / height
        crop = cv2.resize(
            crop,
            (max(1, int(round(width * scale))), max_height),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        logger.warning("Could not JPEG-encode a review crop; storing none.")
        return None
    return bytes(buffer)


def best_reference_crop(
    observations, max_height: int = MAX_HEIGHT
) -> bytes | None:
    """One representative crop from an enrolment, for side-by-side comparison.

    The largest observation, because that is the closest look the enrolment
    footage got -- the same reasoning the branches use when they weight by
    quality.
    """
    usable = [
        o
        for o in observations
        if getattr(o, "crop", None) is not None and o.crop.size > 0
    ]
    if not usable:
        return None
    return encode_crop(max(usable, key=lambda o: o.crop.shape[0]).crop, max_height)
