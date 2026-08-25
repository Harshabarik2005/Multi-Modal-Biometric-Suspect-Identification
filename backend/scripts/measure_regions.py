"""Measure how much identity each face region carries (Phase 11).

    python scripts/measure_regions.py --image path/to/face.jpg
    python scripts/measure_regions.py --image face.jpg --draw regions.png

Covers one region at a time and records how far the face embedding moves. The
drop *is* the region's weight: if blanking the mouth barely changes the
embedding, the mouth was not carrying much identity for this encoder, and a
covered mouth should not cost much quality.

Why measure rather than assume
------------------------------
Dhamecha et al. report the same experiment on *people* -- their Figure 5 gives
the misclassification rate for each disguised facial part. Those numbers
describe human vision. ArcFace is not human vision, and there is no reason its
weighting should match; periocular structure matters far more to a deep
encoder than it does to a person, who leans on hairline and face outline.

So the weights in `occlusion.py` come from here, on the encoder actually
configured. Re-run this after changing `face.model_pack`: weights measured on
one encoder say nothing about another.

The covering is a flat grey block, not a mask. That is the point -- it removes
the region's information without adding a specific accessory's, so what is
measured is the region's contribution rather than a particular disguise's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.types import TrackObservation, cosine_similarity  # noqa: E402
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.embeddings.occlusion import FaceRegion, region_boxes  # noqa: E402

#: Mid-grey. Chosen to sit near the mean of most skin tones, so the block
#: removes structure without introducing a strong new edge that the encoder
#: could react to in its own right.
COVER = (128, 128, 128)


def observation_of(image: np.ndarray) -> TrackObservation:
    return TrackObservation(
        frame_index=0,
        timestamp_s=0.0,
        crop=image,
        box_height=float(image.shape[0]),
        detection_confidence=0.95,
    )


def draw_regions(image: np.ndarray, boxes: dict) -> np.ndarray:
    out = image.copy()
    for region, box in boxes.items():
        cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 1)
        cv2.putText(
            out,
            region.value[:6],
            (box.x1 + 1, max(8, box.y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True, help="A face photo.")
    parser.add_argument("--draw", type=Path, default=None, help="Save a region overlay.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.logging.level)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Could not read {args.image}")
        return 1

    embedder = FaceEmbedder(settings)
    faces = embedder.detect(image)
    if not faces:
        print("No face detected.")
        return 1

    face = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    boxes = region_boxes(tuple(float(v) for v in face.bbox), face.kps)

    if args.draw:
        args.draw.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.draw), draw_regions(image, boxes))
        print(f"Wrote region overlay to {args.draw}")

    clean = embedder.embed_frame(observation_of(image))
    if not clean.has_signal:
        print(f"The clean image produced no embedding: {clean.reason}")
        return 1

    print(f"\nclean quality {clean.quality:.3f}\n")
    print(f"{'region':<14} {'size':>9} {'similarity':>11} {'drop':>7}  weight")
    print("-" * 60)

    drops: dict[FaceRegion, float] = {}
    for region, box in boxes.items():
        if not box.usable:
            print(f"{region.value:<14} {'too small':>9}")
            drops[region] = 0.0
            continue

        covered = image.copy()
        cv2.rectangle(
            covered, (box.x1, box.y1), (box.x2, box.y2), COVER, thickness=-1
        )
        got = embedder.embed_frame(observation_of(covered))

        if not got.has_signal:
            # Covering this region stopped detection entirely, which is the
            # strongest statement possible about how much it mattered.
            print(f"{region.value:<14} {f'{box.width}x{box.height}':>9} "
                  f"{'no face':>11} {'1.000':>7}  (detection lost)")
            drops[region] = 1.0
            continue

        similarity = cosine_similarity(clean.vector, got.vector)
        drop = max(0.0, 1.0 - similarity)
        drops[region] = drop
        print(f"{region.value:<14} {f'{box.width}x{box.height}':>9} "
              f"{similarity:>11.4f} {drop:>7.4f}")

    total = sum(drops.values())
    print("\nREGION_IDENTITY_WEIGHT = {")
    for region in FaceRegion:
        share = drops.get(region, 0.0) / total if total > 0 else 1.0 / len(FaceRegion)
        print(f"    FaceRegion.{region.name}: {share:.3f},")
    print("}")
    print(
        "\nWeights are normalised drops, so they sum to 1 and express relative\n"
        "importance rather than absolute similarity change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
