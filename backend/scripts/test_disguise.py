"""Measure the face branch's behaviour under disguise (Phase 11).

    python scripts/test_disguise.py
    python scripts/test_disguise.py --save-crops

Takes the enrollment fixture, applies each synthetic disguise, and reports two
things per disguise:

* **similarity** to the undisguised reference -- how far the embedding moved.
* **quality** -- whether the branch *noticed*.

The second number is the one that matters. A face branch that returns a
confident embedding for a masked face is worse than one that returns nothing,
because fusion weights confidence and would trust it. What we want to see is
similarity falling and quality falling with it; what would be alarming is
similarity collapsing while quality stays high.

The disguises are synthetic: flat shapes on a photograph, not real masks under
real lighting. Numbers here show whether the branch reacts to occlusion at all,
not how it performs against real disguises. That needs DFW or comparable
footage.
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
from app.core.track_buffer import TrackBufferStore  # noqa: E402
from app.core.types import TrackObservation  # noqa: E402
from app.embeddings.disguise import Disguise, apply_to_region  # noqa: E402
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402

REPO = BACKEND_ROOT.parent
CLIP = REPO / "data" / "test_videos" / "enroll_subject_a.mp4"


def best_crop(settings) -> np.ndarray | None:
    """The largest person crop from the enrollment clip."""
    settings.video.max_frames = 12
    pipeline = DetectionTrackingPipeline(settings)
    store = TrackBufferStore(settings)
    for result, frame in pipeline.stream(CLIP):
        store.update(result, frame)

    best, best_height = None, 0.0
    for buffer in store:
        for observation in buffer:
            if observation.box_height > best_height:
                best, best_height = observation.crop, observation.box_height
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "output")
    args = parser.parse_args(argv)

    if not CLIP.exists():
        print(f"Missing {CLIP}. Run scripts/make_face_fixtures.py first.")
        return 1

    settings = get_settings()
    setup_logging(settings.logging.level)

    crop = best_crop(settings)
    if crop is None:
        print("No person crop found in the enrollment clip.")
        return 1

    embedder = FaceEmbedder(settings)
    clean = embedder.embed_frame(
        TrackObservation(0, 0.0, crop, float(crop.shape[0]), 0.9)
    )
    if not clean.has_signal:
        print(
            "No face found in the clean crop, so there is nothing to compare "
            "against. Check the fixture."
        )
        return 1

    # Locate the face ONCE, then disguise that region rather than the whole
    # body crop. These occlusions are proportioned to a face; applied to a body
    # crop a "mask" lands around the knees and the test measures nothing.
    faces = embedder.detect(crop)
    if not faces:
        print("Could not locate a face box to disguise.")
        return 1
    face_box = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ).bbox

    print(
        f"\nFace located at {[int(v) for v in face_box]} within a "
        f"{crop.shape[1]}x{crop.shape[0]} body crop."
    )
    print(f"Baseline (no disguise): quality {clean.quality:.3f}\n")

    header = (
        f"{'disguise':<22} {'covered':>8} {'detected':>9} "
        f"{'similarity':>11} {'quality':>8}  verdict"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for disguise in Disguise:
        result = apply_to_region(crop, disguise, face_box)
        embedded = embedder.embed_frame(
            TrackObservation(
                0, 0.0, result.image, float(result.image.shape[0]), 0.9
            )
        )

        if args.save_crops:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(args.out_dir / f"disguise_{disguise.value}.png"), result.image
            )

        if not embedded.has_signal:
            print(
                f"{disguise.value:<22} {result.occluded_fraction:>7.0%} "
                f"{'no':>9} {'-':>11} {'-':>8}  no signal (safe)"
            )
            rows.append((disguise.value, None, 0.0))
            continue

        similarity = embedded.similarity(clean)
        # The failure mode worth naming: the embedding moved a long way but
        # the branch still thinks it got a good look.
        risky = similarity is not None and similarity < 0.5 and embedded.quality > 0.3
        verdict = "RISKY: confident but wrong" if risky else "degrades as expected"

        print(
            f"{disguise.value:<22} {result.occluded_fraction:>7.0%} "
            f"{'yes':>9} {similarity:>11.3f} {embedded.quality:>8.3f}  {verdict}"
        )
        rows.append((disguise.value, similarity, embedded.quality))

    print()
    risky = [
        name
        for name, similarity, quality in rows
        if similarity is not None and similarity < 0.5 and quality > 0.3
    ]
    if risky:
        print(
            "CONCERN: " + ", ".join(risky) + "\n"
            "  These produced embeddings far from the reference while still\n"
            "  reporting usable quality. Fusion weights by quality, so it would\n"
            "  trust them. Consider tightening face.min_quality."
        )
    else:
        print(
            "No disguise produced a confidently-wrong embedding: either the\n"
            "branch found no face (safe), or similarity and quality fell\n"
            "together. That is the behaviour fusion needs."
        )

    print(
        "\nThese are synthetic occlusions -- flat shapes on a photograph. They\n"
        "show the branch reacts to occlusion, not how it fares against real\n"
        "disguises. That needs DFW or comparable footage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
