"""Measure re-ID calibration anchors from real footage (Phase 10).

    python scripts/measure_calibration.py --person case-0001         --videos "A:/test/**/*.mp4"

config.yaml ships reid_impostor=0.726 / reid_genuine=0.881, measured on "one
photograph, two people, no variation in pose or lighting", and tells you to
re-derive them before operational use. This is how.

Why it matters more than it sounds
----------------------------------
`ModalityCalibration.calibrate` clips at both ends, so every similarity at or
below the impostor anchor becomes exactly 0.0. If real strangers on your
cameras score well under the shipped 0.726, then genuine matches are being
clipped to zero too -- and the modality contributes nothing to fusion while
still consuming weight. That failure is invisible in a match score: it looks
like re-ID simply disagreeing.

Face is used as the labeller. When it is frontal it is far more reliable than
re-ID, so tracks whose face clearly matches the enrolled subject are genuine,
tracks whose face clearly does not are impostors, and everything between is
left out rather than guessed at. One mislabelled pair poisons the anchor it is
meant to establish.

Read the separation warning seriously
-------------------------------------
The script refuses to suggest anchors when the genuine and impostor
distributions overlap, because no pair of anchors can separate distributions
that are not separated. Measured on classroom footage of students in similar
light shirts, genuine re-ID pairs scored 0.493-0.814 and impostors 0.516-0.710
-- overlapping almost entirely. Re-ID largely encodes clothing, so a room full
of people dressed alike is close to the worst case for it. The honest response
there is to distrust the modality on that footage, not to move the anchors
until the numbers look better.
"""
import argparse
import glob
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.track_buffer import TrackBufferStore  # noqa: E402
from app.core.types import Modality, cosine_similarity  # noqa: E402
from app.db.repository import (  # noqa: E402
    WatchlistRepository, make_engine, session_factory,
)
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.embeddings.reid import ReIDEmbedder  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402

#: Face similarity above this means the track is the enrolled subject; below
#: the lower bound means somebody else. The gap is deliberately left
#: unlabelled -- one mislabelled pair poisons the anchor it establishes.
GENUINE_FACE = 0.55
IMPOSTOR_FACE = 0.30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--person", required=True, help="Enrolled person_id to label against.")
    parser.add_argument(
        "--videos", action="append", required=True,
        help="Glob of footage to measure. Repeatable.",
    )
    parser.add_argument(
        "--genuine-face", type=float, default=GENUINE_FACE,
        help="Face similarity at or above which a track is the subject.",
    )
    parser.add_argument(
        "--impostor-face", type=float, default=IMPOSTOR_FACE,
        help="Face similarity at or below which a track is somebody else.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.logging.level)

    face_embedder = FaceEmbedder(settings)
    reid_embedder = ReIDEmbedder(settings)

    engine = make_engine(f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}")
    person = WatchlistRepository(session_factory(engine)()).load_gallery().get(args.person)
    if person is None:
        print(f"No enrolled person {args.person!r}.")
        return 1
    if Modality.FACE not in person.embeddings or Modality.REID not in person.embeddings:
        print(f"{args.person} needs both a face and a re-ID template to calibrate "
              "against: face does the labelling, re-ID is what gets measured.")
        return 1

    face_template = person.embeddings[Modality.FACE]
    reid_template = person.embeddings[Modality.REID]

    paths: list[str] = []
    for pattern in args.videos:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    if not paths:
        print("No footage matched.")
        return 1

    cap = settings.matching.max_observations_to_embed
    genuine: list[float] = []
    impostor: list[float] = []
    unlabelled = 0

    print(f"{'clip':<20} {'track':>5} {'face':>7} {'reid':>7}  label")
    print("-" * 52)

    for video in paths:
        path = Path(video)
        pipeline = DetectionTrackingPipeline(settings)
        store = TrackBufferStore(settings)
        for result, frame in pipeline.stream(path):
            store.update(result, frame)

        for buffer in sorted(store, key=len, reverse=True):
            if len(buffer) < settings.matching.min_track_observations:
                continue
            sampled = buffer.best(cap) if cap else list(buffer)

            reid = reid_embedder.embed(sampled)
            if not reid.has_signal:
                continue
            reid_sim = cosine_similarity(reid.vector, reid_template.vector)

            face = face_embedder.embed(sampled)
            if not face.has_signal:
                unlabelled += 1
                print(f"{path.stem:<20} {buffer.track_id:>5} {'--':>7} "
                      f"{reid_sim:>7.3f}  (no face, unlabelled)")
                continue

            face_sim = cosine_similarity(face.vector, face_template.vector)
            if face_sim >= args.genuine_face:
                genuine.append(reid_sim)
                label = "GENUINE"
            elif face_sim <= args.impostor_face:
                impostor.append(reid_sim)
                label = "impostor"
            else:
                unlabelled += 1
                label = "(ambiguous)"
            print(f"{path.stem:<20} {buffer.track_id:>5} {face_sim:>7.3f} "
                  f"{reid_sim:>7.3f}  {label}")

    print()
    print(f"labelled by face: {len(genuine)} genuine, {len(impostor)} impostor, "
          f"{unlabelled} left out")

    if not genuine or not impostor:
        print("\nNot enough labelled pairs on both sides to say anything. Add "
              "footage containing the subject AND other people, with the face "
              "visible often enough to label the tracks.")
        return 1

    g, i = np.array(genuine), np.array(impostor)
    print()
    print(f"  genuine  re-ID: min {g.min():.3f}  median {np.median(g):.3f}  max {g.max():.3f}")
    print(f"  impostor re-ID: min {i.min():.3f}  median {np.median(i):.3f}  max {i.max():.3f}")
    print()
    print(f"  shipped anchors: impostor {settings.fusion.reid_impostor}  "
          f"genuine {settings.fusion.reid_genuine}")
    clipped = int((g < settings.fusion.reid_impostor).sum())
    if clipped:
        print(f"  -> {clipped}/{len(g)} GENUINE pairs sit below the shipped impostor "
              f"anchor,\n"
          "so they calibrate to exactly 0.0 and contribute nothing.")

    # Percentiles rather than extremes, so one freak pair cannot set the scale.
    imp = float(np.percentile(i, 90))
    gen = float(np.percentile(g, 50))
    print()
    if gen <= imp:
        print("  NO ANCHORS SUGGESTED.")
        print(f"  The genuine median ({gen:.3f}) is not above the impostor 90th "
              f"percentile ({imp:.3f}):")
        print("  these distributions overlap, and no pair of anchors can separate")
        print("  distributions that are not separated. Moving the anchors until the")
        print("  numbers look better would only relabel noise as confidence.")
        print()
        print("  Re-ID largely encodes clothing, so people dressed alike -- a class,")
        print("  a uniform, a crowd in similar shirts -- is close to its worst case.")
        print("  Distrust the modality on this footage rather than tuning it.")
        return 2

    print("  Suggested, for backend/config.yaml:")
    print(f"    reid_impostor: {imp:.3f}   # 90th percentile of measured impostors")
    print(f"    reid_genuine:  {gen:.3f}   # median of measured genuine pairs")
    print(f"    reid_anchors_measured_on: {settings.reid.weights}")
    print()
    print(f"  Based on {len(g)} genuine and {len(i)} impostor pairs. That is a "
          "small sample;\n"
          "  the more footage and the more enrolled people, the "
          "less these move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
