"""Enroll a person into the watchlist from a video (Phase 2).

Feeds enrollment footage through the Phase-1 detect+track pipeline, collects
the crops belonging to the dominant track, and stores a reference face
embedding for that person.

The footage should show ONE person, ideally rotating through 360 degrees so
the reference covers a range of angles. If several people are detected the
script uses the longest-lived track and says so, rather than silently averaging
strangers into one identity -- a corrupted reference vector is worse than a
failed enrollment, because it produces confident wrong matches forever after.

    python scripts/enroll.py --source ../data/enrollment/raw/ravi.mp4 \
        --person-id ravi --name "Ravi Kumar"

    python scripts/enroll.py --list
    python scripts/enroll.py --inspect ravi
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.track_buffer import TrackBufferStore  # noqa: E402
from app.core.types import Modality  # noqa: E402
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.embeddings.gait import GaitEmbedder  # noqa: E402
from app.matching.gallery import GalleryStore, PersonRecord  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402


def collect_observations(settings, source: str) -> dict[int, list]:
    """Run detect+track over the footage, returning observations per track."""
    # Enrollment wants every usable frame, not a rolling window, so the buffer
    # cap is lifted well above the live-matching default for this run only.
    settings.track_buffer.max_observations = 100_000
    settings.track_buffer.max_tracks = 100

    pipeline = DetectionTrackingPipeline(settings)
    store = TrackBufferStore(settings)

    frames = 0
    for result, frame in pipeline.stream(source):
        store.update(result, frame)
        frames += 1

    print(f"Processed {frames} frames, found {len(store)} track(s).")
    return {buffer.track_id: list(buffer) for buffer in store}


def enroll(args: argparse.Namespace) -> int:
    settings = get_settings(args.config)
    setup_logging(settings.logging.level)

    tracks = collect_observations(settings, args.source)
    if not tracks:
        print(
            "\nNo people were tracked in that footage. Check the video shows a "
            "person clearly, or lower detection.conf_threshold."
        )
        return 1

    # The enrollment subject is the person on screen the longest.
    ordered = sorted(tracks.items(), key=lambda kv: len(kv[1]), reverse=True)
    track_id, observations = ordered[0]

    if len(ordered) > 1:
        others = ", ".join(f"#{tid} ({len(obs)}f)" for tid, obs in ordered[1:6])
        print(
            f"\nWARNING: {len(ordered)} tracks were found. Enrolling the longest "
            f"one, track #{track_id} ({len(observations)} frames).\n"
            f"         Ignored: {others}\n"
            "         If that is the wrong person, re-record with only the "
            "subject in frame -- a reference vector blended from several people "
            "produces confident wrong matches."
        )

    print(f"\nEmbedding {len(observations)} frames from track #{track_id}...")
    embeddings = {}

    face = FaceEmbedder(settings).embed_reference(observations)
    if face.has_signal:
        candidates = face.detail.get("enroll_candidates", 0)
        selected = face.detail.get("enroll_selected", 0)
        print(
            f"  face: {int(selected)} of {int(candidates)} usable frames kept "
            f"(top {settings.face.enroll_top_fraction:.0%} by quality), "
            f"quality {face.quality:.3f}"
        )
        embeddings[Modality.FACE] = face
    else:
        print(
            "  face: no usable face found.\n"
            "        - Is the subject's face visible and reasonably frontal?\n"
            "        - Faces need roughly 112px of height to embed well.\n"
            f"        - Try lowering face.min_quality (now {settings.face.min_quality})."
        )

    if not args.no_gait:
        gait = GaitEmbedder(settings).embed_reference(observations)
        if gait.has_signal:
            print(
                f"  gait: {gait.frames_used} silhouettes, "
                f"{gait.detail.get('cycles', 0):.1f} gait cycles, "
                f"quality {gait.quality:.3f}"
            )
            embeddings[Modality.GAIT] = gait
        else:
            print(
                "  gait: no gait signal. The subject has to be WALKING through\n"
                "        several full step cycles. A 360-degree rotation on the\n"
                "        spot gives an excellent face reference and no gait\n"
                "        reference at all."
            )

    if not embeddings:
        print("\nNothing could be enrolled from this footage.")
        return 1

    person = PersonRecord(
        person_id=args.person_id,
        display_name=args.name or args.person_id,
        embeddings=embeddings,
        enrolled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=args.notes or "",
        source=str(args.source),
    )

    store = GalleryStore(settings)
    if store.person_dir(args.person_id).exists() and not args.force:
        print(
            f"\n{args.person_id!r} is already enrolled. Re-run with --force to "
            "replace that reference."
        )
        return 1

    directory = store.save(person)
    print(f"\nSaved to {directory}")
    print("Stored modalities: " + ", ".join(sorted(m.value for m in embeddings)))
    if Modality.GAIT not in embeddings:
        print(
            "No gait reference stored -- this person will match on face alone "
            "until you enroll them walking."
        )
    print("Re-ID is not built yet (phase 4).")
    return 0


def list_people(args: argparse.Namespace) -> int:
    settings = get_settings(args.config)
    store = GalleryStore(settings)
    ids = store.list_person_ids()
    if not ids:
        print(f"No one is enrolled yet. Enrollment directory: {store.root}")
        return 0

    print(f"{len(ids)} enrolled in {store.root}:\n")
    gallery = store.load_gallery()
    for person in gallery:
        modalities = ", ".join(m.value for m in person.modalities) or "none"
        print(f"  {person.person_id:<20} {person.display_name:<28} [{modalities}]")
        if person.enrolled_at:
            print(f"  {'':<20} enrolled {person.enrolled_at}")
    return 0


def inspect(args: argparse.Namespace) -> int:
    settings = get_settings(args.config)
    store = GalleryStore(settings)
    try:
        person = store.load_person(args.inspect)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"person_id    : {person.person_id}")
    print(f"display_name : {person.display_name}")
    print(f"enrolled_at  : {person.enrolled_at}")
    print(f"source       : {person.source}")
    if person.notes:
        print(f"notes        : {person.notes}")
    for modality in person.modalities:
        embedding = person.embeddings[modality]
        print(f"\n[{modality.value}]")
        print(f"  dim         : {embedding.vector.size}")
        print(f"  quality     : {embedding.quality:.3f}")
        print(f"  frames_used : {embedding.frames_used}")
        for key, value in sorted(embedding.detail.items()):
            print(f"  {key:<12}: {value:.3f}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="Enrollment video path.")
    parser.add_argument("--person-id", help="Unique id, e.g. 'ravi'.")
    parser.add_argument("--name", help="Display name.")
    parser.add_argument("--notes", help="Free-text note stored with the record.")
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing enrollment."
    )
    parser.add_argument(
        "--no-gait", action="store_true",
        help="Skip the gait branch (faster; for rotation-on-the-spot clips).",
    )
    parser.add_argument("--list", action="store_true", help="List enrolled people.")
    parser.add_argument("--inspect", metavar="PERSON_ID", help="Show one record.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        return list_people(args)
    if args.inspect:
        return inspect(args)
    if not args.source or not args.person_id:
        print("--source and --person-id are required (or use --list / --inspect).")
        return 2
    return enroll(args)


if __name__ == "__main__":
    raise SystemExit(main())
