"""Match tracked people in a video against the enrolled watchlist (Phase 2).

Runs the Phase-1 pipeline, buffers each track's crops, embeds their faces, and
compares them against every enrolled person by cosine similarity.

Two things this deliberately does NOT do, both from section 8 of the build plan:

* It takes no action on a match. It prints candidates for a human to confirm.
* It hides nothing. Every match above the threshold is reported with the
  similarity that produced it, and near-misses can be shown with --show-all,
  so a reviewer can see how close the call was.

    python scripts/match.py --source ../data/test_videos/clip.mp4
    python scripts/match.py --source 0 --threshold 0.45 --show-all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.core.track_buffer import TrackBufferStore  # noqa: E402
from app.core.types import Modality  # noqa: E402
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.embeddings.gait import GaitEmbedder  # noqa: E402
from app.matching.gallery import GalleryStore, MatchCandidate  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402

logger = get_logger(__name__)


@dataclass
class TrackVerdict:
    """The best result seen for one track over the whole clip."""

    track_id: int
    best_person_id: str | None = None
    best_person_name: str | None = None
    best_similarity: float = -1.0
    best_frame: int = -1
    probe_quality: float = 0.0
    observations: int = 0
    times_matched: int = 0
    # Which modality produced the best score. Surfaced so the report never
    # implies a gait match carries the same weight as a face match.
    modality: str = "-"
    # Every above-threshold hit, for the audit log section 8 requires.
    hits: list[tuple[int, str, float]] = field(default_factory=list)


def _report(
    verdicts: dict[int, TrackVerdict],
    thresholds: dict[str, float],
    show_all: bool,
) -> int:
    print()
    print("=" * 74)
    print("MATCH REPORT")
    print("=" * 74)

    # Each modality carries its own threshold: a gait score of 0.85 is a far
    # weaker claim than a face score of 0.85, and one shared number would
    # quietly equate them.
    matched = {
        tid: v
        for tid, v in verdicts.items()
        if v.best_person_id
        and v.best_similarity >= thresholds.get(v.modality, 1.1)
    }
    unmatched = {tid: v for tid, v in verdicts.items() if tid not in matched}

    if matched:
        print(f"\nCANDIDATE MATCHES ({len(matched)}) -- require human confirmation\n")
        header = (
            f"  {'track':>5}  {'person':<20} {'name':<20} {'via':>5} "
            f"{'sim':>6}  {'qual':>5}  {'frame':>6}  {'hits':>4}"
        )
        print(header)
        for verdict in sorted(
            matched.values(), key=lambda v: v.best_similarity, reverse=True
        ):
            print(
                f"  {verdict.track_id:>5}  {verdict.best_person_id:<20} "
                f"{(verdict.best_person_name or ''):<20} {verdict.modality:>5} "
                f"{verdict.best_similarity:>6.3f}  {verdict.probe_quality:>5.2f}  "
                f"{verdict.best_frame:>6}  {verdict.times_matched:>4}"
            )
    else:
        print("\nNo track matched anyone on the watchlist.")

    if show_all and unmatched:
        print(f"\nBELOW THRESHOLD ({len(unmatched)}) -- shown for calibration\n")
        for verdict in sorted(
            unmatched.values(), key=lambda v: v.best_similarity, reverse=True
        ):
            if verdict.best_person_id is None:
                detail = "no modality produced an embedding"
            else:
                detail = (
                    f"closest {verdict.best_person_id} at {verdict.best_similarity:+.3f}"
                )
            print(
                f"  track {verdict.track_id:>3}  {verdict.observations:>4} obs  {detail}"
            )

    print()
    print("=" * 74)
    print(
        "No action has been taken. Every candidate above requires a human to\n"
        "confirm before anything follows from it (build plan, section 8)."
    )
    print("=" * 74)
    return len(matched)


def run(args: argparse.Namespace) -> int:
    settings = get_settings(args.config)
    if args.threshold is not None:
        settings.matching.face_threshold = args.threshold
    if args.max_frames is not None:
        settings.video.max_frames = args.max_frames
    setup_logging(settings.logging.level)

    gallery = GalleryStore(settings).load_gallery()
    if len(gallery) == 0:
        print(
            "The watchlist is empty -- nothing to match against.\n"
            "Enroll someone first:  python scripts/enroll.py --source <video> "
            "--person-id <id>"
        )
        return 1

    min_obs = settings.matching.min_track_observations
    rematch_every = settings.matching.rematch_every

    pipeline = DetectionTrackingPipeline(settings)
    store = TrackBufferStore(settings)
    face_embedder = FaceEmbedder(settings)
    gait_embedder = None if args.no_gait else GaitEmbedder(settings)
    verdicts: dict[int, TrackVerdict] = {}

    print(
        f"Watchlist: {len(gallery)} enrolled. Thresholds: face "
        f"{settings.matching.face_threshold:.2f}, gait "
        f"{settings.matching.gait_threshold:.2f}"
    )
    if gait_embedder is None:
        print("Gait fallback disabled (--no-gait).")
    print()

    for result, frame in pipeline.stream(args.source):
        updated = store.update(result, frame)

        for track_id in updated:
            buffer = store.get(track_id)
            if buffer is None or len(buffer) < min_obs:
                continue
            # Re-embedding every frame is wasteful and adds nothing; a track's
            # appearance changes slowly.
            if (
                buffer.last_matched_frame is not None
                and result.frame_index - buffer.last_matched_frame < rematch_every
            ):
                continue
            buffer.last_matched_frame = result.frame_index

            observations = list(buffer)
            verdict = verdicts.setdefault(track_id, TrackVerdict(track_id=track_id))
            verdict.observations = len(buffer)

            probe = face_embedder.embed(observations)
            modality = Modality.FACE
            if not probe.has_signal and gait_embedder is not None:
                # Face failed: turned away, too distant, masked. This is the
                # exact case the project exists for, so fall back to gait
                # rather than abandoning the track.
                probe = gait_embedder.embed(observations)
                modality = Modality.GAIT
            if not probe.has_signal:
                continue

            candidates: list[MatchCandidate] = gallery.rank_single(modality, probe)
            if not candidates:
                continue
            best = candidates[0]

            if best.fused_similarity > verdict.best_similarity:
                verdict.best_similarity = best.fused_similarity
                verdict.best_person_id = best.person.person_id
                verdict.best_person_name = best.person.display_name
                verdict.best_frame = result.frame_index
                verdict.probe_quality = probe.quality
                verdict.modality = modality.value

            active = (
                settings.matching.face_threshold
                if modality is Modality.FACE
                else settings.matching.gait_threshold
            )
            if best.fused_similarity >= active:
                verdict.times_matched += 1
                verdict.hits.append(
                    (
                        result.frame_index,
                        best.person.person_id,
                        best.fused_similarity,
                    )
                )
                # Audit trail: every match decision is logged with what drove it.
                logger.info(
                    "MATCH frame=%d track=%d person=%s %s quality=%.3f",
                    result.frame_index,
                    track_id,
                    best.person.person_id,
                    best.explain(),
                    probe.quality,
                )

    matched = _report(
        verdicts,
        {
            "face": settings.matching.face_threshold,
            "gait": settings.matching.gait_threshold,
        },
        args.show_all,
    )
    return 0 if matched or not args.require_match else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine similarity above which a track counts as a candidate.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-gait", action="store_true",
        help="Face only; skip the gait fallback.",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Also show below-threshold tracks, for threshold calibration.",
    )
    parser.add_argument(
        "--require-match", action="store_true",
        help="Exit non-zero when nothing matched (useful in scripts).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source.isdigit():
        args.source = int(args.source)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
