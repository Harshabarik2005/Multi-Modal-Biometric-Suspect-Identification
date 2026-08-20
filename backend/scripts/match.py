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
from app.embeddings.reid import ReIDEmbedder  # noqa: E402
from app.fusion.baseline import build_strategy  # noqa: E402
from app.fusion.calibration import default_calibrations  # noqa: E402
from app.matching.gallery import GalleryStore, MatchCandidate  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402

logger = get_logger(__name__)


def reid_trusts(gallery, settings) -> dict[Modality, float]:
    """How far each modality's stored reference can still be trusted.

    Only re-ID decays. Face and gait describe the person; re-ID largely
    describes their clothing, so an old reference is far weaker evidence. The
    oldest enrolment in the gallery sets the decay conservatively, since one
    trust value covers the whole ranking pass.
    """
    from datetime import datetime, timezone

    from app.embeddings.reid import trust_at

    half_life = settings.reid.trust_half_life_days
    if half_life <= 0:
        return {}

    oldest_days = 0.0
    now = datetime.now(timezone.utc)
    for person in gallery:
        if not person.enrolled_at:
            continue
        try:
            enrolled = datetime.fromisoformat(person.enrolled_at)
        except ValueError:
            continue
        if enrolled.tzinfo is None:
            enrolled = enrolled.replace(tzinfo=timezone.utc)
        oldest_days = max(oldest_days, (now - enrolled).total_seconds() / 86400.0)

    return {Modality.REID: trust_at(oldest_days, half_life)}


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
    # Which modalities contributed, as initials (e.g. "f+r").
    modality: str = "-"
    # Full per-modality breakdown of the winning score, for the audit log.
    breakdown: str = ""
    # Every above-threshold hit, for the audit log section 8 requires.
    hits: list[tuple[int, str, float]] = field(default_factory=list)


def _report(
    verdicts: dict[int, TrackVerdict],
    threshold: float,
    show_all: bool,
) -> int:
    print()
    print("=" * 74)
    print("MATCH REPORT")
    print("=" * 74)

    # One threshold is correct here, unlike before fusion: scores are on the
    # calibrated [0, 1] scale, where 0.5 means "halfway between a stranger and
    # a genuine match" for every modality alike.
    matched = {
        tid: v
        for tid, v in verdicts.items()
        if v.best_person_id and v.best_similarity >= threshold
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
            if verdict.breakdown:
                print(f"           {verdict.breakdown}")
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
        settings.fusion.threshold = args.threshold
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
    reid_embedder = None if args.no_reid else ReIDEmbedder(settings)
    strategy = build_strategy(
        args.strategy or settings.fusion.strategy, default_calibrations(settings)
    )
    verdicts: dict[int, TrackVerdict] = {}

    print(f"Watchlist: {len(gallery)} enrolled.")
    print(
        f"Fusion: {strategy.name}, threshold {settings.fusion.threshold:.2f} "
        "on the calibrated 0-1 scale."
    )
    disabled = [
        name
        for name, embedder in (("gait", gait_embedder), ("reid", reid_embedder))
        if embedder is None
    ]
    if disabled:
        print("Disabled: " + ", ".join(disabled))
    gait_refs = sum(
        1 for p in gallery if p.embedding(Modality.GAIT) is not None
    )
    need = settings.fusion.gait_min_references_for_centring
    if gait_refs and gait_refs < need:
        print(
            f"Gait comparison is OFF: {gait_refs} gait reference(s), {need} "
            "needed to estimate the population mean. Uncentred gait scores "
            "above 0.93 for everyone, so it is refused rather than reported."
        )
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

            # Every modality that has something to say contributes at once.
            # This replaces the phase-4 fallback ladder: two modalities each
            # moderately agreeing is stronger evidence than either alone, and
            # a ladder cannot express that.
            probes = {}
            probe_face = face_embedder.embed(observations)
            if probe_face.has_signal:
                probes[Modality.FACE] = probe_face
            if gait_embedder is not None:
                probe_gait = gait_embedder.embed(observations)
                if probe_gait.has_signal:
                    probes[Modality.GAIT] = probe_gait
            if reid_embedder is not None:
                probe_reid = reid_embedder.embed(observations)
                if probe_reid.has_signal:
                    probes[Modality.REID] = probe_reid

            if not probes:
                continue

            candidates: list[MatchCandidate] = gallery.rank(
                probes,
                strategy=strategy,
                trusts=reid_trusts(gallery, settings),
                gait_min_references=settings.fusion.gait_min_references_for_centring,
            )
            if not candidates:
                continue
            best = candidates[0]

            if best.fused_similarity > verdict.best_similarity:
                verdict.best_similarity = best.fused_similarity
                verdict.best_person_id = best.person.person_id
                verdict.best_person_name = best.person.display_name
                verdict.best_frame = result.frame_index
                verdict.probe_quality = max(p.quality for p in probes.values())
                verdict.modality = "+".join(
                    sorted(m.value[:1] for m in best.weights if best.weights[m] > 0)
                ) or "-"
                verdict.breakdown = (
                    best.fusion.explain() if best.fusion else best.explain()
                )

            if best.fused_similarity >= settings.fusion.threshold:
                verdict.times_matched += 1
                verdict.hits.append(
                    (
                        result.frame_index,
                        best.person.person_id,
                        best.fused_similarity,
                    )
                )
                # Audit trail: every match decision is logged with what drove it.
                # Audit trail (build plan, section 8): every match decision
                # is logged with the per-modality weights that produced it, so
                # a reviewer can later see what the system actually relied on.
                logger.info(
                    "MATCH frame=%d track=%d person=%s fused=%s | raw %s",
                    result.frame_index,
                    track_id,
                    best.person.person_id,
                    best.fusion.explain() if best.fusion else "n/a",
                    best.explain(),
                )

    matched = _report(verdicts, settings.fusion.threshold, args.show_all)
    return 0 if matched or not args.require_match else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fused score (0-1, calibrated) above which a track is a candidate.",
    )
    parser.add_argument(
        "--strategy", default=None,
        choices=["single_best", "average", "quality_weighted"],
        help="Fusion strategy. Defaults to fusion.strategy in config.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-gait", action="store_true", help="Skip the gait fallback."
    )
    parser.add_argument(
        "--no-reid", action="store_true", help="Skip the re-ID fallback."
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
