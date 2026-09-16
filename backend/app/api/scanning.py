"""Scanning uploaded footage against the watchlist.

The same logic as `scripts/match.py`, factored so the API can call it and
report progress while it runs. The script remains the reference implementation
for anyone working from a terminal.

One thing worth restating here, because this is the path a web UI will use and
a web UI is where the temptation to skip it lives: every hit is recorded as a
**pending** decision. Nothing in this module confirms a match. That still
requires a named person in the review console.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.evidence import encode_crop
from app.core.logging import get_logger
from app.core.track_buffer import TrackBufferStore
from app.core.types import Modality
from app.core.video import VideoReader
from app.fusion.baseline import build_strategy
from app.fusion.calibration import default_calibrations
from app.pipeline import DetectionTrackingPipeline

logger = get_logger(__name__)


def scan_video(
    path: Path,
    gallery,
    repo,
    settings: Settings,
    camera_id: str = "",
    job=None,
    video_index: int = 0,
    video_count: int = 1,
) -> list[dict]:
    """Find watchlist people in one video.

    Returns one entry per track that matched, with the fusion breakdown behind
    it. `repo` may be None to scan without writing decisions -- useful for
    trying a threshold before committing anything to the audit trail.
    """
    from app.embeddings.face import FaceEmbedder
    from app.embeddings.gait import GaitEmbedder
    from app.embeddings.reid import ReIDEmbedder

    if job is not None:
        job.message = f"loading models ({path.name})"

    pipeline = DetectionTrackingPipeline(settings)
    store = TrackBufferStore(settings)
    face_embedder = FaceEmbedder(settings)
    gait_embedder = GaitEmbedder(settings)
    reid_embedder = ReIDEmbedder(settings)
    strategy = build_strategy(settings.fusion.strategy, default_calibrations(settings))

    # Total frames up front so progress means something. Some containers do not
    # report it, in which case progress stays coarse rather than lying.
    try:
        with VideoReader(path) as probe:
            total_frames = probe.meta.frame_count
    except Exception:  # noqa: BLE001
        total_frames = 0

    min_obs = settings.matching.min_track_observations
    rematch_every = settings.matching.rematch_every
    cap = settings.matching.max_observations_to_embed

    best_per_track: dict[int, dict] = {}
    processed = 0

    for result, frame in pipeline.stream(path):
        processed += 1
        store.update(result, frame)

        if job is not None and processed % 10 == 0:
            share = (processed / total_frames) if total_frames else 0.0
            job.progress = min(
                0.98, (video_index + min(share, 1.0)) / max(1, video_count)
            )
            job.message = (
                f"scanning {path.name}: frame {processed}"
                + (f" of {total_frames}" if total_frames else "")
            )

        for track in result.tracks:
            buffer = store.get(track.track_id)
            if buffer is None or len(buffer) < min_obs:
                continue
            if (
                buffer.last_matched_frame is not None
                and result.frame_index - buffer.last_matched_frame < rematch_every
            ):
                continue
            buffer.last_matched_frame = result.frame_index

            # Face and re-ID take the largest crops; gait needs the ordered
            # sequence, because sorting by size destroys its cadence signal,
            # and its own paced ring, because a stride outlasts the main one.
            ordered = list(buffer.gait_observations)
            sampled = buffer.best(cap) if cap else list(buffer)

            probes = {}
            # What each branch said when it had nothing, kept with whichever
            # re-match turns out to be this track's best.
            unused: dict[str, str] = {}
            face = face_embedder.embed(sampled)
            if face.has_signal:
                probes[Modality.FACE] = face
            else:
                unused[Modality.FACE.value] = face.reason or "no signal"
            gait = gait_embedder.embed(ordered)
            if gait.has_signal:
                probes[Modality.GAIT] = gait
            else:
                unused[Modality.GAIT.value] = gait.reason or "no signal"
            reid = reid_embedder.embed(sampled)
            if reid.has_signal:
                probes[Modality.REID] = reid
            else:
                unused[Modality.REID.value] = reid.reason or "no signal"

            if not probes:
                continue

            candidates = gallery.rank(
                probes,
                strategy=strategy,
                gait_min_references=settings.fusion.gait_min_references_for_centring,
                reid_half_life_days=settings.reid.trust_half_life_days,
            )
            if not candidates:
                continue

            best = candidates[0]
            if not best.weights:
                # Nothing was allowed to vote, so every candidate ties at 0.0
                # and "the best" would just be the gallery's first entry.
                continue
            if best.fused_similarity < settings.fusion.threshold:
                continue

            previous = best_per_track.get(track.track_id)
            if previous and previous["score"] >= best.fused_similarity:
                continue

            best_per_track[track.track_id] = {
                "track_id": track.track_id,
                "person_id": best.person.person_id,
                "display_name": best.person.display_name,
                "score": round(best.fused_similarity, 4),
                "frame_index": result.frame_index,
                "timestamp_s": round(result.timestamp_s, 2),
                "video": path.name,
                "weights": {m.value: round(w, 3) for m, w in best.weights.items()},
                "calibrated": (
                    {m.value: round(c, 3) for m, c in best.fusion.calibrated.items()}
                    if best.fusion
                    else {}
                ),
                # The rule that actually ran, not the one that was asked for.
                # quality_weighted falls back to a plain average when every
                # modality scored zero quality, and recording the outer name
                # would put a rule in the audit trail that was never applied
                # (LOG-12).
                "strategy": (
                    best.fusion.strategy if best.fusion else strategy.name
                ),
                # Modalities that had a signal but could not be compared --
                # almost always a reference enrolled with a different model
                # (DES-01). Without this the reviewer sees a face-only match
                # and has no way to know appearance was dropped rather than
                # simply absent.
                # The crop this match was made on. A reviewer confirming an
                # identification needs to see the person, not just the score
                # (DES-02). Captured here because this is the frame the match
                # was actually scored on; by the time the findings are written
                # the frame is long gone.
                "evidence_jpeg": encode_crop(track.crop(frame)),
                "not_compared": {
                    m.value: score.incomparable_reason
                    for m, score in best.scores.items()
                    # Mismatches only; routine refusals are under "unused".
                    if score.model_mismatch
                },
                # Every signal that did not count toward this score, and why:
                # the branch saw nothing, it could not be compared, or it was
                # withheld from voting. Without this a card showing face and
                # appearance says nothing about gait, and all three causes
                # look identical -- absent.
                "unused": {
                    **unused,
                    **{m.value: why for m, why in best.not_counted().items()},
                },
            }

    # Record once per track, using its best moment, rather than once per
    # re-match. Twenty pending decisions for one person walking past is noise
    # that makes the review queue useless.
    findings = []
    for finding in sorted(
        best_per_track.values(), key=lambda f: f["score"], reverse=True
    ):
        if repo is not None:
            from app.core.types import Modality as M

            decision = repo.record_match(
                finding["person_id"],
                track_id=finding["track_id"],
                score=finding["score"],
                strategy=finding["strategy"],
                weights={M(k): v for k, v in finding["weights"].items()},
                calibrated={M(k): v for k, v in finding["calibrated"].items()},
                camera_id=camera_id,
                frame_index=finding["frame_index"],
                evidence_jpeg=finding.get("evidence_jpeg"),
                unused={M(k): v for k, v in finding["unused"].items()},
            )
            finding["decision_id"] = decision.id
            finding["status"] = "pending"
        # Stored, not returned. The image is fetched from its own authenticated
        # route when a reviewer opens the decision, so it is not embedded in
        # every scan response and every job record that holds one.
        finding.pop("evidence_jpeg", None)
        findings.append(finding)

    logger.info(
        "Scanned %s: %d frames, %d candidate(s)", path.name, processed, len(findings)
    )
    return findings
