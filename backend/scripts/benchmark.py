"""Per-stage throughput benchmark (Phase 11, edge deployment).

    python scripts/benchmark.py
    python scripts/benchmark.py --frames 60 --device cpu

Times each stage separately, because "the pipeline runs at N fps" is not
actionable. What you need to know before targeting constrained hardware is
which stage is actually eating the budget -- and in this system it is usually
not detection.

On Jetson-class hardware
------------------------
This measures *this* machine, and cannot predict a Jetson. What it does give
you is the shape of the problem: the relative cost of the stages carries over
even when absolute numbers do not, and any stage that is already slow on a
desktop GPU will be far worse on an embedded one.

The obvious levers, in the order worth trying:

1. **Raise `video.frame_stride`.** Processing every second frame halves the
   whole pipeline's cost. Tracking survives it; gait cadence detection is the
   thing to watch, since it needs enough temporal resolution to see a step.
2. **Run the embedding branches less often**, not faster. They already run
   once per `matching.rematch_every` frames rather than per frame; raising it
   costs little because a person's appearance changes slowly.
3. **Drop to smaller backbones**: `yolov8n` is already the smallest, but
   `osnet_x0_25` is roughly a quarter of `osnet_x1_0`, and `buffalo_s`
   replaces `buffalo_l` for face.
4. **Skip gait on constrained hardware.** It needs a second segmentation pass
   per crop and is the weakest of the three signals, so it is the first thing
   to cut when the budget is tight.

TensorRT export is the standard next step for Jetson, and is out of scope
here: it needs the target device to build an engine, so it cannot be done
meaningfully from a desktop.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.track_buffer import TrackBufferStore  # noqa: E402
from app.core.video import VideoReader  # noqa: E402

REPO = BACKEND_ROOT.parent


class Timer:
    """Accumulates wall time per named stage."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, frames: int) -> list[str]:
        total = sum(self.totals.values())
        lines = [
            f"{'stage':<26} {'total s':>9} {'ms/call':>9} {'calls':>7} {'share':>7}",
            "-" * 62,
        ]
        for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            calls = self.counts[name]
            lines.append(
                f"{name:<26} {seconds:>9.2f} {seconds / calls * 1000:>9.1f} "
                f"{calls:>7} {seconds / total * 100 if total else 0:>6.1f}%"
            )
        lines.append("-" * 62)
        lines.append(f"{'TOTAL':<26} {total:>9.2f}")
        if frames and total:
            lines.append(f"\nend-to-end: {frames / total:.1f} fps over {frames} frames")
        return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", default=str(REPO / "data" / "test_videos" / "synthetic_pan.mp4")
    )
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-gait", action="store_true")
    args = parser.parse_args(argv)

    if not Path(args.source).exists():
        print(f"Missing {args.source}. Run scripts/make_test_video.py first.")
        return 1

    settings = get_settings()
    if args.device:
        settings.device = args.device
    setup_logging("WARNING")

    device = settings.resolve_device()
    print(f"device: {device}   source: {Path(args.source).name}\n")

    timer = Timer()

    with timer.stage("model load: detector"):
        from app.detection.yolo_detector import YOLOPersonDetector

        detector = YOLOPersonDetector(settings, device=device)

    with timer.stage("model load: tracker"):
        from app.tracking.deepsort_tracker import DeepSortTracker

        tracker = DeepSortTracker(settings, device=device)

    with timer.stage("model load: face"):
        from app.embeddings.face import FaceEmbedder

        face = FaceEmbedder(settings)

    with timer.stage("model load: reid"):
        from app.embeddings.reid import ReIDEmbedder

        reid = ReIDEmbedder(settings, device=device)

    gait = None
    if not args.skip_gait:
        with timer.stage("model load: gait seg"):
            from app.embeddings.gait import GaitEmbedder

            gait = GaitEmbedder(settings)
            _ = gait.extractor  # force the lazy segmentation load

    store = TrackBufferStore(settings)
    frames = 0

    with VideoReader(args.source, max_frames=args.frames) as reader:
        for frame_index, timestamp, frame in reader:
            frames += 1
            with timer.stage("detect"):
                detections = detector.detect(frame)
            with timer.stage("track"):
                tracks = tracker.update(detections, frame)

            from app.core.types import FrameResult

            with timer.stage("buffer crops"):
                store.update(
                    FrameResult(frame_index, timestamp, detections, tracks), frame
                )

    # Embed once per track, which is how matching actually uses these -- timing
    # them per frame would overstate their cost several times over.
    cap = settings.matching.max_observations_to_embed
    for buffer in store:
        # Mirror the live matching path, which embeds the best N crops rather
        # than every buffered one.
        ordered = list(buffer)
        sampled = buffer.best(cap) if cap else ordered
        if len(sampled) < 3:
            continue
        with timer.stage("embed: face (per track)"):
            face.embed(sampled)
        with timer.stage("embed: reid (per track)"):
            reid.embed(sampled)
        if gait is not None:
            # Gait needs the ordered sequence, not the largest-N sample.
            with timer.stage("embed: gait (per track)"):
                gait.embed(ordered)

    print("\n".join(timer.report(frames)))

    per_frame = sum(
        seconds
        for name, seconds in timer.totals.items()
        if not name.startswith("model load")
        and "per track" not in name
    )
    if per_frame:
        print(
            f"\nDetection + tracking alone: {frames / per_frame:.1f} fps. The "
            "embedding\nbranches run per track rather than per frame, so their "
            "cost scales with\nhow many people are in view, not with frame rate."
        )
    print(
        "\nThis machine only. Relative stage costs carry over to embedded\n"
        "hardware; absolute numbers do not. See the module docstring for the\n"
        "levers worth pulling, in order."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
