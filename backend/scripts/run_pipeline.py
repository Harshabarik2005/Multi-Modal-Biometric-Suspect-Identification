"""CLI for the Phase-1 detection + tracking pipeline.

Examples
--------
    python scripts/run_pipeline.py --source ../data/test_videos/sample.mp4
    python scripts/run_pipeline.py --source 0 --max-frames 200        # webcam
    python scripts/run_pipeline.py --source clip.mp4 --save-annotated --stride 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_pipeline.py` from anywhere by putting backend/ on
# the path before importing the app package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.pipeline import DetectionTrackingPipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track people in a video (Faceless FRS, Phase 1).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path, stream URL, or camera index (e.g. 0).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (defaults to backend/config.yaml).",
    )
    parser.add_argument(
        "--model", default=None, help="Override detection model, e.g. yolov8s.pt."
    )
    parser.add_argument(
        "--device", default=None, choices=["auto", "cuda", "cpu"],
        help="Override compute device.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override detection confidence threshold.",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Process every Nth frame (default 1).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many processed frames.",
    )
    parser.add_argument(
        "--save-annotated", action="store_true",
        help="Write an annotated video with boxes and track IDs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    settings = get_settings(args.config)
    # CLI flags win over both YAML and env vars.
    if args.model is not None:
        settings.detection.model = args.model
    if args.device is not None:
        settings.device = args.device
    if args.conf is not None:
        settings.detection.conf_threshold = args.conf
    if args.stride is not None:
        settings.video.frame_stride = args.stride
    if args.max_frames is not None:
        settings.video.max_frames = args.max_frames
    if args.save_annotated:
        settings.video.save_annotated = True

    setup_logging(settings.logging.level)

    # A bare integer means a camera index, not a filename.
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    pipeline = DetectionTrackingPipeline(settings)
    report = pipeline.run(source)

    print()
    print("=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    for line in report.summary_lines():
        print(line)
    print("=" * 60)

    if report.frames_processed == 0:
        print("\nNo frames were processed -- check the source path or camera index.")
        return 1
    if report.unique_track_ids == 0:
        print(
            "\nFrames processed but no tracks confirmed. Try lowering --conf, "
            "reducing tracking.n_init, or check that the footage contains people."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
