"""Generate a small test video containing real people, for smoke-testing.

Phase 1 needs *some* footage with people in it before real CCTV clips exist.
This builds one from `bus.jpg`, the sample image that ships with Ultralytics
(it contains several pedestrians), by panning and zooming a crop window across
it. People therefore move between frames, which is exactly what the tracker
needs in order to form tracks.

This is a smoke-test fixture, not a benchmark. Replace it with real enrollment
and CCTV footage in `data/test_videos/` as soon as you have any.

    python scripts/make_test_video.py
    python scripts/make_test_video.py --frames 200 --out ../data/test_videos/pan.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

ASSET_URL = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"


def load_source_image() -> np.ndarray:
    """Find bus.jpg in the installed ultralytics package, else download it."""
    try:
        import ultralytics

        asset = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if asset.is_file():
            image = cv2.imread(str(asset))
            if image is not None:
                print(f"Using bundled asset: {asset}")
                return image
    except ImportError:
        pass

    print(f"Downloading sample image from {ASSET_URL}")
    import urllib.request

    with urllib.request.urlopen(ASSET_URL, timeout=30) as response:
        buffer = np.frombuffer(response.read(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode the downloaded sample image.")
    return image


def build_video(
    image: np.ndarray, out_path: Path, frames: int, fps: int, size: tuple[int, int]
) -> None:
    """Pan/zoom a crop window over `image` and write it out as a video."""
    out_w, out_h = size
    src_h, src_w = image.shape[:2]

    # Upscale so the crop window has room to travel without leaving the image.
    scale = max(2.0, (out_w * 1.6) / src_w, (out_h * 1.6) / src_h)
    big = cv2.resize(
        image, (int(src_w * scale), int(src_h * scale)), interpolation=cv2.INTER_CUBIC
    )
    big_h, big_w = big.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    try:
        for i in range(frames):
            t = i / max(1, frames - 1)

            # Crop window shrinks slightly over the clip (slow zoom in) and
            # slides horizontally with a gentle vertical bob.
            zoom = 1.0 - 0.25 * t
            crop_w = min(big_w, int(out_w * 1.5 * zoom))
            crop_h = min(big_h, int(out_h * 1.5 * zoom))

            max_x = max(0, big_w - crop_w)
            max_y = max(0, big_h - crop_h)
            x = int(max_x * t)
            y = int(max_y * (0.5 + 0.5 * math.sin(2 * math.pi * t)))

            window = big[y : y + crop_h, x : x + crop_w]
            writer.write(cv2.resize(window, (out_w, out_h), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()

    print(f"Wrote {frames} frames ({out_w}x{out_h} @ {fps} fps) -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "test_videos" / "synthetic_pan.mp4",
        help="Output video path.",
    )
    parser.add_argument("--frames", type=int, default=120, help="Number of frames.")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args(argv)

    image = load_source_image()
    build_video(image, args.out, args.frames, args.fps, (args.width, args.height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
