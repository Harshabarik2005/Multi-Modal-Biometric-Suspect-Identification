"""Generate a small test video containing real people, for smoke-testing.

Phase 1 needs *some* footage with people in it before real CCTV clips exist.
This builds one from `bus.jpg`, the sample image that ships with Ultralytics
(it contains several pedestrians): the image is tiled into a wide canvas and a
frame-sized window pans across it. People therefore move between frames, which
is exactly what the tracker needs in order to form tracks.

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


def build_canvas(image: np.ndarray, out_h: int) -> np.ndarray:
    """Build a wide canvas by tiling the source image horizontally.

    bus.jpg is portrait and the output frame is landscape, so any crop window
    wide enough to fill the frame shows only a horizontal band -- people get
    cut off at the knees. Later phases need whole bodies (gait needs a full
    silhouette, re-ID a full body crop), so instead of cropping into the image
    we scale it to fit the frame height and repeat it sideways, alternating
    mirrored copies so the seams are less abrupt. Panning across that gives
    full-height people who enter and leave the frame.
    """
    src_h, src_w = image.shape[:2]
    # Slightly taller than the frame, leaving a little vertical room to pan.
    scale = (out_h * 1.12) / src_h
    tile = cv2.resize(
        image, (int(src_w * scale), int(src_h * scale)), interpolation=cv2.INTER_CUBIC
    )
    mirrored = cv2.flip(tile, 1)
    return np.hstack([tile, mirrored, tile, mirrored])


def build_video(
    image: np.ndarray, out_path: Path, frames: int, fps: int, size: tuple[int, int]
) -> None:
    """Pan a crop window across the tiled canvas and write it out as a video."""
    out_w, out_h = size
    canvas = build_canvas(image, out_h)
    canvas_h, canvas_w = canvas.shape[:2]

    if canvas_w < out_w:
        raise RuntimeError(
            f"Canvas ({canvas_w}px) is narrower than the requested frame ({out_w}px)."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    max_x = canvas_w - out_w
    max_y = max(0, canvas_h - out_h)

    try:
        for i in range(frames):
            t = i / max(1, frames - 1)
            # Steady horizontal pan, with a gentle vertical drift so the
            # tracker sees motion on both axes.
            x = int(max_x * t)
            y = int(max_y * (0.5 + 0.5 * math.sin(2 * math.pi * t)))
            window = canvas[y : y + out_h, x : x + out_w]
            writer.write(window)
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
