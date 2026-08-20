"""Generate face enrollment/probe fixtures for smoke-testing Phase 2.

Builds two clips from `zidane.jpg`, the two-person sample image bundled with
Ultralytics:

* `enroll_subject_a.mp4` -- the right-hand person alone, for enrollment.
* `probe_two_subjects.mp4` -- both people, for matching.

The probe clip is mirrored, dimmed and slightly blurred, so matching is not a
pixel-identical comparison and the embedding has to show some robustness. The
second person is a genuine impostor: they must NOT match the enrolled subject.

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
It proves the machinery: crop -> embed -> store -> load -> cosine -> threshold,
and that a different person scores below a matching one. That is all.

It says NOTHING about real-world accuracy. Both clips derive from a single
photograph, so the "enrollment" covers one pose under one lighting condition,
which is the easiest possible case and nothing like CCTV. Real validation needs
real footage: a person recorded across angles for enrollment, then found in
separate footage shot at a different time.

These are stand-in test images from a computer-vision sample asset, used to
exercise code paths. They are not a watchlist. Real enrollment involves real
people's biometric data and needs a lawful basis and their consent.

    python scripts/make_face_fixtures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_OUT = REPO_ROOT / "data" / "test_videos"

# Region of zidane.jpg holding the right-hand person on their own.
SUBJECT_A_REGION = (700, 0, 1280, 720)  # x1, y1, x2, y2


def load_source() -> np.ndarray:
    import ultralytics

    asset = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    image = cv2.imread(str(asset))
    if image is None:
        raise RuntimeError(f"Could not read the sample image at {asset}")
    print(f"Using bundled asset: {asset}  {image.shape[1]}x{image.shape[0]}")
    return image


def write_clip(
    frames: list[np.ndarray], path: Path, fps: int = 25
) -> None:
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    print(f"  wrote {len(frames)} frames ({width}x{height}) -> {path}")


def drift(image: np.ndarray, count: int, amplitude: int = 14) -> list[np.ndarray]:
    """Small sub-second camera drift, so tracks form across frames.

    Without motion the tracker still works, but a completely static clip is an
    unrealistically easy case even for a smoke test.
    """
    height, width = image.shape[:2]
    pad = amplitude + 2
    canvas = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    frames = []
    for i in range(count):
        t = i / max(1, count - 1)
        dx = int(amplitude * np.sin(2 * np.pi * t))
        dy = int(amplitude * 0.4 * np.cos(2 * np.pi * t))
        x, y = pad + dx, pad + dy
        frames.append(canvas[y : y + height, x : x + width].copy())
    return frames


def degrade(image: np.ndarray) -> np.ndarray:
    """Mirror, dim and soften, so the probe is not the enrollment pixels."""
    out = cv2.flip(image, 1)
    out = cv2.convertScaleAbs(out, alpha=0.82, beta=-12)
    return cv2.GaussianBlur(out, (3, 3), 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args(argv)

    source = load_source()

    print("\nenrollment clip (subject A alone):")
    x1, y1, x2, y2 = SUBJECT_A_REGION
    subject_a = source[y1:y2, x1:x2]
    write_clip(drift(subject_a, args.frames), args.out_dir / "enroll_subject_a.mp4")

    print("\nprobe clip (both subjects, mirrored + dimmed + blurred):")
    write_clip(
        drift(degrade(source), args.frames),
        args.out_dir / "probe_two_subjects.mp4",
    )

    print(
        "\nThese fixtures exercise the matching machinery only. Both derive from\n"
        "one photograph, so they say nothing about accuracy on real footage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
