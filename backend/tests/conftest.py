"""Shared pytest fixtures and the --run-slow opt-in flag."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests that download model weights and process real video.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 480x640 BGR frame of mid-grey."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """A 10-frame video of moving coloured blocks. No people in it.

    Enough to exercise VideoReader; not enough for the detector to find
    anything, which is intentional -- these tests must not need weights.
    """
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    try:
        for i in range(10):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            x = 10 + i * 10
            cv2.rectangle(frame, (x, 40), (x + 20, 80), (0, 200, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def synthetic_people_video() -> Path:
    """Path to the generated smoke-test clip, if it exists."""
    path = REPO_ROOT / "data" / "test_videos" / "synthetic_pan.mp4"
    if not path.exists():
        pytest.skip(
            "Run `python scripts/make_test_video.py` to generate the test clip."
        )
    return path
