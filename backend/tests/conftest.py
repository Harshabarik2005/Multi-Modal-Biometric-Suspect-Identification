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


@pytest.fixture(autouse=True)
def template_encryption_key(monkeypatch):
    """Give every test a throwaway encryption key.

    Writing biometric templates unencrypted is refused unless it is explicitly
    opted into (SEC-06), so the tests run the same path production does rather
    than a plaintext one that would never be exercised in a deployment. Tests
    about the plaintext path clear this themselves.
    """
    from cryptography.fernet import Fernet

    monkeypatch.setenv("FRS_TEMPLATE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", raising=False)


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


# ---------------------------------------------------------------------------
# API clients
#
# Every route except /health now requires a signed-in operator (SEC-01), so
# tests need a real session rather than an anonymous one. `api_client` is
# authenticated; `anon_client` deliberately is not, for the tests that assert
# the door is actually locked.
# ---------------------------------------------------------------------------

TEST_OPERATOR = "tester"
TEST_PASSWORD = "test-password-1234"


@pytest.fixture
def api_engine(monkeypatch):
    # A fixed signing key, so tokens are stable within a test run.
    monkeypatch.setenv("FRS_AUTH_SECRET", "test-signing-secret-not-for-real-use")

    from app.db.repository import make_engine

    return make_engine("sqlite:///:memory:")


@pytest.fixture
def api_app(api_engine):
    """One app instance, shared by the signed-in and anonymous clients."""
    from app.api.main import create_app

    return create_app(engine=api_engine)


@pytest.fixture
def anon_client(api_app, api_engine):
    """A client with no credentials.

    A separate TestClient rather than the same one with its header stripped.
    `api_client` used to be built by mutating this object, so a test asking for
    both got one authenticated client under two names -- and a test asserting
    "this route is locked" passed while proving nothing.
    """
    from fastapi.testclient import TestClient

    with TestClient(api_app) as client:
        client.engine = api_engine
        yield client


@pytest.fixture
def api_client(api_app, api_engine):
    """A client signed in as a test operator."""
    from fastapi.testclient import TestClient

    from app.api.auth import create_operator
    from app.db.repository import session_factory

    session = session_factory(api_engine)()
    try:
        create_operator(
            session, TEST_OPERATOR, TEST_PASSWORD, display_name="Tester", is_admin=True
        )
    except ValueError:
        pass  # another client in this test created it already
    finally:
        session.close()

    with TestClient(api_app) as client:
        client.engine = api_engine
        response = client.post(
            "/api/auth/login",
            json={"username": TEST_OPERATOR, "password": TEST_PASSWORD},
        )
        assert response.status_code == 200, response.text
        client.headers.update(
            {"Authorization": f"Bearer {response.json()['token']}"}
        )
        yield client
