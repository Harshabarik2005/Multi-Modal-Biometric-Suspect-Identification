"""Tests for uploads, background jobs, and media handling.

The job runner and media classification are pure logic and always run. The
tests that actually push a video through the models are marked slow.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.jobs import Job, JobRunner, JobStatus, ProgressReporter
from app.api.media import (
    MediaSummary,
    assess_readiness,
    kind_of,
    observations_from_images,
)
from app.core.config import get_settings
from app.core.types import Modality
from app.db.repository import make_engine

REPO = Path(__file__).resolve().parents[2]
ENROLL_CLIP = REPO / "data" / "test_videos" / "enroll_subject_a.mp4"
PROBE_CLIP = REPO / "data" / "test_videos" / "probe_two_subjects.mp4"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
    from app.api.main import create_app

    app = create_app(engine=make_engine("sqlite:///:memory:"))
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client: TestClient, job_id: str, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s")


class TestJobRunner:
    def test_a_job_runs_and_reports_its_result(self) -> None:
        runner = JobRunner()
        job = runner.submit("test", lambda _: {"answer": 42})
        for _ in range(100):
            if runner.get(job.id).status is JobStatus.SUCCEEDED:
                break
            time.sleep(0.05)
        assert runner.get(job.id).result == {"answer": 42}
        runner.shutdown()

    def test_a_failing_job_records_the_error_rather_than_crashing(self) -> None:
        """A failed job must not take the server down; the browser needs telling."""
        runner = JobRunner()

        def explode(_):
            raise ValueError("something went wrong")

        job = runner.submit("test", explode)
        for _ in range(100):
            if runner.get(job.id).status is JobStatus.FAILED:
                break
            time.sleep(0.05)

        finished = runner.get(job.id)
        assert finished.status is JobStatus.FAILED
        assert "something went wrong" in finished.error
        runner.shutdown()

    def test_jobs_report_progress(self) -> None:
        runner = JobRunner()

        def slow(job: Job):
            reporter = ProgressReporter(job, stages=["a", "b", "c"])
            reporter.stage("first")
            reporter.stage("second")
            return "done"

        job = runner.submit("test", slow)
        for _ in range(100):
            if runner.get(job.id).status is JobStatus.SUCCEEDED:
                break
            time.sleep(0.05)
        assert runner.get(job.id).progress == 1.0
        runner.shutdown()

    def test_unknown_job_is_none(self) -> None:
        runner = JobRunner()
        assert runner.get("nope") is None
        runner.shutdown()

    def test_history_is_bounded_as_jobs_keep_arriving(self) -> None:
        """Pruning is opportunistic, at submit time.

        It only evicts *finished* jobs, because dropping a running one would
        lose the handle the browser is polling. So a burst of submissions with
        nothing yet finished can briefly exceed the limit -- the guarantee is
        that history stays bounded as submissions continue, not that it is
        never momentarily over.
        """
        runner = JobRunner(history=3)
        for _ in range(4):
            runner.submit("test", lambda _: 1)
        # Let them finish so there is something eligible to evict.
        time.sleep(0.5)
        for _ in range(6):
            runner.submit("test", lambda _: 1)
            time.sleep(0.05)

        assert len(runner.recent(limit=100)) <= 5
        runner.shutdown()

    def test_pruning_never_evicts_an_unfinished_job(self) -> None:
        """Evicting a running job would lose the handle the browser polls."""
        import threading

        release = threading.Event()
        runner = JobRunner(history=1)
        blocked = runner.submit("test", lambda _: release.wait(timeout=10))
        time.sleep(0.2)

        for _ in range(5):
            runner.submit("test", lambda _: 1)
            time.sleep(0.05)

        assert runner.get(blocked.id) is not None, (
            "the running job must still be retrievable"
        )
        release.set()
        runner.shutdown()


class TestMediaClassification:
    def test_recognises_images_and_videos(self) -> None:
        assert kind_of(Path("a.jpg")) == "image"
        assert kind_of(Path("a.PNG")) == "image"
        assert kind_of(Path("a.mp4")) == "video"
        assert kind_of(Path("a.MOV")) == "video"
        assert kind_of(Path("a.txt")) == "unknown"
        assert kind_of(Path("a.exe")) == "unknown"


class TestReadiness:
    """The guidance that stops someone storing a profile they think is complete."""

    def test_photos_alone_can_never_support_gait(self) -> None:
        summary = MediaSummary(images=6, observations=6, has_motion_source=False)
        checks = {c.modality: c for c in assess_readiness(summary, get_settings())}

        assert checks[Modality.FACE].ready
        assert checks[Modality.REID].ready
        assert not checks[Modality.GAIT].ready
        assert "no gait" in checks[Modality.GAIT].reason.lower()

    def test_video_with_enough_frames_supports_all_three(self) -> None:
        settings = get_settings()
        summary = MediaSummary(
            videos=1,
            observations=settings.gait.min_frames + 10,
            has_motion_source=True,
        )
        checks = {c.modality: c for c in assess_readiness(summary, settings)}
        assert all(check.ready for check in checks.values())

    def test_a_very_short_video_cannot_support_gait(self) -> None:
        settings = get_settings()
        summary = MediaSummary(videos=1, observations=5, has_motion_source=True)
        checks = {c.modality: c for c in assess_readiness(summary, settings)}
        assert not checks[Modality.GAIT].ready
        assert str(settings.gait.min_frames) in checks[Modality.GAIT].reason

    def test_nothing_found_makes_everything_unready(self) -> None:
        checks = assess_readiness(MediaSummary(), get_settings())
        assert not any(check.ready for check in checks)

    def test_every_check_explains_what_is_needed(self) -> None:
        checks = assess_readiness(MediaSummary(), get_settings())
        for check in checks:
            assert len(check.requirement) > 20


class TestUploadValidation:
    def test_rejects_an_unsupported_file_type(self, client) -> None:
        response = client.post(
            "/api/enroll/preview",
            files=[("files", ("notes.txt", b"hello", "text/plain"))],
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_rejects_an_empty_upload(self, client) -> None:
        response = client.post("/api/enroll", data={
            "person_id": "x", "display_name": "X",
        })
        assert response.status_code == 422

    def test_scanning_refuses_stills(self, client) -> None:
        """Scanning needs motion; a photograph has no tracks to follow."""
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", blank)
        assert ok
        response = client.post(
            "/api/scan",
            files=[("files", ("still.jpg", buffer.tobytes(), "image/jpeg"))],
        )
        assert response.status_code == 400
        assert "video" in response.json()["detail"].lower()

    def test_unknown_job_is_404_with_an_explanation(self, client) -> None:
        response = client.get("/api/jobs/deadbeef")
        assert response.status_code == 404
        assert "restart" in response.json()["detail"].lower()


class TestImageObservations:
    def test_an_image_with_no_person_is_rejected_with_a_reason(self) -> None:
        """The caller needs to know *which* file failed and why."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.jpg"
            cv2.imwrite(str(path), np.full((200, 200, 3), 128, dtype=np.uint8))

            observations, summary = observations_from_images([path], get_settings())
            assert observations == []
            assert summary.rejected
            assert summary.rejected[0][0] == "blank.jpg"

    def test_an_undecodable_file_is_reported_not_crashed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"this is not a jpeg")

            observations, summary = observations_from_images([path], get_settings())
            assert observations == []
            assert "decoded" in summary.rejected[0][1]


@pytest.mark.slow
class TestUploadFlow:
    def _skip_without_fixtures(self) -> None:
        if not ENROLL_CLIP.exists() or not PROBE_CLIP.exists():
            pytest.skip("Run scripts/make_face_fixtures.py first.")

    def test_preview_reports_all_three_signals_for_video(self, client) -> None:
        self._skip_without_fixtures()
        with ENROLL_CLIP.open("rb") as handle:
            response = client.post(
                "/api/enroll/preview",
                files=[("files", ("enroll.mp4", handle, "video/mp4"))],
            )
        assert response.status_code == 200
        body = response.json()
        assert body["videos"] == 1
        assert body["observations"] > 0
        assert {entry["modality"] for entry in body["readiness"]} == {
            "face", "gait", "reid"
        }

    def test_enroll_then_scan_finds_the_person(self, client) -> None:
        """The whole point: enrol from one video, find them in another."""
        self._skip_without_fixtures()

        with ENROLL_CLIP.open("rb") as handle:
            started = client.post(
                "/api/enroll",
                data={
                    "person_id": "ravi",
                    "display_name": "Ravi Kumar",
                    "operator": "tester",
                },
                files=[("files", ("enroll.mp4", handle, "video/mp4"))],
            ).json()

        enrolled = wait_for(client, started["id"])
        assert enrolled["status"] == "succeeded", enrolled.get("error")
        assert "face" in enrolled["result"]["stored"]

        assert [p["person_id"] for p in client.get("/api/watchlist").json()] == ["ravi"]

        with PROBE_CLIP.open("rb") as handle:
            started = client.post(
                "/api/scan",
                data={"camera_id": "cam-test"},
                files=[("files", ("probe.mp4", handle, "video/mp4"))],
            ).json()

        scanned = wait_for(client, started["id"])
        assert scanned["status"] == "succeeded", scanned.get("error")

        findings = scanned["result"]["findings"]
        assert findings, "the enrolled person should have been found"
        assert findings[0]["person_id"] == "ravi"
        assert findings[0]["weights"], "a finding must record what drove it"

        # And it lands in the review queue as PENDING, not confirmed.
        decisions = client.get("/api/decisions").json()
        assert len(decisions) == 1
        assert decisions[0]["status"] == "pending"
        assert decisions[0]["is_actionable"] is False
        assert client.get("/api/alerts").json() == [], (
            "an unreviewed scan result must never be actionable"
        )

    def test_scanning_an_empty_watchlist_fails_clearly(self, client) -> None:
        self._skip_without_fixtures()
        with PROBE_CLIP.open("rb") as handle:
            started = client.post(
                "/api/scan", files=[("files", ("probe.mp4", handle, "video/mp4"))]
            ).json()
        finished = wait_for(client, started["id"])
        assert finished["status"] == "failed"
        assert "watchlist is empty" in finished["error"].lower()

    def test_uploaded_files_are_deleted_after_the_job(self, client) -> None:
        """Enrolment footage is personal data; keeping it serves nothing."""
        self._skip_without_fixtures()
        import tempfile

        before = set(Path(tempfile.gettempdir()).glob("frs-upload-*"))
        with ENROLL_CLIP.open("rb") as handle:
            started = client.post(
                "/api/enroll",
                data={"person_id": "temp", "display_name": "Temp"},
                files=[("files", ("enroll.mp4", handle, "video/mp4"))],
            ).json()
        wait_for(client, started["id"])

        after = set(Path(tempfile.gettempdir()).glob("frs-upload-*"))
        assert not (after - before), "upload directory was not cleaned up"
