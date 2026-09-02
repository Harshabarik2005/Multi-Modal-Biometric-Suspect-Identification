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
def client(api_client):
    """Signed in. Auth is required on every route except /health (SEC-01)."""
    return api_client


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
        frames = settings.gait.min_frames + 10
        summary = MediaSummary(
            videos=1,
            observations=frames,
            video_observations=frames,
            longest_video_run=frames,
            has_motion_source=True,
        )
        checks = {c.modality: c for c in assess_readiness(summary, settings)}
        assert all(check.ready for check in checks.values())

    def test_a_very_short_video_cannot_support_gait(self) -> None:
        settings = get_settings()
        summary = MediaSummary(
            videos=1,
            observations=5,
            video_observations=5,
            longest_video_run=5,
            has_motion_source=True,
        )
        checks = {c.modality: c for c in assess_readiness(summary, settings)}
        assert not checks[Modality.GAIT].ready
        assert str(settings.gait.min_frames) in checks[Modality.GAIT].reason

    def test_photographs_do_not_count_toward_gait(self) -> None:
        """Regression, LOG-04.

        Gait was gated on total observations, so 20 photographs plus a 5-frame
        clip reported "gait ready: 25 continuous frames available" -- the exact
        failure this readiness check exists to prevent, in the check built to
        prevent it.
        """
        settings = get_settings()
        summary = MediaSummary(
            videos=1,
            images=20,
            observations=25,
            video_observations=5,
            longest_video_run=5,
            has_motion_source=True,
        )
        checks = {c.modality: c for c in assess_readiness(summary, settings)}

        assert not checks[Modality.GAIT].ready
        assert "photographs do not count" in checks[Modality.GAIT].reason
        # Face and appearance legitimately use the photos.
        assert checks[Modality.FACE].ready
        assert checks[Modality.REID].ready

    def test_gait_is_judged_on_one_clip_not_the_total(self) -> None:
        """Two short clips do not add up to one long walk."""
        settings = get_settings()
        half = settings.gait.min_frames // 2
        summary = MediaSummary(
            videos=2,
            observations=half * 2,
            video_observations=half * 2,
            longest_video_run=half,  # neither clip alone is long enough
            has_motion_source=True,
        )
        checks = {c.modality: c for c in assess_readiness(summary, settings)}
        assert not checks[Modality.GAIT].ready
        assert "longest single clip" in checks[Modality.GAIT].reason

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


class TestMultipleVideos:
    """Regression, LOG-05: separate recordings are not one continuous walk."""

    def _clip(self, directory: Path, name: str, frames: int = 4):
        import cv2
        import numpy as np

        path = directory / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
        )
        for index in range(frames):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + index, 20), (60 + index, 100), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()
        return path

    def test_frame_indices_stay_monotonic_across_videos(self, tmp_path) -> None:
        """Each video's indices start at 0, so raw concatenation gave
        [0,1,2,3,0,1,2,3] -- not monotonic, despite the docstring saying it was.
        """
        from unittest.mock import patch

        from app.api.media import MediaSummary, observations_from_uploads
        from app.core.types import TrackObservation

        import numpy as np

        def fake_video(path, settings, pipeline=None):
            observations = [
                TrackObservation(i, float(i), np.zeros((40, 20, 3), np.uint8), 40.0, 0.9)
                for i in range(4)
            ]
            summary = MediaSummary(
                videos=1,
                observations=4,
                video_observations=4,
                longest_video_run=4,
                has_motion_source=True,
            )
            return observations, summary

        paths = [self._clip(tmp_path, "a.mp4"), self._clip(tmp_path, "b.mp4")]
        with patch("app.api.media.observations_from_video", side_effect=fake_video):
            observations, summary = observations_from_uploads(paths, get_settings())

        indices = [o.frame_index for o in observations]
        assert indices == sorted(indices), f"not monotonic: {indices}"
        assert len(set(indices)) == len(indices), f"duplicate indices: {indices}"

    def test_gait_gets_one_clip_not_the_concatenation(self, tmp_path) -> None:
        from unittest.mock import patch

        import numpy as np

        from app.api.media import (
            MediaSummary,
            gait_observations,
            observations_from_uploads,
        )
        from app.core.types import TrackObservation

        lengths = iter([6, 3])

        def fake_video(path, settings, pipeline=None):
            count = next(lengths)
            observations = [
                TrackObservation(i, float(i), np.zeros((40, 20, 3), np.uint8), 40.0, 0.9)
                for i in range(count)
            ]
            return observations, MediaSummary(
                videos=1,
                observations=count,
                video_observations=count,
                longest_video_run=count,
                has_motion_source=True,
            )

        paths = [self._clip(tmp_path, "a.mp4"), self._clip(tmp_path, "b.mp4")]
        with patch("app.api.media.observations_from_video", side_effect=fake_video):
            observations, summary = observations_from_uploads(paths, get_settings())

        assert len(observations) == 9
        # Gait sees only the longer clip, not the splice.
        segment = gait_observations(observations, summary)
        assert len(segment) == 6
        assert summary.longest_video_run == 6

    def test_gait_segment_is_empty_for_photos_only(self) -> None:
        from app.api.media import MediaSummary, gait_observations

        summary = MediaSummary(images=5, observations=5)
        assert gait_observations([1, 2, 3, 4, 5], summary) == []


class TestGaitIsOfferedEveryClip:
    """Gait must not be handed a clip by upload order.

    It used to take the single longest run, and ties went to whichever video
    was uploaded first. Ties are the NORMAL case: enrolment caps each video at
    ENROLMENT_MAX_OBSERVATIONS, so any two clips longer than that are both
    exactly 600 frames. A face video uploaded before a walking video therefore
    won, and gait analysed somebody standing still -- reporting, accurately,
    "legs barely move; person is not walking".

    Measured on real enrolment footage: the face clip and the walking clip both
    capped at 600, face was offered first and refused, and the walking clip
    enrolled at quality 0.713 the moment it was tried.
    """

    @staticmethod
    def _summary_of(counts: list[int]):
        from app.api.media import MediaSummary

        summary = MediaSummary(videos=len(counts), has_motion_source=True)
        start = 0
        for count in counts:
            summary.video_segments.append((start, start + count))
            if count > summary.longest_video_run:
                summary.longest_video_run = count
                summary.gait_segment = (start, start + count)
            start += count
        summary.observations = start
        summary.video_observations = start
        return summary

    def test_every_clip_is_offered_not_just_the_longest(self) -> None:
        from app.api.media import gait_candidates

        summary = self._summary_of([600, 600, 545])
        observations = list(range(1745))
        candidates = gait_candidates(observations, summary)

        assert len(candidates) == 3, "all three clips must be offered"
        assert [len(c) for c in candidates] == [600, 600, 545]

    def test_tied_clips_are_both_reachable(self) -> None:
        """The exact failure: two clips at the cap, only one ever tried."""
        from app.api.media import gait_candidates, gait_observations

        summary = self._summary_of([600, 600])
        observations = list(range(1200))

        # The old single-pick path can only ever see the first.
        assert gait_observations(observations, summary) == list(range(600))
        # The new one reaches the second as well.
        reachable = {tuple(c) for c in gait_candidates(observations, summary)}
        assert tuple(range(600, 1200)) in reachable

    def test_longest_is_offered_first(self) -> None:
        """Ordering is a cost heuristic: a longer clip more often holds a
        complete cycle, so trying it first usually means trying it once."""
        from app.api.media import gait_candidates

        summary = self._summary_of([120, 600, 300])
        candidates = gait_candidates(list(range(1020)), summary)
        assert [len(c) for c in candidates] == [600, 300, 120]

    def test_photos_only_offers_nothing(self) -> None:
        from app.api.media import MediaSummary, gait_candidates

        summary = MediaSummary(images=5, observations=5)
        assert gait_candidates([1, 2, 3, 4, 5], summary) == []

    def test_an_older_summary_still_yields_its_one_segment(self) -> None:
        """`video_segments` is new; a summary built before it must not go from
        one candidate to none."""
        from app.api.media import MediaSummary, gait_candidates

        summary = MediaSummary(
            videos=1, has_motion_source=True, longest_video_run=40,
            gait_segment=(10, 50),
        )
        candidates = gait_candidates(list(range(60)), summary)
        assert [len(c) for c in candidates] == [40]


class TestUploadsKeepTheRealFrameRate:
    """Renumbering the uploads must not rewrite how fast they were shot.

    Frame indices are counters and may be renumbered freely. Timestamps are
    not: they are the only record of the capture rate, and gait recovers
    cadence from them (LOG-06). This assigned `float(offset + index)` to both,
    which says one second per frame whatever the camera did. At the resulting
    1Hz, gait's own resolution check refused every enrolment with "frames too
    far apart to measure cadence" -- for everybody, on every upload, which is
    why no watchlist entry ever held a gait template.
    """

    @staticmethod
    def _clip(directory: Path, name: str, frames: int = 4) -> Path:
        """A real file on disk. `observations_from_uploads` classifies by
        suffix but still needs the path to exist for the caller to stage it."""
        path = directory / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
        )
        for index in range(frames):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(
                frame, (10 + index, 20), (60 + index, 100), (200, 200, 200), -1
            )
            writer.write(frame)
        writer.release()
        return path

    @staticmethod
    def _fake_video(fps: float, count: int):
        import numpy as np

        from app.api.media import MediaSummary
        from app.core.types import TrackObservation

        def make(path, settings, pipeline=None):
            observations = [
                TrackObservation(
                    i, i / fps, np.zeros((40, 20, 3), np.uint8), 40.0, 0.9
                )
                for i in range(count)
            ]
            return observations, MediaSummary(
                videos=1,
                observations=count,
                video_observations=count,
                longest_video_run=count,
                has_motion_source=True,
            )

        return make

    def _run(self, tmp_path, fps, count, names=("a.mp4", "b.mp4")):
        from unittest.mock import patch

        from app.api.media import observations_from_uploads

        paths = [self._clip(tmp_path, n) for n in names]
        with patch(
            "app.api.media.observations_from_video",
            side_effect=self._fake_video(fps, count),
        ):
            return observations_from_uploads(paths, get_settings())

    def test_spacing_within_a_clip_survives_renumbering(self, tmp_path) -> None:
        import numpy as np

        from app.api.media import gait_observations

        observations, summary = self._run(tmp_path, fps=30.0, count=20)
        segment = gait_observations(observations, summary)
        gaps = np.diff([o.timestamp_s for o in segment])

        assert float(np.median(gaps)) == pytest.approx(1 / 30.0, abs=1e-6), (
            "the clip was shot at 30fps; the segment handed to gait must still "
            f"say so, not {1 / float(np.median(gaps)):.1f}fps"
        )

    def test_a_thirty_fps_clip_can_resolve_a_half_cycle(self, tmp_path) -> None:
        """The gate that was failing, asserted directly."""
        from app.api.media import gait_observations
        from app.core.config import get_settings as settings_of
        from app.embeddings.gait import resample_cadence
        from app.embeddings.silhouette import Silhouette

        import numpy as np

        observations, summary = self._run(tmp_path, fps=30.0, count=40)
        segment = gait_observations(observations, summary)
        silhouettes = [
            Silhouette(
                frame_index=o.frame_index,
                image=np.zeros((64, 44), np.float32),
                coverage=0.5,
                clipped=False,
                timestamp_s=o.timestamp_s,
            )
            for o in segment
        ]
        cfg = settings_of().gait
        cadence = resample_cadence(
            silhouettes, np.zeros(len(silhouettes), np.float32), cfg.assumed_fps
        )

        assert cadence.rate_hz == pytest.approx(30.0, rel=0.01)
        assert cadence.resolves(
            cfg.min_half_period_s, cfg.min_samples_per_half_period
        ), "a 30fps clip must be able to resolve a half gait cycle"

    def test_timestamps_stay_ordered_across_clips(self, tmp_path) -> None:
        observations, _ = self._run(tmp_path, fps=30.0, count=20)
        stamps = [o.timestamp_s for o in observations]
        assert stamps == sorted(stamps), f"not monotonic: {stamps[:5]}..."
        assert len(set(stamps)) == len(stamps), "duplicate timestamps"

    def test_photos_land_after_the_footage(self, tmp_path) -> None:
        """Guards the new offsetting, rather than reproducing an old failure.

        Worth stating plainly: the previous code did NOT get this wrong. It
        offset photographs by frame index while videos also carried index-as-
        seconds, so both grew at the same rate and the order held. Keeping the
        real capture rate makes video timestamps much smaller than their frame
        count, and an offset still taken from the index would now overshoot in
        the other direction. Nothing here ever broke -- this is the check that
        stops the repair breaking it.
        """
        from unittest.mock import patch

        import numpy as np

        from app.api.media import MediaSummary, observations_from_uploads
        from app.core.types import TrackObservation

        def fake_images(paths, settings, detector=None):
            observations = [
                TrackObservation(i, 0.0, np.zeros((40, 20, 3), np.uint8), 40.0, 0.9)
                for i in range(3)
            ]
            return observations, MediaSummary(images=3, observations=3)

        video = self._clip(tmp_path, "v.mp4")
        photo = tmp_path / "p.jpg"
        cv2.imwrite(str(photo), np.zeros((40, 20, 3), np.uint8))

        with patch(
            "app.api.media.observations_from_video",
            side_effect=self._fake_video(30.0, 300),
        ), patch("app.api.media.observations_from_images", side_effect=fake_images):
            observations, _ = observations_from_uploads(
                [video, photo], get_settings()
            )

        stamps = [o.timestamp_s for o in observations]
        assert stamps == sorted(stamps), (
            "photographs must be timestamped after the video they follow, not "
            "by frame index"
        )
