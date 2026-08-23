"""Regression tests for the security review findings.

Each test names the finding it guards. These are the ones where a silent
regression would be worst: a shared mutable threshold, an unverified TLS
channel, a pickle sink, and a path that escapes its root.
"""

from __future__ import annotations

import ssl
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.repository import make_engine


@pytest.fixture
def client(api_client):
    """Signed in. Auth is required on every route except /health (SEC-01)."""
    return api_client


class TestSettingsIsolation:
    """SEC-02: a request must not rewrite settings for the whole process."""

    def test_settings_are_a_shared_singleton(self) -> None:
        """The property that made the bug possible, asserted so it stays known."""
        assert get_settings() is get_settings()

    def test_a_deep_copy_does_not_leak_back(self) -> None:
        original = get_settings()
        before = original.fusion.threshold

        job_settings = original.model_copy(deep=True)
        job_settings.fusion.threshold = 0.01

        assert get_settings().fusion.threshold == before, (
            "mutating a per-job copy must not touch the shared settings"
        )
        assert job_settings.fusion.threshold == 0.01

    def test_nested_models_are_copied_too(self) -> None:
        """A shallow copy would share the nested FusionSettings object."""
        original = get_settings()
        copy = original.model_copy(deep=True)
        assert copy.fusion is not original.fusion
        assert copy.track_buffer is not original.track_buffer

    def test_scan_rejects_an_out_of_range_threshold(self, client) -> None:
        """Unbounded, one request could set this to 0 and match everyone."""
        import cv2

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        for bad in ("-5", "2.5"):
            response = client.post(
                "/api/scan",
                data={"threshold": bad},
                files=[("files", ("clip.mp4", buffer.tobytes(), "video/mp4"))],
            )
            assert response.status_code == 422, f"threshold={bad} was accepted"


class TestConfigurationIsHonoured:
    """SEC-05: documented settings were silently dropped."""

    def test_database_url_is_a_real_field(self, monkeypatch) -> None:
        monkeypatch.setenv("FRS_DATABASE_URL", "postgresql+psycopg://u:p@h/db")
        get_settings.cache_clear()
        try:
            assert get_settings().database_url == "postgresql+psycopg://u:p@h/db"
        finally:
            monkeypatch.delenv("FRS_DATABASE_URL", raising=False)
            get_settings.cache_clear()

    def test_job_workers_is_a_real_field(self) -> None:
        assert isinstance(get_settings().job_workers, int)

    def test_an_unknown_setting_fails_loudly(self) -> None:
        """extra='ignore' is how FRS_DATABASE_URL vanished without a word."""
        with pytest.raises(Exception):
            Settings(totally_made_up_setting=1)


class TestCredentialsAreNotPrintable:
    """SEC-12: repr(settings) and validation errors print field values."""

    def test_smtp_password_is_masked(self) -> None:
        settings = Settings(alerts={"smtp_password": "hunter2"})
        assert "hunter2" not in repr(settings)
        assert "hunter2" not in str(settings.alerts.smtp_password)
        assert settings.alerts.smtp_password.get_secret_value() == "hunter2"

    def test_twilio_token_is_masked(self) -> None:
        settings = Settings(alerts={"twilio_auth_token": "sk-secret"})
        assert "sk-secret" not in repr(settings)
        assert settings.alerts.twilio_auth_token.get_secret_value() == "sk-secret"


class TestSMTPTransportSecurity:
    """SEC-04: bare starttls() does not verify the server certificate."""

    def test_the_stdlib_default_really_is_unverified(self) -> None:
        """The reason the explicit context is needed, pinned as a fact."""
        insecure = ssl._create_stdlib_context()
        assert insecure.check_hostname is False
        assert insecure.verify_mode == ssl.CERT_NONE

        secure = ssl.create_default_context()
        assert secure.check_hostname is True
        assert secure.verify_mode == ssl.CERT_REQUIRED

    def test_starttls_is_given_a_verifying_context(self) -> None:
        from app.alerts.notifier import SMTPNotifier

        source = Path(SMTPNotifier.send.__code__.co_filename).read_text(
            encoding="utf-8"
        )
        assert "starttls(context=ssl.create_default_context())" in source
        assert "server.starttls()" not in source

    def test_plaintext_smtp_is_refused(self) -> None:
        """An alert names an identified person, a camera and a time."""
        from app.alerts.notifier import SMTPNotifier

        with pytest.raises(ValueError, match="unencrypted"):
            SMTPNotifier(
                host="smtp.example.com",
                recipients=["duty@example.com"],
                use_tls=False,
            )


class TestNoPickleSinks:
    """SEC-03: torch.load with weights_only=False executes arbitrary code."""

    @pytest.mark.parametrize(
        "module_path",
        ["app/embeddings/reid.py", "app/fusion/attention.py"],
    )
    def test_torch_load_is_restricted(self, module_path: str) -> None:
        source = (Path(__file__).resolve().parents[1] / module_path).read_text(
            encoding="utf-8"
        )
        assert "weights_only=False" not in source, (
            f"{module_path} unpickles a file downloaded over the network"
        )
        assert "weights_only=True" in source

    def test_a_downloaded_html_page_is_rejected(self, tmp_path) -> None:
        """A Drive quota page passes is_file() and only fails inside the loader."""
        # gdown is only needed to fetch weights, so it is not a hard dependency;
        # without it there is no download path to test.
        pytest.importorskip("gdown")

        from app.embeddings.reid import ReIDEmbedder

        embedder = ReIDEmbedder.__new__(ReIDEmbedder)
        embedder.cfg = get_settings().reid

        fake = tmp_path / "osnet_x1_0_imagenet.pth"
        fake.write_bytes(b"<!DOCTYPE html><html>Quota exceeded</html>")

        import unittest.mock as mock

        with mock.patch("gdown.download", return_value=str(fake)):
            with pytest.raises(RuntimeError, match="not a torch checkpoint"):
                embedder._download_weights(fake)
        assert not fake.exists(), "the bogus download should be removed"


class TestPersonIdCannotEscapeItsRoot:
    """SEC-07: person_id becomes a directory name."""

    @pytest.mark.parametrize(
        "person_id",
        ["../../../../tmp/pwned", "/tmp/absolute", "a/b", "..", "with space"],
    )
    def test_traversal_shapes_are_rejected_by_the_api(
        self, client, person_id: str
    ) -> None:
        import cv2

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        response = client.post(
            "/api/enroll",
            data={"person_id": person_id, "display_name": "X"},
            files=[("files", ("a.jpg", buffer.tobytes(), "image/jpeg"))],
        )
        assert response.status_code == 422, f"{person_id!r} was accepted"

    @pytest.mark.parametrize("person_id", ["ravi", "subject_a", "p-1.2"])
    def test_ordinary_ids_are_accepted(self, person_id: str) -> None:
        import re

        from app.api.ingest import PERSON_ID_PATTERN

        assert re.match(PERSON_ID_PATTERN, person_id)


class TestJobListingDoesNotLeak:
    """SEC-08: the listing returned every scan result to anyone who asked."""

    def test_results_and_raw_errors_are_withheld(self, client) -> None:
        from app.api.jobs import JobStatus

        runner = client.app.state.job_runner
        job = runner.submit("scan", lambda _: {"findings": [{"person_id": "ravi"}]})
        for _ in range(100):
            if runner.get(job.id).status is JobStatus.SUCCEEDED:
                break
            import time

            time.sleep(0.05)

        listing = client.get("/api/jobs").json()
        assert listing, "the job should be listed"
        assert all(entry["result"] is None for entry in listing), (
            "the listing must not hand out who was found"
        )

        # Fetching it directly, with its id, still returns the result.
        detail = client.get(f"/api/jobs/{job.id}").json()
        assert detail["result"]["findings"][0]["person_id"] == "ravi"

    def test_error_text_is_not_echoed_in_the_listing(self, client) -> None:
        from app.api.jobs import JobStatus

        runner = client.app.state.job_runner

        def explode(_):
            raise RuntimeError("C:/secret/path/to/model.pth is missing")

        job = runner.submit("scan", explode)
        for _ in range(100):
            if runner.get(job.id).status is JobStatus.FAILED:
                break
            import time

            time.sleep(0.05)

        listing = client.get("/api/jobs").json()
        assert all("secret" not in (entry["error"] or "") for entry in listing)


class TestEncryptionIsNotOptional:
    """SEC-06: plaintext biometric templates used to be the default.

    With no key the code wrote raw float32 vectors and logged a warning, so
    every default deployment stored biometric data in the clear. Worse, because
    encryption is recorded per row, a deployment could be half-encrypted and
    still look right in a spot check.
    """

    @staticmethod
    def _no_key(monkeypatch) -> None:
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", raising=False)

    def test_a_gallery_write_without_a_key_is_refused(self, monkeypatch) -> None:
        from app.matching.gallery import GalleryStore

        self._no_key(monkeypatch)
        store = GalleryStore.__new__(GalleryStore)
        with pytest.raises(RuntimeError, match="unencrypted"):
            store._encode_vectors({"face": np.zeros(4, dtype=np.float32)})

    def test_a_template_write_without_a_key_is_refused(self, monkeypatch) -> None:
        from app.db.repository import WatchlistRepository, create_schema, session_factory

        self._no_key(monkeypatch)
        engine = make_engine("sqlite:///:memory:")
        create_schema(engine)
        session = session_factory(engine)()
        repo = WatchlistRepository(session)
        try:
            with pytest.raises(RuntimeError, match="unencrypted"):
                repo._encode(np.zeros(4, dtype=np.float32))
        finally:
            session.close()

    def test_the_refusal_says_how_to_fix_it(self, monkeypatch) -> None:
        """An error nobody can act on just gets worked around."""
        from app.matching.gallery import refuse_plaintext

        self._no_key(monkeypatch)
        with pytest.raises(RuntimeError) as caught:
            refuse_plaintext("a biometric template")

        message = str(caught.value)
        assert "FRS_TEMPLATE_ENCRYPTION_KEY" in message
        assert "Fernet.generate_key" in message
        assert "FRS_ALLOW_PLAINTEXT_TEMPLATES" in message

    def test_the_escape_hatch_is_explicit(self, monkeypatch) -> None:
        """Local development can still write plaintext, but only on purpose."""
        from app.matching.gallery import GalleryStore

        self._no_key(monkeypatch)
        monkeypatch.setenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", "1")

        store = GalleryStore.__new__(GalleryStore)
        blob = store._encode_vectors({"face": np.zeros(4, dtype=np.float32)})
        assert not blob.startswith(b"FRSENC1:")

    def test_a_key_is_enough_on_its_own(self, monkeypatch) -> None:
        """The opt-out is not needed when encryption is actually configured."""
        from cryptography.fernet import Fernet

        from app.matching.gallery import GalleryStore

        monkeypatch.setenv("FRS_TEMPLATE_ENCRYPTION_KEY", Fernet.generate_key().decode())
        monkeypatch.delenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", raising=False)

        store = GalleryStore.__new__(GalleryStore)
        assert store._encode_vectors(
            {"face": np.zeros(4, dtype=np.float32)}
        ).startswith(b"FRSENC1:")


class TestReIDWeightsAreReIDWeights:
    """DES-01: the branch was running ImageNet weights, not re-ID weights.

    OSNet without re-ID training gives generic visual features. Everything
    anchored on the resulting scores -- reid_impostor, reid_threshold, the
    calibration mapping, the fusion weights -- was calibrated against a model
    that does not do the job the branch claims to do.
    """

    def test_the_imagenet_checkpoints_are_still_labelled_as_such(self) -> None:
        """The ids are torchreid's ImageNet ones; the label must not drift."""
        from app.embeddings.reid import PRETRAINED_URLS

        assert (
            PRETRAINED_URLS[("osnet_x1_0", "imagenet")]
            == "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY"
        )

    def test_reid_trained_checkpoints_are_available(self) -> None:
        from app.embeddings.reid import available_weights

        assert {"msmt17", "market1501", "dukemtmcreid"} <= set(
            available_weights("osnet_x1_0")
        )

    def test_the_default_is_not_imagenet(self) -> None:
        from app.core.config import Settings
        from app.embeddings.reid import NON_REID_WEIGHTS

        assert Settings().reid.weights not in NON_REID_WEIGHTS

    def test_the_weights_file_is_named_for_its_training_set(self) -> None:
        """Otherwise switching weights silently reuses the cached old file."""
        from app.embeddings.reid import ReIDEmbedder

        embedder = ReIDEmbedder.__new__(ReIDEmbedder)
        embedder.settings = get_settings()
        embedder.cfg = get_settings().reid
        embedder.weights = "msmt17"
        assert embedder._weights_path().name == "osnet_x1_0_msmt17.pth"

        embedder.weights = "imagenet"
        assert embedder._weights_path().name == "osnet_x1_0_imagenet.pth"

    def test_stale_calibration_is_reported(self) -> None:
        """Anchors belong to one checkpoint; they do not transfer to another."""
        settings = Settings()
        settings.reid.weights = "msmt17"
        settings.fusion.reid_anchors_measured_on = "imagenet"

        message = settings.reid_calibration_mismatch()
        assert message is not None
        assert "imagenet" in message and "msmt17" in message

        settings.fusion.reid_anchors_measured_on = "msmt17"
        assert settings.reid_calibration_mismatch() is None

    def test_the_operator_is_told_not_just_the_log(self, client, monkeypatch) -> None:
        """Someone judging a match must see that its score is uncalibrated."""
        settings = get_settings()
        assert settings.reid_calibration_mismatch() is None, (
            "the shipped config should be self-consistent"
        )
        assert client.get("/api/stats").json()["warnings"] == []

        # Now make it stale, the way switching reid.weights would.
        monkeypatch.setattr(
            settings.fusion, "reid_anchors_measured_on", "imagenet"
        )
        warnings = client.get("/api/stats").json()["warnings"]
        assert any("calibration" in w.lower() for w in warnings)

    def test_an_unavailable_checkpoint_is_refused_by_name(self) -> None:
        """osnet_x0_25 has no re-ID weights; asking must not fall back."""
        from app.core.config import Settings
        from app.embeddings.reid import ReIDEmbedder

        settings = Settings()
        settings.reid.model = "osnet_x0_25"
        settings.reid.weights = "msmt17"

        with pytest.raises(ValueError, match="No 'msmt17' checkpoint"):
            ReIDEmbedder(settings=settings)


class TestEmbeddingsCarryTheirModel:
    """DES-01, second half: a reference is only comparable to a probe from the
    same model.

    Switching `reid.weights` is the right thing to do, and it silently
    invalidates every template already enrolled. The vectors still load, still
    have the right length, and still produce a cosine similarity -- one that
    means nothing, because the two vectors live in unrelated spaces. Nothing
    about that failure is visible in a score.
    """

    @staticmethod
    def _embedding(model_id: str, seed: int) -> "ModalityEmbedding":
        from app.core.types import Modality, ModalityEmbedding, l2_normalize

        rng = np.random.default_rng(seed)
        return ModalityEmbedding(
            modality=Modality.REID,
            vector=l2_normalize(rng.normal(size=64).astype(np.float32)),
            quality=0.9,
            frames_used=10,
            model_id=model_id,
        )

    def test_a_mismatch_is_reported(self) -> None:
        from app.matching.gallery import model_mismatch

        probe = self._embedding("osnet_x1_0/msmt17", 1)
        reference = self._embedding("osnet_x1_0/imagenet", 2)

        message = model_mismatch(probe, reference)
        assert message is not None
        assert "msmt17" in message and "imagenet" in message
        assert "Re-enrol" in message

    def test_matching_models_compare(self) -> None:
        from app.matching.gallery import model_mismatch

        assert (
            model_mismatch(
                self._embedding("osnet_x1_0/msmt17", 1),
                self._embedding("osnet_x1_0/msmt17", 2),
            )
            is None
        )

    def test_an_unknown_model_does_not_strand_old_enrolments(self) -> None:
        """Refusing these would break every watchlist enrolled before this."""
        from app.matching.gallery import model_mismatch

        assert (
            model_mismatch(
                self._embedding("osnet_x1_0/msmt17", 1), self._embedding("", 2)
            )
            is None
        )

    def test_the_gallery_refuses_rather_than_scoring(self) -> None:
        """A cross-model pair must read as 'could not compare', not as a low
        score -- those mean completely different things to fusion."""
        from app.core.types import Modality
        from app.matching.gallery import Gallery, PersonRecord

        gallery = Gallery()
        gallery.add(
            PersonRecord(
                person_id="ravi",
                display_name="Ravi",
                embeddings={Modality.REID: self._embedding("osnet_x1_0/imagenet", 2)},
            )
        )

        probe = self._embedding("osnet_x1_0/msmt17", 1)
        score = gallery.rank({Modality.REID: probe})[0].scores[Modality.REID]

        assert score.similarity is None
        assert "different models" in score.incomparable_reason

    def test_the_same_model_still_scores(self) -> None:
        from app.core.types import Modality
        from app.matching.gallery import Gallery, PersonRecord

        reference = self._embedding("osnet_x1_0/msmt17", 2)
        gallery = Gallery()
        gallery.add(
            PersonRecord(
                person_id="ravi", display_name="Ravi",
                embeddings={Modality.REID: reference},
            )
        )
        score = gallery.rank({Modality.REID: reference})[0].scores[Modality.REID]
        assert score.similarity == pytest.approx(1.0, abs=1e-5)

    def test_the_model_survives_a_database_round_trip(self) -> None:
        from app.core.types import Modality
        from app.db.repository import (
            WatchlistRepository,
            create_schema,
            session_factory,
        )

        engine = make_engine("sqlite:///:memory:")
        create_schema(engine)
        session = session_factory(engine)()
        repo = WatchlistRepository(session)
        try:
            repo.enroll(
                "ravi", "Ravi",
                {Modality.REID: self._embedding("osnet_x1_0/msmt17", 3)},
            )
            record = repo.to_record(repo.get_person("ravi"))
            assert record.embeddings[Modality.REID].model_id == "osnet_x1_0/msmt17"
        finally:
            session.close()

    def test_an_older_database_gains_the_column(self) -> None:
        """create_all does not ALTER, so an upgrade would break on first query."""
        from sqlalchemy import inspect, text

        from app.db.repository import create_schema

        engine = make_engine("sqlite:///:memory:")
        create_schema(engine)
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE templates DROP COLUMN model_id"))
        assert "model_id" not in {
            c["name"] for c in inspect(engine).get_columns("templates")
        }

        create_schema(engine)
        assert "model_id" in {
            c["name"] for c in inspect(engine).get_columns("templates")
        }

    def test_the_console_counts_unstamped_templates(self, client) -> None:
        """The operator should be told their watchlist may predate a change."""
        from sqlalchemy import text

        from app.db.repository import WatchlistRepository, session_factory
        from app.core.types import Modality

        session = session_factory(client.engine)()
        try:
            repo = WatchlistRepository(session)
            repo.enroll(
                "ravi", "Ravi",
                {Modality.REID: self._embedding("osnet_x1_0/msmt17", 4)},
            )
            assert repo.templates_without_a_model() == 0
            assert client.get("/api/stats").json()["warnings"] == []

            # An enrolment from before the model was recorded.
            session.execute(text("UPDATE templates SET model_id = ''"))
            session.commit()
            assert repo.templates_without_a_model() == 1
        finally:
            session.close()

        warnings = client.get("/api/stats").json()["warnings"]
        assert any("do not record which model" in w for w in warnings)


class TestUploadedFootageIsAlwaysDeleted:
    """SEC-11: the delete lived in the job body's own `finally`.

    A job that is cancelled at shutdown, or still queued when the process
    stops, never enters that `finally` -- so a temp directory of somebody's
    biometric footage survived with nothing tracking it.
    """

    def test_cleanup_runs_for_a_job_that_never_starts(self, tmp_path) -> None:
        import threading

        from app.api.jobs import JobRunner

        started = threading.Event()
        release = threading.Event()
        cleaned: list[str] = []

        runner = JobRunner(max_workers=1)
        try:
            # Occupy the single worker so the second job stays queued.
            runner.submit(
                "block",
                lambda job: (started.set(), release.wait(5)),
                cleanup=lambda: cleaned.append("first"),
            )
            assert started.wait(5)

            runner.submit(
                "queued",
                lambda job: cleaned.append("this should never run"),
                cleanup=lambda: cleaned.append("second"),
            )
        finally:
            release.set()
            runner.shutdown()

        assert "second" in cleaned, "the queued job's upload was left on disk"
        assert "this should never run" not in cleaned

    def test_cleanup_runs_once_not_twice(self) -> None:
        from app.api.jobs import JobRunner

        calls = []
        runner = JobRunner(max_workers=1)
        runner.submit("work", lambda job: None, cleanup=lambda: calls.append(1))
        runner.shutdown()
        assert calls == [1]

    def test_cleanup_still_runs_when_the_job_fails(self) -> None:
        from app.api.jobs import JobRunner

        calls = []

        def explode(job):
            raise RuntimeError("boom")

        runner = JobRunner(max_workers=1)
        runner.submit("work", explode, cleanup=lambda: calls.append(1))
        runner.shutdown()
        assert calls == [1]

    def test_the_startup_sweep_removes_orphans(self, monkeypatch, tmp_path) -> None:
        """Nothing in-process survives a kill -9, so old dirs are swept."""
        import tempfile
        import time

        from app.api.ingest import UPLOAD_PREFIX, sweep_stale_uploads

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        old = tmp_path / f"{UPLOAD_PREFIX}orphan"
        old.mkdir()
        (old / "footage.mp4").write_bytes(b"personal data")
        import os

        long_ago = time.time() - 24 * 3600
        os.utime(old, (long_ago, long_ago))

        # A directory a live request may be writing into right now.
        fresh = tmp_path / f"{UPLOAD_PREFIX}inflight"
        fresh.mkdir()

        # And something that is not ours at all.
        other = tmp_path / "someone-elses-tempdir"
        other.mkdir()

        assert sweep_stale_uploads(older_than_hours=6.0) == 1
        assert not old.exists()
        assert fresh.exists(), "deleting an in-flight upload would break a request"
        assert other.exists()


class TestTheReviewerCanSeeThePerson:
    """DES-02: the review card showed a score and no image.

    The human confirmation is the safeguard the whole architecture rests on.
    Without a picture the reviewer can see *how* the system reached its
    conclusion but not *whether* it is right, which makes the safeguard a
    formality that produces an audit trail saying a human checked.
    """

    @staticmethod
    def _crop(height: int = 400, width: int = 160) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    @classmethod
    def _encoded(cls) -> bytes:
        from app.core.evidence import encode_crop

        return encode_crop(cls._crop())

    def _repo(self, engine):
        from app.core.types import Modality, ModalityEmbedding
        from app.db.repository import WatchlistRepository, session_factory

        session = session_factory(engine)()
        repo = WatchlistRepository(session)
        repo.enroll(
            "ravi",
            "Ravi Kumar",
            {
                Modality.FACE: ModalityEmbedding(
                    modality=Modality.FACE,
                    vector=np.ones(8, dtype=np.float32),
                    quality=0.9,
                    frames_used=5,
                )
            },
            reference_jpeg=self._encoded(),
        )
        return repo, session

    def test_a_crop_round_trips_through_the_decision(self, client) -> None:
        repo, session = self._repo(client.engine)
        try:
            original = self._encoded()
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=original
            )
            assert repo.decision_evidence(decision.id) == original
        finally:
            session.close()

    def test_the_stored_image_is_encrypted(self, client) -> None:
        """A picture of an identified person is personal data too."""
        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
            session.refresh(decision)
            assert decision.evidence_jpeg.startswith(b"FRSENC1:")
            assert b"\xff\xd8\xff" not in decision.evidence_jpeg[:64]
        finally:
            session.close()

    def test_the_routes_serve_both_images(self, client) -> None:
        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
        finally:
            session.close()

        seen = client.get(f"/api/decisions/{decision.id}/evidence")
        assert seen.status_code == 200
        assert seen.headers["content-type"] == "image/jpeg"
        assert seen.content.startswith(b"\xff\xd8\xff")

        enrolled = client.get("/api/watchlist/ravi/reference")
        assert enrolled.status_code == 200
        assert enrolled.content.startswith(b"\xff\xd8\xff")

    def test_images_require_a_session(self, anon_client, client) -> None:
        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
        finally:
            session.close()

        assert (
            anon_client.get(f"/api/decisions/{decision.id}/evidence").status_code
            == 401
        )
        assert anon_client.get("/api/watchlist/ravi/reference").status_code == 401

    def test_a_missing_image_is_reported_not_faked(self, client) -> None:
        """A reviewer deciding without evidence must know that is the case."""
        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match("ravi", track_id=2, score=0.9)
        finally:
            session.close()

        assert (
            client.get(f"/api/decisions/{decision.id}/evidence").status_code == 404
        )
        listed = client.get(f"/api/decisions/{decision.id}").json()
        assert listed["has_evidence"] is False
        assert listed["has_reference"] is True

    def test_the_listing_flags_images_without_carrying_them(self, client) -> None:
        """A hundred pending decisions must not mean a hundred JPEGs inline."""
        repo, session = self._repo(client.engine)
        try:
            repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
        finally:
            session.close()

        body = client.get("/api/decisions").json()
        assert body[0]["has_evidence"] is True
        assert "evidence_jpeg" not in body[0]

    def test_evidence_can_be_purged_without_losing_the_decision(self, client) -> None:
        """Keeping photographs indefinitely is not the same as keeping a trail."""
        from datetime import datetime, timedelta, timezone

        from app.db.models import DecisionStatus, MatchDecision

        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
            repo.review(decision.id, "alice", DecisionStatus.CONFIRMED)

            session.execute(
                MatchDecision.__table__.update()
                .where(MatchDecision.id == decision.id)
                .values(created_at=datetime.now(timezone.utc) - timedelta(days=400))
            )
            session.commit()

            assert repo.purge_evidence(older_than_days=365) == 1
            assert repo.decision_evidence(decision.id) is None

            kept = session.get(MatchDecision, decision.id)
            assert kept.score == pytest.approx(0.9)
            assert len(kept.reviews) == 1
        finally:
            session.close()

    def test_a_recent_decision_keeps_its_evidence(self, client) -> None:
        repo, session = self._repo(client.engine)
        try:
            decision = repo.record_match(
                "ravi", track_id=1, score=0.9, evidence_jpeg=self._encoded()
            )
            assert repo.purge_evidence(older_than_days=30) == 0
            assert repo.decision_evidence(decision.id) is not None
        finally:
            session.close()


class TestThePreviewDoesNotBlockTheServer:
    """SEC-09, second half: an async route running YOLO inline.

    `preview_enrollment` is `async def` but the work inside is detection on
    every frame -- seconds of pure CPU. Running it inline blocked every other
    request for the duration, including the job-status polls the browser makes
    while a scan is running.
    """

    def test_the_detection_work_is_offloaded(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "app" / "api" / "ingest.py"
        ).read_text(encoding="utf-8")

        preview = source[source.index("async def preview_enrollment") :]
        preview = preview[: preview.index("\n@router")]

        assert "run_in_threadpool" in preview, (
            "preview runs detection on the event loop, stalling every other "
            "request on the server"
        )
        assert "await run_in_threadpool(\n            observations_from_uploads" in preview
