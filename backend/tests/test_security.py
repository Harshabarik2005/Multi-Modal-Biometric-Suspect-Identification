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
