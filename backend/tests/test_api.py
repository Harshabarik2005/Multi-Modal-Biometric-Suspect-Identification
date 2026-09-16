"""Tests for the database layer and API (Phase 7).

In-memory SQLite, no model weights, no video. The tests that matter most here
are the guardrail ones: that nothing can confirm a match without a human, that
the audit trail cannot be rewritten, and that templates are encrypted.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.types import Modality, ModalityEmbedding, l2_normalize
from app.db.models import DecisionStatus, Template
from app.db.repository import (
    WatchlistRepository,
    create_schema,
    make_engine,
    session_factory,
)


@pytest.fixture
def repo():
    engine = make_engine("sqlite:///:memory:")
    create_schema(engine)
    session = session_factory(engine)()
    yield WatchlistRepository(session)
    session.close()


@pytest.fixture
def client(api_client):
    """Signed in. Auth is required on every route except /health (SEC-01)."""
    return api_client


def embedding(modality: Modality = Modality.FACE, dim: int = 8, seed: int = 0):
    generator = np.random.default_rng(seed)
    return ModalityEmbedding(
        modality=modality,
        vector=l2_normalize(generator.normal(size=dim).astype(np.float32)),
        quality=0.8,
        frames_used=20,
    )


def enroll(repo, person_id="ravi", name="Ravi Kumar", **kwargs):
    return repo.enroll(
        person_id,
        name,
        {Modality.FACE: embedding(), Modality.REID: embedding(Modality.REID, seed=1)},
        **kwargs,
    )


class TestEnrollment:
    def test_round_trip_preserves_the_vector(self, repo) -> None:
        original = embedding()
        repo.enroll("ravi", "Ravi", {Modality.FACE: original})
        record = repo.to_record(repo.get_person("ravi"))
        assert np.allclose(record.embeddings[Modality.FACE].vector, original.vector)

    def test_refuses_a_person_with_no_usable_embedding(self, repo) -> None:
        with pytest.raises(ValueError, match="no modality"):
            repo.enroll(
                "ghost", "Ghost", {Modality.FACE: ModalityEmbedding.empty(Modality.FACE)}
            )

    def test_duplicate_enrollment_is_rejected_unless_replacing(self, repo) -> None:
        enroll(repo)
        with pytest.raises(ValueError, match="already enrolled"):
            enroll(repo)
        enroll(repo, name="Ravi Updated", replace=True)
        assert repo.get_person("ravi").display_name == "Ravi Updated"

    def test_replacing_does_not_duplicate_templates(self, repo) -> None:
        enroll(repo)
        enroll(repo, replace=True)
        assert len(repo.get_person("ravi").templates) == 2

    def test_enrollment_writes_an_audit_event(self, repo) -> None:
        enroll(repo, actor="alice")
        events = repo.audit_trail()
        assert any(e.kind == "enroll" and e.actor == "alice" for e in events)

    def test_gallery_loads_from_the_database(self, repo) -> None:
        enroll(repo)
        gallery = repo.load_gallery()
        assert len(gallery) == 1
        assert gallery.get("ravi") is not None


class TestTemplateEncryption:
    def test_templates_are_encrypted_when_a_key_is_set(self, monkeypatch) -> None:
        """A database dump must not leak biometric vectors."""
        from cryptography.fernet import Fernet

        monkeypatch.setenv("FRS_TEMPLATE_ENCRYPTION_KEY", Fernet.generate_key().decode())
        engine = make_engine("sqlite:///:memory:")
        create_schema(engine)
        session = session_factory(engine)()
        repo = WatchlistRepository(session)

        original = embedding()
        repo.enroll("ravi", "Ravi", {Modality.FACE: original})
        template = repo.get_person("ravi").templates[0]

        assert template.encrypted
        assert template.payload.startswith(b"FRSENC1:")
        # numpy's .npy magic must not be sitting there in the clear.
        assert b"\x93NUMPY" not in template.payload[:64]
        # ...and it still round-trips with the key.
        assert np.allclose(
            repo.to_record(repo.get_person("ravi")).embeddings[Modality.FACE].vector,
            original.vector,
        )
        session.close()

    def test_plaintext_is_flagged_as_unencrypted(self, monkeypatch) -> None:
        """The escape hatch writes plaintext, and says so on the row."""
        monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("FRS_ALLOW_PLAINTEXT_TEMPLATES", "1")
        engine = make_engine("sqlite:///:memory:")
        create_schema(engine)
        session = session_factory(engine)()
        repo = WatchlistRepository(session)

        enroll(repo)
        assert all(not t.encrypted for t in repo.get_person("ravi").templates)
        session.close()


class TestRetirement:
    def test_retiring_keeps_the_row_for_the_audit_trail(self, repo) -> None:
        """Deleting would orphan match decisions and make them unreviewable."""
        enroll(repo)
        repo.record_match("ravi", track_id=1, score=0.9)
        assert repo.retire("ravi", actor="alice")

        person = repo.get_person("ravi")
        assert person is not None
        assert not person.is_active
        assert len(repo.decisions_for("ravi")) == 1

    def test_retired_people_are_excluded_from_the_gallery(self, repo) -> None:
        enroll(repo)
        repo.retire("ravi")
        assert len(repo.load_gallery()) == 0
        assert len(repo.load_gallery(include_retired=True)) == 1

    def test_retiring_twice_reports_no_change(self, repo) -> None:
        enroll(repo)
        assert repo.retire("ravi")
        assert not repo.retire("ravi")


class TestRestore:
    """Retiring without an undo was a one-way door."""

    def test_a_retired_person_can_be_put_back(self, repo) -> None:
        enroll(repo)
        repo.retire("ravi")
        assert repo.restore("ravi", actor="alice")

        person = repo.get_person("ravi")
        assert person.is_active
        assert person.retired_at is None

    def test_restoring_brings_them_back_into_the_gallery(self, repo) -> None:
        """The point of the undo: they can be matched against again."""
        enroll(repo)
        repo.retire("ravi")
        assert len(repo.load_gallery()) == 0
        repo.restore("ravi")
        assert len(repo.load_gallery()) == 1

    def test_restoring_keeps_the_templates_that_were_never_destroyed(self, repo) -> None:
        enroll(repo)
        before = len(repo.get_person("ravi").templates)
        repo.retire("ravi")
        repo.restore("ravi")
        assert len(repo.get_person("ravi").templates) == before

    def test_restoring_someone_active_reports_no_change(self, repo) -> None:
        enroll(repo)
        assert not repo.restore("ravi")

    def test_restoring_writes_an_audit_event(self, repo) -> None:
        enroll(repo)
        repo.retire("ravi")
        repo.restore("ravi", actor="alice")
        assert any(
            e.kind == "restore" and e.actor == "alice" and e.subject == "ravi"
            for e in repo.audit_trail()
        )


class TestErasingBiometrics:
    """Destruction, as distinct from retirement.

    Retiring is a filing decision and reversible. This destroys the templates
    and the enrolment photograph and cannot be undone -- while keeping the row
    and its decisions, so what the system did to this person stays reviewable.
    """

    def test_templates_and_photo_are_destroyed(self, repo) -> None:
        enroll(repo, reference_jpeg=b"pretend-jpeg-bytes")
        assert repo.get_person("ravi").reference_jpeg is not None

        destroyed = repo.erase_biometrics("ravi", actor="alice")

        person = repo.get_person("ravi")
        assert destroyed == 2
        assert person.templates == []
        assert person.reference_jpeg is None, (
            "the enrolment photograph is a picture of them stored beside the "
            "vectors; leaving it makes 'erased' a lie"
        )

    def test_the_record_and_its_decisions_survive(self, repo) -> None:
        enroll(repo)
        repo.record_match("ravi", track_id=1, score=0.9)
        repo.erase_biometrics("ravi")

        assert repo.get_person("ravi") is not None
        assert len(repo.decisions_for("ravi")) == 1, (
            "erasing biometrics must not destroy the record of what the system "
            "already decided about this person"
        )

    def test_they_can_no_longer_be_matched(self, repo) -> None:
        enroll(repo)
        repo.erase_biometrics("ravi")
        gallery = repo.load_gallery()
        assert gallery.get("ravi") is None or len(gallery.get("ravi").embeddings) == 0

    def test_erasing_writes_an_audit_event(self, repo) -> None:
        enroll(repo)
        repo.erase_biometrics("ravi", actor="alice")
        assert any(
            e.kind == "erase_biometrics" and e.actor == "alice"
            for e in repo.audit_trail()
        )

    def test_erasing_an_unknown_person_destroys_nothing(self, repo) -> None:
        assert repo.erase_biometrics("nobody") == 0


class TestPermanentDeletion:
    """Allowed only while it is safe -- refused the moment it is not."""

    def test_an_unmatched_person_can_be_deleted_outright(self, repo) -> None:
        enroll(repo)
        assert repo.delete_permanently("ravi", actor="alice")
        assert repo.get_person("ravi") is None

    def test_deleting_takes_the_templates_with_it(self, repo) -> None:
        enroll(repo)
        repo.delete_permanently("ravi")
        assert repo.session.query(Template).count() == 0

    def test_deletion_is_refused_once_someone_has_been_matched(self, repo) -> None:
        """`MatchDecision.person_pk` has no cascade, so this would leave the
        review queue pointing at nobody."""
        enroll(repo)
        repo.record_match("ravi", track_id=1, score=0.9)

        with pytest.raises(ValueError, match="match decision"):
            repo.delete_permanently("ravi")

        assert repo.get_person("ravi") is not None, "the refusal must be total"
        assert len(repo.decisions_for("ravi")) == 1

    def test_the_refusal_names_the_alternative(self, repo) -> None:
        """A dead end is not an answer; erasure is what they actually want."""
        enroll(repo)
        repo.record_match("ravi", track_id=1, score=0.9)
        with pytest.raises(ValueError, match="[Ee]rase their biometrics"):
            repo.delete_permanently("ravi")

    def test_deleting_an_unknown_person_reports_no_change(self, repo) -> None:
        assert not repo.delete_permanently("nobody")

    def test_decision_count_drives_the_decision(self, repo) -> None:
        enroll(repo)
        assert repo.decision_count("ravi") == 0
        repo.record_match("ravi", track_id=1, score=0.9)
        assert repo.decision_count("ravi") == 1


class TestHumanConfirmation:
    """Section 8: no automated action on a match alone."""

    def test_matches_are_always_created_pending(self, repo) -> None:
        enroll(repo)
        decision = repo.record_match("ravi", track_id=3, score=0.91)
        assert decision.status is DecisionStatus.PENDING
        assert not decision.is_actionable

    def test_a_pending_decision_is_never_actionable(self, repo) -> None:
        enroll(repo)
        repo.record_match("ravi", track_id=3, score=0.99)
        assert repo.actionable_decisions() == []

    def test_confirmation_requires_a_named_operator(self, repo) -> None:
        enroll(repo)
        decision = repo.record_match("ravi", track_id=1, score=0.9)
        for blank in ("", "   "):
            with pytest.raises(ValueError, match="name the operator"):
                repo.review(decision.id, operator=blank, verdict=DecisionStatus.CONFIRMED)

    def test_only_a_human_verdict_is_accepted(self, repo) -> None:
        """The system cannot mark its own decision expired or pending."""
        enroll(repo)
        decision = repo.record_match("ravi", track_id=1, score=0.9)
        for verdict in (DecisionStatus.PENDING, DecisionStatus.EXPIRED):
            with pytest.raises(ValueError, match="CONFIRMED or REJECTED"):
                repo.review(decision.id, operator="alice", verdict=verdict)

    def test_confirmation_makes_it_actionable(self, repo) -> None:
        enroll(repo)
        decision = repo.record_match("ravi", track_id=1, score=0.9)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        assert len(repo.actionable_decisions()) == 1

    def test_rejection_does_not(self, repo) -> None:
        enroll(repo)
        decision = repo.record_match("ravi", track_id=1, score=0.9)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.REJECTED)
        assert repo.actionable_decisions() == []

    def test_reviews_append_rather_than_overwrite(self, repo) -> None:
        """A reviewer changing their mind must leave both judgements visible."""
        enroll(repo)
        decision = repo.record_match("ravi", track_id=1, score=0.9)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        repo.review(
            decision.id, operator="bob", verdict=DecisionStatus.REJECTED,
            reason="wrong person on review",
        )

        assert len(decision.reviews) == 2
        assert [r.operator for r in decision.reviews] == ["alice", "bob"]
        assert decision.status is DecisionStatus.REJECTED

    def test_the_weights_that_drove_a_match_are_recorded(self, repo) -> None:
        """The explainability record -- what the system actually relied on."""
        import json

        enroll(repo)
        decision = repo.record_match(
            "ravi",
            track_id=1,
            score=0.88,
            strategy="quality_weighted",
            weights={Modality.FACE: 0.46, Modality.REID: 0.54},
        )
        stored = json.loads(decision.weights_json)
        assert stored == {"face": 0.46, "reid": 0.54}

    def test_matching_an_unknown_person_is_rejected(self, repo) -> None:
        with pytest.raises(ValueError, match="Unknown person"):
            repo.record_match("nobody", track_id=1, score=0.9)


class TestAPI:
    def _enroll(self, client):
        session = session_factory(client.engine)()
        repo = WatchlistRepository(session)
        enroll(repo)
        session.close()

    def test_health(self, client) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_watchlist_is_empty_initially(self, client) -> None:
        assert client.get("/api/watchlist").json() == []

    def test_lists_an_enrolled_person_without_leaking_vectors(self, client) -> None:
        """Templates are described, never returned.

        An API that hands out biometric vectors is a biometric-vector leak
        with extra steps.
        """
        self._enroll(client)
        body = client.get("/api/watchlist").json()
        assert len(body) == 1
        assert body[0]["person_id"] == "ravi"

        payload = client.get("/api/watchlist/ravi").text
        assert "payload" not in payload
        assert "vector" not in payload
        modalities = {t["modality"] for t in body[0]["templates"]}
        assert modalities == {"face", "reid"}

    def test_unknown_person_is_404(self, client) -> None:
        assert client.get("/api/watchlist/nobody").status_code == 404

    def test_patch_updates_descriptive_fields(self, client) -> None:
        self._enroll(client)
        response = client.patch(
            "/api/watchlist/ravi", json={"display_name": "Ravi K", "notes": "n"}
        )
        assert response.status_code == 200
        assert response.json()["display_name"] == "Ravi K"

    def test_delete_retires_rather_than_removing(self, client) -> None:
        self._enroll(client)
        # No operator query parameter any more: it comes from the session.
        assert client.delete("/api/watchlist/ravi").status_code == 200
        assert client.get("/api/watchlist").json() == []
        assert len(client.get("/api/watchlist?include_retired=true").json()) == 1

    def test_restore_puts_a_retired_person_back(self, client) -> None:
        self._enroll(client)
        client.delete("/api/watchlist/ravi")
        assert client.get("/api/watchlist").json() == []

        assert client.post("/api/watchlist/ravi/restore").status_code == 200
        assert len(client.get("/api/watchlist").json()) == 1

    def test_restoring_someone_active_is_a_404(self, client) -> None:
        self._enroll(client)
        assert client.post("/api/watchlist/ravi/restore").status_code == 404

    def test_erasing_biometrics_empties_the_record_but_keeps_it(self, client) -> None:
        self._enroll(client)
        response = client.delete("/api/watchlist/ravi/biometrics")
        assert response.status_code == 200
        assert response.json()["templates_destroyed"] == 2

        person = client.get("/api/watchlist/ravi").json()
        assert person["templates"] == []
        assert person["has_reference"] is False

    def test_erasing_an_unknown_person_is_a_404(self, client) -> None:
        assert client.delete("/api/watchlist/nobody/biometrics").status_code == 404

    def test_an_unmatched_person_can_be_deleted_permanently(self, client) -> None:
        self._enroll(client)
        assert client.delete("/api/watchlist/ravi/permanently").status_code == 200
        assert client.get("/api/watchlist?include_retired=true").json() == []

    def test_deleting_someone_with_decisions_is_a_409(self, client) -> None:
        """Not a 500 and not a silent partial delete: the review queue would be
        left pointing at nobody."""
        self._enroll(client)
        session = session_factory(client.engine)()
        WatchlistRepository(session).record_match("ravi", track_id=1, score=0.9)
        session.close()

        response = client.delete("/api/watchlist/ravi/permanently")
        assert response.status_code == 409
        assert "erase their biometrics" in response.json()["detail"].lower()
        assert len(client.get("/api/watchlist").json()) == 1

    def test_editing_cannot_blank_a_name(self, client) -> None:
        """Registration insists on a name; editing must not undo that and
        leave a record nobody can recognise in a review queue."""
        self._enroll(client)
        assert client.patch(
            "/api/watchlist/ravi", json={"display_name": ""}
        ).status_code == 422
        assert client.get("/api/watchlist/ravi").json()["display_name"] == "Ravi Kumar"

    def test_person_id_cannot_be_edited(self, client) -> None:
        """It is the identity every decision was filed under."""
        self._enroll(client)
        client.patch("/api/watchlist/ravi", json={"person_id": "someone-else"})
        assert client.get("/api/watchlist/ravi").status_code == 200
        assert client.get("/api/watchlist/someone-else").status_code == 404

    def test_review_requires_a_valid_verdict(self, client) -> None:
        """There is no route that confirms a match without a human verdict."""
        self._enroll(client)
        session = session_factory(client.engine)()
        decision = WatchlistRepository(session).record_match(
            "ravi", track_id=1, score=0.9, strategy="average",
            weights={Modality.FACE: 1.0},
        )
        decision_id = decision.id
        session.close()

        # "pending" is not a human verdict and the schema rejects it.
        bad = client.post(
            f"/api/decisions/{decision_id}/review", json={"verdict": "pending"}
        )
        assert bad.status_code == 422

        missing = client.post(f"/api/decisions/{decision_id}/review", json={})
        assert missing.status_code == 422

    def test_reviewing_requires_being_signed_in(self, anon_client) -> None:
        """The guardrail is only real if the door is locked (SEC-01)."""
        response = anon_client.post(
            "/api/decisions/1/review", json={"verdict": "confirmed"}
        )
        assert response.status_code == 401

    def test_the_operator_comes_from_the_session_not_the_body(self, client) -> None:
        """The whole point of SEC-01.

        The audit trail used to record whatever name the request supplied, so
        every entry was an unverified claim and the trail was repudiable. A
        caller can still put an `operator` field in the body; it is ignored.
        """
        self._enroll(client)
        session = session_factory(client.engine)()
        decision_id = (
            WatchlistRepository(session)
            .record_match("ravi", track_id=1, score=0.9, weights={Modality.FACE: 1.0})
            .id
        )
        session.close()

        response = client.post(
            f"/api/decisions/{decision_id}/review",
            json={
                "verdict": "confirmed",
                "operator": "somebody-else",  # ignored
                "reason": "spoof attempt",
            },
        )
        assert response.status_code == 200
        assert response.json()["reviews"][0]["operator"] == "tester"

    def test_full_review_flow(self, client) -> None:
        self._enroll(client)
        session = session_factory(client.engine)()
        decision_id = (
            WatchlistRepository(session)
            .record_match(
                "ravi", track_id=7, score=0.93, strategy="quality_weighted",
                weights={Modality.FACE: 0.46, Modality.REID: 0.54},
            )
            .id
        )
        session.close()

        pending = client.get("/api/decisions").json()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"
        assert pending[0]["is_actionable"] is False
        assert pending[0]["weights"] == {"face": 0.46, "reid": 0.54}

        # Nothing is actionable before a human looks at it.
        assert client.get("/api/alerts").json() == []

        confirmed = client.post(
            f"/api/decisions/{decision_id}/review",
            json={"verdict": "confirmed", "reason": "clear face"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["is_actionable"] is True
        # The signed-in operator, not anything the body said.
        assert confirmed.json()["reviews"][0]["operator"] == "tester"

        alerts = client.get("/api/alerts").json()
        assert len(alerts) == 1
        assert alerts[0]["id"] == decision_id

    def test_audit_trail_records_enrollment_and_review(self, client) -> None:
        self._enroll(client)
        events = client.get("/api/audit").json()
        assert any(e["kind"] == "enroll" for e in events)

    def test_openapi_schema_builds(self, client) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestUnusedSignals:
    """Why a signal did not count is part of the decision record."""

    def test_reasons_are_stored_and_served(self, repo) -> None:
        from app.api.routes import DecisionOut

        enroll(repo)
        repo.record_match(
            "ravi",
            track_id=1,
            score=0.9,
            weights={Modality.REID: 1.0},
            unused={
                Modality.GAIT: "no complete gait cycle detected",
                Modality.FACE: "no usable faces",
            },
        )
        served = DecisionOut.of(repo.decisions_for("ravi")[0])
        assert served.unused == {
            "gait": "no complete gait cycle detected",
            "face": "no usable faces",
        }

    def test_a_decision_recorded_without_reasons_serves_none(self, repo) -> None:
        from app.api.routes import DecisionOut

        enroll(repo)
        repo.record_match("ravi", track_id=1, score=0.9)
        assert DecisionOut.of(repo.decisions_for("ravi")[0]).unused == {}

    def test_an_existing_database_gains_the_column(self, tmp_path) -> None:
        """create_all never alters a table that already exists."""
        import sqlite3

        from sqlalchemy import inspect

        from app.db.repository import create_schema, make_engine

        path = tmp_path / "old.db"
        engine = make_engine(f"sqlite:///{path}")
        create_schema(engine)
        engine.dispose()
        connection = sqlite3.connect(path)
        connection.execute("ALTER TABLE match_decisions DROP COLUMN unused_json")
        connection.commit()
        connection.close()

        engine = make_engine(f"sqlite:///{path}")
        columns = {c["name"] for c in inspect(engine).get_columns("match_decisions")}
        assert "unused_json" not in columns
        create_schema(engine)
        columns = {c["name"] for c in inspect(engine).get_columns("match_decisions")}
        assert "unused_json" in columns
        engine.dispose()
