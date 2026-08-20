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
from app.db.models import DecisionStatus
from app.db.repository import (
    WatchlistRepository,
    create_schema,
    make_engine,
    session_factory,
)


@pytest.fixture
def repo(monkeypatch):
    monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
    engine = make_engine("sqlite:///:memory:")
    create_schema(engine)
    session = session_factory(engine)()
    yield WatchlistRepository(session)
    session.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("FRS_TEMPLATE_ENCRYPTION_KEY", raising=False)
    from app.api.main import create_app

    engine = make_engine("sqlite:///:memory:")
    app = create_app(engine=engine)
    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client


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

    def test_plaintext_is_flagged_as_unencrypted(self, repo) -> None:
        enroll(repo)
        assert all(not t.encrypted for t in repo.get_person("ravi").templates)


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
        assert client.delete("/api/watchlist/ravi?operator=alice").status_code == 200
        assert client.get("/api/watchlist").json() == []
        assert len(client.get("/api/watchlist?include_retired=true").json()) == 1

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
            f"/api/decisions/{decision_id}/review",
            json={"operator": "alice", "verdict": "pending"},
        )
        assert bad.status_code == 422

        blank = client.post(
            f"/api/decisions/{decision_id}/review",
            json={"operator": "", "verdict": "confirmed"},
        )
        assert blank.status_code == 422

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
            json={"operator": "alice", "verdict": "confirmed", "reason": "clear face"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["is_actionable"] is True
        assert confirmed.json()["reviews"][0]["operator"] == "alice"

        alerts = client.get("/api/alerts").json()
        assert len(alerts) == 1
        assert alerts[0]["id"] == decision_id

    def test_audit_trail_records_enrollment_and_review(self, client) -> None:
        self._enroll(client)
        events = client.get("/api/audit").json()
        assert any(e["kind"] == "enroll" for e in events)

    def test_openapi_schema_builds(self, client) -> None:
        assert client.get("/openapi.json").status_code == 200
