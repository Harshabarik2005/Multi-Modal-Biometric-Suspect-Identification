"""Tests for alerting (Phase 9).

The tests that matter here are the refusal ones. Everything else in the system
that goes wrong can be corrected in the review console; an alert that has been
sent cannot be unsent, so "never fires on an unconfirmed match" has to be
verified rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.alerts.notifier import (
    ALERT_SENT,
    Alert,
    AlertDispatcher,
    AlertError,
    ConsoleNotifier,
    Notifier,
    SMTPNotifier,
    TwilioNotifier,
    render,
)
from app.core.types import Modality, ModalityEmbedding, l2_normalize
from app.db.models import DecisionStatus
from app.db.repository import (
    WatchlistRepository,
    create_schema,
    make_engine,
    session_factory,
)


class RecordingNotifier(Notifier):
    """Captures what it was asked to send instead of sending it."""

    name = "recording"

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[Alert] = []
        self.fail = fail

    def send(self, alert: Alert) -> str:
        if self.fail:
            raise AlertError("transport exploded")
        self.sent.append(alert)
        return f"recording:{alert.decision_id}"


@pytest.fixture
def repo():
    engine = make_engine("sqlite:///:memory:")
    create_schema(engine)
    session = session_factory(engine)()
    repository = WatchlistRepository(session)
    repository.enroll(
        "ravi",
        "Ravi Kumar",
        {
            Modality.FACE: ModalityEmbedding(
                Modality.FACE,
                l2_normalize(np.arange(8, dtype=np.float32)),
                0.8,
            )
        },
    )
    yield repository
    session.close()


_DEFAULT_WEIGHTS = {Modality.FACE: 0.46, Modality.REID: 0.54}


def a_match(repo, score: float = 0.91, weights=None):
    # `weights if weights is not None`, not `weights or ...`: an explicitly
    # empty dict is a meaningful case (no breakdown was recorded) and `or`
    # would silently replace it with the defaults.
    return repo.record_match(
        "ravi",
        track_id=1,
        score=score,
        strategy="quality_weighted",
        weights=_DEFAULT_WEIGHTS if weights is None else weights,
        camera_id="cam-1",
        frame_index=42,
    )


class TestConfirmationGate:
    """Alerts fire only on human-confirmed decisions. Nothing else."""

    def test_pending_decision_never_alerts(self, repo) -> None:
        a_match(repo)
        notifier = RecordingNotifier()
        report = AlertDispatcher(repo, notifier, dry_run=False).dispatch()

        assert notifier.sent == []
        assert report.sent == []

    def test_rejected_decision_never_alerts(self, repo) -> None:
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.REJECTED)

        notifier = RecordingNotifier()
        AlertDispatcher(repo, notifier, dry_run=False).dispatch()
        assert notifier.sent == []

    def test_confirmed_decision_alerts(self, repo) -> None:
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)

        notifier = RecordingNotifier()
        report = AlertDispatcher(repo, notifier, dry_run=False).dispatch()

        assert len(notifier.sent) == 1
        assert notifier.sent[0].person_id == "ravi"
        assert report.sent == [decision.id]

    def test_a_confirmation_later_reversed_stops_alerting(self, repo) -> None:
        """Reviews append; the latest verdict governs what may be acted on."""
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        repo.review(
            decision.id, operator="bob", verdict=DecisionStatus.REJECTED,
            reason="wrong person",
        )

        notifier = RecordingNotifier()
        AlertDispatcher(repo, notifier, dry_run=False).dispatch()
        assert notifier.sent == []


class TestNoDuplicates:
    def test_a_second_run_does_not_re_notify(self, repo) -> None:
        """A duplicate alert reads as a second sighting."""
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)

        notifier = RecordingNotifier()
        dispatcher = AlertDispatcher(repo, notifier, dry_run=False)

        first = dispatcher.dispatch()
        second = dispatcher.dispatch()

        assert len(notifier.sent) == 1
        assert first.sent == [decision.id]
        assert second.skipped_already_sent == [decision.id]

    def test_delivery_is_recorded_in_the_audit_trail(self, repo) -> None:
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        AlertDispatcher(repo, RecordingNotifier(), dry_run=False).dispatch()

        events = [e for e in repo.audit_trail() if e.kind == ALERT_SENT]
        assert len(events) == 1
        assert events[0].subject == "ravi"

    def test_a_failed_delivery_is_not_marked_sent(self, repo) -> None:
        """Otherwise a transient outage would silently swallow the alert."""
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)

        failing = RecordingNotifier(fail=True)
        report = AlertDispatcher(repo, failing, dry_run=False).dispatch()

        assert report.sent == []
        assert len(report.failed) == 1
        assert not [e for e in repo.audit_trail() if e.kind == ALERT_SENT]

        # A later run with a working transport still delivers it.
        working = RecordingNotifier()
        AlertDispatcher(repo, working, dry_run=False).dispatch()
        assert len(working.sent) == 1


class TestDryRun:
    def test_dry_run_delivers_nothing(self, repo) -> None:
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)

        notifier = RecordingNotifier()
        report = AlertDispatcher(repo, notifier, dry_run=True).dispatch()

        assert notifier.sent == []
        assert report.dry_run is True
        assert report.sent == [decision.id]  # would have been sent

    def test_dry_run_does_not_mark_anything_as_alerted(self, repo) -> None:
        """A dry run must not suppress the real alert afterwards."""
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)

        AlertDispatcher(repo, RecordingNotifier(), dry_run=True).dispatch()
        notifier = RecordingNotifier()
        AlertDispatcher(repo, notifier, dry_run=False).dispatch()
        assert len(notifier.sent) == 1

    def test_dispatcher_defaults_to_dry_run(self, repo) -> None:
        assert AlertDispatcher(repo).dry_run is True


class TestRendering:
    def test_alert_names_the_confirming_operator(self, repo) -> None:
        """An alert that does not name its reviewer reads as machine truth."""
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        body = render(decision).body
        assert "Confirmed by: alice" in body
        assert "confirmed by a person, not decided by the system" in body

    def test_alert_carries_the_modality_breakdown(self, repo) -> None:
        decision = a_match(repo)
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        body = render(decision).body
        assert "Face" in body and "Appearance" in body
        assert "54%" in body

    def test_appearance_heavy_matches_carry_a_caution(self, repo) -> None:
        decision = a_match(repo, weights={Modality.FACE: 0.2, Modality.REID: 0.8})
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        alert = render(decision)
        assert alert.appearance_share == pytest.approx(0.8)
        assert "CAUTION" in alert.body

    def test_face_driven_matches_carry_no_caution(self, repo) -> None:
        decision = a_match(repo, weights={Modality.FACE: 0.9, Modality.REID: 0.1})
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        assert "CAUTION" not in render(decision).body

    def test_missing_weights_do_not_crash_rendering(self, repo) -> None:
        decision = a_match(repo, weights={})
        repo.review(decision.id, operator="alice", verdict=DecisionStatus.CONFIRMED)
        assert "no per-modality breakdown" in render(decision).body


class TestTransportConfiguration:
    def test_smtp_requires_a_host(self) -> None:
        with pytest.raises(ValueError, match="host"):
            SMTPNotifier(host="", recipients=["a@example.com"])

    def test_smtp_requires_recipients(self) -> None:
        with pytest.raises(ValueError, match="recipient"):
            SMTPNotifier(host="smtp.example.com", recipients=[])

    def test_twilio_requires_credentials(self) -> None:
        with pytest.raises(ValueError, match="account_sid"):
            TwilioNotifier("", "", "", ["+15550000000"])

    def test_twilio_requires_recipients(self) -> None:
        with pytest.raises(ValueError, match="recipient"):
            TwilioNotifier("sid", "token", "+15551111111", [])

    def test_console_notifier_always_works(self, capsys) -> None:
        reference = ConsoleNotifier().send(
            Alert(1, "ravi", "subject", "body", 0.3)
        )
        assert reference == "console:1"
        assert "subject" in capsys.readouterr().out
