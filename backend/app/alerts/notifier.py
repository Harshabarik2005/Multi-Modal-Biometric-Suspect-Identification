"""Alerting (Phase 9) -- Twilio / SMTP.

**Alerts fire only on human-confirmed decisions.** `AlertDispatcher` reads the
same confirmed-only path the API's `/alerts` endpoint uses and refuses anything
pending, so a match the model raised on its own can never reach a phone. That
gate is checked twice: once when selecting decisions and once per decision
immediately before sending, because this is the one place in the system where a
mistake leaves the building.

Two more deliberate choices:

**Alerts carry the reasoning, not just the score.** A notification saying
"match found, 0.91" invites exactly the unexamined trust the review step exists
to prevent. Every alert states which modalities drove the score and warns when
it rested mostly on clothing.

**Nothing is sent twice.** Dispatch records an audit event per delivery and
skips decisions already alerted, so a re-run does not re-notify. A duplicate
alert reads as a second sighting.

Dry-run is the default. Sending requires explicit configuration *and* an
explicit flag, because the failure mode of accidentally messaging a real
contact list during testing is not recoverable.
"""

from __future__ import annotations

import json
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.message import EmailMessage

from app.core.logging import get_logger
from app.db.models import DecisionStatus, MatchDecision
from app.db.repository import WatchlistRepository

logger = get_logger(__name__)

ALERT_SENT = "alert_sent"


@dataclass
class Alert:
    """A rendered notification, ready to send."""

    decision_id: int
    person_id: str
    subject: str
    body: str
    appearance_share: float = 0.0


class AlertError(RuntimeError):
    """A transport failed. Never raised for a refused (unconfirmed) decision."""


# -- rendering -------------------------------------------------------------

def _weights_summary(weights: dict[str, float]) -> str:
    if not weights:
        return "  (no per-modality breakdown was recorded)"
    total = sum(weights.values()) or 1.0
    names = {"face": "Face", "gait": "Gait", "reid": "Appearance"}
    lines = []
    for modality, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        share = weight / total * 100
        lines.append(f"  {names.get(modality, modality):<11} {share:>5.0f}%")
    return "\n".join(lines)


def render(decision: MatchDecision) -> Alert:
    """Turn a confirmed decision into a notification.

    Includes who confirmed it. An alert that does not name its reviewer invites
    the reader to treat it as machine truth.
    """
    weights = json.loads(decision.weights_json or "{}")
    total = sum(weights.values()) or 1.0
    appearance = weights.get("reid", 0.0) / total

    reviewer = "unknown"
    for review in decision.reviews:
        if review.verdict is DecisionStatus.CONFIRMED:
            reviewer = review.operator

    caution = ""
    if appearance >= 0.5:
        caution = (
            f"\nCAUTION: {appearance * 100:.0f}% of this identification rests on "
            "appearance\n(build and clothing), which is the weakest of the three "
            "signals and\ngoes stale as people change clothes. Weigh it accordingly."
        )

    body = f"""Confirmed identification: {decision.person.display_name}

Person   : {decision.person.person_id}
Camera   : {decision.camera_id or "unspecified"}
Track    : {decision.track_id} (frame {decision.frame_index})
Score    : {decision.score:.3f}  [{decision.strategy or "unknown strategy"}]
Raised   : {decision.created_at.isoformat()}
Confirmed by: {reviewer}

What drove this identification:
{_weights_summary(weights)}
{caution}

This was confirmed by a person, not decided by the system. If that
confirmation was made in error, correct it in the review console -- the
original decision and every review of it are retained.
"""
    return Alert(
        decision_id=decision.id,
        person_id=decision.person.person_id,
        subject=f"[Faceless FRS] Confirmed: {decision.person.display_name}",
        body=body,
        appearance_share=appearance,
    )


# -- transports ------------------------------------------------------------

class Notifier(ABC):
    """One way of delivering an alert."""

    name: str

    @abstractmethod
    def send(self, alert: Alert) -> str:
        """Deliver. Returns a transport reference; raises `AlertError` on failure."""


class ConsoleNotifier(Notifier):
    """Prints the alert. The default, and what dry-run uses."""

    name = "console"

    def send(self, alert: Alert) -> str:
        print("\n" + "=" * 70)
        print(alert.subject)
        print("=" * 70)
        print(alert.body)
        return f"console:{alert.decision_id}"


class SMTPNotifier(Notifier):
    """Email over SMTP."""

    name = "smtp"

    def __init__(
        self,
        host: str,
        port: int = 587,
        username: str = "",
        password: str = "",
        sender: str = "",
        recipients: list[str] | None = None,
        use_tls: bool = True,
        timeout: int = 20,
    ) -> None:
        if not host:
            raise ValueError("SMTP host is required.")
        if not recipients:
            raise ValueError("At least one recipient is required.")
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender or username
        self.recipients = recipients
        self.use_tls = use_tls
        self.timeout = timeout

    def send(self, alert: Alert) -> str:
        message = EmailMessage()
        message["Subject"] = alert.subject
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(alert.body)

        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(message)
        except Exception as exc:  # noqa: BLE001
            raise AlertError(f"SMTP delivery failed: {exc}") from exc
        return f"smtp:{','.join(self.recipients)}"


class TwilioNotifier(Notifier):
    """SMS via Twilio.

    Sends a deliberately short summary with a pointer to the console. An SMS
    cannot carry the full breakdown, and truncating it would leave the reader
    with a score and no context -- which is worse than making them open the
    review console.
    """

    name = "twilio"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_numbers: list[str] | None = None,
        console_url: str = "",
    ) -> None:
        if not (account_sid and auth_token and from_number):
            raise ValueError("Twilio needs account_sid, auth_token and from_number.")
        if not to_numbers:
            raise ValueError("At least one recipient number is required.")
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
        self.console_url = console_url

    def send(self, alert: Alert) -> str:
        try:
            from twilio.rest import Client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AlertError(
                "twilio is not installed. `pip install twilio` to use SMS alerts."
            ) from exc

        caution = " Mostly appearance-based." if alert.appearance_share >= 0.5 else ""
        text = (
            f"Faceless FRS: confirmed identification of {alert.person_id}."
            f"{caution} Full breakdown: {self.console_url or 'review console'}"
        )

        try:
            client = Client(self.account_sid, self.auth_token)
            sids = [
                client.messages.create(body=text, from_=self.from_number, to=number).sid
                for number in self.to_numbers
            ]
        except Exception as exc:  # noqa: BLE001
            raise AlertError(f"Twilio delivery failed: {exc}") from exc
        return f"twilio:{','.join(sids)}"


# -- dispatch --------------------------------------------------------------

@dataclass
class DispatchReport:
    sent: list[int] = field(default_factory=list)
    skipped_already_sent: list[int] = field(default_factory=list)
    refused_unconfirmed: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    dry_run: bool = True

    def summary_lines(self) -> list[str]:
        # ASCII only: Windows consoles default to cp1252 and mangle em-dashes.
        mode = "DRY RUN - nothing was delivered" if self.dry_run else "LIVE"
        lines = [
            f"mode                : {mode}",
            f"sent                : {len(self.sent)}",
            f"already alerted     : {len(self.skipped_already_sent)}",
        ]
        if self.refused_unconfirmed:
            lines.append(
                f"REFUSED (unconfirmed): {len(self.refused_unconfirmed)} "
                f"{self.refused_unconfirmed}"
            )
        if self.failed:
            lines.append(f"failed              : {len(self.failed)}")
            for decision_id, error in self.failed:
                lines.append(f"  decision {decision_id}: {error}")
        return lines


class AlertDispatcher:
    """Sends alerts for confirmed decisions, exactly once each."""

    def __init__(
        self,
        repo: WatchlistRepository,
        notifier: Notifier | None = None,
        dry_run: bool = True,
    ) -> None:
        self.repo = repo
        self.notifier = notifier or ConsoleNotifier()
        self.dry_run = dry_run

    def already_sent(self) -> set[int]:
        """Decision ids that have already been alerted, from the audit trail."""
        sent: set[int] = set()
        for event in self.repo.audit_trail(limit=5000):
            if event.kind != ALERT_SENT:
                continue
            try:
                decision_id = json.loads(event.detail_json or "{}").get("decision_id")
            except json.JSONDecodeError:
                continue
            if decision_id is not None:
                sent.add(int(decision_id))
        return sent

    def dispatch(self, limit: int = 100) -> DispatchReport:
        report = DispatchReport(dry_run=self.dry_run)
        sent_already = self.already_sent()

        for decision in self.repo.actionable_decisions(limit=limit):
            # Checked again per decision, not just in the query. This is the
            # one place where a mistake leaves the building.
            if not decision.is_actionable:
                report.refused_unconfirmed.append(decision.id)
                logger.error(
                    "Refusing to alert on decision %d: status is %s, not confirmed.",
                    decision.id,
                    decision.status.value,
                )
                continue

            if decision.id in sent_already:
                report.skipped_already_sent.append(decision.id)
                continue

            alert = render(decision)

            if self.dry_run:
                ConsoleNotifier().send(alert)
                report.sent.append(decision.id)
                continue

            try:
                reference = self.notifier.send(alert)
            except AlertError as exc:
                report.failed.append((decision.id, str(exc)))
                logger.error("Alert for decision %d failed: %s", decision.id, exc)
                continue

            # Recorded in the same audit trail as everything else, so "who was
            # told, and when" is answerable alongside "who confirmed it".
            self.repo.log(
                ALERT_SENT,
                actor=self.notifier.name,
                subject=decision.person.person_id,
                detail={
                    "decision_id": decision.id,
                    "transport": self.notifier.name,
                    "reference": reference,
                    "score": decision.score,
                },
            )
            self.repo.session.commit()
            report.sent.append(decision.id)

        return report
