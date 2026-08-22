"""Send alerts for confirmed identifications (Phase 9).

    python scripts/send_alerts.py                 # dry run: shows what would go
    python scripts/send_alerts.py --send          # actually deliver
    python scripts/send_alerts.py --transport smtp --send

Alerts fire ONLY on decisions a human has confirmed. A pending decision is
invisible to this script, and the dispatcher re-checks each decision's status
immediately before sending -- this is the one place in the system where a
mistake leaves the building.

Dry run is the default and `--send` is required, because accidentally
messaging a real contact list during testing cannot be undone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.alerts.notifier import (  # noqa: E402
    AlertDispatcher,
    ConsoleNotifier,
    SMTPNotifier,
    TwilioNotifier,
)
from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.repository import (  # noqa: E402
    WatchlistRepository,
    create_schema,
    make_engine,
    session_factory,
)


def build_notifier(settings, transport: str):
    cfg = settings.alerts

    if transport == "console":
        return ConsoleNotifier()

    if transport == "smtp":
        recipients = cfg.split(cfg.smtp_recipients)
        if not cfg.smtp_host or not recipients:
            raise SystemExit(
                "SMTP is not configured. Set alerts.smtp_host and\n"
                "alerts.smtp_recipients (see .env.example for the env-var names)."
            )
        return SMTPNotifier(
            host=cfg.smtp_host,
            port=cfg.smtp_port,
            username=cfg.smtp_username,
            password=cfg.smtp_password.get_secret_value(),
            sender=cfg.smtp_sender,
            recipients=recipients,
            use_tls=cfg.smtp_use_tls,
        )

    if transport == "twilio":
        numbers = cfg.split(cfg.twilio_to_numbers)
        if not (
            cfg.twilio_account_sid
            and cfg.twilio_auth_token.get_secret_value()
            and numbers
        ):
            raise SystemExit(
                "Twilio is not configured. Set alerts.twilio_account_sid,\n"
                "twilio_auth_token, twilio_from_number and twilio_to_numbers."
            )
        return TwilioNotifier(
            account_sid=cfg.twilio_account_sid,
            auth_token=cfg.twilio_auth_token.get_secret_value(),
            from_number=cfg.twilio_from_number,
            to_numbers=numbers,
            console_url=cfg.console_url,
        )

    raise SystemExit(f"Unknown transport {transport!r}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transport", default=None, choices=["console", "smtp", "twilio"])
    parser.add_argument(
        "--send", action="store_true",
        help="Actually deliver. Without this it is a dry run.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.logging.level)

    database_url = args.db_url or (
        f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}"
    )
    engine = make_engine(database_url)
    create_schema(engine)
    session = session_factory(engine)()
    repo = WatchlistRepository(session)

    transport = args.transport or settings.alerts.transport
    notifier = build_notifier(settings, transport)

    if not args.send:
        print(
            "DRY RUN — nothing will be delivered. Re-run with --send to "
            f"actually notify via {transport}.\n"
        )

    dispatcher = AlertDispatcher(repo, notifier=notifier, dry_run=not args.send)
    report = dispatcher.dispatch(limit=args.limit)

    print()
    print("=" * 62)
    print("ALERT DISPATCH")
    print("=" * 62)
    for line in report.summary_lines():
        print(line)
    print("=" * 62)

    if not report.sent and not report.skipped_already_sent:
        print(
            "\nNothing to alert on. Alerts fire only on decisions a human has\n"
            "confirmed in the review console."
        )

    session.close()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
