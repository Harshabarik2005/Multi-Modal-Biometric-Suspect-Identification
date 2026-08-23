"""Manage operator accounts (SEC-01).

    python scripts/manage_operators.py --create harsh --admin
    python scripts/manage_operators.py --list
    python scripts/manage_operators.py --disable someone
    python scripts/manage_operators.py --reset-password harsh

There is no self-registration endpoint, deliberately. Accounts are created from
the machine running the service, by someone with shell access to it -- which is
the appropriate bar for a system that can confirm an identification of a real
person.

The password is read from a prompt, never from an argument: anything passed on
the command line lands in shell history and in the process list, where any
other local user can read it.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.auth import AUTH_SECRET_ENV, create_operator, hash_password  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.models import Operator  # noqa: E402
from app.db.repository import (  # noqa: E402
    create_schema,
    make_engine,
    session_factory,
)


def read_password(prompt: str = "Password: ") -> str:
    first = getpass.getpass(prompt)
    if len(first) < 8:
        raise SystemExit("Password must be at least 8 characters.")
    if first != getpass.getpass("Confirm: "):
        raise SystemExit("Passwords did not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--create", metavar="USERNAME")
    parser.add_argument("--name", default="", help="Display name for --create.")
    parser.add_argument("--admin", action="store_true", help="Make --create an admin.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--disable", metavar="USERNAME")
    parser.add_argument("--enable", metavar="USERNAME")
    parser.add_argument("--reset-password", metavar="USERNAME")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging("WARNING")

    database_url = (
        args.db_url
        or settings.database_url
        or f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}"
    )
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(database_url)
    create_schema(engine)
    session = session_factory(engine)()

    try:
        if args.create:
            password = read_password()
            operator = create_operator(
                session,
                args.create,
                password,
                display_name=args.name,
                is_admin=args.admin,
            )
            print(
                f"Created {operator.username}"
                + (" (admin)" if operator.is_admin else "")
            )
            import os

            if not os.environ.get(AUTH_SECRET_ENV):
                print(
                    f"\nWARNING: {AUTH_SECRET_ENV} is not set, so tokens are signed\n"
                    "with a key generated fresh on every start -- everyone is logged\n"
                    "out whenever the server restarts. Generate one with:\n"
                    '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            return 0

        if args.list:
            operators = session.query(Operator).order_by(Operator.username).all()
            if not operators:
                print(
                    "No operators. Nobody can sign in.\n"
                    "Create one:  python scripts/manage_operators.py --create <name>"
                )
                return 0
            print(f"{'username':<20} {'name':<24} {'admin':>6} {'active':>7} last login")
            print("-" * 78)
            for operator in operators:
                print(
                    f"{operator.username:<20} {operator.display_name:<24} "
                    f"{'yes' if operator.is_admin else '':>6} "
                    f"{'yes' if operator.is_active else 'NO':>7} "
                    f"{operator.last_login_at.isoformat() if operator.last_login_at else '-'}"
                )
            return 0

        for flag, active in ((args.disable, False), (args.enable, True)):
            if not flag:
                continue
            operator = (
                session.query(Operator)
                .filter(Operator.username == flag.strip().lower())
                .first()
            )
            if operator is None:
                print(f"No such operator: {flag}")
                return 1
            operator.is_active = active
            session.commit()
            print(f"{operator.username} is now {'active' if active else 'disabled'}")
            print(
                "Existing tokens stop working immediately -- the account is "
                "re-checked on every request, not trusted from the token."
            )
            return 0

        if args.reset_password:
            operator = (
                session.query(Operator)
                .filter(Operator.username == args.reset_password.strip().lower())
                .first()
            )
            if operator is None:
                print(f"No such operator: {args.reset_password}")
                return 1
            salt, digest = hash_password(read_password("New password: "))
            operator.password_salt, operator.password_hash = salt, digest
            session.commit()
            print(f"Password reset for {operator.username}")
            return 0

        parser.print_help()
        return 2
    except ValueError as exc:
        print(exc)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
