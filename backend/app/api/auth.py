"""Authentication (SEC-01).

Every endpoint was open. Anyone who could reach the port could read the
watchlist and the audit trail, enrol people, run scans, and **confirm a
match** -- the one act that makes an identification actionable and can trigger
an alert. The operator name was typed into a text box and sent as ordinary
request data, so the audit trail recorded unverified, self-asserted names and
was entirely repudiable.

That is the finding that makes the others reachable, and it is the one that
made the build plan's guardrails unenforceable: "a named human confirms" means
nothing if the name is whatever the request says it is.

What this provides
------------------
* Operator accounts with scrypt-hashed passwords, stored in the database.
* `POST /api/auth/login` exchanges a password for a signed, expiring token.
* `require_operator` -- a dependency every route depends on.
* **The operator identity on a review comes from the authenticated principal,
  never from the request body.** That is the whole point: a decision now
  records who was actually logged in.

Why stdlib rather than a framework
----------------------------------
`hashlib.scrypt` and `hmac` are in the standard library and are the right
primitives here. Adding passlib and python-jose for one login route would be
more code to audit, not less. Tokens are signed rather than stored, so there is
no session table to keep consistent; the cost is that a token cannot be
revoked before it expires, which is noted below and is why they are short-lived.

What this is NOT
----------------
Single-factor, no password policy, no lockout, no refresh tokens, no per-route
authorisation beyond "is a valid operator". It is the difference between
"anyone on the network" and "someone who knows a password", which is the gap
the audit identified. A deployment handling real biometric data needs more, and
`docs/AUDIT.md` should stay open on that point.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Operator

logger = get_logger(__name__)

AUTH_SECRET_ENV = "FRS_AUTH_SECRET"

#: scrypt parameters. n=2**14 keeps login around a tenth of a second on this
#: hardware -- slow enough to make guessing expensive, fast enough not to be a
#: denial-of-service lever itself.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LENGTH = 32

#: Tokens are signed, not stored, so they cannot be revoked before expiry.
#: Eight hours is one shift.
TOKEN_TTL_SECONDS = 8 * 60 * 60

bearer_scheme = HTTPBearer(auto_error=False)


# -- password hashing ------------------------------------------------------

def hash_password(password: str) -> tuple[str, str]:
    """Return `(salt_hex, hash_hex)` for a new password."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LENGTH,
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hex: str) -> bool:
    """Constant-time password check.

    `compare_digest` rather than `==`: a short-circuiting comparison leaks how
    much of the hash matched through timing.
    """
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False

    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LENGTH,
    )
    return hmac.compare_digest(digest.hex(), expected_hex)


# -- token signing ---------------------------------------------------------

def _secret() -> bytes:
    """The signing key.

    Taken from the environment. Without one, a random key is generated per
    process: tokens then stop working across a restart, which is inconvenient
    but safe. A hardcoded fallback would be neither.
    """
    configured = os.environ.get(AUTH_SECRET_ENV, "").strip()
    if configured:
        return configured.encode("utf-8")

    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_bytes(32)
        logger.warning(
            "%s is not set, so a random signing key was generated for this "
            "process. Everyone will be logged out when it restarts. Set %s to "
            "a persistent value.",
            AUTH_SECRET_ENV,
            AUTH_SECRET_ENV,
        )
    return _EPHEMERAL_SECRET


_EPHEMERAL_SECRET: bytes | None = None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def issue_token(username: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    payload = json.dumps(
        {"sub": username, "exp": int(time.time()) + ttl_seconds},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}"


def read_token(token: str) -> str | None:
    """Return the username a valid token names, or None.

    Returns None for anything wrong -- bad shape, bad signature, expired --
    rather than distinguishing them, because the difference is only useful to
    someone probing.
    """
    try:
        payload_part, signature_part = token.split(".", 1)
        payload = _unb64(payload_part)
        signature = _unb64(signature_part)
    except (ValueError, AttributeError):
        return None

    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        claims = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(claims, dict) or "sub" not in claims:
        return None
    if int(claims.get("exp", 0)) < time.time():
        return None
    return str(claims["sub"])


# -- dependencies ----------------------------------------------------------

def get_auth_session() -> Session:  # pragma: no cover - set by the app factory
    raise RuntimeError("Auth session dependency was not configured.")


#: Who a review is recorded against when sign-in is switched off.
DEMO_USERNAME = "demo"
DEMO_DISPLAY_NAME = "Demo — no sign-in"


def demo_operator(session: Session) -> Operator:
    """The stand-in principal used while `demo_mode` is on.

    Created on first use rather than seeded, so a database that has never run
    in demo mode does not carry a passwordless account in it. The password
    hash is random and nothing can log in as this account: it exists to be
    written into decision reviews, not to authenticate.
    """
    operator = (
        session.query(Operator).filter(Operator.username == DEMO_USERNAME).first()
    )
    if operator is not None:
        return operator

    salt, digest = hash_password(secrets.token_urlsafe(32))
    operator = Operator(
        username=DEMO_USERNAME,
        display_name=DEMO_DISPLAY_NAME,
        password_salt=salt,
        password_hash=digest,
        is_admin=True,
    )
    session.add(operator)
    session.commit()
    logger.warning(
        "Demo mode: created the %r operator. Every action is recorded against "
        "it, and anyone who can reach this port can take any action.",
        DEMO_USERNAME,
    )
    return operator


def require_operator(
    request: Request,
    session: Annotated[Session, Depends(get_auth_session)],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> Operator:
    """Resolve the caller, or refuse.

    The returned `Operator` is the ONLY acceptable source of identity for a
    review. Anything the request body claims about who is acting is decoration.
    """
    if get_settings().demo_mode:
        # Prototype mode: no sign-in, but still a named operator, so a review
        # is attributable to *something* and the audit trail keeps its shape.
        # The name is deliberately unmistakable -- a trail full of "demo"
        # cannot later be mistaken for a record of who actually decided.
        operator = demo_operator(session)
        request.state.operator = operator
        return operator

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Not signed in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = read_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Session expired or invalid. Sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    operator = session.query(Operator).filter(Operator.username == username).first()
    if operator is None or not operator.is_active:
        # A token outlives a disabled account, so the account is re-checked on
        # every request rather than trusted from the token alone.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "This account is no longer active."
        )

    request.state.operator = operator
    return operator


CurrentOperator = Annotated[Operator, Depends(require_operator)]


# -- account management ----------------------------------------------------

def create_operator(
    session: Session,
    username: str,
    password: str,
    display_name: str = "",
    is_admin: bool = False,
) -> Operator:
    username = (username or "").strip().lower()
    if not username:
        raise ValueError("A username is required.")
    if session.query(Operator).filter(Operator.username == username).first():
        raise ValueError(f"Operator {username!r} already exists.")

    salt, digest = hash_password(password)
    operator = Operator(
        username=username,
        display_name=display_name or username,
        password_salt=salt,
        password_hash=digest,
        is_admin=is_admin,
    )
    session.add(operator)
    session.commit()
    logger.info("Created operator %s", username)
    return operator


def authenticate(session: Session, username: str, password: str) -> Operator | None:
    """Check a username and password. None on any failure.

    A missing account still runs a hash so that "no such user" and "wrong
    password" take the same time, which stops the endpoint being used to
    enumerate accounts.
    """
    operator = (
        session.query(Operator)
        .filter(Operator.username == (username or "").strip().lower())
        .first()
    )

    if operator is None:
        hash_password("dummy-password-to-equalise-timing")
        return None
    if not operator.is_active:
        return None
    if not verify_password(password, operator.password_salt, operator.password_hash):
        return None
    return operator
