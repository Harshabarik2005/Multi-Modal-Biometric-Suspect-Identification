"""FastAPI routes (Phase 7).

Endpoints for the watchlist, match decisions, and the audit trail.

The shape of this API encodes section 8's guardrails rather than leaving them
to the caller:

* There is **no endpoint that confirms a match automatically**. `/match` only
  ever creates PENDING decisions; moving one to CONFIRMED requires
  `/decisions/{id}/review` with a named operator.
* Every match response carries the per-modality weights that produced it, so
  the dashboard can show *why* it fired.
* Retiring someone from the watchlist does not delete them, because match
  decisions reference them and an audit trail with dangling references is not
  auditable.

Embeddings are not accepted or returned over the wire in raw form: enrollment
takes video, and templates stay encrypted server-side. An API that hands out
biometric vectors is a biometric-vector leak with extra steps.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import CurrentOperator, authenticate, issue_token
from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.reid import NON_REID_WEIGHTS
from app.db.models import DecisionStatus, MatchDecision, Operator, Person
from app.db.repository import WatchlistRepository

logger = get_logger(__name__)
router = APIRouter()


# -- dependency ------------------------------------------------------------

def get_session() -> Session:  # pragma: no cover - replaced by the app factory
    raise RuntimeError("Session dependency was not configured.")


def get_repository(
    session: Annotated[Session, Depends(get_session)],
) -> WatchlistRepository:
    return WatchlistRepository(session)


Repo = Annotated[WatchlistRepository, Depends(get_repository)]


# -- schemas ---------------------------------------------------------------

class TemplateOut(BaseModel):
    modality: str
    dim: int
    quality: float
    frames_used: int
    encrypted: bool


class PersonOut(BaseModel):
    person_id: str
    display_name: str
    notes: str = ""
    source: str = ""
    enrolled_at: str | None = None
    retired_at: str | None = None
    is_active: bool = True
    templates: list[TemplateOut] = Field(default_factory=list)

    @classmethod
    def of(cls, person: Person) -> "PersonOut":
        return cls(
            person_id=person.person_id,
            display_name=person.display_name,
            notes=person.notes,
            source=person.source,
            enrolled_at=person.enrolled_at.isoformat() if person.enrolled_at else None,
            retired_at=person.retired_at.isoformat() if person.retired_at else None,
            is_active=person.is_active,
            templates=[
                TemplateOut(
                    modality=t.modality,
                    dim=t.dim,
                    quality=t.quality,
                    frames_used=t.frames_used,
                    encrypted=t.encrypted,
                )
                for t in person.templates
            ],
        )


class PersonUpdate(BaseModel):
    display_name: str | None = None
    notes: str | None = None


class DecisionOut(BaseModel):
    id: int
    person_id: str
    display_name: str
    camera_id: str
    track_id: int
    frame_index: int
    score: float
    strategy: str
    #: What the system relied on. The explainability record.
    weights: dict[str, float] = Field(default_factory=dict)
    calibrated: dict[str, float] = Field(default_factory=dict)
    status: str
    created_at: str
    reviews: list["ReviewOut"] = Field(default_factory=list)
    #: True only when a human has confirmed. Nothing may act otherwise.
    is_actionable: bool = False

    @classmethod
    def of(cls, decision: MatchDecision) -> "DecisionOut":
        return cls(
            id=decision.id,
            person_id=decision.person.person_id,
            display_name=decision.person.display_name,
            camera_id=decision.camera_id,
            track_id=decision.track_id,
            frame_index=decision.frame_index,
            score=decision.score,
            strategy=decision.strategy,
            weights=json.loads(decision.weights_json or "{}"),
            calibrated=json.loads(decision.calibrated_json or "{}"),
            status=decision.status.value,
            created_at=decision.created_at.isoformat(),
            reviews=[ReviewOut.of(r) for r in decision.reviews],
            is_actionable=decision.is_actionable,
        )


class ReviewOut(BaseModel):
    operator: str
    verdict: str
    reason: str = ""
    created_at: str

    @classmethod
    def of(cls, review) -> "ReviewOut":
        return cls(
            operator=review.operator,
            verdict=review.verdict.value,
            reason=review.reason,
            created_at=review.created_at.isoformat(),
        )


class ReviewIn(BaseModel):
    """A human's verdict.

    Note what is NOT here: the operator. Identity comes from the authenticated
    session, never from the request body. A caller cannot assert who they are.
    """

    verdict: str = Field(pattern="^(confirmed|rejected)$")
    reason: str = Field("", max_length=2000)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginOut(BaseModel):
    token: str
    username: str
    display_name: str
    is_admin: bool
    expires_in_seconds: int


class MeOut(BaseModel):
    username: str
    display_name: str
    is_admin: bool


class AuditOut(BaseModel):
    kind: str
    actor: str
    subject: str
    detail: dict = Field(default_factory=dict)
    created_at: str


DecisionOut.model_rebuild()


# -- auth ------------------------------------------------------------------

@router.post("/auth/login", response_model=LoginOut, tags=["auth"])
def login(
    payload: LoginIn,
    session: Annotated[Session, Depends(get_session)],
) -> LoginOut:
    """Exchange a password for a token.

    Deliberately does not say whether the username or the password was wrong;
    the distinction is only useful for enumerating accounts.
    """
    from app.api.auth import TOKEN_TTL_SECONDS
    from app.db.models import utcnow

    operator = authenticate(session, payload.username, payload.password)
    if operator is None:
        logger.warning("Failed sign-in for %r", payload.username)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Incorrect username or password."
        )

    operator.last_login_at = utcnow()
    session.commit()
    logger.info("Operator %s signed in", operator.username)

    return LoginOut(
        token=issue_token(operator.username),
        username=operator.username,
        display_name=operator.display_name,
        is_admin=operator.is_admin,
        expires_in_seconds=TOKEN_TTL_SECONDS,
    )


@router.get("/auth/me", response_model=MeOut, tags=["auth"])
def whoami(operator: CurrentOperator) -> MeOut:
    return MeOut(
        username=operator.username,
        display_name=operator.display_name,
        is_admin=operator.is_admin,
    )


# -- watchlist -------------------------------------------------------------

@router.get("/watchlist", response_model=list[PersonOut], tags=["watchlist"])
def list_watchlist(
    repo: Repo,
    operator: CurrentOperator,
    include_retired: bool = Query(False, description="Include retired entries."),
) -> list[PersonOut]:
    return [PersonOut.of(p) for p in repo.list_people(include_retired=include_retired)]


@router.get("/watchlist/{person_id}", response_model=PersonOut, tags=["watchlist"])
def get_person(person_id: str, repo: Repo, operator: CurrentOperator) -> PersonOut:
    person = repo.get_person(person_id)
    if person is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such person: {person_id}")
    return PersonOut.of(person)


@router.patch("/watchlist/{person_id}", response_model=PersonOut, tags=["watchlist"])
def update_person(
    person_id: str, payload: PersonUpdate, repo: Repo, operator: CurrentOperator
) -> PersonOut:
    """Edit descriptive fields only. Templates are replaced by re-enrolling."""
    person = repo.get_person(person_id)
    if person is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such person: {person_id}")

    if payload.display_name is not None:
        person.display_name = payload.display_name
    if payload.notes is not None:
        person.notes = payload.notes
    repo.log(
        "update",
        actor=operator.username,
        subject=person_id,
        detail=payload.model_dump(exclude_none=True),
    )
    repo.session.commit()
    return PersonOut.of(person)


@router.delete("/watchlist/{person_id}", tags=["watchlist"])
def retire_person(person_id: str, repo: Repo, operator: CurrentOperator) -> dict:
    """Retire, not delete.

    Match decisions reference this person; removing the row would make past
    decisions unreviewable.
    """
    if not repo.retire(person_id, actor=operator.username):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No active watchlist entry for {person_id}",
        )
    return {"person_id": person_id, "retired": True}


# -- decisions -------------------------------------------------------------

@router.get("/decisions", response_model=list[DecisionOut], tags=["decisions"])
def list_decisions(
    repo: Repo,
    operator: CurrentOperator,
    pending_only: bool = Query(True, description="Only decisions awaiting review."),
    limit: int = Query(100, ge=1, le=1000),
) -> list[DecisionOut]:
    decisions = (
        repo.pending_decisions(limit=limit)
        if pending_only
        else list(
            repo.session.scalars(
                repo.session.query(MatchDecision)
                .order_by(MatchDecision.created_at.desc())
                .limit(limit)
                .statement
            )
        )
    )
    return [DecisionOut.of(d) for d in decisions]


@router.get("/decisions/{decision_id}", response_model=DecisionOut, tags=["decisions"])
def get_decision(decision_id: int, repo: Repo, operator: CurrentOperator) -> DecisionOut:
    decision = repo.session.get(MatchDecision, decision_id)
    if decision is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such decision")
    return DecisionOut.of(decision)


@router.post(
    "/decisions/{decision_id}/review", response_model=DecisionOut, tags=["decisions"]
)
def review_decision(
    decision_id: int,
    payload: ReviewIn,
    repo: Repo,
    operator: CurrentOperator,
) -> DecisionOut:
    """Record a human verdict on a candidate match.

    This is the only route to CONFIRMED. Nothing in the system confirms a match
    on its own, which is what makes the human-confirm step structural rather
    than a convention.
    """
    try:
        # The authenticated principal, not anything the body claims.
        decision = repo.review(
            decision_id,
            operator=operator.username,
            verdict=DecisionStatus(payload.verdict),
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return DecisionOut.of(decision)


@router.get("/alerts", response_model=list[DecisionOut], tags=["decisions"])
def actionable_alerts(
    repo: Repo, operator: CurrentOperator, limit: int = Query(100, ge=1, le=1000)
) -> list[DecisionOut]:
    """Human-confirmed decisions only.

    Phase 9's alerting reads this and nothing else, so an unreviewed match can
    never reach a notification.
    """
    return [DecisionOut.of(d) for d in repo.actionable_decisions(limit=limit)]


# -- audit -----------------------------------------------------------------

@router.get("/audit", response_model=list[AuditOut], tags=["audit"])
def audit_trail(
    repo: Repo, operator: CurrentOperator, limit: int = Query(200, ge=1, le=2000)
) -> list[AuditOut]:
    return [
        AuditOut(
            kind=event.kind,
            actor=event.actor,
            subject=event.subject,
            detail=json.loads(event.detail_json or "{}"),
            created_at=event.created_at.isoformat(),
        )
        for event in repo.audit_trail(limit=limit)
    ]


@router.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness only, and deliberately unauthenticated.

    It reports nothing about the watchlist. It used to return how many people
    were enrolled and how many decisions were pending, which is a small but
    real disclosure to an unauthenticated caller -- enough to tell whether a
    given installation is in use.
    """
    return {"status": "ok"}


@router.get("/stats", tags=["meta"])
def stats(repo: Repo, operator: CurrentOperator) -> dict:
    """The counts /health used to give away, plus anything the console should
    be warning its operators about."""
    settings = get_settings()

    # Warnings an operator needs in front of them, not buried in a server log
    # they will never read. Someone judging a candidate match deserves to know
    # the score they are judging is not calibrated for the model that produced
    # it (DES-01).
    warnings: list[str] = []
    mismatch = settings.reid_calibration_mismatch()
    if mismatch:
        warnings.append(mismatch)
    if settings.reid.weights in NON_REID_WEIGHTS:
        warnings.append(
            f"The appearance branch is running {settings.reid.weights!r} "
            "weights, which are not trained for person re-identification. "
            "Appearance similarity between two different people will be much "
            "higher than it should be."
        )

    unstamped = repo.templates_without_a_model()
    if unstamped:
        warnings.append(
            f"{unstamped} stored template(s) do not record which model "
            "produced them, so they cannot be checked against the model now "
            "running. If they were enrolled before reid.weights was changed, "
            "their similarity scores are meaningless. Re-enrol to be sure."
        )

    return {
        "watchlist": len(repo.list_people()),
        "pending_decisions": len(repo.pending_decisions(limit=1000)),
        "warnings": warnings,
    }
