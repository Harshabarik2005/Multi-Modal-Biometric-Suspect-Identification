"""Persistence models (Phase 7).

Three things in this schema exist because section 8 of the build plan requires
them, not because they were convenient:

**Biometric templates are encrypted at rest.** `Template.payload` holds a blob
produced by the same Fernet path the file-based `GalleryStore` uses, so a
database dump does not leak biometric vectors.

**Match decisions are append-only.** `MatchDecision` rows are never updated or
deleted — a confirmation writes a *new* `DecisionReview` row referencing the
original. An audit trail you can edit is not an audit trail, and "the system
said X, then someone changed it to Y" is exactly the thing a review needs to be
able to see.

**No action follows from a match alone.** A `MatchDecision` is created with
`status = PENDING` and stays there until a human writes a review. Nothing
downstream (Phase 9 alerting) is permitted to fire on a pending decision. The
human-confirm step is therefore structural rather than a convention someone can
forget.

Every decision also stores the per-modality fusion weights that produced it, so
a reviewer can later see whether a match rested on a clear face or mostly on a
jacket.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DecisionStatus(str, enum.Enum):
    """Lifecycle of a match. PENDING is the only state the system may create."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Person(Base):
    """Someone on the watchlist."""

    __tablename__ = "people"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(Text, default="")

    #: One representative crop from the enrolment footage, JPEG, encrypted.
    #: Shown beside the match so a reviewer is comparing two pictures rather
    #: than trusting a number (DES-02).
    reference_jpeg: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True, default=None
    )

    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Set rather than deleting the row: match decisions reference this person,
    # and an audit trail with dangling references is not auditable.
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    templates: Mapped[list["Template"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    decisions: Mapped[list["MatchDecision"]] = relationship(back_populates="person")

    @property
    def is_active(self) -> bool:
        return self.retired_at is None


class Template(Base):
    """One modality's reference embedding for one person, encrypted."""

    __tablename__ = "templates"
    __table_args__ = (
        UniqueConstraint("person_pk", "modality", name="uq_template_person_modality"),
        CheckConstraint("dim > 0", name="ck_template_dim_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    person_pk: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"))
    modality: Mapped[str] = mapped_column(String(16), index=True)

    #: Encrypted vector. Never store a raw float array here.
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    #: True when `payload` went through Fernet. False means the deployment has
    #: no key configured, which is a misconfiguration rather than a mode.
    encrypted: Mapped[bool] = mapped_column(default=False)

    dim: Mapped[int] = mapped_column(Integer)
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    frames_used: Mapped[int] = mapped_column(Integer, default=0)

    #: Which model produced this vector, e.g. "osnet_x1_0/msmt17".
    #:
    #: A stored reference is only comparable to a probe from the same model.
    #: Change the checkpoint and every template here still loads, still has
    #: the right length, and still yields a cosine similarity -- one that
    #: means nothing, because the two vectors live in unrelated spaces
    #: (DES-01). Empty on rows enrolled before this column existed, which is
    #: treated as unknown rather than as agreement.
    model_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    person: Mapped[Person] = relationship(back_populates="templates")


class MatchDecision(Base):
    """One candidate identification, awaiting or having received human review.

    Append-only. Rows are never updated: a review is a separate row in
    `DecisionReview` pointing here. `status` is maintained by the repository as
    a denormalised convenience for querying, and is derived from the reviews.
    """

    __tablename__ = "match_decisions"
    __table_args__ = (
        Index("ix_decision_status_created", "status", "created_at"),
        CheckConstraint("score >= -1.0 AND score <= 1.0", name="ck_decision_score_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    person_pk: Mapped[int] = mapped_column(ForeignKey("people.id"))

    camera_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)

    score: Mapped[float] = mapped_column(Float)
    strategy: Mapped[str] = mapped_column(String(32), default="")
    #: JSON: {"face": 0.46, "reid": 0.54}. The explainability record -- what
    #: the system actually relied on when it raised this.
    weights_json: Mapped[str] = mapped_column(Text, default="{}")
    #: JSON: per-modality calibrated scores, for the same reason.
    calibrated_json: Mapped[str] = mapped_column(Text, default="{}")

    #: The crop this match was made on, JPEG, encrypted like a template
    #: (DES-02). The review card previously showed a score and some weight
    #: bars and no image at all, so the human whose confirmation the whole
    #: design rests on could judge how the system reached its conclusion but
    #: not whether it was right.
    #:
    #: Nullable: a decision recorded before this existed, or one where the
    #: frame could not be encoded, still has to load.
    evidence_jpeg: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True, default=None
    )

    status: Mapped[DecisionStatus] = mapped_column(
        Enum(DecisionStatus), default=DecisionStatus.PENDING, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    person: Mapped[Person] = relationship(back_populates="decisions")
    reviews: Mapped[list["DecisionReview"]] = relationship(
        back_populates="decision", order_by="DecisionReview.created_at"
    )

    @property
    def is_actionable(self) -> bool:
        """Whether anything downstream may act on this.

        Only a human-confirmed decision qualifies. Phase 9 alerting must gate
        on this and nothing else.
        """
        return self.status is DecisionStatus.CONFIRMED


class DecisionReview(Base):
    """A human's verdict on a match decision. Append-only.

    A reviewer changing their mind writes another row rather than editing this
    one, so the sequence of judgements stays visible.
    """

    __tablename__ = "decision_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    decision_pk: Mapped[int] = mapped_column(ForeignKey("match_decisions.id"))

    #: The authenticated operator's username, copied at review time. Not
    #: supplied by the caller -- see app/api/auth.py. It used to be free text
    #: from the request body, which made every entry in the audit trail an
    #: unverified claim.
    operator: Mapped[str] = mapped_column(String(120))
    verdict: Mapped[DecisionStatus] = mapped_column(Enum(DecisionStatus))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    #: True when this review overturned an earlier verdict. Overturning is
    #: allowed -- people make mistakes -- but it is recorded as such rather
    #: than looking like a first opinion.
    overturns_previous: Mapped[bool] = mapped_column(default=False)

    decision: Mapped[MatchDecision] = relationship(back_populates="reviews")


class Operator(Base):
    """Someone allowed to use the system.

    Before this existed, `operator` was a free-text field on a review -- so the
    audit trail recorded a claimed name that nobody had verified, and every
    decision in it was repudiable. An identification confirmed by "whoever
    typed alice" is not confirmed by anyone.

    Passwords are scrypt-hashed with a per-account salt; the plaintext is never
    stored and never logged.
    """

    __tablename__ = "operators"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200), default="")

    password_salt: Mapped[str] = mapped_column(String(64))
    password_hash: Mapped[str] = mapped_column(String(128))

    is_active: Mapped[bool] = mapped_column(default=True)
    #: Admins may manage other operators. Everything else is open to any
    #: signed-in operator; finer authorisation is not modelled yet.
    is_admin: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditEvent(Base):
    """Everything else worth recording: enrollments, retirements, exports.

    Append-only, deliberately schema-light. A biometric system should be able
    to answer "who did what, when" about the watchlist itself, not only about
    matches.
    """

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_kind_created", "kind", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    actor: Mapped[str] = mapped_column(String(120), default="system")
    subject: Mapped[str] = mapped_column(String(120), default="")
    detail_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
