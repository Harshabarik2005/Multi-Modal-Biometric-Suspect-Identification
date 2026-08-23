"""Database-backed watchlist and audit trail (Phase 7).

Replaces the file-based `GalleryStore` from Phase 2 while keeping the same
encryption discipline. `Gallery` (the in-memory matcher) is unchanged: this
loads one from the database instead of from disk, which is why the gallery
interface was kept narrow.

The append-only rules from `models.py` are enforced here, in the only code that
writes: `confirm`/`reject` add a `DecisionReview` row and never rewrite the
original decision, and there is deliberately no method that edits or deletes
one.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.logging import get_logger
from app.core.types import Modality, ModalityEmbedding
from app.db.models import (
    AuditEvent,
    Base,
    DecisionReview,
    DecisionStatus,
    MatchDecision,
    Person,
    Template,
)
from app.matching.gallery import (
    _ENCRYPTED_MAGIC,
    TEMPLATE_KEY_ENV,
    Gallery,
    GalleryStore,
    PersonRecord,
)

logger = get_logger(__name__)


def make_engine(url: str, echo: bool = False) -> Engine:
    """Create an engine, with the two SQLite quirks this project trips over.

    `check_same_thread` because FastAPI serves requests from a thread pool.

    `StaticPool` for in-memory databases because each new connection otherwise
    gets its own *separate* empty database -- so `create_schema()` builds
    tables on one connection and the next session opens a blank one and reports
    "no such table". Pinning a single connection makes ``:memory:`` behave the
    way tests expect.
    """
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url.endswith("sqlite://"):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


class WatchlistRepository:
    """Reads and writes the watchlist, templates, and the audit trail."""

    def __init__(self, session: Session, crypto: GalleryStore | None = None) -> None:
        self.session = session
        # Reuse GalleryStore purely for its Fernet handling, so there is one
        # implementation of template encryption rather than two that can drift.
        self.crypto = crypto or GalleryStore()

    # -- template encryption ----------------------------------------------

    def _encode(self, vector: np.ndarray) -> tuple[bytes, bool]:
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(vector, dtype=np.float32))
        raw = buffer.getvalue()

        fernet = self.crypto._fernet()
        if fernet is None:
            logger.warning(
                "Storing a biometric template UNENCRYPTED. Section 8 of the "
                "build plan requires encryption at rest. Set %s.",
                TEMPLATE_KEY_ENV,
            )
            return raw, False
        return _ENCRYPTED_MAGIC + fernet.encrypt(raw), True

    def _decode(self, payload: bytes) -> np.ndarray:
        if payload.startswith(_ENCRYPTED_MAGIC):
            fernet = self.crypto._fernet()
            if fernet is None:
                raise RuntimeError(
                    "This template is encrypted but no key is configured. Set "
                    f"{TEMPLATE_KEY_ENV} to the key used at enrollment."
                )
            payload = fernet.decrypt(payload[len(_ENCRYPTED_MAGIC) :])
        return np.load(io.BytesIO(payload))

    # -- watchlist ---------------------------------------------------------

    def enroll(
        self,
        person_id: str,
        display_name: str,
        embeddings: dict[Modality, ModalityEmbedding],
        notes: str = "",
        source: str = "",
        actor: str = "system",
        replace: bool = False,
    ) -> Person:
        """Add or replace a watchlist entry. Writes an audit event either way."""
        usable = {m: e for m, e in embeddings.items() if e.has_signal}
        if not usable:
            raise ValueError(
                f"Refusing to enroll {person_id}: no modality produced an "
                "embedding. Check the footage actually shows the person."
            )

        existing = self.get_person(person_id)
        if existing is not None:
            if not replace:
                raise ValueError(f"{person_id!r} is already enrolled.")
            for template in list(existing.templates):
                self.session.delete(template)
            # Flush the deletes before inserting the replacements. Without
            # this, SQLAlchemy is free to order the INSERTs first and the
            # (person, modality) unique constraint fires on a re-enrollment.
            self.session.flush()
            person = existing
            person.display_name = display_name
            person.notes = notes
            person.source = source
            person.retired_at = None
        else:
            person = Person(
                person_id=person_id,
                display_name=display_name,
                notes=notes,
                source=source,
            )
            self.session.add(person)
            self.session.flush()

        for modality, embedding in usable.items():
            payload, encrypted = self._encode(embedding.vector)
            self.session.add(
                Template(
                    person_pk=person.id,
                    modality=modality.value,
                    payload=payload,
                    encrypted=encrypted,
                    dim=int(np.asarray(embedding.vector).size),
                    quality=float(embedding.quality),
                    frames_used=int(embedding.frames_used),
                )
            )

        self.log(
            "enroll" if existing is None else "re-enroll",
            actor=actor,
            subject=person_id,
            detail={"modalities": sorted(m.value for m in usable)},
        )
        self.session.commit()
        return person

    def get_person(self, person_id: str) -> Person | None:
        return self.session.scalar(
            select(Person).where(Person.person_id == person_id)
        )

    def list_people(self, include_retired: bool = False) -> list[Person]:
        statement = select(Person).order_by(Person.person_id)
        if not include_retired:
            statement = statement.where(Person.retired_at.is_(None))
        return list(self.session.scalars(statement))

    def retire(self, person_id: str, actor: str = "system") -> bool:
        """Take someone off the watchlist without destroying the audit trail.

        Deleting the row would orphan every match decision that referenced
        them, which would make past decisions unreviewable.
        """
        person = self.get_person(person_id)
        if person is None or person.retired_at is not None:
            return False
        person.retired_at = datetime.now(timezone.utc)
        self.log("retire", actor=actor, subject=person_id)
        self.session.commit()
        return True

    def to_record(self, person: Person) -> PersonRecord:
        embeddings: dict[Modality, ModalityEmbedding] = {}
        for template in person.templates:
            try:
                modality = Modality(template.modality)
            except ValueError:
                logger.warning("Unknown modality %r on %s", template.modality, person.person_id)
                continue
            embeddings[modality] = ModalityEmbedding(
                modality=modality,
                vector=self._decode(template.payload),
                quality=template.quality,
                frames_used=template.frames_used,
            )
        return PersonRecord(
            person_id=person.person_id,
            display_name=person.display_name,
            embeddings=embeddings,
            enrolled_at=person.enrolled_at.isoformat() if person.enrolled_at else "",
            notes=person.notes,
            source=person.source,
        )

    def load_gallery(self, include_retired: bool = False) -> Gallery:
        gallery = Gallery()
        for person in self.list_people(include_retired=include_retired):
            try:
                gallery.add(self.to_record(person))
            except Exception as exc:  # noqa: BLE001 - one bad row must not take
                # the whole watchlist offline; a missing person is safer than a
                # matcher that refuses to start.
                logger.error("Could not load %s: %s", person.person_id, exc)
        return gallery

    # -- match decisions ---------------------------------------------------

    def record_match(
        self,
        person_id: str,
        track_id: int,
        score: float,
        strategy: str = "",
        weights: dict[Modality, float] | None = None,
        calibrated: dict[Modality, float] | None = None,
        camera_id: str = "",
        frame_index: int = 0,
    ) -> MatchDecision:
        """Record a candidate match. Always PENDING -- never auto-confirmed."""
        person = self.get_person(person_id)
        if person is None:
            raise ValueError(f"Unknown person {person_id!r}")

        decision = MatchDecision(
            person_pk=person.id,
            camera_id=camera_id,
            track_id=track_id,
            frame_index=frame_index,
            score=float(score),
            strategy=strategy,
            weights_json=json.dumps(
                {m.value: round(w, 4) for m, w in (weights or {}).items()}
            ),
            calibrated_json=json.dumps(
                {m.value: round(c, 4) for m, c in (calibrated or {}).items()}
            ),
            status=DecisionStatus.PENDING,
        )
        self.session.add(decision)
        self.session.commit()
        return decision

    def review(
        self,
        decision_id: int,
        operator: str,
        verdict: DecisionStatus,
        reason: str = "",
    ) -> MatchDecision:
        """Record a human verdict. Appends a review; never edits the decision.

        `operator` is required and not defaulted. A decision reviewed by
        "someone" is not reviewed.
        """
        if verdict not in (DecisionStatus.CONFIRMED, DecisionStatus.REJECTED):
            raise ValueError(
                f"A human verdict must be CONFIRMED or REJECTED, got {verdict}."
            )
        if not operator or not operator.strip():
            raise ValueError(
                "A review must name the operator who made it. An anonymous "
                "confirmation is not an audit trail."
            )

        decision = self.session.get(MatchDecision, decision_id)
        if decision is None:
            raise ValueError(f"No match decision with id {decision_id}")

        # Re-reviewing is allowed, because people make mistakes and the history
        # has to show the correction. But an overturn is marked as one: without
        # it, `status` -- which /alerts and the dispatcher gate on -- could be
        # flipped back and forth and every review would look like a first
        # opinion.
        previously = decision.status
        overturns = previously in (
            DecisionStatus.CONFIRMED,
            DecisionStatus.REJECTED,
        ) and previously is not verdict

        self.session.add(
            DecisionReview(
                decision_pk=decision.id,
                operator=operator.strip(),
                verdict=verdict,
                reason=reason,
                overturns_previous=overturns,
            )
        )
        # Denormalised for querying; the reviews remain the source of truth.
        decision.status = verdict
        self.log(
            "review",
            actor=operator,
            subject=decision.person.person_id,
            detail={
                "decision_id": decision.id,
                "verdict": verdict.value,
                "score": decision.score,
                "overturns_previous": overturns,
                "previous_status": previously.value,
            },
        )
        self.session.commit()
        return decision

    def pending_decisions(self, limit: int = 100) -> list[MatchDecision]:
        return list(
            self.session.scalars(
                select(MatchDecision)
                .where(MatchDecision.status == DecisionStatus.PENDING)
                .order_by(MatchDecision.created_at.desc())
                .limit(limit)
            )
        )

    def decisions_for(self, person_id: str, limit: int = 100) -> list[MatchDecision]:
        person = self.get_person(person_id)
        if person is None:
            return []
        return list(
            self.session.scalars(
                select(MatchDecision)
                .where(MatchDecision.person_pk == person.id)
                .order_by(MatchDecision.created_at.desc())
                .limit(limit)
            )
        )

    def actionable_decisions(self, limit: int = 100) -> list[MatchDecision]:
        """Only human-confirmed decisions. Phase 9 alerting gates on this."""
        return list(
            self.session.scalars(
                select(MatchDecision)
                .where(MatchDecision.status == DecisionStatus.CONFIRMED)
                .order_by(MatchDecision.created_at.desc())
                .limit(limit)
            )
        )

    # -- audit -------------------------------------------------------------

    def log(
        self,
        kind: str,
        actor: str = "system",
        subject: str = "",
        detail: dict | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            kind=kind,
            actor=actor,
            subject=subject,
            detail_json=json.dumps(detail or {}),
        )
        self.session.add(event)
        return event

    def audit_trail(self, limit: int = 200) -> list[AuditEvent]:
        return list(
            self.session.scalars(
                select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
            )
        )
