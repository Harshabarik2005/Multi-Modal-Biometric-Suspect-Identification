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
from sqlalchemy import create_engine, func, select
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
    refuse_plaintext,
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
    _add_missing_columns(engine)


#: Columns added after a database may already exist in the field. `create_all`
#: creates missing tables but never alters existing ones, so without this an
#: upgrade opens a database that loads fine and fails on the first query
#: touching the new column.
#:
#: This is not a migration system. It handles the one case it can handle
#: safely -- an added, nullable, defaulted column -- and nothing else. Renames,
#: type changes and backfills need a real tool; this exists so that adding a
#: field does not silently break every deployment that predates it.
_ADDED_COLUMNS = {
    "templates": {"model_id": "VARCHAR(64) DEFAULT ''"},
    "decision_reviews": {"overturns_previous": "BOOLEAN DEFAULT 0"},
    "match_decisions": {"evidence_jpeg": "BLOB", "unused_json": "TEXT DEFAULT '{}'"},
    "people": {"reference_jpeg": "BLOB"},
}


def _add_missing_columns(engine: Engine) -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table, columns in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table)}
        for name, definition in columns.items():
            if name in present:
                continue
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                )
            logger.info("Added %s.%s to an existing database", table, name)


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
            refuse_plaintext("a biometric template")
            return raw, False
        return _ENCRYPTED_MAGIC + fernet.encrypt(raw), True

    def _encode_bytes(self, raw: bytes | None) -> bytes | None:
        """Encrypt an evidence image (DES-02).

        A crop of an identified person is personal data in the same sense a
        template is, so it goes through the same Fernet path rather than
        sitting in the clear in a table right next to the encrypted vectors.
        """
        if raw is None:
            return None
        fernet = self.crypto._fernet()
        if fernet is None:
            refuse_plaintext("a review image")
            return raw
        return _ENCRYPTED_MAGIC + fernet.encrypt(raw)

    def _decode_bytes(self, payload: bytes | None) -> bytes | None:
        if payload is None:
            return None
        if not payload.startswith(_ENCRYPTED_MAGIC):
            return payload  # written before evidence was encrypted
        fernet = self.crypto._fernet()
        if fernet is None:
            raise RuntimeError(
                "This review image is encrypted but no key is configured. Set "
                f"{TEMPLATE_KEY_ENV} to the key used when it was recorded."
            )
        return fernet.decrypt(payload[len(_ENCRYPTED_MAGIC) :])

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
        reference_jpeg: bytes | None = None,
    ) -> Person:
        """Add or replace a watchlist entry. Writes an audit event either way.

        `reference_jpeg` is one representative crop from the enrolment
        footage, shown beside a candidate match so the reviewer is comparing
        two pictures rather than trusting a number (DES-02).
        """
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
            if reference_jpeg is not None:
                person.reference_jpeg = self._encode_bytes(reference_jpeg)
        else:
            person = Person(
                person_id=person_id,
                display_name=display_name,
                notes=notes,
                source=source,
                reference_jpeg=self._encode_bytes(reference_jpeg),
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
                    model_id=embedding.model_id,
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

    def templates_without_a_model(self) -> int:
        """Count templates that do not record which model produced them.

        These predate the model being recorded (DES-01). They are allowed --
        refusing them would strand every existing watchlist -- but if any of
        them predate a change to `reid.weights` their similarity scores are
        meaningless, and nothing about that is visible in a score. The console
        reports the count so someone can decide whether to re-enrol.
        """
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(Template)
                .where((Template.model_id == "") | (Template.model_id.is_(None)))
            )
            or 0
        )

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

    def restore(self, person_id: str, actor: str = "system") -> bool:
        """Put a retired person back on the active watchlist.

        Retiring without this was a one-way door: a mis-click took someone off
        the list and the only route back was re-enrolling them from footage
        that may no longer exist. Nothing was destroyed by the retirement, so
        nothing needs rebuilding to undo it.
        """
        person = self.get_person(person_id)
        if person is None or person.retired_at is None:
            return False
        person.retired_at = None
        self.log("restore", actor=actor, subject=person_id)
        self.session.commit()
        return True

    def decision_count(self, person_id: str) -> int:
        """How many match decisions name this person.

        The number that decides whether they can be deleted outright or only
        have their biometrics erased.
        """
        person = self.get_person(person_id)
        if person is None:
            return 0
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(MatchDecision)
                .where(MatchDecision.person_pk == person.id)
            )
            or 0
        )

    def decision_counts(self) -> dict[int, int]:
        """Decisions per person primary key, for the whole watchlist at once.

        One grouped query rather than a count per person. The watchlist view
        needs this for every row it renders -- to tell a record that can be
        deleted from one that can only have its biometrics erased -- and
        reaching through `person.decisions` there would issue a query per
        person on a page whose whole job is to list all of them.
        """
        rows = self.session.execute(
            select(MatchDecision.person_pk, func.count())
            .group_by(MatchDecision.person_pk)
        )
        return {pk: int(count) for pk, count in rows}

    def erase_biometrics(self, person_id: str, actor: str = "system") -> int:
        """Destroy this person's templates and enrolment photo. Irreversible.

        The row survives, and so does every decision that referenced it, so
        past identifications stay reviewable -- a reviewer opening an old
        decision still sees who it named and on what evidence. What is gone is
        the biometric data itself: after this they cannot be matched against
        again, because there is nothing left to match.

        This is what an erasure request actually asks for. Deleting the row
        instead would satisfy nobody: it destroys the audit trail as
        collateral, which is the record of what the system did to that person.

        Returns how many templates were destroyed.
        """
        person = self.get_person(person_id)
        if person is None:
            return 0

        destroyed = len(person.templates)
        person.templates.clear()
        person.reference_jpeg = None
        # Not a template, but it is a photograph of them and it is stored
        # beside the vectors. Leaving it behind would make "erased" a lie.
        self.log(
            "erase_biometrics",
            actor=actor,
            subject=person_id,
            detail={"templates_destroyed": destroyed},
        )
        self.session.commit()
        return destroyed

    def delete_permanently(self, person_id: str, actor: str = "system") -> bool:
        """Remove the row entirely. Only safe when no decision references it.

        Raises ValueError when decisions exist. That is not caution for its own
        sake: `MatchDecision.person_pk` is a foreign key with no cascade, so
        deleting underneath it either fails outright or leaves decisions
        pointing at nothing, depending on whether the database is enforcing
        the constraint. A review queue full of matches against a missing person
        is worse than either outcome. `erase_biometrics` is the answer there.
        """
        person = self.get_person(person_id)
        if person is None:
            return False

        decisions = self.decision_count(person_id)
        if decisions:
            raise ValueError(
                f"{person_id} is named in {decisions} match decision(s). "
                "Deleting the record would leave those decisions pointing at "
                "nobody. Erase their biometrics instead -- that destroys the "
                "templates and the enrolment photo while keeping the decisions "
                "reviewable."
            )

        # Templates go with it: the relationship cascades delete-orphan.
        self.log(
            "delete",
            actor=actor,
            subject=person_id,
            detail={"display_name": person.display_name},
        )
        self.session.delete(person)
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
                model_id=template.model_id or "",
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
        evidence_jpeg: bytes | None = None,
        unused: dict[Modality, str] | None = None,
    ) -> MatchDecision:
        """Record a candidate match. Always PENDING -- never auto-confirmed.

        `evidence_jpeg` is the crop the match was made on. Without it a
        reviewer is asked to confirm an identification of someone they have
        never seen (DES-02).
        """
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
            unused_json=json.dumps(
                {m.value: str(why) for m, why in (unused or {}).items()}
            ),
            evidence_jpeg=self._encode_bytes(evidence_jpeg),
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

    def decision_evidence(self, decision_id: int) -> bytes | None:
        """The crop a decision was made on, decrypted. None if there is none."""
        decision = self.session.get(MatchDecision, decision_id)
        if decision is None:
            return None
        return self._decode_bytes(decision.evidence_jpeg)

    def person_reference(self, person_id: str) -> bytes | None:
        """The enrolment crop for a person, decrypted. None if there is none."""
        person = self.get_person(person_id)
        if person is None:
            return None
        return self._decode_bytes(person.reference_jpeg)

    def purge_evidence(self, older_than_days: float) -> int:
        """Delete review images from decisions older than `older_than_days`.

        The images exist so a human can judge a match. Once a decision is old
        enough that nobody is going to revisit it, keeping a picture of the
        person serves nothing and is one more thing to protect. The decision,
        its score, its weights and its reviews all survive -- the audit trail
        stays complete, it just stops carrying photographs indefinitely.

        Not called automatically: how long to keep evidence is a policy
        decision for whoever runs the deployment, not a default this code
        should pick for them.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        purged = 0
        for decision in self.session.scalars(
            select(MatchDecision).where(
                MatchDecision.created_at < cutoff,
                MatchDecision.evidence_jpeg.is_not(None),
            )
        ):
            decision.evidence_jpeg = None
            purged += 1
        if purged:
            self.log(
                "purge_evidence",
                actor="system",
                detail={"decisions": purged, "older_than_days": older_than_days},
            )
        self.session.commit()
        return purged

    def events_of_kind(self, kind: str) -> list[AuditEvent]:
        """Every event of one kind, oldest first and deliberately unlimited.

        `audit_trail` takes a limit because it backs a listing. This backs a
        question -- "has this already happened?" -- where a limit is a bug:
        enrolments, retirements and reviews share the table, so a windowed scan
        loses old records of one kind as other kinds accumulate, and the answer
        silently flips from yes to no (LOG-08).

        Unbounded is safe for the kinds this is used with. Alerts fire only on
        human-confirmed decisions, so their count is bounded by how fast people
        review, not by how much footage is processed.
        """
        return list(
            self.session.scalars(
                select(AuditEvent)
                .where(AuditEvent.kind == kind)
                .order_by(AuditEvent.created_at)
            )
        )
