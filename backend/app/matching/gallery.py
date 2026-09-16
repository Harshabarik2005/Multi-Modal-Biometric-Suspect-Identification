"""The watchlist gallery: enrolled people and their reference embeddings.

Open-set matching, not classification. There is no fixed roster and no softmax
over N known people -- a track is compared by cosine similarity against every
enrolled person, and the vast majority of people walking past a camera match
nobody. That is the expected outcome, and the reason accuracy is a meaningless
metric here (a system that matched nobody, ever, would score extremely well).

**Biometric templates are personal data.** Section 8 of the build plan requires
them encrypted at rest, so `GalleryStore` encrypts the vectors whenever a key
is configured, and warns loudly on every save when one is not. The plaintext
path exists only so local development is not blocked; it is not a mode to ship.
Set `FRS_TEMPLATE_ENCRYPTION_KEY` to turn encryption on:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Storage is one directory per person under `paths.enrollment_dir`. Phase 7
replaces this with a real database; the interface here is deliberately narrow
so that swap stays cheap.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.types import (
    Modality,
    ModalityEmbedding,
    cosine_similarity,
    l2_normalize,
)

logger = get_logger(__name__)

TEMPLATE_KEY_ENV = "FRS_TEMPLATE_ENCRYPTION_KEY"

#: SEC-06. Encryption at rest used to be opt-in: with no key, templates were
#: written as plain float32 vectors and a warning went to a log nobody reads.
#: That made the insecure state the default, and because encryption is recorded
#: per row a deployment could end up half-encrypted and still look right in a
#: spot check. Writing now refuses unless a key is configured, or this is set
#: to make the choice deliberate and visible.
ALLOW_PLAINTEXT_ENV = "FRS_ALLOW_PLAINTEXT_TEMPLATES"


def model_mismatch(
    probe: ModalityEmbedding, reference: ModalityEmbedding
) -> str | None:
    """Report a probe and reference that came from different models (DES-01).

    Returns None when they are comparable -- including when either side does
    not know what produced it, which is every template enrolled before the
    model was recorded. Refusing those would strand existing watchlists, and
    the honest position is that they are unverified rather than known-wrong.
    A deployment that re-enrols gets the check; one that does not is no worse
    off than before.
    """
    if not probe.model_id or not reference.model_id:
        return None
    if probe.model_id == reference.model_id:
        return None
    return (
        f"enrolled with {reference.model_id}, probe from {probe.model_id} -- "
        "different models, so their similarity is meaningless. Re-enrol."
    )


def plaintext_is_permitted() -> bool:
    return os.environ.get(ALLOW_PLAINTEXT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def refuse_plaintext(what: str) -> None:
    """Raise unless writing unencrypted biometric data has been opted into.

    Called on the write path only. Reading plaintext templates that already
    exist still works, because refusing to read them would strand data someone
    has to migrate rather than protecting anything.
    """
    if plaintext_is_permitted():
        logger.warning(
            "Writing %s UNENCRYPTED because %s is set. This is biometric data; "
            "do not use this setting outside local development.",
            what,
            ALLOW_PLAINTEXT_ENV,
        )
        return

    raise RuntimeError(
        f"""Refusing to write {what} unencrypted.

These are biometric templates -- they identify a real person and cannot be
reissued like a password.

Set an encryption key:
  python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
  export {TEMPLATE_KEY_ENV}=<that key>

Keep the key somewhere other than the database it protects; without it the
enrolments cannot be read back.

For local development with throwaway data only, set {ALLOW_PLAINTEXT_ENV}=1 to write
plaintext deliberately."""
    )
_ENCRYPTED_MAGIC = b"FRSENC1:"

#: person_id becomes a directory name under `paths.enrollment_dir`, so it has
#: to be a safe path component. Must START with an alphanumeric: a plain
#: charset class still admits ".." and ".", which are traversal components made
#: entirely of otherwise-permitted characters. No lookahead, because pydantic
#: v2 validates patterns with the Rust regex crate, which does not support it.
PERSON_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


def validate_person_id(person_id: str) -> str:
    """Return `person_id` if it is a safe path component, else raise.

    Enforced here rather than only at the API boundary, because the CLI writes
    to the same directories and `scripts/enroll.py --person-id ../../..` would
    otherwise escape the enrolment root entirely.
    """
    import re

    if not re.match(PERSON_ID_PATTERN, person_id or ""):
        raise ValueError(
            f"Invalid person_id {person_id!r}. It becomes a directory name, so "
            "it must start with a letter or digit and contain only letters, "
            "digits, dots, dashes and underscores (max 64 characters)."
        )
    return person_id


@dataclass
class PersonRecord:
    """One enrolled person and their reference embedding per modality."""

    person_id: str
    display_name: str
    embeddings: dict[Modality, ModalityEmbedding] = field(default_factory=dict)
    enrolled_at: str = ""
    notes: str = ""
    source: str = ""

    def embedding(self, modality: Modality) -> ModalityEmbedding | None:
        record = self.embeddings.get(modality)
        return record if record is not None and record.has_signal else None

    @property
    def modalities(self) -> list[Modality]:
        return [m for m in Modality if self.embedding(m) is not None]


@dataclass
class ModalityScore:
    """How one modality scored a track against one enrolled person."""

    modality: Modality
    similarity: float | None  # None = this modality could not compare at all
    probe_quality: float
    #: Why, when `similarity` is None. Surfaced in the explainability view so
    #: "no face was visible" and "the stored reference is from a different
    #: model" are distinguishable, which they are not from a bare None.
    incomparable_reason: str = ""
    #: True when the reason is a reference from a different model (DES-01) --
    #: the one refusal a reviewer is warned about, unlike routine ones such as
    #: a person with no gait reference.
    model_mismatch: bool = False


@dataclass
class MatchCandidate:
    """One enrolled person's overall score against one track."""

    person: PersonRecord
    scores: dict[Modality, ModalityScore] = field(default_factory=dict)
    fused_similarity: float = 0.0
    # Per-modality weights that produced `fused_similarity`. Phase 5 fills
    # these with fixed weights, Phase 6 with learned attention. Surfaced in the
    # dashboard so a reviewer can see what drove the match.
    weights: dict[Modality, float] = field(default_factory=dict)
    # Full fusion breakdown (a `FusionResult`) when a strategy was used, for
    # the audit log and the explainability view. Typed loosely to keep the
    # gallery from importing the fusion package at module scope.
    fusion: object | None = None

    @property
    def comparable_modalities(self) -> list[Modality]:
        return [m for m, s in self.scores.items() if s.similarity is not None]

    def explain(self) -> str:
        """One-line human-readable breakdown for logs and the CLI."""
        if not self.scores:
            return "no modality could be compared"
        parts = []
        for modality, score in sorted(self.scores.items(), key=lambda kv: kv[0].value):
            if score.similarity is None:
                parts.append(f"{modality.value}=n/a")
            else:
                weight = self.weights.get(modality)
                suffix = f" w={weight:.2f}" if weight is not None else ""
                parts.append(f"{modality.value}={score.similarity:+.3f}{suffix}")
        return "  ".join(parts)

    def not_counted(self) -> dict[Modality, str]:
        """Modalities that reached this comparison but did not count, and why.

        Either the comparison was refused -- no reference, a different model,
        too few gait references -- or fusion withheld the modality from voting.
        A branch that produced no probe never gets this far; callers add those
        from the probe's own reason.
        """
        reasons = {
            modality: score.incomparable_reason or "could not be compared"
            for modality, score in self.scores.items()
            if score.similarity is None
        }
        if self.fusion is not None:
            reasons.update(self.fusion.withheld)
        return reasons


class Gallery:
    """In-memory watchlist. Compares a probe embedding against every person."""

    def __init__(self, people: list[PersonRecord] | None = None) -> None:
        self._people: dict[str, PersonRecord] = {}
        for person in people or []:
            self.add(person)

    def add(self, person: PersonRecord) -> None:
        if person.person_id in self._people:
            logger.warning("Replacing existing gallery entry %s", person.person_id)
        self._people[person.person_id] = person

    def get(self, person_id: str) -> PersonRecord | None:
        return self._people.get(person_id)

    def remove(self, person_id: str) -> bool:
        return self._people.pop(person_id, None) is not None

    def __len__(self) -> int:
        return len(self._people)

    def __iter__(self):
        return iter(self._people.values())

    @property
    def people(self) -> list[PersonRecord]:
        return list(self._people.values())

    # -- gait population centring -----------------------------------------

    def gait_population_mean(self, min_references: int) -> np.ndarray | None:
        """Mean of every enrolled gait descriptor, or None if too few.

        Every Gait Energy Image looks like a blurry human, so raw cosine
        similarity between two GEI descriptors is dominated by that shared
        shape: measured on synthetic walkers, different walking styles scored
        0.986 against 1.000 for the same style -- a separation of 0.014, which
        is useless. Removing the population mean strips the "generic human"
        component and leaves what actually distinguishes people, taking the
        separation to 0.484.

        Estimating that mean needs several references. Below `min_references`
        this returns None and gait comparison is refused outright, because an
        uncentred gait similarity of 0.96 looks like a strong match and is not.
        """
        vectors = [
            person.embeddings[Modality.GAIT].vector
            for person in self._people.values()
            if person.embedding(Modality.GAIT) is not None
        ]
        if len(vectors) < min_references:
            return None
        return l2_normalize(np.mean(np.stack(vectors), axis=0))

    @staticmethod
    def remove_population_component(
        vector: np.ndarray, mean: np.ndarray
    ) -> np.ndarray:
        """Project out the population mean, then renormalise."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        residual = vector - float(np.dot(vector, mean)) * mean
        return l2_normalize(residual)

    def _gait_similarity(
        self,
        probe: ModalityEmbedding,
        reference: ModalityEmbedding,
        mean: np.ndarray | None,
    ) -> float | None:
        """Gait similarity with the population component removed.

        Returns None when there is no population mean available. That is a
        deliberate refusal rather than a fallback: uncentred gait similarities
        sit above 0.93 for everyone, so returning one would manufacture a
        confident match out of nothing.
        """
        if mean is None:
            return None
        return cosine_similarity(
            self.remove_population_component(probe.vector, mean),
            self.remove_population_component(reference.vector, mean),
        )

    # -- ranking -----------------------------------------------------------

    @staticmethod
    def _reference_trust(
        person: PersonRecord, modality: Modality, half_life_days: float
    ) -> float:
        """How far THIS person's reference for THIS modality can still be trusted.

        Only re-ID decays. Face and gait describe the person; re-ID largely
        describes their clothing, so a three-week-old reference is far weaker
        evidence than one from this morning.

        Resolved here, per person, rather than passed in by the caller. It used
        to be a `trusts` parameter, which meant every call site had to remember
        it -- and the API scan path, the documented primary workflow, did not.
        Nor did the evaluation harness. Something that must never be forgotten
        should not be something a caller can omit.
        """
        if modality is not Modality.REID or half_life_days <= 0:
            return 1.0
        if not person.enrolled_at:
            return 1.0

        from datetime import datetime, timezone

        try:
            enrolled = datetime.fromisoformat(person.enrolled_at)
        except ValueError:
            return 1.0
        if enrolled.tzinfo is None:
            enrolled = enrolled.replace(tzinfo=timezone.utc)

        from app.embeddings.reid import trust_at

        elapsed_days = (
            datetime.now(timezone.utc) - enrolled
        ).total_seconds() / 86400.0
        return trust_at(elapsed_days, half_life_days)

    def rank(
        self,
        probes: dict[Modality, ModalityEmbedding],
        strategy=None,
        gait_min_references: int = 3,
        reid_half_life_days: float = 0.0,
    ) -> list[MatchCandidate]:
        """Score every enrolled person against this track's embeddings.

        Returns candidates sorted best-first. Scoring only -- it deliberately
        does not decide what counts as a match; thresholding is the caller's
        job, and in production a human's.

        With a `strategy`, `fused_similarity` is the fused calibrated score in
        [0, 1] and `weights` records what drove it. Without one, only the
        per-modality scores are filled in.

        `reid_half_life_days` applies staleness decay to each person's own
        re-ID reference. Pass `settings.reid.trust_half_life_days`; 0 disables
        it. It is resolved per person here rather than supplied by the caller,
        because when it was the caller's job two of the three call sites forgot.
        """
        gait_mean = (
            self.gait_population_mean(gait_min_references)
            if Modality.GAIT in probes
            else None
        )
        candidates: list[MatchCandidate] = []

        for person in self._people.values():
            candidate = MatchCandidate(person=person)
            for modality, probe in probes.items():
                reference = person.embedding(modality)
                reason = ""
                is_mismatch = False
                if reference is None or not probe.has_signal:
                    similarity = None
                    if reference is None:
                        reason = "no reference enrolled for this person"
                elif (mismatch := model_mismatch(probe, reference)) is not None:
                    # Refuse rather than score. Cosine similarity between two
                    # different embedding spaces is noise shaped like a number,
                    # and this system's whole discipline is that "could not
                    # compare" must never collapse into "compared and got a
                    # low score" (DES-01).
                    similarity, reason, is_mismatch = None, mismatch, True
                elif modality is Modality.GAIT:
                    similarity = self._gait_similarity(probe, reference, gait_mean)
                    if similarity is None:
                        reason = (
                            f"fewer than {gait_min_references} people have a "
                            "gait reference, so walks cannot be compared yet"
                        )
                else:
                    similarity = probe.similarity(reference)

                candidate.scores[modality] = ModalityScore(
                    modality=modality,
                    similarity=similarity,
                    probe_quality=probe.quality,
                    incomparable_reason=reason,
                    model_mismatch=is_mismatch,
                )

            if strategy is not None:
                from app.fusion.baseline import FusionInput

                result = strategy.fuse(
                    [
                        FusionInput(
                            modality=modality,
                            similarity=score.similarity,
                            quality=score.probe_quality,
                            trust=self._reference_trust(
                                person, modality, reid_half_life_days
                            ),
                        )
                        for modality, score in candidate.scores.items()
                    ]
                )
                candidate.fused_similarity = result.score
                candidate.weights = result.weights
                candidate.fusion = result

            candidates.append(candidate)

        return sorted(candidates, key=lambda c: c.fused_similarity, reverse=True)

    def rank_attention(
        self,
        probes: dict[Modality, ModalityEmbedding],
        model,
    ) -> list[MatchCandidate]:
        """Rank using the learned attention head (Phase 6).

        Structurally different from `rank()`. The Phase-5 strategies fuse
        per-modality *similarities*; the attention head fuses *embeddings* into
        a single adaptive vector, so both the probe and every enrolled person
        are fused first and compared once in that shared space. That is the
        paper's formulation, and it is why the head has to be trained: it is
        learning a joint representation, not a weighted average of scores.
        """
        if not getattr(model, "is_trained", False):
            raise RuntimeError(
                "The attention head is untrained. Ranking with it would produce "
                "noise strictly worse than the Phase-5 fixed rules."
            )

        for modality, probe in probes.items():
            expected = model.dims.get(modality)
            if expected is not None and probe.vector.size != expected:
                raise ValueError(
                    f"The attention head expects {expected}-d {modality.value} "
                    f"embeddings but got {probe.vector.size}-d. A head trained "
                    "on synthetic data cannot fuse real embeddings; retrain it "
                    "on real footage with matching dimensions."
                )

        probe_vector, probe_weights = model.fuse_one(
            {m: e.vector for m, e in probes.items() if e.has_signal},
            {m: e.quality for m, e in probes.items()},
        )

        candidates: list[MatchCandidate] = []
        for person in self._people.values():
            # Drop any modality whose stored reference came from a different
            # model. Here it matters more than in rank(): the head fuses
            # embeddings into one vector before comparing, so a single
            # mismatched modality contaminates the fused representation rather
            # than just contributing one bad per-modality score (DES-01).
            usable = [
                m
                for m in person.modalities
                if m in model.dims
                and (
                    m not in probes
                    or model_mismatch(probes[m], person.embeddings[m]) is None
                )
            ]
            references = {m: person.embeddings[m].vector for m in usable}
            if not references:
                continue
            reference_vector, _ = model.fuse_one(
                references,
                {m: person.embeddings[m].quality for m in references},
            )

            candidate = MatchCandidate(person=person)
            candidate.fused_similarity = cosine_similarity(
                probe_vector, reference_vector
            )
            candidate.weights = {m: w for m, w in probe_weights.items() if w > 0.0}
            for modality, probe in probes.items():
                candidate.scores[modality] = ModalityScore(
                    modality=modality,
                    similarity=None,  # attention compares in the fused space only
                    probe_quality=probe.quality,
                )
            candidates.append(candidate)

        return sorted(candidates, key=lambda c: c.fused_similarity, reverse=True)

    def rank_single(
        self, modality: Modality, probe: ModalityEmbedding
    ) -> list[MatchCandidate]:
        """Rank on one modality alone. This is the Phase-2 matching path.

        Phases 5 and 6 replace the fused score with real fusion; until then
        `fused_similarity` is simply that one modality's similarity.
        """
        candidates = self.rank({modality: probe})
        for candidate in candidates:
            score = candidate.scores.get(modality)
            candidate.fused_similarity = (
                score.similarity if score and score.similarity is not None else -1.0
            )
            candidate.weights = {modality: 1.0}
        return sorted(candidates, key=lambda c: c.fused_similarity, reverse=True)


class GalleryStore:
    """Loads and saves gallery entries under `paths.enrollment_dir`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.paths.enrollment_dir

    # -- encryption --------------------------------------------------------

    def _fernet(self):
        key = os.environ.get(TEMPLATE_KEY_ENV, "").strip()
        if not key:
            return None
        try:
            from cryptography.fernet import Fernet

            return Fernet(key.encode())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{TEMPLATE_KEY_ENV} is set but is not a valid Fernet key: {exc}. "
                "Generate one with: python -c \"from cryptography.fernet import "
                'Fernet; print(Fernet.generate_key().decode())"'
            ) from exc

    def _encode_vectors(self, payload: dict[str, np.ndarray]) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        raw = buffer.getvalue()

        fernet = self._fernet()
        if fernet is None:
            refuse_plaintext("gallery templates")
            return raw
        return _ENCRYPTED_MAGIC + fernet.encrypt(raw)

    def _decode_vectors(self, blob: bytes) -> dict[str, np.ndarray]:
        if blob.startswith(_ENCRYPTED_MAGIC):
            fernet = self._fernet()
            if fernet is None:
                raise RuntimeError(
                    "This enrollment is encrypted but no key is configured. "
                    f"Set {TEMPLATE_KEY_ENV} to the key used at enrollment."
                )
            blob = fernet.decrypt(blob[len(_ENCRYPTED_MAGIC) :])
        with np.load(io.BytesIO(blob)) as data:
            return {name: data[name] for name in data.files}

    # -- persistence -------------------------------------------------------

    def person_dir(self, person_id: str) -> Path:
        return self.root / validate_person_id(person_id)

    def save(self, person: PersonRecord) -> Path:
        directory = self.person_dir(person.person_id)
        directory.mkdir(parents=True, exist_ok=True)

        vectors: dict[str, np.ndarray] = {}
        meta_embeddings: dict[str, dict] = {}
        for modality, embedding in person.embeddings.items():
            if not embedding.has_signal:
                continue
            vectors[modality.value] = np.asarray(embedding.vector, dtype=np.float32)
            meta_embeddings[modality.value] = {
                "quality": embedding.quality,
                "frames_used": embedding.frames_used,
                "dim": int(np.asarray(embedding.vector).size),
                "detail": {k: float(v) for k, v in embedding.detail.items()},
                # Which model produced it. A reference is only comparable to a
                # probe from the same one (DES-01).
                "model_id": embedding.model_id,
            }

        if not vectors:
            raise ValueError(
                f"Refusing to save {person.person_id}: no modality produced an "
                "embedding. Check the enrollment footage actually shows the person."
            )

        (directory / "templates.npz").write_bytes(self._encode_vectors(vectors))

        metadata = {
            "person_id": person.person_id,
            "display_name": person.display_name,
            "enrolled_at": person.enrolled_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": person.notes,
            "source": person.source,
            "encrypted": bool(self._fernet()),
            "embeddings": meta_embeddings,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info(
            "Enrolled %s (%s) with %s",
            person.person_id,
            person.display_name,
            ", ".join(sorted(vectors)),
        )
        return directory

    def load_person(self, person_id: str) -> PersonRecord:
        directory = self.person_dir(person_id)
        meta_path = directory / "metadata.json"
        vec_path = directory / "templates.npz"
        if not meta_path.is_file() or not vec_path.is_file():
            raise FileNotFoundError(f"No enrollment found for {person_id!r}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = self._decode_vectors(vec_path.read_bytes())

        embeddings: dict[Modality, ModalityEmbedding] = {}
        for name, vector in vectors.items():
            try:
                modality = Modality(name)
            except ValueError:
                logger.warning("Unknown modality %r in %s; skipping", name, person_id)
                continue
            info = metadata.get("embeddings", {}).get(name, {})
            embeddings[modality] = ModalityEmbedding(
                modality=modality,
                vector=np.asarray(vector, dtype=np.float32),
                quality=float(info.get("quality", 0.0)),
                frames_used=int(info.get("frames_used", 0)),
                detail={k: float(v) for k, v in (info.get("detail") or {}).items()},
                model_id=str(info.get("model_id", "")),
            )

        return PersonRecord(
            person_id=metadata["person_id"],
            display_name=metadata.get("display_name", metadata["person_id"]),
            embeddings=embeddings,
            enrolled_at=metadata.get("enrolled_at", ""),
            notes=metadata.get("notes", ""),
            source=metadata.get("source", ""),
        )

    def load_gallery(self) -> Gallery:
        gallery = Gallery()
        if not self.root.is_dir():
            logger.info("No enrollment directory at %s; gallery is empty", self.root)
            return gallery

        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not (directory / "metadata.json").is_file():
                continue
            try:
                gallery.add(self.load_person(directory.name))
            except Exception as exc:  # noqa: BLE001 - one bad entry must not
                # take the whole watchlist down; a missing person is safer than
                # a matcher that refuses to start.
                logger.error("Could not load enrollment %s: %s", directory.name, exc)

        logger.info("Loaded %d enrolled %s", len(gallery),
                    "person" if len(gallery) == 1 else "people")
        return gallery

    def list_person_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "metadata.json").is_file()
        )
