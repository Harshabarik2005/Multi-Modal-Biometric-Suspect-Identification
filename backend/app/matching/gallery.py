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
from app.core.types import Modality, ModalityEmbedding, cosine_similarity

logger = get_logger(__name__)

TEMPLATE_KEY_ENV = "FRS_TEMPLATE_ENCRYPTION_KEY"
_ENCRYPTED_MAGIC = b"FRSENC1:"


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

    def rank(
        self, probes: dict[Modality, ModalityEmbedding]
    ) -> list[MatchCandidate]:
        """Score every enrolled person against this track's embeddings.

        Returns candidates sorted best-first. Scoring only -- it deliberately
        does not decide what counts as a match; thresholding is the caller's
        job, and in production a human's.
        """
        candidates: list[MatchCandidate] = []

        for person in self._people.values():
            candidate = MatchCandidate(person=person)
            for modality, probe in probes.items():
                reference = person.embedding(modality)
                similarity = (
                    probe.similarity(reference)
                    if reference is not None and probe.has_signal
                    else None
                )
                candidate.scores[modality] = ModalityScore(
                    modality=modality,
                    similarity=similarity,
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
            logger.warning(
                "Writing biometric templates UNENCRYPTED. Section 8 of the build "
                "plan requires encryption at rest. Set %s to enable it.",
                TEMPLATE_KEY_ENV,
            )
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
        return self.root / person_id

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
