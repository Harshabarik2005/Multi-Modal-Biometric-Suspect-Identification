"""Persistence models (Phase 7).

Watchlist people, their per-modality reference embeddings, match
decisions, and the audit log.

Two constraints from section 8 that belong in the schema itself, not
in application code: biometric templates are encrypted at rest, and
every match decision is written to an append-only audit record
including the attention weights and the confirming operator.
"""


raise NotImplementedError  # scaffolded in Phase 0; see the build plan
