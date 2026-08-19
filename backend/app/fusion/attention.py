"""Keyless attention fusion (Phase 6) -- the core contribution.

Small trainable head that scores each modality per frame, normalises
those into global weights (alpha for face, beta for gait, gamma for
re-ID), and produces one adaptive embedding.

Two deliberate divergences from the reference paper:
  * three modalities rather than two;
  * trained with a metric-learning loss (ArcFace-margin or triplet)
    for open-set matching, instead of a closed-set softmax over a
    fixed 20-person roster.

The per-modality weights are part of the output contract, not an
internal detail -- the dashboard surfaces them so an investigator
can see why a match fired.
"""


raise NotImplementedError  # scaffolded in Phase 0; see the build plan
