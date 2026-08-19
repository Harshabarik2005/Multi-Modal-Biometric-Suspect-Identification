"""Face embedding branch (Phase 2) -- ArcFace / InsightFace.

Takes pose-guided face crops from a track and returns a normalised
512-d identity embedding. Must be occlusion-aware: masks and
sunglasses are expected, not exceptional, and the branch should
report a per-frame quality score that Phase 6's attention head can
use to decide how far to trust this modality.
"""


raise NotImplementedError  # scaffolded in Phase 0; see the build plan
