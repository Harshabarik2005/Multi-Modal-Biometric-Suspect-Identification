"""Gait embedding branch (Phase 3) -- GEI + GaitSet/GaitGL.

Extracts person silhouettes across a track, segments them into gait
cycles, averages each cycle into a Gait Energy Image, and encodes
the GEI sequence into an embedding. Pretrain on CASIA-B (124
subjects, 11 view angles, bag/coat conditions) rather than the
paper's CASIA-A.
"""


raise NotImplementedError  # scaffolded in Phase 0; see the build plan
