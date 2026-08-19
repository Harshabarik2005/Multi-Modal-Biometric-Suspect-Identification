"""Open-set evaluation metrics (Phase 10).

TAR@FAR, ROC-AUC, and CMC / Rank-N -- the metrics that actually
describe watchlist performance. Plain accuracy and log-loss, which
the reference paper reports, do not transfer to an open-set problem
where most people passing a camera are not on the list.

Section 8 also requires a demographic fairness breakdown here, not
just aggregate numbers.
"""


raise NotImplementedError  # scaffolded in Phase 0; see the build plan
