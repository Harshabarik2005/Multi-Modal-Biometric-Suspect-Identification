"""Open-set evaluation metrics (Phase 10).

TAR@FAR, ROC-AUC and CMC / Rank-N -- the metrics that actually describe
watchlist performance.

Why not accuracy
----------------
The reference paper reports accuracy and log-loss over a closed set of 20
people. Neither transfers here. In an open-set watchlist almost everyone who
walks past a camera is on nobody's list, so a system that matched *nobody, ever*
would score extremely well on accuracy while being completely useless. The
question is not "what fraction did it get right" but "at a false-alarm rate we
can actually live with, what fraction of the people we are looking for does it
find" -- which is TAR@FAR.

The vocabulary
--------------
* **Genuine score** -- a comparison between two observations of the same person.
* **Impostor score** -- a comparison between different people.
* **FAR** (false accept rate) -- the fraction of impostor comparisons scoring
  above the threshold. In deployment these are people wrongly flagged.
* **TAR** (true accept rate) -- the fraction of genuine comparisons above it.
* **CMC / Rank-N** -- given a probe whose true match *is* enrolled, how often
  does the correct person appear in the top N.

TAR@FAR and CMC answer different questions and a system can be good at one and
poor at the other. CMC assumes the person is enrolled; TAR@FAR does not, which
is why it is the one that matters for a watchlist.

A note on what a false accept costs
-----------------------------------
FAR is not a symmetric error rate here. A false accept means a person who is on
no watchlist is flagged as someone who is. Choose operating points on that
basis, not on whichever number looks best.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Operating points worth reporting. 1e-3 is a common headline figure; on a
# camera seeing 10,000 people a day it still means ~10 false flags daily.
DEFAULT_FARS: tuple[float, ...] = (0.1, 0.01, 0.001, 0.0001)


@dataclass
class ROCPoint:
    threshold: float
    far: float
    tar: float


@dataclass
class VerificationReport:
    """Genuine-vs-impostor separation: the open-set verification view."""

    genuine_count: int = 0
    impostor_count: int = 0
    auc: float = 0.0
    #: TAR at each requested FAR, plus the threshold that achieves it.
    tar_at_far: dict[float, float] = field(default_factory=dict)
    threshold_at_far: dict[float, float] = field(default_factory=dict)
    eer: float = 0.0
    eer_threshold: float = 0.0
    genuine_mean: float = 0.0
    impostor_mean: float = 0.0
    roc: list[ROCPoint] = field(default_factory=list)

    #: The finest FAR this many impostor pairs can actually distinguish.
    #: 500 pairs cannot resolve anything below 1/500.
    smallest_resolvable_far: float = 1.0
    #: Requested FARs below that floor. Their TAR figures are artefacts of the
    #: sample size and should not be read as results.
    unresolvable_fars: list[float] = field(default_factory=list)

    @property
    def separation(self) -> float:
        return self.genuine_mean - self.impostor_mean

    def summary_lines(self) -> list[str]:
        lines = [
            f"genuine pairs   : {self.genuine_count}",
            f"impostor pairs  : {self.impostor_count}",
            f"ROC-AUC         : {self.auc:.4f}",
            f"EER             : {self.eer:.4f}  (threshold {self.eer_threshold:.3f})",
            f"mean genuine    : {self.genuine_mean:+.4f}",
            f"mean impostor   : {self.impostor_mean:+.4f}",
            f"separation      : {self.separation:+.4f}",
            "",
            "TAR at fixed FAR (threshold in brackets):",
        ]
        for far in sorted(self.tar_at_far, reverse=True):
            flag = "  (below resolution)" if far in self.unresolvable_fars else ""
            lines.append(
                f"  FAR {far:<8g} TAR {self.tar_at_far[far]:.4f}   "
                f"[{self.threshold_at_far[far]:+.3f}]{flag}"
            )
        if self.unresolvable_fars:
            lines.append(
                f"  {self.impostor_count} impostor pairs cannot resolve below "
                f"FAR {self.smallest_resolvable_far:g}."
            )
        return lines


@dataclass
class IdentificationReport:
    """CMC: where the correct person lands in the ranked list."""

    probe_count: int = 0
    ranks: dict[int, float] = field(default_factory=dict)
    mean_rank: float = 0.0

    def rank_n(self, n: int) -> float:
        return self.ranks.get(n, 0.0)

    def summary_lines(self) -> list[str]:
        lines = [f"probes          : {self.probe_count}", "CMC:"]
        for n in sorted(self.ranks):
            lines.append(f"  Rank-{n:<3} {self.ranks[n]:.4f}")
        lines.append(f"mean rank       : {self.mean_rank:.2f}")
        return lines


def roc_curve(
    genuine: np.ndarray, impostor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FAR, TAR and threshold arrays, ordered by increasing threshold.

    Thresholds are every distinct observed score, so the curve passes exactly
    through every achievable operating point rather than an arbitrary grid.
    """
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    impostor = np.asarray(impostor, dtype=np.float64).ravel()
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError(
            "Both genuine and impostor scores are required. With only one of "
            "them there is no trade-off to measure."
        )

    thresholds = np.unique(np.concatenate([genuine, impostor]))
    # Descending so the curve runs from accept-everything to accept-nothing.
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    tar = np.array([(genuine >= t).mean() for t in thresholds])
    far = np.array([(impostor >= t).mean() for t in thresholds])
    return far, tar, thresholds


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Area under the ROC curve.

    Computed as the Mann-Whitney statistic -- the probability that a random
    genuine pair outscores a random impostor pair -- with ties counted as half.
    That is exact, where trapezoidal integration of a sampled curve is not.
    """
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    impostor = np.asarray(impostor, dtype=np.float64).ravel()
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Both genuine and impostor scores are required.")

    combined = np.concatenate([genuine, impostor])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)

    # Average ranks within tied groups, so ties contribute 0.5 rather than 1.
    sorted_values = combined[order]
    start = 0
    for index in range(1, sorted_values.size + 1):
        if index == sorted_values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    genuine_rank_sum = ranks[: genuine.size].sum()
    u = genuine_rank_sum - genuine.size * (genuine.size + 1) / 2.0
    return float(u / (genuine.size * impostor.size))


def tar_at_far(
    genuine: np.ndarray, impostor: np.ndarray, target_far: float
) -> tuple[float, float]:
    """TAR at the tightest threshold whose FAR does not exceed `target_far`.

    Returns `(tar, threshold)`. Picking the threshold from the impostor
    distribution and *then* measuring TAR is the honest order: choosing it to
    flatter TAR would report a false-alarm rate the system does not achieve.
    """
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    impostor = np.asarray(impostor, dtype=np.float64).ravel()
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Both genuine and impostor scores are required.")

    far, tar, thresholds = roc_curve(genuine, impostor)
    admissible = np.where(far <= target_far)[0]
    if admissible.size == 0:
        # Even accepting nothing exceeds the target, which happens when the
        # impostor set is too small to resolve this FAR at all.
        return 0.0, float("inf")

    # Among thresholds meeting the FAR budget, take the one with the best TAR.
    best = admissible[np.argmax(tar[admissible])]
    return float(tar[best]), float(thresholds[best])


def equal_error_rate(
    genuine: np.ndarray, impostor: np.ndarray
) -> tuple[float, float]:
    """The rate where false accepts and false rejects are equal."""
    far, tar, thresholds = roc_curve(genuine, impostor)
    frr = 1.0 - tar
    index = int(np.argmin(np.abs(far - frr)))
    return float((far[index] + frr[index]) / 2.0), float(thresholds[index])


def evaluate_verification(
    genuine: np.ndarray,
    impostor: np.ndarray,
    fars: tuple[float, ...] = DEFAULT_FARS,
) -> VerificationReport:
    """Full verification report from genuine and impostor score arrays."""
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    impostor = np.asarray(impostor, dtype=np.float64).ravel()

    far_curve, tar_curve, thresholds = roc_curve(genuine, impostor)
    eer, eer_threshold = equal_error_rate(genuine, impostor)

    report = VerificationReport(
        genuine_count=genuine.size,
        impostor_count=impostor.size,
        auc=roc_auc(genuine, impostor),
        eer=eer,
        eer_threshold=eer_threshold,
        genuine_mean=float(genuine.mean()),
        impostor_mean=float(impostor.mean()),
        roc=[
            ROCPoint(threshold=float(t), far=float(f), tar=float(r))
            for f, r, t in zip(far_curve, tar_curve, thresholds)
        ],
    )

    for far in fars:
        tar, threshold = tar_at_far(genuine, impostor, far)
        report.tar_at_far[far] = tar
        report.threshold_at_far[far] = threshold

    # A FAR finer than the impostor set can resolve is not measurable: with
    # 500 impostor pairs the smallest non-zero FAR is 1/500 = 0.002, and
    # anything below that is an artefact of the sample size, not a result.
    # This used to read `report.tar_at_far[far] = report.tar_at_far.get(far, 0.0)`
    # -- a self-assignment that changed nothing, so unmeasurable operating
    # points were reported as though they were real.
    report.smallest_resolvable_far = 1.0 / impostor.size if impostor.size else 1.0
    report.unresolvable_fars = sorted(
        far for far in fars if far < report.smallest_resolvable_far
    )
    return report


def evaluate_identification(
    score_matrix: np.ndarray,
    true_indices: np.ndarray,
    max_rank: int = 10,
) -> IdentificationReport:
    """CMC from a (probes x gallery) score matrix.

    `true_indices[i]` is the gallery column holding probe `i`'s real identity.
    Every probe must have its mate enrolled -- that is what CMC measures, and
    it is why CMC alone overstates real watchlist performance, where most
    people have no mate at all.
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    true_indices = np.asarray(true_indices, dtype=int).ravel()

    if score_matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D score matrix, got shape {score_matrix.shape}")
    if score_matrix.shape[0] != true_indices.size:
        raise ValueError(
            f"{score_matrix.shape[0]} probes but {true_indices.size} labels."
        )

    probes, gallery_size = score_matrix.shape
    if probes == 0:
        return IdentificationReport()

    # Rank of the true identity: how many gallery entries outscore it, plus 1.
    true_scores = score_matrix[np.arange(probes), true_indices]
    ranks = (score_matrix > true_scores[:, None]).sum(axis=1) + 1

    report = IdentificationReport(probe_count=probes, mean_rank=float(ranks.mean()))
    for n in range(1, min(max_rank, gallery_size) + 1):
        report.ranks[n] = float((ranks <= n).mean())
    return report


def pairwise_scores(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split all pairwise cosine similarities into genuine and impostor sets.

    Assumes rows are L2-normalised, which every branch in this project
    guarantees, so the Gram matrix is the cosine similarity matrix.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if embeddings.shape[0] != labels.size:
        raise ValueError(
            f"{embeddings.shape[0]} embeddings but {labels.size} labels."
        )

    similarity = embeddings @ embeddings.T
    same = labels[:, None] == labels[None, :]
    # Upper triangle only: a pair is one comparison, and the diagonal is a
    # vector against itself, which is not a comparison at all.
    upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)

    genuine = similarity[upper & same]
    impostor = similarity[upper & ~same]
    return genuine, impostor


@dataclass
class GroupReport:
    """One group's error rates at the SHARED operating threshold."""

    group: str
    genuine_count: int
    impostor_count: int
    #: Fraction of this group's genuine pairs accepted at the shared threshold.
    tar: float
    #: Fraction of this group's impostor pairs accepted at it.
    far: float
    auc: float
    threshold: float
    #: True when the requested FAR was unachievable and the equal-error point
    #: was used instead. Without this, a table of zeros looks like parity.
    threshold_is_fallback: bool = False
    target_far: float = 0.01

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.tar


def fairness_breakdown(
    genuine: np.ndarray,
    impostor: np.ndarray,
    genuine_groups: np.ndarray,
    impostor_groups: np.ndarray,
    target_far: float = 0.01,
) -> dict[str, GroupReport]:
    """Per-group TAR and FAR at ONE shared threshold (build plan, section 8).

    The threshold is derived once, from the pooled impostor distribution, and
    then applied to every group -- because that is what a deployment does. It
    sets one threshold and everyone is judged against it.

    This previously called `evaluate_verification` per group, which derives a
    fresh threshold from each group's own impostor scores. That measures each
    group at its own private operating point, which no deployment uses, and it
    inverted the finding: a group that was being missed 57% of the time at the
    real threshold reported TAR 1.000, while a group being falsely flagged
    reported 0.824. The number that matters is what happens to each group at
    the threshold actually in use.

    Both TAR and FAR are reported per group. They are different harms -- a low
    TAR means a group is missed, a high FAR means a group is falsely flagged --
    and reporting only one hides half the problem.

    Groups are supplied by the caller. This code does not and must not attempt
    to infer demographic attributes from biometric data.
    """
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    impostor = np.asarray(impostor, dtype=np.float64).ravel()
    genuine_groups = np.asarray(genuine_groups).ravel()
    impostor_groups = np.asarray(impostor_groups).ravel()

    if genuine.size != genuine_groups.size or impostor.size != impostor_groups.size:
        raise ValueError("Scores and group labels must be the same length.")
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Both genuine and impostor scores are required.")

    # One threshold, from the pooled impostor set: the operating point a
    # deployment would actually choose.
    pooled_tar, shared_threshold = tar_at_far(genuine, impostor, target_far)

    # If the requested FAR is unachievable, every group scores TAR 0 and the
    # table reads as "all groups equal" when it actually means "this operating
    # point does not exist on this data". Fall back to the equal-error point,
    # which always exists, and record that it happened.
    fell_back = False
    if not np.isfinite(shared_threshold) or pooled_tar <= 0.0:
        _, shared_threshold = equal_error_rate(genuine, impostor)
        fell_back = True

    reports: dict[str, GroupReport] = {}
    for group in sorted(set(genuine_groups.tolist()) | set(impostor_groups.tolist())):
        group_genuine = genuine[genuine_groups == group]
        group_impostor = impostor[impostor_groups == group]
        if group_genuine.size == 0 or group_impostor.size == 0:
            continue

        reports[str(group)] = GroupReport(
            group=str(group),
            genuine_count=int(group_genuine.size),
            impostor_count=int(group_impostor.size),
            tar=float((group_genuine >= shared_threshold).mean()),
            far=float((group_impostor >= shared_threshold).mean()),
            auc=roc_auc(group_genuine, group_impostor),
            threshold=float(shared_threshold),
            threshold_is_fallback=fell_back,
            target_far=target_far,
        )
    return reports


def fairness_summary(reports: dict[str, GroupReport]) -> list[str]:
    """Readable table, with the spread called out.

    The spread is the finding. An aggregate number can look strong while one
    group is served far worse, and that is the whole reason this breakdown
    exists.
    """
    if not reports:
        return ["no group had both genuine and impostor pairs"]

    first = next(iter(reports.values()))
    threshold = first.threshold
    lines = [
        f"All groups judged at the SAME threshold ({threshold:+.3f}), which is",
        "what a deployment does. TAR is how often the group is correctly found;",
        "FAR is how often it is falsely flagged. Both matter, differently.",
    ]
    if first.threshold_is_fallback:
        lines.append("")
        lines.append(
            f"NOTE: FAR {first.target_far:g} is unachievable on this data -- no "
            "threshold reaches\nit with any true accepts at all. Using the "
            "equal-error point instead.\nA row of zeros here would have looked "
            "like parity between groups; it\nwould have meant the operating "
            "point does not exist."
        )
    lines.extend([
        "",
        f"{'group':<14} {'TAR':>8} {'FAR':>8} {'AUC':>8} {'genuine':>8} {'impostor':>9}",
        "-" * 60,
    ])
    for report in sorted(reports.values(), key=lambda r: r.group):
        lines.append(
            f"{report.group:<14} {report.tar:>8.3f} {report.far:>8.3f} "
            f"{report.auc:>8.4f} {report.genuine_count:>8} {report.impostor_count:>9}"
        )

    tars = [r.tar for r in reports.values()]
    fars = [r.far for r in reports.values()]
    if len(reports) > 1:
        lines.append("")
        lines.append(f"TAR spread: {max(tars) - min(tars):.3f}   "
                     f"FAR spread: {max(fars) - min(fars):.3f}")
        worst_missed = min(reports.values(), key=lambda r: r.tar)
        worst_flagged = max(reports.values(), key=lambda r: r.far)
        if max(tars) - min(tars) > 0.1:
            lines.append(
                f"  {worst_missed.group} is MISSED most often "
                f"({worst_missed.miss_rate:.0%} of the time)."
            )
        if max(fars) - min(fars) > 0.01:
            lines.append(
                f"  {worst_flagged.group} is FALSELY FLAGGED most often "
                f"({worst_flagged.far:.1%} of its impostor pairs)."
            )
    return lines
