"""One-shot scaffolding helper: writes the Phase 2+ placeholder modules.

Run once during Phase 0. Delete it (or just ignore it) afterwards -- it exists
so the repo layout matches section 5 of the build plan from day one, without
any half-working code pretending to be real.
"""

from __future__ import annotations

from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent

HEADER = '"""{title}\n\n{body}\n"""\n'

PLACEHOLDERS: dict[str, tuple[str, str]] = {
    "app/embeddings/face.py": (
        "Face embedding branch (Phase 2) -- ArcFace / InsightFace.",
        "Takes pose-guided face crops from a track and returns a normalised\n"
        "512-d identity embedding. Must be occlusion-aware: masks and\n"
        "sunglasses are expected, not exceptional, and the branch should\n"
        "report a per-frame quality score that Phase 6's attention head can\n"
        "use to decide how far to trust this modality.",
    ),
    "app/embeddings/gait.py": (
        "Gait embedding branch (Phase 3) -- GEI + GaitSet/GaitGL.",
        "Extracts person silhouettes across a track, segments them into gait\n"
        "cycles, averages each cycle into a Gait Energy Image, and encodes\n"
        "the GEI sequence into an embedding. Pretrain on CASIA-B (124\n"
        "subjects, 11 view angles, bag/coat conditions) rather than the\n"
        "paper's CASIA-A.",
    ),
    "app/embeddings/reid.py": (
        "Re-ID embedding branch (Phase 4) -- OSNet via torchreid.",
        "Whole-body appearance embedding: build, clothing, gross shape. This\n"
        "is the modality that degrades fastest over time -- Phase 11 adds a\n"
        "trust decay so a sighting weeks apart is weighted well below one\n"
        "from the same day.",
    ),
    "app/fusion/baseline.py": (
        "Baseline fusion (Phase 5) -- the number the real contribution beats.",
        "Simple rule-based and average fusion over the three modality\n"
        "embeddings, mirroring the 'average fusion' row of the reference\n"
        "paper's Table I. Deliberately dumb: fixed weights, no learning.",
    ),
    "app/fusion/attention.py": (
        "Keyless attention fusion (Phase 6) -- the core contribution.",
        "Small trainable head that scores each modality per frame, normalises\n"
        "those into global weights (alpha for face, beta for gait, gamma for\n"
        "re-ID), and produces one adaptive embedding.\n\n"
        "Two deliberate divergences from the reference paper:\n"
        "  * three modalities rather than two;\n"
        "  * trained with a metric-learning loss (ArcFace-margin or triplet)\n"
        "    for open-set matching, instead of a closed-set softmax over a\n"
        "    fixed 20-person roster.\n\n"
        "The per-modality weights are part of the output contract, not an\n"
        "internal detail -- the dashboard surfaces them so an investigator\n"
        "can see why a match fired.",
    ),
    "app/api/routes.py": (
        "FastAPI routes (Phase 7).",
        "Endpoints: enrollment, matching, watchlist CRUD, alert history.\n"
        "Every match response carries the attention weights and the\n"
        "contributing frame references, so the human-confirm step in the UI\n"
        "has something to reason about.",
    ),
    "app/db/models.py": (
        "Persistence models (Phase 7).",
        "Watchlist people, their per-modality reference embeddings, match\n"
        "decisions, and the audit log.\n\n"
        "Two constraints from section 8 that belong in the schema itself, not\n"
        "in application code: biometric templates are encrypted at rest, and\n"
        "every match decision is written to an append-only audit record\n"
        "including the attention weights and the confirming operator.",
    ),
    "app/alerts/notifier.py": (
        "Alerting (Phase 9) -- Twilio / SMTP.",
        "Fires only after a human confirms a match. No automated action is\n"
        "taken on a model decision alone (section 8).",
    ),
    "../eval/metrics.py": (
        "Open-set evaluation metrics (Phase 10).",
        "TAR@FAR, ROC-AUC, and CMC / Rank-N -- the metrics that actually\n"
        "describe watchlist performance. Plain accuracy and log-loss, which\n"
        "the reference paper reports, do not transfer to an open-set problem\n"
        "where most people passing a camera are not on the list.\n\n"
        "Section 8 also requires a demographic fairness breakdown here, not\n"
        "just aggregate numbers.",
    ),
    "../eval/ablation.py": (
        "Ablation harness (Phase 10).",
        "Mirrors Table I of the reference paper: each modality alone, then\n"
        "naive/average fusion, then keyless attention fusion, on identical\n"
        "splits. This is what shows the attention head earns its complexity.",
    ),
}

TODO = "\n\nraise NotImplementedError  # scaffolded in Phase 0; see the build plan\n"


def main() -> int:
    for rel, (title, body) in PLACEHOLDERS.items():
        path = (BACKEND / rel).resolve()
        if path.exists() and path.stat().st_size > 0:
            print(f"skip (exists) {path.relative_to(REPO)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER.format(title=title, body=body) + TODO, encoding="utf-8")
        print(f"wrote        {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
