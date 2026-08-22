"""Evaluation harness CLI (Phase 10).

    python eval/run_evaluation.py --synthetic
    python eval/run_evaluation.py --synthetic --people 60 --sightings 6 --fairness

Produces the ablation table -- each modality alone, then each fusion rule --
with TAR@FAR, ROC-AUC, EER and CMC on identical pairs.

WHAT --synthetic PROVES AND DOES NOT PROVE
------------------------------------------
It proves the harness: that the metrics compute correctly, that the ablation
compares like with like, and that a more informative modality produces a better
curve.

It proves NOTHING about this system's real accuracy. Synthetic identities are
random vectors with tuned noise. Real face, gait and re-ID embeddings have
structure this does not reproduce, and the noise levels here are chosen, not
measured.

To evaluate for real you need labelled footage: several people, each recorded
more than once, ideally at different times and from different cameras. Run them
through the pipeline, collect per-modality embeddings per sighting, and build
`Observation` records from that. Everything downstream of that point already
works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "backend") not in sys.path:
    sys.path.insert(0, str(REPO / "backend"))

from app.core.types import Modality  # noqa: E402
from app.fusion.calibration import ModalityCalibration  # noqa: E402
from eval.ablation import Observation, build_pairs, run_ablation  # noqa: E402
from eval.metrics import (  # noqa: E402
    evaluate_identification,
    fairness_breakdown,
    fairness_summary,
)

# Noise per modality, mirroring the ordering measured in phases 2-5: face is by
# far the most discriminative, re-ID the least, gait in between but weak.
NOISE = {Modality.FACE: 0.18, Modality.GAIT: 0.50, Modality.REID: 0.42}
AVAILABILITY = {Modality.FACE: 0.55, Modality.GAIT: 0.45, Modality.REID: 0.95}
DIM = 64


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def synthetic_population(
    people: int, sightings: int, seed: int = 0, groups: int = 0
) -> list[Observation]:
    """Several people, each seen several times, with modalities coming and going."""
    generator = np.random.default_rng(seed)
    observations: list[Observation] = []

    for index in range(people):
        latent = {m: unit(generator.normal(size=DIM)) for m in NOISE}
        group = f"group-{index % groups}" if groups else ""

        # A per-person quality offset, so some people are systematically harder
        # to capture. Without this every person is equally easy and the
        # fairness breakdown has nothing to find.
        difficulty = generator.uniform(0.8, 1.3) if groups else 1.0

        for _ in range(sightings):
            observation = Observation(label=f"person-{index}", group=group)
            for modality, vector in latent.items():
                if generator.random() > AVAILABILITY[modality]:
                    continue
                scale = NOISE[modality] * difficulty * generator.uniform(0.6, 1.4)
                observation.embeddings[modality] = unit(
                    vector + generator.normal(scale=scale, size=DIM)
                ).astype(np.float32)
                observation.qualities[modality] = float(
                    np.clip(1.0 - scale, 0.0, 1.0)
                )
            if observation.embeddings:
                observations.append(observation)

    return observations


def calibrations_from_pairs(pairs) -> dict[Modality, ModalityCalibration]:
    """Derive calibration anchors from the data being evaluated.

    Using the project's configured anchors would be wrong here: they were
    measured on real footage, and these are synthetic scores on a different
    scale. Anchors must come from the same distribution as the scores they
    calibrate, or the fusion rows are comparing miscalibrated inputs and the
    ablation measures the mismatch rather than the fusion.
    """
    from eval.ablation import score_single_modality

    result: dict[Modality, ModalityCalibration] = {}
    for modality in (Modality.FACE, Modality.GAIT, Modality.REID):
        genuine, impostor, _ = score_single_modality(pairs, modality)
        if genuine.size == 0 or impostor.size == 0:
            continue
        impostor_anchor = float(np.percentile(impostor, 95))
        genuine_anchor = float(np.percentile(genuine, 50))
        if genuine_anchor <= impostor_anchor:
            # This modality cannot separate at all on this data; give it a
            # minimal valid range rather than crashing the whole run.
            genuine_anchor = impostor_anchor + 1e-3
        result[modality] = ModalityCalibration(
            modality, impostor_anchor, genuine_anchor
        )
    return result


def identification_view(observations: list[Observation]) -> None:
    """CMC using face alone, as a worked example of the rank metrics."""
    gallery: dict[str, Observation] = {}
    probes: list[Observation] = []
    for observation in observations:
        if not observation.has(Modality.FACE):
            continue
        if observation.label not in gallery:
            gallery[observation.label] = observation
        else:
            probes.append(observation)

    if len(gallery) < 2 or not probes:
        print("  not enough face observations for a CMC curve")
        return

    labels = list(gallery)
    matrix = np.array(
        [
            [
                float(
                    np.dot(
                        probe.embeddings[Modality.FACE],
                        gallery[label].embeddings[Modality.FACE],
                    )
                )
                for label in labels
            ]
            for probe in probes
        ]
    )
    true_indices = np.array([labels.index(p.label) for p in probes])
    report = evaluate_identification(matrix, true_indices, max_rank=5)
    for line in report.summary_lines():
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Evaluate on generated data. Required until real footage exists.",
    )
    parser.add_argument("--people", type=int, default=40)
    parser.add_argument("--sightings", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fairness", action="store_true",
        help="Also break results down by demographic group.",
    )
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument(
        "--attention-head", type=Path, default=None,
        help="Path to a trained attention head to include in the ablation.",
    )
    args = parser.parse_args(argv)

    if not args.synthetic:
        print(
            "Real-data evaluation needs labelled footage that does not exist yet:\n"
            "several people, each recorded more than once, ideally at different\n"
            "times and cameras. Run them through the pipeline, collect\n"
            "per-modality embeddings per sighting, and build eval.ablation\n"
            "Observation records. Everything downstream already works.\n\n"
            "Run with --synthetic to exercise the harness."
        )
        return 2

    observations = synthetic_population(
        args.people, args.sightings, args.seed, args.groups if args.fairness else 0
    )
    pairs = build_pairs(observations, max_pairs=args.max_pairs, seed=args.seed)
    genuine_pairs = sum(1 for p in pairs if p.same)

    print(
        f"{len(observations)} sightings of {args.people} people -> "
        f"{len(pairs)} pairs ({genuine_pairs} genuine, "
        f"{len(pairs) - genuine_pairs} impostor)\n"
    )

    attention = None
    if args.attention_head and args.attention_head.is_file():
        from app.fusion.attention import KeylessAttentionFusion

        attention = KeylessAttentionFusion.load(args.attention_head)
        print(f"Including attention head from {args.attention_head}\n")

    calibrations = calibrations_from_pairs(pairs)
    result = run_ablation(pairs, calibrations, attention_model=attention)

    print("=" * 90)
    print("ABLATION 1 of 2 -- NATIVE COVERAGE")
    print("Each row on the pairs it can actually score. This is how useful each")
    print("configuration is in deployment, but the AUCs are NOT comparable across")
    print("rows, because a rarely-available modality is graded on easy cases only.")
    print("=" * 90)
    print(result.table())
    print("=" * 90)

    from eval.ablation import common_subset

    shared = common_subset(pairs)
    shared_genuine = sum(1 for p in shared if p.same)
    print()
    print("=" * 90)
    print("ABLATION 2 of 2 -- COMMON SUBSET (all three modalities present)")
    print(
        f"{len(shared)} pairs ({shared_genuine} genuine). Every row sees exactly"
    )
    print("these, so the AUCs ARE comparable. This is the apples-to-apples view --")
    print("and an unrepresentatively easy one, since a hidden face is the norm.")
    print("=" * 90)
    if shared_genuine and shared_genuine < 30:
        print(
            f"  WARNING: only {shared_genuine} genuine pairs. TAR resolves in steps\n"
            f"  of {1 / shared_genuine:.1%}, so the TAR columns are coarse here and\n"
            "  small gaps between rows are noise. Raise --people / --sightings.\n"
        )
    if len(shared) < 20 or shared_genuine < 5:
        print(
            f"  Not enough pairs with all three modalities present "
            f"({len(shared)}, {shared_genuine} genuine).\n"
            "  Raise --people or --sightings to get a usable common subset."
        )
    else:
        shared_result = run_ablation(
            shared, calibrations, attention_model=attention
        )
        print(shared_result.table(note_coverage=False))
    print("=" * 90)

    # The finding the two tables exist to make visible.
    face_row = result.row("face only")
    fusion_row = result.row("fusion:quality_weighted")
    if face_row and fusion_row and face_row.report and fusion_row.report:
        print()
        print("WHAT FUSION ACTUALLY BUYS")
        print(
            f"  face alone : AUC {face_row.auc:.4f} on {face_row.coverage:.0%} of pairs"
        )
        print(
            f"  fused      : AUC {fusion_row.auc:.4f} on "
            f"{fusion_row.coverage:.0%} of pairs"
        )
        if face_row.auc > fusion_row.auc and fusion_row.coverage > face_row.coverage:
            print(
                "\n  Face is the stronger signal WHEN IT IS THERE, and fusion does\n"
                "  not beat it on the pairs where both can be scored. What fusion\n"
                "  buys is COVERAGE: a usable score on "
                f"{fusion_row.coverage:.0%} of pairs against\n"
                f"  face's {face_row.coverage:.0%}. In deployment the alternative to a "
                "fused score on a\n"
                "  turned-away person is not a better score -- it is NO score at all,\n"
                "  which is the entire premise of this project."
            )

    print("\nIDENTIFICATION (CMC, face only, as a worked example)")
    identification_view(observations)

    if args.fairness:
        print("\n" + "=" * 78)
        print("FAIRNESS BREAKDOWN  (build plan, section 8)")
        print("=" * 78)
        from eval.ablation import score_fusion

        genuine, impostor, _ = score_fusion(pairs, "quality_weighted", calibrations)

        # Collect each score's group in the same order score_fusion produced
        # them, so a score and its label cannot drift apart. Cross-group pairs
        # carry no group and are dropped from BOTH lists together: a pair
        # spanning two groups belongs to neither, and assigning it to one
        # measures that group's false-accept rate on the wrong population.
        kept_genuine, kept_impostor = [], []
        genuine_groups, impostor_groups = [], []
        genuine_index = impostor_index = 0
        cross_group = 0

        for pair in pairs:
            if not any(pair.a.has(m) and pair.b.has(m) for m in calibrations):
                continue  # score_fusion skipped this one too
            if pair.same:
                score = genuine[genuine_index] if genuine_index < len(genuine) else None
                genuine_index += 1
                target, labels = kept_genuine, genuine_groups
            else:
                score = (
                    impostor[impostor_index] if impostor_index < len(impostor) else None
                )
                impostor_index += 1
                target, labels = kept_impostor, impostor_groups

            if score is None:
                continue
            if not pair.group:
                cross_group += 1
                continue
            target.append(score)
            labels.append(pair.group)

        if cross_group:
            print(
                f"{cross_group} cross-group pairs excluded -- a pair spanning "
                "two groups\nbelongs to neither.\n"
            )

        if len(kept_genuine) < 2 or len(kept_impostor) < 2:
            print(
                "Not enough within-group pairs to break down. Raise --people, "
                "or use fewer --groups."
            )
        else:
            reports = fairness_breakdown(
                np.array(kept_genuine),
                np.array(kept_impostor),
                np.array(genuine_groups),
                np.array(impostor_groups),
            )
            for line in fairness_summary(reports):
                print(line)

    print(
        "\nNOTE: these are SYNTHETIC identities with chosen noise levels. The\n"
        "harness is validated; this system's real accuracy is not. That needs\n"
        "labelled footage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
