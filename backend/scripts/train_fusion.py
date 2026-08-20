"""Train the keyless attention fusion head (Phase 6).

    python scripts/train_fusion.py --synthetic          # demo on synthetic data
    python scripts/train_fusion.py --synthetic --triplets 3000 --epochs 120

WHAT THE SYNTHETIC MODE DOES AND DOES NOT PROVE
-----------------------------------------------
It proves the machinery: the head trains, the loss falls, absent modalities are
handled, and it learns to weight an informative modality above a merely
available one. That last point is the whole contribution -- the Phase-5 fixed
rules were measured giving re-ID MORE weight than face (0.54 vs 0.46) because
they weight by how good a look you got, not by how much the modality is worth.

It proves NOTHING about real footage. The synthetic identities are random
vectors with tuned noise; real face, gait and re-ID embeddings have structure
this does not reproduce. A head trained here must not be used for real
matching, and the training report says so.

Training on real data needs labelled same/different pairs from actual footage:
the same person seen twice, and other people. That is the same blocker as
gait's positive validation, and it is where real data stops being optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.types import Modality  # noqa: E402
from app.fusion.attention import KeylessAttentionFusion  # noqa: E402
from app.fusion.training import (  # noqa: E402
    evaluate_separation,
    synthetic_triplets,
    train,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Train on generated data. Required until real triplets exist.",
    )
    parser.add_argument("--triplets", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--shared-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to save the head. Defaults to models_dir/attention_fusion.pt",
    )
    parser.add_argument(
        "--save-anyway", action="store_true",
        help="Save even when the head is badly overfitted.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.logging.level)

    if not args.synthetic:
        print(
            "Real-data training is not implemented: it needs labelled\n"
            "same/different pairs from actual footage, which do not exist yet.\n"
            "Run with --synthetic to exercise the training machinery.\n\n"
            "To build real triplets you need the same person recorded twice\n"
            "(different times, ideally different cameras) plus other people,\n"
            "run through the pipeline to collect per-modality embeddings."
        )
        return 2

    dims = {Modality.FACE: 64, Modality.GAIT: 64, Modality.REID: 64}
    print(
        f"Generating {args.triplets} synthetic triplets "
        f"(face is least noisy but often absent; re-ID is noisier but almost "
        f"always present)...\n"
    )
    triplets = synthetic_triplets(args.triplets, dims, seed=args.seed)

    model = KeylessAttentionFusion(dims, shared_dim=args.shared_dim, seed=args.seed)
    report = train(
        model,
        triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        patience=args.patience,
        seed=args.seed,
    )

    print()
    print("=" * 66)
    print("TRAINING REPORT")
    print("=" * 66)
    for line in report.summary_lines():
        print(line)

    held_out = synthetic_triplets(400, dims, seed=args.seed + 9999)
    separation = evaluate_separation(model, held_out)["separation"]
    print(f"\nheld-out separation : {separation:+.4f}")

    face = report.mean_weights_when_present.get(Modality.FACE, 0.0)
    reid = report.mean_weights_when_present.get(Modality.REID, 0.0)
    print()
    if face > reid:
        print(
            f"The head learned to weight face ({face:.3f}) above re-ID "
            f"({reid:.3f}) when both\nare present -- the behaviour the Phase-5 "
            "fixed rules got backwards."
        )
    else:
        print(
            f"WARNING: the head did NOT learn to prefer face ({face:.3f}) over "
            f"re-ID ({reid:.3f}).\nMeasured across seeds, this happens when the "
            "head is under-resourced: 64-d\nmodalities with a 128-d shared "
            "space and 1500+ triplets got it right 6/6\ntimes, while a 64-d "
            "shared space with 800 triplets managed only 2/6. Try\n"
            "--shared-dim 128 --triplets 1500 or more."
        )

    overfitted = report.overfitting_ratio >= 3.0
    print("=" * 66)
    print(
        "This head is trained on SYNTHETIC data. It must not be used for real\n"
        "matching -- the matching CLI will refuse it unless you point at it\n"
        "explicitly, and it would be worse than the Phase-5 fixed rules."
    )
    print("=" * 66)

    if overfitted and not args.save_anyway:
        print(
            f"\nNot saving: train/validation separation ratio is "
            f"{report.overfitting_ratio:.2f}x, which means the head has largely\n"
            "memorised its training triplets. Use --save-anyway to override, or\n"
            "train with more data."
        )
        return 1

    out = args.out or (settings.paths.models_dir / "attention_fusion.pt")
    model.save(out)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
