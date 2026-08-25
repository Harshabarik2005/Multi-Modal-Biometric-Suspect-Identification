"""Fit the occlusion detector and print its coefficients (Phase 11).

    python scripts/train_occlusion.py --images "A:/data/*.jpeg"
    python scripts/train_occlusion.py --images "faces/*.jpg" --paste

Covers each face region with a variety of synthetic occluders, extracts the ITE
feature from covered and uncovered versions, and fits a logistic regression
separating them. Prints an `OCCLUSION_MODEL` block to paste into
`app/embeddings/occlusion.py`.

No scikit-learn. requirements.txt turns it down as "a large dependency for four
functions" and that judgement holds here: a two-class logistic fit is twenty
lines of gradient descent, and keeping it here means the shipped coefficients
are reproducible from this file alone.

Why the occluders vary
----------------------
Training on one flat grey rectangle produces a detector that recognises flat
grey rectangles. The colours, and the two textured occluders, exist so the fit
has to find something more general than an intensity. The textured ones matter
most: a rule based on flatness alone scores AUC 1.000 on every flat occluder
and 0.000 on stripes -- not "no better than chance", but confidently backwards,
because a striped patch has *more* local variation than skin. Anything that
survives here has had to learn texture, not brightness.

Two validations run automatically, and both are printed:

* **Held out by occluder** -- fit without one occluder, test on it. Catches a
  model that has memorised the specific things it was shown.
* **Transfer to `disguise.py`** -- test against the mask, sunglasses and hood
  shapes the robustness harness actually draws, which have edges, shading and
  non-rectangular outlines this training set never contains.

Neither makes this a real-disguise detector. Both are synthetic. What they
establish is that the features generalise beyond the exact thing fitted, which
is the most that can be shown without real disguise footage.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.embeddings.disguise import Disguise, apply_to_region  # noqa: E402
from app.embeddings.face import FaceEmbedder  # noqa: E402
from app.embeddings.occlusion import (  # noqa: E402
    FEATURE_DIM,
    OCCLUDED_THRESHOLD,
    FaceRegion,
    ite_features,
    region_boxes,
)


def occluders(rng: np.random.Generator) -> dict:
    """Synthetic coverings. Flat ones are easy; the textured ones are the test."""
    return {
        "grey": lambda p: np.full_like(p, 150),
        "black": lambda p: np.full_like(p, 20),
        "white": lambda p: np.full_like(p, 235),
        "blue": lambda p: (np.zeros_like(p) + np.array([160, 90, 40], np.uint8)),
        "skin_toned": lambda p: np.full_like(
            p, 0
        ) + np.array([120, 150, 185], np.uint8),
        "noise": lambda p: rng.integers(0, 255, p.shape, dtype=np.uint8),
        "stripes": lambda p: (
            np.where((np.arange(p.shape[0])[:, None, None] // 3) % 2 == 0, 40, 200)
            .astype(np.uint8)
            * np.ones_like(p)
        ),
    }


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 1500,
    lr: float = 0.6,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain batch gradient descent on the logistic loss.

    Standardises first, and returns the standardisation with the weights: the
    detector has to apply the identical transform or the coefficients mean
    nothing. Classes are reweighted because there are far more covered samples
    than clean ones -- one clean patch per region against one per occluder --
    and without it the fit buys accuracy by calling everything covered.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.maximum(std, 1e-8)
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])

    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    sample_weight = np.where(y > 0, len(y) / (2 * positives), len(y) / (2 * negatives))

    w = np.zeros(z.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -60, 60)))
        gradient = z.T @ (sample_weight * (p - y)) / len(y)
        gradient += l2 * np.r_[w[:-1], 0.0]
        w -= lr * gradient
    return w, mean, std


def predict(w, mean, std, features) -> np.ndarray:
    z = np.hstack(
        [(np.asarray(features, dtype=np.float64) - mean) / np.maximum(std, 1e-8),
         np.ones((len(features), 1))]
    )
    return 1.0 / (1.0 + np.exp(-np.clip(z @ w, -60, 60)))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC by rank. 0.5 is chance; below 0.5 means confidently backwards."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives, negatives = labels.sum(), (1 - labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores)
    ranked = labels[order]
    tp = np.cumsum(ranked) / positives
    fp = np.cumsum(1 - ranked) / negatives
    return float(np.trapz(tp, fp))


def face_regions_of(embedder: FaceEmbedder, image: np.ndarray):
    faces = embedder.detect(image)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    box = tuple(float(v) for v in face.bbox)
    return face, region_boxes(box, face.kps)


def collect(embedder, paths, rng):
    """One clean and one covered sample per region per occluder."""
    rows = []
    used = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        face, boxes = face_regions_of(embedder, image)
        if boxes is None:
            print(f"  skipped {Path(path).name}: no face detected")
            continue
        used += 1
        for region, box in boxes.items():
            if not box.usable:
                continue
            clean = box.crop(image)
            if clean.size == 0 or min(clean.shape[:2]) < 8:
                continue
            rows.append((ite_features(clean), 0, region, "none", Path(path).name))
            for name, make in occluders(rng).items():
                covered = image.copy()
                patch = box.crop(covered)
                patch[:] = make(patch)
                rows.append(
                    (ite_features(box.crop(covered)), 1, region, name, Path(path).name)
                )
    return rows, used


def transfer_check(embedder, paths, w, mean, std) -> None:
    """Test against the shapes `disguise.py` actually draws.

    Different rendering path, different shapes, soft edges -- none of it in the
    training set. Reports, per disguise, whether the regions it covers come out
    above the threshold and the regions it leaves alone stay below.
    """
    print("\nTRANSFER TO disguise.py SHAPES (never trained on)")
    print("=" * 72)
    print(f"{'disguise':<22} {'regions called covered':<34} {'clean regions wrong'}")
    print("-" * 72)

    for disguise in Disguise:
        if disguise is Disguise.BLUR:
            # Blur is degradation, not occlusion. Reported separately below.
            continue
        called, false_alarms, faces = [], [], 0
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            face, boxes = face_regions_of(embedder, image)
            if boxes is None:
                continue
            faces += 1
            covered = apply_to_region(
                image.copy(), disguise, tuple(float(v) for v in face.bbox)
            ).image
            for region, box in boxes.items():
                if not box.usable:
                    continue
                patch = box.crop(covered)
                if patch.size == 0 or min(patch.shape[:2]) < 8:
                    continue
                probability = float(predict(w, mean, std, [ite_features(patch)])[0])
                if probability >= OCCLUDED_THRESHOLD:
                    called.append(region)

            for region, box in boxes.items():
                if not box.usable:
                    continue
                patch = box.crop(image)
                if patch.size == 0 or min(patch.shape[:2]) < 8:
                    continue
                probability = float(predict(w, mean, std, [ite_features(patch)])[0])
                if probability >= OCCLUDED_THRESHOLD:
                    false_alarms.append(region)

        counts: dict[FaceRegion, int] = {}
        for region in called:
            counts[region] = counts.get(region, 0) + 1
        summary = ", ".join(
            f"{r.value}({c}/{faces})"
            for r, c in sorted(counts.items(), key=lambda kv: -kv[1])
        ) or "(none)"
        wrong = len(false_alarms) / max(1, faces)
        print(f"{disguise.value:<22} {summary[:33]:<34} {wrong:.2f} per face")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images", action="append", required=True,
        help="Glob of face photographs. Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paste", action="store_true",
        help="Print only the OCCLUSION_MODEL block.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.logging.level)
    rng = np.random.default_rng(args.seed)

    paths = []
    for pattern in args.images:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("No images matched.")
        return 1

    embedder = FaceEmbedder(settings)
    rows, used = collect(embedder, paths, rng)
    if not rows:
        print("No usable face regions found in any image.")
        return 1

    features = np.stack([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])
    if not args.paste:
        print(f"{len(rows)} region samples from {used} face image(s)")
        print(f"  clean={int((labels == 0).sum())} covered={int((labels == 1).sum())}"
              f"  feature dim={FEATURE_DIM}")

        names = sorted({r[3] for r in rows} - {"none"})
        print("\nHELD OUT BY OCCLUDER (fit without it, then test on it)")
        print("=" * 72)
        for held in names:
            train = [r for r in rows if r[3] != held]
            test = [r for r in rows if r[3] in (held, "none")]
            hw, hm, hs = fit_logistic(
                np.stack([r[0] for r in train]), np.array([r[1] for r in train])
            )
            probability = predict(hw, hm, hs, [r[0] for r in test])
            auc = roc_auc(probability, np.array([r[1] for r in test]))
            verdict = (
                "ok" if auc >= 0.9
                else "WEAK" if auc >= 0.6
                else "BACKWARDS -- ranks this covering as cleaner than skin"
            )
            print(f"  {held:<12} AUC={auc:.3f}  {verdict}")

    w, mean, std = fit_logistic(features, labels)

    if not args.paste:
        transfer_check(embedder, paths, w, mean, std)
        print("\n\nPaste into app/embeddings/occlusion.py:")
        print("=" * 72)

    def fmt(values) -> str:
        return "[\n        " + ",\n        ".join(
            ", ".join(f"{v:.6g}" for v in values[i : i + 6])
            for i in range(0, len(values), 6)
        ) + ",\n    ]"

    print("OCCLUSION_MODEL: dict[str, list[float]] = {")
    print(f'    "weights": {fmt(w)},')
    print(f'    "mean": {fmt(mean)},')
    print(f'    "std": {fmt(std)},')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
