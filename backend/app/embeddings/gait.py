"""Gait embedding branch (Phase 3) -- silhouettes, gait cycles, GEI.

Unlike face and re-ID, this branch implements `EmbeddingBranch.embed()`
directly over a whole sequence, because a single frame of a person walking
carries no gait information at all. That asymmetry is the reason the branch
interface takes sequences rather than frames.

Pipeline
--------
1. **Silhouettes** -- YOLOv8-seg cuts the person out of each buffered crop,
   normalised onto the standard 64x44 gait canvas (see `silhouette.py`).
2. **Gait cycles** -- the silhouette's width oscillates as the legs open and
   close. Autocorrelating that signal recovers the cadence without needing to
   find individual footfalls, which is far more robust at CCTV resolution than
   peak-picking.
3. **Gait Energy Image** -- silhouettes averaged over whole cycles. Averaging
   over a *whole* number of cycles matters: a partial cycle biases the GEI
   toward whichever leg happened to be forward when the clip ended, so two
   recordings of the same person would produce different GEIs.
4. **Encoder** -- turns the GEI into a comparable vector.

On the encoder, and an honest limitation
----------------------------------------
The build plan calls for GaitSet/GaitGL pretrained on CASIA-B. Those weights
*are* downloadable from OpenGait's GitHub releases, but OpenGait ships **no
licence file at all**, and the weights are useless without vendoring its model
definitions. Copying unlicensed research code into this project is a real
legal risk, so it is not the default.

The default `gei` encoder is therefore a **classical descriptor, not a learned
embedding**: pooled GEI pixels plus row/column projection profiles, L2
normalised. GEI-based recognition predates deep gait models and genuinely
works, but it is markedly weaker than a trained encoder and much weaker than
ArcFace. That is stated plainly rather than hidden, because a modality that
silently over-claims its reliability would corrupt the Phase-6 attention
weights that decide how far to trust it.

The preprocessing deliberately produces exactly the 64x44 input a learned gait
model expects, so swapping one in later is an encoder change and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.core.config import GaitSettings, Settings, get_settings
from app.core.logging import get_logger
from app.core.types import (
    Modality,
    ModalityEmbedding,
    TrackObservation,
    l2_normalize,
)
from app.embeddings.base import EmbeddingBranch
from app.embeddings.silhouette import Silhouette, SilhouetteExtractor

logger = get_logger(__name__)


def cadence_signal(silhouettes: Sequence[Silhouette]) -> np.ndarray:
    """Per-frame signal that oscillates once per half gait cycle.

    Uses the width of the lower third of the silhouette -- the legs. Whole-body
    width also oscillates, but arm swing and carried bags contaminate it; the
    legs are where the periodicity actually lives.
    """
    values = []
    for silhouette in silhouettes:
        image = silhouette.image
        legs = image[int(image.shape[0] * 0.66) :, :]
        # Mean occupied width across the leg rows.
        values.append(float(legs.sum(axis=1).mean()))
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class CadenceSamples:
    """A cadence signal placed on a uniform time grid (LOG-06).

    Autocorrelation assumes evenly spaced samples. The silhouettes feeding it
    are not evenly spaced: `video.frame_stride` skips source frames, and the
    extractor drops any frame whose mask failed. Treating the survivors as
    uniform makes the recovered period a function of how many masks happened to
    work, which is not a property of the person's walk.
    """

    #: Signal resampled onto a uniform grid.
    signal: np.ndarray
    #: Grid spacing, in samples per second.
    rate_hz: float
    #: Fraction of grid points that had a real sample near them. Low means the
    #: track is mostly interpolation, and a period recovered from invented
    #: values is invented too.
    coverage: float
    #: How long the track spans, in seconds.
    duration_s: float

    def lag_bounds(self, min_seconds: float, max_seconds: float) -> tuple[int, int]:
        """Autocorrelation lag bounds, in grid samples, for a period in seconds."""
        return (
            max(2, int(round(min_seconds * self.rate_hz))),
            max(2, int(round(max_seconds * self.rate_hz))),
        )

    def resolves(self, min_seconds: float, min_samples: float) -> bool:
        """Whether the shortest period of interest is sampled finely enough.

        Deriving the lag bounds from the real sample rate stops gait silently
        reporting nothing on a strided stream, but it does not conjure signal
        that is not there. At five samples a second a half gait cycle spans
        about two of them, and autocorrelation on two samples does not return
        "unsure" -- it locks onto the full cycle and reports a confident number
        that is twice the truth.

        A wrong cadence is worse than no cadence: it feeds the GEI's
        trustworthiness and the attention head weighs it as real evidence. So
        the branch refuses below this rather than guessing.
        """
        return min_seconds * self.rate_hz >= min_samples


def resample_cadence(
    silhouettes: Sequence[Silhouette],
    values: np.ndarray,
    fallback_fps: float,
) -> CadenceSamples:
    """Put a cadence signal on a uniform time grid.

    Falls back to assuming uniform sampling at `fallback_fps` when the
    silhouettes carry no usable timestamps -- which is the case for anything
    constructed by hand, and reproduces the behaviour this had before
    timestamps existed.
    """
    times = np.asarray(
        [s.timestamp_s for s in silhouettes], dtype=np.float64
    )
    usable = times.size == values.size and times.size >= 2 and np.all(times >= 0)
    if usable:
        usable = bool(np.all(np.diff(times) > 0)) and float(times[-1] - times[0]) > 0

    if not usable:
        rate = fallback_fps if fallback_fps > 0 else 25.0
        return CadenceSamples(
            signal=np.asarray(values, dtype=np.float32),
            rate_hz=rate,
            coverage=1.0,
            duration_s=float(values.size / rate) if values.size else 0.0,
        )

    gaps = np.diff(times)
    step = float(np.median(gaps))
    if step <= 0:
        step = float(times[-1] - times[0]) / max(1, times.size - 1)

    duration = float(times[-1] - times[0])
    count = max(2, int(round(duration / step)) + 1)
    grid = np.linspace(times[0], times[-1], count)
    resampled = np.interp(grid, times, values).astype(np.float32)

    # A grid point is "real" when a sample sits within half a step of it.
    # Anything else is interpolation across a gap, and a period recovered
    # mostly from interpolation is a property of np.interp, not of a walk.
    nearest = np.abs(grid[:, None] - times[None, :]).min(axis=1)
    coverage = float(np.mean(nearest <= step / 2.0))

    return CadenceSamples(
        signal=resampled,
        rate_hz=1.0 / step,
        coverage=coverage,
        duration_s=duration,
    )


def swing_ratio(signal: np.ndarray) -> float:
    """How much the cadence signal actually swings, relative to its level.

    Autocorrelation measures whether a signal *repeats*, not whether it moves.
    Segmentation jitter around a stationary person repeats quite happily at a
    low amplitude, which is enough to fool periodicity alone. A real walk swings
    the leg-region width substantially, so this is the second, independent gate.

    Uses p90-p10 rather than max-min so one bad mask cannot manufacture swing.
    """
    if signal.size < 4:
        return 0.0
    mean = float(np.mean(signal))
    if mean <= 1e-6:
        return 0.0
    return float((np.percentile(signal, 90) - np.percentile(signal, 10)) / mean)


def area_stability(silhouettes: Sequence[Silhouette]) -> float:
    """Coefficient of variation of normalised silhouette area.

    A walking person's silhouette redistributes pixels -- legs open, arms
    swing -- but its total area stays roughly constant, because a body does not
    change size. When the segmenter intermittently gains or loses a chunk of
    the person, area jumps, and the resulting signal can be both periodic and
    high-amplitude, passing both other gates while containing no gait at all.

    Lower is more stable. Real walking measures around 0.03-0.05.
    """
    if not silhouettes:
        return float("inf")
    areas = np.array([float(s.image.sum()) for s in silhouettes], dtype=np.float32)
    mean = float(areas.mean())
    if mean <= 1e-6:
        return float("inf")
    return float(areas.std() / mean)


def estimate_half_period(
    signal: np.ndarray, min_lag: int, max_lag: int
) -> tuple[int | None, float]:
    """Recover the half-cycle period by autocorrelation.

    Returns `(period, strength)`, where strength in [0, 1] is how periodic the
    signal actually is. A low strength means the person was not walking
    steadily -- standing, turning, or the silhouettes were too noisy -- and the
    caller should treat the gait reading as untrustworthy rather than pretend a
    cycle was found.
    """
    if signal.size < 4:
        return None, 0.0

    centred = signal - signal.mean()
    norm = float(np.dot(centred, centred))
    if norm <= 1e-8:
        # Perfectly flat: the legs never moved, so there is no gait here.
        return None, 0.0

    max_lag = min(max_lag, signal.size - 1)
    if max_lag < min_lag:
        return None, 0.0

    scores: dict[int, float] = {}
    for lag in range(min_lag, max_lag + 1):
        overlap = centred[: signal.size - lag]
        shifted = centred[lag:]
        if overlap.size < 2:
            break
        denom = float(np.linalg.norm(overlap) * np.linalg.norm(shifted))
        if denom <= 1e-8:
            continue
        scores[lag] = float(np.dot(overlap, shifted) / denom)

    if not scores:
        return None, 0.0

    best_score = max(scores.values())
    if best_score <= 0.0:
        return None, 0.0

    # A periodic signal correlates just as well at every multiple of its
    # period, and long lags compare fewer samples, which can push their score
    # *above* the fundamental's. Taking the argmax would then report a cadence
    # two or three times too slow. Pick the smallest lag that is within a hair
    # of the best score -- that is the fundamental.
    tolerance = 0.95 * best_score
    fundamental = min(lag for lag, score in scores.items() if score >= tolerance)
    return fundamental, max(0.0, scores[fundamental])


def gait_energy_image(
    silhouettes: Sequence[Silhouette], half_period: int | None
) -> np.ndarray | None:
    """Average silhouettes over a whole number of gait cycles.

    A full cycle is two half-periods (left step, right step). Truncating to a
    whole number of cycles keeps the GEI phase-independent: averaged over a
    partial cycle it would be biased toward whichever leg was forward at the
    cut, and the same person recorded twice would not match themselves.
    """
    if not silhouettes:
        return None

    frames = np.stack([s.image for s in silhouettes])

    if half_period:
        cycle = half_period * 2
        usable = (len(frames) // cycle) * cycle
        if usable >= cycle:
            frames = frames[:usable]

    return frames.mean(axis=0)


class GEIDescriptorEncoder:
    """Classical GEI descriptor. No trained weights.

    Combines a spatially pooled GEI with its row and column projection
    profiles. The pooled image keeps coarse body shape; the profiles capture
    height distribution and stance width, which is where much of the
    inter-person variation in a GEI actually sits.
    """

    def __init__(
        self,
        height: int,
        width: int,
        gei_height: int = 64,
        gei_width: int = 44,
    ) -> None:
        self.height = height
        self.width = width
        # Profiles are taken from the FULL-resolution GEI, not the pooled copy:
        # they exist to capture fine height distribution and stance width, and
        # pooling first would throw away exactly the detail they are for. The
        # encoder therefore has to know the GEI's shape to report `dim`.
        self.gei_height = gei_height
        self.gei_width = gei_width

    @property
    def dim(self) -> int:
        return self.height * self.width + self.gei_height + self.gei_width

    def encode(self, gei: np.ndarray) -> np.ndarray:
        pooled = cv2.resize(
            gei, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        row_profile = gei.mean(axis=1)
        column_profile = gei.mean(axis=0)

        vector = np.concatenate(
            [pooled.ravel(), row_profile.ravel(), column_profile.ravel()]
        ).astype(np.float32)
        return l2_normalize(vector)


class GaitEmbedder(EmbeddingBranch):
    """Gait embeddings from a sequence of observations of one track."""

    modality = Modality.GAIT

    def __init__(
        self,
        settings: Settings | None = None,
        gait: GaitSettings | None = None,
        extractor: SilhouetteExtractor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = gait or self.settings.gait
        self.min_observations = self.cfg.min_frames

        if self.cfg.encoder == "gei":
            self.encoder = GEIDescriptorEncoder(
                self.cfg.descriptor_height,
                self.cfg.descriptor_width,
                self.cfg.silhouette_height,
                self.cfg.silhouette_width,
            )
        else:
            raise ValueError(
                f"Unknown gait encoder {self.cfg.encoder!r}. Only 'gei' is "
                "implemented. A learned encoder would need OpenGait's model "
                "code, which ships without a licence -- see the module docstring."
            )

        self.embedding_dim = self.encoder.dim
        self._extractor = extractor

    @property
    def model_id(self) -> str:
        # The GEI descriptor is not learned weights, but its geometry is what
        # defines the space: change the descriptor grid and old vectors are a
        # different length and a different meaning.
        return (
            f"gei/{self.cfg.descriptor_height}x{self.cfg.descriptor_width}"
            f"@{self.cfg.silhouette_height}x{self.cfg.silhouette_width}"
        )

    @property
    def extractor(self) -> SilhouetteExtractor:
        """Segmentation model, loaded on first use rather than at construction.

        Keeps `GaitEmbedder()` cheap to build in tests that never segment
        anything.
        """
        if self._extractor is None:
            self._extractor = SilhouetteExtractor(self.settings, self.cfg)
        return self._extractor

    # -- quality -----------------------------------------------------------

    def _quality(
        self, silhouettes: Sequence[Silhouette], periodicity: float, cycles: float
    ) -> float:
        """How much to trust this gait reading, in [0, 1].

        Three multiplied factors, mirroring the face branch's logic:

        * **Periodicity** -- did the person actually walk steadily, or stand
          and turn? An unsteady signal means the GEI is averaging unrelated
          poses together.
        * **Cycle count** -- one cycle is the bare minimum; two or more is a
          much more stable average.
        * **Silhouette cleanliness** -- the fraction of frames where the body
          was fully in view. A body running off the top or bottom of the frame
          normalises to the wrong height; one running off the side has a
          truncated width, which is the very quantity the cadence signal is
          read from. Either way the shape is misleading rather than merely
          noisy, so these are downweighted rather than averaged in as equals.
        """
        if not silhouettes:
            return 0.0

        unclipped = sum(1 for s in silhouettes if not s.clipped) / len(silhouettes)
        cycle_factor = min(1.0, cycles / 2.0)
        return float(np.clip(periodicity * cycle_factor * unclipped, 0.0, 1.0))

    # -- embedding ---------------------------------------------------------

    def embed(self, observations: Sequence[TrackObservation]) -> ModalityEmbedding:
        if len(observations) < self.cfg.min_frames:
            return ModalityEmbedding.empty(
                self.modality, reason="fewer frames than one gait cycle"
            )

        silhouettes = self.extractor.extract(observations)
        if len(silhouettes) < self.cfg.min_frames:
            return ModalityEmbedding.empty(
                self.modality, reason="too few usable silhouettes"
            )

        # Cadence is a rate, so it is recovered in real time rather than in
        # frames: the stride skips source frames and the extractor drops any
        # whose mask failed, so the survivors are not evenly spaced (LOG-06).
        cadence = resample_cadence(
            silhouettes, cadence_signal(silhouettes), self.cfg.assumed_fps
        )
        if cadence.coverage < self.cfg.min_cadence_coverage:
            return ModalityEmbedding.empty(
                self.modality,
                reason="too many gaps in the silhouette sequence to read cadence",
            )

        if not cadence.resolves(
            self.cfg.min_half_period_s, self.cfg.min_samples_per_half_period
        ):
            # Sampled too coarsely to tell a walk from its own harmonics.
            logger.warning(
                "Gait refused: %.1f samples/second cannot resolve a %.2fs half "
                "cycle. Lower video.frame_stride, or accept that this footage "
                "carries no gait.",
                cadence.rate_hz,
                self.cfg.min_half_period_s,
            )
            return ModalityEmbedding.empty(
                self.modality,
                reason="frames too far apart to measure cadence",
            )

        signal = cadence.signal
        min_lag, max_lag = cadence.lag_bounds(
            self.cfg.min_half_period_s, self.cfg.max_half_period_s
        )
        half_period, periodicity = estimate_half_period(signal, min_lag, max_lag)
        swing = swing_ratio(signal)

        # In seconds, so this measures the walk rather than the sample count.
        half_period_s = half_period / cadence.rate_hz if half_period else 0.0
        cycles = (
            cadence.duration_s / (half_period_s * 2) if half_period_s > 0 else 0.0
        )
        if half_period is None or cycles < 1.0:
            # Not one complete gait cycle: a GEI here would encode a pose, not
            # a walk, and would not match the same person on another day.
            return ModalityEmbedding.empty(
                self.modality, reason="no complete gait cycle detected"
            )

        # Two independent gates, because either alone is foolable. A person
        # standing still while the camera pans produces mask jitter that
        # repeats (passing periodicity) at a tiny amplitude (failing swing);
        # a single lurching movement produces amplitude without repetition.
        # Only something that both repeats AND swings is a walk.
        if periodicity < self.cfg.min_periodicity:
            return ModalityEmbedding.empty(
                self.modality, reason="signal not periodic enough to be a walk"
            )
        if swing < self.cfg.min_swing_ratio:
            return ModalityEmbedding.empty(
                self.modality, reason="legs barely move; person is not walking"
            )

        # Third gate, on a physical invariant rather than a tuned statistic:
        # a body does not change size while walking. Unstable segmentation can
        # produce a signal that both repeats and swings hard, satisfying the
        # two gates above while containing no gait whatsoever.
        area_cv = area_stability(silhouettes)
        if area_cv > self.cfg.max_area_cv:
            return ModalityEmbedding.empty(
                self.modality,
                reason="silhouette area unstable; segmentation is failing",
            )

        gei = gait_energy_image(silhouettes, half_period)
        if gei is None:
            return ModalityEmbedding.empty(self.modality, reason="GEI failed")

        quality = self._quality(silhouettes, periodicity, cycles)
        if quality < self.cfg.min_quality:
            return ModalityEmbedding.empty(
                self.modality, reason="gait quality below threshold"
            )

        return ModalityEmbedding(
            modality=self.modality,
            model_id=self.model_id,
            vector=self.encoder.encode(gei),
            quality=quality,
            frames_used=len(silhouettes),
            detail={
                "half_period": float(half_period),
                # Also in seconds, because the sample count means nothing
                # without knowing the rate it was sampled at.
                "half_period_s": float(half_period_s),
                "cadence_hz": float(cadence.rate_hz),
                "cadence_coverage": float(cadence.coverage),
                "cycles": float(cycles),
                "periodicity": float(periodicity),
                "swing_ratio": float(swing),
                "area_cv": float(area_cv),
                "silhouettes": float(len(silhouettes)),
                "clipped_fraction": float(
                    sum(1 for s in silhouettes if s.clipped) / len(silhouettes)
                ),
                "mean_coverage": float(np.mean([s.coverage for s in silhouettes])),
            },
        )
