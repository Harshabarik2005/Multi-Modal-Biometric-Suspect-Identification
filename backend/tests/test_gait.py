"""Tests for the gait branch.

The signal-processing core -- cadence detection, GEI construction, the
descriptor encoder, silhouette normalisation -- is tested with synthetic
silhouettes and needs no model weights, so it runs on every `pytest`. Tests
that need YOLOv8-seg are marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.embeddings.gait import (
    GEIDescriptorEncoder,
    GaitEmbedder,
    area_stability,
    cadence_signal,
    estimate_half_period,
    gait_energy_image,
    swing_ratio,
)
from app.embeddings.silhouette import Silhouette, SilhouetteExtractor


def walking_silhouette(
    phase: float, height: int = 64, width: int = 44, stride: float = 1.0
) -> np.ndarray:
    """A crude walking figure: fixed torso, legs scissoring with `phase`."""
    image = np.zeros((height, width), dtype=np.float32)
    centre = width // 2

    # Torso: a stable column in the upper two thirds.
    torso_half = max(2, int(width * 0.10))
    image[: int(height * 0.66), centre - torso_half : centre + torso_half] = 1.0

    # Legs: two lines opening and closing with the gait phase.
    spread = int(round(stride * (width * 0.22) * abs(np.sin(phase))))
    for row in range(int(height * 0.66), height):
        depth = (row - int(height * 0.66)) / max(1, height - int(height * 0.66))
        offset = int(spread * depth)
        for x in (centre - offset, centre + offset):
            image[row, max(0, x - 1) : min(width, x + 2)] = 1.0
    return image


def walking_sequence(
    frames: int, half_period: int = 10, stride: float = 1.0, clipped: bool = False
) -> list[Silhouette]:
    return [
        Silhouette(
            frame_index=i,
            image=walking_silhouette(np.pi * i / half_period, stride=stride),
            coverage=0.3,
            clipped=clipped,
        )
        for i in range(frames)
    ]


class TestCadenceDetection:
    def test_recovers_a_known_period(self) -> None:
        silhouettes = walking_sequence(60, half_period=10)
        period, strength = estimate_half_period(cadence_signal(silhouettes), 5, 40)
        assert period == pytest.approx(10, abs=1)
        assert strength > 0.7

    def test_recovers_a_different_period(self) -> None:
        silhouettes = walking_sequence(60, half_period=7)
        period, _ = estimate_half_period(cadence_signal(silhouettes), 5, 40)
        assert period == pytest.approx(7, abs=1)

    def test_a_standing_person_has_no_cadence(self) -> None:
        """Standing still must not be reported as a gait cycle."""
        still = [
            Silhouette(i, walking_silhouette(0.0), 0.3, False) for i in range(40)
        ]
        period, strength = estimate_half_period(cadence_signal(still), 5, 40)
        assert period is None or strength < 0.2

    def test_too_short_a_signal_returns_nothing(self) -> None:
        assert estimate_half_period(np.array([1.0, 2.0]), 5, 40) == (None, 0.0)

    def test_flat_signal_does_not_divide_by_zero(self) -> None:
        period, strength = estimate_half_period(np.ones(40, dtype=np.float32), 5, 40)
        assert period is None
        assert strength == 0.0


class TestGaitEnergyImage:
    def test_averages_to_the_silhouette_shape(self) -> None:
        gei = gait_energy_image(walking_sequence(40, half_period=10), 10)
        assert gei.shape == (64, 44)
        assert 0.0 <= gei.min() and gei.max() <= 1.0

    def test_torso_is_solid_and_feet_are_blurred(self) -> None:
        """The GEI's whole point: static parts stay sharp, moving parts blur.

        Measured at the ANKLES, not the hips. Legs join at the hip, so the top
        of the leg region is occupied in every frame and saturates to 1.0 --
        that is anatomy, not a bug. The scissoring only opens up further down.
        """
        gei = gait_energy_image(walking_sequence(40, half_period=10), 10)
        torso = gei[: int(64 * 0.6)]
        ankles = gei[int(64 * 0.92) :]
        assert torso.max() == pytest.approx(1.0, abs=1e-6)
        assert ankles.max() < 0.99, "feet should be smeared across the average"
        assert ankles.mean() < torso.mean()

    def test_truncates_to_whole_cycles(self) -> None:
        """A partial cycle biases the GEI toward whichever leg was forward.

        Same walk, different clip lengths, must give near-identical GEIs -- so
        the same person recorded twice matches themselves.
        """
        full = gait_energy_image(walking_sequence(40, half_period=10), 10)
        ragged = gait_energy_image(walking_sequence(47, half_period=10), 10)
        assert np.abs(full - ragged).max() < 1e-6

    def test_empty_sequence_returns_none(self) -> None:
        assert gait_energy_image([], 10) is None


class TestDescriptorEncoder:
    def test_dimension_matches_declared_dim(self) -> None:
        """`dim` must match what `encode` actually returns.

        The profiles come from the full-resolution 64x44 GEI, not the pooled
        32x22 copy, so the total is 32*22 + 64 + 44. A gallery that trusted a
        wrong `dim` would reject valid vectors as malformed.
        """
        encoder = GEIDescriptorEncoder(32, 22, gei_height=64, gei_width=44)
        gei = gait_energy_image(walking_sequence(40), 10)
        assert encoder.dim == 32 * 22 + 64 + 44
        assert encoder.encode(gei).size == encoder.dim

    def test_declared_dim_tracks_a_different_gei_shape(self) -> None:
        encoder = GEIDescriptorEncoder(16, 11, gei_height=128, gei_width=88)
        gei = np.zeros((128, 88), dtype=np.float32)
        gei[10:100, 20:60] = 0.5
        assert encoder.encode(gei).size == encoder.dim == 16 * 11 + 128 + 88

    def test_output_is_unit_length(self) -> None:
        encoder = GEIDescriptorEncoder(32, 22)
        vector = encoder.encode(gait_energy_image(walking_sequence(40), 10))
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_same_walk_scores_higher_than_a_different_stride(self) -> None:
        """The descriptor has to separate walking styles at all, or it is noise."""
        from app.core.types import cosine_similarity

        encoder = GEIDescriptorEncoder(32, 22)
        narrow_a = encoder.encode(
            gait_energy_image(walking_sequence(40, 10, stride=0.4), 10)
        )
        narrow_b = encoder.encode(
            gait_energy_image(walking_sequence(60, 10, stride=0.4), 10)
        )
        wide = encoder.encode(
            gait_energy_image(walking_sequence(40, 10, stride=1.8), 10)
        )

        same = cosine_similarity(narrow_a, narrow_b)
        different = cosine_similarity(narrow_a, wide)
        assert same > different, f"same walk {same:.4f} !> different {different:.4f}"
        assert same > 0.99


class TestSilhouetteNormalisation:
    def _extractor(self) -> SilhouetteExtractor:
        # Bypass __init__ so no segmentation weights are loaded; only the pure
        # geometry of `normalise` is under test here.
        extractor = SilhouetteExtractor.__new__(SilhouetteExtractor)
        extractor.cfg = get_settings().gait
        return extractor

    def test_scales_any_input_size_to_the_standard_canvas(self) -> None:
        extractor = self._extractor()
        for size in ((200, 90), (37, 20), (640, 300)):
            mask = np.zeros(size, dtype=bool)
            mask[10 : size[0] - 10, 5 : size[1] - 5] = True
            out = extractor.normalise(mask)
            assert out.shape == (64, 44)

    def test_centres_on_mass_so_position_in_frame_does_not_matter(self) -> None:
        """Two identical people at different x positions must normalise alike."""
        extractor = self._extractor()
        left = np.zeros((120, 200), dtype=bool)
        left[10:110, 20:50] = True
        right = np.zeros((120, 200), dtype=bool)
        right[10:110, 150:180] = True

        assert np.abs(extractor.normalise(left) - extractor.normalise(right)).max() < 1e-6

    def test_empty_mask_returns_none(self) -> None:
        assert self._extractor().normalise(np.zeros((100, 50), dtype=bool)) is None

    def test_single_pixel_mask_returns_none(self) -> None:
        mask = np.zeros((100, 50), dtype=bool)
        mask[50, 25] = True
        assert self._extractor().normalise(mask) is None


class TestGaitEmbedder:
    def _embedder(self) -> GaitEmbedder:
        settings = get_settings()
        # Pass a sentinel extractor so nothing tries to load seg weights; the
        # tests below all bail out before segmentation is reached.
        return GaitEmbedder(settings, extractor=object())

    def test_reports_no_signal_below_min_frames(self) -> None:
        from app.core.types import TrackObservation

        embedder = self._embedder()
        observations = [
            TrackObservation(i, i / 25.0, np.zeros((100, 50, 3), np.uint8), 100.0, 0.9)
            for i in range(5)
        ]
        result = embedder.embed(observations)
        assert not result.has_signal

    def test_rejects_an_unknown_encoder(self) -> None:
        settings = get_settings()
        saved = settings.gait.encoder
        settings.gait.encoder = "opengait"
        try:
            with pytest.raises(ValueError, match="licence"):
                GaitEmbedder(settings)
        finally:
            settings.gait.encoder = saved

    def test_quality_falls_when_the_body_is_clipped(self) -> None:
        embedder = self._embedder()
        clean = walking_sequence(40, clipped=False)
        clipped = walking_sequence(40, clipped=True)
        assert embedder._quality(clean, 0.9, 2.0) > embedder._quality(clipped, 0.9, 2.0)

    def test_quality_falls_when_the_walk_is_unsteady(self) -> None:
        embedder = self._embedder()
        silhouettes = walking_sequence(40)
        assert embedder._quality(silhouettes, 0.95, 2.0) > embedder._quality(
            silhouettes, 0.2, 2.0
        )

    def test_quality_rewards_more_complete_cycles(self) -> None:
        embedder = self._embedder()
        silhouettes = walking_sequence(40)
        assert embedder._quality(silhouettes, 0.9, 2.0) > embedder._quality(
            silhouettes, 0.9, 1.0
        )

    def test_quality_of_nothing_is_zero(self) -> None:
        assert self._embedder()._quality([], 0.9, 2.0) == 0.0


class TestWalkGates:
    """The three independent gates that stop non-walking being read as gait.

    Each exists because the others are individually foolable. Discovered by
    running the branch on a panned photograph of stationary people: 3 of 4
    tracks produced confident gait embeddings, one at quality 0.66, purely
    from segmentation jitter.
    """

    def test_periodicity_alone_is_foolable(self) -> None:
        """A low-amplitude repeating wobble passes autocorrelation."""
        jitter = np.tile([10.0, 10.2, 10.0, 9.8], 15).astype(np.float32)
        _, strength = estimate_half_period(jitter, 5, 40)
        assert strength > 0.5, "this wobble does repeat"
        # ...but it barely moves, which is what the swing gate is for.
        assert swing_ratio(jitter) < 0.12

    def test_swing_alone_is_foolable(self) -> None:
        """One big lurch has amplitude without repetition."""
        lurch = np.concatenate(
            [np.full(30, 5.0), np.full(30, 12.0)]
        ).astype(np.float32)
        assert swing_ratio(lurch) > 0.12, "this does swing"
        _, strength = estimate_half_period(lurch, 5, 40)
        assert strength < 0.99, "but it is not periodic within the lag range"

    def test_area_stability_separates_walking_from_broken_segmentation(self) -> None:
        """A body does not change size while walking.

        Real walking measured 0.033-0.045 across strides and cadences; a track
        whose segmentation kept gaining and losing chunks measured 0.159.
        """
        for stride in (0.4, 1.0, 1.8):
            for half_period in (7, 10, 14):
                walk = walking_sequence(60, half_period=half_period, stride=stride)
                assert area_stability(walk) < 0.10

        # Silhouettes whose area swings wildly: segmentation failing, not gait.
        broken = []
        for i in range(40):
            image = walking_silhouette(np.pi * i / 10)
            if i % 3 == 0:
                image = image * 0.0  # the segmenter lost the person entirely
            broken.append(Silhouette(i, image, 0.3, False))
        assert area_stability(broken) > 0.10

    def test_area_stability_of_nothing_is_infinite(self) -> None:
        assert area_stability([]) == float("inf")

    def test_swing_ratio_of_a_flat_signal_is_zero(self) -> None:
        assert swing_ratio(np.full(40, 7.0, dtype=np.float32)) == 0.0

    def test_swing_ratio_of_a_short_signal_is_zero(self) -> None:
        assert swing_ratio(np.array([1.0, 2.0])) == 0.0

    def test_all_three_gates_pass_for_a_real_walk(self) -> None:
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        walk = walking_sequence(60, half_period=10, stride=1.0)
        signal = cadence_signal(walk)

        # The hand-built sequence carries no timestamps, so this falls back to
        # assuming uniform sampling -- which is what it is.
        cadence = resample_cadence(walk, signal, settings.gait.assumed_fps)
        min_lag, max_lag = cadence.lag_bounds(
            settings.gait.min_half_period_s, settings.gait.max_half_period_s
        )
        _, periodicity = estimate_half_period(cadence.signal, min_lag, max_lag)
        assert periodicity >= settings.gait.min_periodicity
        assert swing_ratio(signal) >= settings.gait.min_swing_ratio
        assert area_stability(walk) <= settings.gait.max_area_cv


@pytest.mark.slow
class TestGaitOnRealFootage:
    def test_stationary_people_produce_no_gait_embedding(self) -> None:
        """Nobody in the pan clip is walking, so nothing may report gait.

        This is the safety-critical direction: a false gait signal would be
        weighted as real evidence by the Phase-6 attention head. Before the
        three gates existed, 3 of 4 tracks here produced gait embeddings.
        """
        from pathlib import Path as _Path

        from app.core.track_buffer import TrackBufferStore
        from app.pipeline import DetectionTrackingPipeline

        clip = _Path(__file__).resolve().parents[2] / "data" / "test_videos" / "synthetic_pan.mp4"
        if not clip.exists():
            pytest.skip("Run `python scripts/make_test_video.py` first.")

        settings = get_settings()
        previous = settings.video.max_frames
        settings.video.max_frames = 60
        try:
            pipeline = DetectionTrackingPipeline(settings)
            store = TrackBufferStore(settings)
            for result, frame in pipeline.stream(clip):
                store.update(result, frame)

            embedder = GaitEmbedder(settings)
            embedded = [
                buffer.track_id
                for buffer in store
                if embedder.embed(list(buffer)).has_signal
            ]
            assert not embedded, (
                f"tracks {embedded} reported gait, but nobody in this clip walks"
            )
        finally:
            settings.video.max_frames = previous


def timed_walk(
    frames: int,
    half_period_s: float = 0.4,
    fps: float = 25.0,
    frame_stride: int = 1,
    drop: set[int] | None = None,
) -> list[Silhouette]:
    """A walk sampled the way the pipeline actually samples one.

    `frame_stride` skips source frames, `drop` removes frames whose mask
    failed. Both leave the surviving silhouettes unevenly spaced in the source
    frame numbering, and `timestamp_s` is what says how far apart they really
    are.
    """
    drop = drop or set()
    out = []
    for step in range(frames):
        source_frame = step * frame_stride
        if source_frame in drop:
            continue
        seconds = source_frame / fps
        out.append(
            Silhouette(
                frame_index=source_frame,
                image=walking_silhouette(np.pi * seconds / half_period_s),
                coverage=0.3,
                clipped=False,
                timestamp_s=seconds,
            )
        )
    return out


class TestCadenceIsMeasuredInSeconds:
    """LOG-06: cadence assumed a gapless 25fps stride-1 stream.

    `min_half_period=7` / `max_half_period=40` were counts of *processed*
    frames. Set `video.frame_stride: 5` and a real half-cycle of ~12 source
    frames arrives as ~2.4 samples, below the floor -- so gait reported no
    signal for everyone, and it read as poor footage rather than as a config
    value nobody connected to the symptom.
    """

    def test_a_strided_stream_still_finds_the_cycle(self) -> None:
        """The case that silently disabled gait for everybody."""
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        walk = timed_walk(120, half_period_s=0.48, fps=25.0, frame_stride=2)

        cadence = resample_cadence(
            walk, cadence_signal(walk), settings.gait.assumed_fps
        )
        # Two source frames apart at 25fps is one sample per 0.08s.
        assert cadence.rate_hz == pytest.approx(12.5, rel=0.05)

        min_lag, max_lag = cadence.lag_bounds(
            settings.gait.min_half_period_s, settings.gait.max_half_period_s
        )
        assert min_lag >= 2
        period, strength = estimate_half_period(cadence.signal, min_lag, max_lag)

        assert period is not None, "the cycle was lost to frame-count lag bounds"
        assert period / cadence.rate_hz == pytest.approx(0.48, abs=0.1)
        assert strength > 0.7

    def test_the_old_frame_count_bounds_would_have_missed_it(self) -> None:
        """The bug, asserted, so the fix cannot be quietly undone."""
        walk = timed_walk(120, half_period_s=0.48, fps=25.0, frame_stride=2)
        # 7 and 40 were the old hardcoded bounds, in processed frames. The real
        # half cycle arrives as 6 samples, under the floor of 7.
        period, _ = estimate_half_period(cadence_signal(walk), 7, 40)
        assert period is None or period > 6 * 1.5, (
            "the old bounds should not have been able to find a 6-sample cycle"
        )

    def test_too_coarse_a_stream_is_refused_not_guessed_at(self) -> None:
        """Deriving the bounds correctly does not conjure absent signal.

        At five samples a second a half cycle spans about two of them, and
        autocorrelation does not report uncertainty -- it locks onto the full
        cycle and returns a confident number twice the truth. That is worse
        than nothing, because it gets weighted as evidence.
        """
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        walk = timed_walk(40, half_period_s=0.48, fps=25.0, frame_stride=5)
        cadence = resample_cadence(
            walk, cadence_signal(walk), settings.gait.assumed_fps
        )
        assert cadence.rate_hz == pytest.approx(5.0, rel=0.05)

        assert not cadence.resolves(
            settings.gait.min_half_period_s,
            settings.gait.min_samples_per_half_period,
        )

        # And it would indeed have been wrong: the recovered period is the
        # full cycle, not the half cycle.
        period, _ = estimate_half_period(
            cadence.signal,
            *cadence.lag_bounds(
                settings.gait.min_half_period_s, settings.gait.max_half_period_s
            ),
        )
        assert period is not None
        assert period / cadence.rate_hz > 0.48 * 1.5

    def test_a_normal_stream_resolves(self) -> None:
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        for stride in (1, 2):
            walk = timed_walk(
                200 // stride, half_period_s=0.48, fps=25.0, frame_stride=stride
            )
            cadence = resample_cadence(
                walk, cadence_signal(walk), settings.gait.assumed_fps
            )
            assert cadence.resolves(
                settings.gait.min_half_period_s,
                settings.gait.min_samples_per_half_period,
            ), f"stride {stride} should still support gait"

    def test_the_recovered_period_does_not_depend_on_the_stride(self) -> None:
        """Cadence is a property of the walk, not of how it was sampled."""
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        seconds = []
        for stride in (1, 2):
            walk = timed_walk(
                200 // stride, half_period_s=0.48, fps=25.0, frame_stride=stride
            )
            cadence = resample_cadence(
                walk, cadence_signal(walk), settings.gait.assumed_fps
            )
            period, _ = estimate_half_period(
                cadence.signal,
                *cadence.lag_bounds(
                    settings.gait.min_half_period_s,
                    settings.gait.max_half_period_s,
                ),
            )
            assert period is not None, f"lost the cycle at stride {stride}"
            seconds.append(period / cadence.rate_hz)

        assert max(seconds) - min(seconds) < 0.15, (
            f"the recovered period moved with the stride: {seconds}"
        )

    def test_dropped_masks_do_not_stretch_the_period(self) -> None:
        """`extract` drops frames whose mask failed; the survivors are not
        uniformly sampled, and treating them as if they were shifts the
        recovered period."""
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        # Lose a scattered fifth of the frames, as a flaky segmenter would.
        dropped = {i for i in range(200) if i % 5 == 3}
        walk = timed_walk(200, half_period_s=0.48, fps=25.0, drop=dropped)

        cadence = resample_cadence(
            walk, cadence_signal(walk), settings.gait.assumed_fps
        )
        period, _ = estimate_half_period(
            cadence.signal,
            *cadence.lag_bounds(
                settings.gait.min_half_period_s, settings.gait.max_half_period_s
            ),
        )
        assert period is not None
        assert period / cadence.rate_hz == pytest.approx(0.48, abs=0.1)

    def test_a_mostly_missing_sequence_is_reported_not_interpolated(self) -> None:
        """A period recovered from interpolation describes np.interp."""
        from app.embeddings.gait import resample_cadence

        settings = get_settings()
        # One long contiguous outage in the middle.
        walk = timed_walk(
            120, half_period_s=0.48, fps=25.0, drop=set(range(20, 100))
        )
        cadence = resample_cadence(
            walk, cadence_signal(walk), settings.gait.assumed_fps
        )
        assert cadence.coverage < settings.gait.min_cadence_coverage

    def test_untimed_silhouettes_fall_back_to_uniform(self) -> None:
        """Hand-built sequences carry no timestamps and must still work."""
        from app.embeddings.gait import resample_cadence

        walk = walking_sequence(60, half_period=10)
        cadence = resample_cadence(walk, cadence_signal(walk), 25.0)
        assert cadence.rate_hz == 25.0
        assert cadence.coverage == 1.0
        assert cadence.signal.size == 60
