"""Configuration loading for Faceless FRS.

Layered config, highest priority first:

    CLI flags  >  environment variables  >  config.yaml  >  field defaults

CLI flags are applied by the caller (see `scripts/run_pipeline.py`); the rest
is handled here. Nested settings use the `FRS_` prefix with `__` as the
separator, e.g. `FRS_DETECTION__CONF_THRESHOLD=0.5`.

Import `get_settings()` rather than constructing `Settings` directly, so the
YAML file is read once per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# repo_root/backend/app/core/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_CONFIG_PATH = BACKEND_ROOT / "config.yaml"

# Which YAML file the next Settings() construction should read. Set by
# get_settings(); the sources hook below has no other way to receive it.
_active_config_path: Path = DEFAULT_CONFIG_PATH


class PathSettings(BaseModel):
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    enrollment_dir: Path = Path("data/enrollment")
    test_videos_dir: Path = Path("data/test_videos")
    output_dir: Path = Path("data/output")

    def resolved(self, root: Path = REPO_ROOT) -> "PathSettings":
        """Return a copy with every relative path anchored at the repo root."""
        return PathSettings(
            **{
                name: (value if value.is_absolute() else root / value)
                for name, value in self
            }
        )


class DetectionSettings(BaseModel):
    model: str = "yolov8n.pt"
    conf_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    person_class_id: int = 0
    imgsz: int = 640
    min_box_height: int = 60
    half: bool = True


class TrackingSettings(BaseModel):
    max_age: int = 30
    n_init: int = 3
    max_cosine_distance: float = 0.35
    nn_budget: int | None = 100
    embedder: str = "mobilenet"
    embedder_gpu: bool = True


class FaceSettings(BaseModel):
    """Phase 2: ArcFace / InsightFace face branch."""

    # InsightFace model pack. buffalo_l is the accurate default (~280MB,
    # downloads to ~/.insightface on first use); buffalo_s is smaller/faster.
    model_pack: str = "buffalo_l"
    det_size: int = 640
    # Requires onnxruntime-gpu built for the installed CUDA; the plain
    # onnxruntime wheel is CPU-only and this silently falls back.
    use_gpu: bool = False

    # Frames needed before the branch will report a face embedding at all.
    min_frames: int = 1
    # Per-frame quality floor. Below this the frame is treated as no signal.
    min_quality: float = Field(0.15, ge=0.0, le=1.0)

    # Yaw (degrees) up to which a face counts as fully frontal, and the yaw at
    # which it is considered useless. ArcFace degrades sharply in between.
    frontal_yaw_deg: float = 20.0
    max_yaw_deg: float = 65.0
    # Face height in pixels at which resolution stops being a limiting factor.
    ideal_face_height: float = 112.0

    # Enrollment keeps only this top fraction of usable frames by quality, so
    # profile and back-of-head frames do not blur the reference vector.
    enroll_top_fraction: float = Field(0.4, gt=0.0, le=1.0)
    enroll_min_frames: int = 10
    #: Penalise quality by how much of the face is actually visible (Phase 11).
    #:
    #: On by default, because without it a covered face scores HIGHER than a
    #: clear one -- an opaque shape makes the detector more confident while yaw
    #: and pixel count do not move -- and quality is what fusion weights by.
    #: Turn it off to reproduce pre-Phase-11 numbers, or where a deployment
    #: sees no occlusion and wants the milliseconds back.
    occlusion_aware: bool = True


class GaitSettings(BaseModel):
    """Phase 3: gait branch -- silhouettes, gait cycles, GEI."""

    # YOLOv8 segmentation model used to cut silhouettes out of body crops.
    seg_model: str = "yolov8n-seg.pt"
    seg_conf: float = Field(0.30, ge=0.0, le=1.0)

    # Standard gait silhouette canvas (GaitSet / GaitGL / GaitBase all use
    # 64x44). Keeping this shape means a learned encoder can be dropped in
    # later without redoing the preprocessing.
    silhouette_height: int = 64
    silhouette_width: int = 44

    # A gait cycle is roughly 20-30 frames at 25fps. Below `min_frames` there
    # is not enough signal and the branch reports nothing rather than guessing.
    min_frames: int = 20
    # Plausible half-cycle period, IN SECONDS (LOG-06).
    #
    # These were frame counts, which quietly assumed the stream was 25fps with
    # no frames skipped. Set video.frame_stride to 5 and a real half-cycle of
    # ~12 source frames becomes ~2.4 samples -- under the old floor of 7 -- so
    # gait reported no signal for everybody, and it read as poor footage rather
    # than as configuration. The lag bounds are now derived from the measured
    # sample rate, so they follow the stride and the source frame rate.
    #
    # 0.28s to 1.6s covers a run through to a slow walk. At 25fps with stride 1
    # they work out to 7 and 40 samples, which is what the old frame counts
    # were, so the behaviour on the footage this was tuned against is unchanged.
    min_half_period_s: float = 0.28
    max_half_period_s: float = 1.6

    # Sampling rate assumed when silhouettes carry no timestamps -- hand-built
    # sequences, mostly. Real footage always carries them.
    assumed_fps: float = 25.0

    # Below this fraction of the cadence grid backed by a real silhouette, the
    # signal is mostly interpolation across gaps and any period recovered from
    # it describes np.interp rather than a walk.
    min_cadence_coverage: float = Field(0.6, ge=0.0, le=1.0)

    # Samples the shortest half cycle of interest must span before cadence is
    # believed. Below about three, autocorrelation does not fail -- it locks
    # onto the full cycle and confidently reports twice the real period, which
    # then gets weighted as evidence. At 25fps this allows frame_stride up to
    # 2; beyond that gait refuses rather than guessing.
    min_samples_per_half_period: float = 3.0

    # Autocorrelation strength below which the signal is not a walk. Panning
    # over a stationary person produces weak periodicity from mask jitter --
    # measured around 0.18-0.32 on such footage, so this sits well above it.
    min_periodicity: float = Field(0.55, ge=0.0, le=1.0)
    # The leg-region width of a real walk swings substantially. Noise wobbles.
    # Relative swing = (p90 - p10) / mean of the cadence signal.
    min_swing_ratio: float = Field(0.12, ge=0.0)
    # A human body does not change size while walking. Large frame-to-frame
    # variation in silhouette area means the segmenter is gaining and losing
    # chunks of the person, and its "periodicity" is segmentation noise rather
    # than gait. Measured: real walking 0.033-0.045, unstable segmentation
    # 0.159, so this sits with headroom on both sides.
    max_area_cv: float = Field(0.10, ge=0.0)

    # Encoder producing the final vector from a GEI.
    #   "gei"  - classical descriptor, no trained weights required
    #   "opengait" - learned encoder; needs OpenGait model code, see docs
    encoder: str = "gei"
    # Side length the GEI is pooled to before flattening, for the classical
    # descriptor. 32x22 keeps spatial structure without a huge vector.
    descriptor_height: int = 32
    descriptor_width: int = 22

    # Below this the silhouette is too poor to use (mask barely covers the box,
    # or the person is clipped by the frame edge).
    min_silhouette_coverage: float = Field(0.12, ge=0.0, le=1.0)
    min_quality: float = Field(0.25, ge=0.0, le=1.0)


class ReIDSettings(BaseModel):
    """Phase 4: OSNet whole-body appearance re-identification."""

    # OSNet variant. x1_0 is the accurate default (2.7M params, ~11MB weights);
    # x0_25 is markedly faster if throughput matters.
    model: str = "osnet_x1_0"

    # Which checkpoint to load (DES-01). This branch used to load OSNet's
    # ImageNet weights -- generic classification features, not a model trained
    # to tell people apart -- while presenting the result as re-identification.
    #
    #   msmt17        re-ID. 4,101 identities, 15 cameras, indoor and outdoor,
    #                 day and night. The most varied single re-ID dataset, so
    #                 the best default for footage from cameras it has never
    #                 seen. osnet_x1_0 only.
    #   market1501    re-ID, but a single university campus. Higher benchmark
    #                 numbers, narrower conditions.
    #   dukemtmcreid  re-ID, outdoor campus.
    #   multi_source  trained on MSMT17 + DukeMTMC + CUHK03 together, for
    #                 transfer to unseen cameras. osnet_ibn_x1_0 only.
    #   imagenet      NOT re-ID. Available for every variant, including the
    #                 small fast ones, and it will warn loudly on every start.
    weights: str = "msmt17"

    # OSNet's training input size. Do not change without retraining.
    input_height: int = 256
    input_width: int = 128
    #: Crops per forward pass. The batch used to be "however many there are",
    #: which is fine at 30fps for a few seconds and fatal otherwise: enrolment
    #: keeps up to 600 observations PER video, so three 60fps clips handed the
    #: model ~1800 crops at once and asked a 4GB card for 3.41GiB in a single
    #: allocation. Chunking costs nothing measurable -- the GPU is saturated
    #: long before 64 crops -- and makes memory a function of this number
    #: rather than of how long somebody filmed for.
    max_batch: int = 64

    min_frames: int = 3
    min_quality: float = Field(0.20, ge=0.0, le=1.0)

    # Box height in pixels beyond which resolution stops limiting quality.
    ideal_box_height: float = 192.0
    # A standing person is roughly 2.5x taller than wide; boxes far from this
    # usually mean a partial body or two people merged into one detection.
    ideal_aspect: float = 2.5

    # Days after which a stored re-ID reference is worth half its original
    # weight. Re-ID largely encodes clothing, so it goes stale in a way face
    # and gait do not -- a strong match weeks later probably means a similar
    # jacket rather than the same person. Set to 0 to disable decay.
    trust_half_life_days: float = 3.0


class FusionSettings(BaseModel):
    """Phases 5-6: combining the three modalities into one score.

    The anchors put the modalities on a comparable scale. They are MEASUREMENTS,
    not guesses, but from very few clips -- re-derive them from the Phase-10
    TAR@FAR curve on real footage before trusting any of this operationally.
    """

    # "single_best", "average" (the paper's baseline), or "quality_weighted".
    strategy: str = "quality_weighted"

    # Face: measured on real crops. Different people 0.03, same person 0.95.
    face_impostor: float = 0.03
    face_genuine: float = 0.95

    # Gait: measured on synthetic walkers AFTER population centring. Without
    # centring the descriptor barely separates at all (different styles 0.986
    # vs same 1.000), which is why gait comparisons are refused when the
    # gallery is too small to centre against.
    gait_impostor: float = 0.52
    gait_genuine: float = 0.95

    # Re-ID: re-measured under the msmt17 checkpoint (DES-01). The previous
    # values -- impostor 0.755, genuine 0.980 -- came from OSNet's ImageNet
    # weights, which are not a re-ID model at all.
    #
    # Both checkpoints, same fixtures, enrolment clip against probe clip:
    #
    #                  impostor   genuine   separation
    #   imagenet         0.765     0.942      0.177
    #   msmt17           0.726     0.881      0.155
    #
    # The impostor drops, which is the expected effect of a model trained to
    # tell people apart. The genuine score drops too, and by more -- these
    # fixtures are two crops of the same photograph, so ImageNet's generic
    # features were partly matching the image rather than the person.
    #
    # Still a very small sample: one photograph, two people, no variation in
    # pose or lighting. Re-derive from the Phase-10 TAR@FAR curve on real
    # footage before operational use.
    reid_impostor: float = 0.726
    reid_genuine: float = 0.881

    # Which checkpoint the two numbers above were measured against. When this
    # disagrees with reid.weights the calibration does not apply to the model
    # actually running, and the mismatch is reported rather than left for
    # someone to notice.
    reid_anchors_measured_on: str = "msmt17"

    # Fused score above which a track is a candidate match. On the calibrated
    # scale, 0.5 means "halfway between a stranger and a genuine match".
    threshold: float = Field(0.55, ge=0.0, le=1.0)

    # Gait descriptors are dominated by the shared "generic human" shape, so
    # they must be compared with the population mean removed. That needs enough
    # references to estimate a mean; below this many, gait refuses to compare
    # rather than returning a misleadingly high similarity.
    gait_min_references_for_centring: int = 3
    #: Whether `gait_impostor` and `gait_genuine` were measured on real footage.
    #:
    #: They were not; both come from synthetic walkers. Until they are, gait is
    #: still computed and compared, and the reviewer is told that it was not
    #: counted and why -- but it carries no weight in the score.
    #:
    #: On real footage every centred gait similarity measured so far -- genuine
    #: and impostor alike, -0.76 to +0.41 -- sits below the 0.52 impostor
    #: anchor, so each one calibrates to exactly 0.0. Gait's separation is 0.43
    #: and its trust is always 1.0, so at the gait quality measured on the test
    #: clips (0.97) it would have taken 21-65% of the weight of every one of the
    #: eight test identifications while contributing nothing, and seven of them
    #: would have fallen under the threshold. That is the failure separation
    #: weighting was introduced for with re-ID, in a modality with twice the pull.
    #:
    #: Set True only after re-measuring both anchors on real walks: several
    #: people, each filmed walking on two separate occasions. One genuine pair
    #: exists so far, which measures nothing.
    gait_anchors_validated: bool = False


class TrackBufferSettings(BaseModel):
    """Per-track frame buffering that feeds the embedding branches."""

    # Hard cap on observations held per track. Unbounded buffers are how a
    # busy camera exhausts memory: 20 tracks x thousands of frames of crops.
    max_observations: int = 64
    #: Gait's own history per track: at most this many observations...
    gait_max_observations: int = 64
    #: ...thinned towards this many a second. 0 keeps every frame.
    #:
    #: `max_observations` is a frame count, so the time it covers depends on
    #: the camera: 2.6s at 25fps, 1.07s at 60fps. Gait will not speak until it
    #: has seen one whole stride, a little over a second for a normal walk.
    #: Measured on a clean side-on walk: of 76 windows of 64 frames at 60fps,
    #: not one held a complete stride. Thinned towards 20 a second, 64
    #: observations span 3.15s and 83% of windows pass every gait gate.
    #:
    #: A whole number of frames is skipped between the ones gait keeps, never a
    #: wall-clock slot -- see `TrackBuffer.gait_stride` for what uneven spacing
    #: costs. At 25fps and 29.97fps nothing is skipped and gait sees exactly
    #: what it saw before; at 60fps every third frame is kept.
    #:
    #: Its own ring, not a thinned main ring, because face and appearance choose
    #: their crops from the main ring by box height. Pacing that was tried and
    #: measured: someone walking into a camera has their tallest boxes mid-walk,
    #: when the face is still small, so a 3.2s window traded the close-ups for
    #: them -- on one clip an identification fell from 0.713 to 0.569 as its
    #: face score dropped from 0.83 to 0.59. The rings hold the same observation
    #: objects, so gait costs only the crops it keeps after the main ring has
    #: let them go.
    gait_sample_hz: float = Field(20.0, ge=0.0)
    # Crops are stored downscaled to this height; anything taller is resized.
    # 256px is comfortably above what ArcFace and OSNet need.
    store_height: int = 256
    # Skip observations whose box is shorter than this - too small to embed.
    min_box_height: int = 60
    # Ceiling on simultaneously buffered tracks; least-recently-seen is dropped.
    max_tracks: int = 50
    #: How near a frame edge a box may come, as a fraction of that dimension,
    #: before the person is treated as cut off.
    #:
    #: This was an exact comparison against the edge, and a detector box on a
    #: body running out of shot stops a pixel or two short of it. Measured on a
    #: clip of someone walking into the camera until only head and shoulders
    #: remained: 309 of 491 boxes ended within 4px of the bottom edge and 10
    #: were flagged, so the fragments reached gait presented as whole bodies.
    edge_margin_fraction: float = Field(0.01, ge=0.0, le=0.1)


class MatchingSettings(BaseModel):
    """Open-set matching against the watchlist gallery."""

    # Cosine similarity above which a track is considered a candidate match.
    # Calibrate on real footage via the Phase-10 TAR@FAR curve rather than
    # trusting this default.
    face_threshold: float = Field(0.40, ge=-1.0, le=1.0)
    # Gait, compared with the population mean removed. Measured on synthetic
    # walkers: different styles peaked at 0.516, same style 1.000. WITHOUT
    # centring, everything scores above 0.93 and this threshold would match
    # every person alive - which is why uncentred gait refuses to compare at
    # all rather than returning a number.
    gait_threshold: float = Field(0.70, ge=-1.0, le=1.0)
    # Re-ID cosine similarities cluster in a high, narrow band, so face-like
    # thresholds do not transfer.
    #
    # This was 0.88, from measurements taken against OSNet's ImageNet weights.
    # Re-measured under the msmt17 re-ID checkpoint (DES-01), the enrolled
    # subject scores 0.881 against their own probe clip and 0.726 against the
    # impostor -- so the old 0.88 sat directly on top of the genuine score and
    # would have rejected a correct match on a rounding error. A stale
    # threshold does not announce itself; it just stops finding people.
    #
    # 0.80 is the midpoint of the measured pair, leaving similar margin either
    # side. Still conservative in intent, because a false identification is
    # worse than a missed one, but no longer conservative past the point of
    # not working. Calibrate properly with the Phase-10 TAR@FAR curve: both
    # figures come from a single photograph of two people, which is far easier
    # than a real cross-camera, cross-day match.
    reid_threshold: float = Field(0.80, ge=-1.0, le=1.0)
    # Minimum observations before a track is matched at all, so an identity is
    # never asserted off a single frame.
    min_track_observations: int = 5
    # Re-match a live track at most this often (in processed frames).
    rematch_every: int = 15
    # Cap on how many of a track's buffered observations are embedded per
    # match. Measured: embedding all 64 buffered crops made the face branch
    # 70% of total runtime at 5.3s per track. Aggregation is quality-weighted,
    # so the poorest crops barely move the result -- embedding the largest few
    # gets nearly the same vector for a fraction of the cost. 0 means no cap.
    max_observations_to_embed: int = 16


class AlertSettings(BaseModel):
    """Phase 9: notifications for confirmed identifications.

    Sending is OFF by default and requires an explicit flag as well as
    configuration. Accidentally messaging a real contact list during testing is
    not recoverable, so the safe state is the default state.
    """

    # "console" (prints), "smtp", or "twilio".
    transport: str = "console"

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    # SecretStr so the value cannot leak through repr(settings) or a pydantic
    # validation error, both of which print field values verbatim.
    smtp_password: SecretStr = SecretStr("")
    smtp_sender: str = ""
    smtp_use_tls: bool = True
    #: Comma-separated in config/env; parsed into a list at use.
    smtp_recipients: str = ""

    twilio_account_sid: str = ""
    twilio_auth_token: SecretStr = SecretStr("")
    twilio_from_number: str = ""
    twilio_to_numbers: str = ""

    #: Included in SMS, which is too short to carry the full breakdown.
    console_url: str = "http://localhost:4173"

    @staticmethod
    def split(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]


class VideoSettings(BaseModel):
    frame_stride: int = Field(1, ge=1)
    max_frames: int | None = None
    save_annotated: bool = False


class LoggingSettings(BaseModel):
    level: str = "INFO"
    progress_every: int = Field(30, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRS_",
        env_nested_delimiter="__",
        # "forbid", not "ignore". With "ignore", FRS_DATABASE_URL was accepted
        # and silently dropped because no such field existed -- so anyone
        # following the README to move the watchlist onto Postgres got a local
        # SQLite file holding the biometric templates and the whole audit
        # trail, with no error and no warning. A mistyped setting must fail
        # loudly, not quietly do the insecure thing.
        extra="forbid",
    )

    project_name: str = "Faceless FRS"
    device: Literal["auto", "cuda", "cpu"] = "auto"

    #: SQLAlchemy URL. Defaults to SQLite under the data directory.
    #: Prefer this over `serve.py --db-url`, which puts the password in the
    #: process command line where any local user can read it.
    database_url: str | None = None

    #: Background job workers. One by default: the models contend for the same
    #: GPU memory, so concurrent scans slow each other and risk running out.
    job_workers: int = Field(1, ge=1, le=8)

    paths: PathSettings = PathSettings()
    detection: DetectionSettings = DetectionSettings()
    tracking: TrackingSettings = TrackingSettings()
    video: VideoSettings = VideoSettings()
    logging: LoggingSettings = LoggingSettings()
    face: FaceSettings = FaceSettings()
    gait: GaitSettings = GaitSettings()
    reid: ReIDSettings = ReIDSettings()
    fusion: FusionSettings = FusionSettings()
    alerts: AlertSettings = AlertSettings()
    track_buffer: TrackBufferSettings = TrackBufferSettings()
    matching: MatchingSettings = MatchingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources so environment variables beat the YAML file.

        Pydantic's default puts `init` first, which would let config.yaml
        override an env var if the YAML were passed as kwargs. Loading YAML as
        its own lowest-priority source is what makes the documented precedence
        actually hold.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_active_config_path),
        )

    #: Skip sign-in entirely and treat every request as a fixed demo operator.
    #:
    #: For showing the prototype, where a login screen is friction and there is
    #: nothing real behind it. It does NOT delete the mechanism: reviews are
    #: still recorded against a named operator, so the audit trail keeps its
    #: shape and turning this back off restores real accounts with no
    #: migration. The operator it records is obviously fake, which is the
    #: honest thing for a trail nobody should later mistake for evidence.
    #:
    #: Off by default, and it has to stay that way. An identification system
    #: that ships open because the safe setting was the one you had to
    #: remember is the shape of SEC-06, which this project already fixed once.
    demo_mode: bool = False

    def resolve_device(self) -> str:
        """Turn `device: auto` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # torch missing - detection would fail anyway
            return "cpu"

    def reid_calibration_mismatch(self) -> str | None:
        """Report re-ID anchors that were measured against a different model.

        The impostor and genuine anchors are what put re-ID on a comparable
        scale with face and gait. They are properties of a specific checkpoint:
        measured on one model, they say nothing about another. DES-01 found
        them measured against ImageNet weights, so switching to a re-ID
        checkpoint -- the right thing to do -- leaves the calibration stale in
        the opposite direction.

        Returns None when they agree, so callers can treat it as a flag.
        """
        measured = self.fusion.reid_anchors_measured_on
        running = self.reid.weights
        if measured == running:
            return None
        return (
            f"Re-ID calibration was measured against {measured!r} weights but "
            f"{running!r} weights are loaded. reid_impostor="
            f"{self.fusion.reid_impostor} and reid_genuine="
            f"{self.fusion.reid_genuine} do not describe the model that is "
            "running, so the calibrated re-ID score and anything derived from "
            "it -- the fused score, the threshold, the fusion weights -- are "
            "unreliable. Re-derive them with eval/ and set "
            "fusion.reid_anchors_measured_on to match."
        )


@lru_cache(maxsize=None)
def get_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings for `config_path` (default `backend/config.yaml`).

    Cached per path; call `get_settings.cache_clear()` in tests that change
    the environment and need a fresh read.
    """
    global _active_config_path
    _active_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    settings = Settings()
    # Anchor relative paths so the pipeline works from any working directory.
    settings.paths = settings.paths.resolved()
    return settings
