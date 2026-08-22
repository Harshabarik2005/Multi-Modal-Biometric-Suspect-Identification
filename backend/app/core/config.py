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
    # Plausible half-cycle period in frames, used to bound the autocorrelation
    # search. At 25fps a half gait cycle is ~10-15 frames; below 7 you are into
    # sprinting territory, and short lags are exactly where segmentation jitter
    # produces fake periodicity.
    min_half_period: int = 7
    max_half_period: int = 40

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
    # OSNet's training input size. Do not change without retraining.
    input_height: int = 256
    input_width: int = 128

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

    # Re-ID: measured on real crops. Different people 0.755, same person 0.980.
    reid_impostor: float = 0.755
    reid_genuine: float = 0.980

    # Fused score above which a track is a candidate match. On the calibrated
    # scale, 0.5 means "halfway between a stranger and a genuine match".
    threshold: float = Field(0.55, ge=0.0, le=1.0)

    # Gait descriptors are dominated by the shared "generic human" shape, so
    # they must be compared with the population mean removed. That needs enough
    # references to estimate a mean; below this many, gait refuses to compare
    # rather than returning a misleadingly high similarity.
    gait_min_references_for_centring: int = 3


class TrackBufferSettings(BaseModel):
    """Per-track frame buffering that feeds the embedding branches."""

    # Hard cap on observations held per track. Unbounded buffers are how a
    # busy camera exhausts memory: 20 tracks x thousands of frames of crops.
    max_observations: int = 64
    # Crops are stored downscaled to this height; anything taller is resized.
    # 256px is comfortably above what ArcFace and OSNet need.
    store_height: int = 256
    # Skip observations whose box is shorter than this - too small to embed.
    min_box_height: int = 60
    # Ceiling on simultaneously buffered tracks; least-recently-seen is dropped.
    max_tracks: int = 50


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
    # thresholds do not transfer. Measured on real crops: two different people
    # scored 0.755, the same person across one track scored 0.980. A threshold
    # of 0.75 would therefore have matched strangers. 0.88 sits between the
    # two, leaning conservative because a false identification is worse than a
    # missed one. Calibrate properly with the Phase-10 TAR@FAR curve -- the
    # same-person figure here comes from one continuous track (identical
    # clothing, lighting and seconds apart), which is far easier than a real
    # cross-camera, cross-day match.
    reid_threshold: float = Field(0.88, ge=-1.0, le=1.0)
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
    console_url: str = "http://localhost:5173"

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

    def resolve_device(self) -> str:
        """Turn `device: auto` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # torch missing - detection would fail anyway
            return "cpu"


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
