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

from pydantic import BaseModel, Field
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
    # Minimum observations before a track is matched at all, so an identity is
    # never asserted off a single frame.
    min_track_observations: int = 5
    # Re-match a live track at most this often (in processed frames).
    rematch_every: int = 15


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
        extra="ignore",
    )

    project_name: str = "Faceless FRS"
    device: Literal["auto", "cuda", "cpu"] = "auto"

    paths: PathSettings = PathSettings()
    detection: DetectionSettings = DetectionSettings()
    tracking: TrackingSettings = TrackingSettings()
    video: VideoSettings = VideoSettings()
    logging: LoggingSettings = LoggingSettings()
    face: FaceSettings = FaceSettings()
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
