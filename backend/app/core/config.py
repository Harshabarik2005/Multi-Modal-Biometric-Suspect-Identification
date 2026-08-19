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
