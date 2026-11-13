"""Runtime settings, read from environment variables with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass
class Settings:
    models_config: Path = field(default_factory=lambda: _env_path("VISION_MODELS_CONFIG", REPO_ROOT / "configs/models.yaml"))
    taxonomy_config: Path = field(default_factory=lambda: _env_path("VISION_TAXONOMY_CONFIG", REPO_ROOT / "configs/taxonomy.yaml"))
    cache_dir: Path = field(default_factory=lambda: _env_path("VISION_CACHE_DIR", REPO_ROOT / "models_cache"))
    output_dir: Path = field(default_factory=lambda: _env_path("VISION_OUTPUT_DIR", REPO_ROOT / "outputs"))
    device: str = field(default_factory=lambda: os.environ.get("VISION_DEVICE", "auto"))  # auto|cuda|mps|cpu
    default_model: str | None = field(default_factory=lambda: os.environ.get("VISION_DEFAULT_MODEL"))
    # Allow automatic weight downloads. Set VISION_OFFLINE=1 to forbid network access.
    offline: bool = field(default_factory=lambda: os.environ.get("VISION_OFFLINE", "0") == "1")
    # Image validation limits
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("VISION_MAX_UPLOAD_MB", "25")))
    min_side: int = 32
    max_side: int = 12000
    max_infer_side: int = 1920  # larger images are downscaled before inference (boxes mapped back)
    # Video limits
    max_video_mb: float = field(default_factory=lambda: float(os.environ.get("VISION_MAX_VIDEO_MB", "300")))
    max_video_seconds: float = field(default_factory=lambda: float(os.environ.get("VISION_MAX_VIDEO_SECONDS", "600")))

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Route all HF downloads into our cache unless the user configured HF_HOME already.
        os.environ.setdefault("HF_HOME", str(self.cache_dir / "hf"))
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """For tests."""
    global _settings
    _settings = None
