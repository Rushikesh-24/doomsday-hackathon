"""Detector interface shared by every model backend."""
from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision.schemas import ModelInfo
from vision.taxonomy import Taxonomy

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when weights are missing / cannot be downloaded / fail to load."""

    def __init__(self, key: str, reason: str):
        super().__init__(f"model '{key}' unavailable: {reason}")
        self.key = key
        self.reason = reason


@dataclass
class RawDetection:
    xyxy: np.ndarray        # (4,) float, pixel coords in the image given to detect()
    score: float
    class_name: str         # canonical taxonomy class
    prompt: str             # prompt / raw label that produced it
    detector: str           # model key


class Detector(ABC):
    """A loaded model that turns an RGB image into canonical-class detections."""

    def __init__(self, key: str, spec: dict[str, Any], taxonomy: Taxonomy, device: str, cache_dir: Path,
                 offline: bool = False):
        self.key = key
        self.spec = spec
        self.taxonomy = taxonomy
        self.device = device
        self.cache_dir = cache_dir
        self.offline = offline
        self.base_conf = float(spec.get("conf", 0.25))
        self._lock = threading.Lock()  # torch modules are not guaranteed thread-safe
        self.revision: str | None = None
        self.weights_sha256: str | None = None
        self.load_seconds: float | None = None

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]: ...

    def supported_classes(self) -> list[str]:
        return [c for c in self.taxonomy.names()]

    # ---- public ----------------------------------------------------------
    def detect(self, image_rgb: np.ndarray, conf: float | None = None,
               classes: list[str] | None = None) -> list[RawDetection]:
        """Return raw detections above `conf` (defaults to the model's base threshold,
        scaled down slightly so per-class thresholds can be applied afterwards)."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("expected HxWx3 RGB image")
        c = self.base_conf if conf is None else conf
        with self._lock:
            dets = self._detect(image_rgb, c)
        if classes:
            dets = [d for d in dets if d.class_name in classes]
        return dets

    def info(self) -> ModelInfo:
        return ModelInfo(
            key=self.key,
            name=self.spec.get("weights") or self.spec.get("hf_id") or self.key,
            version=self.spec.get("version", "unknown"),
            source=self.spec.get("source"),
            license=self.spec.get("license"),
            device=self.device,
            weights_sha256=self.weights_sha256,
            revision=self.revision,
        )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
