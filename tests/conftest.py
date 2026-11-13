"""Shared fixtures. Unit tests use `ColorBlobDetector`, a tiny deterministic detector that
finds solid-colour rectangles, so the pipeline/video/API mechanics are tested without
downloading weights. Real-model tests live in test_integration_models.py."""
from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True, scope="session")
def _isolated_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    os.environ["VISION_OUTPUT_DIR"] = str(out)
    os.environ.setdefault("VISION_CACHE_DIR", str(ROOT / "models_cache"))
    from vision import config
    config.reset_settings()
    yield


# RGB colour -> (canonical class, prompt)
COLOR_CLASSES = {
    (255, 0, 0): ("staircase", "stairs"),
    (0, 0, 255): ("handrail", "handrail"),
    (0, 255, 0): ("obstacle", "chair"),
    (255, 255, 0): ("walkway", "walkway"),
}


def make_blob_detector(registry):
    from vision.models.base import Detector, RawDetection

    class ColorBlobDetector(Detector):
        def load(self):
            self.load_seconds = 0.0

        def _detect(self, rgb, conf):
            out = []
            for color, (cname, prompt) in COLOR_CLASSES.items():
                diff = np.abs(rgb.astype(int) - np.array(color)).sum(axis=2)
                mask = (diff < 90).astype(np.uint8)  # tolerant to video compression
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
                for i in range(1, n):
                    x, y, w, h, area = stats[i]
                    if area < 50:
                        continue
                    score = 0.9 if area > 400 else 0.2  # small blobs fall under threshold
                    if score >= conf:
                        out.append(RawDetection(np.array([x, y, x + w, y + h], float), score, cname, prompt, self.key))
            return out

    spec = {"type": "fake", "conf": 0.5, "version": "blob-1", "license": "test"}
    det = ColorBlobDetector("blob", spec, registry.taxonomy, "cpu", registry.settings.cache_dir)
    det.load()
    registry.specs["blob"] = spec
    registry._models["blob"] = det
    registry.default_key = "blob"
    return det


@pytest.fixture
def blob_pipeline():
    from vision.config import get_settings
    from vision.inference.pipeline import VisionPipeline
    from vision.models.registry import ModelRegistry
    reg = ModelRegistry(get_settings())
    make_blob_detector(reg)
    return VisionPipeline(registry=reg)


def scene(w=640, h=480) -> np.ndarray:
    """Synthetic scene: yellow walkway, red stairs, blue handrail beside stairs, green chair on walkway."""
    img = np.full((h, w, 3), 128, np.uint8)
    img[300:480, 0:640] = (255, 255, 0)          # walkway
    img[100:300, 100:300] = (255, 0, 0)          # stairs
    img[90:290, 305:315] = (0, 0, 255)           # handrail just right of stairs
    img[380:460, 300:360] = (0, 255, 0)          # chair centred on walkway
    img[20:27, 600:607] = (0, 255, 0)            # tiny blob (49px) -> ignored
    return img


def to_bytes(rgb: np.ndarray, fmt="JPEG", **kw) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format=fmt, **kw)
    return b.getvalue()


def png_bytes(rgb: np.ndarray) -> bytes:
    return to_bytes(rgb, "PNG")  # lossless: colours survive exactly
