"""Ultralytics backends: YOLO-World (open vocabulary) and closed-set COCO YOLO."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from vision.models.base import Detector, ModelUnavailableError, RawDetection, sha256_file

log = logging.getLogger(__name__)


def resolve_ultralytics_weights(weights: str, cache_dir: Path, offline: bool, key: str) -> Path:
    """Return a local path for `weights`, downloading the official Ultralytics release asset
    into our cache if needed. Custom (fine-tuned) weights may be given as a path."""
    p = Path(weights)
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise ModelUnavailableError(key, f"weights file not found: {p}")
        return p
    target = cache_dir / "ultralytics" / weights
    if target.exists() and target.stat().st_size > 0:
        return target
    if offline:
        raise ModelUnavailableError(key, f"weights {weights} not cached at {target} and VISION_OFFLINE=1")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        log.info("Downloading %s from Ultralytics GitHub releases -> %s", weights, target)
        out = Path(attempt_download_asset(str(target)))
    except Exception as e:  # network errors, 404 ...
        raise ModelUnavailableError(key, f"download of {weights} failed: {e}") from e
    if not out.exists():
        raise ModelUnavailableError(key, f"download of {weights} produced no file")
    return out


class YoloWorldDetector(Detector):
    """YOLO-World v2. All taxonomy prompts are embedded once with CLIP at load time;
    per-request class restrictions are applied as output filters (no re-embedding)."""

    def load(self) -> None:
        t = time.time()
        path = resolve_ultralytics_weights(self.spec["weights"], self.cache_dir, self.offline, self.key)
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(str(path))
            self.prompts = self.taxonomy.all_prompts()
            # CLIP ViT-B/32 text encoder is downloaded by Ultralytics on first use (OpenAI CDN).
            self.model.set_classes(self.prompts)
        except ModelUnavailableError:
            raise
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load YOLO-World: {e}") from e
        self.imgsz = int(self.spec.get("imgsz", 640))
        self.weights_sha256 = sha256_file(path)
        self.load_seconds = time.time() - t

    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]:
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        res = self.model.predict(bgr, conf=conf, iou=0.5, imgsz=self.imgsz, device=self.device,
                                 verbose=False, max_det=300)[0]
        out = []
        if res.boxes is None:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, s, c in zip(boxes, scores, cls):
            prompt = self.prompts[c]
            cname = self.taxonomy.class_for_prompt(prompt)
            if cname:
                out.append(RawDetection(b, float(s), cname, prompt, self.key))
        return out


class YoloCocoDetector(Detector):
    """Closed-set YOLO (COCO-80 or a fine-tuned checkpoint). Only COCO labels mapped in the taxonomy (`coco_labels`)
    are reported; it cannot detect stairs/ramps/etc. and does not pretend to."""

    def load(self) -> None:
        t = time.time()
        path = resolve_ultralytics_weights(self.spec["weights"], self.cache_dir, self.offline, self.key)
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load YOLO: {e}") from e
        self.names = self.model.names
        self.imgsz = int(self.spec.get("imgsz", 640))
        self.weights_sha256 = sha256_file(path)
        self.load_seconds = time.time() - t

    def _map(self, label: str) -> str | None:
        # COCO label via taxonomy `coco_labels`, or a fine-tuned model whose names ARE canonical classes
        return self.taxonomy.class_for_coco(label) or (label if label in self.taxonomy.classes else None)

    def supported_classes(self) -> list[str]:
        return sorted({self._map(n) for n in self.names.values()} - {None})

    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]:
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        res = self.model.predict(bgr, conf=conf, iou=0.5, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int)):
            label = self.names[c]
            cname = self._map(label)
            if cname:
                out.append(RawDetection(b, float(s), cname, label, self.key))
        return out
