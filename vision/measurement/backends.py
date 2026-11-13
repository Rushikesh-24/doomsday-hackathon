"""Model backends for measurement: metric depth, semantic segmentation (ADE20K) and
box-prompted instance masks (SAM). Each is loaded once and cached; accelerator failures
fall back to CPU permanently for that model."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from vision.config import REPO_ROOT, get_settings
from vision.device import select_device
from vision.measurement.schemas import ModelRef
from vision.models.base import ModelUnavailableError
from vision.models.hf_models import _resolve_hf

log = logging.getLogger(__name__)

# ADE20K-150 label groups used by the geometry
GROUND_LABELS = {"floor", "road", "sidewalk", "path", "earth", "grass", "field", "land", "rug", "runway", "dirt track"}
STAIR_LABELS = {"stairs", "stairway", "step"}
RAIL_LABELS = {"railing", "bannister"}
DOOR_LABELS = {"door", "screen door", "double door"}
SKY_LABELS = {"sky"}


@lru_cache(maxsize=2)
def load_measurement_config(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs/measurement.yaml").read_text())


class _HFBackend:
    role = "model"

    def __init__(self, key: str, spec: dict, device: str, offline: bool):
        self.key, self.spec, self.device, self.offline = key, spec, device, offline
        self._lock = threading.Lock()
        self.revision: str | None = None
        self.load_seconds: float | None = None

    def ref(self) -> ModelRef:
        return ModelRef(role=self.role, key=self.key, name=self.spec["hf_id"], version=self.spec.get("version", "?"),
                        license=self.spec.get("license"), device=self.device, revision=self.revision)

    def _place(self):
        try:
            self.model.to(self.device)
        except Exception as e:
            log.warning("%s: cannot use %s (%s) -> CPU", self.key, self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
        self.model.eval()

    def _run(self, fn):
        try:
            with torch.inference_mode():
                return fn()
        except (RuntimeError, NotImplementedError, TypeError) as e:
            if self.device == "cpu":
                raise
            log.warning("%s: inference failed on %s (%s) -> CPU", self.key, self.device, str(e)[:160])
            self.device = "cpu"
            self.model.to("cpu")
            with torch.inference_mode():
                return fn()

    @staticmethod
    def _to(inputs, device):
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.float() if v.dtype == torch.float64 else v
                v = v.to(device)
            out[k] = v
        return out


@dataclass
class DepthResult:
    depth: np.ndarray            # (H, W) float32 metres (network scale, before calibration)
    fov_deg: float | None        # horizontal FOV predicted by the model, if any
    model: ModelRef
    inference_ms: float


class DepthBackend(_HFBackend):
    role = "depth"

    def load(self):
        t = time.time()
        path, self.revision = _resolve_hf(self.spec["hf_id"], self.offline, self.key)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.processor = AutoImageProcessor.from_pretrained(path)
            self.model = AutoModelForDepthEstimation.from_pretrained(path)
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load depth model: {e}") from e
        self._place()
        self.load_seconds = time.time() - t

    def predict(self, rgb: np.ndarray) -> DepthResult:
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        t = time.perf_counter()
        with self._lock:
            def fn():
                inputs = self._to(self.processor(images=img, return_tensors="pt"), self.device)
                out = self.model(**inputs)
                pp = self.processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
                fov = getattr(out, "field_of_view", None)
                return pp["predicted_depth"].float().cpu().numpy(), (float(fov[0]) if fov is not None else None)
            depth, fov = self._run(fn)
        depth = np.asarray(depth, dtype=np.float32).reshape(h, w)
        return DepthResult(depth, fov if self.spec.get("estimates_fov") else None, self.ref(),
                           (time.perf_counter() - t) * 1000)


class SemanticBackend(_HFBackend):
    role = "semantic_segmentation"

    def load(self):
        t = time.time()
        path, self.revision = _resolve_hf(self.spec["hf_id"], self.offline, self.key)
        try:
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            self.processor = AutoImageProcessor.from_pretrained(path)
            self.model = UperNetForSemanticSegmentation.from_pretrained(path)
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load segmentation model: {e}") from e
        self._place()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.load_seconds = time.time() - t

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        with self._lock:
            def fn():
                inputs = self._to(self.processor(images=img, return_tensors="pt"), self.device)
                out = self.model(**inputs)
                return self.processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()
            return self._run(fn).astype(np.int32)

    def mask_of(self, seg: np.ndarray, names: set[str]) -> np.ndarray:
        ids = [i for i, n in self.id2label.items() if n in names]
        return np.isin(seg, ids)


class BoxSegmenter(_HFBackend):
    role = "box_segmenter"

    def load(self):
        t = time.time()
        path, self.revision = _resolve_hf(self.spec["hf_id"], self.offline, self.key)
        try:
            from transformers import SamModel, SamProcessor
            self.processor = SamProcessor.from_pretrained(path)
            self.model = SamModel.from_pretrained(path)
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load SAM: {e}") from e
        self._place()
        self.load_seconds = time.time() - t

    def predict(self, rgb: np.ndarray, boxes: list[list[float]]) -> list[tuple[np.ndarray, float]]:
        """One mask per box prompt: (bool mask HxW, predicted IoU score)."""
        if not boxes:
            return []
        img = Image.fromarray(rgb)
        with self._lock:
            def fn():
                inputs = self._to(self.processor(images=img, input_boxes=[boxes], return_tensors="pt"), self.device)
                out = self.model(**inputs)
                masks = self.processor.image_processor.post_process_masks(
                    out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
                return masks.numpy(), out.iou_scores.cpu().numpy()[0]
            masks, ious = self._run(fn)
        res = []
        for i in range(len(boxes)):
            j = int(np.argmax(ious[i]))
            res.append((masks[i, j].astype(bool), float(ious[i, j])))
        return res


class MeasurementModels:
    """Lazy, cached access to all measurement backends."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_measurement_config()
        s = get_settings()
        self.device = select_device(s.device)
        self.offline = s.offline
        self._cache: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    def _get(self, key: str, factory):
        with self._lock:
            if key not in self._cache:
                try:
                    b = factory()
                    b.load()
                except ModelUnavailableError as e:
                    self._errors[key] = e.reason
                    raise
                self._cache[key] = b
                self._errors.pop(key, None)
                log.info("measurement model %s loaded on %s in %.1fs", key, b.device, b.load_seconds or 0)
            return self._cache[key]

    def depth(self, key: str) -> DepthBackend:
        specs = self.cfg["models"]["depth"]
        if key not in specs or key == "default":
            raise KeyError(f"unknown depth model '{key}'; available: {[k for k in specs if k != 'default']}")
        if specs[key].get("accelerator_only") and self.device == "cpu":
            raise ModelUnavailableError(key, "this depth model needs a GPU (CUDA/MPS); on CPU it takes minutes per image")
        return self._get(key, lambda: DepthBackend(key, specs[key], self.device, self.offline))

    def semantic(self) -> SemanticBackend:
        return self._get("semantic", lambda: SemanticBackend("upernet_ade20k", self.cfg["models"]["segmentation"],
                                                             self.device, self.offline))

    def box_segmenter(self) -> BoxSegmenter:
        return self._get("sam", lambda: BoxSegmenter("slimsam", self.cfg["models"]["box_segmenter"],
                                                     self.device, self.offline))

    def describe(self) -> dict:
        d = {k: {"hf_id": v["hf_id"], "license": v.get("license"), "loaded": k in self._cache,
                 "error": self._errors.get(k)} for k, v in self.cfg["models"]["depth"].items() if k != "default"}
        d["semantic"] = {"hf_id": self.cfg["models"]["segmentation"]["hf_id"], "loaded": "semantic" in self._cache,
                         "error": self._errors.get("semantic")}
        d["sam"] = {"hf_id": self.cfg["models"]["box_segmenter"]["hf_id"], "loaded": "sam" in self._cache,
                    "error": self._errors.get("sam")}
        return d


_models: MeasurementModels | None = None


def get_measurement_models() -> MeasurementModels:
    global _models
    if _models is None:
        _models = MeasurementModels()
    return _models
