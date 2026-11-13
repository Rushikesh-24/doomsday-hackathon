"""Model registry: reads configs/models.yaml, lazily loads each model ONCE and caches it."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from vision.config import Settings, get_settings
from vision.device import select_device
from vision.models.base import Detector, ModelUnavailableError
from vision.taxonomy import Taxonomy, load_taxonomy

log = logging.getLogger(__name__)


def _detector_class(type_: str):
    if type_ == "yolo_world":
        from vision.models.ultralytics_models import YoloWorldDetector
        return YoloWorldDetector
    if type_ == "yolo_coco":
        from vision.models.ultralytics_models import YoloCocoDetector
        return YoloCocoDetector
    if type_ == "grounding_dino":
        from vision.models.hf_models import GroundingDinoDetector
        return GroundingDinoDetector
    if type_ == "owlv2":
        from vision.models.hf_models import Owlv2Detector
        return Owlv2Detector
    raise ValueError(f"unknown model type '{type_}'")


class ModelRegistry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        cfg = yaml.safe_load(Path(self.settings.models_config).read_text())
        self.specs: dict[str, dict[str, Any]] = cfg["models"]
        self.default_key: str = self.settings.default_model or cfg.get("default_model") or next(iter(self.specs))
        if self.default_key not in self.specs:
            raise ValueError(f"default model '{self.default_key}' not in {list(self.specs)}")
        self.taxonomy: Taxonomy = load_taxonomy(str(self.settings.taxonomy_config))
        self.device = select_device(self.settings.device)
        self._models: dict[str, Detector] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()  # re-entrant: ensembles load their members inside get()
        log.info("ModelRegistry: %d models configured, default=%s, device=%s",
                 len(self.specs), self.default_key, self.device)

    def _build(self, key: str) -> Detector:
        spec = self.specs[key]
        common = dict(taxonomy=self.taxonomy, device=self.device, cache_dir=self.settings.cache_dir,
                      offline=self.settings.offline)
        if spec["type"] == "ensemble":
            from vision.models.ensemble import EnsembleDetector
            members = [self.get(m) for m in spec["members"]]
            det = EnsembleDetector(key, spec, members=members, **common)
        else:
            det = _detector_class(spec["type"])(key, spec, **common)
        det.load()
        return det

    def get(self, key: str | None = None) -> Detector:
        key = key or self.default_key
        if key not in self.specs:
            raise KeyError(f"unknown model '{key}'. Available: {list(self.specs)}")
        if key in self._models:
            return self._models[key]
        with self._lock:
            if key in self._models:
                return self._models[key]
            log.info("Loading model %s ...", key)
            try:
                det = self._build(key)
            except ModelUnavailableError as e:
                self._errors[key] = e.reason
                log.error("%s", e)
                raise
            except Exception as e:  # unexpected
                self._errors[key] = str(e)
                log.exception("Unexpected error loading %s", key)
                raise ModelUnavailableError(key, str(e)) from e
            self._errors.pop(key, None)
            self._models[key] = det
            log.info("Loaded %s on %s in %.1fs", key, det.device, det.load_seconds or 0)
            return det

    def is_loaded(self, key: str) -> bool:
        return key in self._models

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for key, spec in self.specs.items():
            d = {
                "key": key,
                "type": spec["type"],
                "version": spec.get("version"),
                "source": spec.get("source"),
                "license": spec.get("license"),
                "open_vocabulary": spec.get("open_vocabulary", False),
                "base_confidence": spec.get("conf"),
                "default": key == self.default_key,
                "loaded": key in self._models,
                "error": self._errors.get(key),
            }
            if spec["type"] == "ensemble":
                d["members"] = spec["members"]
            if key in self._models:
                m = self._models[key]
                d["device"] = m.device
                d["load_seconds"] = round(m.load_seconds or 0, 2)
                d["supported_classes"] = m.supported_classes()
                d["weights_sha256"] = m.weights_sha256
                d["revision"] = m.revision
            out.append(d)
        return out


_registry: ModelRegistry | None = None
_rlock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _rlock:
        if _registry is None:
            _registry = ModelRegistry()
        return _registry
