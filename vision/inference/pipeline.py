"""End-to-end image pipeline: validate -> preprocess -> detect -> threshold -> NMS ->
attributes -> JSON + annotated image. Model inference lives in `vision.models`; this
module contains no HTTP code."""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision.accessibility.attributes import enrich
from vision.config import Settings, get_settings
from vision.models.base import RawDetection
from vision.models.registry import ModelRegistry, get_registry
from vision.postprocessing.nms import (classwise_nms, drop_degenerate, filter_by_threshold, merge_fragments,
                                      suppress_contained_duplicates)
from vision.preprocessing.image import InvalidImageError, PreparedImage, from_array, load_image
from vision.schemas import BatchResult, BBox, Detection, ImageInfo, ImageResult, SourceRef, Timing
from vision.visualization.draw import annotate

log = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class VisionPipeline:
    def __init__(self, registry: ModelRegistry | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.taxonomy = self.registry.taxonomy

    # ------------------------------------------------------------------ helpers
    def _owner(self, model_key: str, class_name: str) -> str:
        """Which (member) model produces `class_name` for `model_key`."""
        spec = self.registry.specs[model_key]
        if spec["type"] != "ensemble":
            return model_key
        mc = spec.get("member_classes", {}) or {}
        for m, cls in mc.items():
            if class_name in cls:
                return m
        return next(m for m in spec["members"] if m not in mc)

    def thresholds(self, model_key: str, conf: float | None,
                   class_conf: dict[str, float] | None = None) -> dict[str, float]:
        """Effective per-class confidence thresholds actually applied."""
        out = {}
        for c in self.taxonomy.classes.values():
            if class_conf and c.name in class_conf:
                out[c.name] = class_conf[c.name]
                continue
            base = conf if conf is not None else self.registry.specs[self._owner(model_key, c.name)].get("conf", 0.25)
            out[c.name] = round(base * c.threshold_scale, 4)
        return out

    def _threshold_fn(self, conf: float | None, class_conf: dict[str, float] | None):
        def thr(d: RawDetection) -> float:
            if class_conf and d.class_name in class_conf:
                return class_conf[d.class_name]
            base = conf if conf is not None else self.registry.specs[d.detector].get("conf", 0.25)
            return base * self.taxonomy.classes[d.class_name].threshold_scale
        return thr

    # ------------------------------------------------------------------ core
    def analyze_prepared(self, img: PreparedImage, *, model: str | None = None, conf: float | None = None,
                         class_conf: dict[str, float] | None = None, classes: list[str] | None = None,
                         iou: float = 0.5, include_auxiliary: bool = False, source: SourceRef | None = None,
                         save_dir: Path | None = None, save_annotated: bool = True, save_json: bool = True,
                         request_id: str | None = None, extra_detections=None) -> ImageResult:
        t0 = time.perf_counter()
        key = model or self.registry.default_key
        detector = self.registry.get(key)
        rid = request_id or uuid.uuid4().hex[:12]
        if classes:
            unknown = set(classes) - set(self.taxonomy.names())
            if unknown:
                raise ValueError(f"unknown classes {sorted(unknown)}; valid: {self.taxonomy.names()}")
        t1 = time.perf_counter()

        # Run slightly below the lowest effective threshold so per-class thresholds can act after.
        min_scale = min(c.threshold_scale for c in self.taxonomy.classes.values())
        raw_conf = None
        if conf is not None or class_conf:
            cands = ([conf * min_scale] if conf is not None else []) + list((class_conf or {}).values())
            raw_conf = max(0.001, min(cands))
        raw = detector.detect(img.infer_rgb, conf=raw_conf, classes=classes)
        t2 = time.perf_counter()

        # map to original resolution
        for d in raw:
            d.xyxy = np.asarray(d.xyxy, dtype=float) * img.scale
        n_raw = len(raw)
        dets = filter_by_threshold(raw, self._threshold_fn(conf, class_conf))
        dets = drop_degenerate(dets)
        dets = classwise_nms(dets, iou)
        dets = merge_fragments(dets, {c.name for c in self.taxonomy.classes.values() if c.merge_fragments})
        dets = suppress_contained_duplicates(dets)
        dets.sort(key=lambda d: -d.score)

        ts = now_iso()
        src = source or SourceRef(name=img.name, kind="image")
        info = detector.info()
        out: list[Detection] = []
        for i, d in enumerate(dets):
            out.append(Detection(
                id=f"{rid}-{i:03d}", class_name=d.class_name, category=self.taxonomy.category(d.class_name),
                confidence=round(d.score, 4), bbox=BBox.from_xyxy(d.xyxy, img.width, img.height),
                source=src, prompt=d.prompt, detector=d.detector, model_name=info.name,
                model_version=info.version, timestamp=ts))
        t3 = time.perf_counter()
        findings = enrich(out, img.rgb)
        if not include_auxiliary:
            visible = [d for d in out if d.category != "auxiliary"]
        else:
            visible = out
        t4 = time.perf_counter()

        counts: dict[str, int] = {}
        for d in visible:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        warnings = []
        if n_raw and not dets:
            warnings.append(f"{n_raw} raw candidates were all below threshold")
        if img.scale > 1.0:
            warnings.append(f"image downscaled by {img.scale:.2f}x for inference; boxes mapped back to original size")
        res = ImageResult(
            request_id=rid,
            image=ImageInfo(name=img.name, width=img.width, height=img.height, format=img.format,
                            exif_transposed=img.exif_transposed),
            model=info,
            thresholds=self.thresholds(key, conf, class_conf),
            detections=visible, counts_by_class=counts, findings=findings,
            timing=Timing(preprocess_ms=round((t1 - t0) * 1000, 1), inference_ms=round((t2 - t1) * 1000, 1),
                          postprocess_ms=round((t3 - t2) * 1000, 1), attributes_ms=round((t4 - t3) * 1000, 1),
                          total_ms=round((time.perf_counter() - t0) * 1000, 1)),
            processed_at=ts, warnings=warnings)
        if save_dir is not None and (save_annotated or save_json):
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{Path(img.name).stem}_{rid}"  # unique per request: stored results must never be overwritten
            if save_annotated:
                ann = annotate(img.rgb, out if include_auxiliary else visible, self.taxonomy)
                p = save_dir / f"{stem}_annotated.jpg"
                cv2.imwrite(str(p), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
                res.annotated_image = str(p)
            if save_json:
                # keep the (EXIF-corrected) source so measurements can later reuse it without a re-upload
                cv2.imwrite(str(save_dir / f"{stem}_source.jpg"), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                p = save_dir / f"{stem}.json"
                res.json_path = str(p)
                p.write_text(res.model_dump_json(indent=2))
                try:  # index for reuse by the measurement engine (POST /measurements/analyze vision_request_id)
                    from vision.measurement.store import register_vision_result
                    register_vision_result(rid, str(p), img.name, img.width, img.height)
                except Exception as e:  # never break detection because of the index
                    log.warning("could not index vision result: %s", e)
        log.info("%s: %d detections (%d raw) model=%s infer=%.0fms total=%.0fms", img.name, len(visible), n_raw,
                 key, res.timing.inference_ms, res.timing.total_ms)
        return res

    def analyze_image(self, data: bytes | str | Path, name: str | None = None, **kw) -> ImageResult:
        s = self.settings
        t = time.perf_counter()
        img = load_image(data, name, max_bytes=int(s.max_upload_mb * 1e6), min_side=s.min_side,
                         max_side=s.max_side, max_infer_side=s.max_infer_side)
        pre_ms = (time.perf_counter() - t) * 1000
        if "save_dir" not in kw:
            kw["save_dir"] = s.output_dir / "images" / datetime.now().strftime("%Y%m%d")
        res = self.analyze_prepared(img, **kw)
        res.timing.preprocess_ms = round(res.timing.preprocess_ms + pre_ms, 1)
        res.timing.total_ms = round(res.timing.total_ms + pre_ms, 1)
        return res

    def analyze_array(self, rgb: np.ndarray, name: str, **kw) -> ImageResult:
        return self.analyze_prepared(from_array(rgb, name, self.settings.max_infer_side), **kw)

    def analyze_batch(self, items: list[tuple[bytes | str | Path, str | None]], **kw) -> BatchResult:
        t = time.perf_counter()
        bid = uuid.uuid4().hex[:12]
        kw.setdefault("save_dir", self.settings.output_dir / "batches" / bid)
        results, errors = [], []
        for i, (data, name) in enumerate(items):
            nm = name or (Path(data).name if isinstance(data, (str, Path)) else f"image_{i}")
            try:
                results.append(self.analyze_image(data, nm, request_id=f"{bid}-{i:02d}", **kw))
            except InvalidImageError as e:
                errors.append({"name": nm, "error": str(e)})
        br = BatchResult(batch_id=bid, results=results, errors=errors,
                         total_ms=round((time.perf_counter() - t) * 1000, 1))
        p = Path(kw["save_dir"]) if kw.get("save_dir") else None
        if p:
            p.mkdir(parents=True, exist_ok=True)
            (p / "batch.json").write_text(br.model_dump_json(indent=2))
        return br


_pipeline: VisionPipeline | None = None


def get_pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline


def dump(result, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.model_dump(mode="json"), indent=2))
