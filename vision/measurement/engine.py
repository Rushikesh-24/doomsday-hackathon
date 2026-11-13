"""Measurement engine:
image -> facility detections (reused, never re-run when supplied) -> per-object masks (SAM box prompts)
-> semantic context (ADE20K) -> metric depth -> camera intrinsics -> gravity (ground plane / vanishing
point) -> scale recovery -> geometric analyzers -> ground-truth comparison -> rules -> JSON + image."""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision.config import get_settings
from vision.measurement import store
from vision.measurement.analyzers import ANALYZERS
from vision.measurement.backends import (GROUND_LABELS, RAIL_LABELS, SKY_LABELS, STAIR_LABELS, DOOR_LABELS,
                                         MeasurementModels, get_measurement_models, load_measurement_config)
from vision.measurement.camera import exif_focal_35mm, resolve_intrinsics, undistort
from vision.measurement.context import Gravity, SceneContext
from vision.measurement.geometry import angle_deg, fit_plane, orient_up, vertical_vanishing_up
from vision.measurement.rules import assess
from vision.measurement.scale import recover_scale
from vision.measurement.schemas import (CameraInfo, FacilityResult, GroundTruthComparison, MeasurementBatchResult,
                                        MeasurementReport, MeasurementStatus as MS, ModelRef, ScaleInfo, ScaleStatus)
from vision.measurement.visualize import render
from vision.preprocessing.image import load_image
from vision.schemas import ImageInfo, ImageResult

log = logging.getLogger(__name__)
MEASURED_CLASSES = set(ANALYZERS)
MASKED_CLASSES = MEASURED_CLASSES - {"corridor", "walkway"} | {"handrail"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MeasurementEngine:
    def __init__(self, models: MeasurementModels | None = None, vision_pipeline=None):
        self.models = models or get_measurement_models()
        self.cfg = load_measurement_config()
        self._vision = vision_pipeline
        self.settings = get_settings()

    @property
    def vision(self):
        if self._vision is None:
            from vision.inference.pipeline import get_pipeline
            self._vision = get_pipeline()
        return self._vision

    # ------------------------------------------------------------------ detections
    def _resolve_detections(self, img, detections, vision_request_id, model) -> tuple[ImageResult, str]:
        if detections is not None:
            res = detections if isinstance(detections, ImageResult) else ImageResult.model_validate(detections)
            src = "provided"
        elif vision_request_id:
            p = store.get_vision_result_path(vision_request_id)
            if not p or not Path(p).exists():
                raise KeyError(f"vision result '{vision_request_id}' not found; analyze the image first or pass "
                               f"detections")
            res = ImageResult.model_validate_json(Path(p).read_text())
            if res.request_id != vision_request_id:
                raise KeyError(f"stored file for '{vision_request_id}' now holds '{res.request_id}' (overwritten); "
                               f"re-run detection")
            src = "stored"
        else:
            res = self.vision.analyze_prepared(img, model=model, save_dir=self.settings.output_dir / "images" /
                                               datetime.now().strftime("%Y%m%d"))
            return res, "computed"
        if (res.image.width, res.image.height) != (img.width, img.height):
            raise ValueError(f"detections were made on a {res.image.width}x{res.image.height} image but this image is "
                             f"{img.width}x{img.height}; they cannot be reused")
        return res, src

    # ------------------------------------------------------------------ masks
    def _masks(self, ctx: SceneContext, dets) -> None:
        todo = [d for d in dets if d.class_name in MASKED_CLASSES]
        boxes = [ctx.box(d) for d in todo]
        try:
            sam = self.models.box_segmenter()
            res = sam.predict(ctx.rgb, boxes) if boxes else []
            ctx.models.append(sam.ref())
        except Exception as e:  # SAM unavailable -> boxes/semantic fallback
            log.warning("SAM unavailable (%s); using semantic/box masks", e)
            res = [(None, 0.0)] * len(todo)
        sem = {"staircase": STAIR_LABELS, "door": DOOR_LABELS, "elevator": DOOR_LABELS, "handrail": RAIL_LABELS}
        for d, (m, iou) in zip(todo, res):
            box = ctx.box_mask(d, 0.03)
            src = "sam"
            if m is not None:
                m = m & box
            area_box = box.sum()
            if m is None or m.sum() < 0.12 * area_box:
                lab = sem.get(d.class_name)
                m2 = ctx.label_mask(lab) & box if lab else None
                if m2 is not None and m2.sum() > 0.12 * area_box:
                    m, src = m2, "ade20k_semantic_in_box"
                elif d.class_name in ("door", "elevator"):
                    m, src = ctx.box_mask(d, 0.0), "detection_box (fallback)"
                else:
                    m, src = (m if m is not None else np.zeros_like(box)), "sam (weak)"
            elif d.class_name == "staircase":
                m = m | (ctx.label_mask(STAIR_LABELS) & box)
                src = "sam + ade20k_stairs"
            ctx.masks[d.id] = (m, src, float(iou))

    # ------------------------------------------------------------------ gravity
    def _gravity(self, ctx: SceneContext) -> None:
        g = ctx.label_mask(GROUND_LABELS)
        for did, (m, _, _) in ctx.masks.items():
            g &= ~m
        for d in ctx.detections:
            if d.class_name in ("obstacle", "person", "staircase", "ramp"):
                g &= ~ctx.box_mask(d, 0.02)
        gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY)
        vp_up, n_lines, vp_sig = vertical_vanishing_up(gray, ctx.K)
        ground = None
        if g.sum() > 2000:
            ys, xs = np.nonzero(g)
            sel = np.random.default_rng(0).choice(len(ys), min(len(ys), 25000), replace=False)
            X = ctx.P[ys[sel], xs[sel]].astype(np.float64)
            pl = fit_plane(X, rel_thresh=0.02)
            if pl is not None and pl.inlier_ratio > 0.45:
                up = orient_up(pl.normal)
                ref = vp_up if vp_up is not None else np.array([0, -1.0, 0])
                if angle_deg(up, ref) < (15 if vp_up is not None else 50):
                    ground = pl
                    full = np.zeros_like(g)
                    full[ys[sel][pl.inliers], xs[sel][pl.inliers]] = True
                    ctx.ground_mask = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & g
        if ground is not None:
            up = orient_up(ground.normal)
            ctx.ground = ground
            agree = angle_deg(up, vp_up) if vp_up is not None else None
            ctx.gravity = Gravity(up, "ground_plane", ground.normal_sigma_deg + 0.5, agree)
        elif vp_up is not None:
            ctx.gravity = Gravity(vp_up, "vanishing_point", vp_sig)
        else:
            ctx.gravity = Gravity(None, "none", None)
        ctx._vp_lines = n_lines

    # ------------------------------------------------------------------ main
    def analyze(self, data: bytes | str | Path, name: str | None = None, *, detections=None,
                vision_request_id: str | None = None, calibration: dict | None = None,
                ground_truth: list[dict] | None = None, depth_model: str | None = None,
                rules_profile: str | None = None, detection_ids: list[str] | None = None,
                vision_model: str | None = None, scale_override: dict | None = None,
                depth_override: np.ndarray | None = None, save: bool = True, run_id: str | None = None,
                save_dir: Path | None = None) -> MeasurementReport:
        T = {}
        t0 = time.perf_counter()
        if isinstance(data, (str, Path)):
            name = name or Path(data).name
            data = Path(data).read_bytes()
        name = name or "image"
        s = self.settings
        img = load_image(data, name, max_bytes=int(s.max_upload_mb * 1e6), min_side=s.min_side, max_side=s.max_side,
                         max_infer_side=s.max_infer_side)
        calibration = calibration or {}
        f35 = exif_focal_35mm(data)
        T["decode"] = time.perf_counter() - t0
        t = time.perf_counter()
        res, det_src = self._resolve_detections(img, detections, vision_request_id, vision_model)
        T["detections"] = time.perf_counter() - t
        rid = run_id or uuid.uuid4().hex[:12]
        # analysis resolution
        W, H = img.width, img.height
        sa = min(1.0, self.cfg["analysis"]["max_side"] / max(W, H))
        rgb = img.rgb
        user_K = calibration.get("intrinsics")
        if user_K and user_K.get("dist"):
            Ktmp = resolve_intrinsics(W, H, user_K, None, None, 26, 0.15)
            rgb = undistort(rgb, Ktmp)
        rgb_a = cv2.resize(rgb, (round(W * sa), round(H * sa)), interpolation=cv2.INTER_AREA) if sa < 1 else rgb
        warnings = []
        # semantic scene parsing
        t = time.perf_counter()
        seg, semantic, seg_ref = None, None, None
        try:
            semantic = self.models.semantic()
            seg = semantic.predict(rgb_a)
            seg_ref = semantic.ref()
        except Exception as e:
            warnings.append(f"semantic segmentation unavailable ({e}); ground/railing evidence disabled")
        T["segmentation"] = time.perf_counter() - t
        sky = float(semantic.mask_of(seg, SKY_LABELS).mean()) if seg is not None else 0.0
        outdoor = sky > 0.02
        # depth
        t = time.perf_counter()
        dkey = depth_model or self.cfg["models"]["depth"].get("default", "auto")
        if dkey == "auto":
            dkey = "da2_metric_outdoor_small" if outdoor else "da2_metric_indoor_small"
        if depth_override is not None:
            dep = depth_override.astype(np.float32)
            if dep.shape != rgb_a.shape[:2]:
                dep = cv2.resize(dep, (rgb_a.shape[1], rgb_a.shape[0]), interpolation=cv2.INTER_NEAREST)
            dref = ModelRef(role="depth", key="override", name="supplied depth", version="-")
            fov = None
        else:
            db = self.models.depth(dkey)
            dr = db.predict(rgb_a)
            dep, dref, fov = dr.depth, dr.model, dr.fov_deg
        T["depth"] = time.perf_counter() - t
        Ko = resolve_intrinsics(W, H, user_K, f35, fov, self.cfg["camera"]["prior_focal_35mm"],
                                self.cfg["camera"]["prior_focal_rel_sigma"])
        Ka = Ko.scaled(sa) if sa < 1 else Ko
        Ka.width, Ka.height = rgb_a.shape[1], rgb_a.shape[0]
        vis_ref = ModelRef(role="detection", key=res.model.key, name=res.model.name, version=res.model.version,
                           license=res.model.license, device=res.model.device, revision=res.model.revision)
        ctx = SceneContext(run_id=rid, rgb=rgb_a, to_orig=1 / sa, K=Ka, depth_net=dep, depth_model=dref, seg=seg,
                           semantic=semantic, detections=list(res.detections), cfg=self.cfg, timestamp=_now(),
                           models=[m for m in (vis_ref, dref, seg_ref) if m is not None])
        ctx.set_scale(ScaleInfo(status=ScaleStatus.uncalibrated_monocular, scale_factor=1.0,
                                rel_sigma=self.cfg["uncertainty"]["uncalibrated_depth_scale_rel"]))
        targets = [d for d in res.detections if d.class_name in MEASURED_CLASSES and
                   (not detection_ids or d.id in detection_ids)]
        if detection_ids:
            missing = set(detection_ids) - {d.id for d in res.detections}
            if missing:
                raise KeyError(f"detection ids not in this result: {sorted(missing)}")
        t = time.perf_counter()
        self._masks(ctx, [d for d in res.detections if d.class_name in MASKED_CLASSES or d in targets])
        T["masks"] = time.perf_counter() - t
        t = time.perf_counter()
        self._gravity(ctx)
        scale = recover_scale(ctx, calibration, scale_override)
        ctx.set_scale(scale)
        if ctx.ground is not None:
            ctx.ground.d *= scale.scale_factor
        T["scale"] = time.perf_counter() - t
        t = time.perf_counter()
        facilities = []
        for d in targets:
            m = ctx.masks.get(d.id)
            fr = FacilityResult(detection_id=d.id, class_name=d.class_name, detection_confidence=d.confidence,
                                bbox=d.bbox.xyxy(), mask_area_px=int(m[0].sum() / sa / sa) if m else None,
                                mask_source=m[1] if m else None)
            if m is not None:
                from vision.measurement.geometry import touching_sides
                fr.truncated_sides = touching_sides(m[0], self.cfg["analysis"]["border_margin_px"])
                zz = ctx.P[..., 2][m[0]]
                fr.mean_depth_m = round(float(np.median(zz)), 3) if zz.size else None
            try:
                fr.measurements = ANALYZERS[d.class_name](ctx, d)
            except Exception as e:
                log.exception("analyzer failed for %s", d.id)
                fr.notes.append(f"analyzer error: {e.__class__.__name__}: {e}")
                fr.measurements = [ctx.refuse(d, "analysis", f"internal analyzer error: {e}")]
            facilities.append(fr)
        T["analysis"] = time.perf_counter() - t
        self._apply_ground_truth(facilities, ground_truth or [], warnings)
        items, summary = assess(facilities, rules_profile)
        g = ctx.gravity
        scene = {"indoor_outdoor": "outdoor" if outdoor else "indoor", "sky_fraction": round(sky, 3),
                 "depth_model": dref.key, "analysis_size": [rgb_a.shape[1], rgb_a.shape[0]],
                 "gravity_source": g.source, "gravity_sigma_deg": None if g.sigma_deg is None else round(g.sigma_deg, 2),
                 "gravity_ground_vs_vp_deg": None if g.agreement_deg is None else round(g.agreement_deg, 2),
                 "vertical_lines": getattr(ctx, "_vp_lines", 0),
                 "ground_plane_inlier_ratio": round(ctx.ground.inlier_ratio, 3) if ctx.ground else None,
                 "camera_height_m": round(abs(ctx.ground.d), 2) if ctx.ground and scale.calibrated else None,
                 "measured_detections": len(targets), "detections_total": len(res.detections)}
        cam = CameraInfo(fx=Ko.fx, fy=Ko.fy, cx=Ko.cx, cy=Ko.cy, width=W, height=H, hfov_deg=round(Ko.hfov_deg, 2),
                         source=Ko.source, focal_rel_sigma=Ko.focal_rel_sigma, distortion_corrected=bool(Ko.dist))
        warnings += Ko.notes
        rep = MeasurementReport(run_id=rid, image=ImageInfo(name=name, width=W, height=H, format=img.format,
                                                            exif_transposed=img.exif_transposed),
                                vision_request_id=res.request_id, detections_source=det_src, camera=cam, scale=scale,
                                models=ctx.models, scene=scene, facilities=facilities, assessment=items,
                                assessment_summary=summary, timing_ms={}, processed_at=_now(), warnings=warnings)
        T["total"] = time.perf_counter() - t0
        rep.timing_ms = {k: round(v * 1000, 1) for k, v in T.items()}
        if save:
            out = save_dir or s.output_dir / "measurements" / rid
            out.mkdir(parents=True, exist_ok=True)
            ann = render(img.rgb, rep, {k: v[0] for k, v in ctx.masks.items() if k in {f.detection_id for f in facilities}},
                         calibration)
            p = out / f"{Path(name).stem}_measured.jpg"
            cv2.imwrite(str(p), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rep.annotated_image = str(p)
            rep.json_path = str(out / "report.json")
            Path(rep.json_path).write_text(rep.model_dump_json(indent=2))
            try:
                store.save_report(rep)
            except Exception as e:
                rep.warnings.append(f"history store unavailable: {e}")
        log.info("measured %s: %d facilities, scale=%s, %.1fs", name, len(facilities), scale.status.value, T["total"])
        return rep

    @staticmethod
    def _apply_ground_truth(facilities: list[FacilityResult], gt: list[dict], warnings: list[str]) -> None:
        """Manual measurements: compare with the engine value, then mark the record VERIFIED."""
        for g in gt:
            hit = False
            for f in facilities:
                if f.detection_id != g.get("detection_id"):
                    continue
                for m in f.measurements:
                    if m.name != g.get("name"):
                        continue
                    hit = True
                    manual = float(g["value"])
                    ev = m.value if isinstance(m.value, (int, float)) and not isinstance(m.value, bool) else m.provisional_value
                    cmp_ = GroundTruthComparison(manual_value=manual, engine_value=ev)
                    if ev is not None:
                        err = float(ev) - manual
                        cmp_.error, cmp_.abs_error = round(err, 4), round(abs(err), 4)
                        cmp_.rel_error = round(err / manual, 4) if manual else None
                        if m.uncertainty and m.status == MS.ESTIMATED:
                            cmp_.within_reported_uncertainty = abs(err) <= 1.96 * m.uncertainty
                    m.ground_truth = cmp_
                    m.notes.append(f"engine status before verification: {m.status.value}")
                    m.status = MS.VERIFIED
                    m.value = manual
                    m.uncertainty = float(g.get("uncertainty", 0.005))
                    m.verification_required = False
                    m.verification_reason = g.get("source", "manual on-site measurement supplied by user")
            if not hit:
                warnings.append(f"ground truth entry not matched: {g}")

    # ------------------------------------------------------------------ batch / multi-image
    def analyze_batch(self, items: list[tuple[bytes | str | Path, str]], *, calibrations: dict | None = None,
                      detections: dict | None = None, facility_groups: dict | None = None,
                      ground_truth: dict | None = None, **kw) -> MeasurementBatchResult:
        bid = uuid.uuid4().hex[:12]
        calibrations, detections, ground_truth = calibrations or {}, detections or {}, ground_truth or {}
        reports: dict[str, MeasurementReport] = {}
        raw: dict[str, bytes] = {}
        errors = []
        for data, name in items:
            if isinstance(data, (str, Path)):
                name = name or Path(data).name
                data = Path(data).read_bytes()
            raw[name] = data
            try:
                reports[name] = self.analyze(data, name, calibration=calibrations.get(name),
                                             detections=detections.get(name), ground_truth=ground_truth.get(name), **kw)
            except Exception as e:
                errors.append({"name": name, "error": f"{e.__class__.__name__}: {e}"})
        fused = []
        if facility_groups:
            members = self._resolve_groups(facility_groups, reports)
            self._transfer_scale(members, reports, raw, calibrations, ground_truth, kw, detections)
            fused = self._fuse(members, reports)
        return MeasurementBatchResult(batch_id=bid, reports=list(reports.values()), fused=fused, errors=errors)

    @staticmethod
    def _resolve_groups(groups: dict, reports: dict) -> dict[str, list[tuple[str, str]]]:
        out = {}
        for fid, mem in groups.items():
            lst = []
            for m in mem:
                rep = reports.get(m["image"])
                if rep is None:
                    continue
                if "detection_id" in m:
                    lst.append((m["image"], m["detection_id"]))
                else:  # class + rank by confidence
                    fs = sorted([f for f in rep.facilities if f.class_name == m["class"]],
                                key=lambda f: -f.detection_confidence)
                    if len(fs) > m.get("index", 0):
                        lst.append((m["image"], fs[m.get("index", 0)].detection_id))
            out[fid] = lst
        return out

    def _transfer_scale(self, members, reports, raw, calibrations, ground_truth, kw, detections=None) -> None:
        calibrated = lambda r: r.scale.status not in (ScaleStatus.uncalibrated_monocular,)
        for fid, lst in members.items():
            cal = [(img, did) for img, did in lst if calibrated(reports[img])]
            unc = [(img, did) for img, did in lst if not calibrated(reports[img])]
            for img_u, did_u in unc:
                fu = next(f for f in reports[img_u].facilities if f.detection_id == did_u)
                ratios, rels = [], []
                for img_c, did_c in cal:
                    fc = next(f for f in reports[img_c].facilities if f.detection_id == did_c)
                    for mc in fc.measurements:
                        if mc.unit != "m" or mc.status != MS.ESTIMATED or mc.lower_bound or not mc.value:
                            continue
                        mu = next((x for x in fu.measurements if x.name == mc.name and x.provisional_value), None)
                        if mu and not mu.lower_bound:
                            ratios.append(mc.value / mu.provisional_value)
                            rels.append((mc.uncertainty or 0) / mc.value)
                if not ratios:
                    continue
                s = float(np.median(ratios))
                rel = float(math.sqrt(np.mean(np.square(rels)) + (np.std(ratios) / s) ** 2 + 0.04 ** 2))
                kw2 = {k: v for k, v in kw.items() if k not in ("scale_override",)}
                rep = reports[img_u]
                dets = (detections or {}).get(img_u) or self._load_res(rep)  # reuse, never re-detect
                reports[img_u] = self.analyze(raw[img_u], img_u, detections=dets,
                                              calibration=calibrations.get(img_u),
                                              ground_truth=ground_truth.get(img_u),
                                              scale_override={"scale": s, "rel_sigma": rel,
                                                              "description": f"from facility '{fid}' in "
                                                                             f"{[c[0] for c in cal]}"}, **kw2)

    @staticmethod
    def _load_res(rep: MeasurementReport):
        p = store.get_vision_result_path(rep.vision_request_id) if rep.vision_request_id else None
        return json.loads(Path(p).read_text()) if p and Path(p).exists() else None

    @staticmethod
    def _fuse(members, reports) -> list[dict]:
        out = []
        for fid, lst in members.items():
            by: dict[str, list] = {}
            for img, did in lst:
                f = next((f for f in reports[img].facilities if f.detection_id == did), None)
                if not f:
                    continue
                for m in f.measurements:
                    if m.status in (MS.ESTIMATED, MS.VERIFIED) and isinstance(m.value, (int, float)) \
                            and not isinstance(m.value, bool) and m.uncertainty:
                        by.setdefault(m.name, []).append((img, m))
            for name, vals in by.items():
                v = np.array([m.value for _, m in vals], float)
                s = np.array([max(m.uncertainty, 1e-6) for _, m in vals], float)
                w = 1 / s ** 2
                mu = float((w * v).sum() / w.sum())
                sig = float(1 / math.sqrt(w.sum()))
                chi2 = float((((v - mu) / s) ** 2).sum())
                dof = len(v) - 1
                out.append({"facility_id": fid, "name": name, "unit": vals[0][1].unit, "value": round(mu, 4),
                            "uncertainty": round(sig, 4), "n_images": len(v),
                            "consistent": bool(dof == 0 or chi2 / dof < 4.0),
                            "chi2_per_dof": None if dof == 0 else round(chi2 / dof, 2),
                            "status": "VERIFIED" if any(m.status == MS.VERIFIED for _, m in vals) else "ESTIMATED",
                            "inputs": [{"image": i, "measurement_id": m.id, "value": m.value,
                                        "uncertainty": m.uncertainty} for i, m in vals],
                            "method": "inverse-variance weighted mean across images"})
        return out


_engine: MeasurementEngine | None = None


def get_engine() -> MeasurementEngine:
    global _engine
    if _engine is None:
        _engine = MeasurementEngine()
    return _engine
