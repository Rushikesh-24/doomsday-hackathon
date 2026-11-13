"""Per-image analysis context shared by all analyzers, and the single place where
measurement status / uncertainty / refusal rules are applied."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vision.measurement.camera import Intrinsics
from vision.measurement.geometry import Plane, normal_map, point_map
from vision.measurement.schemas import (Measurement, MeasurementStatus as MS, ModelRef, Overlay, ScaleInfo,
                                        ScaleStatus)
from vision.schemas import Detection


@dataclass
class Gravity:
    up: np.ndarray | None
    source: str              # ground_plane | vanishing_point | none
    sigma_deg: float | None
    agreement_deg: float | None = None   # angle between ground-plane and vanishing-point estimates


@dataclass
class SceneContext:
    run_id: str
    rgb: np.ndarray                  # analysis-resolution RGB
    to_orig: float                   # multiply analysis px by this to get original px
    K: Intrinsics                    # analysis-resolution intrinsics
    depth_net: np.ndarray            # network depth (metres, uncalibrated)
    depth_model: ModelRef
    seg: np.ndarray | None
    semantic: object | None          # SemanticBackend (for label masks)
    detections: list[Detection]
    cfg: dict
    timestamp: str
    scale: ScaleInfo | None = None
    gravity: Gravity | None = None
    ground: Plane | None = None
    ground_mask: np.ndarray | None = None
    masks: dict[str, tuple[np.ndarray, str, float]] = field(default_factory=dict)  # det_id -> (mask, source, score)
    models: list[ModelRef] = field(default_factory=list)
    _counter: int = 0
    P: np.ndarray | None = None
    N: np.ndarray | None = None

    # ------------------------------------------------------------------ geometry state
    def set_scale(self, scale: ScaleInfo) -> None:
        self.scale = scale
        self.P = point_map(self.depth_net * scale.scale_factor, self.K)
        self.N = normal_map(self.P)

    def label_mask(self, names: set[str]) -> np.ndarray:
        if self.seg is None or self.semantic is None:
            return np.zeros(self.rgb.shape[:2], bool)
        return self.semantic.mask_of(self.seg, names)

    def box(self, det: Detection) -> list[float]:
        return [v / self.to_orig for v in det.bbox.xyxy()]

    def box_mask(self, det: Detection, pad: float = 0.0) -> np.ndarray:
        h, w = self.rgb.shape[:2]
        x1, y1, x2, y2 = self.box(det)
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        m = np.zeros((h, w), bool)
        m[max(0, int(y1 - ph)):min(h, int(math.ceil(y2 + ph))), max(0, int(x1 - pw)):min(w, int(math.ceil(x2 + pw)))] = True
        return m

    def orig(self, pts) -> list[list[float]]:
        return [[round(float(x) * self.to_orig, 1), round(float(y) * self.to_orig, 1)] for x, y in np.atleast_2d(pts)]

    def overlay(self, lines=(), points=(), polygon=None, label_at=None) -> Overlay:
        L = [[round(float(v) * self.to_orig, 1) for v in ln] for ln in lines]
        return Overlay(lines=L, points=self.orig(points) if len(points) else [],
                       polygon=self.orig(polygon) if polygon is not None and len(polygon) else None,
                       label_at=self.orig([label_at])[0] if label_at is not None else None)

    # ------------------------------------------------------------------ uncertainty
    def length_rel_sigma(self, z: float | None, geom_rel: float = 0.0, along_view: bool = False,
                         same_plane: bool = False) -> float:
        """Relative 1-sigma of an absolute length at depth z (calibrated scale). Depth terms are the
        errors measured on real photos with laser depth (configs/measurement.yaml)."""
        u = self.cfg["uncertainty"]
        s = self.scale
        rel = s.rel_sigma ** 2 + geom_rel ** 2
        if s.status != ScaleStatus.uncalibrated_monocular:
            if same_plane:   # ratio to a reference on the same planar object: depth errors largely cancel
                dr = u.get("same_plane_ratio_rel", 0.03)
            else:
                dr = u["depth_relative_rel"]
                if z and s.reference_depth_m:
                    dr += u["depth_relative_per_log_ratio"] * abs(math.log(max(z, 1e-3) / s.reference_depth_m))
            rel += dr ** 2
            if along_view and not same_plane:
                rel += u.get("along_view_extra_rel", 0.0) ** 2
        # lateral lengths also scale with focal error unless the reference absorbed it at similar depth
        if self.K.source == "prior" and s.status in (ScaleStatus.uncalibrated_monocular,):
            rel += self.K.focal_rel_sigma ** 2
        return math.sqrt(rel)

    # ------------------------------------------------------------------ measurement factory
    def _id(self) -> str:
        self._counter += 1
        return f"{self.run_id}-m{self._counter:03d}"

    def _base(self, det: Detection, name: str, **kw) -> Measurement:
        return Measurement(id=self._id(), detection_id=det.id, facility_class=det.class_name, name=name,
                           timestamp=self.timestamp, models=kw.pop("models", self.models),
                           scale_status=self.scale.status if self.scale else None, **kw)

    def refuse(self, det: Detection, name: str, reason: str, unit: str | None = None, method: str = "-",
               scale_dependent: bool = True) -> Measurement:
        return self._base(det, name, value=None, unit=unit, confidence=0.0, status=MS.INSUFFICIENT_DATA,
                          method=method, refusal_reason=reason, scale_dependent=scale_dependent,
                          verification_required=True, verification_reason="not measured")

    def manual(self, det: Detection, name: str, reason: str, unit: str | None = None) -> Measurement:
        return self._base(det, name, value=None, unit=unit, confidence=0.0, status=MS.REQUIRES_MANUAL_MEASUREMENT,
                          method="not observable from imagery", refusal_reason=reason,
                          verification_required=True, verification_reason=reason)

    def length(self, det: Detection, name: str, value_m: float, *, z: float | None, geom_rel: float, method: str,
               quality: float = 1.0, lower_bound: bool = False, overlay: Overlay | None = None,
               notes: list[str] | None = None, extra_abs_sigma: float = 0.0, along_view: bool = False) -> Measurement:
        """Absolute length (metres). Status depends on calibration."""
        same_plane = any(r.used and r.kind == "object" and r.detection_id == det.id for r in self.scale.references)
        rel = self.length_rel_sigma(z, geom_rel, along_view=along_view, same_plane=same_plane)
        if along_view and not same_plane:
            notes = list(notes or []) + ["length runs along the viewing direction, where monocular depth error is "
                                         "largest (validated median error ~60%)"]
        sigma = math.hypot(value_m * rel, extra_abs_sigma)
        conf = float(np.clip((1 - rel / 0.8) * quality, 0.0, 1.0))
        notes = list(notes or [])
        if lower_bound:
            notes.append("object truncated by the image border or occluded: value is a lower bound")
        if not self.scale.calibrated:
            u = self.cfg["uncertainty"]["uncalibrated_depth_scale_rel"]
            return self._base(det, name, value=None, unit="m", confidence=0.0, status=MS.REQUIRES_CALIBRATION,
                              method=method, provisional_value=round(value_m, 3), lower_bound=lower_bound,
                              uncertainty=None, overlay=overlay, verification_required=True,
                              verification_reason="no metric calibration: provide a reference measurement, marker or "
                                                  "known object",
                              notes=notes + [f"provisional value from uncalibrated monocular depth "
                                             f"(typical scale error +-{u:.0%}); not a measurement"])
        prior = self.scale.status == ScaleStatus.prior_reference_object
        return self._base(det, name, value=round(value_m, 3), unit="m", uncertainty=round(sigma, 3),
                          interval_95=[round(value_m - 1.96 * sigma, 3), round(value_m + 1.96 * sigma, 3)],
                          lower_bound=lower_bound, confidence=round(conf * (0.8 if prior else 1.0), 3),
                          status=MS.ESTIMATED, method=method, overlay=overlay, notes=notes,
                          verification_required=True,
                          verification_reason=("scale from typical object dimensions (prior)" if prior else
                                               "image-based estimate; confirm on site before any compliance decision"))

    def scale_free(self, det: Detection, name: str, value, unit: str | None, *, sigma: float | None, method: str,
                   confidence: float, overlay: Overlay | None = None, notes: list[str] | None = None,
                   lower_bound: bool = False, verification_reason: str | None = None) -> Measurement:
        iv = None
        if sigma is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            iv = [round(value - 1.96 * sigma, 3), round(value + 1.96 * sigma, 3)]
        return self._base(det, name, value=value if not isinstance(value, float) else round(value, 3), unit=unit,
                          uncertainty=None if sigma is None else round(sigma, 3), interval_95=iv,
                          lower_bound=lower_bound, confidence=round(float(np.clip(confidence, 0, 1)), 3),
                          status=MS.ESTIMATED, method=method, scale_dependent=False, overlay=overlay,
                          notes=notes or [], verification_required=True,
                          verification_reason=verification_reason or "image-based estimate; confirm on site")
