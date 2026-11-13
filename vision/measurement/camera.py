"""Camera intrinsics: user calibration > EXIF 35mm-equivalent focal > depth-model FOV > prior."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

DIAG_35MM = math.hypot(36.0, 24.0)  # 43.27 mm, 35mm-equivalence is defined on the diagonal


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str
    focal_rel_sigma: float
    dist: list[float] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics for an image resized by factor s (new = old * s)."""
        return Intrinsics(self.fx * s, self.fy * s, self.cx * s, self.cy * s, round(self.width * s),
                          round(self.height * s), self.source, self.focal_rel_sigma, self.dist, list(self.notes))

    def with_focal_factor(self, f: float) -> "Intrinsics":
        return Intrinsics(self.fx * f, self.fy * f, self.cx, self.cy, self.width, self.height, self.source,
                          self.focal_rel_sigma, self.dist, self.notes)


def exif_focal_35mm(data: bytes) -> float | None:
    try:
        img = Image.open(io.BytesIO(data))
        exif = img.getexif()
        sub = exif.get_ifd(0x8769)
        f35 = sub.get(0xA405) or exif.get(0xA405)
        if f35:
            f35 = float(f35)
            return f35 if 8 <= f35 <= 600 else None
    except Exception:
        return None
    return None


def from_focal_35mm(f35: float, w: int, h: int, source: str, rel_sigma: float) -> Intrinsics:
    f = f35 * math.hypot(w, h) / DIAG_35MM
    return Intrinsics(f, f, w / 2, h / 2, w, h, source, rel_sigma)


def from_hfov(hfov_deg: float, w: int, h: int, source: str, rel_sigma: float) -> Intrinsics:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return Intrinsics(f, f, w / 2, h / 2, w, h, source, rel_sigma)


def resolve_intrinsics(w: int, h: int, user: dict | None, exif_f35: float | None, model_fov: float | None,
                       prior_f35: float, prior_sigma: float) -> Intrinsics:
    """`user` may contain fx,fy,cx,cy[,dist] (pixels of the ORIGINAL image), or hfov_deg, or focal_35mm."""
    if user:
        if "fx" in user:
            fx = float(user["fx"])
            K = Intrinsics(fx, float(user.get("fy", fx)), float(user.get("cx", w / 2)), float(user.get("cy", h / 2)),
                           w, h, "user_intrinsics", float(user.get("focal_rel_sigma", 0.01)), user.get("dist"))
            return K
        if "hfov_deg" in user:
            return from_hfov(float(user["hfov_deg"]), w, h, "user_fov", float(user.get("focal_rel_sigma", 0.03)))
        if "focal_35mm" in user:
            return from_focal_35mm(float(user["focal_35mm"]), w, h, "user_focal_35mm",
                                   float(user.get("focal_rel_sigma", 0.03)))
    if exif_f35:
        # EXIF 35mm-equivalent focal is usually accurate to a few % (crop/rounding)
        return from_focal_35mm(exif_f35, w, h, "exif_focal_35mm", 0.04)
    if model_fov:
        return from_hfov(model_fov, w, h, "depth_model_fov", 0.08)
    K = from_focal_35mm(prior_f35, w, h, "prior", prior_sigma)
    K.notes.append(f"no calibration/EXIF: assumed {prior_f35} mm-equivalent focal (+-{prior_sigma:.0%})")
    return K


def undistort(rgb: np.ndarray, K: Intrinsics) -> np.ndarray:
    if not K.dist:
        return rgb
    d = np.asarray(K.dist, dtype=float)
    return cv2.undistort(rgb, K.K, d, None, K.K)
