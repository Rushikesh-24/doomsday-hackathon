"""Annotated measurement image: object boundaries, measurement lines with end ticks, reference
segments/markers, step nosing points and value labels with status."""
from __future__ import annotations

import cv2
import numpy as np

from vision.measurement.schemas import MeasurementReport, MeasurementStatus as MS

COL = {"ESTIMATED": (60, 200, 60), "VERIFIED": (255, 200, 0), "REQUIRES_CALIBRATION": (0, 165, 255),
       "INSUFFICIENT_DATA": (80, 80, 255), "REQUIRES_MANUAL_MEASUREMENT": (180, 180, 180)}
SHORT = {"doorway_width": "W", "clear_width_estimate": "clear", "door_height": "H", "clear_width": "W",
         "ramp_length": "L", "running_slope_percent": "slope", "riser_height": "riser", "tread_depth": "tread",
         "stair_width": "W", "step_count": "steps", "min_available_clearance": "free"}


def _label(m) -> str:
    n = SHORT.get(m.name, m.name)
    if m.status == MS.INSUFFICIENT_DATA:
        return f"{n}: refused"
    if m.status == MS.REQUIRES_CALIBRATION:
        return f"{n}: ? (needs calibration)"
    if m.value is None:
        return f"{n}: -"
    if isinstance(m.value, float):
        u = m.unit or ""
        if u == "m":
            s = f"{m.value:.2f}" + (f"+-{m.uncertainty:.2f}" if m.uncertainty else "") + " m"
        else:
            s = f"{m.value:.1f}" + (f"+-{m.uncertainty:.1f}" if m.uncertainty else "") + f" {u}"
    else:
        s = f"{m.value}{' ' + m.unit if m.unit else ''}"
    return f"{n}: {'>=' if m.lower_bound else ''}{s}"


def _text(img, txt, org, color, scale):
    (tw, th), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(img.shape[1] - tw - 4, x))
    y = max(th + 4, min(img.shape[0] - 4, y))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + b), (20, 20, 20), -1)
    cv2.putText(img, txt, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _tick_line(img, ln, color, t):
    x1, y1, x2, y2 = (int(round(v)) for v in ln)
    cv2.line(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
    d = np.array([x2 - x1, y2 - y1], float)
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * 6 * t
    for x, y in ((x1, y1), (x2, y2)):
        cv2.line(img, (int(x - n[0]), int(y - n[1])), (int(x + n[0]), int(y + n[1])), color, t, cv2.LINE_AA)


def render(rgb: np.ndarray, report: MeasurementReport, masks: dict[str, np.ndarray], calibration: dict | None) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]
    sc = max(0.45, min(1.0, max(H, W) / 1600))
    t = max(1, int(round(2 * sc)))
    # object boundaries
    for fac in report.facilities:
        m = masks.get(fac.detection_id)
        if m is not None:
            mm = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (255, 0, 255), t, cv2.LINE_AA)
        x1, y1 = fac.bbox[0], fac.bbox[1]
        _text(img, f"{fac.class_name} {fac.detection_id[-3:]}", (x1, y1 - 4), (255, 0, 255), 0.5 * sc + 0.1)
    # measurement lines / points
    for fac in report.facilities:
        y_off = 0
        for m in fac.measurements:
            col = COL.get(m.status.value, (255, 255, 255))
            ov = m.overlay
            if ov:
                for ln in ov.lines:
                    _tick_line(img, ln, col, t)
                for p in ov.points:
                    cv2.circle(img, (int(p[0]), int(p[1])), max(2, t + 1), (0, 255, 255), -1, cv2.LINE_AA)
            if m.name in SHORT:
                anchor = (ov.lines[0][0], ov.lines[0][1] - 6) if ov and ov.lines else \
                    (fac.bbox[0], fac.bbox[3] + 16 * sc + y_off)
                if not (ov and ov.lines):
                    y_off += 18 * sc
                _text(img, _label(m), anchor, col, 0.48 * sc + 0.08)
    # references
    for ref in (calibration or {}).get("references", []) or []:
        if ref.get("type") == "segment":
            (x1, y1), (x2, y2) = ref["points"]
            _tick_line(img, [x1, y1, x2, y2], (255, 255, 0), t + 1)
            _text(img, f"REF {ref['length_m']} m", (x1, y1 - 8), (255, 255, 0), 0.5 * sc + 0.1)
    hdr = (f"scale: {report.scale.status.value} (x{report.scale.scale_factor:.3f} +-{report.scale.rel_sigma:.0%})  "
           f"camera: {report.camera.source} hfov {report.camera.hfov_deg:.0f}deg  "
           f"gravity: {report.scene.get('gravity_source')}")
    cv2.rectangle(img, (0, 0), (W, int(26 * sc) + 8), (20, 20, 20), -1)
    cv2.putText(img, hdr, (6, int(20 * sc) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc + 0.05, (255, 255, 255), 1, cv2.LINE_AA)
    # legend
    y = H - 8
    for k, c in COL.items():
        (tw, _), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sc + 0.05, 1)
        _text(img, k, (W - tw - 10, y), c, 0.4 * sc + 0.05)
        y -= int(18 * sc) + 4
    return img
