"""Render detections onto images."""
from __future__ import annotations

import cv2
import numpy as np

from vision.schemas import Detection
from vision.taxonomy import Taxonomy

# Large scene-level boxes (corridor/walkway) are drawn thin and dashed-looking so they don't hide objects.
_SCENE = {"corridor", "walkway"}


def _label(d: Detection) -> str:
    s = f"{d.class_name} {d.confidence:.2f}"
    if d.track_id is not None:
        s = f"#{d.track_id} " + s
    sc = d.attributes.get("visible_step_count")
    if sc is not None and sc.status.value == "estimated":
        s += f" ~{sc.value} steps"
    if d.class_name == "obstacle" and d.prompt:
        s = s.replace("obstacle", f"obstacle:{d.prompt}")
    po = d.attributes.get("potential_obstruction")
    if po is not None and po.value is True:
        s += " !BLOCKS"
    return s


def annotate(rgb: np.ndarray, detections: list[Detection], taxonomy: Taxonomy,
             show_auxiliary: bool = False, header: str | None = None) -> np.ndarray:
    """Return a BGR uint8 image with boxes + labels drawn (ready for cv2.imwrite)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(1.2, max(h, w) / 1400))
    thick = max(1, int(round(2 * scale)))
    # draw big scene boxes first so object boxes are on top
    order = sorted(detections, key=lambda d: -(d.bbox.width * d.bbox.height))
    for d in order:
        if d.category == "auxiliary" and not show_auxiliary:
            continue
        r, g, b = taxonomy.color(d.class_name)
        color = (int(b), int(g), int(r))
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox.xyxy())
        t = 1 if d.class_name in _SCENE else thick
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        text = _label(d)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, thick - 1))
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        if d.class_name in _SCENE:  # put scene labels at the bottom-left inside the box
            ty = min(y2 - 6, h - 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        cv2.putText(img, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 0, 0) if lum > 140 else (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    if header:
        cv2.rectangle(img, (0, 0), (w, int(28 * scale) + 6), (20, 20, 20), -1)
        cv2.putText(img, header, (8, int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                    (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return img
