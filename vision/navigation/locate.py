"""Where am I, and what is in front of me?" from one phone photo, for the mobile app's camera guide.

Localisation (keyframe-level, not 6-DoF):
  1. DINOv2 global descriptors shortlist the most similar keyframes of the scan (vision.ailab.models).
  2. SIFT + ratio test + RANSAC fundamental matrix re-ranks the shortlist by geometrically consistent inliers.
     Look-alike places (two similar staircases) score high on (1) but fail (2), so a wrong match is
     reported as "low" confidence instead of a confident wrong position.
  3. Heading = the matched keyframe's viewing direction, corrected by the horizontal shift of the inliers
     (if the scene appears shifted right in the photo, the phone is turned left of the keyframe).
Scene: the facility detector runs on the same photo; detections are turned into short spoken hints with a
direction (left / ahead / right), hazards first.
"""
from __future__ import annotations

import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_sift_cache: dict[str, tuple[float, list]] = {}
_lock = threading.Lock()
MAX_SIDE = 800
HAZARDS = {"staircase": "Stairs", "obstacle": "Obstacle", "ramp": "Ramp"}
USEFUL = {"door": "Door", "entrance_exit": "Exit", "elevator": "Lift", "handrail": "Handrail",
          "signboard": "Sign", "tactile_paving": "Tactile paving", "accessibility_symbol": "Accessibility sign",
          "corridor": "Corridor", "walkway": "Walkway"}


def _gray(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("L")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    return np.asarray(im)


def _sift():
    import cv2
    return cv2.SIFT_create(nfeatures=1500)


def _keyframe_features(job: Path, frame: str):
    import cv2
    key = str(job / "frames" / frame)
    with _lock:
        hit = _sift_cache.get(key)
    mt = (job / "frames" / frame).stat().st_mtime
    if hit and hit[0] == mt:
        return hit[1]
    g = _gray(Image.open(job / "frames" / frame))
    kp, des = _sift().detectAndCompute(g, None)
    val = [np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des, g.shape[::-1]]
    with _lock:
        if len(_sift_cache) > 4000:
            _sift_cache.clear()
        _sift_cache[key] = (mt, val)
    return val


def _verify(q, k) -> tuple[int, float]:
    """Geometric inliers between query and keyframe features, and the inliers' mean horizontal shift
    (query x - keyframe x, as a fraction of image width)."""
    import cv2
    qp, qd, qsize = q
    kp, kd, _ = k
    if qd is None or kd is None or len(qd) < 8 or len(kd) < 8:
        return 0, 0.0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd, kd, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return len(good) // 3, 0.0
    a = np.float32([qp[g.queryIdx] for g in good])
    b = np.float32([kp[g.trainIdx] for g in good])
    F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    if mask is None:
        return 0, 0.0
    inl = mask.ravel().astype(bool)
    shift = float(np.median(a[inl, 0] - b[inl, 0]) / qsize[0]) if inl.any() else 0.0
    return int(inl.sum()), shift


def _yaw(v) -> float:
    return math.atan2(v[2], v[0])        # angle on the plan (x right, z down), 0 = +x


def _side(cx: float) -> str:
    return "on your left" if cx < 0.36 else "on your right" if cx > 0.64 else "ahead"


def describe(im: Image.Image) -> dict:
    """Facility detections -> short hints, hazards first. Never raises: an empty scene is a valid answer."""
    from vision.inference.pipeline import get_pipeline
    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    t = time.perf_counter()
    res = get_pipeline().analyze_array(rgb, "camera.jpg", save_dir=None, save_annotated=False, save_json=False)
    seen, hints = [], []
    for d in sorted(res.detections, key=lambda d: -d.confidence):
        name = HAZARDS.get(d.class_name) or USEFUL.get(d.class_name)
        if not name:
            continue
        b = d.bbox
        cx = (b.x1n + b.x2n) / 2
        area = (b.x2n - b.x1n) * (b.y2n - b.y1n)
        item = {"class_name": d.class_name, "label": name, "side": _side(cx), "confidence": round(d.confidence, 2),
                "near": area > 0.18 or b.y2n > 0.92, "hazard": d.class_name in HAZARDS,
                "bbox": [round(b.x1n, 3), round(b.y1n, 3), round(b.x2n, 3), round(b.y2n, 3)]}
        if any(s["class_name"] == item["class_name"] and s["side"] == item["side"] for s in seen):
            continue
        seen.append(item)
    seen.sort(key=lambda s: (not s["hazard"], not s["near"], -s["confidence"]))
    for s in seen[:4]:
        lead = "Careful: " if s["hazard"] and s["near"] else ""
        hints.append(f"{lead}{s['label']} {s['side']}{' close by' if s['near'] and not s['hazard'] else ''}.")
    return {"objects": seen, "hints": hints, "model": res.model.key if hasattr(res.model, "key") else None,
            "ms": round((time.perf_counter() - t) * 1000)}


def locate(job: Path, data: bytes, with_scene: bool = True, shortlist: int = 8) -> dict:
    from vision.ailab.models import keyframe_index, _embed
    t = time.perf_counter()
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ValueError(f"not an image: {e}")
    im = ImageOps.exif_transpose(im).convert("RGB")
    traj, E = keyframe_index(job)
    if not len(E):
        raise ValueError("this map has no registered keyframes")
    reg = [i for i, c in enumerate(traj) if c.get("center_local")]       # graph node camK = K-th registered camera
    node_of = {i: f"cam{k}" for k, i in enumerate(reg)}
    sim = E @ _embed([im])[0]
    cand = [int(i) for i in np.argsort(-sim) if int(i) in node_of][:shortlist]
    q = _keyframe_query(im)
    scored = []
    for i in cand:
        inl, shift = _verify(q, _keyframe_features(job, traj[i]["frame"]))
        scored.append((inl, float(sim[i]), i, shift))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    inl, s, best, shift = scored[0]
    # the strongest rival that is somewhere else (neighbouring keyframes along the walk are not rivals)
    bp = np.array(traj[best]["center_local"])
    near = 8 * float(np.median(np.linalg.norm(np.diff([traj[i]["center_local"] for i in reg], axis=0), axis=1)))
    rival = max((n for n, _, i, _ in scored[1:]
                 if np.linalg.norm(np.array(traj[i]["center_local"]) - bp) > near), default=0)
    conf = ("high" if inl >= 60 and inl >= 1.5 * max(rival, 1) else
            "medium" if inl >= 25 else "low")
    cam = traj[best]
    fwd = cam.get("forward_local") or [1, 0, 0]
    # horizontal shift -> yaw offset, assuming a ~65 deg horizontal field of view
    yaw = _yaw(fwd) - shift * math.radians(65)
    out = {
        "node": node_of[best], "position": cam["center_local"], "heading": round(yaw, 4),
        "confidence": conf, "inliers": inl, "similarity": round(s, 4), "frame": cam["frame"],
        "candidates": [{"node": node_of[i], "inliers": n, "similarity": round(v, 4)} for n, v, i, _ in scored[:4]],
        "method": "DINOv2 shortlist + SIFT/RANSAC verification over the scan's keyframes (keyframe-level)",
    }
    if with_scene:
        try:
            out["scene"] = describe(im)
        except Exception as e:                         # detector unavailable: localisation still answers
            out["scene"] = {"objects": [], "hints": [], "error": str(e)}
    out["ms"] = round((time.perf_counter() - t) * 1000)
    return out


def _keyframe_query(im: Image.Image):
    g = _gray(im)
    kp, des = _sift().detectAndCompute(g, None)
    return [np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des, g.shape[::-1]]


def nearest_place(job: Path, pos: list[float]) -> dict | None:
    from vision.navigation.places import list_places
    ps = list_places(job)
    if not ps:
        return None
    p = np.array(pos)
    d = [float(np.linalg.norm(np.array(x["position"]) - p)) for x in ps]
    k = int(np.argmin(d))
    return {"name": ps[k]["name"], "id": ps[k]["id"], "distance": round(d[k], 2)}


__all__ = ["locate", "describe", "nearest_place"]
