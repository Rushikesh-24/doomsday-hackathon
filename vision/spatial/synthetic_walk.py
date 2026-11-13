"""Ray-cast synthetic corridor walkthrough with world-anchored texture, exact camera poses and an
optional ArUco marker of known size. Used by tests to check SfM, scale recovery and registration against
exact ground truth (it does not replace real-footage validation)."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def _texture(X: np.ndarray, axes: tuple[int, int], k: float, seed: int) -> np.ndarray:
    a = np.floor(X[..., axes[0]] * k).astype(np.int64)
    b = np.floor(X[..., axes[1]] * k).astype(np.int64)
    h = (a * 73856093) ^ (b * 19349663) ^ seed
    rng = np.abs(h) % 997
    base = np.stack([(rng * 37) % 200 + 40, (rng * 91) % 200 + 40, (rng * 53) % 200 + 40], -1).astype(np.float32)
    fine = 25 * np.sin(X[..., axes[0]] * 41.0) * np.cos(X[..., axes[1]] * 37.0)
    return np.clip(base + fine[..., None], 0, 255)


def render_walk(n_frames: int = 40, w: int = 640, h: int = 480, hfov: float = 70.0, width: float = 2.4,
                length: float = 9.0, cam_h: float = 1.5, walk: float = 3.0, marker: dict | None = None):
    """Returns frames (list of BGR uint8), poses (list of (R, t) world->camera), K, gt dict."""
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1)
    hw = width / 2
    planes = [  # (normal, d, texture axes, bounds-check, seed): world Y up
        ((0, 1, 0), 0.0, (0, 2), 1), ((0, 1, 0), 2.8, (0, 2), 2),
        ((1, 0, 0), -hw, (2, 1), 3), ((1, 0, 0), hw, (2, 1), 4), ((0, 0, 1), length, (0, 1), 5)]
    frames, poses = [], []
    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)
        C = np.array([0.25 * math.sin(2 * math.pi * s), cam_h + 0.03 * math.sin(6 * math.pi * s), 0.5 + walk * s])
        yaw = math.radians(8 * math.sin(2 * math.pi * s))
        Rw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])  # cam->world (y up)
        flip = np.diag([1.0, -1.0, 1.0])                       # camera y is down
        R_cw = Rw @ flip                                       # camera -> world
        dw = dc @ R_cw.T
        best = np.full((h, w), np.inf)
        img = np.zeros((h, w, 3), np.float32)
        for n, d, axes, seed in planes:
            n = np.array(n, float)
            den = dw @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (d - C @ n) / den
            X = C + dw * t[..., None]
            ok = (t > 0) & (X[..., 1] >= -1e-6) & (X[..., 1] <= 2.8 + 1e-6) & (np.abs(X[..., 0]) <= hw + 1e-6) & \
                 (X[..., 2] <= length + 1e-6)
            upd = ok & (t < best)
            best[upd] = t[upd]
            tex = _texture(X, axes, 6.0, seed)
            img[upd] = tex[upd]
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        R = R_cw.T
        tt = -R @ C
        if marker:
            size, cz = marker["size_m"], length - 1e-3
            c3 = np.array([[-size / 2, 1.6 + size / 2, cz], [size / 2, 1.6 + size / 2, cz],
                           [size / 2, 1.6 - size / 2, cz], [-size / 2, 1.6 - size / 2, cz]])
            pc = c3 @ R.T + tt
            uv = (pc[:, :2] / pc[:, 2:]) * f + [w / 2, h / 2]
            mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker["dictionary"])),
                                               marker.get("id", 3), 240)
            src = np.float32([[0, 0], [239, 0], [239, 239], [0, 239]])
            Hm = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
            warped = cv2.warpPerspective(cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR), Hm, (w, h))
            inside = cv2.warpPerspective(np.full((240, 240), 255, np.uint8), Hm, (w, h)) > 0
            quiet = cv2.dilate(inside.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            bgr[quiet] = 255
            bgr[inside] = warped[inside]
        frames.append(bgr)
        poses.append((R, tt))
    centers = np.array([-R.T @ t for R, t in poses])
    return frames, poses, K, {"centers": centers, "path_length": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())}


def write_video(frames: list[np.ndarray], path: Path, fps: float = 10.0) -> Path:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path
