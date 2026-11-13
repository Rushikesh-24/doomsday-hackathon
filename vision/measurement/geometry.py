"""3D geometry helpers: point maps, normals, robust planes, ray casting, gravity direction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision.measurement.camera import Intrinsics


def point_map(depth: np.ndarray, K: Intrinsics) -> np.ndarray:
    """(H,W,3) camera-frame points (x right, y down, z forward) from a z-depth map."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    return np.stack([(u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z], axis=-1)


def normal_map(P: np.ndarray, smooth: int = 7) -> np.ndarray:
    """Per-pixel surface normals (unit, oriented towards the camera) from a point map."""
    Ps = cv2.GaussianBlur(P, (smooth, smooth), 0) if smooth > 1 else P
    dx = np.zeros_like(Ps)
    dy = np.zeros_like(Ps)
    dx[:, 1:-1] = Ps[:, 2:] - Ps[:, :-2]
    dy[1:-1] = Ps[2:] - Ps[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    flip = (n * Ps).sum(-1) > 0
    n[flip] *= -1
    return n


def ray(u: float, v: float, K: Intrinsics) -> np.ndarray:
    r = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
    return r / np.linalg.norm(r)


@dataclass
class Plane:
    normal: np.ndarray        # unit, oriented towards the camera
    d: float                  # n.x + d = 0
    inliers: np.ndarray       # bool mask over the input points
    inlier_ratio: float
    rms: float                # residual RMS of inliers (same unit as points)
    normal_sigma_deg: float   # approx. 1-sigma angular uncertainty of the normal
    extent: float             # RMS spread of inliers in-plane

    def distance(self, X: np.ndarray) -> np.ndarray:
        return X @ self.normal + self.d

    def intersect_ray(self, r: np.ndarray) -> np.ndarray | None:
        den = float(self.normal @ r)
        if abs(den) < 1e-6:
            return None
        t = -self.d / den
        return r * t if t > 0 else None


def fit_plane(X: np.ndarray, rel_thresh: float = 0.015, iters: int = 300, seed: int = 0,
              abs_thresh: float | None = None) -> Plane | None:
    """RANSAC plane + SVD refinement. Threshold scales with distance (monocular depth error grows with z)."""
    X = np.asarray(X, dtype=np.float64)
    n_pts = len(X)
    if n_pts < 30:
        return None
    rng = np.random.default_rng(seed)
    thr = abs_thresh if abs_thresh is not None else rel_thresh * float(np.median(X[:, 2]))
    sub = X if n_pts <= 20000 else X[rng.choice(n_pts, 20000, replace=False)]
    best, best_cnt = None, -1
    for _ in range(iters):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ p[0]
        cnt = int((np.abs(sub @ n + d) < thr).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, d), cnt
    if best is None:
        return None
    n, d = best
    inl = np.abs(X @ n + d) < thr
    for _ in range(2):  # refine with SVD on inliers
        Y = X[inl]
        if len(Y) < 10:
            return None
        c = Y.mean(0)
        _, s, vt = np.linalg.svd(Y - c, full_matrices=False)
        n = vt[2]
        d = -n @ c
        inl = np.abs(X @ n + d) < thr
    Y = X[inl]
    if n @ Y.mean(0) > 0:
        n, d = -n, -d
    res = Y @ n + d
    rms = float(np.sqrt(np.mean(res ** 2)))
    c = Y.mean(0)
    ext = float(np.sqrt(np.mean(np.sum((Y - c) ** 2, 1) - res ** 2)) + 1e-9)
    # first-order normal uncertainty: residual / (extent * sqrt(N_eff)); N_eff capped (correlated depth errors)
    sig = math.degrees(math.atan(rms / (ext * math.sqrt(min(len(Y), 200)))))
    return Plane(n, float(d), inl, float(inl.mean()), rms, sig, ext)


def orient_up(n: np.ndarray) -> np.ndarray:
    """Camera y points down, so 'up' has negative y."""
    return n if n[1] < 0 else -n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return math.degrees(math.acos(min(1.0, c)))


def vertical_vanishing_up(gray: np.ndarray, K: Intrinsics, max_tilt_deg: float = 25.0):
    """Gravity 'up' direction from the vertical vanishing point of near-vertical line segments
    (building edges, door frames). Returns (up_vector or None, n_inlier_lines, sigma_deg)."""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50, minLineLength=int(0.06 * h), maxLineGap=6)
    if lines is None:
        return None, 0, None
    L = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))  # 0 = vertical
        if ang <= max_tilt_deg:
            l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
            nrm = np.linalg.norm(l[:2])
            if nrm > 0:
                L.append(l / nrm)
    if len(L) < 6:
        return None, len(L), None
    L = np.array(L)
    Kinv = np.linalg.inv(K.K)
    rng = np.random.default_rng(0)
    best, best_inl = None, None
    nrm_all = (K.K.T @ L.T).T
    nrm_all /= np.linalg.norm(nrm_all, axis=1, keepdims=True)
    for _ in range(300):
        i, j = rng.choice(len(L), 2, replace=False)
        v = np.cross(L[i], L[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        d = Kinv @ v
        d /= np.linalg.norm(d)
        # angular consistency: each line's back-projected plane normal must be orthogonal to d
        err = np.degrees(np.arcsin(np.clip(np.abs(nrm_all @ d), 0, 1)))
        inl = err < 1.5
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = d, inl
    if best is None or best_inl.sum() < 6:
        return None, int(0 if best_inl is None else best_inl.sum()), None
    nrm = (K.K.T @ L[best_inl].T).T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(nrm)
    d = vt[2]
    up = orient_up(d)
    err = np.degrees(np.arcsin(np.clip(np.abs(nrm @ d), 0, 1)))
    sigma = float(np.sqrt(np.mean(err ** 2)) / math.sqrt(best_inl.sum()) + 0.5)
    return up, int(best_inl.sum()), sigma


def robust_extent(v: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float, float]:
    a, b = np.percentile(v, [lo, hi])
    return float(b - a), float(a), float(b)


def touching_sides(mask: np.ndarray, margin: int = 3) -> list[str]:
    h, w = mask.shape
    s = []
    if mask[:margin].any():
        s.append("top")
    if mask[-margin:].any():
        s.append("bottom")
    if mask[:, :margin].any():
        s.append("left")
    if mask[:, -margin:].any():
        s.append("right")
    return s


def project(X: np.ndarray, K: Intrinsics) -> np.ndarray:
    X = np.atleast_2d(X)
    return np.stack([X[:, 0] / X[:, 2] * K.fx + K.cx, X[:, 1] / X[:, 2] * K.fy + K.cy], 1)
