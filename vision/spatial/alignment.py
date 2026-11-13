"""Scale recovery and coordinate alignment for SfM / splat reconstructions.

Frames (all transforms stored explicitly as 4x4 matrices, row-major, applied to column vectors):
  sfm     COLMAP world frame, arbitrary scale/orientation (cameras: x_cam = R x_sfm + t)
  local   gravity-aligned metric(-ish) frame: x_local = s * R_g * (x_sfm - c)      (+Y = up, origin = first camera)
  geo     local ENU metres from >= 3 ground control points (Umeyama similarity); never from a single GPS tag

Scale status (strict):
  relative              no metric information
  approximate_metric    priors only (typical phone height while walking, monocular metric depth) - NOT calibrated
  scale_calibrated      ArUco markers (triangulated in 3D) and/or user reference segments
  georeferenced         scale_calibrated + >= 3 GCPs with residuals reported
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ model I/O
class SfmModel:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        P = json.loads((model_dir / "poses.json").read_text())
        self.camera = P["camera"]
        self.images = {im["name"]: im for im in P["images"]}
        d = np.load(model_dir / "points.npz")
        self.ids = d["ids"]
        self.xyz = d["xyz"]
        self.rgb = d["rgb"]
        self.track_len = d["track_len"]
        self.error = d["error"]
        self._id2row = {int(i): r for r, i in enumerate(self.ids)}
        self.obs = {k[5:]: d[k] for k in d.files if k.startswith("obs::")}
        sp = model_dir / "suspect_frames.json"
        self.suspect = set(json.loads(sp.read_text())) if sp.exists() else set()

    def K(self) -> np.ndarray:
        p = self.camera["params"]
        f, cx, cy = p[0], p[1], p[2]
        if self.camera["model"] in ("PINHOLE", "OPENCV", "RADIAL", "FULL_OPENCV"):
            return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    def dist(self) -> np.ndarray:
        p = self.camera["params"]
        m = self.camera["model"]
        if m == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0])
        if m == "RADIAL":
            return np.array([p[3], p[4], 0, 0])
        if m == "OPENCV":
            return np.array(p[4:8])
        return np.zeros(4)

    def pose(self, name: str):
        im = self.images[name]
        return np.array(im["R"]), np.array(im["t"])

    def center(self, name: str) -> np.ndarray:
        R, t = self.pose(name)
        return -R.T @ t

    def points_in_frame(self, name: str):
        """(uv (n,2), xyz (n,3)) of sparse points observed in this frame."""
        o = self.obs.get(name)
        if o is None or not len(o):
            return np.zeros((0, 2)), np.zeros((0, 3))
        rows = [self._id2row.get(int(i)) for i in o[:, 2]]
        keep = [r is not None for r in rows]
        rows = np.array([r for r in rows if r is not None], dtype=int)
        return o[np.array(keep), :2], self.xyz[rows]

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """distorted pixel coords -> normalised camera coords (x/z, y/z)."""
        if not len(uv):
            return uv
        pts = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), self.K(), self.dist())
        return pts.reshape(-1, 2)


# ------------------------------------------------------------------ geometry helpers
def triangulate(obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float] | None:
    """DLT triangulation from >= 2 (R, t, xn) observations with xn = normalised image coords.
    Returns (X, max reprojection error in normalised units)."""
    if len(obs) < 2:
        return None
    A = []
    for R, t, xn in obs:
        P = np.hstack([R, t[:, None]])
        A.append(xn[0] * P[2] - P[0])
        A.append(xn[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    errs = []
    for R, t, xn in obs:
        pc = R @ X + t
        if pc[2] <= 0:
            return None
        errs.append(np.linalg.norm(pc[:2] / pc[2] - xn))
    return X, float(max(errs))


def gravity_up(model: SfmModel) -> dict:
    """Up direction in the SfM frame. Primary: mean camera 'up' (-y_cam) of a hand-held walkthrough,
    refined by a RANSAC ground plane of sparse points below the cameras when one is found."""
    Rs = [np.array(im["R"]) for im in model.images.values()]
    ups = np.array([R.T @ np.array([0, -1.0, 0]) for R in Rs])
    u0 = ups.mean(0)
    spread = float(np.degrees(np.mean(np.arccos(np.clip(ups @ (u0 / np.linalg.norm(u0)), -1, 1)))))
    u0 /= np.linalg.norm(u0)
    out = {"up": u0.tolist(), "method": "mean_camera_up", "camera_up_spread_deg": round(spread, 2),
           "confidence": "low" if spread > 20 else "medium"}
    cams = np.array([model.center(n) for n in model.images])
    h = model.xyz @ u0
    below = model.xyz[(h < np.percentile(cams @ u0, 5)) & (model.track_len >= 3)]
    if len(below) >= 60:
        rng = np.random.default_rng(0)
        scale = float(np.median(np.linalg.norm(np.diff(cams, axis=0), axis=1))) * 3 or 0.05
        best = None
        for _ in range(400):
            p = below[rng.choice(len(below), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-12:
                continue
            n /= np.linalg.norm(n)
            if n @ u0 < 0:
                n = -n
            if np.degrees(np.arccos(np.clip(n @ u0, -1, 1))) > 25:
                continue
            d = -n @ p[0]
            inl = np.abs(below @ n + d) < 0.15 * scale
            if best is None or inl.sum() > best[2].sum():
                best = (n, d, inl)
        if best is not None and best[2].sum() >= 40 and best[2].mean() > 0.35:
            Y = below[best[2]]
            c = Y.mean(0)
            n = np.linalg.svd(Y - c)[2][2]
            n = n if n @ u0 > 0 else -n
            cam_h = float(np.median(cams @ n + (-n @ c)))
            out.update(up=n.tolist(), method="ground_plane", ground_inliers=int(best[2].sum()),
                       ground_point=c.tolist(), camera_height_sfm_units=round(cam_h, 5),
                       agreement_with_camera_up_deg=round(float(np.degrees(np.arccos(np.clip(n @ u0, -1, 1)))), 2),
                       confidence="medium-high")
    return out


def _rot_to_y(up: np.ndarray) -> np.ndarray:
    y = np.array([0, 1.0, 0])
    v = np.cross(up, y)
    c = float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float(np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean())
    t = mu_d - s * R @ mu_s
    return s, R, t


# ------------------------------------------------------------------ scale sources
def aruco_scale(model: SfmModel, frames_dir: Path, dictionary: str, size_m: float) -> list[dict]:
    """Detect ArUco markers in registered frames, triangulate their corners, compare edge lengths."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    seen: dict[int, list] = {}
    for name in model.images:
        img = cv2.imread(str(frames_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        R, t = model.pose(name)
        for c, i in zip(corners, ids.ravel()):
            xn = model.undistort_points(c.reshape(4, 2))
            seen.setdefault(int(i), []).append((R, t, xn))
    out = []
    for mid, views in seen.items():
        if len(views) < 2:
            out.append({"marker_id": mid, "views": len(views), "used": False, "reason": "seen in < 2 registered frames"})
            continue
        corners3 = []
        for k in range(4):
            tri = triangulate([(R, t, xn[k]) for R, t, xn in views])
            if tri is None or tri[1] > 0.01:
                corners3 = None
                break
            corners3.append(tri[0])
        if corners3 is None:
            out.append({"marker_id": mid, "views": len(views), "used": False, "reason": "triangulation inconsistent"})
            continue
        C = np.array(corners3)
        edges = np.linalg.norm(np.roll(C, -1, 0) - C, axis=1)
        s = size_m / float(edges.mean())
        out.append({"marker_id": mid, "views": len(views), "used": True, "scale_m_per_unit": s,
                    # floor 2%: synthetic validation with exact GT showed 1.9% error at 0.7% predicted
                    "rel_sigma": float(max(edges.std() / edges.mean() / math.sqrt(4), 0.02)),
                    "edge_lengths_sfm": edges.round(6).tolist(), "corners_sfm": C.round(6).tolist()})
    return out


def segment_scale(model: SfmModel, ref: dict) -> dict:
    """User reference: {"frame": name, "points": [[x,y],[x,y]] (frame pixels), "length_m": L}.
    Endpoints are placed on a plane fitted to sparse points observed near the segment in that frame."""
    name = ref["frame"]
    if name not in model.images:
        return {"used": False, "reason": f"frame {name} is not registered in the reconstruction", **ref}
    uv, X = model.points_in_frame(name)
    a, b = np.array(ref["points"], float)
    seg = b - a
    L = np.linalg.norm(seg) + 1e-9
    tt = np.clip(((uv - a) @ seg) / L ** 2, 0, 1)
    dist = np.linalg.norm(uv - (a + tt[:, None] * seg), axis=1)
    near = dist < max(30.0, 0.25 * L)
    if near.sum() < 8:
        return {"used": False, "reason": f"only {int(near.sum())} sparse points near the segment", **ref}
    Y = X[near]
    c = Y.mean(0)
    n = np.linalg.svd(Y - c)[2][2]
    R, t = model.pose(name)
    C = -R.T @ t
    ends = []
    for p in (a, b):
        xn = model.undistort_points(p[None])[0]
        d = R.T @ np.array([xn[0], xn[1], 1.0])
        den = d @ n
        if abs(den) < 1e-9:
            return {"used": False, "reason": "segment parallel to local surface", **ref}
        ends.append(C + d * ((c - C) @ n) / den)
    l_sfm = float(np.linalg.norm(ends[1] - ends[0]))
    rms = float(np.sqrt(np.mean(((Y - c) @ n) ** 2)))
    return {"used": True, "scale_m_per_unit": float(ref["length_m"]) / l_sfm, "length_sfm": l_sfm,
            "rel_sigma": float(min(0.3, 0.02 + rms / max(l_sfm, 1e-9))), **ref}


def camera_height_prior(gravity: dict, prior_m: float = 1.40, sigma_m: float = 0.15) -> dict | None:
    h = gravity.get("camera_height_sfm_units")
    if not h or h <= 0:
        return None
    return {"used": True, "scale_m_per_unit": prior_m / h, "rel_sigma": sigma_m / prior_m,
            "note": f"assumes the phone was held ~{prior_m} m above the floor (+-{sigma_m} m)"}


def combine(items: list[dict]) -> tuple[float, float] | None:
    use = [i for i in items if i.get("used")]
    if not use:
        return None
    ls = np.log([i["scale_m_per_unit"] for i in use])
    w = np.array([1 / max(i["rel_sigma"], 1e-3) ** 2 for i in use])
    mu = float((w * ls).sum() / w.sum())
    sig = float(1 / math.sqrt(w.sum()))
    if len(use) > 1:
        sig = max(sig, float(np.sqrt((w * (ls - mu) ** 2).sum() / w.sum())))
    return math.exp(mu), sig


def align(model: SfmModel, frames_dir: Path, options: dict, mono_scale: dict | None = None) -> dict:
    grav = gravity_up(model)
    up = np.array(grav["up"])
    Rg = _rot_to_y(up)
    names = sorted(model.images)
    origin = model.center(names[0])
    sources = {"aruco": [], "segments": [], "camera_height_prior": None, "monocular_depth": mono_scale}
    for m in options.get("markers", []) or []:
        sources["aruco"] += aruco_scale(model, frames_dir, m.get("dictionary", "DICT_4X4_50"), float(m["size_m"]))
    for r in options.get("references", []) or []:
        sources["segments"].append(segment_scale(model, r))
    cal = combine(sources["aruco"] + sources["segments"])
    status, scale, rel = "relative", None, None
    if cal:
        status, (scale, rel) = "scale_calibrated", cal
    else:
        pri = []
        chp = camera_height_prior(grav, options.get("camera_height_m", 1.40))
        sources["camera_height_prior"] = chp
        if chp:
            pri.append(chp)
        if mono_scale and mono_scale.get("used"):
            pri.append(mono_scale)
        c = combine(pri)
        if c and len(pri) >= 2:
            r = max(p["scale_m_per_unit"] for p in pri) / min(p["scale_m_per_unit"] for p in pri)
            tol = math.exp(2.5 * math.hypot(*[p["rel_sigma"] for p in pri[:2]]))
            if r > tol:   # the priors contradict each other: do not average a contradiction into a number
                sources["prior_conflict"] = (f"priors disagree by x{r:.1f} (> x{tol:.1f} allowed by their "
                                             f"uncertainties); e.g. camera-height prior on stairs/slopes")
                c = None
        if c:
            status, (scale, rel) = "approximate_metric", c
    s = scale if scale else 1.0
    T_local = np.eye(4)
    T_local[:3, :3] = s * Rg
    T_local[:3, 3] = -s * Rg @ origin
    res = {"scale_status": status, "metres_per_sfm_unit": scale, "scale_rel_sigma": rel,
           "metric": status in ("scale_calibrated", "georeferenced"),
           "gravity": grav, "sources": sources,
           "frames": {"sfm": "COLMAP world (arbitrary similarity)",
                      "local": "gravity-aligned (+Y up), origin = first registered camera; units = metres if "
                               "scale_status is scale_calibrated/approximate_metric, else SfM units",
                      "geo": "local ENU metres (only when georeferenced)"},
           "T_local_from_sfm": T_local.tolist(), "georeferenced": False, "T_geo_from_local": None}
    gcps = options.get("gcps") or []
    if len(gcps) >= 3 and status == "scale_calibrated":
        src = np.array([g["local"] for g in gcps], float)
        dst = np.array([g["enu_m"] for g in gcps], float)
        sg, Rg2, tg = umeyama(src, dst)
        res_err = np.linalg.norm((sg * (Rg2 @ src.T)).T + tg - dst, axis=1)
        Tg = np.eye(4)
        Tg[:3, :3] = sg * Rg2
        Tg[:3, 3] = tg
        res.update(georeferenced=True, scale_status="georeferenced", T_geo_from_local=Tg.tolist(),
                   gcp_residuals_m=res_err.round(3).tolist(), gcp_rmse_m=round(float(np.sqrt((res_err ** 2).mean())), 3))
    elif gcps:
        res["georeference_note"] = "need >= 3 ground control points and a scale-calibrated reconstruction"
    return res


def apply(T: list | np.ndarray, X: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    X = np.atleast_2d(X)
    return X @ T[:3, :3].T + T[:3, 3]
