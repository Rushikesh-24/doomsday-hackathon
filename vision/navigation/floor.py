"""Vector floor plan of the walked corridor, measured from the scan.

For every ~0.25 m along the (smoothed) walked path, the scan points at body height are projected onto the
path's normal; the nearest dense band of points on each side is taken as the wall. The result is a clean,
drawable corridor (left / right wall polylines + a fill polygon) and a per-sample clear-width estimate that
the accessibility score uses (narrow passages for wheelchair users).

Widths are estimates from a monocular scan (approximate metric scale): they are labelled as such everywhere.
Where the scan has no wall evidence on a side, the side is marked "open" (unknown, not blocked) and drawn
with a default half-width.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SAMPLE = 0.25          # metres between corridor samples
DEFAULT_HALF = 0.9     # half-width drawn where there is no wall evidence
MIN_HALF, MAX_HALF = 0.3, 2.6
FORMAT = 2


def _local_points(job: Path) -> np.ndarray:
    from vision.navigation.places import _splat_centres
    al = json.loads((job / "alignment.json").read_text())
    T = np.array(al["T_local_from_sfm"], float)
    pts = []
    sp = job / "sfm/models/0/points.npz"
    if sp.exists():
        xyz = np.load(sp)["xyz"]
        pts.append((np.c_[xyz, np.ones(len(xyz))] @ T.T)[:, :3])
    sc = _splat_centres(job / "splat/splat_local.ply")
    if sc is not None and len(sc):
        pts.append(sc)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _smooth(P: np.ndarray, w: int = 5) -> np.ndarray:
    if len(P) < 3:
        return P
    k = np.ones(w) / w
    pad = np.pad(P, ((w // 2, w // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(P.shape[1])], 1)


def _resample(P: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < step:
        return P[[0, -1]] if len(P) > 1 else P
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, P[:, i]) for i in range(P.shape[1])], 1)


def _median_filter(a: np.ndarray, w: int = 9) -> np.ndarray:
    h = w // 2
    pad = np.pad(a, h, mode="edge")
    return np.array([np.nanmedian(pad[i:i + w]) if np.isfinite(pad[i:i + w]).any() else np.nan
                     for i in range(len(a))])


def analyse(job: Path) -> dict:
    """Corridor geometry in the local frame (x, z on the floor; +Y up)."""
    traj = json.loads((job / "trajectory.json").read_text())
    cams = np.array([c["center_local"] for c in traj["cameras"] if c.get("center_local")], float)
    metric = traj.get("scale_status") in ("metric", "approximate_metric")
    if len(cams) < 2:
        raise ValueError("not enough registered cameras for a floor plan")
    step_med = float(np.median(np.linalg.norm(np.diff(cams, axis=0), axis=1))) or 1e-3
    unit = 1.0 if metric else step_med / 0.6          # non-metric scans: express metres in SfM units
    eye = float(np.median(cams[:, 1]))
    P = _local_points(job)
    # body height band: knee to just above the eye (walls, doors, furniture; not floor or ceiling)
    band = P[(P[:, 1] > eye - 1.1 * unit) & (P[:, 1] < eye + 0.35 * unit)] if len(P) else P
    XZ = band[:, [0, 2]]
    path = _resample(_smooth(cams[:, [0, 2]]), SAMPLE * unit)
    n = len(path)
    tan = np.gradient(path, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nor = np.stack([-tan[:, 1], tan[:, 0]], 1)          # left of travel
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    half_strip = 0.35 * unit
    for i in range(n):
        d = XZ - path[i]
        along = d @ tan[i]
        lat = d @ nor[i]
        m = (np.abs(along) < half_strip) & (np.abs(lat) > MIN_HALF * unit) & (np.abs(lat) < MAX_HALF * unit)
        if m.sum() < 4:
            continue
        lat = lat[m]
        for side, sel, arr in ((1, lat > 0, left), (-1, lat < 0, right)):
            v = np.sort(np.abs(lat[sel]))
            if len(v) >= 4:
                # nearest distance where at least 4 points pile up within 12 cm = a wall / furniture face
                bins = np.arange(MIN_HALF * unit, MAX_HALF * unit + 0.12 * unit, 0.06 * unit)
                h, edges = np.histogram(v, bins=bins)
                cum = np.convolve(h, [1, 1], mode="same")
                hit = np.nonzero(cum >= 4)[0]
                if len(hit):
                    arr[i] = edges[hit[0]]
    left_f, right_f = _median_filter(left), _median_filter(right)
    open_l, open_r = ~np.isfinite(left_f), ~np.isfinite(right_f)
    lh = np.where(open_l, DEFAULT_HALF * unit, left_f)
    rh = np.where(open_r, DEFAULT_HALF * unit, right_f)
    width = left_f + right_f                               # nan where either side is unknown
    L = path + nor * lh[:, None]
    R = path - nor * rh[:, None]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] / unit
    known = np.isfinite(width)
    w_m = width / unit
    return {
        "format": FORMAT, "units": traj["units"], "metric": metric, "unit_per_m": unit,
        "centre": path.round(3).tolist(), "left": L.round(3).tolist(), "right": R.round(3).tolist(),
        "open_left": open_l.tolist(), "open_right": open_r.tolist(),
        "width_m": [None if not np.isfinite(x) else round(float(x), 2) for x in w_m],
        "distance_m": s.round(2).tolist(),
        "summary": {
            "length_m": round(float(s[-1]), 1) if n else 0.0,
            "measured_fraction": round(float(known.mean()), 2) if n else 0.0,
            "median_width_m": round(float(np.nanmedian(w_m)), 2) if known.any() else None,
            "min_width_m": round(float(np.nanmin(w_m)), 2) if known.any() else None,
            "points_used": int(len(XZ)),
        },
        "note": "Clear width estimated from the scan's points at body height (approximate scale).",
    }


def ensure_floor(job: Path) -> dict:
    """Cached analysis; recomputed when the trajectory or alignment changes."""
    out = job / "plan" / "floor.json"
    src = [job / "trajectory.json", job / "alignment.json"]
    newest = max(p.stat().st_mtime for p in src if p.exists())
    if out.exists() and out.stat().st_mtime >= newest:
        try:
            f = json.loads(out.read_text())
            if f.get("format") == FORMAT:
                return f
        except Exception:
            pass
    f = analyse(job)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(f))
    return f


def narrow_stretches(floor: dict, limit_m: float = 0.9, min_len_m: float = 0.5) -> list[dict]:
    """Runs of samples whose measured clear width is below `limit_m` (wheelchair needs ~0.9 m)."""
    w, s, c = floor["width_m"], floor["distance_m"], floor["centre"]
    out, i = [], 0
    while i < len(w):
        if w[i] is not None and w[i] < limit_m:
            j = i
            while j + 1 < len(w) and w[j + 1] is not None and w[j + 1] < limit_m:
                j += 1
            if s[j] - s[i] >= min_len_m or j == i and min_len_m <= 0:
                seg = [x for x in w[i:j + 1] if x is not None]
                mid = (i + j) // 2
                out.append({"from_m": s[i], "to_m": s[j], "min_width_m": min(seg), "at": c[mid],
                            "points": c[i:j + 1]})
            i = j + 1
        else:
            i += 1
    return out


__all__ = ["analyse", "ensure_floor", "narrow_stretches"]
