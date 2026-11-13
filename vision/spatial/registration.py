"""Facility-to-reconstruction registration.

Per observation (track detection in a registered keyframe):
  support  = sparse SfM points observed in that frame whose 2D position lies inside the (shrunk) box,
             restricted to the dominant depth cluster (rejects background seen through the box)
  centre   = box-centre viewing ray intersected with a plane fitted to the support points (fallback:
             median support depth along the ray)
Per 2D track: positions from observations are combined (median) and checked against multi-view ray
triangulation of the box centres (independent geometric consistency check).
Tracks of the same class whose 3D boxes overlap / centres are close are merged into one persistent
facility. Observations without enough 3D support keep position=None and a reason - no invented coordinates.
"""
from __future__ import annotations

import numpy as np

from vision.spatial.alignment import SfmModel, apply, triangulate

MIN_SUPPORT = 6
# extended facilities: different views see different parts, so box centres are NOT the same 3D point and
# multi-view triangulation of centres is not a valid consistency check for them
EXTENDED = {"staircase", "handrail", "walkway", "corridor", "ramp", "tactile_paving"}


def _support(model: SfmModel, name: str, bbox, shrink: float = 0.1):
    uv, X = model.points_in_frame(name)
    if not len(uv):
        return None, "frame has no triangulated points"
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    inside = (uv[:, 0] > x1 + shrink * w) & (uv[:, 0] < x2 - shrink * w) & (uv[:, 1] > y1 + shrink * h) & \
             (uv[:, 1] < y2 - shrink * h)
    if inside.sum() < MIN_SUPPORT:
        return None, f"only {int(inside.sum())} sparse points inside the box"
    Xi = X[inside]
    R, t = model.pose(name)
    z = (Xi @ R.T + t)[:, 2]
    # dominant depth cluster: points within +-30% of the 35th percentile depth (front surface)
    z0 = np.percentile(z, 35)
    keep = np.abs(z - z0) < 0.3 * z0
    if keep.sum() < MIN_SUPPORT:
        return None, "box points spread over very different depths (no consistent surface)"
    return Xi[keep], None


def _ray_on_support(R, t, xn, sup: np.ndarray) -> np.ndarray:
    """Where the box-centre viewing ray meets the facility surface spanned by its support points
    (least-squares plane); falls back to the median support depth along the ray."""
    C = -R.T @ t
    d = R.T @ np.array([xn[0], xn[1], 1.0])
    d /= np.linalg.norm(d)
    c = sup.mean(0)
    _, sv, vt = np.linalg.svd(sup - c)
    n = vt[2]
    planar = sv[2] < 0.35 * sv[1]                 # thin along the normal -> a usable surface
    den = float(d @ n)
    if planar and abs(den) > 0.15:
        lam = float((c - C) @ n) / den
        if lam > 0:
            return C + lam * d
    lam = float(np.median((sup - C) @ d))
    return C + lam * d


def register(model: SfmModel, tracks: list[dict], T_local: list, scale_status: str, video_id: str,
             merge_dist_rel: float = 0.04) -> dict:
    cams = np.array([model.center(n) for n in model.images])
    scene_extent = float(np.linalg.norm(cams.max(0) - cams.min(0)) + np.linalg.norm(np.ptp(model.xyz, 0)) * 0.1) or 1.0
    per_track = []
    for tr in tracks:
        if not tr.get("confirmed"):
            continue
        obs_out, centres, supports, rays = [], [], [], []
        for o in tr["observations"]:
            rec = {"frame": o["frame"], "detection_id": o["detection_id"], "t": o["t"], "bbox": o["bbox"],
                   "confidence": o["confidence"]}
            if o["frame"] not in model.images:
                rec.update(position_sfm=None, reason="frame not registered in the primary reconstruction")
                obs_out.append(rec)
                continue
            if o["frame"] in model.suspect:
                rec.update(position_sfm=None, reason="frame pose temporally inconsistent (possible aliasing)")
                obs_out.append(rec)
                continue
            R, t = model.pose(o["frame"])
            x1, y1, x2, y2 = o["bbox"]
            xn = model.undistort_points(np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
            rays.append((R, t, xn))
            sup, why = _support(model, o["frame"], o["bbox"])
            if sup is None:
                rec.update(position_sfm=None, reason=why)
            else:
                c = _ray_on_support(R, t, xn, sup)
                centres.append(c)
                supports.append(sup)
                rec.update(position_sfm=c.round(6).tolist(), support_points=len(sup))
            obs_out.append(rec)
        entry = {"track_id": tr["track_id"], "class_name": tr["class_name"], "observations": obs_out,
                 "track_confidence": tr.get("track_confidence"), "n_observations": len(obs_out)}
        tri = triangulate(rays) if len(rays) >= 2 else None
        baseline = 0.0
        if len(rays) >= 2:
            cs = np.array([-R.T @ t for R, t, _ in rays])
            baseline = float(np.linalg.norm(cs.max(0) - cs.min(0)))
        if centres:
            C = np.array(centres)
            pos = np.median(C, 0)
            allp = np.vstack(supports)
            lo, hi = np.percentile(allp, 10, 0), np.percentile(allp, 90, 0)
            obj_diag = float(np.linalg.norm(hi - lo)) or scene_extent * 0.01
            spread = float(np.median(np.linalg.norm(C - pos, axis=1))) / obj_diag   # relative to the object
            tri_dev = None
            compact = tr["class_name"] not in EXTENDED
            if compact and tri is not None and tri[1] < 0.05 and baseline > 0.02 * scene_extent:
                tri_dev = float(np.linalg.norm(tri[0] - pos)) / max(obj_diag, 1e-9)
            conf = (min(1.0, len(centres) / 4) * min(1.0, len(allp) / 60) * float(np.exp(-spread / 0.5))
                    * (1.0 if tri_dev is None else float(np.exp(-tri_dev / 1.0))))
            entry.update(position_sfm=pos.round(6).tolist(), bbox_sfm=[lo.round(6).tolist(), hi.round(6).tolist()],
                         support_points=int(len(allp)), registered_observations=len(centres),
                         observation_spread_rel_to_object=round(spread, 4), object_extent_sfm=round(obj_diag, 5),
                         ray_triangulation_deviation_rel_to_object=None if tri_dev is None else round(tri_dev, 4),
                         triangulation_check="applied" if tri_dev is not None else
                         ("not applicable (extended facility)" if not compact else "insufficient baseline"),
                         registration_confidence=round(conf, 3), method="sparse_points_in_box (+ray triangulation check)")
        elif tr["class_name"] not in EXTENDED and tri is not None and tri[1] < 0.02 and baseline > 0.05 * scene_extent:
            entry.update(position_sfm=tri[0].round(6).tolist(), bbox_sfm=None, support_points=0,
                         registered_observations=len(rays), registration_confidence=0.2,
                         method="multi-view ray triangulation of box centres only (no surface support)")
        else:
            entry.update(position_sfm=None, bbox_sfm=None, registration_confidence=0.0,
                         reason="no reliable 3D position: too few sparse points in the boxes and insufficient "
                                "baseline for triangulation")
        per_track.append(entry)
    # merge tracks into persistent facilities (same class, close in 3D)
    facilities = []
    for e in sorted(per_track, key=lambda e: -(e.get("registration_confidence") or 0)):
        target = None
        if e.get("position_sfm") is not None:
            p = np.array(e["position_sfm"])
            for f in facilities:
                if f["class_name"] != e["class_name"] or f.get("position_sfm") is None:
                    continue
                d = np.linalg.norm(np.array(f["position_sfm"]) - p) / scene_extent
                overlap = False
                if f.get("bbox_sfm") and e.get("bbox_sfm"):
                    a0, a1 = np.array(f["bbox_sfm"][0]), np.array(f["bbox_sfm"][1])
                    b0, b1 = np.array(e["bbox_sfm"][0]), np.array(e["bbox_sfm"][1])
                    overlap = bool(np.all(np.minimum(a1, b1) >= np.maximum(a0, b0)))
                if d < merge_dist_rel or overlap:
                    target = f
                    break
        if target is None:
            facilities.append({**e, "track_ids": [e["track_id"]], "merged_from": 1})
        else:
            target["track_ids"].append(e["track_id"])
            target["merged_from"] += 1
            target["observations"] += e["observations"]
            target["n_observations"] += e["n_observations"]
            if e.get("support_points"):
                target["support_points"] = (target.get("support_points") or 0) + e["support_points"]
    out = []
    for i, f in enumerate(sorted(facilities, key=lambda f: (f["class_name"], -(f.get("registration_confidence") or 0)))):
        f = dict(f)
        f.pop("track_id", None)
        f["facility_id"] = f"{video_id}-fac{i:03d}"
        f["detection_ids"] = [o["detection_id"] for o in f["observations"]]
        f["frames"] = sorted({o["frame"] for o in f["observations"]})
        if f.get("position_sfm") is not None:
            f["position_local"] = apply(T_local, np.array(f["position_sfm"]))[0].round(4).tolist()
            if f.get("bbox_sfm"):
                corners = np.array([[x, y, z] for x in (f["bbox_sfm"][0][0], f["bbox_sfm"][1][0])
                                    for y in (f["bbox_sfm"][0][1], f["bbox_sfm"][1][1])
                                    for z in (f["bbox_sfm"][0][2], f["bbox_sfm"][1][2])])
                cl = apply(T_local, corners)
                f["bbox_local"] = [cl.min(0).round(4).tolist(), cl.max(0).round(4).tolist()]
            f["position_units"] = "metres" if scale_status != "relative" else "sfm_units"
            f["position_scale_status"] = scale_status
        else:
            f["position_local"] = None
        out.append(f)
    reg = [f for f in out if f.get("position_sfm") is not None]
    return {"facilities": out, "tracks_considered": len(per_track), "facilities_total": len(out),
            "facilities_registered": len(reg), "facilities_unregistered": len(out) - len(reg),
            "scene_extent_sfm": round(scene_extent, 4), "merge_threshold_rel": merge_dist_rel}
