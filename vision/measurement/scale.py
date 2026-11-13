"""Metric scale recovery.

Monocular metric-depth networks give depth with a global scale error that is often 10-30%, so
absolute lengths are only reported once the scale is anchored by:
  1. user reference segments   {"type":"segment","points":[[x,y],[x,y]],"length_m":1.0}
  2. user-identified objects   {"type":"object","detection_id":"...","dimension":"height"|"width","length_m":2.1}
  3. ArUco markers             {"type":"marker","dictionary":"DICT_4X4_50","size_m":0.15[, "id": 7]}
  4. (optional) priors         typical door height / adult height (weaker, flagged as prior)
  5. multi-image transfer      scale propagated from another calibrated photo of the same facility
Each reference implies a scale factor s = true_length / network_length; references are combined in
log space with inverse-variance weights and checked for mutual consistency.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision.measurement.analyzers import planar_extents, rays_on_plane
from vision.measurement.context import SceneContext
from vision.measurement.geometry import fit_plane
from vision.measurement.schemas import ReferenceUse, ScaleInfo, ScaleStatus


def _segment(ctx: SceneContext, ref: dict) -> ReferenceUse:
    pts = np.asarray(ref["points"], float) / ctx.to_orig
    L_true = float(ref["length_m"])
    desc = ref.get("note") or f"user segment {ref['points']}"
    h, w = ctx.rgb.shape[:2]
    if not ((0 <= pts[:, 0]) & (pts[:, 0] < w) & (0 <= pts[:, 1]) & (pts[:, 1] < h)).all():
        return ReferenceUse(kind="segment", description=desc, true_length_m=L_true, used=False,
                            reason="segment endpoints outside the image")
    len_px = float(np.linalg.norm(pts[1] - pts[0]))
    if len_px < 15:
        return ReferenceUse(kind="segment", description=desc, true_length_m=L_true, used=False,
                            reason=f"segment too short ({len_px * ctx.to_orig:.0f} px)")
    band = np.zeros((h, w), np.uint8)
    cv2.line(band, tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[1]), 1, thickness=max(9, int(len_px * 0.08)))
    ys, xs = np.nonzero(band)
    X = ctx.P[ys, xs].astype(np.float64)
    pl = fit_plane(X, rel_thresh=0.02, iters=200)
    if pl is None or pl.inlier_ratio < 0.5:
        # non-planar neighbourhood: use per-endpoint median depth
        Xe = [np.nanmedian(ctx.P[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].reshape(-1, 3), 0) for x, y in pts]
        L = float(np.linalg.norm(Xe[1] - Xe[0]))
        geo = 0.05
    else:
        Xe = rays_on_plane(pts, ctx, pl)
        L = float(np.linalg.norm(Xe[1] - Xe[0]))
        geo = pl.rms / max(L, 1e-6)
    u = ctx.cfg["uncertainty"]
    rel = math.sqrt(u["user_reference_rel"] ** 2 + (2 * u["pixel_localisation_px"] / len_px) ** 2 + geo ** 2)
    return ReferenceUse(kind="segment", description=desc, true_length_m=L_true, measured_unscaled=round(L, 4),
                        implied_scale=L_true / L, rel_sigma=rel), float(np.mean([Xe[0][2], Xe[1][2]]))


def _object(ctx: SceneContext, ref: dict) -> tuple[ReferenceUse, float | None]:
    did = ref["detection_id"]
    dim = ref.get("dimension", "height")
    L_true = float(ref["length_m"])
    desc = f"user: {dim} of {did} = {L_true} m"
    det = next((d for d in ctx.detections if d.id == did), None)
    if det is None:
        return ReferenceUse(kind="object", description=desc, true_length_m=L_true, used=False,
                            reason="detection_id not found in this image"), None
    if det.class_name == "person":
        r = _person_height(ctx, det)
        if isinstance(r, str):
            return ReferenceUse(kind="object", description=desc, true_length_m=L_true, used=False, reason=r), None
        Lh, z = r
        if dim != "height":
            return ReferenceUse(kind="object", description=desc, used=False, reason="only person height supported"), None
        return ReferenceUse(kind="object", description=desc, true_length_m=L_true, measured_unscaled=round(Lh, 4),
                            implied_scale=L_true / Lh, rel_sigma=0.04, detection_id=did), z
    mask = ctx.masks.get(det.id, (None,))[0]
    if mask is None:
        return ReferenceUse(kind="object", description=desc, true_length_m=L_true, used=False,
                            reason="no mask for this detection"), None
    ex = planar_extents(ctx, mask, ctx.gravity.up if ctx.gravity else None)
    if ex is None or "error" in ex:
        return ReferenceUse(kind="object", description=desc, true_length_m=L_true, used=False,
                            reason=(ex or {}).get("error", "plane fit failed")), None
    L = ex["width"] if dim == "width" else ex["height"]
    px = ex["width_px"] if dim == "width" else ex["height_px"]
    rel = math.sqrt(ctx.cfg["uncertainty"]["user_reference_rel"] ** 2 + (2 * 2.0 / max(px, 1)) ** 2 +
                    (ex["plane"].rms / max(L, 1e-6)) ** 2)
    return ReferenceUse(kind="object", description=desc, true_length_m=L_true, measured_unscaled=round(L, 4),
                        implied_scale=L_true / L, rel_sigma=rel, detection_id=did), ex["z"]


def _markers(ctx: SceneContext, ref: dict) -> list[tuple[ReferenceUse, float]]:
    name = ref.get("dictionary", "DICT_4X4_50")
    size = float(ref["size_m"])
    try:
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    except Exception as e:
        return [(ReferenceUse(kind="marker", description=f"ArUco {name}", used=False, reason=f"aruco unavailable: {e}"), 0)]
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [(ReferenceUse(kind="marker", description=f"ArUco {name} {size} m", true_length_m=size, used=False,
                              reason="no marker found"), 0)]
    out = []
    for c, i in zip(corners, ids.ravel()):
        if "id" in ref and int(ref["id"]) != int(i):
            continue
        c = c.reshape(4, 2)
        poly = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(poly, c.astype(np.int32), 1)
        ys, xs = np.nonzero(cv2.dilate(poly, np.ones((5, 5), np.uint8)))
        pl = fit_plane(ctx.P[ys, xs].astype(np.float64), rel_thresh=0.02, iters=150)
        side_px = float(np.mean(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)))
        desc = f"ArUco {name} id {int(i)} ({size} m)"
        if pl is None:
            out.append((ReferenceUse(kind="marker", description=desc, true_length_m=size, used=False,
                                     reason="marker depth not planar"), 0))
            continue
        Xc = rays_on_plane(c, ctx, pl)
        L = float(np.mean(np.linalg.norm(np.roll(Xc, -1, 0) - Xc, axis=1)))
        rel = math.sqrt((0.5 / side_px) ** 2 + 0.003 ** 2 + (pl.rms / max(L, 1e-6)) ** 2 / 4)
        out.append((ReferenceUse(kind="marker", description=desc, true_length_m=size, measured_unscaled=round(L, 4),
                                 implied_scale=size / L, rel_sigma=rel), float(Xc[:, 2].mean())))
    return out


def _person_height(ctx: SceneContext, det) -> tuple[float, float] | str:
    """Single-view metrology: feet on the ground plane, body vertical along gravity."""
    if ctx.ground is None or ctx.gravity is None or ctx.gravity.up is None:
        return "needs a ground plane and gravity direction"
    x1, y1, x2, y2 = ctx.box(det)
    h, w = ctx.rgb.shape[:2]
    if y1 <= 2 or y2 >= h - 3 or x1 <= 2 or x2 >= w - 3:
        return "person truncated by the image border"
    if (y2 - y1) / max(x2 - x1, 1) < 1.8:
        return "person not standing upright / partially visible"
    foot_band = [x1, y2 - 0.15 * (y2 - y1), x2, y2]
    for o in ctx.detections:
        if o.id == det.id or o.class_name not in ("person", "obstacle"):
            continue
        ox1, oy1, ox2, oy2 = ctx.box(o)
        ix = max(0.0, min(foot_band[2], ox2) - max(foot_band[0], ox1))
        iy = max(0.0, min(foot_band[3], oy2) - max(foot_band[1], oy1))
        if ix * iy > 0.25 * (foot_band[2] - foot_band[0]) * (foot_band[3] - foot_band[1]) and oy2 >= y2 - 2:
            return "feet may be occluded by another object/person"
    from vision.measurement.geometry import ray
    F = ctx.ground.intersect_ray(ray((x1 + x2) / 2, y2, ctx.K))
    if F is None:
        return "feet ray does not hit the ground plane"
    up = ctx.gravity.up
    rh = ray((x1 + x2) / 2, y1, ctx.K)
    # closest point between line F + t*up and the head ray s*rh
    A = np.stack([up, -rh], 1)
    sol, *_ = np.linalg.lstsq(A, -F, rcond=None)
    t = float(sol[0])
    if t <= 0:
        return "inconsistent geometry"
    return t, float(F[2])


def _priors(ctx: SceneContext) -> list[tuple[ReferenceUse, float]]:
    pri = ctx.cfg["reference_priors"]
    out = []
    for det in ctx.detections:
        if det.class_name == "door" and det.confidence >= 0.3 and det.id in ctx.masks:
            mask = ctx.masks[det.id][0]
            from vision.measurement.geometry import touching_sides
            if touching_sides(mask, 3):
                continue
            ex = planar_extents(ctx, mask, ctx.gravity.up if ctx.gravity else None)
            if not ex or "error" in ex:
                continue
            ar = ex["height"] / max(ex["width"], 1e-6)
            if not 1.7 <= ar <= 3.3:   # double doors / gates / windows are not used
                continue
            p = pri["door_height_m"]
            rel = math.hypot(p["sigma"] / p["mean"], 2 * 2.0 / max(ex["height_px"], 1))
            out.append((ReferenceUse(kind="prior:door", description=f"door {det.id} assumed {p['mean']} m tall",
                                     true_length_m=p["mean"], measured_unscaled=round(ex["height"], 4),
                                     implied_scale=p["mean"] / ex["height"], rel_sigma=rel), ex["z"]))
        if det.class_name == "person" and det.confidence >= 0.5:
            r = _person_height(ctx, det)
            if isinstance(r, str):
                continue
            p = pri["person_height_m"]
            out.append((ReferenceUse(kind="prior:person", description=f"person {det.id} assumed {p['mean']} m tall",
                                     true_length_m=p["mean"], measured_unscaled=round(r[0], 4),
                                     implied_scale=p["mean"] / r[0], rel_sigma=p["sigma"] / p["mean"] + 0.03), r[1]))
    return out


def combine(refs: list[tuple[ReferenceUse, float]], status: ScaleStatus, notes: list[str]) -> ScaleInfo:
    used = [(r, z) for r, z in refs if r.used and r.implied_scale and r.implied_scale > 0]
    if not used:
        raise ValueError("no usable reference")
    ls = np.log([r.implied_scale for r, _ in used])
    w = np.array([1 / max(r.rel_sigma, 1e-3) ** 2 for r, _ in used])
    mu = float((w * ls).sum() / w.sum())
    sig = float(1 / math.sqrt(w.sum()))
    if len(used) > 1:
        # consistency: mark references that disagree beyond 3 sigma, inflate uncertainty by the spread
        for (r, _), l in zip(used, ls):
            if abs(l - mu) > 3 * math.hypot(r.rel_sigma, sig):
                notes.append(f"reference '{r.description}' disagrees with the others "
                             f"(implies x{math.exp(l - mu):.2f}); check it")
        spread = float(np.sqrt((w * (ls - mu) ** 2).sum() / w.sum()))
        sig = max(sig, spread / math.sqrt(len(used)))
    zref = float(np.average([z for _, z in used], weights=w)) if any(z for _, z in used) else None
    if zref is not None:
        zref *= math.exp(mu)   # reference depths were measured on the unscaled network depth
    return ScaleInfo(status=status, scale_factor=math.exp(mu), rel_sigma=sig, reference_depth_m=zref,
                     references=[r for r, _ in refs], notes=notes)


def recover_scale(ctx: SceneContext, calibration: dict | None, scale_override: dict | None = None) -> ScaleInfo:
    """ctx must hold the UNSCALED point map (scale factor 1)."""
    calibration = calibration or {}
    notes: list[str] = []
    if scale_override:
        return ScaleInfo(status=ScaleStatus.calibrated_multi_image, scale_factor=float(scale_override["scale"]),
                         rel_sigma=float(scale_override["rel_sigma"]),
                         reference_depth_m=scale_override.get("reference_depth_m"),
                         references=[ReferenceUse(kind="multi_image", description=scale_override.get("description", ""),
                                                  implied_scale=float(scale_override["scale"]),
                                                  rel_sigma=float(scale_override["rel_sigma"]))],
                         notes=["scale transferred from another calibrated image of the same facility"])
    user: list[tuple[ReferenceUse, float]] = []
    markers: list[tuple[ReferenceUse, float]] = []
    for ref in calibration.get("references", []) or []:
        t = ref.get("type")
        try:
            if t == "segment":
                r = _segment(ctx, ref)
                user.append(r if isinstance(r, tuple) else (r, 0.0))
            elif t == "object":
                r, z = _object(ctx, ref)
                user.append((r, z or 0.0))
            elif t == "marker":
                markers.extend(_markers(ctx, ref))
            else:
                user.append((ReferenceUse(kind=str(t), description=str(ref), used=False,
                                          reason="unknown reference type"), 0.0))
        except (KeyError, TypeError, ValueError) as e:
            user.append((ReferenceUse(kind=str(t), description=str(ref), used=False, reason=f"invalid: {e}"), 0.0))
    all_user = user + markers
    if any(r.used for r, _ in all_user):
        st = ScaleStatus.calibrated_marker if any(r.used for r, _ in markers) and not any(r.used for r, _ in user) \
            else ScaleStatus.calibrated_reference
        return combine(all_user, st, notes)
    if all_user:
        notes.append("all supplied references were rejected: " + "; ".join(r.reason or "" for r, _ in all_user))
    if calibration.get("use_priors", True):
        pri = _priors(ctx)
        if pri:
            s = combine(pri, ScaleStatus.prior_reference_object, notes + [
                "scale from typical object dimensions; supply a measured reference for calibrated results"])
            s.references = [r for r, _ in all_user] + s.references
            return s
    u = ctx.cfg["uncertainty"]["uncalibrated_depth_scale_rel"]
    notes.append("no reference available: absolute lengths are withheld (REQUIRES_CALIBRATION)")
    return ScaleInfo(status=ScaleStatus.uncalibrated_monocular, scale_factor=1.0, rel_sigma=u,
                     references=[r for r, _ in all_user], notes=notes)
