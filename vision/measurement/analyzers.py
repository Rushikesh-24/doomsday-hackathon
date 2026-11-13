"""Geometric analyzers. All work on the calibrated point map in SceneContext (camera frame,
x right, y down, z forward). Planar quantities use ray-plane intersection of the object's mask
pixels with a robustly fitted plane, which avoids the depth bleeding that monocular networks
show at object boundaries."""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision.accessibility.geometry import _find_peaks, estimate_step_edges
from vision.measurement.backends import RAIL_LABELS
from vision.measurement.context import SceneContext
from vision.measurement.geometry import Plane, angle_deg, fit_plane, project, robust_extent, touching_sides
from vision.measurement.schemas import Measurement
from vision.schemas import Detection

# ----------------------------------------------------------------------------- helpers


def mask_pixels(mask: np.ndarray, max_n: int = 40000, seed: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    uv = np.stack([xs, ys], 1).astype(np.float64)
    if len(uv) > max_n:
        uv = uv[np.random.default_rng(seed).choice(len(uv), max_n, replace=False)]
    return uv


def rays(uv: np.ndarray, ctx: SceneContext) -> np.ndarray:
    K = ctx.K
    r = np.stack([(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy, np.ones(len(uv))], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def rays_on_plane(uv: np.ndarray, ctx: SceneContext, plane: Plane) -> np.ndarray:
    r = rays(uv, ctx)
    den = r @ plane.normal
    den = np.where(np.abs(den) < 1e-6, np.nan, den)
    t = -plane.d / den
    X = r * t[:, None]
    X[t <= 0] = np.nan
    return X


def points_of(mask: np.ndarray, ctx: SceneContext, max_n: int = 40000) -> tuple[np.ndarray, np.ndarray]:
    uv = mask_pixels(mask, max_n)
    X = ctx.P[uv[:, 1].astype(int), uv[:, 0].astype(int)].astype(np.float64)
    ok = np.isfinite(X).all(1) & (X[:, 2] > 0)
    return X[ok], uv[ok]


def in_plane_axes(n: np.ndarray, up: np.ndarray):
    """vertical-in-plane axis v (towards up) and horizontal axis h = v x n (unit)."""
    v = up - (up @ n) * n
    if np.linalg.norm(v) < 1e-6:  # plane is horizontal: pick camera x as horizontal
        h = np.array([1.0, 0, 0]) - (np.array([1.0, 0, 0]) @ n) * n
        h /= np.linalg.norm(h)
        return np.cross(n, h), h
    v /= np.linalg.norm(v)
    h = np.cross(v, n)
    return v, h / np.linalg.norm(h)


def grazing_angle(plane: Plane, X: np.ndarray) -> float:
    c = np.nanmedian(X, 0)
    return angle_deg(plane.normal, c / np.linalg.norm(c))


def seg_line(ctx: SceneContext, A: np.ndarray, B: np.ndarray) -> list[float]:
    p = project(np.stack([A, B]), ctx.K)
    return [p[0, 0], p[0, 1], p[1, 0], p[1, 1]]


def mask_polygon(mask: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True)
    return c.reshape(-1, 2).astype(float)


def check_mask(ctx: SceneContext, det: Detection, mask: np.ndarray) -> str | None:
    a = ctx.cfg["analysis"]
    if mask is None or mask.sum() < a["min_mask_pixels"]:
        return f"object region too small ({0 if mask is None else int(mask.sum())} px < {a['min_mask_pixels']} px)"
    return None


# ----------------------------------------------------------------------------- planar objects


def planar_extents(ctx: SceneContext, mask: np.ndarray, up: np.ndarray | None) -> dict | None:
    """Fit a plane to the mask's 3D points and measure width/height in that plane via ray casting."""
    X, uv = points_of(mask, ctx)
    if len(X) < ctx.cfg["analysis"]["min_valid_points"]:
        return {"error": f"only {len(X)} valid depth points"}
    pl = fit_plane(X, rel_thresh=0.02)
    if pl is None or pl.inlier_ratio < ctx.cfg["analysis"]["min_plane_inlier_ratio"]:
        return {"error": f"surface is not planar enough (inlier ratio {0 if pl is None else pl.inlier_ratio:.2f})"}
    g = grazing_angle(pl, X)
    if g > ctx.cfg["analysis"]["max_grazing_angle_deg"]:
        return {"error": f"viewed too obliquely ({g:.0f} deg from the surface normal)"}
    upv = up if up is not None else np.array([0, -1.0, 0])
    v, h = in_plane_axes(pl.normal, upv)
    Y = rays_on_plane(mask_pixels(mask), ctx, pl)
    Y = Y[np.isfinite(Y).all(1)]
    a, b = Y @ h, Y @ v
    w, a0, a1 = robust_extent(a, 0.5, 99.5)
    ht, b0, b1 = robust_extent(b, 0.5, 99.5)
    c = Y.mean(0)
    ca, cb = c @ h, c @ v
    base = c - ca * h - cb * v
    P = lambda aa, bb: base + aa * h + bb * v
    mid_b, mid_a = (b0 + b1) / 2, (a0 + a1) / 2
    return {"width": w, "height": ht, "plane": pl, "grazing_deg": g, "z": float(np.median(X[:, 2])),
            "width_line": seg_line(ctx, P(a0, mid_b), P(a1, mid_b)),
            "height_line": seg_line(ctx, P(mid_a, b0), P(mid_a, b1)),
            "width_px": float(np.ptp(project(np.stack([P(a0, mid_b), P(a1, mid_b)]), ctx.K)[:, 0])),
            "height_px": float(abs(np.diff(project(np.stack([P(mid_a, b0), P(mid_a, b1)]), ctx.K)[:, 1]))[0]),
            "up_used": "gravity" if up is not None else "camera_y_assumed"}


def analyze_door(ctx: SceneContext, det: Detection) -> list[Measurement]:
    out = []
    mask, src, _ = ctx.masks.get(det.id, (None, None, 0))
    reason = check_mask(ctx, det, mask)
    if reason:
        return [ctx.refuse(det, n, reason, "m") for n in ("doorway_width", "clear_width_estimate", "door_height")]
    trunc = touching_sides(mask, ctx.cfg["analysis"]["border_margin_px"])
    up = ctx.gravity.up if ctx.gravity else None
    ex = planar_extents(ctx, mask, up)
    if ex is None or "error" in ex:
        r = ex["error"] if ex else "plane fit failed"
        return [ctx.refuse(det, n, r, "m") for n in ("doorway_width", "clear_width_estimate", "door_height")]
    px = ctx.cfg["uncertainty"]["pixel_localisation_px"]
    notes = [f"mask: {src}", f"plane rms {ex['plane'].rms:.3f}, grazing {ex['grazing_deg']:.0f} deg"]
    if ex["up_used"] != "gravity":
        notes.append("gravity direction unknown: assumed camera is upright")
    lr_trunc = bool({"left", "right"} & set(trunc))
    tb_trunc = bool({"top", "bottom"} & set(trunc))
    q = 0.9 if ex["up_used"] == "gravity" else 0.7
    geom_w = math.hypot(2 * px / max(ex["width_px"], 1), ex["plane"].rms / max(ex["width"], 1e-6))
    lab = [(ex["width_line"][0] + ex["width_line"][2]) / 2, (ex["width_line"][1] + ex["width_line"][3]) / 2]
    if lr_trunc:
        out.append(ctx.refuse(det, "doorway_width", f"door truncated at image {sorted(trunc)}: width not observable", "m"))
        out.append(ctx.refuse(det, "clear_width_estimate", "door width not observable", "m"))
    else:
        out.append(ctx.length(det, "doorway_width", ex["width"], z=ex["z"], geom_rel=geom_w, quality=q,
                              method="SAM mask rays intersected with RANSAC door plane; horizontal extent (p0.5-p99.5)",
                              overlay=ctx.overlay(lines=[ex["width_line"]], label_at=lab), notes=notes))
        ded = ctx.cfg["door"]["clear_width_deduction_m"]
        is_slide = det.class_name == "elevator"
        dm, ds = (0.0, 0.02) if is_slide else (ded["mean"], ded["sigma"])
        m = ctx.length(det, "clear_width_estimate", ex["width"] - dm, z=ex["z"], geom_rel=geom_w, quality=q * 0.85,
                       method=("doorway width (sliding lift doors: no leaf deduction)" if is_slide else
                               f"doorway width minus {dm:.2f} m for leaf thickness + stops (door open 90 deg)"),
                       extra_abs_sigma=ds, notes=notes + ["clear width depends on the open leaf; measure on site"])
        out.append(m)
    gh = math.hypot(2 * px / max(ex["height_px"], 1), ex["plane"].rms / max(ex["height"], 1e-6))
    out.append(ctx.length(det, "door_height", ex["height"], z=ex["z"], geom_rel=gh, quality=q, lower_bound=tb_trunc,
                          method="vertical extent in door plane",
                          overlay=ctx.overlay(lines=[ex["height_line"]]), notes=notes))
    if not (lr_trunc or tb_trunc):
        r = ex["width"] / ex["height"]
        out.append(ctx.scale_free(det, "width_to_height_ratio", r, None, sigma=r * math.hypot(geom_w, gh),
                                  method="ratio of in-plane extents (scale-independent)", confidence=q))
    out.append(ctx.manual(det, "threshold_height", "door threshold height is a few mm - below image resolution", "m"))
    out.append(ctx.manual(det, "opening_force_and_swing", "door swing and opening force require on-site check"))
    return out


# ----------------------------------------------------------------------------- ramps


def _slope_with_focal(ctx: SceneContext, X: np.ndarray, Xg: np.ndarray | None, f: float) -> float | None:
    """Recompute ramp slope after rescaling the focal length by f (sensitivity analysis)."""
    S = np.array([1 / f, 1 / f, 1.0])
    pl = fit_plane(X * S, rel_thresh=0.02, iters=150)
    if pl is None:
        return None
    if Xg is not None:
        g = fit_plane(Xg * S, rel_thresh=0.02, iters=150)
        if g is None:
            return None
        up = g.normal
    else:
        return None
    return angle_deg(pl.normal, up)


def railing_points(ctx: SceneContext, det: Detection, region_pad: float = 0.5) -> np.ndarray:
    m = ctx.label_mask(RAIL_LABELS)
    for d in ctx.detections:
        if d.class_name == "handrail" and d.id in ctx.masks:
            m = m | ctx.masks[d.id][0]
    m &= ctx.box_mask(det, region_pad)
    if m.sum() < 30:
        return np.zeros((0, 3))
    X, _ = points_of(m, ctx, 8000)
    return X


def handrail_coverage(ctx, X_rail, center, along, lateral, half_width, a_range, up, plane) -> dict:
    if len(X_rail) == 0:
        return {"left": 0.0, "right": 0.0, "n": 0}
    rel = X_rail - center
    a = rel @ along
    l = rel @ lateral
    hgt = plane.distance(X_rail) if plane is not None else rel @ up
    near = (np.abs(l) < 2.0 * half_width) & (np.abs(hgt) < 1.5 * half_width + 0.3 * (a_range[1] - a_range[0]))
    bins = np.linspace(a_range[0], a_range[1], 21)
    res = {"n": int(near.sum())}
    for side, sel in (("left", l < 0), ("right", l >= 0)):
        s = near & sel
        if s.sum() < 10:
            res[side] = 0.0
            continue
        hist, _ = np.histogram(a[s], bins)
        res[side] = float((hist > 0).mean())
    return res


def analyze_ramp(ctx: SceneContext, det: Detection) -> list[Measurement]:
    names_len = ("ramp_length", "horizontal_run", "rise", "clear_width")
    mask, src, _ = ctx.masks.get(det.id, (None, None, 0))
    reason = check_mask(ctx, det, mask)
    if reason:
        return [ctx.refuse(det, "running_slope_percent", reason, "%", scale_dependent=False)] + \
               [ctx.refuse(det, n, reason, "m") for n in names_len]
    X, uv = points_of(mask, ctx)
    pl = fit_plane(X, rel_thresh=0.02) if len(X) >= ctx.cfg["analysis"]["min_valid_points"] else None
    if pl is None or pl.inlier_ratio < ctx.cfg["analysis"]["min_plane_inlier_ratio"]:
        r = "ramp surface not planar in depth (no dominant plane)"
        return [ctx.refuse(det, "running_slope_percent", r, "%", scale_dependent=False)] + \
               [ctx.refuse(det, n, r, "m") for n in names_len]
    out: list[Measurement] = []
    gv = ctx.gravity
    notes = [f"mask: {src}", f"ramp plane inliers {pl.inlier_ratio:.2f}, rms {pl.rms:.3f}"]
    if gv is None or gv.up is None:
        out.append(ctx.refuse(det, "running_slope_percent", "no gravity reference: neither a ground plane nor "
                              "vertical vanishing point could be found", "%", scale_dependent=False))
        up = None
    else:
        up = gv.up
        theta = angle_deg(pl.normal, up)
        # sensitivity to focal length (dominant unknown for uncalibrated photos)
        sig_f = 0.0
        if gv.source == "ground_plane" and ctx.ground is not None:
            Xg, _ = points_of(ctx.ground_mask, ctx, 20000)
            fs = ctx.K.focal_rel_sigma
            alts = [_slope_with_focal(ctx, X, Xg, 1 + fs), _slope_with_focal(ctx, X, Xg, 1 - fs)]
            alts = [a for a in alts if a is not None]
            if alts:
                sig_f = max(abs(a - theta) for a in alts)
        floor = ctx.cfg["uncertainty"]["slope_model_floor_deg"]
        sig = math.sqrt(pl.normal_sigma_deg ** 2 + (gv.sigma_deg or 2.0) ** 2 + sig_f ** 2 + floor ** 2)
        if gv.agreement_deg is not None:
            sig = math.hypot(sig, 0.5 * gv.agreement_deg)
        grade = 100 * math.tan(math.radians(theta))
        sig_grade = 100 * math.radians(sig) / math.cos(math.radians(theta)) ** 2
        conf = float(np.clip(1 - sig / 6.0, 0.05, 0.95)) * (0.9 if gv.source == "ground_plane" else 0.7)
        notes_s = notes + [f"gravity from {gv.source} (sigma {gv.sigma_deg or 0:.1f} deg)",
                           f"focal-length sensitivity {sig_f:.2f} deg (camera: {ctx.K.source})"]
        out.append(ctx.scale_free(det, "running_slope_deg", theta, "deg", sigma=sig, confidence=conf,
                                  method="angle between RANSAC ramp-surface normal and gravity", notes=notes_s))
        out.append(ctx.scale_free(det, "running_slope_percent", grade, "%", sigma=sig_grade, confidence=conf,
                                  method="100 * tan(running slope); scale-independent", notes=notes_s))
        out.append(ctx.scale_free(det, "running_slope_ratio", f"1:{(1 / max(math.tan(math.radians(theta)), 1e-6)):.1f}",
                                  None, sigma=None, confidence=conf, method="rise:run from slope angle"))
    upv = up if up is not None else np.array([0, -1.0, 0])
    v, h = in_plane_axes(pl.normal, upv)   # v = uphill direction in plane
    Y = rays_on_plane(mask_pixels(mask), ctx, pl)
    Y = Y[np.isfinite(Y).all(1)]
    a, b = Y @ v, Y @ h
    L, a0, a1 = robust_extent(a, 0.2, 99.8)
    W, b0, b1 = robust_extent(b, 0.3, 99.7)
    trunc = touching_sides(mask, ctx.cfg["analysis"]["border_margin_px"])
    c = Y.mean(0)
    base = c - (c @ v) * v - (c @ h) * h
    Pp = lambda aa, bb: base + aa * v + bb * h
    lline = seg_line(ctx, Pp(a0, (b0 + b1) / 2), Pp(a1, (b0 + b1) / 2))
    wline = seg_line(ctx, Pp((a0 + a1) / 2, b0), Pp((a0 + a1) / 2, b1))
    z = float(np.median(X[:, 2]))
    px = ctx.cfg["uncertainty"]["pixel_localisation_px"]
    lpx = math.hypot(lline[2] - lline[0], lline[3] - lline[1])
    wpx = math.hypot(wline[2] - wline[0], wline[3] - wline[1])
    gl = math.hypot(2 * px / max(lpx, 1), pl.rms / max(L, 1e-6))
    gw = math.hypot(2 * px / max(wpx, 1), pl.rms / max(W, 1e-6))
    lb = bool(trunc)
    vis_note = ["visible extent only - the ramp may continue outside the frame or behind occluders"]
    out.append(ctx.length(det, "ramp_length", L, z=z, geom_rel=gl, lower_bound=lb, along_view=True, overlay=ctx.overlay(lines=[lline]),
                          method="extent along the in-plane uphill axis (ray-plane intersection)", notes=vis_note))
    if up is not None:
        th = math.radians(angle_deg(pl.normal, up))
        out.append(ctx.length(det, "horizontal_run", L * math.cos(th), z=z, geom_rel=gl, lower_bound=lb, along_view=True,
                              method="ramp_length * cos(slope)", notes=vis_note))
        out.append(ctx.length(det, "rise", L * math.sin(th), z=z, geom_rel=math.hypot(gl, 0.3), lower_bound=lb, along_view=True,
                              method="ramp_length * sin(slope) (sensitive to slope error)", notes=vis_note))
    out.append(ctx.length(det, "clear_width", W, z=z, geom_rel=gw, lower_bound=bool({"left", "right"} & set(trunc)),
                          overlay=ctx.overlay(lines=[wline]),
                          method="ramp-surface extent across the uphill axis; clear width between handrails may be "
                                 "smaller", notes=[f"mask: {src}"]))
    # handrails along the ramp
    Xr = railing_points(ctx, det)
    cov = handrail_coverage(ctx, Xr, base, v, h, W / 2 if W > 0 else 0.5, (a0, a1), upv, pl)
    present = max(cov["left"], cov["right"]) >= 0.3
    out.append(ctx.scale_free(det, "handrail_present", bool(present), None, sigma=None,
                              confidence=0.6 if present else 0.4,
                              method="ADE20K railing/bannister pixels + handrail detections along the ramp",
                              notes=[f"{cov['n']} railing points near ramp"] + ([] if present else
                                    ["no railing found: absence of evidence is not evidence of absence"])))
    for side in ("left", "right"):
        out.append(ctx.scale_free(det, f"handrail_coverage_{side}", cov[side], "fraction", sigma=0.15,
                                  confidence=0.5, method="fraction of the ramp length with railing evidence on that "
                                                         "side (camera-relative side)"))
    out.append(ctx.scale_free(det, "handrail_continuous", bool(max(cov["left"], cov["right"]) >= 0.85), None,
                              sigma=None, confidence=0.45, method="coverage >= 85% of visible ramp length on a side"))
    # surface condition indicators
    out.extend(surface_condition(ctx, det, mask, pl, L))
    return out


def surface_condition(ctx: SceneContext, det: Detection, mask: np.ndarray, pl: Plane, L: float) -> list[Measurement]:
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2HSV)
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if m.sum() < 200:
        return [ctx.refuse(det, "surface_irregularity_ratio", "surface region too small", None, scale_dependent=False)]
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = float(np.abs(lap[m]).mean() / (gray[m].std() + 1e-6))
    spec = float(((hsv[..., 2] > 240) & (hsv[..., 1] < 30))[m].mean())
    rough = pl.rms / max(L, 1e-6)
    cond = "regular"
    flags = []
    if rough > 0.03:
        flags.append("depth surface deviates from a plane (possible unevenness, steps or wrong mask)")
    if spec > 0.08:
        flags.append("specular highlights (possible wet/glossy surface)")
    if tex > 1.2:
        flags.append("high fine-texture energy (possible cracks/gravel/grating)")
    if flags:
        cond = "irregularity indicators"
    return [ctx.scale_free(det, "surface_irregularity_ratio", rough, "rms/length", sigma=None, confidence=0.35,
                           method="RANSAC plane residual RMS / ramp length"),
            ctx.scale_free(det, "surface_condition", cond, None, sigma=None, confidence=0.3,
                           method="plane residual + Laplacian texture energy + specular fraction",
                           notes=flags or ["no indicator triggered"] + [f"texture={tex:.2f} specular={spec:.3f}"],
                           verification_reason="surface condition indicators are weak; inspect on site"),
            ctx.manual(det, "slip_resistance", "slip resistance needs a pendulum/tribometer test")]


# ----------------------------------------------------------------------------- stairs


def _edge_candidates(ctx: SceneContext, mask: np.ndarray, det: Detection, n_strips: int = 7) -> list[tuple[int, float, float]]:
    """Horizontal-edge peaks along several vertical strips through the stair mask. (strip, u, v)."""
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 0, 1, ksize=3))
    ys, xs = np.nonzero(mask)
    x0, x1 = np.percentile(xs, [10, 90])
    sw = max(3, int((x1 - x0) / (n_strips * 2.5)))
    out = []
    for k, xc in enumerate(np.linspace(x0, x1, n_strips)):
        xa, xb = int(xc - sw), int(xc + sw)
        col = mask[:, xa:xb].mean(1) > 0.6
        rows = np.nonzero(col)[0]
        if len(rows) < 20:
            continue
        r0, r1 = rows.min(), rows.max()
        prof = gy[r0:r1 + 1, xa:xb].mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 1).ravel()
        trend = cv2.blur(prof.reshape(-1, 1), (1, 41)).ravel()
        sig = prof - trend
        if sig.std() < 1e-6:
            continue
        for p in _find_peaks(sig, min_distance=3, prominence=0.4 * sig.std()):
            out.append((k, float(xc), float(r0 + p)))
    return out


def _strip_runs(t: np.ndarray, strips: np.ndarray, lo: float = 0.55, hi: float = 1.8) -> list[dict]:
    """Per strip: the longest run of edges whose consecutive spacings vary smoothly (depth-model
    distortion makes spacing drift along the flight, so a single global grid is not valid).
    Each step may produce one edge (nosing) or two (nosing + riser base); the every-other-edge
    interpretation is chosen when it is clearly more regular."""
    from vision.accessibility.geometry import _longest_consistent_run
    runs = []
    for k in np.unique(strips):
        ts = np.sort(t[strips == k])
        if len(ts) < 4:
            continue
        best = None
        for cand, per_step in ((ts, 1), (ts[0::2], 2), (ts[1::2], 2)):
            if len(cand) < 3:
                continue
            sp = np.diff(cand)
            if (sp <= 0).any():
                continue
            i0, i1 = _longest_consistent_run(sp, lo, hi)
            run = sp[i0:i1]
            if len(run) < 2:
                continue
            reg = float(np.std(np.log(run[1:] / run[:-1]))) if len(run) > 2 else 0.5
            score = len(run) * (1 - min(reg, 0.9))
            if best is None or (score > best["score"] * 1.3 if per_step == 2 else score > best["score"]):
                best = dict(score=score, n=len(run) + 1, span=float(cand[i1] - cand[i0]), reg=reg, per_step=per_step,
                            coverage=float((cand[i1] - cand[i0]) / max(np.ptp(ts), 1e-9)), strip=int(k),
                            t_first=float(cand[i0]), t_last=float(cand[i1]))
        if best:
            runs.append(best)
    return runs


def analyze_stairs(ctx: SceneContext, det: Detection) -> list[Measurement]:
    names = ("riser_height", "tread_depth", "stair_width")
    mask, src, _ = ctx.masks.get(det.id, (None, None, 0))
    reason = check_mask(ctx, det, mask)
    if reason:
        return [ctx.refuse(det, "step_count", reason, "steps", scale_dependent=False)] + \
               [ctx.refuse(det, n, reason, "m") for n in names]
    out: list[Measurement] = []
    trunc = touching_sides(mask, ctx.cfg["analysis"]["border_margin_px"])
    gv = ctx.gravity
    up = gv.up if gv and gv.up is not None else None
    # --- 2D estimate (fallback / cross-check)
    ys, xs = np.nonzero(mask)
    crop = ctx.rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    est2d = estimate_step_edges(crop)
    # --- 3D: lift edge candidates, fit nosing line, uniform grid
    cands = _edge_candidates(ctx, mask, det)
    res3d = None
    why = None
    if up is None:
        why = "no gravity reference"
    elif len(cands) < 6:
        why = f"only {len(cands)} step-edge candidates"
    else:
        uvs = np.array([[c[1], c[2]] for c in cands])
        strips = np.array([c[0] for c in cands])
        # run direction: horizontal component of riser normals (they face down the stairs)
        Xs, uvm = points_of(mask, ctx, 20000)
        Ns = ctx.N[uvm[:, 1].astype(int), uvm[:, 0].astype(int)]
        riser = np.abs(Ns @ up) < 0.4
        r = Ns[riser].mean(0) if riser.sum() > 50 else -np.median(Xs, 0)
        r = r - (r @ up) * up
        # Depth is too noisy to localise single nosings, but it is smooth along a strip: fit a
        # low-order 3D curve X(v) per strip from ALL strip pixels and map the precise 2D edge rows
        # through it (perspective-correct, noise-free positions).
        Xe = np.full((len(uvs), 3), np.nan)
        for k in np.unique(strips):
            sel = strips == k
            u0 = int(uvs[sel][0, 0])
            sw = max(2, int(0.02 * ctx.rgb.shape[1]))
            col = mask[:, max(0, u0 - sw):u0 + sw + 1].mean(1) > 0.6
            rows = np.nonzero(col)[0]
            if len(rows) < 20:
                continue
            Xr = np.nanmedian(ctx.P[rows, max(0, u0 - sw):u0 + sw + 1], axis=1)
            okr = np.isfinite(Xr).all(1)
            if okr.sum() < 15:
                continue
            for dim in range(3):
                c = np.polyfit(rows[okr], Xr[okr, dim], 3)
                Xe[sel, dim] = np.polyval(c, uvs[sel, 1])
        ok = np.isfinite(Xe).all(1)
        Xe, uvs, strips = Xe[ok], uvs[ok], strips[ok]
        if len(Xe) < 6 or np.linalg.norm(r) < 1e-6:
            why = "cannot lift step edges to 3D"
        else:
            r /= np.linalg.norm(r)
            hgt = Xe @ up
            dd = Xe @ r
            # robust line fit hgt = a + b*dd (nosings are collinear)
            best = None
            rng = np.random.default_rng(0)
            thr = 0.02 * float(np.median(Xe[:, 2]))
            for _ in range(300):
                i, j = rng.choice(len(dd), 2, replace=False)
                if abs(dd[i] - dd[j]) < 1e-6:
                    continue
                b = (hgt[j] - hgt[i]) / (dd[j] - dd[i])
                a = hgt[i] - b * dd[i]
                inl = np.abs(hgt - (a + b * dd)) / math.sqrt(1 + b * b) < thr
                if best is None or inl.sum() > best[2].sum():
                    best = (a, b, inl)
            if best is None or best[2].sum() < 5:
                why = "step edges are not collinear in 3D"
            else:
                a, b, inl = best
                b = np.polyfit(dd[inl], hgt[inl], 1)[0]
                pitch = math.degrees(math.atan(abs(b)))
                t = (dd + b * hgt) / math.sqrt(1 + b * b)       # coordinate along the nosing line
                runs = _strip_runs(t, strips)
                runs = [x for x in runs if x["n"] >= 3 and x["coverage"] >= 0.5]
                if len(runs) < 2 or not (5 <= pitch <= 60):
                    why = (f"step edges not regular in enough strips ({len(runs)})" if 5 <= pitch <= 60
                           else f"implausible pitch {pitch:.0f} deg")
                else:
                    ns = np.array([x["n"] for x in runs])
                    n_steps = int(np.median(ns))
                    mad = float(np.median(np.abs(ns - n_steps)))
                    Dm = float(np.median([x["span"] / (x["n"] - 1) for x in runs]))
                    agree_frac = float(np.mean(np.abs(ns - n_steps) <= 1))
                    keep = np.abs(strips - runs[0]["strip"]) >= 0  # all edges for overlay
                    res3d = dict(n=n_steps, mad=mad, D=Dm, pitch=pitch, agree=agree_frac, b=b, r=r,
                                 line_inl=float(inl.mean()),
                                 n_strips=len(runs), Xe=Xe[inl], uv=uvs[keep],
                                 per_step=int(np.median([x["per_step"] for x in runs])))
    lb = bool({"top", "bottom"} & set(trunc))
    if res3d:
        conf = float(np.clip(res3d["agree"], 0, 1)) * min(1.0, res3d["n_strips"] / 4)
        agree = est2d.get("count") is not None and abs(est2d["count"] - res3d["n"]) <= max(1, 0.15 * res3d["n"])
        notes = [f"median over {res3d['n_strips']} image strips ({res3d['agree']:.0%} within +-1); "
                 f"{res3d['per_step']} detected edge(s) per step",
                 f"2D edge-periodicity estimate: {est2d.get('count')} ({'agrees' if agree else 'disagrees'})"]
        sig_n = max(1.0, 1.4826 * res3d["mad"])  # validated: off-by-one at the flight ends is common
        out.append(ctx.scale_free(det, "step_count", res3d["n"], "steps", sigma=sig_n,
                                  confidence=float(np.clip(conf + (0.1 if agree else -0.15), 0.05, 0.95)), lower_bound=lb,
                                  method="horizontal step edges per image strip, mapped through a smooth per-strip 3D "
                                         "depth curve (perspective-corrected), longest regular run, median over strips",
                                  overlay=ctx.overlay(points=res3d["uv"]), notes=notes + (
                                      ["staircase continues beyond the image: count is a lower bound"] if lb else [])))
        pconf = float(np.clip(res3d["line_inl"], 0, 1)) * (0.8 if ctx.gravity.source == "ground_plane" else 0.65)
        out.append(ctx.scale_free(det, "pitch_deg", res3d["pitch"], "deg", sigma=3.0, confidence=pconf,
                                  method="slope of the 3D nosing line relative to gravity",
                                  notes=[f"{res3d['line_inl']:.0%} of lifted edges on the nosing line; gravity from "
                                         f"{ctx.gravity.source}"]))
        rr = math.tan(math.radians(res3d["pitch"]))
        out.append(ctx.scale_free(det, "riser_to_tread_ratio", rr, None, sigma=rr * 0.15, confidence=pconf * 0.9,
                                  method="tan(pitch); scale-independent"))
        z = float(np.median(res3d["Xe"][:, 2]))
        th = math.radians(res3d["pitch"])
        note_d = ["mean nosing spacing over the run; monocular depth compresses distant steps, so individual "
                  "steps may differ"]
        out.append(ctx.length(det, "riser_height", res3d["D"] * math.sin(th), z=z, geom_rel=0.15, quality=conf,
                              method="mean nosing spacing * sin(pitch)", notes=note_d))
        out.append(ctx.length(det, "tread_depth", res3d["D"] * math.cos(th), z=z, geom_rel=0.15, quality=conf, along_view=True,
                              method="mean nosing spacing * cos(pitch)", notes=note_d))
    else:
        c2 = est2d.get("count")
        if c2 and est2d.get("confidence", 0) >= 0.25:
            out.append(ctx.scale_free(det, "step_count", int(c2), "steps", sigma=1.5, confidence=est2d["confidence"] * 0.8,
                                      lower_bound=lb, method="2D edge periodicity (3D method unavailable: " + why + ")",
                                      notes=["2D method is perspective-sensitive"]))
        else:
            out.append(ctx.refuse(det, "step_count", f"3D: {why}; 2D: {est2d.get('reason', 'low regularity')}",
                                  "steps", scale_dependent=False))
        out.append(ctx.refuse(det, "riser_height", f"step geometry not recovered ({why})", "m"))
        out.append(ctx.refuse(det, "tread_depth", f"step geometry not recovered ({why})", "m"))
    # width: across the run direction, within the stair mask, on the nosing plane
    if res3d is not None and up is not None:
        w_ax = np.cross(up, res3d["r"])
        w_ax /= np.linalg.norm(w_ax)
        pitch_dir = res3d["r"] + res3d["b"] * up
        n_pl = np.cross(pitch_dir, w_ax)
        n_pl /= np.linalg.norm(n_pl)
        c = np.median(res3d["Xe"], 0)
        pl = Plane(n_pl if n_pl @ c < 0 else -n_pl, 0.0, np.zeros(0, bool), 1.0, 0.0, 0.0, 0.0)
        pl.d = -float(pl.normal @ c)
        Xm, _ = points_of(mask, ctx, 30000)
        near = np.abs(pl.distance(Xm)) < 0.08 * float(np.median(Xm[:, 2]))   # reject grazing / background pixels
        if near.sum() < 100:
            out.append(ctx.refuse(det, "stair_width", "too few stair pixels near the nosing plane", "m"))
            Y = None
        else:
            Y = Xm[near]
            wv = Y @ w_ax
            W, w0, w1 = robust_extent(wv, 2, 98)
            cc = Y.mean(0)
            line = seg_line(ctx, cc + (w0 - cc @ w_ax) * w_ax, cc + (w1 - cc @ w_ax) * w_ax)
        if Y is not None:
            out.append(ctx.length(det, "stair_width", W, z=float(np.median(Y[:, 2])), geom_rel=0.08,
                                  lower_bound=bool({"left", "right"} & set(trunc)), overlay=ctx.overlay(lines=[line]),
                                  method="lateral extent (p2-p98) of stair-mask depth points lying on the nosing plane",
                                  notes=["includes balustrade/wall-to-wall extent visible in the mask"]))
        Xr = railing_points(ctx, det, 0.3)
        sides = set()
        if len(Xr):
            lat = (Xr - c) @ w_ax
            if (lat < -0.25 * W).sum() > 20:
                sides.add("left")
            if (lat > 0.25 * W).sum() > 20:
                sides.add("right")
    else:
        out.append(ctx.refuse(det, "stair_width", f"stair geometry not recovered ({why})", "m"))
        sides = set()
        if ctx.label_mask(RAIL_LABELS)[ctx.box_mask(det, 0.3)].sum() > 50:
            sides.add("unknown side")
    # handrail / tactile evidence (reuses detection attributes from the vision engine)
    attr_rail = det.attributes.get("handrail_present")
    rail = bool(sides) or bool(attr_rail and attr_rail.value is True)
    out.append(ctx.scale_free(det, "handrail_present", rail, None, sigma=None, confidence=0.65 if rail else 0.4,
                              method="handrail detections (vision engine) + ADE20K railing/bannister pixels",
                              notes=[f"sides with railing evidence: {sorted(sides) or 'none'}"] +
                                    ([] if rail else ["not detected - verify; absence of detection is not proof"])))
    out.append(ctx.scale_free(det, "handrail_both_sides", {"left", "right"} <= sides, None, sigma=None,
                              confidence=0.45, method="railing evidence on both lateral sides of the stair"))
    tac = det.attributes.get("tactile_warning_surface")
    out.append(ctx.scale_free(det, "tactile_warning_surface", bool(tac and tac.value is True), None, sigma=None,
                              confidence=0.55 if tac and tac.value else 0.35,
                              method="tactile-paving detection adjacent to the stair (vision engine)",
                              notes=[] if tac and tac.value else ["not detected - verify on site"]))
    out.append(ctx.manual(det, "nosing_contrast_and_profile", "nosing projection/contrast needs close inspection"))
    return out


# ----------------------------------------------------------------------------- corridors / walkways


def analyze_path(ctx: SceneContext, det: Detection) -> list[Measurement]:
    from vision.measurement.backends import GROUND_LABELS
    H, W = ctx.rgb.shape[:2]
    region = ctx.box_mask(det, 0.0)
    floor = ctx.label_mask(GROUND_LABELS) & region
    obst = [d for d in ctx.detections if d.class_name in ("obstacle", "person")]
    occl = np.zeros_like(floor)
    for d in obst:
        floor &= ~ctx.box_mask(d, 0.0)
        occl |= ctx.box_mask(d, 0.06)
    if floor.sum() < ctx.cfg["analysis"]["min_mask_pixels"]:
        return [ctx.refuse(det, n, "walkable floor not segmented in this region", "m")
                for n in ("clear_width", "min_available_clearance")]
    Xf, _ = points_of(floor, ctx, 30000)
    pl = fit_plane(Xf, rel_thresh=0.02)
    if pl is None or pl.inlier_ratio < 0.5:
        return [ctx.refuse(det, n, "floor is not planar in depth", "m") for n in ("clear_width", "min_available_clearance")]
    if ctx.gravity and ctx.gravity.up is not None and angle_deg(pl.normal, ctx.gravity.up) > 20:
        return [ctx.refuse(det, n, "segmented floor is not horizontal (likely wrong surface)", "m")
                for n in ("clear_width", "min_available_clearance")]
    uv = mask_pixels(floor, 60000)
    Y = rays_on_plane(uv, ctx, pl)
    ok = np.isfinite(Y).all(1)
    Y, uv = Y[ok], uv[ok]
    # corridor axis = principal in-plane direction of the floor (usually away from the camera)
    Q = Y - Y.mean(0)
    Q -= np.outer(Q @ pl.normal, pl.normal)
    _, _, vt = np.linalg.svd(Q[::max(1, len(Q) // 5000)], full_matrices=False)
    ax = vt[0]
    if ax @ np.array([0, 0, 1.0]) < 0:
        ax = -ax
    lat = np.cross(pl.normal, ax)
    s, l = Y @ ax, Y @ lat
    s0, s1 = np.percentile(s, [3, 97])
    edges = np.linspace(s0, s1, 11)
    widths = []
    m = ctx.cfg["analysis"]["border_margin_px"]
    for i in range(10):
        sel = (s >= edges[i]) & (s < edges[i + 1])
        if sel.sum() < 40:
            continue
        lw, la, lb_ = robust_extent(l[sel], 1, 99)
        u_sel = uv[sel, 0]
        trunc = bool(u_sel.min() <= m or u_sel.max() >= W - 1 - m)
        # a floor boundary next to an object box is an occlusion edge, not a wall: lower bound only
        li, ri = int(np.argmin(l[sel])), int(np.argmax(l[sel]))
        for (bu, bv) in (uv[sel][li], uv[sel][ri]):
            if occl[min(H - 1, int(bv)), min(W - 1, int(bu))]:
                trunc = True
        smid = (edges[i] + edges[i + 1]) / 2
        base = Y.mean(0) - (Y.mean(0) @ ax) * ax - (Y.mean(0) @ lat) * lat
        A, B = base + smid * ax + la * lat, base + smid * ax + lb_ * lat
        widths.append(dict(w=lw, trunc=trunc, la=la, lb=lb_, s=smid, line=seg_line(ctx, A, B), z=float((A[2] + B[2]) / 2)))
    if not widths:
        return [ctx.refuse(det, n, "too few floor points per cross-section", "m")
                for n in ("clear_width", "min_available_clearance")]
    full = [w for w in widths if not w["trunc"]]
    notes = ["floor = ADE20K floor/road/sidewalk/path/... pixels inside the detection, minus objects",
             "width is between the visible floor boundaries; skirting, mats or mis-segmentation shift it"]
    out = []
    if full:
        wmin = min(full, key=lambda w: w["w"])
        out.append(ctx.length(det, "clear_width", wmin["w"], z=wmin["z"], geom_rel=0.05,
                              overlay=ctx.overlay(lines=[w["line"] for w in widths], label_at=None),
                              method="minimum floor width over 10 cross-sections along the corridor axis "
                                     "(ray-plane intersection with fitted floor plane)",
                              notes=notes + [f"{len(full)}/{len(widths)} cross-sections with both floor edges visible "
                                             f"(others truncated or occluded by objects/people)"]))
    else:
        wmin = min(widths, key=lambda w: w["w"])
        out.append(ctx.length(det, "clear_width", wmin["w"], z=wmin["z"], geom_rel=0.05, lower_bound=True,
                              overlay=ctx.overlay(lines=[w["line"] for w in widths]),
                              method="minimum floor width over cross-sections; every cross-section has a floor edge "
                                     "outside the image or hidden by objects/people, so this is a lower bound",
                              notes=notes))
    # obstacles standing on this floor
    clear = []
    perm = False
    for d in obst:
        x1, y1, x2, y2 = ctx.box(d)
        fy = min(H - 1, y2 - 1)
        foot = np.array([[x1, fy], [x2, fy], [(x1 + x2) / 2, fy]])
        if not region[int(fy), int(min(W - 1, (x1 + x2) / 2))]:
            continue
        F = rays_on_plane(foot, ctx, pl)
        if not np.isfinite(F).all():
            continue
        fs = F[2] @ ax
        cs = min(widths, key=lambda w: abs(w["s"] - fs))
        o0, o1 = sorted([F[0] @ lat, F[1] @ lat])
        gaps = [max(0.0, o0 - cs["la"]), max(0.0, cs["lb"] - o1)]
        free = max(gaps)
        transient = d.class_name == "person"
        perm |= not transient
        g_line = seg_line(ctx, F[0], F[1])
        clear.append(dict(free=free, det=d, z=float(F[2][2]), transient=transient, line=g_line, trunc=cs["trunc"]))
    if clear:
        worst = min([c for c in clear if not c["transient"]] or clear, key=lambda c: c["free"])
        out.append(ctx.length(det, "min_available_clearance", worst["free"], z=worst["z"], geom_rel=0.08,
                              lower_bound=worst["trunc"], overlay=ctx.overlay(lines=[worst["line"]]),
                              method="largest free lateral gap beside the obstacle footprint (box bottom edge on floor "
                                     "plane) in the nearest cross-section",
                              notes=[f"limiting object: {worst['det'].prompt or worst['det'].class_name} "
                                     f"({worst['det'].id}){' - transient person' if worst['transient'] else ''}",
                                     "footprint from the 2D box bottom edge: wide-based objects may be underestimated"]))
    out.append(ctx.scale_free(det, "permanent_obstruction_present", bool(perm), None, sigma=None,
                              confidence=0.55, method="non-person obstacle detections standing on the floor region",
                              notes=[f"{len(clear)} objects on this path ({sum(c['transient'] for c in clear)} people)"]))
    return out


ANALYZERS = {"door": analyze_door, "elevator": analyze_door, "ramp": analyze_ramp, "staircase": analyze_stairs,
             "corridor": analyze_path, "walkway": analyze_path}
