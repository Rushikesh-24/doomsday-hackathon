#!/usr/bin/env python
"""Measurement validation. Reports only errors that were actually computed.

  synthetic   exact analytic ground truth; isolates the geometric algorithms (depth scaled by a known
              wrong factor to emulate monocular scale error, then calibrated with a reference)
  diode       DIODE laser-scanned depth (Vasiljevic et al. 2019): the SAME detections/masks are measured
              with the laser depth (truth) and with the monocular network depth (uncalibrated, and
              calibrated with one reference taken from the laser geometry) -> depth-model-induced error
  manual      your tape-measured ground truth for real photos (the only true field validation):
              JSON [{"image": "path.jpg", "calibration": {...}, "ground_truth": [{"detection_id" | "class"+"index",
                     "name": "doorway_width", "value": 0.91}]}]

    python scripts/validate_measurements.py synthetic
    python scripts/validate_measurements.py diode --dir samples/validation/diode --limit 40
    python scripts/validate_measurements.py manual --file campus_gt.json
"""
import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("outputs/validation")


def stats(errs):
    e = np.asarray(errs, float)
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mean": round(float(e.mean()), 4), "mae": round(float(np.abs(e).mean()), 4),
            "median_abs": round(float(np.median(np.abs(e))), 4), "max_abs": round(float(np.abs(e).max()), 4)}


# ------------------------------------------------------------------ synthetic
def synthetic():
    from vision.measurement import synthetic as S
    from vision.measurement.engine import MeasurementEngine
    stairs = {"staircase": [f"riser{k}" for k in range(8)] + [f"tread{k}" for k in range(8)]}
    scenes = [
        ("ramp 1:12", S.ramp_scene, {"ramp": ["ramp"]}, {"slope_deg": 4.76}),
        ("ramp 1:8", S.ramp_scene, {"ramp": ["ramp"]}, {"slope_deg": 7.13}),
        ("ramp 1:20", S.ramp_scene, {"ramp": ["ramp"]}, {"slope_deg": 2.86, "width": 1.2}),
        ("door 0.9x2.1", S.door_scene, {"door": ["door"]}, {}),
        ("door 1.2x2.0 @6m", S.door_scene, {"door": ["door"]}, {"width": 1.2, "height": 2.0, "z": 6.0}),
        ("stairs 8x0.16/0.30", S.stairs_scene, stairs, {}),
        ("stairs 8x0.18/0.26", S.stairs_scene, stairs, {"riser": 0.18, "tread": 0.26}),
        ("corridor 1.8", S.corridor_scene, {"corridor": ["floor", "wl", "wr"]}, {}),
        ("corridor 1.2", S.corridor_scene, {"corridor": ["floor", "wl", "wr"]}, {"width": 1.2}),
    ]
    rows = []
    for label, fn, groups, kw in scenes:
        for scale_error in (0.75, 1.0, 1.35):
            png, z, res, fm, gt, sc = S.build(fn, groups, scale_error=scale_error, **kw)
            eng = MeasurementEngine(models=fm, vision_pipeline=object())
            for zr in (2.2, 3.0, 5.0):   # first visible ground reference
                A, B = sc.project(np.array([[-0.3, 0, zr], [0.3, 0, zr]]))
                if (A.min() >= 0 and B.min() >= 0 and A[0] < sc.w and B[0] < sc.w and A[1] < sc.h and B[1] < sc.h):
                    break
            cal = {"intrinsics": sc.K(), "references": [{"type": "segment", "points": [A.tolist(), B.tolist()],
                                                         "length_m": 0.6}]}
            rep = eng.analyze(png, f"{label}.png", detections=res.model_dump(), calibration=cal, save=False)
            for f in rep.facilities:
                for m in f.measurements:
                    if m.name in gt and isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
                        err = float(m.value) - gt[m.name]
                        rows.append({"scene": label, "scale_error": scale_error, "name": m.name, "unit": m.unit,
                                     "truth": round(gt[m.name], 4), "estimate": m.value, "error": round(err, 4),
                                     "rel_error": round(err / gt[m.name], 4) if gt[m.name] else None,
                                     "sigma": m.uncertainty,
                                     "within_95": None if not m.uncertainty else abs(err) <= 1.96 * m.uncertainty,
                                     "status": m.status.value})
                    elif m.name in gt:
                        rows.append({"scene": label, "scale_error": scale_error, "name": m.name, "truth": gt[m.name],
                                     "estimate": None, "status": m.status.value, "refusal": m.refusal_reason})
    by = {}
    for r in rows:
        if r.get("error") is not None:
            by.setdefault(r["name"], []).append(r)
    summary = {k: {**stats([x["error"] for x in v]), "rel_mae": round(float(np.mean([abs(x["rel_error"]) for x in v])), 4),
                   "within_95_rate": round(float(np.mean([x["within_95"] for x in v if x["within_95"] is not None])), 3)
                   if any(x["within_95"] is not None for x in v) else None}
               for k, v in by.items()}
    return {"kind": "synthetic", "rows": rows, "summary": summary,
            "refused": [r for r in rows if r.get("estimate") is None]}


# ------------------------------------------------------------------ DIODE
DIODE_K = {"fx": 886.81, "fy": 927.06, "cx": 512.0, "cy": 384.0, "focal_rel_sigma": 0.01}


def diode(directory: Path, limit: int):
    from vision.inference.pipeline import get_pipeline
    from vision.measurement.engine import get_engine
    eng = get_engine()
    pipe = get_pipeline()
    pngs = sorted(directory.rglob("*.png"))[:limit]
    rows, plane_check = [], []
    for p in pngs:
        dpath = p.with_name(p.stem + "_depth.npy")
        mpath = p.with_name(p.stem + "_depth_mask.npy")
        if not dpath.exists() or not mpath.exists():
            continue
        gt = np.load(dpath).squeeze().astype(np.float32)
        valid = np.load(mpath).astype(bool)
        if valid.mean() < 0.6:
            continue
        gt_filled = gt.copy()
        gt_filled[~valid] = np.median(gt[valid])
        det = pipe.analyze_image(p, save_dir=OUT / "diode_detections")
        if not any(d.class_name in ("door", "elevator", "ramp", "staircase", "corridor", "walkway") for d in det.detections):
            continue
        base = dict(detections=det.model_dump(), save=False)
        cal_gt = {"intrinsics": DIODE_K, "use_priors": False}
        rep_gt = eng.analyze(p, p.name, calibration=cal_gt, depth_override=gt_filled, **base)
        # laser depth is metric: mark it as calibrated by giving it a self-consistent scale (no correction)
        rep_net = eng.analyze(p, p.name, calibration=cal_gt, **base)
        truth = {}
        for f in rep_gt.facilities:
            for m in f.measurements:
                v = m.value if m.value is not None else m.provisional_value
                if isinstance(v, (int, float)) and not isinstance(v, bool) and m.status.value not in ("INSUFFICIENT_DATA",):
                    truth[(m.detection_id, m.name)] = (float(v), m.unit, m.status.value)
        # calibration reference for the network run: the first absolute length measured on laser depth
        ref = next(((k, v) for k, v in truth.items() if v[1] == "m" and k[1] in ("door_height", "doorway_width",
                                                                                     "clear_width", "ramp_length")), None)
        rep_cal = None
        if ref:
            (did, name), (val, _, _) = ref
            dim = "height" if name == "door_height" else "width"
            if name in ("door_height", "doorway_width"):
                cal = {"intrinsics": DIODE_K, "use_priors": False,
                       "references": [{"type": "object", "detection_id": did, "dimension": dim, "length_m": val}]}
                rep_cal = eng.analyze(p, p.name, calibration=cal, **base)
        for rep, mode in ((rep_net, "uncalibrated_network"), (rep_cal, "calibrated_network")):
            if rep is None:
                continue
            for f in rep.facilities:
                for m in f.measurements:
                    key = (m.detection_id, m.name)
                    if key not in truth:
                        continue
                    if mode == "calibrated_network" and ref and key == ref[0]:
                        continue  # the reference itself
                    v = m.value if m.value is not None else m.provisional_value
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    t = truth[key][0]
                    rows.append({"image": p.name, "mode": mode, "detection": m.detection_id, "class": m.facility_class,
                                 "name": m.name, "unit": m.unit, "laser_value": round(t, 4), "network_value": round(float(v), 4),
                                 "error": round(float(v) - t, 4), "rel_error": round((float(v) - t) / t, 4) if t else None,
                                 "status": m.status.value})
    by = {}
    for r in rows:
        by.setdefault((r["mode"], r["name"]), []).append(r)
    summary = {f"{k[0]}:{k[1]}": {**stats([x["error"] for x in v]),
                                   "rel_mae": round(float(np.mean([abs(x["rel_error"]) for x in v if x["rel_error"] is not None])), 4)}
               for k, v in by.items()}
    return {"kind": "diode", "images_used": len({r["image"] for r in rows}), "rows": rows, "summary": summary,
            "note": "truth = same pipeline on DIODE laser depth with DIODE intrinsics; errors isolate the monocular "
                    "depth model, not detection/segmentation"}


# ------------------------------------------------------------------ DIODE geometry (detection-independent)
def diode_geometry(directory: Path, limit: int, pairs: int = 300, depth_model: str | None = None):
    """Real photos + laser depth: measures the three error sources that matter for measurements,
    on the SAME pixels: (1) uncalibrated metric-scale error, (2) ground-plane orientation error
    (bounds slope error), (3) error of calibrated 3D lengths vs the depth ratio to the reference."""
    import cv2
    from vision.measurement.backends import GROUND_LABELS, get_measurement_models
    from vision.measurement.geometry import angle_deg, fit_plane
    mm = get_measurement_models()
    sem = mm.semantic()
    K = DIODE_K
    rows, length_rows = [], []
    rng = np.random.default_rng(0)
    for p in sorted(directory.rglob("*.png"))[:limit]:
        dp, mp = p.with_name(p.stem + "_depth.npy"), p.with_name(p.stem + "_depth_mask.npy")
        if not (dp.exists() and mp.exists()):
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        gt = np.load(dp).squeeze().astype(np.float32)
        valid = np.load(mp).astype(bool) & (gt > 0.3)
        seg = sem.predict(rgb)
        outdoor = sem.mask_of(seg, {"sky"}).mean() > 0.02
        key = depth_model or ("da2_metric_outdoor_small" if outdoor else "da2_metric_indoor_small")
        net = mm.depth(key).predict(rgb).depth
        H, W = gt.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        def pts(z):
            return np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], -1)
        Pg, Pn = pts(gt), pts(net)
        ratio = net[valid] / gt[valid]
        row = {"image": p.name, "depth_model": key, "valid_frac": round(float(valid.mean()), 3),
               "scale_ratio_net_over_laser": round(float(np.median(ratio)), 3),
               "scale_ratio_iqr": [round(float(x), 3) for x in np.percentile(ratio, [25, 75])]}
        gmask = sem.mask_of(seg, GROUND_LABELS) & valid
        if gmask.sum() > 3000:
            ys, xs = np.nonzero(gmask)
            sel = rng.choice(len(ys), min(20000, len(ys)), replace=False)
            pg = fit_plane(Pg[ys[sel], xs[sel]].astype(np.float64), rel_thresh=0.01)
            if pg is not None and pg.inlier_ratio > 0.5:
                inl = pg.inliers
                pn = fit_plane(Pn[ys[sel][inl], xs[sel][inl]].astype(np.float64), rel_thresh=0.02)
                if pn is not None:
                    row["ground_plane_angle_error_deg"] = round(angle_deg(pg.normal, pn.normal), 3)
                    # calibrated lengths: reference = one ground segment, then other random ground segments
                    yy, xx = ys[sel][inl], xs[sel][inl]
                    A = rng.integers(0, len(yy), pairs * 3)
                    B = rng.integers(0, len(yy), pairs * 3)
                    LA = np.linalg.norm(Pg[yy[A], xx[A]] - Pg[yy[B], xx[B]], axis=1)
                    ok = (LA > 0.5) & (LA < 8)
                    A, B, LA = A[ok][:pairs], B[ok][:pairs], LA[ok][:pairs]
                    if len(A) > 10:
                        LN = np.linalg.norm(Pn[yy[A], xx[A]] - Pn[yy[B], xx[B]], axis=1)
                        zs = (gt[yy[A], xx[A]] + gt[yy[B], xx[B]]) / 2
                        # (a) oracle global scale -> pure shape error of the network depth
                        s_or = float(np.median(gt[valid] / net[valid]))
                        # (b) realistic single reference: a 1-3 m ground segment near the median depth
                        cand = np.nonzero((LA > 1) & (LA < 3))[0]
                        r0 = cand[np.argmin(np.abs(zs[cand] - np.median(zs)))] if len(cand) else 0
                        s_ref = LA[r0] / LN[r0]
                        for i in range(len(A)):
                            if i == r0:
                                continue
                            length_rows.append({"image": p.name, "laser_m": float(LA[i]),
                                                "rel_error": float((s_ref * LN[i] - LA[i]) / LA[i]),
                                                "rel_error_oracle_scale": float((s_or * LN[i] - LA[i]) / LA[i]),
                                                "log_depth_ratio": float(abs(np.log(zs[i] / zs[r0]))),
                                                "endpoint_log_ratio": float(abs(np.log(gt[yy[A[i]], xx[A[i]]] /
                                                                                      gt[yy[B[i]], xx[B[i]]])))})
        rows.append(row)
    lr = np.array([[r["rel_error"], r["log_depth_ratio"]] for r in length_rows]) if length_rows else np.zeros((0, 2))
    bins = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 3.0)]
    by_ratio = {f"|ln z/zref| {a}-{b}": {"n": int(((lr[:, 1] >= a) & (lr[:, 1] < b)).sum()),
                                         "rel_mae": round(float(np.abs(lr[(lr[:, 1] >= a) & (lr[:, 1] < b), 0]).mean()), 4)
                                         if ((lr[:, 1] >= a) & (lr[:, 1] < b)).any() else None}
                for a, b in bins}
    sr = [r["scale_ratio_net_over_laser"] for r in rows]
    ga = [r["ground_plane_angle_error_deg"] for r in rows if "ground_plane_angle_error_deg" in r]
    lo = np.array([r["rel_error_oracle_scale"] for r in length_rows]) if length_rows else np.zeros(0)
    lat = [r for r in length_rows if r["endpoint_log_ratio"] < 0.05]
    dep = [r for r in length_rows if r["endpoint_log_ratio"] >= 0.3]
    summary = {"images": len(rows),
               "length_rel_error_oracle_global_scale": stats(lo),
               "lateral_lengths (endpoints at ~same depth)": {
                   "oracle_scale": stats([r["rel_error_oracle_scale"] for r in lat]),
                   "single_reference": stats([r["rel_error"] for r in lat])},
               "along_view_lengths (endpoint depth ratio > 1.35)": {
                   "oracle_scale": stats([r["rel_error_oracle_scale"] for r in dep]),
                   "single_reference": stats([r["rel_error"] for r in dep])},
               "uncalibrated_scale_ratio": {"median": round(float(np.median(sr)), 3) if sr else None,
                                            "min": min(sr) if sr else None, "max": max(sr) if sr else None,
                                            "mean_abs_log_error": round(float(np.mean(np.abs(np.log(sr)))), 3) if sr else None},
               "ground_plane_angle_error_deg": stats(ga),
               "calibrated_length_rel_error": {**(stats(lr[:, 0]) if len(lr) else {"n": 0}), "by_depth_ratio": by_ratio}}
    return {"kind": "diode_geometry", "rows": rows, "summary": summary,
            "note": "DIODE val subset (laser depth, official intrinsics); errors of the monocular depth model only"}


# ------------------------------------------------------------------ manual
def manual(file: Path):
    from vision.measurement.engine import get_engine
    eng = get_engine()
    rows = []
    for item in json.loads(file.read_text()):
        rep = eng.analyze(item["image"], calibration=item.get("calibration"))
        gts = []
        for g in item["ground_truth"]:
            if "detection_id" not in g:
                fs = sorted([f for f in rep.facilities if f.class_name == g["class"]], key=lambda f: -f.detection_confidence)
                if len(fs) <= g.get("index", 0):
                    rows.append({"image": item["image"], "name": g["name"], "error": None, "note": "facility not detected"})
                    continue
                g = {**g, "detection_id": fs[g.get("index", 0)].detection_id}
            gts.append(g)
        rep = eng.analyze(item["image"], calibration=item.get("calibration"), ground_truth=gts,
                          detections=json.loads(Path(rep.json_path).read_text()) if False else None,
                          vision_request_id=rep.vision_request_id)
        for f in rep.facilities:
            for m in f.measurements:
                if m.ground_truth:
                    rows.append({"image": item["image"], "name": m.name, "manual": m.ground_truth.manual_value,
                                 "engine": m.ground_truth.engine_value, "error": m.ground_truth.error,
                                 "rel_error": m.ground_truth.rel_error,
                                 "within_95": m.ground_truth.within_reported_uncertainty,
                                 "engine_status": next((n for n in m.notes if n.startswith("engine status")), None)})
    by = {}
    for r in rows:
        if r.get("error") is not None:
            by.setdefault(r["name"], []).append(r["error"])
    return {"kind": "manual", "rows": rows, "summary": {k: stats(v) for k, v in by.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "diode", "diode-geometry", "manual"])
    ap.add_argument("--dir", default="samples/validation/diode")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--file")
    ap.add_argument("--depth-model", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    res = (synthetic() if a.mode == "synthetic" else diode(Path(a.dir), a.limit) if a.mode == "diode"
           else diode_geometry(Path(a.dir), a.limit, depth_model=a.depth_model) if a.mode == "diode-geometry" else manual(Path(a.file)))
    res["created"] = datetime.now().isoformat(timespec="seconds")
    p = OUT / f"{a.mode}{'_' + a.depth_model if a.depth_model else ''}.json"
    p.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res["summary"], indent=2))
    if res.get("refused"):
        print("refused:", [(r["scene"], r["scale_error"], r["name"], r.get("refusal")) for r in res["refused"]])
    print("written", p)


if __name__ == "__main__":
    main()
