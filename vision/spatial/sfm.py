"""Structure-from-Motion stage (COLMAP via pycolmap, BSD-3).

Runs in a separate worker process (`python -m vision.spatial.sfm <frames_dir> <out_dir> [json-options]`)
because pycolmap and PyTorch ship different OpenMP runtimes that abort when loaded together.
Outputs only plain files so downstream stages never import pycolmap:
  sfm.json        attempts, per-model stats, primary model id, failure reasons
  models/<k>/poses.json   camera intrinsics + world->camera (R, t) per registered frame
  models/<k>/points.npz   xyz, rgb, error, track length, and per-image observations (for registration)
  sparse_points.ply       primary model point cloud (downloadable)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def _write_points_ply(path: Path, xyz, rgb) -> None:
    import numpy as np
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def _frame_index(name: str) -> int | None:
    import re
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def temporal_check(rec, jump_factor: float = 5.0) -> dict:
    """Video frames are ordered in time, so consecutive registered frames must be close in space. A camera
    that 'teleports' away and back (> jump_factor x the median step on both sides) has an aliased pose
    (typical on repetitive structures such as staircases matched across different flights)."""
    import numpy as np
    ims = [(i.name, i) for i in rec.images.values() if i.has_pose]
    ims = [(n, i) for n, i in ims if _frame_index(n) is not None]
    ims.sort(key=lambda x: _frame_index(x[0]))
    if len(ims) < 4:
        return {"suspect_frames": [], "jumps": 0, "consistent": len(ims)}
    C = np.array([np.asarray(i.projection_center()) for _, i in ims])
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    med = float(np.median(step)) or 1e-9
    big = step > jump_factor * med
    suspects = set()
    for k in range(1, len(ims) - 1):
        if big[k - 1] and big[k]:          # far from both temporal neighbours
            suspects.add(ims[k][0])
    # a jump with no single culprit: flag the shorter side of the split
    for k in np.nonzero(big)[0]:
        if ims[k][0] not in suspects and ims[k + 1][0] not in suspects:
            suspects.add(ims[k + 1][0])
    return {"suspect_frames": sorted(suspects), "jumps": int(big.sum()), "consistent": len(ims) - len(suspects),
            "median_step": med}


def _export(rec, out: Path) -> dict:
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    (out / "colmap").mkdir(exist_ok=True)
    rec.write(str(out / "colmap"))          # standard COLMAP binary model (for external splat tools)
    cam = next(iter(rec.cameras.values()))
    model = str(cam.model).split(".")[-1]
    imgs = []
    obs = {}   # image name -> list of (x, y, point3D_id)
    for im in rec.images.values():
        if not im.has_pose:
            continue
        p = im.cam_from_world()
        imgs.append({"name": im.name, "image_id": int(im.image_id), "R": p.rotation.matrix().tolist(),
                     "t": [float(v) for v in p.translation],
                     "center": [float(v) for v in im.projection_center()]})
        o = []
        for p2 in im.points2D:
            if p2.has_point3D():
                o.append((float(p2.xy[0]), float(p2.xy[1]), int(p2.point3D_id)))
        obs[im.name] = o
    pids = list(rec.points3D.keys())
    pts = [rec.points3D[i] for i in pids]
    xyz = np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
    err = np.array([p.error for p in pts], dtype=np.float32)
    trk = np.array([p.track.length() for p in pts], dtype=np.int32)
    obs_flat = {k: np.array(v, dtype=np.float64).reshape(-1, 3) for k, v in obs.items()}
    np.savez_compressed(out / "points.npz", ids=np.array(pids, dtype=np.int64), xyz=xyz, rgb=rgb, error=err,
                        track_len=trk, **{f"obs::{k}": v for k, v in obs_flat.items()})
    poses = {"camera": {"model": model, "params": [float(v) for v in cam.params], "width": cam.width,
                        "height": cam.height},
             "convention": "x_cam = R @ x_world + t (COLMAP cam_from_world); camera looks along +z, y down",
             "images": sorted(imgs, key=lambda d: d["name"])}
    (out / "poses.json").write_text(json.dumps(poses, indent=1))
    return {"registered": len(imgs), "points": len(pids), "camera": poses["camera"],
            "mean_reprojection_error_px": round(float(rec.compute_mean_reprojection_error()), 3),
            "mean_track_length": round(float(trk.mean()) if len(trk) else 0, 2),
            "mean_observations_per_image": round(float(np.mean([len(v) for v in obs.values()])) if obs else 0, 1)}


def run_sfm(frames_dir: Path, out_dir: Path, opts: dict) -> dict:
    import pycolmap
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in frames_dir.glob("*.jpg"))
    report = {"frames": len(names), "attempts": [], "models": {}, "primary_model": None, "status": "failed",
              "tool": f"pycolmap {pycolmap.__version__} (COLMAP, BSD-3)", "reasons": []}
    if len(names) < 3:
        report["reasons"].append(f"only {len(names)} frames: need >= 3 (preferably 30+)")
        return report
    attempts = [
        {"name": "sequential", "matcher": "sequential", "overlap": opts.get("overlap", 12), "init_tri": 4.0, "init_inl": 50},
        {"name": "exhaustive_relaxed", "matcher": "exhaustive", "init_tri": 2.0, "init_inl": 30},
    ]
    if len(names) > opts.get("exhaustive_max", 180):
        attempts[1]["matcher"] = "sequential"
        attempts[1]["overlap"] = 25
    candidates = []   # (consistent_frames, -jumps, attempt, key, rec, check)
    for a in attempts:
        t0 = time.time()
        work = out_dir / f"attempt_{a['name']}"
        work.mkdir(parents=True, exist_ok=True)
        db = work / "database.db"
        if db.exists():
            db.unlink()
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = opts.get("camera_model", "SIMPLE_RADIAL")
        if opts.get("camera_params"):          # user calibration, e.g. "f,cx,cy,k" for SIMPLE_RADIAL
            ro.camera_params = opts["camera_params"]
        eo = pycolmap.FeatureExtractionOptions()
        eo.max_image_size = opts.get("max_image_size", 1600)
        eo.use_gpu = False
        pycolmap.extract_features(db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                                  extraction_options=eo)
        mo = pycolmap.FeatureMatchingOptions()
        mo.use_gpu = False
        if a["matcher"] == "sequential":
            po = pycolmap.SequentialPairingOptions()
            po.overlap = a["overlap"]
            po.quadratic_overlap = True
            pycolmap.match_sequential(db, matching_options=mo, pairing_options=po)
        else:
            pycolmap.match_exhaustive(db, matching_options=mo)
        io = pycolmap.IncrementalPipelineOptions()
        io.mapper.init_min_tri_angle = a["init_tri"]
        io.mapper.init_min_num_inliers = a["init_inl"]
        io.min_model_size = 5
        io.max_runtime_seconds = opts.get("max_runtime_s", 1800)
        if opts.get("camera_params") and opts.get("fix_intrinsics", True):
            io.ba_refine_focal_length = False
            io.ba_refine_extra_params = False
        sparse = work / "sparse"
        sparse.mkdir(exist_ok=True)
        recs = pycolmap.incremental_mapping(db, frames_dir, sparse, options=io)
        checks = {int(k): temporal_check(r) for k, r in recs.items()}
        report["attempts"].append({"attempt": a["name"], "seconds": round(time.time() - t0, 1),
                                   "models": {int(k): r.num_reg_images() for k, r in recs.items()},
                                   "temporal_jumps": {k: c["jumps"] for k, c in checks.items()},
                                   "consistent_frames": {k: c["consistent"] for k, c in checks.items()}})
        for k, r in recs.items():
            c = checks[int(k)]
            candidates.append((c["consistent"], -c["jumps"], a["name"], int(k), r, c))
        best_now = max((c[0] for c in candidates), default=0)
        if best_now >= opts.get("good_enough_fraction", 0.6) * len(names):
            break
    best = None
    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        top = candidates[0]
        best = (top[2], [c for c in candidates if c[2] == top[2]])
    if best is None:
        report["reasons"].append("COLMAP could not initialise any model: too little texture, too little parallax "
                                 "(camera only rotating), or too little overlap between frames")
        return report
    name, group = best
    group.sort(key=lambda c: (c[0], c[1]), reverse=True)
    for rank, (_, _, _, k, rec, chk) in enumerate(group):
        info = _export(rec, out_dir / "models" / str(rank))
        info["attempt"] = name
        info["temporal_check"] = chk
        (out_dir / "models" / str(rank) / "suspect_frames.json").write_text(json.dumps(chk["suspect_frames"]))
        report["models"][str(rank)] = info
    report["primary_model"] = "0"
    prim = report["models"]["0"]
    report["registration_rate"] = round(prim["registered"] / len(names), 3)
    report["consistent_rate"] = round(prim["temporal_check"]["consistent"] / len(names), 3)
    if prim["temporal_check"]["suspect_frames"]:
        report["reasons"].append(f"{len(prim['temporal_check']['suspect_frames'])} frames have temporally inconsistent "
                                 f"('teleporting') poses - likely aliasing on repetitive structure; they are excluded "
                                 f"from splat training and facility registration")
    reg_all = sum(m["registered"] for m in report["models"].values())
    report["registered_any_model"] = reg_all
    report["unregistered_frames"] = len(names) - reg_all
    report["status"] = "ok" if report["consistent_rate"] >= 0.3 else "partial"
    if len(report["models"]) > 1:
        report["reasons"].append(f"reconstruction split into {len(report['models'])} disconnected models (each has its "
                                 f"own coordinate frame); facilities are only placed in the model of their frames")
    if report["status"] == "partial":
        report["reasons"].append(f"only {report['registration_rate']:.0%} of frames registered in the primary model")
    import numpy as np
    d = np.load(out_dir / "models" / "0" / "points.npz")
    _write_points_ply(out_dir / "sparse_points.ply", d["xyz"], d["rgb"])
    return report


def run_in_subprocess(frames_dir: Path, out_dir: Path, opts: dict, timeout: float = 7200,
                      log_path: Path | None = None) -> dict:
    """Launch the worker; returns sfm.json content (status=failed with reason on crash/timeout)."""
    cmd = [sys.executable, "-m", "vision.spatial.sfm", str(frames_dir), str(out_dir), json.dumps(opts)]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_path or out_dir / "sfm.log", "w")
    try:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                           cwd=str(Path(__file__).resolve().parents[2]))
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        log.close()
    p = out_dir / "sfm.json"
    if rc == 0 and p.exists():
        return json.loads(p.read_text())
    return {"status": "failed", "reasons": [f"SfM worker exited with {rc}; see sfm.log"], "models": {},
            "primary_model": None}


if __name__ == "__main__":
    frames, out = Path(sys.argv[1]), Path(sys.argv[2])
    options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    rep = run_sfm(frames, out, options)
    (out / "sfm.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("status", "primary_model", "reasons")}))
