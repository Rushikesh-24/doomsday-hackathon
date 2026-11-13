"""Resumable spatial pipeline:
quality -> frames -> tracking -> sfm -> reconstruction -> alignment -> registration -> export

Every stage writes its artifacts into the job directory and records its status in state.json. A re-run
skips stages that are `done` (unless forced), so failed stages can be retried and interrupted jobs
resumed without repeating successful work. Heavy native work (COLMAP, splat training) runs in worker
processes. A stage whose prerequisite failed is marked `blocked`, never faked.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision.spatial import frames as F
from vision.spatial import quality as Q

log = logging.getLogger(__name__)
STAGES = ["quality", "frames", "tracking", "sfm", "reconstruction", "alignment", "registration", "export"]
DEPENDS = {"frames": ["quality"], "tracking": ["frames"], "sfm": ["frames"], "reconstruction": ["sfm"],
           "alignment": ["sfm"], "registration": ["alignment", "tracking"], "export": []}
DEFAULTS = {"preset": "fast", "tracker": "iou_centre", "vision_model": None, "measure_facilities": True,
            "max_measured_facilities": 6, "splat_backend": "auto", "references": [], "markers": [], "gcps": [],
            "camera_height_m": 1.40, "sfm": {}, "ground_truth": []}


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w,x,y,z) quaternions, broadcasting over rows."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SpatialPipeline:
    def __init__(self, job_dir: Path, video: Path, options: dict | None = None, on_progress=None, video_id: str = "vid"):
        self.dir = job_dir
        self.video = video
        self.video_id = video_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.opts = {**DEFAULTS, **(options or {})}
        self.on_progress = on_progress
        sp = self.dir / "state.json"
        self.state = json.loads(sp.read_text()) if sp.exists() else {
            "stages": {s: {"status": "pending"} for s in STAGES}, "created": _now(), "options": self.opts}
        self.state["options"] = self.opts
        self._save()

    # ------------------------------------------------------------------ state
    def _save(self):
        (self.dir / "state.json").write_text(json.dumps(self.state, indent=2, default=str))

    def _set(self, stage, **kw):
        self.state["stages"][stage].update(kw)
        self._save()

    def progress(self) -> dict:
        st = self.state["stages"]
        done = sum(1 for s in STAGES if st[s]["status"] in ("done", "failed", "blocked", "skipped"))
        cur = next((s for s in STAGES if st[s]["status"] == "running"), None)
        frac = st[cur].get("fraction", 0.0) if cur else 0.0
        return {"stage": cur, "overall": round((done + frac) / len(STAGES), 4),
                "stages": {s: {k: st[s].get(k) for k in ("status", "fraction", "message", "seconds", "error")}
                           for s in STAGES}}

    def _report(self, stage, frac, msg):
        self._set(stage, fraction=round(float(frac), 4), message=msg)
        if self.on_progress:
            self.on_progress(self.progress())

    # ------------------------------------------------------------------ run
    def run(self, force: set[str] | None = None, only: list[str] | None = None) -> dict:
        import os
        (self.dir / "RUNNING").write_text(str(os.getpid()))   # liveness marker for interrupted-job detection
        try:
            return self._run(force, only)
        finally:
            (self.dir / "RUNNING").unlink(missing_ok=True)

    def _run(self, force, only) -> dict:
        force = set(force or ())
        for s in force & set(STAGES):          # forced stages restart from scratch (truthful progress)
            self.state["stages"][s] = {"status": "pending"}
        self._save()
        stop_after = self.opts.get("stop_after")
        for stage in STAGES:
            if only and stage not in only:
                continue
            if stop_after and STAGES.index(stage) > STAGES.index(stop_after):
                if self.state["stages"][stage]["status"] == "pending":
                    self._set(stage, status="skipped", error=f"stop_after={stop_after}")
                continue
            st = self.state["stages"][stage]
            if st["status"] == "done" and stage not in force:
                continue
            bad = [d for d in DEPENDS.get(stage, []) if self.state["stages"][d]["status"] != "done"]
            if bad and stage != "export":
                self._set(stage, status="blocked", error=f"prerequisite stage(s) not done: {bad}")
                continue
            t0 = time.time()
            self._set(stage, status="running", started=_now(), fraction=0.0, error=None, message=None)
            try:
                out = getattr(self, f"_stage_{stage}")()
                self._set(stage, status=out.pop("_status", "done"), seconds=round(time.time() - t0, 1),
                          finished=_now(), fraction=1.0, summary=out)
            except Exception as e:
                log.exception("stage %s failed", stage)
                self._set(stage, status="failed", seconds=round(time.time() - t0, 1), finished=_now(),
                          error=f"{e.__class__.__name__}: {e}", traceback=traceback.format_exc()[-2000:])
            if self.on_progress:
                self.on_progress(self.progress())
        statuses = {s: self.state["stages"][s]["status"] for s in STAGES}
        ok = all(v in ("done", "skipped") for v in statuses.values())
        self.state["status"] = ("completed" if ok else
                                "completed_with_failures" if statuses["export"] == "done" else "failed")
        self._save()
        return self.state

    # ------------------------------------------------------------------ stages
    def _stage_quality(self):
        rep = Q.analyze_quality(self.video, progress=lambda f, m: self._report("quality", f, m))
        d = rep.__dict__.copy()
        (self.dir / "quality_frames.json").write_text(json.dumps(d.pop("frames")))
        (self.dir / "quality.json").write_text(json.dumps(d, indent=2, default=str))
        return {"suitability": rep.suitability, "score": rep.suitability_score, "issues": [i["code"] for i in rep.issues]}

    def _stage_frames(self):
        q = json.loads((self.dir / "quality.json").read_text())
        frames = json.loads((self.dir / "quality_frames.json").read_text())
        cap = F.PRESET_KEYFRAMES.get(self.opts["preset"], 160)
        picks = F.select_keyframes(frames, max_keyframes=cap)
        if len(picks) < 3:
            raise ValueError(f"only {len(picks)} usable keyframes")
        rows = F.extract_frames(self.video, picks, self.dir / "frames", crop=q["video"].get("letterbox_active_area"))
        self._contact_sheet(rows)
        return {"keyframes": len(rows), "forced_imperfect": sum(r["forced_imperfect"] for r in rows),
                "cropped_letterbox": bool(q["video"].get("letterbox_active_area"))}

    def _contact_sheet(self, rows, n=12):
        pick = rows[:: max(1, len(rows) // n)][:n]
        tiles = []
        for r in pick:
            im = cv2.imread(str(self.dir / "frames" / r["name"]))
            s = 200 / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * s), 200))
            cv2.putText(im, f"t={r['t']:.1f}s", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(im)
        rows_img = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        w = max(r.shape[1] for r in rows_img)
        rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows_img]
        cv2.imwrite(str(self.dir / "frames_preview.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _stage_tracking(self):
        from vision.spatial.tracking import detect_frames, track
        rows = json.loads((self.dir / "frames.json").read_text())
        res = detect_frames(self.dir / "frames", rows, self.dir / "detections", model=self.opts["vision_model"],
                            progress=lambda f, m: self._report("tracking", f * 0.95, m))
        q = json.loads((self.dir / "quality.json").read_text())
        kf_fps = len(rows) / max(q["video"].get("duration_s") or 1, 1e-3)
        out = track(res, rows, method=self.opts["tracker"], fps=kf_fps)
        if self.opts.get("compare_trackers", True):
            other = "bytetrack" if self.opts["tracker"] != "bytetrack" else "iou_centre"
            try:
                o = track(res, rows, method=other, fps=kf_fps)
                out["comparison"] = {other: {"confirmed_tracks": o["confirmed_tracks"],
                                             "per_class_confirmed": o["per_class_confirmed"]}}
            except Exception as e:
                out["comparison"] = {other: {"error": str(e)}}
        (self.dir / "tracks.json").write_text(json.dumps(out, indent=1))
        return {"detections": out["detections"], "confirmed_tracks": out["confirmed_tracks"],
                "per_class": out["per_class_confirmed"], "comparison": out.get("comparison")}

    def _stage_sfm(self):
        from vision.spatial.sfm import run_in_subprocess
        self._report("sfm", 0.05, "COLMAP feature extraction / matching / mapping (worker process)")
        rep = run_in_subprocess(self.dir / "frames", self.dir / "sfm", self.opts.get("sfm", {}))
        if rep.get("status") == "failed" or not rep.get("primary_model"):
            raise RuntimeError("SfM failed: " + "; ".join(rep.get("reasons", [])))
        prim = rep["models"]["0"]
        return {"_status": "done", "status": rep["status"], "registered": prim["registered"], "frames": rep["frames"],
                "points": prim["points"], "models": len(rep["models"]), "reasons": rep.get("reasons")}

    def _stage_reconstruction(self):
        for stale in ("progress.json", "splat.json", "splat.ply", "splat_local.ply"):
            (self.dir / "splat" / stale).unlink(missing_ok=True)      # never report a previous run's output
        cmd = [sys.executable, "-m", "vision.spatial.splat.run", str(self.dir), self.opts["preset"],
               self.opts["splat_backend"]]
        log_f = open(self.dir / "splat.log", "w")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[2]))
        prog = self.dir / "splat" / "progress.json"
        while p.poll() is None:
            time.sleep(3)
            if prog.exists():
                try:
                    j = json.loads(prog.read_text())
                    self._report("reconstruction", j["fraction"], j["message"])
                except Exception:
                    pass
        log_f.close()
        sj = self.dir / "splat" / "splat.json"
        info = json.loads(sj.read_text()) if sj.exists() else {"status": "failed", "error": f"worker exit {p.returncode}"}
        if info.get("status") != "done":
            raise RuntimeError(f"Gaussian splat reconstruction failed: {info.get('error')}")
        return {k: info.get(k) for k in ("num_gaussians", "iterations", "seconds", "device", "test_psnr", "test_ssim",
                                         "train_psnr", "cpu_fallback", "backend")}

    def _mono_scale(self, model, max_frames=8) -> dict | None:
        """Metres per SfM unit from monocular metric depth at sparse points (approximate, +-45% measured)."""
        try:
            from vision.measurement.backends import get_measurement_models
            mm = get_measurement_models()
            sem = mm.semantic()
        except Exception as e:
            return {"used": False, "reason": f"depth model unavailable: {e}"}
        ratios = []
        names = sorted(model.images)[:: max(1, len(model.images) // max_frames)][:max_frames]
        for n in names:
            bgr = cv2.imread(str(self.dir / "frames" / n))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            seg = sem.predict(cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2)))
            key = "da2_metric_outdoor_small" if sem.mask_of(seg, {"sky"}).mean() > 0.02 else "da2_metric_indoor_small"
            dep = mm.depth(key).predict(rgb).depth
            uv, X = model.points_in_frame(n)
            if len(uv) < 20:
                continue
            R, t = model.pose(n)
            z = (X @ R.T + t)[:, 2]
            u = np.clip(uv[:, 0].astype(int), 0, dep.shape[1] - 1)
            v = np.clip(uv[:, 1].astype(int), 0, dep.shape[0] - 1)
            ok = z > 0
            ratios.append(float(np.median(dep[v[ok], u[ok]] / z[ok])))
        if not ratios:
            return {"used": False, "reason": "no frames with enough sparse points"}
        return {"used": True, "scale_m_per_unit": float(np.median(ratios)), "rel_sigma": 0.45,
                "per_frame": [round(r, 4) for r in ratios],
                "note": "monocular metric depth (DA-V2) vs SfM depth; validated scale error ~45% (docs/MEASUREMENT.md)"}

    def _stage_alignment(self):
        from vision.spatial.alignment import SfmModel, align, apply
        model = SfmModel(self.dir / "sfm/models/0")
        mono = self._mono_scale(model) if self.opts.get("monocular_scale", True) else None
        res = align(model, self.dir / "frames", self.opts, mono_scale=mono)
        (self.dir / "alignment.json").write_text(json.dumps(res, indent=2, default=str))
        # trajectory (both frames)
        fr = {r["name"]: r for r in json.loads((self.dir / "frames.json").read_text())}
        traj = []
        T = np.array(res["T_local_from_sfm"])
        Rl = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
        for n in sorted(model.images, key=lambda n: fr.get(n, {}).get("t", 0)):
            R, t = model.pose(n)
            c = model.center(n)
            Rcw = Rl @ R.T                                 # camera axes in local frame
            traj.append({"frame": n, "t": fr.get(n, {}).get("t"), "center_sfm": c.round(6).tolist(),
                         "center_local": apply(T, c)[0].round(4).tolist(),
                         "forward_local": (Rcw @ np.array([0, 0, 1.0])).round(4).tolist(),
                         "up_local": (Rcw @ np.array([0, -1.0, 0])).round(4).tolist(),
                         "R_world_to_cam_sfm": R.round(6).tolist(), "t_world_to_cam_sfm": t.round(6).tolist()})
        units = "m" if res["scale_status"] != "relative" else "sfm_units"
        P = np.array([x["center_local"] for x in traj])
        length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        (self.dir / "trajectory.json").write_text(json.dumps({"units": units, "scale_status": res["scale_status"],
                                                              "length": round(length, 3), "cameras": traj}, indent=1))
        self._plot_trajectory(P, model, T, units, res["scale_status"])
        return {"scale_status": res["scale_status"], "metres_per_sfm_unit": res["metres_per_sfm_unit"],
                "scale_rel_sigma": res["scale_rel_sigma"], "gravity_method": res["gravity"]["method"],
                "trajectory_length": round(length, 3), "trajectory_units": units, "georeferenced": res["georeferenced"]}

    def _plot_trajectory(self, P, model, T, units, status):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from vision.spatial.alignment import apply
        pts = apply(T, model.xyz[model.track_len >= 3])
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for a, (i, j, lab) in zip(ax, ((0, 2, "top view (X-Z)"), (2, 1, "side view (Z-Y, Y up)"))):
            a.scatter(pts[:, i], pts[:, j], s=0.3, c=model.rgb[model.track_len >= 3] / 255, alpha=0.5)
            a.plot(P[:, i], P[:, j], "r-", lw=1.5)
            a.scatter(P[:1, i], P[:1, j], c="lime", s=40, label="start", zorder=3)
            a.scatter(P[-1:, i], P[-1:, j], c="k", s=40, label="end", zorder=3)
            a.set_title(lab)
            a.set_aspect("equal")
            a.set_xlabel(units)
            lo, hi = np.percentile(pts[:, [i, j]], [2, 98], 0)
            lo, hi = np.minimum(lo, P[:, [i, j]].min(0)), np.maximum(hi, P[:, [i, j]].max(0))   # whole path visible
            pad = (hi - lo) * 0.1
            a.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            a.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax[0].legend()
        fig.suptitle(f"Camera trajectory ({len(P)} registered frames) - scale: {status}")
        fig.tight_layout()
        fig.savefig(self.dir / "trajectory.png", dpi=110)
        plt.close(fig)

    def _stage_registration(self):
        from vision.spatial.alignment import SfmModel
        from vision.spatial.registration import register
        model = SfmModel(self.dir / "sfm/models/0")
        al = json.loads((self.dir / "alignment.json").read_text())
        tracks = json.loads((self.dir / "tracks.json").read_text())["tracks"]
        res = register(model, tracks, al["T_local_from_sfm"], al["scale_status"], self.video_id)
        if self.opts.get("measure_facilities"):
            self._measure(res, model, al)
        (self.dir / "facilities.json").write_text(json.dumps(res, indent=1, default=str))
        return {k: res[k] for k in ("facilities_total", "facilities_registered", "facilities_unregistered")}

    def _measure(self, res, model, al):
        """Run the measurement engine on each measurable facility's best frame, reusing its detections."""
        from vision.measurement.engine import get_engine
        from vision.schemas import ImageResult
        eng = get_engine()
        n = 0
        for f in sorted(res["facilities"], key=lambda f: -(f.get("registration_confidence") or 0)):
            if f["class_name"] not in ("staircase", "ramp", "door", "elevator", "corridor", "walkway"):
                continue
            if n >= self.opts["max_measured_facilities"]:
                break
            best = max(f["observations"], key=lambda o: o["confidence"])
            dpath = self.dir / "detections" / (Path(best["frame"]).stem + ".json")
            dets = ImageResult.model_validate_json(dpath.read_text())
            override = None
            if al["scale_status"] in ("scale_calibrated", "georeferenced") and best["frame"] in model.images:
                override = self._frame_depth_scale(model, best["frame"], al)
            try:
                rep = eng.analyze(self.dir / "frames" / best["frame"], best["frame"], detections=dets,
                                  detection_ids=[best["detection_id"]], scale_override=override,
                                  calibration={"use_priors": False},
                                  save_dir=self.dir / "measurements" / f["facility_id"])
                fac = rep.facilities[0] if rep.facilities else None
                f["measurement_report"] = str(Path(rep.json_path).relative_to(self.dir))
                f["measurements"] = [{"name": m.name, "value": m.value, "unit": m.unit, "uncertainty": m.uncertainty,
                                      "status": m.status.value, "provisional_value": m.provisional_value,
                                      "measurement_id": m.id} for m in (fac.measurements if fac else [])]
                f["measurement_scale"] = rep.scale.status.value
            except Exception as e:
                f["measurement_error"] = f"{e.__class__.__name__}: {e}"
            n += 1

    def _frame_depth_scale(self, model, name, al) -> dict | None:
        """Scale for the measurement engine from the METRIC reconstruction: network depth vs metric SfM depth."""
        try:
            from vision.measurement.backends import get_measurement_models
            mm = get_measurement_models()
            bgr = cv2.imread(str(self.dir / "frames" / name))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dep = mm.depth("da2_metric_indoor_small").predict(rgb).depth
        except Exception:
            return None
        uv, X = model.points_in_frame(name)
        R, t = model.pose(name)
        z = (X @ R.T + t)[:, 2] * al["metres_per_sfm_unit"]
        u = np.clip(uv[:, 0].astype(int), 0, dep.shape[1] - 1)
        v = np.clip(uv[:, 1].astype(int), 0, dep.shape[0] - 1)
        r = z / dep[v, u]
        if len(r) < 20:
            return None
        s = float(np.median(r))
        spread = float(np.median(np.abs(np.log(r / s))))
        return {"scale": s, "rel_sigma": math.hypot(al["scale_rel_sigma"] or 0.05, spread),
                "description": "from the scale-calibrated SfM reconstruction (sparse points in this frame)",
                "reference_depth_m": float(np.median(z))}

    def _stage_export(self):
        assets = {}
        def add(name, rel, desc):
            p = self.dir / rel
            if p.exists():
                assets[name] = {"path": rel, "bytes": p.stat().st_size, "description": desc}
        if self.state["stages"]["reconstruction"]["status"] == "done" and (self.dir / "alignment.json").exists():
            self._transform_splat()
        add("splat_ply", "splat/splat.ply", "3D Gaussian splat (3DGS PLY, SfM frame)")
        add("splat_local_ply", "splat/splat_local.ply", "3D Gaussian splat in the gravity-aligned local frame "
            "(base colour only: higher-order SH are not rotated; use splat_ply for view-dependent colour)")
        add("splat_preview", "splat/preview_gt_vs_render.jpg", "held-out frames: ground truth (left) vs render (right)")
        add("splat_pytorch_ply", "splat_pytorch/splat.ply", "earlier in-house PyTorch splat kept for comparison")
        add("sparse_points_ply", "sfm/sparse_points.ply", "COLMAP sparse point cloud (SfM frame)")
        add("colmap_model", "sfm/models/0/colmap/cameras.bin", "COLMAP binary model (cameras/images/points3D .bin)")
        add("trajectory_json", "trajectory.json", "camera poses per keyframe (SfM + local frames)")
        add("trajectory_png", "trajectory.png", "trajectory plot")
        add("facilities_json", "facilities.json", "facility-to-3D registration")
        add("tracks_json", "tracks.json", "2D facility tracks across keyframes")
        add("quality_json", "quality.json", "capture quality report")
        add("frames_preview", "frames_preview.jpg", "extracted keyframe examples")
        add("alignment_json", "alignment.json", "scale + coordinate transforms")
        meta = self._evaluation()
        (self.dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        add("metadata_json", "metadata.json", "reconstruction metadata & quality evaluation")
        (self.dir / "assets.json").write_text(json.dumps(assets, indent=1))
        return {"assets": list(assets)}

    def _transform_splat(self):
        from vision.spatial.splat.trainer import load_ply
        al = json.loads((self.dir / "alignment.json").read_text())
        T = np.array(al["T_local_from_sfm"])
        s = float(np.cbrt(np.linalg.det(T[:3, :3])))
        Rm = T[:3, :3] / s
        p = load_ply(self.dir / "splat/splat.ply")
        xyz = np.stack([p["x"], p["y"], p["z"]], 1) @ T[:3, :3].T + T[:3, 3]
        q = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1)        # wxyz (3DGS convention)
        qn = quat_mul(mat_to_quat(Rm)[None], q)                                  # rotate each Gaussian
        props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
                 "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        n = len(xyz)
        data = np.concatenate([xyz, np.zeros((n, 3)), np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1),
                               p["opacity"][:, None],
                               np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1) + math.log(s),
                               qn], 1).astype(np.float32)
        header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + \
                 "".join(f"property float {k}\n" for k in props) + "end_header\n"
        with open(self.dir / "splat/splat_local.ply", "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())

    def _evaluation(self) -> dict:
        st = self.state["stages"]
        rd = lambda p: json.loads((self.dir / p).read_text()) if (self.dir / p).exists() else None
        q, sfm, sp, al, fac = rd("quality.json"), rd("sfm/sfm.json"), rd("splat/splat.json"), rd("alignment.json"), rd("facilities.json")
        frames = rd("frames.json") or []
        ev = {"video": (q or {}).get("video"), "capture_quality": {k: (q or {}).get(k) for k in ("suitability",
                                                                                                  "suitability_score", "stats")},
              "stage_seconds": {s: st[s].get("seconds") for s in STAGES},
              "stage_status": {s: st[s]["status"] for s in STAGES}, "keyframes": len(frames)}
        if sfm and sfm.get("models"):
            prim = sfm["models"]["0"]
            names = {f["name"] for f in frames}
            reg = {im["name"] for im in json.loads((self.dir / "sfm/models/0/poses.json").read_text())["images"]}
            ts = sorted(f["t"] for f in frames if f["name"] in reg)
            span = (max(f["t"] for f in frames) - min(f["t"] for f in frames)) if frames else 0
            model_pts = np.load(self.dir / "sfm/models/0/points.npz")["xyz"]
            lo, hi = np.percentile(model_pts, [2, 98], 0)
            vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
            ev["registration"] = {
                "frames": len(frames), "registered_primary": prim["registered"],
                "pose_completeness": round(prim["registered"] / max(len(frames), 1), 3),
                "registered_any_model": sfm.get("registered_any_model"), "models": len(sfm["models"]),
                "unregistered_frames": sorted(names - reg)[:500],
                "time_coverage": round((ts[-1] - ts[0]) / span, 3) if len(ts) > 1 and span else 0.0,
                "mean_reprojection_error_px": prim["mean_reprojection_error_px"],
                "mean_track_length": prim["mean_track_length"], "points": prim["points"],
                "points_per_registered_image": round(prim["points"] / max(prim["registered"], 1), 1),
                "point_density_per_unit3": round(prim["points"] / vol, 2),
                "density_units": "sfm_units^-3 (2-98 percentile box)", "attempts": sfm.get("attempts"),
                "notes": sfm.get("reasons")}
        if sp:
            ev["gaussian_splat"] = {k: sp.get(k) for k in ("status", "backend", "num_gaussians", "iterations", "seconds",
                                                           "device", "train_psnr", "test_psnr", "test_ssim", "test_views",
                                                           "cpu_fallback", "error", "train_size")}
            ev["gaussian_splat"]["metric_note"] = ("PSNR/SSIM on held-out keyframes (every 8th), masked to valid "
                                                   "pixels; measures view synthesis, not geometric accuracy")
        if al:
            ev["alignment"] = {k: al.get(k) for k in ("scale_status", "metres_per_sfm_unit", "scale_rel_sigma",
                                                      "georeferenced", "gcp_rmse_m")}
            ev["alignment"]["gravity"] = {k: al["gravity"].get(k) for k in ("method", "confidence",
                                                                            "camera_up_spread_deg",
                                                                            "agreement_with_camera_up_deg")}
        if fac:
            regd = [f for f in fac["facilities"] if f.get("position_sfm") is not None]
            ev["facilities"] = {"total": fac["facilities_total"], "registered": fac["facilities_registered"],
                                "mean_registration_confidence": round(float(np.mean(
                                    [f["registration_confidence"] for f in regd])), 3) if regd else None}
        gt = self.opts.get("ground_truth") or []
        if gt and fac and al and al["scale_status"] != "relative":
            pos = {f["facility_id"]: np.array(f["position_local"]) for f in fac["facilities"] if f.get("position_local")}
            rows = []
            for g in gt:
                a, b = pos.get(g["facility_a"]), pos.get(g["facility_b"])
                if a is None or b is None:
                    rows.append({**g, "error": None, "note": "facility not registered"})
                    continue
                d = float(np.linalg.norm(a - b))
                rows.append({**g, "reconstructed_m": round(d, 3), "error_m": round(d - g["distance_m"], 3)})
            ev["ground_truth_comparison"] = rows
        elif gt:
            ev["ground_truth_comparison"] = "not computed: reconstruction has no metric scale"
        return ev
