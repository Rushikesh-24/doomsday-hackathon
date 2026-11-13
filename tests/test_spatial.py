"""Spatial pipeline tests. SfM / scale / registration are checked against exact synthetic ground truth
(ray-cast textured corridor walk with known poses and a 0.40 m ArUco marker). Real-footage results are
documented in docs/SPATIAL.md (not asserted here: they depend on model weights and take minutes)."""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vision.spatial import frames as F
from vision.spatial import quality as Q
from vision.spatial.alignment import SfmModel, align, combine, triangulate, umeyama
from vision.spatial.synthetic_walk import render_walk, write_video


# ------------------------------------------------------------------ splat core
def test_composite_matches_autograd_reference():
    from vision.spatial.splat.composite import Composite, composite_reference
    torch.manual_seed(0)
    N = 14
    m = (torch.rand(N, 2, dtype=torch.float64) * 8).requires_grad_()
    cn = torch.tensor([[0.2, 0.05, 0.3]], dtype=torch.float64).repeat(N, 1).requires_grad_()
    op = (torch.rand(N, dtype=torch.float64) * 0.6 + 0.2).requires_grad_()
    cl = torch.rand(N, 3, dtype=torch.float64).requires_grad_()
    bg = torch.rand(3, dtype=torch.float64).requires_grad_()
    pad = torch.stack([torch.randperm(N)[:12], torch.cat([torch.randperm(N)[:5], torch.full((7,), N)])])
    ys, xs = torch.meshgrid(torch.arange(4.), torch.arange(4.), indexing="ij")
    px = torch.stack([xs.reshape(-1) * 2 + .5, xs.reshape(-1) * 2 + 4.5]).double()
    py = torch.stack([ys.reshape(-1) * 2 + .5] * 2).double()
    f = lambda *a: Composite.apply(*a[:4], pad, px, py, a[4], 3)
    assert torch.allclose(f(m, cn, op, cl, bg), composite_reference(m, cn, op, cl, pad, px, py, bg), atol=1e-12)
    assert torch.autograd.gradcheck(f, (m, cn, op, cl, bg), eps=1e-6, atol=1e-5)


def test_splat_trains_and_ply_roundtrip(tmp_path):
    from vision.spatial.splat import trainer as T
    frames, poses, K, gt = render_walk(n_frames=10, w=128, h=96)
    views = [T.View(f"f{i}", R, t, K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                    cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
             for i, (fr, (R, t)) in enumerate(zip(frames, poses))]
    rng = np.random.default_rng(0)   # seed points on the true surfaces (floor + back wall)
    xyz = np.concatenate([np.stack([rng.uniform(-1.2, 1.2, 800), np.zeros(800), rng.uniform(1, 9, 800)], 1),
                          np.stack([rng.uniform(-1.2, 1.2, 400), rng.uniform(0, 2.8, 400), np.full(400, 9.0)], 1)])
    T.PRESETS["_t"] = dict(iterations=60, train_long_side=128, max_gaussians=5000, tile_k=64, densify_until=0.6)
    res = T.train(views, xyz, np.full((len(xyz), 3), 0.5), tmp_path, preset="_t", device="cpu", test_every=5)
    assert res.loss_curve[-1][1] < res.loss_curve[0][1]
    p = T.load_ply(Path(res.ply_path))
    assert len(p["x"]) == res.num_gaussians and {"f_dc_0", "opacity", "rot_3", "scale_2"} <= set(p)
    assert res.test_psnr is not None and Path(res.preview_path).exists()
    # backend-independent evaluator reproduces the trainer's own held-out score
    from vision.spatial.splat.evaluate import evaluate
    ev = evaluate(Path(res.ply_path), [v for i, v in enumerate(views) if i % 5 == 0], device="cpu")
    assert ev["psnr"] == pytest.approx(res.test_psnr, abs=0.05) and ev["sh_degree"] == 0


def test_cpu_fallback_reduces_preset(monkeypatch):
    from vision.spatial.splat import run as R
    monkeypatch.setenv("VISION_DEVICE", "cpu")
    from vision.config import reset_settings
    reset_settings()
    assert R.hardware_check()["device"] == "cpu"
    reset_settings()


# ------------------------------------------------------------------ quality & frames
def _video(tmp_path, kind):
    frames, *_ = render_walk(n_frames=40, w=320, h=240)
    if kind == "dark":
        frames = [(f * 0.12).astype(np.uint8) for f in frames]
    elif kind == "blurry":
        frames = [cv2.GaussianBlur(f, (0, 0), 6) for f in frames]
    elif kind == "static":
        frames = [frames[0]] * 40
    elif kind == "shaky":
        rng = np.random.default_rng(1)
        frames = [cv2.warpAffine(f, cv2.getRotationMatrix2D((160, 120), rng.normal(0, 6), 1.0), (320, 240))
                  for f in frames]
    return write_video(frames, tmp_path / f"{kind}.mp4", fps=10)


@pytest.mark.parametrize("kind,code", [("dark", "POOR_LIGHTING"), ("static", "LITTLE_MOVEMENT"),
                                       ("shaky", "CAMERA_SHAKE"), ("short", "TOO_SHORT")])
def test_quality_flags_problems(tmp_path, kind, code):
    if kind == "short":
        frames, *_ = render_walk(n_frames=12, w=320, h=240)
        p = write_video(frames, tmp_path / "short.mp4", fps=10)
    else:
        p = _video(tmp_path, kind)
    rep = Q.analyze_quality(p, analysis_fps=10)
    assert code in [i["code"] for i in rep.issues], rep.issues
    assert rep.recommendations


def test_good_capture_scores_higher_than_bad(tmp_path):
    good = Q.analyze_quality(_video(tmp_path, "good"), analysis_fps=10)
    bad = Q.analyze_quality(_video(tmp_path, "dark"), analysis_fps=10)
    assert good.suitability_score > bad.suitability_score
    assert "LOW_OVERLAP" not in [i["code"] for i in good.issues]


def test_invalid_video_rejected(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(Q.VideoError):
        Q.probe(p)
    q = tmp_path / "x.avi"
    q.write_bytes(b"RIFF....")
    with pytest.raises(Q.VideoError):
        Q.probe(q)


def test_keyframe_selection_skips_duplicates_and_blur():
    fr = []
    for i in range(60):
        fr.append({"index": i, "t": i / 10, "sharpness": 50 if i % 7 == 0 else 300, "blurry": i % 7 == 0,
                   "motion_px": 0.1 if 20 <= i < 40 else 6.0, "overlap": 0.95, "near_duplicate": 20 <= i < 40})
    picks = F.select_keyframes(fr, target_motion_px=18, max_keyframes=100)
    idx = [p["index"] for p in picks]
    assert not any(i % 7 == 0 for i in idx)                  # blurry frames avoided
    assert sum(20 <= i < 40 for i in idx) <= 2                # static stretch not oversampled
    assert len(F.select_keyframes(fr, target_motion_px=18, max_keyframes=5)) <= 5


# ------------------------------------------------------------------ geometry
def test_umeyama_and_triangulation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    a = 0.7
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Y = (2.5 * (Rz @ X.T)).T + [1, 2, 3]
    s, R, t = umeyama(X, Y)
    assert s == pytest.approx(2.5) and np.allclose(R, Rz) and np.allclose(t, [1, 2, 3])
    P = np.array([0.3, -0.2, 4.0])
    obs = []
    for c in ([0, 0, 0], [1, 0, 0], [0, 1, 0.5]):
        R0, t0 = np.eye(3), -np.array(c, float)
        pc = R0 @ P + t0
        obs.append((R0, t0, pc[:2] / pc[2]))
    Xt, err = triangulate(obs)
    assert np.allclose(Xt, P, atol=1e-9) and err < 1e-9
    assert combine([{"used": True, "scale_m_per_unit": 2.0, "rel_sigma": 0.02},
                    {"used": True, "scale_m_per_unit": 2.0, "rel_sigma": 0.02}])[0] == pytest.approx(2.0)
    assert combine([{"used": False}]) is None


@pytest.fixture(scope="module")
def synth_sfm(tmp_path_factory):
    """Real COLMAP run (worker process) on a synthetic walk with exact GT poses + 0.40 m ArUco marker."""
    from vision.spatial.sfm import run_in_subprocess
    d = tmp_path_factory.mktemp("synwalk")
    frames, poses, K, gt = render_walk(n_frames=30, marker={"dictionary": "DICT_4X4_50", "size_m": 0.4})
    (d / "frames").mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / "frames" / f"frame_{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rep = run_in_subprocess(d / "frames", d / "sfm", {})
    return d, rep, gt, poses, K


def test_sfm_recovers_trajectory(synth_sfm):
    d, rep, gt, *_ = synth_sfm
    assert rep["status"] == "ok" and rep["registration_rate"] >= 0.9, rep
    m = SfmModel(d / "sfm/models/0")
    names = sorted(m.images)
    est = np.array([m.center(n) for n in names])
    ref = gt["centers"][[int(n[6:12]) for n in names]]
    s, R, t = umeyama(est, ref)
    rmse = float(np.sqrt((np.linalg.norm((s * (R @ est.T)).T + t - ref, axis=1) ** 2).mean()))
    assert rmse < 0.03 * gt["path_length"], rmse
    assert rep["models"]["0"]["temporal_check"]["jumps"] == 0          # continuous walk: no teleporting poses
    assert (d / "sfm/sparse_points.ply").exists() and (d / "sfm/models/0/colmap").exists()


def test_aruco_scale_and_scale_status(synth_sfm):
    d, rep, gt, *_ = synth_sfm
    m = SfmModel(d / "sfm/models/0")
    names = sorted(m.images)
    est = np.array([m.center(n) for n in names])
    s_true = umeyama(est, gt["centers"][[int(n[6:12]) for n in names]])[0]
    al = align(m, d / "frames", {"markers": [{"dictionary": "DICT_4X4_50", "size_m": 0.4}]})
    assert al["scale_status"] == "scale_calibrated" and al["metric"]
    assert al["metres_per_sfm_unit"] == pytest.approx(s_true, rel=0.04)
    # without references: never claims metric
    al2 = align(m, d / "frames", {})
    assert al2["scale_status"] in ("relative", "approximate_metric") and not al2["metric"]
    assert al2["georeferenced"] is False
    # gravity: local +Y should match world up (synthetic world is Y-up)
    T = np.array(al["T_local_from_sfm"])
    up_local = T[:3, :3] @ np.array(al["gravity"]["up"])
    assert up_local[1] / np.linalg.norm(up_local) > 0.99
    # GCPs need a calibrated scale and >= 3 points
    al3 = align(m, d / "frames", {"gcps": [{"local": [0, 0, 0], "enu_m": [0, 0, 0]}]})
    assert al3["georeferenced"] is False and "georeference_note" in al3


def test_registration_places_facility_and_refuses_without_support(synth_sfm):
    from vision.spatial.registration import register
    d, rep, gt, poses, K = synth_sfm
    m = SfmModel(d / "sfm/models/0")
    al = align(m, d / "frames", {"markers": [{"dictionary": "DICT_4X4_50", "size_m": 0.4}]})
    names = sorted(m.images)
    # the ArUco marker on the back wall (world centre (0, 1.6, 9)) observed as a "signboard" in every frame
    obs = []
    for n in names:
        i = int(n[6:12])
        R, t = poses[i]
        c3 = np.array([[-0.2, 1.8, 9.0], [0.2, 1.4, 9.0]])
        pc = c3 @ R.T + t
        uv = pc[:, :2] / pc[:, 2:] * K[0, 0] + K[:2, 2]
        x1, y1 = uv.min(0) - 25
        x2, y2 = uv.max(0) + 25
        obs.append({"frame": n, "frame_index": i, "t": i / 10, "detection_id": f"d{i}", "bbox": [x1, y1, x2, y2],
                    "confidence": 0.8})
    tracks = [{"track_id": 1, "class_name": "signboard", "observations": obs, "confirmed": True,
               "track_confidence": 0.8},
              {"track_id": 2, "class_name": "signboard", "observations": obs[:4], "confirmed": True},  # same object
              {"track_id": 3, "class_name": "door", "confirmed": True,   # box outside the textured surfaces
               "observations": [{"frame": names[0], "t": 0, "detection_id": "x", "bbox": [0, 0, 3, 3],
                                 "confidence": 0.5}]}]
    res = register(m, tracks, al["T_local_from_sfm"], al["scale_status"], "vid")
    signs = [f for f in res["facilities"] if f["class_name"] == "signboard"]
    assert len(signs) == 1 and set(signs[0]["track_ids"]) == {1, 2}            # merged in 3D
    # independent reference in the SfM frame: the marker centre from triangulated ArUco corners
    mk = next(x for x in al["sources"]["aruco"] if x.get("used"))
    centre = np.array(mk["corners_sfm"]).mean(0)
    err_m = np.linalg.norm(np.array(signs[0]["position_sfm"]) - centre) * al["metres_per_sfm_unit"]
    assert err_m < 0.15, err_m                                                   # metres
    assert signs[0]["registration_confidence"] > 0.3 and signs[0]["position_units"] == "metres"
    door = next(f for f in res["facilities"] if f["class_name"] == "door")
    assert door["position_sfm"] is None and door["position_local"] is None and "reason" in door


# ------------------------------------------------------------------ pipeline + API
def test_pipeline_resume_and_failure(tmp_path):
    from vision.spatial.pipeline import SpatialPipeline
    frames, *_ = render_walk(n_frames=30, w=320, h=240)
    v = write_video(frames, tmp_path / "walk.mp4", fps=10)
    p = SpatialPipeline(tmp_path / "job", v, {"stop_after": "frames"})
    st = p.run()
    assert st["stages"]["quality"]["status"] == "done" and st["stages"]["frames"]["status"] == "done"
    assert st["stages"]["sfm"]["status"] == "skipped"
    t_frames = st["stages"]["frames"]["finished"]
    p2 = SpatialPipeline(tmp_path / "job", v, {"stop_after": "frames"})
    st2 = p2.run()
    assert st2["stages"]["frames"]["finished"] == t_frames            # completed stage not repeated
    # failure: a 2-frame static clip cannot yield keyframes -> downstream blocked, never faked
    w = write_video([frames[0]] * 6, tmp_path / "static.mp4", fps=10)
    p3 = SpatialPipeline(tmp_path / "job2", w, {"stop_after": "reconstruction"})
    st3 = p3.run()
    assert st3["stages"]["frames"]["status"] == "failed"
    assert st3["stages"]["sfm"]["status"] == "blocked" and st3["stages"]["reconstruction"]["status"] == "blocked"
    assert not (tmp_path / "job2" / "splat" / "splat.ply").exists()


def test_failed_reconstruction_is_reported(tmp_path):
    import subprocess
    import sys
    job = tmp_path / "job"
    (job / "sfm/models/0").mkdir(parents=True)
    (job / "sfm/models/0/poses.json").write_text(json.dumps({"camera": {"model": "SIMPLE_PINHOLE", "params": [100, 50, 50],
                                                                        "width": 100, "height": 100}, "images": []}))
    np.savez(job / "sfm/models/0/points.npz", xyz=np.zeros((0, 3)), rgb=np.zeros((0, 3)))
    r = subprocess.run([sys.executable, "-m", "vision.spatial.splat.run", str(job), "fast"], capture_output=True,
                       cwd=str(Path(__file__).resolve().parent.parent))
    assert r.returncode != 0
    info = json.loads((job / "splat/splat.json").read_text())
    assert info["status"] == "failed" and info["error"] and not (job / "splat/splat.ply").exists()


def test_spatial_api(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_PRELOAD", "0")
    monkeypatch.delenv("VISION_BRUSH_BIN", raising=False)     # refusal path is checked below
    from fastapi.testclient import TestClient
    from api.app import app
    frames, *_ = render_walk(n_frames=30, w=320, h=240)
    v = write_video(frames, tmp_path / "walk.mp4", fps=10)
    with TestClient(app) as c:
        assert c.post("/spatial/videos/upload", files={"file": ("x.mp4", b"\x00" * 50, "video/mp4")}).status_code == 422
        assert c.post("/spatial/videos/upload", files={"file": ("x.avi", b"RIFF", "video/avi")}).status_code == 415
        r = c.post("/spatial/videos/upload", files={"file": ("walk.mp4", v.read_bytes(), "video/mp4")})
        assert r.status_code == 201, r.text
        vid = r.json()["video_id"]
        assert c.post(f"/spatial/videos/{vid}/analyze", json={"preset": "ultra"}).status_code == 400
        assert c.post(f"/spatial/videos/{vid}/analyze", json={"splat_backend": "brush"}).status_code == 400
        j = c.post(f"/spatial/videos/{vid}/analyze", json={"stop_after": "frames"}).json()
        for _ in range(120):
            pr = c.get(j["progress_url"]).json()
            if pr["status"] not in ("queued", "running"):
                break
            time.sleep(0.5)
        assert pr["status"] == "completed", pr
        assert c.get(f"/spatial/videos/{vid}/quality").json()["suitability"] in ("good", "fair", "poor")
        job = c.get(f"/spatial/jobs/{j['job_id']}").json()
        assert job["stages"]["frames"]["status"] == "done"
        assert any(x["id"] == j["job_id"] for x in c.get("/spatial/reconstructions").json()["reconstructions"])
        assert c.get(f"/spatial/reconstructions/{j['job_id']}/assets").status_code == 404   # export skipped
        assert c.get(f"/spatial/reconstructions/{j['job_id']}/facilities").status_code == 404
        assert c.get("/spatial/jobs/nope").status_code == 404


@pytest.mark.parametrize("method", ["iou_centre", "bytetrack"])
def test_same_facility_over_20_frames_is_one_track(method):
    from vision.schemas import BBox, Detection, ImageInfo, ImageResult, ModelInfo, SourceRef, Timing
    from vision.spatial.tracking import track
    res, rows = [], []
    for i in range(20):
        x = 100 + 8 * i                    # staircase drifting across the frame as the camera walks
        dets = [Detection(id=f"r{i}-000", class_name="staircase", category="facility", confidence=0.6,
                          bbox=BBox.from_xyxy([x, 200, x + 300, 600], 1280, 720), source=SourceRef(name=f"f{i}"),
                          model_name="m", model_version="1", timestamp="t")]
        res.append(ImageResult(request_id=f"r{i}", image=ImageInfo(name=f"f{i}", width=1280, height=720),
                               model=ModelInfo(key="m", name="m", version="1"), thresholds={}, detections=dets,
                               counts_by_class={}, timing=Timing(preprocess_ms=0, inference_ms=0, postprocess_ms=0,
                                                                 attributes_ms=0, total_ms=0), processed_at="t"))
        rows.append({"name": f"f{i}.jpg", "frame_index": i * 5, "t": i * 0.5})
    out = track(res, rows, method=method, fps=2)
    confirmed = [t for t in out["tracks"] if t["confirmed"]]
    assert len(confirmed) == 1 and confirmed[0]["n_observations"] >= 19
    assert confirmed[0]["observations"][0]["detection_id"] == "r0-000" or method == "bytetrack"


@pytest.mark.skipif(not __import__("os").environ.get("VISION_BRUSH_BIN"), reason="Brush binary not configured")
def test_brush_backend_on_synthetic_walk(synth_sfm):
    """Approved external backend: trains on a holdout-free COLMAP copy, PLY scored by the shared evaluator."""
    from vision.spatial.splat.brush import run_brush
    from vision.spatial.splat.evaluate import evaluate
    from vision.spatial.splat.run import build_views
    d, *_ = synth_sfm
    views, *_ = build_views(d, 320)
    test = [v for i, v in enumerate(views) if i % 8 == 0]
    out = d / "brush"
    out.mkdir(exist_ok=True)
    r = run_brush(d, "fast", out, holdout=[v.name for v in test], steps=600, max_res=320)
    ev = evaluate(Path(r["ply_path"]), test, device="cpu")
    assert ev["sh_degree"] == 3 and ev["num_gaussians"] > 1000
    assert ev["psnr"] > 15, ev["psnr"]
