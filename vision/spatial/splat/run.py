"""Gaussian-splat training worker: `python -m vision.spatial.splat.run <job_dir> <preset> [backend]`.

Reads the primary SfM model (plain files), prepares undistorted training views + validity masks,
checks hardware, trains, and writes <job_dir>/splat/{splat.ply, splat.json, preview, progress.json}.
Exit code != 0 and splat.json status=failed on any failure - a missing PLY is never reported as success.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def hardware_check() -> dict:
    import torch
    from vision.device import select_device
    from vision.config import get_settings
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    dev = select_device(get_settings().device)          # honours VISION_DEVICE=cpu
    return {"device": dev, "cuda": cuda, "mps": mps, "torch": torch.__version__,
            "cuda_name": torch.cuda.get_device_name(0) if cuda else None}


def build_views(job: Path, long_side: int):
    from vision.spatial.splat.trainer import View
    P = json.loads((job / "sfm/models/0/poses.json").read_text())
    cam = P["camera"]
    p = cam["params"]
    model = cam["model"]
    if model == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist = np.array([p[3], 0, 0, 0])
    elif model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist = np.zeros(4)
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]; dist = np.zeros(4)
    elif model == "OPENCV":
        fx, fy, cx, cy = p[:4]; dist = np.array(p[4:8])
    else:
        raise ValueError(f"unsupported camera model {model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    W, H = cam["width"], cam["height"]
    s = min(1.0, long_side / max(W, H))
    w, h = round(W * s), round(H * s)
    valid = cv2.undistort(np.full((H, W), 255, np.uint8), K, dist) > 250
    valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sp = job / "sfm/models/0/suspect_frames.json"
    suspects = set(json.loads(sp.read_text())) if sp.exists() else set()
    views = []
    for im in P["images"]:
        if im["name"] in suspects:            # aliased poses would corrupt the optimisation
            continue
        bgr = cv2.imread(str(job / "frames" / im["name"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.undistort(bgr, K, dist), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        views.append(View(im["name"], np.array(im["R"]), np.array(im["t"]), fx * s, fy * s, cx * s, cy * s, rgb, valid))
    d = np.load(job / "sfm/models/0/points.npz")
    return views, d["xyz"], d["rgb"].astype(np.float32) / 255, {"train_size": [w, h], "camera_model": model}


def main(job: Path, preset: str, backend: str = "auto") -> int:
    from vision.spatial.splat import trainer
    if backend == "auto":   # approved external Brush when configured, else the in-house trainer
        from vision.spatial.splat.brush import available
        backend = "brush" if available()[0] else "pytorch"
    out = job / "splat"
    out.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "backend_requested": backend, "preset": preset, "started": time.time()}
    prog = out / "progress.json"

    def progress(frac, msg):
        prog.write_text(json.dumps({"fraction": round(float(frac), 4), "message": msg, "time": time.time()}))
    try:
        status["backend"] = backend
        hw = hardware_check()
        status["hardware"] = hw
        if backend == "brush":
            from vision.spatial.splat.brush import MAX_RES, run_brush
            from vision.spatial.splat.evaluate import evaluate
            views, xyz, _, info = build_views(job, MAX_RES[preset])
            status.update(info)
            status["initial_points"] = int(len(xyz))
            test = [v for i, v in enumerate(views) if i % 8 == 0] if len(views) >= 16 else []
            progress(0.05, "Brush training (external binary)")
            status.update(run_brush(job, preset, out, holdout=[v.name for v in test]))
            progress(0.9, "evaluating held-out views")
            # same evaluator / views / masks as the in-house trainer (at 320 px for comparability)
            ev_views, *_ = build_views(job, 320)
            ev_test = [v for i, v in enumerate(ev_views) if i % 8 == 0] if len(ev_views) >= 16 else []
            ev_train = [v for i, v in enumerate(ev_views) if i % 8 != 0][::5]
            te = evaluate(Path(status["ply_path"]), ev_test, device=hw["device"], previews=4) if ev_test else None
            tr = evaluate(Path(status["ply_path"]), ev_train, device=hw["device"])
            status.update(num_gaussians=tr["num_gaussians"], sh_degree=tr["sh_degree"], train_psnr=tr["psnr"],
                          test_psnr=te["psnr"] if te else None, test_ssim=te["ssim"] if te else None,
                          test_views=te["views"] if te else [], device=f"brush(webgpu/metal); eval {hw['device']}",
                          eval_note="evaluated at 320 px with the in-house rasterizer incl. SH degree 3")
            if te and te["previews"]:
                import cv2
                H0 = min(p.shape[0] for p in te["previews"])
                sheet = np.concatenate([p[:H0] for p in te["previews"]], 0)
                pp = out / "preview_gt_vs_render.jpg"
                cv2.imwrite(str(pp), cv2.cvtColor((np.clip(sheet, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                status["preview_path"] = str(pp)
        else:
            P = dict(trainer.PRESETS[preset])
            if hw["device"] == "cpu":
                # CPU fallback: honest, reduced job (full presets would take many hours)
                P.update(iterations=min(P["iterations"], 400), train_long_side=min(P["train_long_side"], 240))
                status["cpu_fallback"] = "no CUDA/MPS GPU: reduced to 400 iterations at 240 px"
            trainer.PRESETS["_run"] = P
            views, xyz, rgb, info = build_views(job, P["train_long_side"])
            status.update(info)
            status["initial_points"] = int(len(xyz))
            if len(views) < 3:
                raise ValueError(f"only {len(views)} registered views: need >= 3")
            res = trainer.train(views, xyz, rgb, out, preset="_run", device=hw["device"], progress=progress)
            status.update(res.__dict__)
            status["preset_params"] = P
        ply = Path(status.get("ply_path", out / "splat.ply"))
        if not ply.exists() or ply.stat().st_size < 1000:
            raise RuntimeError("training finished without producing a PLY")
        # quality gate (both backends): a collapsed / diverged optimisation is a failure, not a reconstruction
        if True:
            n0 = status.get("initial_points") or 0
            if (status.get("num_gaussians") or 0) < max(200, 0.1 * n0):
                raise RuntimeError(f"optimisation collapsed: {status.get('num_gaussians')} Gaussians left "
                                   f"(from {n0} initial points)")
            if status.get("train_psnr") is not None and status["train_psnr"] < 12.0:
                raise RuntimeError(f"optimisation diverged: train PSNR {status['train_psnr']} dB < 12 dB")
        status["status"] = "done"
    except Exception as e:
        status.update(status="failed", error=f"{e.__class__.__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    status["finished"] = time.time()
    (out / "splat.json").write_text(json.dumps(status, indent=2, default=str))
    return 0 if status["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "auto"))
