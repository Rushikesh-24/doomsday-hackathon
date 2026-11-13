"""Optional backend: Brush (github.com/ArthurBrussee/brush, Apache-2.0) - WebGPU/Metal 3DGS trainer.

Approved by the operator on 2026-11-13 for v0.3.0 brush-app-aarch64-apple-darwin (archive sha256 pinned below,
verified before extraction). Still opt-in: it only runs when VISION_BRUSH_BIN points at the binary.

Fair evaluation: Brush trains on a copy of the COLMAP model from which the held-out frames have been
deregistered (it never sees them); its PLY (SH degree 3, same frame as COLMAP - verified) is then scored by
vision.spatial.splat.evaluate with the same views/masks/rasterizer as the in-house trainer.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PINNED_ARCHIVE_SHA256 = {"v0.3.0/brush-app-aarch64-apple-darwin.tar.xz":
                         "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c"}
STEPS = {"fast": 5000, "balanced": 15000, "high": 30000}
MAX_RES = {"fast": 640, "balanced": 960, "high": 1600}


def available() -> tuple[bool, str]:
    b = os.environ.get("VISION_BRUSH_BIN")
    if not b:
        return False, "VISION_BRUSH_BIN not set (Brush backend requires explicit operator approval)"
    if not Path(b).is_file() or not os.access(b, os.X_OK):
        return False, f"{b} is not an executable file"
    return True, b


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_FILTER = """
import sys, json, pycolmap
src, dst, names = sys.argv[1], sys.argv[2], set(json.loads(sys.argv[3]))
r = pycolmap.Reconstruction(src)
fids = {im.frame_id for im in r.images.values() if im.name in names}
for f in fids:
    r.deregister_frame(f)
r.write(dst)
print(len(fids))
"""


def prepare_dataset(job: Path, out: Path, holdout: list[str]) -> Path:
    src = job / "sfm/models/0/colmap"
    if not src.exists():
        raise RuntimeError("no COLMAP binary model found for Brush")
    ds = out / "brush_dataset"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "sparse/0").mkdir(parents=True)
    os.symlink((job / "frames").resolve(), ds / "images")
    r = subprocess.run([sys.executable, "-c", _FILTER, str(src), str(ds / "sparse/0"), json.dumps(holdout)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"could not prepare Brush dataset: {r.stderr[-500:]}")
    return ds


def run_brush(job: Path, preset: str, out: Path, holdout: list[str] | None = None, steps: int | None = None,
              max_res: int | None = None) -> dict:
    ok, binary = available()
    if not ok:
        raise RuntimeError(binary)
    ds = prepare_dataset(job, out, holdout or [])
    steps = steps or STEPS[preset]
    max_res = max_res or MAX_RES[preset]
    for old in out.glob("export_*.ply"):
        old.unlink()
    cmd = [binary, str(ds), "--total-steps", str(steps), "--export-every", str(steps), "--export-path", str(out),
           "--max-resolution", str(max_res), "--seed", "0"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    secs = time.time() - t0
    (out / "brush.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    plys = sorted(out.glob("export_*.ply"), key=lambda p: p.stat().st_mtime)
    if r.returncode != 0 or not plys:
        raise RuntimeError(f"brush exited {r.returncode}; see brush.log")
    plys[-1].replace(out / "splat.ply")
    return {"ply_path": str(out / "splat.ply"), "iterations": steps, "seconds": round(secs, 1),
            "max_resolution": max_res, "command": [Path(cmd[0]).name] + cmd[1:], "binary_sha256": sha256(Path(binary)),
            "holdout_frames_removed": len(holdout or [])}
