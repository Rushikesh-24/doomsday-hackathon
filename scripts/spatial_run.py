#!/usr/bin/env python
"""Run the video -> 3D pipeline from the command line (same job store as the API).

    python scripts/spatial_run.py samples/videos/walkthrough_stairway.mp4 --preset fast
    python scripts/spatial_run.py --resume <job_id> --force reconstruction
    python scripts/spatial_run.py video.mp4 --marker DICT_4X4_50:0.15     # ArUco for metric scale
"""
import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.spatial import jobs as J  # noqa: E402
from vision.spatial.pipeline import SpatialPipeline  # noqa: E402
from vision.spatial.quality import probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--preset", default="fast", choices=["fast", "balanced", "high"])
    ap.add_argument("--resume", default=None, help="job id to resume / retry")
    ap.add_argument("--force", nargs="*", default=[], help="stages to re-run")
    ap.add_argument("--marker", action="append", default=[], help="DICT_NAME:size_m")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--tracker", default="iou_centre", choices=["iou_centre", "bytetrack"])
    ap.add_argument("--backend", default=None, choices=["auto", "pytorch", "brush"], help="splat backend")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("vision.inference.pipeline", "httpx", "httpx2"):
        logging.getLogger(n).setLevel(logging.WARNING)
    if a.resume:
        j = J.get_job(a.resume)
        v = J.get_video(j["video_id"])
        jid, opts = a.resume, dict(j["options"])
        if a.backend:
            opts["splat_backend"] = a.backend
            J.update_job(jid, options=opts)
    else:
        src = Path(a.video)
        meta = probe(src)
        tmp = Path(tempfile.mkdtemp(dir=J.root())) / src.name
        shutil.copy2(src, tmp)
        vid = J.add_video(tmp, src.name, meta)
        opts = {"preset": a.preset, "tracker": a.tracker, "splat_backend": a.backend or "auto",
                "markers": [{"dictionary": m.split(":")[0], "size_m": float(m.split(":")[1])} for m in a.marker]}
        if a.stop_after:
            opts["stop_after"] = a.stop_after
        jid = J.create_job(vid, opts)
        v = J.get_video(vid)
    print(f"video {v['id']}  job {jid}  dir {J.job_dir(jid)}", flush=True)
    last = [None]

    def prog(p):
        J.update_job(jid, progress=p)
        key = (p.get("stage"), round((p.get("overall") or 0) * 50))
        if key != last[0]:
            last[0] = key
            st = p["stages"].get(p.get("stage") or "", {}) if p.get("stage") else {}
            print(f"[{time.strftime('%H:%M:%S')}] {p.get('overall', 0):.0%} {p.get('stage')}: {st.get('message') or ''}",
                  flush=True)
    J.update_job(jid, status="running")
    pipe = SpatialPipeline(J.job_dir(jid), Path(v["path"]), opts, on_progress=prog, video_id=v["id"])
    st = pipe.run(force=set(a.force))
    J.update_job(jid, status=st["status"], progress=pipe.progress())
    print("\nstatus:", st["status"])
    for s, x in st["stages"].items():
        print(f"  {s:<15} {x['status']:<9} {x.get('seconds')}s  {json.dumps(x.get('summary') or x.get('error'), default=str)[:300]}")


if __name__ == "__main__":
    main()
