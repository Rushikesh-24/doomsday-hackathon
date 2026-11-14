"""Save a finished reconstruction (with its tagged places and floor plans) as a portable demo bundle.

    .venv/Scripts/python demo/export_demo.py <reconstruction_id> <bundle_name> [--title "..."] [--video path.mp4]

Writes demo/bundles/<bundle_name>/ (bundle.json + job/). Heavy COLMAP scratch (feature databases) and logs are
left out; everything the 3D Explorer, Wayfinding and photo localisation need is kept. Re-run it after you tag
or move places so the stage copy is up to date. Load it on any machine with demo/load_demo.py.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SKIP_DIRS = {"attempt_sequential", "attempt_exhaustive_relaxed", "__pycache__"}
SKIP_FILES = {"RUNNING", "sfm.log", "splat.log", "database.db", "database.db-shm", "database.db-wal"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rid", help="reconstruction (job) id, e.g. 5eecc657ba9a")
    ap.add_argument("name", help="bundle name, letters/digits only, e.g. pcc")
    ap.add_argument("--title", default=None, help="human title shown in the demo README")
    ap.add_argument("--video", default=None, help="optional source video to include for the upload part of the demo")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9]{2,20}", a.name):
        sys.exit("bundle name must be 2-20 lowercase letters/digits")
    from vision.spatial import jobs as J
    src = J.job_dir(a.rid)
    job = J.get_job(a.rid)
    if job is None or not (src / "trajectory.json").exists():
        sys.exit(f"reconstruction {a.rid} not found or not finished")
    with sqlite3.connect(J.root() / "spatial.db") as c:
        row = c.execute("SELECT name, meta FROM videos WHERE id=?", (job["video_id"],)).fetchone()
    out = ROOT / "demo" / "bundles" / a.name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out / "job", ignore=lambda d, names: [n for n in names if n in SKIP_DIRS or n in SKIP_FILES])
    if a.video:
        shutil.copy2(a.video, out / ("source" + Path(a.video).suffix.lower()))
    places = json.loads((src / "places.json").read_text())["places"] if (src / "places.json").exists() else []
    meta = {"name": a.name, "title": a.title or (row[0] if row else a.name), "source_job": a.rid,
            "status": job["status"], "options": job["options"], "video_name": row[0] if row else None,
            "video_meta": json.loads(row[1]) if row else {}, "places": [p["name"] for p in places],
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "bundle.json").write_text(json.dumps(meta, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"exported {a.rid} -> {out.relative_to(ROOT)} ({size:.1f} MB, {len(places)} places: {', '.join(meta['places'])})")


if __name__ == "__main__":
    main()
