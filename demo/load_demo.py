"""Install the stage demo into the local outputs/ store (safe to re-run; it never touches your own scans).

    .venv/Scripts/python demo/load_demo.py                 # every bundle in demo/bundles
    .venv/Scripts/python demo/load_demo.py pcc             # just one
    .venv/Scripts/python demo/load_demo.py --reset         # also restore the bundle's places / plans
    .venv/Scripts/python demo/load_demo.py --photos        # also analyse the sample photos (API must be running)

Each bundle becomes reconstruction id `demo<name>` (e.g. demopcc), so links like
http://localhost:3000/explorer?id=demopcc and /navigation?id=demopcc&to=CG%20Lab are stable for the show.
Places you add while rehearsing are kept unless you pass --reset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLES = ROOT / "demo" / "bundles"
KEEP_ON_RELOAD = ("places.json", "plan")


def install(bundle: Path, reset: bool) -> str:
    from vision.spatial import jobs as J
    meta = json.loads((bundle / "bundle.json").read_text())
    rid, vid = f"demo{meta['name']}", f"demo{meta['name']}"[:10]
    dst = J.job_dir(rid)
    kept = {}
    if dst.exists() and not reset:     # keep places / plan alignment edited during rehearsal
        tmp = J.root() / f"_keep_{uuid.uuid4().hex[:6]}"
        tmp.mkdir()
        for k in KEEP_ON_RELOAD:
            if (dst / k).exists():
                shutil.move(str(dst / k), tmp / k)
                kept[k] = tmp / k
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(bundle / "job", dst)
    for k, p in kept.items():
        if (dst / k).exists():
            shutil.rmtree(dst / k) if (dst / k).is_dir() else (dst / k).unlink()
        shutil.move(str(p), dst / k)
    if kept:
        shutil.rmtree(next(iter(kept.values())).parent, ignore_errors=True)
    (dst / "RUNNING").unlink(missing_ok=True)
    vdir = J.root() / "videos" / vid
    vdir.mkdir(parents=True, exist_ok=True)
    src_video = next(bundle.glob("source.*"), None)
    vpath = vdir / (src_video.name if src_video else "source.mp4")
    if src_video and not vpath.exists():
        shutil.copy2(src_video, vpath)
    now = J.now()
    with sqlite3.connect(J.root() / "spatial.db") as c:
        c.executescript(J._SCHEMA)
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?)",
                  (vid, meta.get("video_name") or f"{meta['name']}.mp4", str(vpath), json.dumps(meta.get("video_meta", {})), now))
        prog = {"stage": None, "overall": 1.0, "stages": {}}
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                  (rid, vid, meta.get("status", "completed"), json.dumps(prog), json.dumps(meta.get("options", {})), now, now, None))
    places = json.loads((dst / "places.json").read_text())["places"] if (dst / "places.json").exists() else []
    print(f"  {meta['title']}: reconstruction {rid} - {len(places)} places ({', '.join(p['name'] for p in places)})"
          + (" - kept your rehearsal edits" if kept else ""))
    return rid


def analyse_photos(api: str) -> None:
    imgs = sorted((ROOT / "samples" / "images").glob("*.jpg"))
    for f in imgs:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{f.name}\"\r\n"
                f"Content-Type: image/jpeg\r\n\r\n").encode() + f.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{api}/vision/analyze-image", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                n = len(json.loads(r.read())["detections"])
            print(f"  analysed {f.name}: {n} detections")
        except Exception as e:      # keep going; the photo part of the demo is optional
            print(f"  {f.name}: failed ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="bundle names (default: all)")
    ap.add_argument("--reset", action="store_true", help="restore the bundle's places and floor plans")
    ap.add_argument("--photos", action="store_true", help="also analyse samples/images through the running API")
    ap.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    bundles = [BUNDLES / n for n in a.names] if a.names else sorted(p for p in BUNDLES.iterdir() if (p / "bundle.json").exists())
    if not bundles:
        sys.exit("no bundles in demo/bundles - create one with demo/export_demo.py")
    print("Installing demo reconstructions:")
    ids = [install(b, a.reset) for b in bundles]
    if a.photos:
        print("Analysing sample photos:")
        analyse_photos(a.api)
    print("\nOpen:")
    for rid in ids:
        print(f"  http://localhost:3000/explorer?id={rid}")
        print(f"  http://localhost:3000/navigation?id={rid}")


if __name__ == "__main__":
    main()
