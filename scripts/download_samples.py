#!/usr/bin/env python
"""Download the openly-licensed demo inputs from Wikimedia Commons and record provenance.

    python scripts/download_samples.py            # images (already committed) + demo video
    python scripts/download_samples.py --video-only

Images -> samples/images/ (+ SOURCES.json), video -> samples/videos/ (MP4 via ffmpeg, git-ignored).
"""
import argparse
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "campus-vision-research/0.1 (open-source test data)"}
IMAGES = {
    "stairs_handrail_sydney.jpg": "File:Staircase in the University of Sydney Main Quadrangle.jpg",
    "stairs_oau_campus.jpg": "File:Staircase leading to the Amphitheater at Obafemi Awolowo University.jpg",
    "ramp_hull_museum.jpg": "File:Hull Maritime Museum New Wheelchair Ramp, Kingston upon Hull Jun24.jpg",
    "ramp_arts.jpg": "File:Arts Ramp (12648473343).jpg",
    "elevator_lobby_hotel.jpg": "File:Elevator door in the lobby of the Hotel.jpg",
    "tactile_paving_doncaster.jpg": "File:Tactile paving at Doncaster railway station.jpg",
    "tactile_paving_singapore.jpg": "File:Tactile paving in a Mass Rapid Transit station in Singapore - 20131105.jpg",
    "corridor_stirling_students.jpg": "File:Students walking down a corridor at the University of Stirling.jpg",
    "campus_entrance_rmcad.jpg": "File:The Tri-Boro Building, front entrance, RMCAD campus 2023-08-13.jpg",
    "accessible_toilet_signs.jpg": "File:Spitalfields Market accessible toilet door signs Tower Hamlets London England.jpg",
}
VIDEOS = {"access_city_logrono.mp4": "File:Access City Award 2015 - Logroño.webm",
          # real handheld walkthroughs used by the spatial (3D reconstruction) pipeline
          "walkthrough_metro_corridor.mp4": "File:Moscow metro Krasnoselskaya vestibule corridor 2023-03.webm",
          "walkthrough_stairway.mp4": "File:Up the Lake Creek Falls Stairway.ogv"}


def info(title: str) -> dict:
    p = {"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
         "iiprop": "url|extmetadata", "iiurlwidth": "1280"}
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p), headers=UA), timeout=40))
    return list(d["query"]["pages"].values())[0]["imageinfo"][0]


def meta(name, title, ii):
    m = ii["extmetadata"]
    return {"file": name, "title": title, "page": ii["descriptionurl"],
            "license": m.get("LicenseShortName", {}).get("value"),
            "author": re.sub("<[^>]+>", "", m.get("Artist", {}).get("value", "unknown")).strip()}


def fetch(url, dest: Path):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-only", action="store_true")
    a = ap.parse_args()
    if not a.video_only:
        d = ROOT / "samples/images"
        d.mkdir(parents=True, exist_ok=True)
        man = []
        for name, title in IMAGES.items():
            ii = info(title)
            if not (d / name).exists():
                fetch(ii.get("thumburl") or ii["url"], d / name)
                time.sleep(1)
            man.append(meta(name, title, ii))
            print("image ok", name)
        (d / "SOURCES.json").write_text(json.dumps(man, indent=2))
    vd = ROOT / "samples/videos"
    vd.mkdir(parents=True, exist_ok=True)
    man = []
    for name, title in VIDEOS.items():
        ii = info(title)
        out = vd / name
        if not out.exists():
            raw = vd / ("_" + Path(ii["url"]).name)
            fetch(ii["url"], raw)
            ff = shutil.which("ffmpeg")
            if not ff:
                raise SystemExit("ffmpeg needed to convert WebM -> MP4 (brew install ffmpeg)")
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'", "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)],
                           check=True)
            raw.unlink()
        man.append(meta(name, title, ii))
        print("video ok", name)
    (vd / "SOURCES.json").write_text(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
