#!/usr/bin/env python
"""Stream the official DIODE validation archive (laser-scanned metric depth, Vasiljevic et al. 2019,
https://diode-dataset.org) and keep a subset for measurement validation. Only RGB/depth/mask arrays
are extracted (no code). Intrinsics (from the DIODE paper/devkit): fx=886.81, fy=927.06, cx=512, cy=384
for 1024x768 images.

    python scripts/fetch_diode_subset.py --per-scan 4 --out samples/validation/diode
"""
import argparse
import collections
import io
import tarfile
import urllib.request
from pathlib import Path

URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/validation/diode")
    ap.add_argument("--per-scan", type=int, default=4, help="samples kept per scan")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = collections.Counter()
    keep_stems = set()
    with urllib.request.urlopen(URL, timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = Path(m.name)
                if p.suffix not in (".png", ".npy") or ".." in p.parts:
                    continue
                stem = p.name.replace("_depth_mask.npy", "").replace("_depth.npy", "").replace(".png", "")
                scan = "/".join(p.parts[-4:-1])  # split/scene/scan
                if stem not in keep_stems:
                    if kept[scan] >= a.per_scan:
                        continue
                    keep_stems.add(stem)
                    kept[scan] += 1
                kind = p.parts[-4] if len(p.parts) >= 4 else "x"   # indoors / outdoor
                dest = out / kind / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(m)
                dest.write_bytes(f.read())
    print("kept", len(keep_stems), "samples from", len(kept), "scans")


if __name__ == "__main__":
    main()
