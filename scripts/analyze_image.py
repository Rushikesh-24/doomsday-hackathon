#!/usr/bin/env python
"""Analyze one or more images (files or folders) from the command line.

    python scripts/analyze_image.py samples/images/stairs_oau_campus.jpg
    python scripts/analyze_image.py samples/images --model owlv2_base --conf 0.2 --out outputs/demo
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.inference.pipeline import get_pipeline  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None, help="override base confidence threshold")
    ap.add_argument("--classes", default=None, help="comma-separated subset")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    files = []
    for i in a.inputs:
        p = Path(i)
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    pipe = get_pipeline()
    classes = a.classes.split(",") if a.classes else None
    for f in files:
        r = pipe.analyze_image(f, model=a.model, conf=a.conf, classes=classes, save_dir=Path(a.out))
        print(f"\n== {f.name}  ({r.image.width}x{r.image.height})  model={r.model.key}  "
              f"inference={r.timing.inference_ms:.0f}ms")
        for d in r.detections:
            attrs = {k: v.value for k, v in d.attributes.items()
                     if v.status.value in ("detected", "estimated") and v.value not in (None, [], {})}
            print(f"  {d.class_name:<20} {d.confidence:.2f}  [{d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}]"
                  f"  {json.dumps(attrs, default=str)[:160]}")
        for fd in r.findings:
            print(f"  ! {fd.code}: {fd.message}")
        print(f"  annotated: {r.annotated_image}\n  json:      {r.json_path}")


if __name__ == "__main__":
    main()
