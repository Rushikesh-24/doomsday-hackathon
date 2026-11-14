"""Draw a clean, architect-style floor-plan sketch of a demo bundle for the "upload your plan" part of the demo.

    .venv/Scripts/python demo/make_sample_plan.py pcc

It is drawn from the scan itself (walked corridor offset into walls, rooms at the tagged places) and then
rotated / rescaled on purpose, so on stage you can show the plan being dragged into alignment.
It is a SKETCH derived from the scan, not an official building plan - say so if asked.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main(name: str) -> None:
    job = ROOT / "demo" / "bundles" / name / "job"
    cams = np.array([c["center_local"] for c in json.loads((job / "trajectory.json").read_text())["cameras"]])
    places = json.loads((job / "places.json").read_text())["places"] if (job / "places.json").exists() else []
    xz = cams[:, [0, 2]]
    # smooth the walk into a corridor centre line
    k = 5
    pad = np.r_[np.repeat(xz[:1], k, 0), xz, np.repeat(xz[-1:], k, 0)]
    centre = np.array([pad[i:i + 2 * k + 1].mean(0) for i in range(len(xz))])
    lo, hi = centre.min(0) - 4, centre.max(0) + 4
    S = 40                                              # px per metre in the sketch
    W, H = int((hi - lo)[0] * S), int((hi - lo)[1] * S)
    img = Image.new("RGB", (W, H), (252, 251, 247))
    d = ImageDraw.Draw(img)
    for gx in range(0, W, S):                           # 1 m drafting grid
        d.line([(gx, 0), (gx, H)], fill=(236, 234, 226))
    for gy in range(0, H, S):
        d.line([(0, gy), (W, gy)], fill=(236, 234, 226))
    P = lambda p: ((p[0] - lo[0]) * S, (p[1] - lo[1]) * S)
    tang = np.gradient(centre, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    half = 1.1                                          # corridor half width (m)
    for side in (-1, 1):
        wall = centre + side * half * nrm
        d.line([P(p) for p in wall], fill=(40, 44, 52), width=7, joint="curve")
    # rooms: a box on the far side of the wall at every tagged room-like place
    f_lab, f_small = font(26), font(16)
    for pl in places:
        p = np.array([pl["position"][0], pl["position"][2]])
        i = int(np.argmin(np.linalg.norm(centre - p, axis=1)))
        if pl["kind"] in ("lab", "classroom", "office", "library", "toilet", "canteen"):
            side = 1 if i % 2 == 0 else -1
            c = centre[i] + side * (half + 2.3) * nrm[i]
            t, n = tang[i], nrm[i] * side
            corners = [c + a * 2.6 * t + b * 2.2 * n for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            d.polygon([P(q) for q in corners], fill=(244, 240, 228), outline=(40, 44, 52), width=6)
            door = centre[i] + side * half * nrm[i]
            d.line([P(door - 0.5 * t), P(door + 0.5 * t)], fill=(252, 251, 247), width=9)
            d.arc([*(np.array(P(door)) - 0.9 * S), *(np.array(P(door)) + 0.9 * S)], 0, 90, fill=(120, 120, 120), width=2)
            tx, ty = P(c)
            d.text((tx, ty), pl["name"].upper(), font=f_lab, fill=(30, 30, 30), anchor="mm")
        elif pl["kind"] == "stairs":
            for j in range(-6, 7):
                q = centre[i] + j * 0.28 * tang[i]
                d.line([P(q - half * nrm[i]), P(q + half * nrm[i])], fill=(90, 90, 90), width=3)
            d.text(P(centre[i] + (half + 0.8) * nrm[i]), "STAIRS DN", font=f_small, fill=(60, 60, 60), anchor="mm")
        else:
            d.text(P(centre[i]), pl["name"].upper(), font=f_small, fill=(70, 70, 70), anchor="mm")
    d.rectangle([W - 330, H - 80, W - 20, H - 20], outline=(40, 44, 52), width=3)
    d.text((W - 175, H - 58), "FLOOR PLAN - SKETCH", font=f_small, fill=(30, 30, 30), anchor="mm")
    d.text((W - 175, H - 36), "derived from 3D scan", font=f_small, fill=(110, 110, 110), anchor="mm")
    # deliberately mis-registered: rotate and change scale so alignment is part of the demo
    out = img.rotate(17, expand=True, fillcolor=(252, 251, 247)).resize((int(W * 1.25), int(H * 1.25)))
    dst = ROOT / "demo" / "floorplans" / f"{name}_floor_sketch.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    print(f"wrote {dst.relative_to(ROOT)} ({out.width}x{out.height}); scan is {(hi - lo).round(1).tolist()} m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pcc")
