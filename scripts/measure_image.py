#!/usr/bin/env python
"""Measure accessibility geometry in a photo.

    python scripts/measure_image.py samples/images/ramp_hull_museum.jpg
    python scripts/measure_image.py photo.jpg --ref "x1,y1,x2,y2=1.00"            # tape-measured segment (px, m)
    python scripts/measure_image.py photo.jpg --object "<detection_id>:height=2.10"
    python scripts/measure_image.py photo.jpg --marker DICT_4X4_50:0.15 --no-priors
    python scripts/measure_image.py photo.jpg --focal35 26 --profile india_hg_2021
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.measurement.engine import get_engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ref", action="append", default=[], help="x1,y1,x2,y2=metres (original-image pixels)")
    ap.add_argument("--object", action="append", default=[], help="detection_id:height|width=metres")
    ap.add_argument("--marker", default=None, help="DICT_NAME:size_m")
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--focal35", type=float, default=None)
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--depth-model", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--vision-request-id", default=None)
    ap.add_argument("--ground-truth", default=None, help="JSON file: [{detection_id,name,value}]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    refs = []
    for r in a.ref:
        pts, L = r.split("=")
        x1, y1, x2, y2 = map(float, pts.split(","))
        refs.append({"type": "segment", "points": [[x1, y1], [x2, y2]], "length_m": float(L)})
    for o in a.object:
        did, rest = o.split(":", 1)
        dim, L = rest.split("=")
        refs.append({"type": "object", "detection_id": did, "dimension": dim, "length_m": float(L)})
    if a.marker:
        d, s = a.marker.split(":")
        refs.append({"type": "marker", "dictionary": d, "size_m": float(s)})
    cal = {"references": refs, "use_priors": not a.no_priors}
    if a.focal35:
        cal["intrinsics"] = {"focal_35mm": a.focal35}
    if a.hfov:
        cal["intrinsics"] = {"hfov_deg": a.hfov}
    gt = json.loads(Path(a.ground_truth).read_text()) if a.ground_truth else None
    r = get_engine().analyze(a.image, calibration=cal, depth_model=a.depth_model, rules_profile=a.profile,
                             vision_request_id=a.vision_request_id, ground_truth=gt,
                             save_dir=Path(a.out) if a.out else None)
    print(f"{r.image.name}  {r.image.width}x{r.image.height}  detections={r.detections_source} ({r.vision_request_id})")
    print(f"camera: {r.camera.source} hfov={r.camera.hfov_deg} deg | depth: {r.scene['depth_model']} | "
          f"gravity: {r.scene['gravity_source']} | scale: {r.scale.status.value} x{r.scale.scale_factor:.3f} "
          f"(+-{r.scale.rel_sigma:.1%})")
    for ref in r.scale.references:
        print(f"  ref {ref.kind}: {ref.description} -> {'used' if ref.used else 'REJECTED: ' + str(ref.reason)}"
              + (f" implies x{ref.implied_scale:.3f}" if ref.implied_scale else ""))
    for n in r.scale.notes:
        print("  note:", n)
    for f in r.facilities:
        print(f"\n[{f.class_name} {f.detection_id}] conf={f.detection_confidence:.2f} mask={f.mask_source} "
              f"truncated={f.truncated_sides or '-'} depth~{f.mean_depth_m}")
        for m in f.measurements:
            if m.status.value in ("ESTIMATED", "VERIFIED"):
                v = m.value
                vs = f"{v:.3f}" if isinstance(v, float) else str(v)
                u = f" +-{m.uncertainty:.3f}" if m.uncertainty is not None else ""
                print(f"  {m.name:<30} {('>= ' if m.lower_bound else '') + vs}{u} {m.unit or ''}  [{m.status.value}, "
                      f"conf {m.confidence:.2f}]")
            elif m.status.value == "REQUIRES_CALIBRATION":
                print(f"  {m.name:<30} -- REQUIRES_CALIBRATION (provisional {m.provisional_value} m, not a measurement)")
            else:
                print(f"  {m.name:<30} -- {m.status.value}: {m.refusal_reason}")
    print(f"\nassessment ({r.assessment_summary.profile}): {r.assessment_summary.counts}")
    for it in r.assessment:
        if it.outcome.value in ("POTENTIAL_ISSUE", "POSSIBLE_ISSUE"):
            print(f"  {it.outcome.value:<16} {it.rule_id:<24} {it.message}")
    print(r.assessment_summary.statement)
    print("annotated:", r.annotated_image, "\njson:", r.json_path)


if __name__ == "__main__":
    main()
