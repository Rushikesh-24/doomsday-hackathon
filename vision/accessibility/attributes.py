"""Accessibility intelligence: derive per-detection attributes and scene-level findings.

Every attribute has an explicit `status`:
  detected             - backed directly by model detections (e.g. a handrail box overlaps the stairs)
  estimated            - 2D pixel heuristic (approximate, image-plane only)
  not_detected         - looked for it and found nothing; NOT proof of absence
  requires_measurement - physically meaningful quantity that needs on-site measurement/depth
  not_determinable     - cannot be inferred from a single 2D image with this pipeline
"""
from __future__ import annotations

import numpy as np

from vision.accessibility.geometry import crop, dominant_line_angle, estimate_step_edges, expand, yellow_fraction
from vision.postprocessing.nms import intersection_over_first
from vision.schemas import Attribute, AttributeStatus as S, Detection, SceneFinding

import cv2


def A(value, status, method=None, note=None, confidence=None) -> Attribute:
    return Attribute(value=value, status=status, method=method, note=note,
                     confidence=None if confidence is None else round(float(confidence), 3))


def _related(det: Detection, others: list[Detection], classes: set[str], fx=1.3, fy=1.3,
             min_cover=0.3) -> list[Detection]:
    """Detections of `classes` whose box is at least `min_cover` inside det's expanded box."""
    region = expand(det.bbox.xyxy(), fx, fy)
    return [o for o in others if o.id != det.id and o.class_name in classes
            and intersection_over_first(o.bbox.xyxy(), region) >= min_cover]


def _measure_req(what: str) -> Attribute:
    return A(None, S.requires_measurement, note=f"{what} cannot be measured from a single uncalibrated 2D image; "
                                                 "needs on-site measurement or depth/stereo + camera calibration")


# ---------------------------------------------------------------- per class
def _staircase(d, dets, rgb):
    at = {}
    rails = _related(d, dets, {"handrail"}, 1.4, 1.2)
    at["handrail_present"] = (A(True, S.detected, "handrail detection overlaps expanded stair box",
                                confidence=max(r.confidence for r in rails))
                              if rails else
                              A(False, S.not_detected, "no handrail detection near stairs",
                                note="absence of a detection is not proof that no handrail exists"))
    at["handrail_ids"] = A([r.id for r in rails], S.detected if rails else S.not_detected)
    if len(rails) >= 2:
        cx = (d.bbox.x1 + d.bbox.x2) / 2
        sides = {("left" if (r.bbox.x1 + r.bbox.x2) / 2 < cx else "right") for r in rails}
        at["handrail_sides_visible"] = A(sorted(sides), S.estimated, "handrail box centres vs stair centre")
    tactile = _related(d, dets, {"tactile_paving"}, 1.3, 1.6, 0.2)
    at["tactile_warning_surface"] = (A(True, S.detected, "tactile paving detection at/near stairs",
                                       confidence=max(t.confidence for t in tactile))
                                     if tactile else
                                     A(False, S.not_detected, "no tactile paving detection near stairs",
                                       note="absence of a detection is not proof of absence"))
    est = estimate_step_edges(crop(rgb, d.bbox.xyxy()))
    if est.get("count") is not None and est["count"] >= 2 and est["confidence"] >= 0.25:
        at["visible_step_count"] = A(est["count"], S.estimated,
                                     "row profile of horizontal-edge energy, peaks counted after deskew",
                                     note="counts visible step edges; occlusion/viewpoint can bias it",
                                     confidence=est["confidence"])
    else:
        at["visible_step_count"] = A(est.get("count"), S.not_determinable,
                                     "row profile of horizontal-edge energy",
                                     note=f"edge pattern not regular enough ({est.get('reason', 'low regularity')})",
                                     confidence=est.get("confidence"))
    at["riser_height"] = _measure_req("Riser height")
    at["tread_depth"] = _measure_req("Tread depth")
    at["direction_up_or_down"] = A(None, S.not_determinable, note="ascending vs descending is ambiguous in 2D")
    return at


def _ramp(d, dets, rgb):
    at = {}
    gray = cv2.cvtColor(crop(rgb, d.bbox.xyxy()), cv2.COLOR_RGB2GRAY) if d.bbox.width > 8 else None
    ang, n = dominant_line_angle(gray) if gray is not None else (None, 0)
    if ang is not None:
        side = "right" if ang < 0 else "left"
        at["image_plane_edge_angle_deg"] = A(round(-ang, 1), S.estimated,
                                             "length-weighted median of Hough line angles inside ramp box",
                                             note="angle in the IMAGE plane (perspective-distorted), not the ramp gradient",
                                             confidence=min(1.0, n / 10))
        at["approx_orientation"] = A(f"edges rise toward image {side}" if abs(ang) > 3 else "edges near horizontal (frontal view)",
                                     S.estimated, "sign of dominant edge angle")
    else:
        at["approx_orientation"] = A(None, S.not_determinable, note="no dominant straight edges found")
    rails = _related(d, dets, {"handrail"}, 1.3, 1.3)
    at["handrail_present"] = (A(True, S.detected, "handrail detection overlaps ramp",
                                confidence=max(r.confidence for r in rails))
                              if rails else A(False, S.not_detected, note="absence of detection is not proof"))
    at["handrail_ids"] = A([r.id for r in rails], S.detected if rails else S.not_detected)
    at["gradient_ratio"] = _measure_req("Ramp gradient (e.g. 1:12)")
    at["clear_width"] = _measure_req("Clear width")
    at["landing_present"] = A(None, S.not_determinable)
    return at


def _door(d, dets, rgb):
    at = {}
    ar = d.bbox.height / max(d.bbox.width, 1e-6)
    at["bbox_aspect_ratio_h_over_w"] = A(round(ar, 2), S.estimated, "bounding box geometry")
    # A frontal single door is typically ~2.0-2.6 tall:wide; much taller boxes suggest an oblique view.
    view = "frontal" if ar <= 2.8 else "oblique / side view"
    if d.prompt == "double door" or ar < 1.2:
        view = "frontal (wide / double door)"
    at["viewing_orientation"] = A(view, S.estimated, "box aspect ratio vs typical door proportions",
                                  note="orientation of the door relative to the camera, not its swing direction")
    at["opening_orientation"] = A(None, S.not_determinable,
                                  note="hinge side / swing direction is not reliably inferable from a single "
                                       "2D view; needs annotated training data or on-site survey")
    if d.prompt in ("glass door", "double door"):
        at["door_type"] = A(d.prompt, S.detected, "matched text prompt", confidence=d.confidence)
    signs = _related(d, dets, {"accessibility_symbol", "signboard"}, 1.6, 1.4)
    at["accessibility_indicators"] = (A([{"id": s.id, "class": s.class_name} for s in signs], S.detected,
                                        "sign/symbol detections on or beside the door")
                                      if signs else A([], S.not_detected))
    lifts = [o for o in dets if o.class_name == "elevator"
             and intersection_over_first(d.bbox.xyxy(), o.bbox.xyxy()) > 0.6]
    if lifts:
        at["part_of_elevator"] = A(lifts[0].id, S.detected, "door box lies inside elevator box")
    at["clear_opening_width"] = _measure_req("Clear opening width")
    at["threshold_height"] = _measure_req("Threshold height")
    return at


def _elevator(d, dets, rgb):
    at = {"lift_entrance_detected": A(True, S.detected, confidence=d.confidence)}
    panels = _related(d, dets, {"elevator_panel"}, 2.0, 1.3, 0.5)
    at["status_indicators"] = (A([{"id": p.id, "prompt": p.prompt} for p in panels], S.detected,
                                 "call-button / floor-display detections beside the lift")
                               if panels else A([], S.not_detected,
                                                note="small panels are often missed by open-vocab models"))
    signs = _related(d, dets, {"accessibility_symbol", "signboard"}, 2.0, 1.4)
    at["accessibility_signage"] = (A([{"id": s.id, "class": s.class_name} for s in signs], S.detected)
                                   if signs else A([], S.not_detected))
    at["door_state"] = A(None, S.not_determinable)
    at["car_dimensions"] = _measure_req("Lift car size")
    return at


def _path_relation(d, dets):
    """Relation of an obstacle/person to detected walkways/corridors."""
    paths = [o for o in dets if o.class_name in ("walkway", "corridor")]
    if not paths:
        return {"relative_to_walkway": A("no walkway/corridor detected", S.not_determinable),
                "potential_obstruction": A(None, S.not_determinable,
                                           note="no path detected to compare against")}
    fx, fy = (d.bbox.x1 + d.bbox.x2) / 2, d.bbox.y2  # ground contact point ~ bottom-centre
    best = None
    for p in paths:
        inside = p.bbox.x1 <= fx <= p.bbox.x2 and p.bbox.y1 <= fy <= p.bbox.y2 + 0.05 * p.bbox.height
        if inside:
            ratio = d.bbox.width / max(p.bbox.width, 1e-6)
            if best is None or ratio > best[1]:
                best = (p, ratio)
    if best is None:
        return {"relative_to_walkway": A("outside detected path", S.estimated, "ground-contact point vs path boxes"),
                "potential_obstruction": A(False, S.estimated, "ground-contact point outside path boxes")}
    p, ratio = best
    pc = (p.bbox.x1 + p.bbox.x2) / 2
    lateral = (fx - pc) / max(p.bbox.width / 2, 1e-6)
    pos = "centre" if abs(lateral) < 0.33 else ("left side" if lateral < 0 else "right side")
    return {
        "relative_to_walkway": A({"path_id": p.id, "path_class": p.class_name, "lateral_position": pos},
                                 S.estimated, "ground-contact point inside path box"),
        "image_width_fraction_of_path": A(round(ratio, 3), S.estimated,
                                          "obstacle box width / path box width (image plane)",
                                          note="not a metric clear-width; perspective affects it"),
        "potential_obstruction": A(bool(ratio >= 0.15 or pos == "centre"), S.estimated,
                                   "on path and (centred or >=15% of path width)"),
        "remaining_clear_width": _measure_req("Remaining clear width"),
    }


def _obstacle(d, dets, rgb):
    at = {"obstacle_type": A(d.prompt, S.detected, f"label from {d.detector or 'model'}", confidence=d.confidence)}
    at.update(_path_relation(d, dets))
    return at


def _person(d, dets, rgb):
    at = {"dynamic": A(True, S.detected, note="people are transient; not a permanent obstruction")}
    at.update(_path_relation(d, dets))
    return at


def _handrail(d, dets, rgb):
    hosts = [o for o in dets if o.class_name in ("staircase", "ramp")
             and intersection_over_first(d.bbox.xyxy(), expand(o.bbox.xyxy(), 1.4, 1.2)) >= 0.3]
    return {"associated_with": A([{"id": h.id, "class": h.class_name} for h in hosts],
                                 S.detected if hosts else S.not_detected, "box overlap with stairs/ramp"),
            "grip_height": _measure_req("Handrail height"),
            "continuity": A(None, S.not_determinable)}


def _tactile(d, dets, rgb):
    yf = yellow_fraction(crop(rgb, d.bbox.xyxy()))
    return {"high_contrast_yellow": A(yf >= 0.15, S.estimated, "HSV yellow pixel fraction in box",
                                      note=f"yellow fraction={yf:.2f}", confidence=min(1.0, abs(yf - 0.15) / 0.15)),
            "pattern_type": A(None, S.not_determinable,
                              note="warning (blister) vs guidance (bar) needs a fine-tuned classifier")}


def _path(d, dets, rgb):
    obs = [o for o in dets if o.class_name in ("obstacle", "person")
           and d.bbox.x1 <= (o.bbox.x1 + o.bbox.x2) / 2 <= d.bbox.x2 and d.bbox.y1 <= o.bbox.y2 <= d.bbox.y2 * 1.05]
    return {"objects_on_path": A([{"id": o.id, "class": o.class_name, "type": o.prompt} for o in obs],
                                 S.detected if obs else S.not_detected),
            "clear_width": _measure_req("Clear path width"),
            "surface_condition": A(None, S.not_determinable)}


def _entrance(d, dets, rgb):
    near = lambda cls: _related(d, dets, cls, 2.0, 1.6, 0.3)
    ramps, lifts, stairs = near({"ramp"}), near({"elevator"}), near({"staircase"})
    if ramps or lifts:
        sf = A(True, S.estimated, "ramp/elevator detected near entrance",
               note="visible step-free option; continuity of the route not verified")
    elif stairs:
        sf = A(False, S.estimated, "only stairs detected near entrance",
               note="a step-free route may exist outside the frame")
    else:
        sf = A(None, S.not_determinable)
    doors = near({"door"})
    return {"step_free_access_visible": sf,
            "door_ids": A([x.id for x in doors], S.detected if doors else S.not_detected),
            "entrance_or_exit": A(None, S.not_determinable, note="needs signage text (OCR) or map context")}


def _signboard(d, dets, rgb):
    sym = _related(d, dets, {"accessibility_symbol"}, 1.2, 1.2, 0.5)
    return {"contains_accessibility_symbol": A(bool(sym), S.detected if sym else S.not_detected),
            "sign_kind": A(d.prompt, S.detected, "matched text prompt", confidence=d.confidence),
            "text_content": A(None, S.not_determinable, note="OCR not yet integrated (see DEVELOPMENT_STATUS.md)")}


def _symbol(d, dets, rgb):
    tgt = _related(d, dets, {"door", "elevator", "entrance_exit", "ramp"}, 2.5, 2.0, 0.2)
    return {"linked_facilities": A([{"id": t.id, "class": t.class_name} for t in tgt],
                                   S.estimated if tgt else S.not_detected, "proximity to facility boxes")}


HANDLERS = {
    "staircase": _staircase, "ramp": _ramp, "door": _door, "elevator": _elevator,
    "obstacle": _obstacle, "person": _person, "handrail": _handrail, "tactile_paving": _tactile,
    "corridor": _path, "walkway": _path, "entrance_exit": _entrance, "signboard": _signboard,
    "accessibility_symbol": _symbol,
}


def enrich(detections: list[Detection], rgb: np.ndarray) -> list[SceneFinding]:
    """Fill `attributes` on each detection in place and return scene-level findings."""
    for d in detections:
        h = HANDLERS.get(d.class_name)
        if h:
            try:
                d.attributes.update(h(d, detections, rgb))
            except Exception as e:  # attribute heuristics must never break detection output
                d.attributes["_error"] = A(str(e), S.not_determinable)
    return scene_findings(detections)


def scene_findings(dets: list[Detection]) -> list[SceneFinding]:
    f: list[SceneFinding] = []
    by = lambda c: [d for d in dets if d.class_name == c]
    stairs, ramps, lifts = by("staircase"), by("ramp"), by("elevator")
    no_rail = [s.id for s in stairs if s.attributes.get("handrail_present") and s.attributes["handrail_present"].value is False]
    if no_rail:
        f.append(SceneFinding(code="STAIRS_WITHOUT_VISIBLE_HANDRAIL", severity="warning", detection_ids=no_rail,
                              message="Staircase(s) detected with no handrail detection nearby (verify on site)."))
    if stairs and not (ramps or lifts):
        f.append(SceneFinding(code="NO_STEP_FREE_ALTERNATIVE_IN_VIEW", severity="warning",
                              detection_ids=[s.id for s in stairs],
                              message="Stairs visible but no ramp or lift detected in this view."))
    if ramps or lifts:
        f.append(SceneFinding(code="STEP_FREE_FACILITY_VISIBLE", detection_ids=[x.id for x in ramps + lifts],
                              message="Ramp or lift detected in view."))
    obstr = [d.id for d in dets if d.attributes.get("potential_obstruction")
             and d.attributes["potential_obstruction"].value is True and d.class_name == "obstacle"]
    if obstr:
        f.append(SceneFinding(code="POTENTIAL_PATH_OBSTRUCTION", severity="warning", detection_ids=obstr,
                              message="Object(s) located on a detected walkway/corridor may obstruct the route."))
    tp = by("tactile_paving")
    if tp:
        f.append(SceneFinding(code="TACTILE_PAVING_PRESENT", detection_ids=[t.id for t in tp],
                              message="Tactile paving detected."))
    return f
