"""Measurement engine tests on analytic ray-cast scenes with exact ground truth (no weights needed).
The fake depth backend can be globally mis-scaled (scale_error) to mimic a monocular network whose
metric scale is wrong, which is exactly what calibration must correct."""
import math

import numpy as np
import pytest

from vision.measurement import synthetic as S
from vision.measurement.camera import from_focal_35mm, resolve_intrinsics
from vision.measurement.engine import MeasurementEngine
from vision.measurement.rules import evaluate_rule, load_rules
from vision.measurement.schemas import (AssessmentOutcome as O, FacilityResult, Measurement, MeasurementStatus as MS,
                                        ScaleStatus)

STAIRS = {"staircase": [f"riser{k}" for k in range(8)] + [f"tread{k}" for k in range(8)]}


def run(scene_fn, groups, calib=None, scale_error=1.0, gt=None, profile=None, **kw):
    png, z, res, fm, truth, sc = S.build(scene_fn, groups, scale_error=scale_error, **kw)
    eng = MeasurementEngine(models=fm, vision_pipeline=object())
    cal = {"intrinsics": sc.K(), **(calib(sc) if callable(calib) else (calib or {}))}
    rep = eng.analyze(png, "syn.png", detections=res.model_dump(), calibration=cal, ground_truth=gt,
                      rules_profile=profile, save=False)
    return rep, truth, sc


def M(rep, name, cls=None):
    for f in rep.facilities:
        if cls and f.class_name != cls:
            continue
        for m in f.measurements:
            if m.name == name:
                return m
    raise KeyError(name)


def ground_ref(z, half=0.4):
    def f(sc):
        A, B = sc.project(np.array([[-half, 0, z], [half, 0, z]]))
        return {"references": [{"type": "segment", "points": [A.tolist(), B.tolist()], "length_m": 2 * half}]}
    return f


# ------------------------------------------------------------------ geometry accuracy (exact GT)
def test_ramp_slope_scale_free_even_when_depth_is_misscaled():
    rep, gt, _ = run(S.ramp_scene, {"ramp": ["ramp"]}, calib={"use_priors": False}, scale_error=1.37)
    m = M(rep, "running_slope_deg")
    assert m.status == MS.ESTIMATED and not m.scale_dependent
    assert abs(m.value - gt["running_slope_deg"]) < 0.3
    assert M(rep, "ramp_length").status == MS.REQUIRES_CALIBRATION  # lengths withheld
    assert M(rep, "ramp_length").value is None and M(rep, "ramp_length").provisional_value > 0


def test_ramp_lengths_after_calibration_correct_network_scale_error():
    rep, gt, _ = run(S.ramp_scene, {"ramp": ["ramp"]}, calib=ground_ref(2.2), scale_error=1.37)
    assert rep.scale.status == ScaleStatus.calibrated_reference
    assert rep.scale.scale_factor == pytest.approx(1 / 1.37, rel=0.03)
    for name, tol in (("clear_width", 0.05), ("ramp_length", 0.07)):
        m = M(rep, name)
        assert m.status == MS.ESTIMATED and m.unit == "m" and m.uncertainty > 0
        assert abs(m.value - gt[name]) / gt[name] < tol, (name, m.value, gt[name])
        assert abs(m.value - gt[name]) < 1.96 * m.uncertainty  # truth inside reported 95% band


def test_steep_ramp_flagged():
    rep, gt, _ = run(S.ramp_scene, {"ramp": ["ramp"]}, slope_deg=9.0, calib=ground_ref(2.2))
    it = next(a for a in rep.assessment if a.rule_id == "ramp_running_slope")
    assert it.outcome in (O.POTENTIAL_ISSUE, O.POSSIBLE_ISSUE)
    assert "compliant" not in rep.assessment_summary.statement.lower().replace("no compliance", "")


def test_door_width_calibrated_and_ratio():
    rep, gt, _ = run(S.door_scene, {"door": ["door"]}, calib=ground_ref(3.0), scale_error=0.8)
    w = M(rep, "doorway_width")
    assert w.status == MS.ESTIMATED and abs(w.value - gt["doorway_width"]) / gt["doorway_width"] < 0.04
    r = M(rep, "width_to_height_ratio")
    assert abs(r.value - 0.9 / 2.1) < 0.02 and r.status == MS.ESTIMATED
    assert M(rep, "threshold_height").status == MS.REQUIRES_MANUAL_MEASUREMENT
    assert w.overlay and w.overlay.lines


def test_door_prior_scale_is_flagged_as_prior():
    rep, gt, _ = run(S.door_scene, {"door": ["door"]}, scale_error=1.3)
    assert rep.scale.status == ScaleStatus.prior_reference_object
    w = M(rep, "doorway_width")
    assert w.status == MS.ESTIMATED and "prior" in w.verification_reason
    assert abs(w.value - 0.9) / 0.9 < 0.06


def test_object_reference_by_detection_id():
    rep, gt, _ = run(S.door_scene, {"door": ["door"]}, scale_error=1.25,
                     calib={"references": [{"type": "object", "detection_id": "syn-000", "dimension": "height",
                                            "length_m": 2.1}]})
    assert rep.scale.status == ScaleStatus.calibrated_reference
    assert M(rep, "doorway_width").value == pytest.approx(0.9, rel=0.04)


def test_stairs_count_riser_tread():
    rep, gt, _ = run(S.stairs_scene, STAIRS, calib=ground_ref(2.2), scale_error=1.2)
    n = M(rep, "step_count")
    assert n.value == gt["step_count"] and not n.scale_dependent
    assert abs(M(rep, "riser_height").value - gt["riser_height"]) < 0.02
    assert abs(M(rep, "tread_depth").value - gt["tread_depth"]) < 0.03
    assert abs(M(rep, "pitch_deg").value - gt["pitch_deg"]) < 3


def test_corridor_width_and_obstacle_clearance():
    def calib(sc):
        A, B = sc.project(np.array([[-0.5, 0, 5.0], [0.5, 0, 5.0]]))
        return {"references": [{"type": "segment", "points": [A.tolist(), B.tolist()], "length_m": 1.0}]}
    rep, gt, sc = run(S.corridor_scene, {"corridor": ["floor", "wl", "wr"], "obstacle": ["box"]}, calib=calib,
                      obstacle=(-0.9, -0.1, 5.5, 0.8))
    w = M(rep, "clear_width")
    assert w.status == MS.ESTIMATED and abs(w.value - 1.8) / 1.8 < 0.06
    c = M(rep, "min_available_clearance")
    assert abs(c.value - gt["min_available_clearance"]) < 0.12
    assert M(rep, "permanent_obstruction_present").value is True


# ------------------------------------------------------------------ refusals
def test_refuses_tiny_object():
    rep, *_ = run(S.door_scene, {"door": ["door"]}, width=0.12, height=0.25, z=25.0, calib=ground_ref(3.0))
    m = M(rep, "doorway_width")
    assert m.status == MS.INSUFFICIENT_DATA and m.value is None and "too small" in m.refusal_reason


def test_refuses_truncated_door_width():
    rep, *_ = run(S.door_scene, {"door": ["door"]}, x0=-2.5, width=1.2, z=3.0, calib=ground_ref(2.6))
    m = M(rep, "doorway_width")
    assert m.status == MS.INSUFFICIENT_DATA and "truncated" in m.refusal_reason


def test_rejects_reference_off_image():
    rep, *_ = run(S.door_scene, {"door": ["door"]}, calib={"use_priors": False, "references": [
        {"type": "segment", "points": [[-50, 10], [5000, 10]], "length_m": 1}]})
    assert rep.scale.status == ScaleStatus.uncalibrated_monocular
    assert rep.scale.references[0].used is False


def test_inconsistent_references_reported():
    def calib(sc):
        A, B = sc.project(np.array([[-0.4, 0, 3.0], [0.4, 0, 3.0]]))
        C, D = sc.project(np.array([[-0.4, 0, 3.3], [0.4, 0, 3.3]]))
        return {"references": [{"type": "segment", "points": [A.tolist(), B.tolist()], "length_m": 0.8},
                               {"type": "segment", "points": [C.tolist(), D.tolist()], "length_m": 2.0}]}
    rep, *_ = run(S.door_scene, {"door": ["door"]}, calib=calib)
    assert any("disagrees" in n for n in rep.scale.notes)


# ------------------------------------------------------------------ verification & rules
def test_ground_truth_marks_verified_and_reports_error():
    rep, *_ = run(S.door_scene, {"door": ["door"]}, calib=ground_ref(3.0),
                  gt=[{"detection_id": "syn-000", "name": "doorway_width", "value": 0.9}])
    m = M(rep, "doorway_width")
    assert m.status == MS.VERIFIED and m.value == 0.9 and not m.verification_required
    assert m.ground_truth.engine_value is not None and abs(m.ground_truth.error) < 0.04


def _m(value, sigma, status=MS.ESTIMATED, lb=False, name="riser_height"):
    return Measurement(id="x", detection_id="d", facility_class="staircase", name=name, value=value, unit="m",
                       uncertainty=sigma, confidence=0.5, status=status, method="t", lower_bound=lb, timestamp="t")


@pytest.mark.parametrize("value,sigma,status,lb,expected", [
    (0.25, 0.02, MS.ESTIMATED, False, O.POTENTIAL_ISSUE),
    (0.19, 0.02, MS.ESTIMATED, False, O.POSSIBLE_ISSUE),
    (0.17, 0.02, MS.ESTIMATED, False, O.POSSIBLE_ISSUE),   # threshold inside the band
    (0.12, 0.01, MS.ESTIMATED, False, O.NO_ISSUE_INDICATED),
    (0.15, 0.0, MS.VERIFIED, False, O.MEETS_THRESHOLD_VERIFIED),
    (0.15, 0.01, MS.ESTIMATED, True, O.CANNOT_ASSESS),     # lower bound can't prove a max rule
    (None, None, MS.REQUIRES_CALIBRATION, False, O.CANNOT_ASSESS),
])
def test_rule_outcomes(value, sigma, status, lb, expected):
    rule = next(r for r in load_rules()["profiles"]["ada_2010"]["rules"] if r["id"] == "stair_riser_max")
    fac = FacilityResult(detection_id="d", class_name="staircase", detection_confidence=0.9, bbox=[0, 0, 1, 1])
    assert evaluate_rule(rule, "ada_2010", fac, _m(value, sigma, status, lb)).outcome == expected


def test_unverified_profile_values_are_not_configured():
    rep, *_ = run(S.door_scene, {"door": ["door"]}, calib=ground_ref(3.0), profile="india_hg_2021")
    it = next(a for a in rep.assessment if a.rule_id == "door_clear_width")
    assert it.outcome == O.NOT_CONFIGURED


def test_absent_handrail_is_possible_not_potential_issue():
    rep, *_ = run(S.stairs_scene, STAIRS, calib=ground_ref(2.2))
    it = next(a for a in rep.assessment if a.rule_id == "stair_handrails")
    assert it.outcome == O.POSSIBLE_ISSUE


# ------------------------------------------------------------------ camera
def test_intrinsics_sources():
    k = from_focal_35mm(26, 4000, 3000, "x", 0.04)
    assert k.fx == pytest.approx(26 * 5000 / math.hypot(36, 24), rel=1e-6)
    # 35mm-equivalence is on the diagonal: 4:3 sensor at 26 mm -> 67.3 deg horizontal
    assert abs(k.hfov_deg - 67.3) < 0.3
    assert resolve_intrinsics(100, 80, None, None, None, 26, 0.15).source == "prior"
    assert resolve_intrinsics(100, 80, None, 28, None, 26, 0.15).source == "exif_focal_35mm"
    assert resolve_intrinsics(100, 80, {"fx": 90}, 28, 70, 26, 0.15).source == "user_intrinsics"


# ------------------------------------------------------------------ multi-image
def test_multi_image_scale_transfer_and_fusion():
    png1, z1, res1, fm1, _, sc1 = S.build(S.door_scene, {"door": ["door"]}, scale_error=0.85)
    png2, z2, res2, fm2, _, sc2 = S.build(S.door_scene, {"door": ["door"]}, scale_error=1.3, z=5.0, x0=-0.6)

    class Switch:  # serve each image's own fake models
        def __init__(self):
            self.cur = fm1

        def __getattr__(self, k):
            return getattr(self.cur, k)

    sw = Switch()
    eng = MeasurementEngine(models=sw, vision_pipeline=object())
    orig = eng.analyze

    def analyze(data, name, **kw):
        sw.cur = fm1 if name == "a.png" else fm2
        return orig(data, name, **kw)
    eng.analyze = analyze
    A, B = sc1.project(np.array([[-0.4, 0, 3.0], [0.4, 0, 3.0]]))
    br = eng.analyze_batch([(png1, "a.png"), (png2, "b.png")],
                           calibrations={"a.png": {"intrinsics": sc1.K(), "references": [
                               {"type": "segment", "points": [A.tolist(), B.tolist()], "length_m": 0.8}]},
                               "b.png": {"intrinsics": sc2.K(), "use_priors": False}},
                           detections={"a.png": res1.model_dump(), "b.png": res2.model_dump()},
                           facility_groups={"door1": [{"image": "a.png", "class": "door"},
                                                      {"image": "b.png", "class": "door"}]}, save=False)
    rb = next(r for r in br.reports if r.image.name == "b.png")
    assert rb.scale.status == ScaleStatus.calibrated_multi_image
    assert M(rb, "doorway_width").value == pytest.approx(0.9, rel=0.06)
    f = next(x for x in br.fused if x["name"] == "doorway_width")
    assert f["n_images"] == 2 and f["value"] == pytest.approx(0.9, rel=0.05) and f["consistent"]


# ------------------------------------------------------------------ API + history
def test_measurement_api_and_history(monkeypatch, tmp_path):
    png, z, res, fm, _, sc = S.build(S.door_scene, {"door": ["door"]})
    import vision.measurement.engine as E
    monkeypatch.setattr(E, "_engine", MeasurementEngine(models=fm, vision_pipeline=object()))
    monkeypatch.setenv("VISION_PRELOAD", "0")
    import json
    from fastapi.testclient import TestClient
    from api.app import app
    A, B = sc.project(np.array([[-0.4, 0, 3.0], [0.4, 0, 3.0]]))
    cal = json.dumps({"intrinsics": sc.K(), "references": [{"type": "segment", "points": [A.tolist(), B.tolist()],
                                                            "length_m": 0.8}]})
    with TestClient(app) as c:
        for _ in range(2):
            r = c.post("/measurements/analyze", files={"file": ("door.png", png, "image/png")},
                       data={"detections": res.model_dump_json(), "calibration": cal})
            assert r.status_code == 200, r.text
        j = r.json()
        assert j["detections_source"] == "provided"
        assert c.get(j["annotated_image"]).status_code == 200
        latest = c.get("/measurements/syn-000").json()
        assert any(m["name"] == "doorway_width" and m["status"] == "ESTIMATED" for m in latest["measurements"])
        hist = c.get("/measurements/syn-000/history").json()
        assert hist["count"] >= 2
        assert c.get("/measurements/nope/history").status_code == 404
        bad = c.post("/measurements/analyze", files={"file": ("door.png", png, "image/png")},
                     data={"calibration": "{not json"})
        assert bad.status_code == 422
        wrong = c.post("/measurements/analyze?detection_ids=zzz", files={"file": ("door.png", png, "image/png")},
                       data={"detections": res.model_dump_json()})
        assert wrong.status_code == 400


def test_aruco_marker_calibration():
    import io

    import cv2
    from PIL import Image
    png, z, res, fm, gt, sc = S.build(S.door_scene, {"door": ["door"]}, scale_error=1.4)
    rgb = np.array(Image.open(io.BytesIO(png)))
    # 0.30 m marker printed on the wall next to the door (plane z = 4.002, x 0.8..1.1, y 1.0..1.3)
    size = 0.30
    corners3d = np.array([[0.8, 1.3, 4.002], [1.1, 1.3, 4.002], [1.1, 1.0, 4.002], [0.8, 1.0, 4.002]])
    dst = sc.project(corners3d).astype(np.float32)
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 200)
    mk = cv2.copyMakeBorder(mk, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    src = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(cv2.cvtColor(mk, cv2.COLOR_GRAY2RGB), Hm, (sc.w, sc.h), borderValue=(0, 0, 0))
    inside = cv2.warpPerspective(np.full((200, 200), 255, np.uint8), Hm, (sc.w, sc.h)) > 0
    quiet = cv2.dilate(inside.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    rgb[quiet] = 255                       # white quiet zone around the marker
    rgb[inside] = warped[inside]
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format="PNG")
    eng = MeasurementEngine(models=fm, vision_pipeline=object())
    rep = eng.analyze(b.getvalue(), "marker.png", detections=res.model_dump(), save=False,
                      calibration={"intrinsics": sc.K(), "use_priors": False,
                                   "references": [{"type": "marker", "dictionary": "DICT_4X4_50", "size_m": size}]})
    assert rep.scale.status == ScaleStatus.calibrated_marker, rep.scale
    assert rep.scale.scale_factor == pytest.approx(1 / 1.4, rel=0.03)
    assert M(rep, "doorway_width").value == pytest.approx(0.9, rel=0.04)
