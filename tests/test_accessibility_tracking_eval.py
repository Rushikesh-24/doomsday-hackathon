import numpy as np
import pytest

from vision.accessibility.geometry import dominant_line_angle, estimate_step_edges, yellow_fraction
from vision.evaluation.metrics import evaluate, summarize_results
from vision.schemas import BBox, Detection, SourceRef
from vision.tracking.tracker import IoUTracker


def stairs_image(n_steps=8, h=400, w=300, skew=0):
    """Alternating tread/riser bands, as a frontal staircase photo would produce."""
    img = np.full((h, w, 3), 200, np.uint8)
    step_h = h // n_steps
    for i in range(n_steps):
        y = i * step_h
        img[y:y + step_h // 3] = 60  # dark riser shadow band -> one strong edge per step
    if skew:
        import cv2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


@pytest.mark.parametrize("n", [5, 8, 12])
def test_step_edge_estimate(n):
    est = estimate_step_edges(stairs_image(n))
    # each band contributes a top and bottom edge; the heuristic should lock on to ~n or ~2n peaks
    assert est["count"] is not None
    assert abs(est["count"] - n) <= 1 or abs(est["count"] - 2 * n) <= 2, est


def test_step_estimate_on_flat_region_is_not_confident():
    est = estimate_step_edges(np.full((200, 200, 3), 128, np.uint8))
    assert est["confidence"] < 0.25
    assert estimate_step_edges(stairs_image(5, h=100))["count"] is None  # too small to resolve


def test_dominant_angle_sign():
    import cv2
    img = np.zeros((200, 300), np.uint8)
    cv2.line(img, (10, 150), (290, 50), 255, 3)   # rises to the right
    ang, n = dominant_line_angle(img)
    assert n >= 1 and ang < -10


def test_yellow_fraction():
    y = np.zeros((10, 10, 3), np.uint8); y[:] = (250, 210, 0)
    assert yellow_fraction(y) > 0.9 and yellow_fraction(np.zeros((10, 10, 3), np.uint8)) == 0


def D(i, cls, box, conf=0.8):
    return Detection(id=str(i), class_name=cls, category="facility", confidence=conf,
                     bbox=BBox.from_xyxy(box, 1000, 1000), source=SourceRef(name="v"),
                     model_name="m", model_version="1", timestamp="t")


def test_tracker_keeps_identity_across_motion_and_gaps():
    tr = IoUTracker(min_hits=2, max_age=2)
    ids = []
    for f in range(6):
        if f == 3:  # missed detection in one frame
            tr.update([], f, f)
            continue
        x = 100 + 60 * f  # moves 60px per sampled frame (IoU still > 0.3 for 200px box)
        dets = [D(f, "staircase", [x, 100, x + 200, 300]), D(100 + f, "door", [800, 100, 900, 400])]
        tr.update(dets, f, f)
        ids.append((dets[0].track_id, dets[1].track_id))
    assert len({a for a, _ in ids}) == 1 and len({b for _, b in ids}) == 1
    assert len(tr.confirmed_tracks()) == 2


def test_tracker_large_jump_uses_centre_fallback_and_class_gate():
    tr = IoUTracker(min_hits=1)
    a = D(0, "door", [0, 0, 100, 200]); tr.update([a], 0, 0)
    b = D(1, "door", [50, 0, 150, 200]); tr.update([b], 1, 1)        # IoU 0.33 -> match
    c = D(2, "ramp", [50, 0, 150, 200]); tr.update([c], 2, 2)        # other class -> new track
    assert a.track_id == b.track_id != c.track_id


def test_tracker_retires_and_single_hits_unconfirmed():
    tr = IoUTracker(min_hits=2, max_age=1)
    tr.update([D(0, "ramp", [0, 0, 50, 50])], 0, 0)
    for f in range(1, 4):
        tr.update([], f, f)
    assert tr.active == [] and len(tr.finished) == 1 and tr.confirmed_tracks() == []


def test_evaluate_precision_recall_ap():
    gt = {"a.jpg": [{"class": "door", "bbox": [0, 0, 10, 10]}, {"class": "door", "bbox": [20, 0, 30, 10]}],
          "b.jpg": [{"class": "door", "bbox": [0, 0, 10, 10]}]}
    pred = {"a.jpg": [{"class": "door", "bbox": [0, 0, 10, 10], "confidence": 0.9},
                      {"class": "door", "bbox": [50, 50, 60, 60], "confidence": 0.8},   # FP
                      {"class": "door", "bbox": [0, 0, 10, 10], "confidence": 0.3}],    # duplicate -> FP
            "b.jpg": [{"class": "door", "bbox": [1, 0, 10, 10], "confidence": 0.7}]}
    r = evaluate(pred, gt)["per_class"]["door"]
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)
    assert r["precision"] == 0.5 and r["recall"] == pytest.approx(2 / 3, abs=1e-4)
    # PR points: (1/3,1),(1/3,.5),(2/3,2/3),(2/3,.5) -> AP = 1/3*1 + 1/3*2/3
    assert r["ap@0.5"] == pytest.approx(1 / 3 + 2 / 9, abs=1e-4)


def test_summarize_results(blob_pipeline, tmp_path):
    from tests.conftest import png_bytes, scene
    rs = [blob_pipeline.analyze_image(png_bytes(scene()), f"{i}.png", save_dir=tmp_path) for i in range(2)]
    s = summarize_results(rs)
    assert s["images"] == 2 and s["counts_by_class"]["staircase"] == 2
    assert sum(s["confidence_histogram"]["counts"]) == s["detections"]
