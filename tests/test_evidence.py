"""Facility evidence for review: the video frames a 3D facility was seen in, with normalised boxes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.test_mobile_api import _completed_map


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_facility_evidence_frames_and_boxes(client):
    from vision.spatial import jobs as J
    rid = _completed_map("evidencemap")
    d = J.job_dir(rid)
    (d / "frames").mkdir(exist_ok=True)
    for n in (10, 20, 30):
        Image.new("RGB", (400, 800), (90, 90, 90)).save(d / "frames" / f"frame_{n:06d}.jpg")
    fac = json.loads((d / "facilities.json").read_text())
    fac["facilities"].append({
        "facility_id": "stairs1", "class_name": "staircase", "position_local": [5, 0, 0], "registration_confidence": 0.6,
        "observations": [
            {"frame": "frame_000010.jpg", "t": 1.0, "bbox": [40, 80, 200, 400], "confidence": 0.4, "position_sfm": [1, 2, 3]},
            {"frame": "frame_000020.jpg", "t": 2.0, "bbox": [0, 0, 400, 800], "confidence": 0.9, "position_sfm": None,
             "reason": "too far"},
            {"frame": "frame_000030.jpg", "t": 3.0, "bbox": [100, 100, 300, 300], "confidence": 0.7, "position_sfm": [1, 2, 3]},
            {"frame": "frame_000099.jpg", "t": 9.0, "bbox": [0, 0, 1, 1], "confidence": 0.99},          # frame missing
            {"frame": "../secret.jpg", "t": 9.0, "bbox": [0, 0, 1, 1], "confidence": 0.99},             # not a frame
        ]})
    (d / "facilities.json").write_text(json.dumps(fac))
    r = client.get(f"/spatial/reconstructions/{rid}/facilities/stairs1/evidence")
    assert r.status_code == 200
    e = r.json()
    assert e["class_name"] == "staircase" and e["total"] == 3 and e["registered"] is True
    # sightings that placed the facility in 3D first, then by confidence
    assert [o["frame"] for o in e["observations"]] == ["frame_000030.jpg", "frame_000010.jpg", "frame_000020.jpg"]
    assert e["observations"][1]["bbox_n"] == [0.1, 0.1, 0.5, 0.5]
    assert e["observations"][2]["placed"] is False and e["observations"][2]["reason"] == "too far"
    img = client.get(e["observations"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get(f"/spatial/reconstructions/{rid}/frames/..%2Ffacilities.json").status_code == 404
    assert client.get(f"/spatial/reconstructions/{rid}/frames/frame_000099.jpg").status_code == 404
    assert client.get(f"/spatial/reconstructions/{rid}/facilities/nope/evidence").status_code == 404
    assert client.get(f"/spatial/reconstructions/{rid}/facilities/stairs1/evidence?limit=1").json()["total"] == 3
