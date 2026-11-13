"""Platform API (web app backend): overview, reviews, campus CRUD, navigation graph/routing, frame serving,
AI model registry. Runs against the session's isolated output dir; no model weights are loaded."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_overview_shape(client):
    r = client.get("/platform/overview")
    assert r.status_code == 200
    d = r.json()
    for k in ("images", "detections_by_class", "measurements", "spatial", "compliance", "reviews", "activity"):
        assert k in d
    assert d["images"]["distinct"] >= 0


def test_reviews_roundtrip_and_validation(client):
    assert client.post("/platform/facilities/det-x/review", json={"status": "great"}).status_code == 400
    r = client.post("/platform/facilities/det-x/review", json={"status": "needs_review", "note": "check rail"})
    assert r.status_code == 200
    client.post("/platform/facilities/det-x/review", json={"status": "verified"})
    h = client.get("/platform/facilities/det-x/reviews").json()["items"]
    assert [x["status"] for x in h][:2] in (["verified", "needs_review"], ["needs_review", "verified"])
    from vision.platform.store import latest_reviews
    assert latest_reviews()["det-x"]["status"] == "verified"


def test_campus_crud(client):
    assert client.post("/platform/campus/buildings", json={"name": "  "}).status_code == 400
    b = client.post("/platform/campus/buildings", json={"name": "Library", "description": "north"}).json()
    f = client.post(f"/platform/campus/buildings/{b['id']}/floors", json={"name": "Ground", "level": 0}).json()
    assert client.post("/platform/campus/buildings/nope/floors", json={"name": "x"}).status_code == 404
    assert client.post("/platform/campus/links", json={"asset_kind": "image", "asset_id": "abc",
                                                        "building_id": b["id"], "floor_id": f["id"]}).status_code == 200
    c = client.get("/platform/campus").json()
    bb = next(x for x in c["buildings"] if x["id"] == b["id"])
    assert bb["floors"][0]["name"] == "Ground" and bb["assets"][0]["id"] == "abc"
    assert "image:abc" in c["links"]
    client.delete("/platform/campus/links/image/abc")
    assert "image:abc" not in client.get("/platform/campus").json()["links"]
    client.delete(f"/platform/campus/buildings/{b['id']}")
    assert all(x["id"] != b["id"] for x in client.get("/platform/campus").json()["buildings"])


def _nav_job(with_rail: bool) -> str:
    """Straight walk of 10 cameras; a staircase spans cameras 4-6. Optional handrail alongside."""
    from vision.spatial import jobs as J
    rid = "navtest" + ("rail" if with_rail else "norail")
    d = J.job_dir(rid)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    cams = [{"frame": f"frame_{i:06d}.jpg", "t": i * 0.5, "center_local": [float(i), 0.0, 0.0],
             "center_sfm": [float(i), 0.0, 0.0]} for i in range(10)]
    (d / "trajectory.json").write_text(json.dumps({"cameras": cams, "units": "sfm_units", "scale_status": "relative"}))
    fac = [{"facility_id": "st1", "class_name": "staircase", "position_local": [5.0, 0.0, 0.0],
            "bbox_local": [[4.0, -0.5, -0.5], [6.0, 0.5, 0.5]], "registration_confidence": 0.8},
           {"facility_id": "door1", "class_name": "door", "position_local": [9.0, 0.3, 0.0],
            "registration_confidence": 0.7},
           {"facility_id": "lost", "class_name": "ramp", "position_local": None, "reason": "too few observations"}]
    if with_rail:
        fac.append({"facility_id": "hr1", "class_name": "handrail", "position_local": [5.0, 0.4, 0.0],
                    "bbox_local": [[4.0, 0.3, -0.2], [6.0, 0.5, 0.2]], "registration_confidence": 0.6})
    (d / "facilities.json").write_text(json.dumps({"facilities": fac}))
    (d / "frames" / "frame_000000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return rid


def test_navigation_profiles(client):
    rid = _nav_job(with_rail=False)
    g = client.get(f"/navigation/{rid}/graph").json()
    assert g["facilities_unregistered"] == 1 and "wheelchair" in g["profiles"]
    std = client.post(f"/navigation/{rid}/route", json={"goal": "door", "profile": "standard"}).json()
    assert std["feasible"] and std["goal"] == "door1" and std["units"] == "units"
    assert "stairs_no_handrail" in std["warnings"]
    assert any("no handrail registered" in s["instruction"] for s in std["steps"])
    wc = client.post(f"/navigation/{rid}/route", json={"goal": "door", "profile": "wheelchair"}).json()
    assert not wc["feasible"] and "only via stairs" in wc["reason"]
    assert client.post(f"/navigation/{rid}/route", json={"goal": "elevator"}).status_code == 404
    assert client.post(f"/navigation/{rid}/route", json={"goal": "door", "profile": "x"}).status_code == 400


def test_navigation_handrail_overlap(client):
    rid = _nav_job(with_rail=True)
    r = client.post(f"/navigation/{rid}/route", json={"goal": "door1", "profile": "standard"}).json()
    assert r["feasible"] and "stairs" in r["warnings"] and "stairs_no_handrail" not in r["warnings"]


def test_navigation_missing_job(client):
    assert client.get("/navigation/doesnotexist/graph").status_code == 404


def test_frame_serving_validates_names(client):
    rid = _nav_job(with_rail=False)
    assert client.get(f"/spatial/reconstructions/{rid}/frames/frame_000000.jpg").status_code == 200
    assert client.get(f"/spatial/reconstructions/{rid}/frames/frame_000001.jpg").status_code == 404
    assert client.get(f"/spatial/reconstructions/{rid}/frames/state.json").status_code == 400
    assert client.get(f"/spatial/reconstructions/{rid}/frames/..%2Fstate.json").status_code in (400, 404)


def test_ai_model_registry_licences(client):
    m = client.get("/ai/models").json()
    assert {k: v["license"] for k, v in m.items()} == {"florence2": "MIT", "whisper": "Apache-2.0",
                                                         "dinov2": "Apache-2.0"}


def test_transcribe_rejects_garbage(client):
    r = client.post("/ai/transcribe", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 422


def test_describe_rejects_non_image(client):
    r = client.post("/ai/describe", files={"file": ("x.jpg", b"nope", "image/jpeg")}, data={"task": "ocr"})
    assert r.status_code == 422


def test_unknown_library_item_is_404(client):
    assert client.get("/platform/library/images/doesnotexist").status_code == 404
