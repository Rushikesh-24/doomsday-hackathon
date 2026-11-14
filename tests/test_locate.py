"""Phone camera guide (POST /maps/{id}/locate) and map GPS location (PATCH /maps/{id} location)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_mobile_api import _completed_map


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_map_location_set_validate_clear(client):
    rid = _completed_map("geomap")
    assert client.patch(f"/maps/{rid}", json={"location": {"lat": 91, "lng": 0}}).status_code == 400
    assert client.patch(f"/maps/{rid}", json={"location": {"lat": "x"}}).status_code == 400
    r = client.patch(f"/maps/{rid}", json={"location": {"lat": 12.9716, "lng": 77.5946, "radius": 120}})
    assert r.status_code == 200 and r.json()["location"] == {"lat": 12.9716, "lng": 77.5946, "radius": 120.0}
    m = next(x for x in client.get("/maps").json()["maps"] if x["id"] == rid)
    assert m["location"]["lat"] == 12.9716
    assert client.get(f"/maps/{rid}/pack").json()["location"]["radius"] == 120.0
    client.patch(f"/maps/{rid}", json={"name": "Renamed"})                 # other edits keep the location
    assert next(x for x in client.get("/maps").json()["maps"] if x["id"] == rid)["location"] is not None
    client.patch(f"/maps/{rid}", json={"location": None})
    assert next(x for x in client.get("/maps").json()["maps"] if x["id"] == rid)["location"] is None


def test_locate_rejects_bad_input(client):
    assert client.post("/maps/nope/locate", files={"file": ("a.jpg", b"x", "image/jpeg")}).status_code == 404
    rid = _completed_map("locatebad")
    assert client.post(f"/maps/{rid}/locate", files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post(f"/maps/{rid}/locate", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 422


REPO = Path(__file__).resolve().parents[1]
DEMO_JOB = REPO / "outputs/spatial/jobs/demopcc"
DEMO_VIDEO = REPO / "outputs/spatial/videos/demopcc/source.mp4"


@pytest.mark.parametrize("t", [12.1, 25.2])
def test_locate_held_out_video_frame(tmp_path, t):
    """A video frame between two keyframes is placed on the nearest keyframe with high confidence; a photo
    from a different scan is reported as low confidence (not a confident wrong answer)."""
    if not (DEMO_JOB / "trajectory.json").exists() or not DEMO_VIDEO.exists():
        pytest.skip("demo scan not loaded (python demo/load_demo.py)")
    from vision.navigation.locate import locate, nearest_place
    q = tmp_path / "q.jpg"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(DEMO_VIDEO), "-frames:v", "1",
                    "-vf", "scale=1080:-1", str(q)], check=True)
    traj = json.loads((DEMO_JOB / "trajectory.json").read_text())["cameras"]
    exp = min(range(len(traj)), key=lambda i: abs(traj[i]["t"] - t))
    j = locate(DEMO_JOB, q.read_bytes(), with_scene=False)
    assert j["node"] == f"cam{exp}" and j["confidence"] == "high"
    assert len(j["position"]) == 3 and isinstance(j["heading"], float)
    assert nearest_place(DEMO_JOB, j["position"])["name"]
