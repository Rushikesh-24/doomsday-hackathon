"""Named places, free-text wayfinding, turn-by-turn directions and floor plans (vision/navigation/places.py)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def _l_walk(rid: str) -> str:
    """Metric L-shaped walk: 10 m east (+x) then 6 m south (+z), cameras every metre, a lift beside the corner,
    and a tiny sparse model + identity alignment so a plan can be generated."""
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    (d / "sfm/models/0").mkdir(parents=True, exist_ok=True)
    pts = [[float(x), 0.0, 0.0] for x in range(11)] + [[10.0, 0.0, float(z)] for z in range(1, 7)]
    cams = [{"frame": f"frame_{i:06d}.jpg", "t": i * 0.5, "center_local": p, "center_sfm": p} for i, p in enumerate(pts)]
    (d / "trajectory.json").write_text(json.dumps({"cameras": cams, "units": "m", "scale_status": "approximate_metric"}))
    fac = [{"facility_id": "lift1", "class_name": "elevator", "position_local": [10.6, 0.0, -0.4],
            "registration_confidence": 0.8}]
    (d / "facilities.json").write_text(json.dumps({"facilities": fac}))
    (d / "alignment.json").write_text(json.dumps({"T_local_from_sfm": np.eye(4).tolist(), "scale_status": "approximate_metric"}))
    rng = np.random.default_rng(0)
    walls = np.r_[np.c_[rng.uniform(0, 11, 400), rng.uniform(-1, 1, 400), np.full(400, -1.5)],
                  np.c_[np.full(300, 11.5), rng.uniform(-1, 1, 300), rng.uniform(-1.5, 7, 300)]]
    np.savez_compressed(d / "sfm/models/0/points.npz", xyz=walls)
    return rid


@pytest.fixture(scope="module")
def rid(client):
    return _l_walk("placestest")


def test_place_crud_and_validation(client, rid):
    base = f"/navigation/{rid}/places"
    assert client.post(base, json={"name": "", "position": [0, 0, 0]}).status_code == 400
    assert client.post(base, json={"name": "X", "position": [0, 0]}).status_code == 400
    assert client.post(base, json={"name": "X", "kind": "spaceship", "position": [0, 0, 0]}).status_code == 400
    ent = client.post(base, json={"name": "Main Entrance", "kind": "entrance", "position": [0, 0, 0], "is_start": True})
    assert ent.status_code == 201
    cc = client.post(base, json={"name": "CC Lab", "kind": "lab", "position": [5, 0, 0.5], "aliases": "computer centre"}).json()
    cg = client.post(base, json={"name": "CG Lab", "kind": "lab", "position": [10, 0, 6], "description": "Room 204",
                                 "access": {"step_free": True, "notes": "Wide door"}}).json()
    assert client.post(base, json={"name": "cg lab", "position": [1, 0, 1]}).status_code == 400   # duplicate name
    assert cc["aliases"] == ["computer centre"]
    ps = client.get(base).json()
    assert [p["name"] for p in ps["places"]] == ["Main Entrance", "CC Lab", "CG Lab"] and "lab" in ps["kinds"]
    r = client.patch(f"{base}/{cg['id']}", json={"aliases": ["graphics lab"]})
    assert r.status_code == 200 and r.json()["aliases"] == ["graphics lab"]
    assert client.patch(f"{base}/nope", json={"name": "Y"}).status_code == 404
    tmp = client.post(base, json={"name": "Temp", "position": [2, 0, 0]}).json()
    assert client.delete(f"{base}/{tmp['id']}").status_code == 204
    assert client.delete(f"{base}/{tmp['id']}").status_code == 404
    g = client.get(f"/navigation/{rid}/graph").json()
    assert {n["name"] for n in g["nodes"] if n["kind"] == "place"} == {"Main Entrance", "CC Lab", "CG Lab"}


def test_route_by_name_has_turns_and_landmarks(client, rid):
    r = client.post(f"/navigation/{rid}/route", json={"goal": "cg LAB", "profile": "standard"})
    assert r.status_code == 200
    rt = r.json()
    assert rt["feasible"] and rt["start_name"] == "Main Entrance" and rt["goal_name"] == "CG Lab"
    kinds = [s["kind"] for s in rt["directions"]]
    assert kinds[0] == "start" and kinds[-1] == "arrive" and "turn" in kinds
    turn = next(s for s in rt["directions"] if s["kind"] == "turn")
    assert turn["text"] == "Turn right."           # east then south (+z) is a right turn seen from above (+Y)
    walk_text = " ".join(s["text"] for s in rt["directions"] if s["kind"] == "walk")
    assert "about 10 m" in walk_text and "CC Lab on your right" in walk_text
    assert "Room 204" in rt["directions"][-1]["text"] and "Wide door" in rt["directions"][-1]["text"]
    assert rt["spoken"].startswith("Start at Main Entrance")


def test_ask_free_text(client, rid):
    a = client.post(f"/navigation/{rid}/ask", json={"text": "How do I get from the computer centre to the graphics lab?",
                                                    "profile": "wheelchair"}).json()
    assert a["interpreted"]["start_name"] == "CC Lab" and a["interpreted"]["goal_name"] == "CG Lab"
    assert a["route"]["feasible"] and a["route"]["directions"][0]["text"].startswith("Start at CC Lab. Step-free route")
    lift = client.post(f"/navigation/{rid}/ask", json={"text": "where's the nearest lift"}).json()
    assert lift["route"]["goal"] == "lift1"
    assert client.post(f"/navigation/{rid}/ask", json={"text": "take me to the moon"}).status_code == 404
    assert client.post(f"/navigation/{rid}/ask", json={"text": " "}).status_code == 400


def test_plan_generate_upload_and_align(client, rid):
    p = client.get(f"/navigation/{rid}/plan").json()
    g = p["generated"]
    assert g["width"] > 0 and g["height"] > 0 and g["points_used"] > 0 and p["uploaded"] is None
    img = client.get(g["url"])
    assert img.status_code == 200 and Image.open(io.BytesIO(img.content)).size == (g["width"], g["height"])
    # the walk's start (0,0) and end (10,6) map inside the image
    for x, z in ((0, 0), (10, 6)):
        u, v = (x - g["origin"][0]) * g["px_per_unit"], (z - g["origin"][1]) * g["px_per_unit"]
        assert 0 <= u < g["width"] and 0 <= v < g["height"]
    assert client.patch(f"/navigation/{rid}/plan/transform", json={"scale": 2}).status_code == 400   # nothing uploaded
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    assert client.post(f"/navigation/{rid}/plan/upload", files={"file": ("x.png", b"nope", "image/png")}).status_code == 422
    u = client.post(f"/navigation/{rid}/plan/upload", files={"file": ("floor.png", buf.getvalue(), "image/png")}).json()
    assert u["uploaded"]["name"] == "floor.png" and u["uploaded"]["transform"]["scale"] > 0
    t = client.patch(f"/navigation/{rid}/plan/transform", json={"rotation": 12.5, "x": 30, "opacity": 3}).json()
    assert t["uploaded"]["transform"]["rotation"] == 12.5 and t["uploaded"]["transform"]["opacity"] == 1.0
    assert client.get(t["uploaded"]["url"]).status_code == 200
    assert client.delete(f"/navigation/{rid}/plan/uploaded").json()["uploaded"] is None
    assert client.get(f"/navigation/{rid}/plan/image/uploaded").status_code == 404


def test_sfm_progress_parser():
    from vision.spatial.sfm import parse_progress
    log = (["=== Feature extraction ==="] + [f"Processed file [{i}/10]" for i in range(1, 11)] +
           [f"Processing image [{i}/10]" for i in range(1, 6)])
    f, msg = parse_progress(log, 10)
    assert 0.35 < f < 0.4 and msg == "Matching frames 5/10"
    log += ["Finding good initial image pair"] + [f"Registering image #{i} (num_reg_frames={i})" for i in range(2, 7)]
    f, msg = parse_progress(log, 10)
    assert msg.startswith("Solving camera poses: 6/10") and 0.75 < f < 0.85
    log += ["Finding good initial image pair", "Registering image #9 (num_reg_frames=2)"]
    f2, msg = parse_progress(log, 10)       # a split model keeps counting forward
    assert f2 > f and "8/10" in msg


def test_tagged_stairs_block_wheelchair_and_tagged_lift_is_found(client):
    rid = _l_walk("placesstairs")
    base = f"/navigation/{rid}/places"
    client.post(base, json={"name": "Old stairs", "kind": "stairs", "position": [5, 0, 0]})
    client.post(base, json={"name": "Office", "kind": "office", "position": [10, 0, 6]})
    client.post(base, json={"name": "Service lift", "kind": "lift", "position": [1, 0, 0.5]})
    std = client.post(f"/navigation/{rid}/route", json={"goal": "Office", "profile": "standard"}).json()
    assert std["feasible"] and "stairs" in std["warnings"]
    wc = client.post(f"/navigation/{rid}/route", json={"goal": "Office", "profile": "wheelchair"}).json()
    assert not wc["feasible"] and "only via stairs" in wc["reason"]
    near = client.post(f"/navigation/{rid}/ask", json={"text": "nearest lift please"}).json()
    assert near["route"]["goal_name"] == "Service lift"     # tagged lift beats the detected one further away


def test_sfm_progress_parser_exhaustive_retry():
    from vision.spatial.sfm import parse_progress
    log = ["=== Feature extraction ===", "Processed file [10/10]", "num_reg_frames=4",
           "=== Feature extraction ===", "Processed file [10/10]", "Processing block [2/4, 3/4]"]
    f, msg = parse_progress(log, 10)
    assert msg == "Matching all frame pairs: block 7/16 (retry 2)" and 0.25 < f < 0.5
