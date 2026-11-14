"""Corridor measurement, accessibility score, score API, map pack v2 and deleting a reconstruction."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def _corridor(rid: str, stairs: bool = False) -> str:
    """Straight 12 m metric walk along +x; walls 0.8 m each side (1.6 m clear) except a 0.6 m pinch at x 6-8."""
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    (d / "sfm/models/0").mkdir(parents=True, exist_ok=True)
    cams = [{"frame": f"f{i}.jpg", "t": i * 0.5, "center_local": [i * 0.5, 0.0, 0.0]} for i in range(25)]
    (d / "trajectory.json").write_text(json.dumps({"cameras": cams, "units": "m", "scale_status": "approximate_metric"}))
    fac = [{"facility_id": f"{rid}-sign", "class_name": "signboard", "position_local": [1.0, 0.3, 0.7]},
           {"facility_id": f"{rid}-obs", "class_name": "obstacle", "position_local": [9.0, -0.8, 0.4]}]
    if stairs:
        fac.append({"facility_id": f"{rid}-st", "class_name": "staircase", "position_local": [10.5, -0.5, 0],
                    "bbox_local": [[10.0, -1.5, -1.0], [12.5, 1.0, 1.0]]})
    (d / "facilities.json").write_text(json.dumps({"facilities": fac}))
    (d / "alignment.json").write_text(json.dumps({"T_local_from_sfm": np.eye(4).tolist()}))
    rng = np.random.default_rng(1)
    xs = rng.uniform(-1, 13, 6000)
    half = np.where((xs > 6) & (xs < 8), 0.3, 0.8)
    side = rng.choice([-1, 1], 6000)
    pts = np.c_[xs, rng.uniform(-0.9, 0.2, 6000), side * half + rng.normal(0, 0.01, 6000)]
    np.savez_compressed(d / "sfm/models/0/points.npz", xyz=pts)
    with sqlite3.connect(J.root() / "spatial.db") as c:
        c.executescript(J._SCHEMA)
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?)", (rid, f"{rid}.mp4", "x", "{}", J.now()))
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                  (rid, rid, "completed", "{}", "{}", J.now(), J.now(), None))
    return rid


def test_floor_measures_width_and_the_pinch():
    from vision.navigation.floor import analyse, narrow_stretches
    from vision.spatial import jobs as J
    f = analyse(J.job_dir(_corridor("floorwidth")))
    w = np.array([x if x is not None else np.nan for x in f["width_m"]])
    x = np.array([c[0] for c in f["centre"]])
    wide = w[(x > 1.5) & (x < 4.5)]
    assert np.nanmedian(wide) == pytest.approx(1.6, abs=0.15)
    assert np.nanmin(w[(x > 6.6) & (x < 7.4)]) < 0.75
    n = narrow_stretches(f, 0.9)
    assert len(n) == 1 and 6.0 <= n[0]["at"][0] <= 8.0 and n[0]["min_width_m"] < 0.75
    assert len(f["left"]) == len(f["right"]) == len(f["centre"])


def test_score_flags_pinch_obstacle_and_stairs():
    from vision.navigation.score import WEIGHTS, compute
    from vision.spatial import jobs as J
    assert sum(WEIGHTS.values()) == 100
    s = compute(J.job_dir(_corridor("scorestairs", stairs=True)))
    kinds = {i["kind"] for i in s["issues"]}
    assert {"narrow", "obstacle", "no_alternative", "no_handrail"} <= kinds
    assert 0 <= s["score"] < 85 and s["grade"] in "BCDE"
    cat = {c["key"]: c for c in s["categories"]}
    assert cat["vertical"]["score"] == 0 and cat["stair_safety"]["score"] == 0
    assert 0 < cat["clear_width"]["score"] < 1
    assert all(len(i["position"]) == 3 for i in s["issues"])
    assert [i["severity"] for i in s["issues"]] == sorted((i["severity"] for i in s["issues"]),
                                                          key=["high", "medium", "low"].index)


def test_accessibility_api_pack_v2_and_delete(client):
    rid = _corridor("scoreapi")
    r = client.get(f"/spatial/reconstructions/{rid}/accessibility")
    assert r.status_code == 200
    a = r.json()
    assert a["score"] > 0 and len(a["categories"]) == 6 and len(a["corridor"]["left"]) > 10
    m = next(x for x in client.get("/maps").json()["maps"] if x["id"] == rid)
    assert m["accessibility"]["grade"] == a["grade"]
    p = client.get(f"/maps/{rid}/pack").json()
    assert p["format"] == 2 and p["accessibility"]["score"] == a["score"] and len(p["floor"]["left"]) > 10
    assert client.get("/spatial/reconstructions/nope/accessibility").status_code == 404
    # delete removes outputs, job and map
    from vision.spatial import jobs as J
    assert client.delete(f"/spatial/jobs/{rid}").json()["deleted"] is True
    assert not J.job_dir(rid).exists() and J.get_job(rid) is None
    assert rid not in [x["id"] for x in client.get("/maps?include_hidden=true").json()["maps"]]
    assert client.delete(f"/spatial/jobs/{rid}").status_code == 404
