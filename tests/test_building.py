"""Multi-floor buildings: auto-linked lifts / stairs, floor stacking, cross-floor routes and directions."""
from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

# building frame: every floor is a 12 m corridor along +x at z = 0; the lift door is at x = 2, the stairwell
# landing at x = 10 (both 1 m off the corridor). Each floor's scan has its OWN local frame (rotated / shifted).
LIFT, STAIRS = (2.0, 1.0), (10.0, 1.0)
FRAMES = {0: (0.0, 0.0, 0.0), 1: (90.0, 5.0, -3.0), 2: (-35.0, -4.0, 7.0)}   # yaw deg, dx, dz (building <- local)


def _to_local(level, x, z):
    yaw, dx, dz = FRAMES[level]
    a = math.radians(yaw)
    px, pz = x - dx, z - dz
    return [math.cos(a) * px + math.sin(a) * pz, 0.0, -math.sin(a) * px + math.cos(a) * pz]


def _floor_scan(rid: str, level: int, rooms: dict[str, tuple[str, float]]) -> str:
    from vision.navigation import places as PL
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    (d / "sfm/models/0").mkdir(parents=True, exist_ok=True)
    cams = [{"frame": f"f{i}.jpg", "t": i * 0.5, "center_local": _to_local(level, i * 0.5, 0.0)} for i in range(25)]
    (d / "trajectory.json").write_text(json.dumps({"cameras": cams, "units": "m", "scale_status": "approximate_metric"}))
    (d / "facilities.json").write_text(json.dumps({"facilities": []}))
    (d / "alignment.json").write_text(json.dumps({"T_local_from_sfm": np.eye(4).tolist()}))
    rng = np.random.default_rng(level)
    xs = rng.uniform(-1, 13, 3000)
    side = rng.choice([-1, 1], 3000)
    pts = np.array([_to_local(level, x, s * 1.0) for x, s in zip(xs, side)])
    pts[:, 1] = rng.uniform(-0.9, 0.2, 3000)
    np.savez_compressed(d / "sfm/models/0/points.npz", xyz=pts)
    with sqlite3.connect(J.root() / "spatial.db") as c:
        c.executescript(J._SCHEMA)
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?)", (rid, f"floor{level}.mp4", "x", "{}", J.now()))
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)", (rid, rid, "completed", "{}", "{}", J.now(), J.now(), None))
    for p in PL.list_places(d):
        PL.delete_place(d, p["id"])
    PL.add_place(d, {"name": "Lift 1", "kind": "lift", "position": _to_local(level, *LIFT)})
    PL.add_place(d, {"name": "Main stairs", "kind": "stairwell", "position": _to_local(level, *STAIRS)})
    for name, (kind, x) in rooms.items():
        PL.add_place(d, {"name": name, "kind": kind, "position": _to_local(level, x, 0.6), "is_start": kind == "entrance"})
    return rid


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def bid(client):
    from vision.navigation import building as B
    scans = [_floor_scan("bfloor0", 0, {"Main entrance": ("entrance", 0.0), "Office": ("office", 6.0)}),
             _floor_scan("bfloor1", 1, {"CC Lab": ("lab", 7.0)}),
             _floor_scan("bfloor2", 2, {"CG Lab": ("lab", 12.0), "Toilet": ("toilet", 4.0)})]
    b = B.create_building("Test Block", floors=3)
    for f, rid in zip(b["floors"], scans):
        B.update_floor(b["id"], f["id"], scan=rid)
    B.auto_connectors(b["id"])
    return b["id"]


def test_auto_connectors_and_stacking(bid):
    from vision.navigation import building as B
    r = B.auto_connectors(bid)                                      # already linked by the fixture
    assert r["created"] == [] and sorted(r["updated"]) == ["Lift 1", "Main stairs"]
    b = r["building"]
    lift = next(k for k in b["connectors"] if k["kind"] == "lift")
    assert lift["floors"] == [0, 1, 2] and lift["name"] == "Lift 1"
    g = B.merged_graph(bid)
    # every floor's lift and stairwell landings land on the same building x/z (the shaft is vertical)
    for name, (x, z) in (("Lift 1", LIFT), ("Main stairs", STAIRS)):
        for p in [p for p in g["places"] if p["name"] == name]:
            assert p["position"][0] == pytest.approx(x, abs=0.05) and p["position"][2] == pytest.approx(z, abs=0.05)
    ys = {f["level"]: np.median([n["pos"][1] for n in g["nodes"] if n["floor"] == f["level"]]) for f in g["floors"]}
    assert ys[1] - ys[0] == pytest.approx(3.5, abs=0.01) and ys[2] - ys[1] == pytest.approx(3.5, abs=0.01)
    assert g["start"] == next(p["id"] for p in g["places"] if p["name"] == "Main entrance")


def test_wheelchair_takes_the_lift_and_directions_say_so(bid):
    from vision.navigation import building as B
    rt = B.route(bid, None, "CG Lab", "wheelchair")
    assert rt["feasible"] and rt["floors"] == [0, 2]
    assert rt["rides"] == [{"name": "Lift 1", "kind": "lift", "from_floor": 0, "to_floor": 2}]
    texts = [d["text"] for d in rt["directions"]]
    assert texts[0].startswith("Start at Main entrance")
    assert "Take Lift 1 up to the Second floor (2 floors)." in texts
    assert texts[-1].startswith("You have arrived at CG Lab")
    assert {d["floor"] for d in rt["directions"]} == {0, 2}
    # floor-qualified names and the way back down
    back = B.route(bid, "CG Lab", "Office on the ground floor", "wheelchair")
    assert back["feasible"] and any(t["text"] == "Take Lift 1 down to the Ground floor (2 floors)." for t in back["directions"])


def test_lift_out_of_service(bid):
    from vision.navigation import building as B
    lift = next(k for k in B.building(bid)["connectors"] if k["kind"] == "lift")
    B.save_connector(bid, {**lift, "out_of_service": True}, lift["id"])
    try:
        w = B.route(bid, None, "CG Lab", "wheelchair")
        assert not w["feasible"] and "only via stairs" in w["reason"] and "Lift 1 out of service" in w["reason"]
        s = B.route(bid, None, "CG Lab", "standard")
        assert s["feasible"] and s["rides"][0]["kind"] == "stairs"
        assert "Take Main stairs up to the Second floor (2 floors)." in [d["text"] for d in s["directions"]]
        lv = B.route(bid, None, "CG Lab", "low_vision")
        assert any("handrail" in n for d in lv["directions"] for n in d.get("notes", []))
    finally:
        B.save_connector(bid, {**lift, "out_of_service": False}, lift["id"])


def test_lift_serving_only_some_floors(bid):
    from vision.navigation import building as B
    lift = next(k for k in B.building(bid)["connectors"] if k["kind"] == "lift")
    keep = [x for x in lift["landings"] if B.building(bid)["floors"][1]["id"] != x["floor_id"]]
    B.save_connector(bid, {**lift, "landings": keep}, lift["id"])
    try:
        assert B.building(bid)["connectors"][0]["floors"] == [0, 2]
        w = B.route(bid, None, "CC Lab", "wheelchair")      # floor 1 is only reachable by stairs now
        assert not w["feasible"]
        assert B.route(bid, None, "CG Lab", "wheelchair")["feasible"]
        s = B.summary(bid)
        assert s["step_free_floors"] == 2 and s["total_floors"] == 3
    finally:
        B.save_connector(bid, lift, lift["id"])
    assert B.summary(bid)["step_free_floors"] == 3


def test_building_api_model_pack_and_maps(client, bid):
    r = client.get(f"/buildings/{bid}")
    assert r.status_code == 200 and len(r.json()["floors"]) == 3
    m = client.get(f"/buildings/{bid}/model").json()
    assert [f["level"] for f in m["floors"]] == [0, 1, 2] and len(m["connectors"]) == 2
    assert all(len(f["left"]) > 5 for f in m["floors"])
    rt = client.post(f"/buildings/{bid}/route", json={"to": "Toilet", "profile": "wheelchair"}).json()
    assert rt["feasible"] and rt["goal_name"] == "Toilet"
    assert client.post(f"/buildings/{bid}/route", json={"to": "Moon"}).status_code == 404
    p = client.get(f"/maps/b_{bid}/pack")
    assert p.status_code == 200
    pk = p.json()
    assert pk["kind"] == "building" and len(pk["floors"]) == 3 and pk["profiles"]["wheelchair"]["stairs"] is None
    assert any("lift" in e["tags"] for e in pk["edges"])
    assert client.get(f"/maps/b_{bid}/pack", headers={"If-None-Match": p.headers["etag"]}).status_code == 304
    ms = client.get("/maps").json()["maps"]
    row = next(x for x in ms if x["id"] == f"b_{bid}")
    assert row["kind"] == "building" and row["floors"] == 3
    # manual alignment overrides the automatic one
    fid = client.get(f"/buildings/{bid}").json()["floors"][1]["id"]
    client.patch(f"/buildings/{bid}/floors/{fid}", json={"align": {"yaw": 10, "dx": 1, "dz": 2}})
    m = client.get(f"/buildings/{bid}/model").json()
    t1 = m["transforms"]["bfloor1"]
    assert t1["source"] == "manual" and t1["yaw_deg"] == pytest.approx(10)
    client.patch(f"/buildings/{bid}/floors/{fid}", json={"align": None})


def _walk_scan(rid: str, pts: list[list[float]], places: dict[str, tuple[str, list[float]]]) -> str:
    """A scan whose cameras follow `pts` exactly (identity frame), with tagged places."""
    from vision.navigation import places as PL
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    (d / "sfm/models/0").mkdir(parents=True, exist_ok=True)
    cams = [{"frame": f"f{i}.jpg", "t": i * 0.5, "center_local": p} for i, p in enumerate(pts)]
    (d / "trajectory.json").write_text(json.dumps({"cameras": cams, "units": "m", "scale_status": "approximate_metric"}))
    (d / "facilities.json").write_text(json.dumps({"facilities": []}))
    (d / "alignment.json").write_text(json.dumps({"T_local_from_sfm": np.eye(4).tolist()}))
    np.savez_compressed(d / "sfm/models/0/points.npz", xyz=np.array(pts) + np.array([0, -0.3, 1.0]))
    with sqlite3.connect(J.root() / "spatial.db") as c:
        c.executescript(J._SCHEMA)
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?)", (rid, f"{rid}.mp4", "x", "{}", J.now()))
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)", (rid, rid, "completed", "{}", "{}", J.now(), J.now(), None))
    for name, (kind, pos) in places.items():
        PL.add_place(d, {"name": name, "kind": kind, "position": pos, "is_start": kind == "entrance"})
    return rid


def test_stairs_video_between_floors_and_two_scans_on_one_floor():
    """Ground floor = two walks joined by a shared place name; a stairs video climbs to the first floor."""
    from vision.navigation import building as B
    g_a = _walk_scan("sv_ga", [[x * 0.5, 0, 0] for x in range(13)], {"Entrance": ("entrance", [0, 0, 0]),
                                                                       "Junction": ("other", [6, 0, 0])})
    g_b = _walk_scan("sv_gb", [[6 + x * 0.5, 0, 0] for x in range(13)], {"Junction": ("other", [6, 0, 0]),
                                                                           "Stair foot": ("other", [12, 0, 0])})
    climb = [[12, 0, 0], [12, 0, 0.5]] + [[12, 0.35 * k, 0.5 + 0.4 * k] for k in range(1, 11)] +             [[12 + 0.5 * x, 3.5, 4.5] for x in range(1, 8)]
    st = _walk_scan("sv_st", climb, {"Stair foot": ("other", [12, 0, 0]), "Stair top": ("other", [12.5, 3.5, 4.5])})
    f1 = _walk_scan("sv_f1", [[12.5 + x * 0.5, 3.5, 4.5] for x in range(17)], {"Stair top": ("other", [12.5, 3.5, 4.5]),
                                                                                 "Room 101": ("classroom", [20, 3.5, 4.5])})
    b = B.create_building("Stairs Block", floors=2)
    f0, f1_ = b["floors"][0]["id"], b["floors"][1]["id"]
    B.add_scan(b["id"], g_a, f0)
    B.add_scan(b["id"], g_b, f0)
    B.add_scan(b["id"], st, f0, f1_)
    B.add_scan(b["id"], f1, f1_)
    lv = B.node_levels(B_job(st), 0, 1)
    assert lv["levels"][0] == 0 and lv["levels"][-1] == 1 and any(lv["stairs"])
    bb = B.building(b["id"])
    assert [p["name"] for p in bb["floors"][1]["places"] if p["scan"] == st] == ["Stair top"]
    rt = B.route(b["id"], None, "Room 101", "standard")
    assert rt["feasible"] and rt["floors"] == [0, 1]
    texts = [d["text"] for d in rt["directions"]]
    assert "Take the stairs up to the First floor." in texts and texts[-1].startswith("You have arrived at Room 101")
    w = B.route(b["id"], None, "Room 101", "wheelchair")
    assert not w["feasible"] and "only via stairs" in w["reason"]
    m = B.model(b["id"])
    assert {f["level"] for f in m["floors"]} == {0, 1}
    assert any(c["kind"] == "flight" for c in m["connectors"])
    assert len(next(f for f in m["floors"] if f["level"] == 0)["segments"]) >= 2      # both ground walks drawn


def B_job(rid):
    from vision.spatial import jobs as J
    return J.job_dir(rid)
