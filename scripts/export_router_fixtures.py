"""Write golden routing fixtures for the mobile app's Dart port (mobile/test/fixtures/router_*.json).

    .venv/Scripts/python scripts/export_router_fixtures.py [map_id ...]   (default: demopcc)
    .venv/Scripts/python scripts/export_router_fixtures.py --building    (synthetic 4-floor building)

Each fixture holds a map pack plus the server's answers to a set of requests; the Dart test
(mobile/test/router_parity_test.dart) must reproduce them exactly. Run it again after changing the
routing or directions logic on either side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUESTS = [
    ("take me to CG lab", "standard", None), ("take me to CG lab", "wheelchair", None),
    ("from upper landing to the graphics lab", "wheelchair", None),
    ("from upper landing to the graphics lab", "low_vision", None), ("from upper landing to the graphics lab", "standard", None),
    ("nearest lift", "wheelchair", None), ("where is the CC lab", "standard", None),
    ("take me to the lobby", "low_vision", "CG Lab"), ("nearest entrance", "standard", "cam40"),
    ("how do I get from the computer centre to the main staircase", "standard", None),
]


def case(g, text, profile, start):
    from vision.navigation.graph import route
    from vision.navigation.places import directions, parse_request, resolve
    try:
        q = parse_request(g, text)
        st = resolve(g, q["start"] or start or "start")
        rt = directions(g, route(g, st, q["goal"], profile))
    except KeyError as e:
        return {"text": text, "profile": profile, "start": start, "error": str(e).strip("'\"")}
    keep = ("feasible", "reason", "goal", "nodes", "length", "units", "warnings", "start_name", "goal_name")
    return {"text": text, "profile": profile, "start": start, "resolved_start": st, "parsed": q,
            "expected": {**{k: rt.get(k) for k in keep},
                         "directions": [{k: s.get(k) for k in ("kind", "text", "notes", "angle")} for s in rt.get("directions", [])]}}


def main(ids: list[str]) -> None:
    from vision.navigation.mappack import build_pack
    from vision.navigation.graph import build
    from vision.navigation.places import attach_places
    from vision.spatial import jobs as J
    out_dir = ROOT / "mobile" / "test" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rid in ids:
        d = J.job_dir(rid)
        g = attach_places(build(d), d)
        pack = build_pack(rid)
        cases = [case(g, *r) for r in REQUESTS]
        (out_dir / f"router_{rid}.json").write_text(json.dumps({"pack": pack, "cases": cases}, indent=1))
        print(f"{rid}: {len(cases)} cases, {sum('error' in c for c in cases)} expected errors")


BUILDING_REQUESTS = [
    (None, "Seminar Hall", "wheelchair"), (None, "Seminar Hall", "standard"), (None, "CG Lab", "low_vision"),
    ("CG Lab", "Office on the ground floor", "wheelchair"), ("Classroom 101", "Toilet", "standard"),
    (None, "Library", "wheelchair"), ("Seminar Hall", "CC Lab", "low_vision"), (None, "Moon base", "standard"),
]


def building_fixture() -> None:
    """A synthetic 4-floor building (tests/test_building.py geometry) in a throw-away output dir; its pack and the
    server's cross-floor routes go to mobile/test/fixtures/building_demo.json."""
    import os
    import tempfile
    os.environ["VISION_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="accessmap_building_")
    import tests.test_building as T
    from vision.navigation import building as B
    T.FRAMES[3] = (160.0, 3.0, 4.0)
    rooms = {0: {"Main entrance": ("entrance", 0.0), "Office": ("office", 6.0), "Library": ("library", 12.0)},
             1: {"CC Lab": ("lab", 7.0), "Classroom 101": ("classroom", 12.0)},
             2: {"CG Lab": ("lab", 12.0), "Toilet": ("toilet", 4.0)}, 3: {"Seminar Hall": ("other", 11.0)}}
    scans = [T._floor_scan(f"demofloor{lv}", lv, r) for lv, r in rooms.items()]
    b = B.create_building("Main Block", floors=4)
    for f, rid in zip(b["floors"], scans):
        B.update_floor(b["id"], f["id"], scan=rid)
    B.auto_connectors(b["id"])
    g = B.merged_graph(b["id"])
    cases = []
    for start, goal, profile in BUILDING_REQUESTS:
        try:
            rt = B.route(b["id"], start, goal, profile, g=g)
        except KeyError as e:
            cases.append({"start": start, "goal": goal, "profile": profile, "error": str(e).strip("'\"")})
            continue
        keep = ("feasible", "reason", "goal", "nodes", "length", "units", "start_name", "goal_name", "floors", "path_floors")
        cases.append({"start": start, "goal": goal, "profile": profile, "expected": {
            **{k: rt.get(k) for k in keep},
            "directions": [{k: s.get(k) for k in ("kind", "text", "notes", "angle", "floor", "to_floor")}
                           for s in rt.get("directions", [])]}})
    out = ROOT / "mobile" / "test" / "fixtures" / "building_demo.json"
    out.write_text(json.dumps({"pack": B.pack(b["id"]), "cases": cases}, indent=1))
    print(f"building: {len(cases)} cases -> {out.name}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--building"]:
        building_fixture()
    else:
        main(sys.argv[1:] or ["demopcc"])
