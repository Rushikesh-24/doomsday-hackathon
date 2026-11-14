"""Multi-floor buildings: walkthrough scans stacked and stitched into one navigable building.

How a building is made
  * A building (platform.db `buildings`) has floors (`floors`: name + level, ground = 0).
  * Any number of scans belong to a building (`building_scans`). A scan starts on one floor and may end on
    another: a video that climbs the stairs from the ground floor to the first floor is a scan from level 0 to
    level 1. Where it changes floor is read from the scan's own (gravity-aligned) camera height: the flat part at
    the start is the lower floor, the flat part at the end the upper floor, and the climb in between becomes a
    stairs connector ("Take the stairs up to the First floor"; wheelchair routes cannot use it).
  * Scans are stitched together:
      - visually (`stitch`): keyframes of one scan are located in another (DINOv2 shortlist + SIFT/RANSAC, the
        camera-guide matcher). Matching keyframes give the rigid transform between the two scans and a junction
        edge in the graph, so the "left" and "right" walks of a floor join, and the bottom / top of a stair
        video join the corridors of the two floors it connects;
      - by name: lift doors / stairwell landings tagged with the same name on several floors ("Lift 1") become
        lift / stairs connectors, and (because shafts are vertical) also tie the floors' positions together;
      - or by hand (yaw, dx, dz per scan).
  * The merged graph (node ids "<scan>|<node>", each node knows its floor) is routed with the normal router,
    and the directions are the single-floor directions per floor leg, joined by lift / stairs steps.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_lock = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS building_meta (building_id TEXT PRIMARY KEY, floor_height REAL);
CREATE TABLE IF NOT EXISTS floor_align (floor_id TEXT PRIMARY KEY, yaw REAL, dx REAL, dz REAL);
CREATE TABLE IF NOT EXISTS connectors (id TEXT PRIMARY KEY, building_id TEXT, kind TEXT, name TEXT, landings TEXT,
    out_of_service INT DEFAULT 0, created TEXT);
CREATE TABLE IF NOT EXISTS building_scans (building_id TEXT, scan TEXT, floor_id TEXT, to_floor_id TEXT, ord INT,
    PRIMARY KEY (building_id, scan));
CREATE TABLE IF NOT EXISTS scan_align (building_id TEXT, scan TEXT, yaw REAL, dx REAL, dz REAL,
    PRIMARY KEY (building_id, scan));
"""
DEFAULT_FLOOR_HEIGHT = 3.5
LIFT_WAIT_M = 8.0          # a lift ride "costs" this much walking plus 2 m per floor (calling and waiting)
STAIRS_M_PER_FLOOR = 2.6   # stair flights per floor, walked length relative to the floor height
CONNECTOR_KINDS = {"lift": "lift", "stairwell": "stairs", "stairs": "stairs"}
ORDINALS = ["Ground", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db():
    from vision.platform import store as PS
    c = PS._db()
    c.executescript(_SCHEMA)
    return c


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def floor_label(level: int) -> str:
    if level < 0:
        return f"Basement {-level}"
    return f"{ORDINALS[level]} floor" if level < len(ORDINALS) else f"Floor {level}"


def _job_ready(rid: str | None) -> bool:
    from vision.spatial import jobs as J
    if not rid or J.get_job(rid) is None:
        return False
    d = J.job_dir(rid)
    return (d / "trajectory.json").exists() and (d / "facilities.json").exists()


def _stitch_path(bid: str) -> Path:
    from vision.config import get_settings
    d = get_settings().output_dir / "buildings" / bid
    d.mkdir(parents=True, exist_ok=True)
    return d / "stitch.json"


# ------------------------------------------------------------------ structure
def building(bid: str) -> dict:
    """Building with its floors, scans (with the floors they span, places and readiness) and connectors."""
    from vision.navigation.places import list_places
    from vision.spatial import jobs as J
    with _lock, _db() as c:
        b = c.execute("SELECT id, name, description, lat, lon FROM buildings WHERE id=?", (bid,)).fetchone()
        if not b:
            raise KeyError(f"building {bid} not found")
        fl = c.execute("SELECT id, name, level FROM floors WHERE building_id=? ORDER BY level", (bid,)).fetchall()
        rows = c.execute("SELECT scan, floor_id, to_floor_id, ord FROM building_scans WHERE building_id=? ORDER BY ord",
                         (bid,)).fetchall()
        legacy = c.execute("SELECT asset_id, floor_id FROM links WHERE asset_kind='reconstruction' AND building_id=?",
                           (bid,)).fetchall()
        meta = c.execute("SELECT floor_height FROM building_meta WHERE building_id=?", (bid,)).fetchone()
        al = {r[0]: {"yaw": r[1], "dx": r[2], "dz": r[3]} for r in c.execute(
            "SELECT scan, yaw, dx, dz FROM scan_align WHERE building_id=?", (bid,))}
        con = c.execute("SELECT id, kind, name, landings, out_of_service FROM connectors WHERE building_id=? "
                        "ORDER BY kind, name", (bid,)).fetchall()
    level = {fid: int(lv) for fid, _, lv in fl}
    seen = {r[0] for r in rows}
    rows = list(rows) + [(rid, fid, None, 1000 + i) for i, (rid, fid) in enumerate(legacy) if rid not in seen]
    scans = []
    for rid, fid, tid, ordv in rows:
        if fid not in level or J.get_job(rid) is None:
            continue
        ready = _job_ready(rid)
        lv0 = level[fid]
        lv1 = level.get(tid, lv0) if tid else lv0
        scans.append({"scan": rid, "name": _scan_name(rid), "floor_id": fid, "to_floor_id": tid if tid else fid,
                      "level": lv0, "to_level": lv1, "ready": ready, "order": ordv,
                      "places": list_places(J.job_dir(rid)) if ready else [], "manual_align": al.get(rid)})
    floors = []
    for fid, name, lv in fl:
        mine = [s for s in scans if s["level"] == lv or s["to_level"] == lv]
        floors.append({"id": fid, "name": name, "level": int(lv), "label": floor_label(int(lv)),
                       "scans": [s["scan"] for s in mine],
                       "scan": next((s["scan"] for s in mine if s["level"] == s["to_level"]), mine[0]["scan"] if mine else None),
                       "ready": any(s["ready"] for s in mine),
                       "places": [{**p, "scan": s["scan"]} for s in mine if s["ready"] for p in s["places"]
                                  if _place_level(s, p) == int(lv)]})
    connectors = [{"id": r[0], "kind": r[1], "name": r[2], "landings": json.loads(r[3] or "[]"),
                   "out_of_service": bool(r[4])} for r in con]
    for k in connectors:
        k["floors"] = sorted({level[x["floor_id"]] for x in k["landings"] if x["floor_id"] in level})
    return {"id": b[0], "name": b[1], "description": b[2], "lat": b[3], "lon": b[4],
            "floor_height": float(meta[0]) if meta and meta[0] else DEFAULT_FLOOR_HEIGHT,
            "floors": floors, "scans": scans, "connectors": connectors}


def _scan_name(rid: str) -> str:
    from vision.spatial import jobs as J
    j = J.get_job(rid)
    v = J.get_video(j["video_id"]) if j else None
    return Path(v["name"]).stem if v and v.get("name") else rid


def _place_level(scan: dict, place: dict) -> int:
    """Floor of a place: single-floor scans -> their floor; scans that change floor -> by the place's height."""
    if scan["level"] == scan["to_level"]:
        return scan["level"]
    from vision.spatial import jobs as J
    lv = node_levels(J.job_dir(scan["scan"]), scan["level"], scan["to_level"])
    y = place["position"][1]
    return lv["level_of_y"](y)


def list_buildings() -> list[dict]:
    with _lock, _db() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM buildings ORDER BY name")]
    out = []
    for bid in ids:
        b = building(bid)
        out.append({"id": b["id"], "name": b["name"], "floors": len(b["floors"]),
                    "floors_ready": sum(f["ready"] for f in b["floors"]), "scans": len(b["scans"]),
                    "connectors": len(b["connectors"])})
    return out


def create_building(name: str, floors: int = 0, floor_height: float | None = None, description: str | None = None) -> dict:
    from vision.platform import store as PS
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ValueError("name must be 1-80 characters")
    b = PS.create_building(name, description, None, None)
    for lv in range(max(0, min(int(floors), 30))):
        PS.create_floor(b["id"], floor_label(lv), lv)
    if floor_height:
        set_floor_height(b["id"], floor_height)
    return building(b["id"])


def set_floor_height(bid: str, h: float) -> None:
    h = float(h)
    if not 2.0 <= h <= 10.0:
        raise ValueError("floor height must be 2-10 m")
    with _lock, _db() as c:
        c.execute("INSERT OR REPLACE INTO building_meta VALUES (?,?)", (bid, h))


def add_scan(bid: str, scan: str, floor_id: str, to_floor_id: str | None = None) -> dict:
    """Add a scan to the building: it starts on `floor_id` and ends on `to_floor_id` (stairs videos) or stays."""
    from vision.spatial import jobs as J
    if J.get_job(scan) is None:
        raise KeyError(f"scan {scan} not found")
    with _lock, _db() as c:
        fl = {r[0] for r in c.execute("SELECT id FROM floors WHERE building_id=?", (bid,))}
        if floor_id not in fl or (to_floor_id and to_floor_id not in fl):
            raise KeyError("floor not found in this building")
        n = c.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM building_scans WHERE building_id=?", (bid,)).fetchone()[0]
        old = c.execute("SELECT ord FROM building_scans WHERE building_id=? AND scan=?", (bid, scan)).fetchone()
        c.execute("INSERT OR REPLACE INTO building_scans VALUES (?,?,?,?,?)",
                  (bid, scan, floor_id, to_floor_id if to_floor_id and to_floor_id != floor_id else None,
                   old[0] if old else n))
        c.execute("DELETE FROM links WHERE asset_kind='reconstruction' AND asset_id=? AND building_id=?", (scan, bid))
    return building(bid)


def remove_scan(bid: str, scan: str) -> dict:
    with _lock, _db() as c:
        c.execute("DELETE FROM building_scans WHERE building_id=? AND scan=?", (bid, scan))
        c.execute("DELETE FROM links WHERE asset_kind='reconstruction' AND asset_id=? AND building_id=?", (scan, bid))
        c.execute("DELETE FROM scan_align WHERE building_id=? AND scan=?", (bid, scan))
    return building(bid)


def align_scan(bid: str, scan: str, align: dict | None) -> dict:
    with _lock, _db() as c:
        if align is None:
            c.execute("DELETE FROM scan_align WHERE building_id=? AND scan=?", (bid, scan))
        else:
            c.execute("INSERT OR REPLACE INTO scan_align VALUES (?,?,?,?,?)",
                      (bid, scan, float(align.get("yaw", 0)), float(align.get("dx", 0)), float(align.get("dz", 0))))
    return building(bid)


def update_floor(bid: str, fid: str, name: str | None = None, level: int | None = None, scan: str | None = "__keep",
                 align: dict | None | str = "__keep") -> dict:
    """Rename / re-level a floor. `scan` sets the floor's single scan (None clears the floor's scans that start
    and end on it); `align` sets the manual placement of that floor's scans."""
    with _lock, _db() as c:
        if not c.execute("SELECT 1 FROM floors WHERE id=? AND building_id=?", (fid, bid)).fetchone():
            raise KeyError(f"floor {fid} not found")
        if name is not None:
            if not name.strip():
                raise ValueError("floor name cannot be empty")
            c.execute("UPDATE floors SET name=? WHERE id=?", (name.strip()[:60], fid))
        if level is not None:
            c.execute("UPDATE floors SET level=? WHERE id=?", (int(level), fid))
    if scan != "__keep":
        for s in building(bid)["scans"]:
            if s["floor_id"] == fid and s["to_floor_id"] == fid:
                remove_scan(bid, s["scan"])
        if scan:
            add_scan(bid, scan, fid)
    if align != "__keep":
        for s in building(bid)["scans"]:
            if s["floor_id"] == fid:
                align_scan(bid, s["scan"], align)
    return building(bid)


def add_floor(bid: str, name: str | None, level: int) -> dict:
    from vision.platform import store as PS
    PS.create_floor(bid, (name or floor_label(int(level))).strip(), int(level))
    return building(bid)


def delete_floor(bid: str, fid: str) -> dict:
    with _lock, _db() as c:
        c.execute("DELETE FROM floors WHERE id=? AND building_id=?", (fid, bid))
        c.execute("DELETE FROM links WHERE floor_id=?", (fid,))
        c.execute("DELETE FROM building_scans WHERE building_id=? AND (floor_id=? OR to_floor_id=?)", (bid, fid, fid))
        for cid, lands in c.execute("SELECT id, landings FROM connectors WHERE building_id=?", (bid,)).fetchall():
            keep = [x for x in json.loads(lands or "[]") if x["floor_id"] != fid]
            c.execute("UPDATE connectors SET landings=? WHERE id=?", (json.dumps(keep), cid))
    return building(bid)


# ------------------------------------------------------------------ floors along a scan
def node_levels(job: Path, lv0: int, lv1: int) -> dict:
    """For a scan that goes from floor lv0 to lv1: the floor of each registered camera, from its height.
    The first / last flat stretches are lv0 / lv1; cameras in between are on the stairs."""
    traj = json.loads((job / "trajectory.json").read_text())
    ys = np.array([c["center_local"][1] for c in traj["cameras"] if c.get("center_local")], float)
    n = len(ys)
    if lv0 == lv1 or n < 4:
        return {"levels": [lv0] * n, "stairs": [False] * n, "y0": float(np.median(ys)) if n else 0.0,
                "y1": float(np.median(ys)) if n else 0.0, "level_of_y": lambda y: lv0}
    k = max(2, n // 8)
    y0, y1 = float(np.median(ys[:k])), float(np.median(ys[-k:]))
    dy = y1 - y0
    if abs(dy) < 1e-6:
        dy = 1e-6
    frac = (ys - y0) / dy                    # 0 on the start floor, 1 on the end floor
    steps = lv1 - lv0

    def level_of_y(y: float) -> int:
        f = (y - y0) / dy
        return int(round(lv0 + max(0.0, min(1.0, f)) * steps))

    stairs = [not (f < 0.18 or f > 0.82) for f in frac]
    # also mark the first / last stair camera's neighbours so the flight starts at a landing node
    levels = [level_of_y(y) for y in ys]
    return {"levels": levels, "stairs": stairs, "y0": y0, "y1": y1, "level_of_y": level_of_y, "frac": frac.tolist()}


# ------------------------------------------------------------------ connectors (lifts, named stairwells)
def save_connector(bid: str, body: dict, cid: str | None = None) -> dict:
    kind = body.get("kind", "lift")
    if kind not in ("lift", "stairs"):
        raise ValueError("kind must be lift or stairs")
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("name is required (e.g. 'Lift 1')")
    b = building(bid)
    places = {f["id"]: {(p["scan"], p["id"]) for p in f["places"]} for f in b["floors"]}
    landings = []
    for x in body.get("landings") or []:
        if x.get("floor_id") not in places:
            raise ValueError(f"unknown floor {x.get('floor_id')}")
        scan = x.get("scan") or next((s for s, pid in places[x["floor_id"]] if pid == x.get("place_id")), None)
        if (scan, x.get("place_id")) not in places[x["floor_id"]]:
            raise ValueError(f"place {x.get('place_id')} is not on that floor")
        landings = [y for y in landings if y["floor_id"] != x["floor_id"]] + [
            {"floor_id": x["floor_id"], "scan": scan, "place_id": x["place_id"]}]
    with _lock, _db() as c:
        if cid:
            if not c.execute("SELECT 1 FROM connectors WHERE id=? AND building_id=?", (cid, bid)).fetchone():
                raise KeyError(f"connector {cid} not found")
            c.execute("UPDATE connectors SET kind=?, name=?, landings=?, out_of_service=? WHERE id=?",
                      (kind, name, json.dumps(landings), int(bool(body.get("out_of_service"))), cid))
        else:
            cid = uuid.uuid4().hex[:10]
            c.execute("INSERT INTO connectors VALUES (?,?,?,?,?,?,?)",
                      (cid, bid, kind, name, json.dumps(landings), int(bool(body.get("out_of_service"))), _now()))
    return next(k for k in building(bid)["connectors"] if k["id"] == cid)


def delete_connector(bid: str, cid: str) -> None:
    with _lock, _db() as c:
        c.execute("DELETE FROM connectors WHERE id=? AND building_id=?", (cid, bid))


def auto_connectors(bid: str) -> dict:
    """Group lift / stairwell places with the same name across floors into connectors (create or update)."""
    b = building(bid)
    groups: dict[tuple[str, str], dict] = {}
    for f in b["floors"]:
        for p in f["places"]:
            kind = CONNECTOR_KINDS.get(p["kind"])
            if not kind:
                continue
            g = groups.setdefault((kind, _norm(p["name"])), {"kind": kind, "name": p["name"], "landings": {}})
            g["landings"].setdefault(f["id"], (p["scan"], p["id"]))
    existing = {(k["kind"], _norm(k["name"])): k for k in b["connectors"]}
    created, updated, single = [], [], []
    for key, g in groups.items():
        lands = [{"floor_id": fid, "scan": s, "place_id": pid} for fid, (s, pid) in g["landings"].items()]
        if len(lands) < 2:
            single.append(f"{g['name']} ({len(lands)} floor)")
            continue
        if key in existing:
            k = existing[key]
            merged = {x["floor_id"]: x for x in k["landings"]}
            merged.update({x["floor_id"]: x for x in lands})
            save_connector(bid, {"kind": k["kind"], "name": k["name"], "out_of_service": k["out_of_service"],
                                 "landings": list(merged.values())}, k["id"])
            updated.append(k["name"])
        else:
            save_connector(bid, {"kind": g["kind"], "name": g["name"], "landings": lands})
            created.append(g["name"])
    return {"created": created, "updated": updated, "only_one_floor": single, "building": building(bid)}


# ------------------------------------------------------------------ visual stitching between scans
def stitch(bid: str, max_queries: int = 12) -> dict:
    """Find where scans overlap by locating keyframes of one scan inside another. Pairs that share a floor are
    tried (including the bottom / top of stairs videos). Saves matches + a rigid transform per pair."""
    from vision.navigation.locate import locate
    from vision.spatial import jobs as J
    b = building(bid)
    scans = [s for s in b["scans"] if s["ready"]]
    info = {}
    for s in scans:
        d = J.job_dir(s["scan"])
        traj = json.loads((d / "trajectory.json").read_text())
        cams = [c for c in traj["cameras"] if c.get("center_local")]
        lv = node_levels(d, s["level"], s["to_level"])
        info[s["scan"]] = {"dir": d, "cams": cams, "levels": lv["levels"], "stairs": lv["stairs"]}
    pairs = []
    for i, x in enumerate(scans):
        for y in scans[i + 1:]:
            shared = set(range(min(x["level"], x["to_level"]), max(x["level"], x["to_level"]) + 1)) & \
                set(range(min(y["level"], y["to_level"]), max(y["level"], y["to_level"]) + 1))
            if shared:
                pairs.append((x["scan"], y["scan"], sorted(shared)))
    results = []
    for xs, ys, shared in pairs:
        X, Y = info[xs], info[ys]
        # query frames: Y keyframes on the shared floors (not on the stairs), spread along the walk
        cand = [k for k, c in enumerate(Y["cams"]) if Y["levels"][k] in shared and not Y["stairs"][k]]
        if not cand:
            continue
        pick = [cand[int(round(t))] for t in np.linspace(0, len(cand) - 1, min(max_queries, len(cand)))]
        matches = []
        for k in dict.fromkeys(pick):
            c = Y["cams"][k]
            try:
                r = locate(X["dir"], (Y["dir"] / "frames" / c["frame"]).read_bytes(), with_scene=False)
            except Exception:
                continue
            kx = int(r["node"][3:])
            # weak matches are kept too: they only count if several agree geometrically (see _pair_transform)
            if r["inliers"] < 10 or X["levels"][kx] not in shared or X["stairs"][kx]:
                continue
            fy = c.get("forward_local") or [1, 0, 0]
            matches.append({"x_node": r["node"], "y_node": f"cam{k}", "inliers": r["inliers"], "confidence": r["confidence"],
                            "x_pos": r["position"], "y_pos": c["center_local"], "x_heading": r["heading"],
                            "y_heading": math.atan2(fy[2], fy[0]), "level": X["levels"][kx]})
        tr = _pair_transform(matches)
        results.append({"x": xs, "y": ys, "shared_levels": shared, "queries": len(pick), "matches": matches,
                        "transform": tr})
    out = {"building_id": bid, "created": _now(), "pairs": results,
           "scans": {s["scan"]: s["name"] for s in scans}}
    _stitch_path(bid).write_text(json.dumps(out, indent=1))
    return stitch_summary(bid)


def _circ_median(a: list[float]) -> float:
    best, val = None, 0.0
    for c in a:
        s = sum(abs(math.atan2(math.sin(x - c), math.cos(x - c))) for x in a)
        if best is None or s < best:
            best, val = s, c
    return val


def _pair_transform(matches: list[dict]) -> dict | None:
    """Rigid 2D transform taking scan Y's floor coordinates (x, z) into scan X's, from located keyframes.
    Yaw from the headings (robust), offset from the positions. A located view must agree with the consensus on
    both direction (< 25 deg) and position (< 2 m): look-alike corridors give matches that point at one spot
    from all along a walk, which fails the position test. Without a high-confidence view, 3 agreeing views are
    needed."""
    if not matches:
        return None
    yaws = [m["x_heading"] - m["y_heading"] for m in matches]
    high = [a for m, a in zip(matches, yaws) if m.get("confidence") == "high"]
    yaw = _circ_median(high or yaws)          # strong views decide the direction when there are any
    keep = [m for m, a in zip(matches, yaws) if abs(math.atan2(math.sin(a - yaw), math.cos(a - yaw))) < math.radians(25)]
    strong = any(m.get("confidence") == "high" for m in keep)
    # without a strong view, most located views must look the same way (a real overlap) - scattered directions
    # mean the matcher is grabbing look-alike corridors
    if not keep or (not strong and len(keep) < 0.6 * len(matches)):
        return None
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    ts = np.array([np.array([m["x_pos"][0], m["x_pos"][2]]) - R @ np.array([m["y_pos"][0], m["y_pos"][2]]) for m in keep])
    t = np.median(ts, 0)
    good = [m for m, tt in zip(keep, ts) if np.linalg.norm(tt - t) < 2.0]
    if not good or (len(good) < 3 and not any(m.get("confidence") == "high" for m in good)):
        return None
    if len(good) >= 2:
        t = np.mean([tt for tt in ts if np.linalg.norm(tt - t) < 2.0], 0)
    return {"yaw": float(yaw), "dx": float(t[0]), "dz": float(t[1]), "n": len(good),
            "inliers": int(sum(m["inliers"] for m in good)), "good": [m["y_node"] for m in good]}


def stitch_summary(bid: str) -> dict | None:
    p = _stitch_path(bid)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"created": s["created"], "pairs": [
        {"x": r["x"], "y": r["y"], "x_name": s["scans"].get(r["x"], r["x"]), "y_name": s["scans"].get(r["y"], r["y"]),
         "shared_levels": r["shared_levels"], "queries": r["queries"], "matches": len(r["matches"]),
         "linked": bool(r["transform"]), "agreeing": (r["transform"] or {}).get("n", 0)} for r in s["pairs"]]}


def _stitch(bid: str) -> list[dict]:
    p = _stitch_path(bid)
    return json.loads(p.read_text())["pairs"] if p.exists() else []


# ------------------------------------------------------------------ placing scans in the building frame
def _compose(a: dict, b: dict) -> dict:
    """a after b: x -> Ra (Rb x + tb) + ta."""
    c, s = math.cos(a["yaw"]), math.sin(a["yaw"])
    return {"yaw": a["yaw"] + b["yaw"], "dx": c * b["dx"] - s * b["dz"] + a["dx"], "dz": s * b["dx"] + c * b["dz"] + a["dz"]}


def _inverse(t: dict) -> dict:
    c, s = math.cos(-t["yaw"]), math.sin(-t["yaw"])
    return {"yaw": -t["yaw"], "dx": -(c * t["dx"] - s * t["dz"]), "dz": -(s * t["dx"] + c * t["dz"])}


def _rigid2d(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted 2D rotation + translation taking src points onto dst (no scale: scans are metric)."""
    w = w / w.sum()
    cs, cd = (src * w[:, None]).sum(0), (dst * w[:, None]).sum(0)
    H = ((src - cs) * w[:, None]).T @ (dst - cd)
    yaw = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return yaw, cd - R @ cs


def transforms(b: dict) -> dict[str, dict]:
    """Per scan: yaw (rad), dx, dz (scan x/z -> building x/z) and how the placement was found.
    Sources, best first: manual; visual stitching (pairs with the most agreeing matches); shared lift / stairwell
    names (vertical shafts: same x/z on every floor). The first ground-floor scan is the reference."""
    scans = [s for s in b["scans"] if s["ready"]]
    if not scans:
        return {}
    ids = [s["scan"] for s in scans]
    edges: list[tuple[str, str, dict, float, str]] = []   # (x, y, T_x<-y, weight, how)
    for r in _stitch(b["id"]):
        if r["transform"] and r["x"] in ids and r["y"] in ids:
            t = r["transform"]
            edges.append((r["x"], r["y"], {"yaw": t["yaw"], "dx": t["dx"], "dz": t["dz"]}, 10.0 + t["n"],
                          f"stitched to {{other}} ({t['n']} matching views)"))
    # named landings (lift / stairwell / any place with the same name in two scans of the same floor)
    by_name: dict[str, list[tuple[str, dict, int]]] = {}
    for s in scans:
        for p in s["places"]:
            vertical = p["kind"] in CONNECTOR_KINDS
            by_name.setdefault(_norm(p["name"]), []).append((s["scan"], p, _place_level(s, p) if not vertical else -999))
    pair_pts: dict[tuple[str, str], list] = {}
    for name, lst in by_name.items():
        for i, (sa, pa, la) in enumerate(lst):
            for sb, pb, lb in lst[i + 1:]:
                if sa == sb or la != lb:
                    continue
                w = 3.0 if pa["kind"] == "lift" else 1.0
                pair_pts.setdefault((sa, sb), []).append(([pa["position"][0], pa["position"][2]],
                                                         [pb["position"][0], pb["position"][2]], w))
    for (sa, sb), pts in pair_pts.items():
        src = np.array([q for _, q, _ in pts])
        dst = np.array([p for p, _, _ in pts])
        w = np.array([x for _, _, x in pts])
        if len({(round(a, 1), round(c, 1)) for a, c in src.tolist()}) >= 2:
            yaw, t = _rigid2d(src, dst, w)
            how = f"aligned on {len(pts)} shared lift / stair landings"
        else:
            yaw, t = 0.0, (dst - src).mean(0)
            how = "offset from 1 shared landing (rotation not known: adjust by hand if needed)"
        edges.append((sa, sb, {"yaw": yaw, "dx": float(t[0]), "dz": float(t[1])}, float(w.sum()), how + " with {other}"))
    # stacked stairwells: a stairs video that starts on the floor where another one ended climbs the same shaft,
    # one floor higher. Stairwells look identical floor to floor (so visual matching cannot tell them apart), but
    # that also means the second climb sits right above the first: same start position and direction.
    for sb in scans:
        if sb["level"] == sb["to_level"]:
            continue
        below = next((sa for sa in scans if sa["to_level"] == sb["level"] and sa["level"] != sa["to_level"]
                      and sa["scan"] != sb["scan"]), None)
        if below is None:
            continue
        a0, b0 = _start_pose(b, below["scan"]), _start_pose(b, sb["scan"])
        yaw = a0[2] - b0[2]
        c, s_ = math.cos(yaw), math.sin(yaw)
        t = {"yaw": yaw, "dx": a0[0] - (c * b0[0] - s_ * b0[1]), "dz": a0[1] - (s_ * b0[0] + c * b0[1])}
        edges.append((below["scan"], sb["scan"], t, 6.0, "stacked over {other} (same stairwell)"))
    out: dict[str, dict] = {}
    for s in scans:
        if s["manual_align"]:
            a = s["manual_align"]
            out[s["scan"]] = {"yaw": math.radians(a["yaw"]), "dx": a["dx"], "dz": a["dz"], "source": "manual"}
    # reference: a ground-floor walk in the biggest linked group (an unlinked walk must not cut routes off)
    comp = {sid: sid for sid in ids}

    def find(x):
        while comp[x] != x:
            x = comp[x]
        return x
    for x, y, *_ in edges:
        if x in comp and y in comp:
            comp[find(x)] = find(y)
    size = {}
    for sid in ids:
        size[find(sid)] = size.get(find(sid), 0) + 1
    ref = min(scans, key=lambda s: (-size[find(s["scan"])], s["level"] != 0, abs(s["level"]),
                                    s["to_level"] != s["level"], s["order"]))
    if ref["scan"] not in out:
        out[ref["scan"]] = {"yaw": 0.0, "dx": 0.0, "dz": 0.0, "source": "reference (first ground-floor scan)"}
    # grow from placed scans along the strongest constraint each time
    while True:
        best = None
        for x, y, t, w, how in edges:
            if x in out and y not in out:
                cand = (_compose(out[x], t), y, how.replace("{other}", _scan_name(x)))
            elif y in out and x not in out:
                cand = (_compose(out[y], _inverse(t)), x, how.replace("{other}", _scan_name(y)))
            else:
                continue
            if best is None or w > best[0]:
                best = (w, cand)
        if best is None:
            break
        tr, sid, how = best[1]
        out[sid] = {**tr, "source": how}
    shift = 0.0
    for s in scans:
        if s["scan"] not in out:
            shift += 30.0
            out[s["scan"]] = {"yaw": 0.0, "dx": shift, "dz": 0.0,
                              "source": "not linked: run 'Stitch scans' or tag the same lift / stairwell in both"}
    return out


def _start_pose(b: dict, scan: str) -> tuple[float, float, float]:
    """x, z and heading (rad) of a scan's first camera, in its own frame."""
    from vision.spatial import jobs as J
    traj = json.loads((J.job_dir(scan) / "trajectory.json").read_text())
    c = next(c for c in traj["cameras"] if c.get("center_local"))
    f = c.get("forward_local") or [1, 0, 0]
    return c["center_local"][0], c["center_local"][2], math.atan2(f[2], f[0])


def _tx2(t: dict, p) -> list[float]:
    c, s = math.cos(t["yaw"]), math.sin(t["yaw"])
    return [round(c * p[0] - s * p[1] + t["dx"], 3), round(s * p[0] + c * p[1] + t["dz"], 3)]


# ------------------------------------------------------------------ merged graph
def merged_graph(bid: str) -> dict:
    from vision.navigation.graph import build
    from vision.navigation.places import attach_places
    from vision.spatial import jobs as J
    b = building(bid)
    T = transforms(b)
    fh = b["floor_height"]
    nodes, edges, places, steps = [], [], [], []
    units, scale = "m", "approximate_metric"
    used_levels = set()
    stair_flights = []
    for s in b["scans"]:
        if not s["ready"] or s["scan"] not in T:
            continue
        d = J.job_dir(s["scan"])
        g = attach_places(build(d), d)
        t = T[s["scan"]]
        lv = node_levels(d, s["level"], s["to_level"])
        pre = f"{s['scan']}|"
        if g["units"] != "m":
            units, scale = g["units"], g["scale_status"]
        steps.append(g["median_step"])
        cam_level = {f"cam{k}": lv["levels"][k] for k in range(len(lv["levels"]))}
        cam_stairs = {f"cam{k}": lv["stairs"][k] for k in range(len(lv["stairs"]))}
        attach = {n["id"]: n.get("attached_to") for n in g["nodes"]}

        def level_of(nid):
            base = attach.get(nid) or nid
            return cam_level.get(base, s["level"])

        dy = (lv["y1"] - lv["y0"]) or 1e-6
        for n in g["nodes"]:
            p = n["pos"]
            L = level_of(n["id"])
            used_levels.add(L)
            if s["level"] == s["to_level"]:
                y = L * fh   # one floor: flat (short phone scans drift in height; that is not a slope)
            else:   # stairs videos: heights mapped linearly between the two floors
                y = (s["level"] + (p[1] - lv["y0"]) / dy * (s["to_level"] - s["level"])) * fh
            xz = _tx2(t, [p[0], p[2]])
            nodes.append({**n, "id": pre + n["id"], "pos": [xz[0], round(y, 4), xz[1]], "floor": L, "scan": s["scan"],
                          **({"attached_to": pre + n["attached_to"]} if n.get("attached_to") else {})})
        flight = {"scan": s["scan"], "name": s["name"], "from": s["level"], "to": s["to_level"], "nodes": []}
        for e in g["edges"]:
            ne = {**e, "a": pre + e["a"], "b": pre + e["b"]}
            if s["level"] != s["to_level"] and (cam_stairs.get(e["a"]) or cam_stairs.get(e["b"]) or
                                                level_of(e["a"]) != level_of(e["b"])):
                ne["tags"] = sorted({*e["tags"], "stairs", "connector"})
                ne.update({"connector": f"flight-{s['scan']}", "connector_name": "the stairs", "connector_kind": "stairs"})
                flight["nodes"] += [ne["a"], ne["b"]]
            edges.append(ne)
        if flight["nodes"]:
            # draw the climb in walking order: stair cameras plus one on each side (the landings)
            st = [k for k, x in enumerate(lv["stairs"]) if x]
            ks = range(max(0, st[0] - 1), min(len(lv["stairs"]), st[-1] + 2)) if st else []
            flight["nodes"] = [f"{pre}cam{k}" for k in ks]
            stair_flights.append(flight)
        for p in g["places"]:
            L = level_of(p["id"])
            q = next(n for n in nodes if n["id"] == pre + p["id"])
            places.append({**p, "id": pre + p["id"], "position": q["pos"], "floor": L,
                           "floor_name": next((f["name"] for f in b["floors"] if f["level"] == L), floor_label(L)),
                           "attached_to": pre + p["attached_to"], "scan": s["scan"], "is_start": bool(p.get("is_start"))})
    byid = {n["id"]: n for n in nodes}
    # junctions between stitched scans: the matched keyframes are the same spot
    for r in _stitch(bid):
        if not r["transform"]:
            continue
        good = set(r["transform"]["good"])
        for m in r["matches"]:
            a, c = f"{r['x']}|{m['x_node']}", f"{r['y']}|{m['y_node']}"
            if m["y_node"] not in good or a not in byid or c not in byid or byid[a]["floor"] != byid[c]["floor"]:
                continue
            pa, pc = byid[a]["pos"], byid[c]["pos"]
            edges.append({"a": a, "b": c, "length": round(max(0.3, math.hypot(pa[0] - pc[0], pa[2] - pc[2])), 3),
                          "tags": ["junction"], "facilities": []})
    # stacked stairs: where the lower flight arrives on a floor, the next flight starts
    ready_scans = [s for s in b["scans"] if s["ready"] and s["scan"] in T]
    for sb in ready_scans:
        if sb["level"] == sb["to_level"]:
            continue
        below = next((sa for sa in ready_scans if sa["to_level"] == sb["level"] and sa["level"] != sa["to_level"]
                      and sa["scan"] != sb["scan"]), None)
        if below is None:
            continue
        arrive = [n for n in nodes if n["scan"] == below["scan"] and n["kind"] == "trajectory" and n["floor"] == sb["level"]]
        start = next((n for n in nodes if n["scan"] == sb["scan"] and n["kind"] == "trajectory"), None)
        if arrive and start:
            a = min(arrive, key=lambda n: math.hypot(n["pos"][0] - start["pos"][0], n["pos"][2] - start["pos"][2]))
            edges.append({"a": a["id"], "b": start["id"], "length": round(max(0.5, math.hypot(
                a["pos"][0] - start["pos"][0], a["pos"][2] - start["pos"][2])), 3), "tags": ["junction"], "facilities": []})
    # junctions by name: the same place name in two scans on the same floor is the same spot
    seen_names: dict[tuple[str, int], str] = {}
    for p in places:
        key = (_norm(p["name"]), p["floor"])
        if key in seen_names and byid[seen_names[key]]["scan"] != p["scan"]:
            edges.append({"a": seen_names[key], "b": p["id"], "length": 0.3, "tags": ["junction"], "facilities": []})
        else:
            seen_names.setdefault(key, p["id"])
    lvl = {f["id"]: f["level"] for f in b["floors"]}
    ids = set(byid)
    connectors = []
    for k in b["connectors"]:
        stops = sorted(((lvl[x["floor_id"]], f"{x.get('scan')}|{x['place_id']}") for x in k["landings"]
                        if x["floor_id"] in lvl and f"{x.get('scan')}|{x['place_id']}" in ids))
        connectors.append({**k, "stops": [s_[1] for s_ in stops], "floors": [s_[0] for s_ in stops]})
        if k["out_of_service"]:
            continue
        for (la, a), (lb, bb) in zip(stops, stops[1:]):
            n = abs(lb - la)
            if k["kind"] == "lift":
                length, tags = LIFT_WAIT_M + 2.0 * n, ["connector", "lift"]
            else:
                length, tags = STAIRS_M_PER_FLOOR * fh * n, ["connector", "stairs"]
            edges.append({"a": a, "b": bb, "length": round(length, 3), "tags": tags, "facilities": [],
                          "connector": k["id"], "connector_name": k["name"], "connector_kind": k["kind"]})
    floors = [{"id": f["id"], "level": f["level"], "name": f["name"], "label": f["label"],
               "scans": [s["scan"] for s in b["scans"] if s["ready"] and (s["level"] == f["level"] or s["to_level"] == f["level"])]}
              for f in b["floors"] if f["level"] in used_levels]
    lowest = min((f["level"] for f in floors), default=0)
    starts = [p for p in places if p["is_start"] and p["floor"] == lowest]
    for p in places:
        p["is_start"] = bool(starts) and p is starts[0]
    ref = next((s["scan"] for s in b["scans"] if s["ready"] and T.get(s["scan"], {}).get("source", "").startswith("reference")), None)
    start = starts[0]["id"] if starts else (f"{ref}|cam0" if ref else (nodes[0]["id"] if nodes else None))
    return {"building_id": bid, "name": b["name"], "nodes": nodes, "edges": edges, "places": places,
            "floors": floors, "connectors": connectors, "flights": stair_flights, "units": units, "scale_status": scale,
            "floor_height": fh, "median_step": float(np.median(steps)) if steps else 1e-6, "start": start,
            "transforms": {k: {"yaw_deg": round(math.degrees(v["yaw"]), 2), "dx": round(v["dx"], 3),
                               "dz": round(v["dz"], 3), "source": v["source"]} for k, v in T.items()},
            "note": "graph covers the walked corridors; scans are joined where they overlap, and floors by stairs / lifts"}


# ------------------------------------------------------------------ routing + directions
_FLOOR_RE = re.compile(r"\b(?:on|at)?\s*(?:the\s+)?(?:(ground)|(\d+)(?:st|nd|rd|th)?|(first|second|third|fourth|fifth))"
                       r"\s+(?:floor|level)\b|\b(?:floor|level)\s+(\d+)\b", re.I)
_WORD_LEVEL = {"ground": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _split_floor(text: str) -> tuple[str, int | None]:
    m = _FLOOR_RE.search(text)
    if not m:
        return text, None
    lv = 0 if m.group(1) else int(m.group(2) or m.group(4)) if (m.group(2) or m.group(4)) else _WORD_LEVEL[m.group(3).lower()]
    return (text[:m.start()] + text[m.end():]).strip(" ,-()"), lv


def resolve(g: dict, ref: str) -> list[str]:
    """Node id / place name or alias (optionally "... on the 2nd floor") / facility class -> candidate goals."""
    from vision.navigation.places import CLASS_WORDS
    ids = {n["id"] for n in g["nodes"]}
    if ref in ids:
        return [ref]
    if _norm(ref) in ("start", "entrance0", "begin", "groundentrance"):
        return [g["start"]]
    full = _norm(ref)
    hits = [p for p in g["places"] if full == _norm(p["name"]) or full in (_norm(a) for a in p.get("aliases", []))]
    if hits:                                   # an exact place name wins ("Third floor corridor" is a name)
        return [p["id"] for p in hits]
    name, lv = _split_floor(ref)
    key = _norm(name)
    hits = [p for p in g["places"] if key == _norm(p["name"]) or key in (_norm(a) for a in p.get("aliases", []))]
    if lv is not None:
        hits = [p for p in hits if p["floor"] == lv]
    if hits:
        return [p["id"] for p in hits]
    classes = {n.get("class_name") for n in g["nodes"] if n.get("class_name")}
    for cls, words in CLASS_WORDS.items():
        if cls in classes and (key == _norm(cls) or key in (_norm(w) for w in words)):
            return [n["id"] for n in g["nodes"] if n.get("class_name") == cls and (lv is None or n["floor"] == lv)]
    raise KeyError(f"no place called '{ref}' in this building")


def route(bid: str, start: str | None, goal: str, profile: str = "standard", g: dict | None = None) -> dict:
    from vision.navigation.graph import route as one_route
    g = g or merged_graph(bid)
    s = resolve(g, start)[0] if start else g["start"]
    best = None
    for cand in resolve(g, goal):
        r = one_route(g, s, cand, profile)
        if r.get("feasible") and (best is None or r["length"] < best["length"]):
            best = r
    if best is None:
        r = one_route(g, s, resolve(g, goal)[0], profile)
        why = r.get("reason", "no route").replace("within this reconstruction", "in this building")
        down = [k["name"] for k in g["connectors"] if k["out_of_service"]]
        if down:
            why += f" ({', '.join(down)} out of service)"
        return {"feasible": False, "profile": profile, "reason": why, "path": [], "steps": [], "directions": [],
                "start": s, "floors": []}
    return directions(g, best, s)


def directions(g: dict, rt: dict, start: str) -> dict:
    """Single-floor directions for each floor leg, joined by lift / stairs steps."""
    from vision.navigation.places import HELP, directions as one_dirs
    byid = {n["id"]: n for n in g["nodes"]}
    emap = {}
    for e in g["edges"]:
        emap[(e["a"], e["b"])] = e
        emap[(e["b"], e["a"])] = e
    ids = rt["nodes"]
    segs, cur = [], [ids[0]]
    rides = []
    for a, b in zip(ids, ids[1:]):
        e = emap[(a, b)]
        if "connector" in e["tags"]:
            if rides and rides[-1]["connector"] == e["connector"] and rides[-1]["to"] == a and cur == [a]:
                rides[-1]["to"] = b      # same ride continues (lift past a floor, or along a stair flight)
            else:
                segs.append(cur)
                rides.append({"connector": e["connector"], "name": e["connector_name"], "kind": e["connector_kind"],
                              "from": a, "to": b})
            cur = [b]
        else:
            cur.append(b)
    segs.append(cur)
    # a stair flight that starts and ends on the same floor (a few steps) is not a floor change: walk it
    prof = HELP.get(rt["profile"], HELP["standard"])
    names = {f["level"]: f["name"] for f in g["floors"]}
    out = []
    for i, seg in enumerate(segs):
        lv = byid[seg[0]]["floor"]
        if len(seg) > 1:
            sub = {"feasible": True, "profile": rt["profile"], "nodes": seg, "units": rt["units"],
                   "length": sum(emap[(a, b)]["length"] for a, b in zip(seg, seg[1:])),
                   "path": [byid[x]["pos"] for x in seg]}
            d = one_dirs(g, sub)["directions"]
            if i > 0:
                d = [x for x in d if x["kind"] != "start"]
            if i < len(segs) - 1:
                d = [x for x in d if x["kind"] != "arrive"]
        elif i == 0:
            d = [{"kind": "start", "text": f"Start at {byid[seg[0]].get('name') or 'your position'}. {prof['intro']}".strip(),
                  "at": seg[0]}]
        else:
            d = []
        if i == len(segs) - 1 and len(seg) == 1:
            gname = byid[seg[0]].get("name") or "your destination"
            d.append({"kind": "arrive", "text": f"You have arrived at {gname}.", "at": seg[0]})
        for x in d:
            x["floor"] = lv
        out.extend(d)
        if i < len(rides):
            r = rides[i]
            la, lb = byid[r["from"]]["floor"], byid[r["to"]]["floor"]
            n = abs(lb - la)
            way = "up" if lb > la else "down"
            span = f" ({n} floors)" if n > 1 else ""
            if la == lb:
                txt = f"Take {r['name']} ({'a few steps' if r['kind'] == 'stairs' else 'lift'})."
            else:
                txt = f"Take {r['name']} {way} to the {names.get(lb, floor_label(lb))}{span}."
            if r["kind"] == "lift":
                notes = ["Step-free: lift."] if rt["profile"] == "wheelchair" else []
            else:
                notes = [prof["stairs"]] if prof.get("stairs") else []
            out.append({"kind": "lift" if r["kind"] == "lift" else "stairs", "text": txt, "notes": notes,
                        "at": r["from"], "to": r["to"], "floor": la, "to_floor": lb, "connector": r["connector"]})
    start_n, goal_n = byid[ids[0]], byid[ids[-1]]
    rt.update({
        "directions": out,
        "start_name": start_n.get("name") or "your position",
        "goal_name": goal_n.get("name") or (goal_n.get("class_name") or "destination").replace("_", " "),
        "start_floor": start_n["floor"], "goal_floor": goal_n["floor"],
        "floors": list(dict.fromkeys(byid[seg[0]]["floor"] for seg in segs)),   # floors walked on (not passed)
        "path_floors": [byid[x]["floor"] for x in ids],
        "rides": [{"name": r["name"], "kind": r["kind"], "from_floor": byid[r["from"]]["floor"],
                   "to_floor": byid[r["to"]]["floor"]} for r in rides],
        "spoken": " ".join(s["text"] + (" " + " ".join(s["notes"]) if s.get("notes") else "") for s in out),
    })
    return rt


# ------------------------------------------------------------------ render model / phone pack / summary
def _corridor_pieces(job: Path, t: dict, scan: dict) -> dict[int, list[dict]]:
    """The scan's measured corridor, cut into pieces per floor (stair parts are left out), in the building frame."""
    from vision.navigation.floor import ensure_floor
    fl = ensure_floor(job)
    if scan["level"] == scan["to_level"]:
        lvls = [scan["level"]] * len(fl["centre"])
        stairs = [False] * len(fl["centre"])
    else:
        traj = json.loads((job / "trajectory.json").read_text())
        cams = np.array([[c["center_local"][0], c["center_local"][2]] for c in traj["cameras"] if c.get("center_local")])
        lv = node_levels(job, scan["level"], scan["to_level"])
        near = [int(np.argmin(np.linalg.norm(cams - np.array(p), axis=1))) for p in fl["centre"]]
        lvls = [lv["levels"][k] for k in near]
        stairs = [lv["stairs"][k] for k in near]
    pieces: dict[int, list[dict]] = {}
    i = 0
    n = len(lvls)
    while i < n:
        if stairs[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not stairs[j + 1] and lvls[j + 1] == lvls[i]:
            j += 1
        if j - i >= 2:
            pieces.setdefault(lvls[i], []).append({
                "left": [_tx2(t, p) for p in fl["left"][i:j + 1]], "right": [_tx2(t, p) for p in fl["right"][i:j + 1]],
                "centre": [_tx2(t, p) for p in fl["centre"][i:j + 1]], "scan": scan["scan"]})
        i = j + 1
    return pieces


def model(bid: str) -> dict:
    """Everything the stacked 3D building view draws, in the building frame."""
    from vision.spatial import jobs as J
    g = merged_graph(bid)
    b = building(bid)
    T = transforms(b)
    per_floor: dict[int, list[dict]] = {}
    for s in b["scans"]:
        if s["ready"] and s["scan"] in T:
            for lv, pcs in _corridor_pieces(J.job_dir(s["scan"]), T[s["scan"]], s).items():
                per_floor.setdefault(lv, []).extend(pcs)
    floors = []
    for f in g["floors"]:
        pcs = per_floor.get(f["level"], [])
        first = pcs[0] if pcs else {"left": [], "right": [], "centre": []}
        floors.append({**f, "y": f["level"] * g["floor_height"], "segments": pcs,
                       "left": first["left"], "right": first["right"], "centre": first["centre"],
                       "transform": {"source": "; ".join(sorted({g["transforms"][sc]["source"] for sc in f["scans"]
                                                                   if sc in g["transforms"]}))},
                       "places": [p for p in g["places"] if p["floor"] == f["level"]]})
    pos = {n["id"]: n["pos"] for n in g["nodes"]}
    shafts = [{"id": k["id"], "name": k["name"], "kind": k["kind"], "out_of_service": k["out_of_service"],
               "floors": k["floors"], "points": [[pos[s][0], k["floors"][i] * g["floor_height"], pos[s][2]]
                                                 for i, s in enumerate(k["stops"])]} for k in g["connectors"]]
    flights = [{"id": f"flight-{fl['scan']}", "name": f"Stairs ({fl['name']})", "kind": "flight", "out_of_service": False,
                "floors": sorted({fl["from"], fl["to"]}), "points": [pos[x] for x in fl["nodes"] if x in pos]}
               for fl in g["flights"]]
    return {"id": bid, "name": g["name"], "floor_height": g["floor_height"], "floors": floors,
            "connectors": shafts + flights, "start": g["start"], "transforms": g["transforms"],
            "stitch": stitch_summary(bid),
            "missing": [{"id": f["id"], "name": f["name"], "level": f["level"],
                         "why": "no scan yet" if not f["scans"] else "scan not finished"}
                        for f in b["floors"] if not f["ready"]]}


def summary(bid: str) -> dict:
    """Is every floor reachable step-free from the entrance? Plus the floors' own scores."""
    from vision.navigation.score import ensure_score
    from vision.spatial import jobs as J
    g = merged_graph(bid)

    def reachable(no_stairs: bool) -> set[str]:
        adj: dict[str, list[str]] = {}
        for e in g["edges"]:
            if no_stairs and "stairs" in e["tags"]:
                continue
            adj.setdefault(e["a"], []).append(e["b"])
            adj.setdefault(e["b"], []).append(e["a"])
        seen, todo = {g["start"]}, [g["start"]]
        while todo:
            for v in adj.get(todo.pop(), []):
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        return seen
    walk, wheel = reachable(False), reachable(True)
    out = []
    for f in g["floors"]:
        mine = {n["id"] for n in g["nodes"] if n["floor"] == f["level"] and n["kind"] == "trajectory"}
        ok, reach = bool(mine & wheel), bool(mine & walk)
        single = [sc for sc in f["scans"]]
        score = None
        for sc in single:
            try:
                s_ = ensure_score(J.job_dir(sc))
                score = {"score": s_["score"], "grade": s_["grade"]}
                break
            except Exception:
                continue
        out.append({"level": f["level"], "name": f["name"], "step_free_from_entrance": ok,
                    "reachable_from_entrance": reach, "score": score})
    lifts = [k for k in g["connectors"] if k["kind"] == "lift"]
    return {"floors": out, "step_free_floors": sum(x["step_free_from_entrance"] for x in out),
            "reachable_floors": sum(x["reachable_from_entrance"] for x in out), "total_floors": len(out),
            "lifts": [{"name": k["name"], "floors": k["floors"], "out_of_service": k["out_of_service"]} for k in lifts]}


def fingerprint(bid: str) -> str:
    from vision.navigation.mappack import _fingerprint
    from vision.spatial import jobs as J
    b = building(bid)
    h = hashlib.sha256(json.dumps({k: b[k] for k in ("name", "floor_height", "connectors")}, sort_keys=True,
                                  default=str).encode())
    for f in b["floors"]:
        h.update(json.dumps([f["id"], f["name"], f["level"]]).encode())
    for s in b["scans"]:
        h.update(json.dumps([s["scan"], s["level"], s["to_level"], s["manual_align"], s["order"]], default=str).encode())
        if s["ready"]:
            h.update(_fingerprint(J.job_dir(s["scan"])).encode())
    p = _stitch_path(bid)
    if p.exists():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def pack(bid: str) -> dict:
    """Offline building pack for the phone: same shape as a map pack (merged graph, places, profiles, help) plus
    the floors (walls / path / places per floor, in the building frame) and the lifts and stairs."""
    from vision.navigation.graph import PROFILES
    from vision.navigation.places import CLASS_WORDS, HELP, KIND_TO_CLASS, KINDS, NEAR_CLASSES
    m = model(bid)
    g = merged_graph(bid)
    if not m["floors"]:
        raise ValueError("no floor of this building has a finished scan yet")
    pts = [p for f in m["floors"] for sg in f["segments"] for p in (sg["left"] + sg["right"] + sg["centre"])]
    pts += [[n["pos"][0], n["pos"][2]] for n in g["nodes"]]
    lo, hi = np.min(pts, 0) - 3, np.max(pts, 0) + 3
    ppu = 1400 / float(max(hi - lo))
    profiles = {k: {kk: (None if isinstance(vv, float) and math.isinf(vv) else vv) for kk, vv in v.items()}
                for k, v in PROFILES.items()}
    floors = []
    for f in m["floors"]:
        segs = [{"left": sg["left"], "right": sg["right"], "centre": sg["centre"],
                 "open_left": [False] * len(sg["left"]), "open_right": [False] * len(sg["right"]), "width_m": []}
                for sg in f["segments"]]
        first_scan = f["scans"][0] if f["scans"] else None
        floors.append({"level": f["level"], "name": f["name"], "label": f["label"], "scan": first_scan,
                       "scans": f["scans"],
                       "yaw_deg": g["transforms"].get(first_scan, {}).get("yaw_deg", 0.0) if first_scan else 0.0,
                       "yaw_by_scan": {sc: g["transforms"][sc]["yaw_deg"] for sc in f["scans"] if sc in g["transforms"]},
                       "floor": segs[0] if segs else {"left": [], "right": [], "centre": [], "open_left": [],
                                                       "open_right": [], "width_m": []},
                       "segments": segs,
                       "path": [n["pos"] for n in g["nodes"] if n["floor"] == f["level"] and n["kind"] == "trajectory"]})
    return {
        "format": 2, "kind": "building", "id": f"b_{bid}", "building_id": bid, "name": g["name"], "description": None,
        "version": fingerprint(bid), "built": _now(), "units": g["units"], "scale_status": g["scale_status"],
        "median_step": g["median_step"], "note": g["note"],
        "nodes": [{k: n[k] for k in ("id", "kind", "pos", "name", "class_name", "place_kind", "floor") if k in n}
                  for n in g["nodes"]],
        "edges": [{"a": e["a"], "b": e["b"], "length": round(e["length"], 5), "tags": e["tags"],
                   **({k: e[k] for k in ("connector", "connector_name", "connector_kind")} if "connector" in e else {})}
                  for e in g["edges"]],
        "places": g["places"], "path": [n["pos"] for n in g["nodes"] if n["kind"] == "trajectory"][:1],
        "start_forward": None, "start": g["start"], "floor_height": g["floor_height"],
        "floors": floors, "connectors": [{k: c[k] for k in ("id", "name", "kind", "floors", "stops", "out_of_service")}
                                         for c in g["connectors"]],
        "plan": {"width": float((hi - lo)[0] * ppu), "height": float((hi - lo)[1] * ppu), "px_per_unit": ppu,
                 "origin": [float(lo[0]), float(lo[1])], "units": g["units"], "image": ""},
        "profiles": profiles, "help": HELP, "near_classes": sorted(NEAR_CLASSES), "class_words": CLASS_WORDS,
        "kind_to_class": KIND_TO_CLASS, "place_kinds": KINDS, "location": None,
    }
