"""Map packs: the self-contained, versioned bundle the mobile app downloads for offline wayfinding.

A pack carries everything routing needs on the device - the navigation graph (walked path + registered
facilities + tagged places, with edge tags), the profile cost tables and the wording used for directions -
plus the floor-plan geometry. The plan image is served separately (`/maps/{id}/plan.png`).
The app runs the same Dijkstra + directions logic (ported 1:1, see mobile/lib/core/router.dart), so offline
answers equal online ones. `version` is a content hash: the app re-downloads only when it changes.

The registry (which scans are offered to the app, and under what name) lives in platform.db `maps`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from vision.config import get_settings

_lock = threading.RLock()
_SCHEMA = """CREATE TABLE IF NOT EXISTS maps (id TEXT PRIMARY KEY, name TEXT, hidden INT DEFAULT 0, description TEXT,
    updated TEXT);"""
PACK_FORMAT = 2   # 2: vector floor (walls) + accessibility score and issues


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(get_settings().output_dir / "platform.db", timeout=30)
    c.executescript(_SCHEMA)
    cols = {r[1] for r in c.execute("PRAGMA table_info(maps)")}
    for col in ("lat", "lng", "radius"):          # v2: where the building is, for the app's "you are here"
        if col not in cols:
            c.execute(f"ALTER TABLE maps ADD COLUMN {col} REAL")
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry() -> dict[str, dict]:
    with _lock, _db() as c:
        rows = c.execute("SELECT id, name, hidden, description, updated, lat, lng, radius FROM maps").fetchall()
    return {r[0]: {"name": r[1], "hidden": bool(r[2]), "description": r[3], "updated": r[4],
                   "location": None if r[5] is None else {"lat": r[5], "lng": r[6], "radius": r[7] or 80.0}}
            for r in rows}


_UNSET = object()


def _check_location(loc) -> dict | None:
    if loc is None:
        return None
    try:
        lat, lng = float(loc["lat"]), float(loc["lng"])
        radius = float(loc.get("radius") or 80.0)
    except (KeyError, TypeError, ValueError):
        raise ValueError("location must be {lat, lng, radius?} in degrees / metres")
    if not (-90 <= lat <= 90 and -180 <= lng <= 180) or not (10 <= radius <= 5000):
        raise ValueError("lat must be -90..90, lng -180..180 and radius 10..5000 m")
    return {"lat": round(lat, 7), "lng": round(lng, 7), "radius": round(radius, 1)}


def set_meta(rid: str, name: str | None = None, hidden: bool | None = None, description: str | None = None,
             location=_UNSET) -> dict:
    """location: {lat, lng, radius?} to pin the map to a place on Earth, None to clear, omitted to keep."""
    cur = registry().get(rid, {"name": None, "hidden": False, "description": None, "location": None})
    if name is not None:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("name must be 1-80 characters")
        cur["name"] = name
    if hidden is not None:
        cur["hidden"] = bool(hidden)
    if description is not None:
        cur["description"] = description.strip()[:300] or None
    if location is not _UNSET:
        cur["location"] = _check_location(location)
    loc = cur.get("location") or {}
    with _lock, _db() as c:
        c.execute("INSERT OR REPLACE INTO maps (id, name, hidden, description, updated, lat, lng, radius) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (rid, cur["name"], int(cur["hidden"]), cur["description"], _now(), loc.get("lat"), loc.get("lng"),
                   loc.get("radius")))
    return {**cur, "id": rid}


def _ready(job: Path) -> bool:
    return (job / "trajectory.json").exists() and (job / "facilities.json").exists()


def _fingerprint(job: Path) -> str:
    """Content hash of everything a pack contains. The plan is generated first (once) so listing and
    downloading agree, and its timestamps are ignored so regenerating identical geometry keeps the version."""
    from vision.navigation.places import ensure_plan
    plan = ensure_plan(job).get("generated") or {}
    h = hashlib.sha256(f"format{PACK_FORMAT}".encode())
    for name in ("trajectory.json", "facilities.json", "places.json", "alignment.json"):
        f = job / name
        if f.exists():
            h.update(name.encode())
            h.update(f.read_bytes())
    h.update(json.dumps({k: plan.get(k) for k in ("width", "height", "px_per_unit", "origin", "points_used")},
                        sort_keys=True).encode())
    from vision.navigation.floor import FORMAT as FLOOR_FORMAT
    h.update(f"floor{FLOOR_FORMAT}".encode())
    return h.hexdigest()[:16]


def _default_name(rid: str) -> str:
    from vision.spatial import jobs as J
    j = J.get_job(rid)
    v = J.get_video(j["video_id"]) if j else None
    stem = Path(v["name"]).stem if v and v.get("name") else rid
    return stem.replace("_", " ").replace("-", " ").strip().title() or rid


def list_maps(include_hidden: bool = False) -> list[dict]:
    """Every finished reconstruction that has a navigation graph, as a map the app can download."""
    from vision.spatial import jobs as J
    from vision.navigation.places import list_places
    reg = registry()
    out = []
    for j in J.list_jobs():
        if not j["status"].startswith("completed"):
            continue
        d = J.job_dir(j["id"])
        if not _ready(d):
            continue
        meta = reg.get(j["id"], {})
        if meta.get("hidden") and not include_hidden:
            continue
        traj = json.loads((d / "trajectory.json").read_text())
        places = list_places(d)
        try:
            from vision.navigation.score import ensure_score
            sc = ensure_score(d)
            score = {"score": sc["score"], "grade": sc["grade"], "issues": sc["counts"]}
        except Exception:
            score = None
        out.append({"id": j["id"], "kind": "floor", "name": meta.get("name") or _default_name(j["id"]), "accessibility": score,
                    "location": meta.get("location"),
                    "description": meta.get("description"), "hidden": bool(meta.get("hidden")),
                    "version": _fingerprint(d), "updated": j["updated"], "places": len(places),
                    "place_names": [p["name"] for p in places][:12], "units": traj.get("units"),
                    "scale_status": traj.get("scale_status"), "length": traj.get("length")})
    out[:0] = _building_maps()
    return out


def _building_maps() -> list[dict]:
    """Multi-floor buildings with at least one finished floor, offered as one map each (id "b_<building id>")."""
    from vision.navigation import building as B
    out = []
    try:
        ids = [b["id"] for b in B.list_buildings() if b["floors_ready"]]
    except Exception:
        return out
    for bid in ids:
        try:
            b = B.building(bid)
            s = B.summary(bid)
            places = [p for f in b["floors"] for p in f["places"]]
            out.append({"id": f"b_{bid}", "kind": "building", "name": b["name"], "description": b.get("description"),
                        "hidden": False, "version": B.fingerprint(bid), "updated": None, "places": len(places),
                        "place_names": [p["name"] for p in places][:12], "units": "m", "location": None,
                        "floors": s["total_floors"], "step_free_floors": s["step_free_floors"],
                        "lifts": [k["name"] for k in b["connectors"] if k["kind"] == "lift"], "accessibility": None})
        except Exception:
            continue
    return out


def build_pack(rid: str) -> dict:
    from vision.navigation.graph import PROFILES, build
    from vision.navigation.places import (CLASS_WORDS, HELP, KIND_TO_CLASS, KINDS, NEAR_CLASSES, attach_places,
                                          ensure_plan)
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    if J.get_job(rid) is None or not _ready(d):
        raise KeyError(f"map {rid} not found or not finished")
    g = attach_places(build(d), d)
    traj = json.loads((d / "trajectory.json").read_text())
    plan = ensure_plan(d).get("generated") or {}
    meta = registry().get(rid, {})
    from vision.navigation.floor import ensure_floor
    from vision.navigation.score import ensure_score
    fl = ensure_floor(d)
    sc = ensure_score(d)
    # JSON has no infinity: forbidden costs travel as null
    profiles = {k: {kk: (None if isinstance(vv, float) and math.isinf(vv) else vv) for kk, vv in v.items()}
                for k, v in PROFILES.items()}
    return {
        "format": PACK_FORMAT, "id": rid, "name": meta.get("name") or _default_name(rid),
        "description": meta.get("description"), "version": _fingerprint(d), "built": _now(),
        "location": meta.get("location"),
        "units": g["units"], "scale_status": g["scale_status"], "median_step": g["median_step"],
        "note": g["note"],
        "nodes": [{k: n[k] for k in ("id", "kind", "pos", "name", "class_name", "place_kind", "frame") if k in n}
                  for n in g["nodes"]],
        "edges": [{"a": e["a"], "b": e["b"], "length": round(e["length"], 5), "tags": e["tags"]} for e in g["edges"]],
        "places": g["places"],
        "path": [c["center_local"] for c in traj["cameras"] if c.get("center_local")],
        "start_forward": next((c.get("forward_local") for c in traj["cameras"] if c.get("forward_local")), None),
        "plan": {k: plan.get(k) for k in ("width", "height", "px_per_unit", "origin", "units")} | {
            "image": f"/maps/{rid}/plan.png"},
        "floor": {k: fl[k] for k in ("centre", "left", "right", "open_left", "open_right", "width_m")},
        "accessibility": {k: sc[k] for k in ("score", "grade", "categories", "issues", "counts", "note")},
        "profiles": profiles, "help": HELP, "near_classes": sorted(NEAR_CLASSES), "class_words": CLASS_WORDS,
        "kind_to_class": KIND_TO_CLASS, "place_kinds": KINDS,
    }


def plan_image(rid: str) -> Path:
    from vision.navigation.places import ensure_plan
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    if J.get_job(rid) is None or not _ready(d):
        raise KeyError(f"map {rid} not found")
    ensure_plan(d)
    return d / "plan" / "generated.png"
