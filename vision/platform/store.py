"""Platform persistence (SQLite): facility reviews, campus structure (buildings / floors) and asset links.
Nothing is pre-seeded: the campus is whatever users create."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from vision.config import get_settings

_lock = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, entity_id TEXT, entity_kind TEXT, status TEXT, note TEXT,
    reviewer TEXT, created TEXT);
CREATE INDEX IF NOT EXISTS idx_rev_entity ON reviews(entity_id, created);
CREATE TABLE IF NOT EXISTS buildings (id TEXT PRIMARY KEY, name TEXT, description TEXT, lat REAL, lon REAL,
    created TEXT);
CREATE TABLE IF NOT EXISTS floors (id TEXT PRIMARY KEY, building_id TEXT, name TEXT, level INT, created TEXT);
CREATE TABLE IF NOT EXISTS links (asset_kind TEXT, asset_id TEXT, building_id TEXT, floor_id TEXT, created TEXT,
    PRIMARY KEY (asset_kind, asset_id));
"""
REVIEW_STATUSES = ("verified", "rejected", "needs_review", "fixed")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(get_settings().output_dir / "platform.db", timeout=30)
    c.executescript(_SCHEMA)
    return c


# ------------------------------------------------------------------ reviews
def add_review(entity_id: str, entity_kind: str, status: str, note: str | None, reviewer: str | None) -> dict:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status must be one of {REVIEW_STATUSES}")
    rid = uuid.uuid4().hex[:12]
    with _lock, _db() as c:
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?)", (rid, entity_id, entity_kind, status, note, reviewer,
                                                                  now()))
    return {"id": rid, "entity_id": entity_id, "status": status, "note": note, "reviewer": reviewer, "created": now()}


def latest_reviews() -> dict[str, dict]:
    with _lock, _db() as c:
        rows = c.execute("SELECT entity_id, status, note, reviewer, created FROM reviews ORDER BY created").fetchall()
    return {r[0]: {"status": r[1], "note": r[2], "reviewer": r[3], "created": r[4]} for r in rows}


def review_history(entity_id: str) -> list[dict]:
    with _lock, _db() as c:
        rows = c.execute("SELECT id, status, note, reviewer, created FROM reviews WHERE entity_id=? ORDER BY created DESC",
                         (entity_id,)).fetchall()
    return [{"id": r[0], "status": r[1], "note": r[2], "reviewer": r[3], "created": r[4]} for r in rows]


# ------------------------------------------------------------------ campus
def create_building(name: str, description: str | None, lat: float | None, lon: float | None) -> dict:
    bid = uuid.uuid4().hex[:10]
    with _lock, _db() as c:
        c.execute("INSERT INTO buildings VALUES (?,?,?,?,?,?)", (bid, name, description, lat, lon, now()))
    return {"id": bid, "name": name, "description": description, "lat": lat, "lon": lon}


def delete_building(bid: str) -> None:
    with _lock, _db() as c:
        c.execute("DELETE FROM buildings WHERE id=?", (bid,))
        c.execute("DELETE FROM floors WHERE building_id=?", (bid,))
        c.execute("DELETE FROM links WHERE building_id=?", (bid,))


def create_floor(building_id: str, name: str, level: int) -> dict:
    fid = uuid.uuid4().hex[:10]
    with _lock, _db() as c:
        if not c.execute("SELECT 1 FROM buildings WHERE id=?", (building_id,)).fetchone():
            raise KeyError(f"building {building_id} not found")
        c.execute("INSERT INTO floors VALUES (?,?,?,?,?)", (fid, building_id, name, level, now()))
    return {"id": fid, "building_id": building_id, "name": name, "level": level}


def link_asset(asset_kind: str, asset_id: str, building_id: str, floor_id: str | None) -> dict:
    with _lock, _db() as c:
        if not c.execute("SELECT 1 FROM buildings WHERE id=?", (building_id,)).fetchone():
            raise KeyError(f"building {building_id} not found")
        c.execute("INSERT OR REPLACE INTO links VALUES (?,?,?,?,?)", (asset_kind, asset_id, building_id, floor_id, now()))
    return {"asset_kind": asset_kind, "asset_id": asset_id, "building_id": building_id, "floor_id": floor_id}


def unlink_asset(asset_kind: str, asset_id: str) -> None:
    with _lock, _db() as c:
        c.execute("DELETE FROM links WHERE asset_kind=? AND asset_id=?", (asset_kind, asset_id))


def campus() -> dict:
    with _lock, _db() as c:
        b = c.execute("SELECT id,name,description,lat,lon,created FROM buildings ORDER BY name").fetchall()
        f = c.execute("SELECT id,building_id,name,level FROM floors ORDER BY level").fetchall()
        l = c.execute("SELECT asset_kind,asset_id,building_id,floor_id FROM links").fetchall()
    out = []
    for bid, name, desc, lat, lon, created in b:
        out.append({"id": bid, "name": name, "description": desc, "lat": lat, "lon": lon, "created": created,
                    "floors": [{"id": x[0], "name": x[2], "level": x[3]} for x in f if x[1] == bid],
                    "assets": [{"kind": x[0], "id": x[1], "floor_id": x[3]} for x in l if x[2] == bid]})
    return {"buildings": out, "links": {f"{x[0]}:{x[1]}": {"building_id": x[2], "floor_id": x[3]} for x in l}}
