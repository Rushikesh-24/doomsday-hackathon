"""Multi-floor buildings: floors (one scan each), lifts / stairs connecting them, the stacked 3D model and routes
across floors. See vision/navigation/building.py.

  GET    /buildings                               list
  POST   /buildings                               {name, floors?, floor_height?}
  GET    /buildings/{id}                          floors (scan, places, alignment) + connectors + summary
  PATCH  /buildings/{id}                          {floor_height}
  DELETE /buildings/{id}
  POST   /buildings/{id}/floors                   {name?, level}
  PATCH  /buildings/{id}/floors/{fid}             {name?, level?, scan? (null clears), align? ({yaw,dx,dz} | null = auto)}
  POST   /buildings/{id}/scans                    {scan, floor_id, to_floor_id?}  a scan starting on a floor (stairs videos
                                                   end on another floor)
  PATCH  /buildings/{id}/scans/{scan}             {floor_id?, to_floor_id?, align? ({yaw,dx,dz} | null = automatic)}
  DELETE /buildings/{id}/scans/{scan}
  POST   /buildings/{id}/stitch                   find where scans overlap (visual matching) and join them
  DELETE /buildings/{id}/floors/{fid}
  POST   /buildings/{id}/connectors/auto          link lifts / stairwells with the same name on several floors
  POST   /buildings/{id}/connectors               {kind: lift|stairs, name, landings: [{floor_id, place_id}], out_of_service}
  PATCH  /buildings/{id}/connectors/{cid}         same body
  DELETE /buildings/{id}/connectors/{cid}
  GET    /buildings/{id}/model                    stacked floors for the 3D building view
  POST   /buildings/{id}/route                    {from?, to, profile} -> route across floors with directions
The phone gets a building as one offline pack: GET /maps lists it as id "b_<id>", GET /maps/b_<id>/pack.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/buildings", tags=["buildings"])


def _B():
    from vision.navigation import building as B
    return B


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("")
def buildings():
    return {"buildings": _B().list_buildings()}


@router.post("", status_code=201)
def create(body: dict[str, Any] = Body(...)):
    return _call(_B().create_building, body.get("name"), int(body.get("floors") or 0), body.get("floor_height"),
                 body.get("description"))


@router.get("/{bid}")
def get(bid: str):
    B = _B()
    b = _call(B.building, bid)
    import math
    b["stitch"] = B.stitch_summary(bid)
    try:
        b["summary"] = B.summary(bid) if any(s["ready"] for s in b["scans"]) else None
        T = B.transforms(b)
        for s in b["scans"]:
            t = T.get(s["scan"])
            s["transform"] = None if t is None else {"yaw_deg": round(math.degrees(t["yaw"]), 2), "dx": round(t["dx"], 3),
                                                     "dz": round(t["dz"], 3), "source": t["source"]}
        for f in b["floors"]:
            mine = [s for s in b["scans"] if s["level"] == f["level"] or s["to_level"] == f["level"]]
            f["transform"] = next((s["transform"] for s in mine if s.get("transform")), None)
            f["manual_align"] = next((s["manual_align"] for s in mine if s.get("manual_align")), None)
    except Exception as e:                    # a broken scan must not hide the editor
        b["summary"], b["error"] = None, str(e)
    return b


@router.patch("/{bid}")
def patch(bid: str, body: dict[str, Any] = Body(...)):
    B = _B()
    _call(B.building, bid)
    if "floor_height" in body:
        _call(B.set_floor_height, bid, body["floor_height"])
    return get(bid)


@router.delete("/{bid}")
def delete(bid: str):
    from vision.platform import store as PS
    B = _B()
    _call(B.building, bid)
    with B._lock, B._db() as c:
        c.execute("DELETE FROM connectors WHERE building_id=?", (bid,))
        c.execute("DELETE FROM building_meta WHERE building_id=?", (bid,))
        c.execute("DELETE FROM floor_align WHERE floor_id IN (SELECT id FROM floors WHERE building_id=?)", (bid,))
    PS.delete_building(bid)
    return {"deleted": bid}


@router.post("/{bid}/floors", status_code=201)
def add_floor(bid: str, body: dict[str, Any] = Body(...)):
    B = _B()
    _call(B.building, bid)
    return _call(B.add_floor, bid, body.get("name"), int(body.get("level", 0)))


@router.patch("/{bid}/floors/{fid}")
def floor(bid: str, fid: str, body: dict[str, Any] = Body(...)):
    kw = {k: body[k] for k in ("name", "level") if k in body}
    if "scan" in body:
        kw["scan"] = body["scan"]
    if "align" in body:
        kw["align"] = body["align"]
    _call(_B().update_floor, bid, fid, **kw)
    return get(bid)


@router.delete("/{bid}/floors/{fid}")
def del_floor(bid: str, fid: str):
    _call(_B().delete_floor, bid, fid)
    return get(bid)


@router.post("/{bid}/scans", status_code=201)
def add_scan(bid: str, body: dict[str, Any] = Body(...)):
    if not body.get("scan") or not body.get("floor_id"):
        raise HTTPException(400, "scan and floor_id are required")
    _call(_B().add_scan, bid, body["scan"], body["floor_id"], body.get("to_floor_id"))
    return get(bid)


@router.patch("/{bid}/scans/{scan}")
def edit_scan(bid: str, scan: str, body: dict[str, Any] = Body(...)):
    B = _B()
    cur = next((s for s in _call(B.building, bid)["scans"] if s["scan"] == scan), None)
    if cur is None:
        raise HTTPException(404, "scan is not part of this building")
    if "floor_id" in body or "to_floor_id" in body:
        _call(B.add_scan, bid, scan, body.get("floor_id", cur["floor_id"]), body.get("to_floor_id", cur["to_floor_id"]))
    if "align" in body:
        _call(B.align_scan, bid, scan, body["align"])
    return get(bid)


@router.delete("/{bid}/scans/{scan}")
def del_scan(bid: str, scan: str):
    _call(_B().remove_scan, bid, scan)
    return get(bid)


@router.post("/{bid}/stitch")
async def stitch(bid: str):
    """Visual matching between scans that share a floor (a few seconds per pair). Joins overlapping walks and the
    bottom / top of stairs videos to the corridors."""
    import asyncio
    from api.app import _infer_sem
    B = _B()
    _call(B.building, bid)
    async with _infer_sem:
        s = await asyncio.to_thread(B.stitch, bid)
    return {"stitch": s, "building": get(bid)}


@router.post("/{bid}/connectors/auto")
def auto(bid: str):
    r = _call(_B().auto_connectors, bid)
    r["building"] = get(bid)
    return r


@router.post("/{bid}/connectors", status_code=201)
def add_connector(bid: str, body: dict[str, Any] = Body(...)):
    return _call(_B().save_connector, bid, body)


@router.patch("/{bid}/connectors/{cid}")
def edit_connector(bid: str, cid: str, body: dict[str, Any] = Body(...)):
    return _call(_B().save_connector, bid, body, cid)


@router.delete("/{bid}/connectors/{cid}")
def del_connector(bid: str, cid: str):
    _call(_B().delete_connector, bid, cid)
    return {"deleted": cid}


@router.get("/{bid}/model")
def model(bid: str):
    return _call(_B().model, bid)


@router.post("/{bid}/route")
def route(bid: str, body: dict[str, Any] = Body(...)):
    if not body.get("to"):
        raise HTTPException(400, "'to' is required")
    prof = body.get("profile", "standard")
    if prof not in ("standard", "wheelchair", "low_vision"):
        raise HTTPException(400, "profile must be standard, wheelchair or low_vision")
    return _call(_B().route, bid, body.get("from"), body["to"], prof)
