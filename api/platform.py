"""Platform routes for the web application: overview, library, facilities + reviews, campus, compliance,
navigation and additional AI capabilities. Thin layer over vision.* modules."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from vision.platform import aggregate as A
from vision.platform import store as PS

router = APIRouter(tags=["platform"])


def _img(data: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(422, f"not a decodable image: {e}")


# ------------------------------------------------------------------ overview / library
@router.get("/platform/overview")
def overview():
    return A.overview()


@router.get("/platform/library/images")
def library_images():
    return {"items": A.library()}


@router.get("/platform/library/images/{request_id}")
def library_image(request_id: str):
    res = A.load_result(request_id)
    if res is None:
        raise HTTPException(404, "analysis not found")
    item = next((x for x in A.library(latest_only=False) if x["request_id"] == request_id), None)
    return {"item": item, "result": res, "measurement": A.latest_measurement_for(request_id)}


@router.post("/measurements/analyze-stored")
async def measure_stored(body: dict[str, Any] = Body(...)):
    """Measure a stored analysis (its saved source image + detections are reused; no re-upload)."""
    from api.app import _err, _infer_sem, _public_url
    from vision.measurement.engine import get_engine
    rid = body.get("vision_request_id")
    item = next((x for x in A.library(latest_only=False) if x["request_id"] == rid), None)
    if item is None:
        raise HTTPException(404, "analysis not found")
    if not item["has_source"]:
        raise HTTPException(409, "this analysis has no stored source image; re-upload it via /measurements/analyze")
    src = A.get_settings().output_dir / item["source_url"].removeprefix("/vision/outputs/")
    try:
        async with _infer_sem:
            rep = await asyncio.to_thread(get_engine().analyze, src, item["image_name"], vision_request_id=rid,
                                          calibration=body.get("calibration"), rules_profile=body.get("rules_profile"),
                                          ground_truth=body.get("ground_truth"))
    except Exception as e:
        raise _err(e)
    rep.annotated_image = _public_url(rep.annotated_image) or rep.annotated_image
    rep.json_path = _public_url(rep.json_path) or rep.json_path
    return rep


# ------------------------------------------------------------------ facilities
@router.get("/platform/facilities")
def facilities(q: str | None = None, class_name: str | None = Query(None, alias="class"), status: str | None = None):
    items = A.facilities(q, class_name, status)
    return {"items": items, "total": len(items)}


@router.post("/platform/facilities/{entity_id}/review")
def review(entity_id: str, body: dict[str, Any] = Body(...)):
    try:
        return PS.add_review(entity_id, body.get("kind", "facility"), body.get("status"), body.get("note"),
                             body.get("reviewer"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/platform/facilities/{entity_id}/reviews")
def reviews(entity_id: str):
    return {"items": PS.review_history(entity_id)}


# ------------------------------------------------------------------ campus
@router.get("/platform/campus")
def campus():
    c = PS.campus()
    linked = set(c["links"])
    images = [{"kind": "image", "id": x["request_id"], "name": x["image_name"], "thumb": x["annotated_url"],
               "provenance": x["provenance"]} for x in A.library()]
    recon = [{"kind": "reconstruction", "id": j["id"], "name": j["id"], "status": j["status"]} for j in A.spatial_jobs()]
    assets = images + recon
    return {**c, "assets": assets, "unassigned": [a for a in assets if f"{a['kind']}:{a['id']}" not in linked]}


@router.post("/platform/campus/buildings", status_code=201)
def new_building(body: dict[str, Any] = Body(...)):
    if not (body.get("name") or "").strip():
        raise HTTPException(400, "name is required")
    return PS.create_building(body["name"].strip(), body.get("description"), body.get("lat"), body.get("lon"))


@router.delete("/platform/campus/buildings/{bid}")
def del_building(bid: str):
    PS.delete_building(bid)
    return {"deleted": bid}


@router.post("/platform/campus/buildings/{bid}/floors", status_code=201)
def new_floor(bid: str, body: dict[str, Any] = Body(...)):
    try:
        return PS.create_floor(bid, body.get("name") or f"Level {body.get('level', 0)}", int(body.get("level", 0)))
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/platform/campus/links")
def link(body: dict[str, Any] = Body(...)):
    try:
        return PS.link_asset(body["asset_kind"], body["asset_id"], body["building_id"], body.get("floor_id"))
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/platform/campus/links/{asset_kind}/{asset_id}")
def unlink(asset_kind: str, asset_id: str):
    PS.unlink_asset(asset_kind, asset_id)
    return {"unlinked": f"{asset_kind}:{asset_id}"}


# ------------------------------------------------------------------ compliance
@router.get("/platform/compliance")
def compliance():
    return A.compliance()


# ------------------------------------------------------------------ navigation
def _job_dir(rid: str) -> Path:
    from vision.spatial import jobs as J
    d = J.job_dir(rid)
    if not (d / "trajectory.json").exists() or not (d / "facilities.json").exists():
        raise HTTPException(404, "reconstruction has no trajectory/facility registration yet")
    return d


def _graph(rid: str) -> dict:
    from vision.navigation.graph import build
    from vision.navigation.places import attach_places
    d = _job_dir(rid)
    try:
        return attach_places(build(d), d)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/navigation/{rid}/graph")
def nav_graph(rid: str):
    from vision.navigation.graph import PROFILES
    return {**_graph(rid), "profiles": {k: v["label"] for k, v in PROFILES.items()}}


@router.get("/spatial/reconstructions/{rid}/frames/{frame}")
def recon_frame(rid: str, frame: str):
    """Keyframe image of a reconstruction (used to show localisation matches)."""
    import re

    from fastapi.responses import FileResponse
    from vision.spatial import jobs as J
    if not re.fullmatch(r"[A-Za-z0-9]+", rid) or not re.fullmatch(r"[A-Za-z0-9_-]+\.(jpg|jpeg|png)", frame):
        raise HTTPException(400, "invalid frame name")
    f = J.job_dir(rid) / "frames" / frame
    if not f.is_file():
        raise HTTPException(404, "frame not found")
    return FileResponse(f)


@router.post("/navigation/{rid}/route")
def nav_route(rid: str, body: dict[str, Any] = Body(...)):
    """start/goal: node id, tagged place name or alias ('CG Lab'), facility class, or 'start'."""
    from vision.navigation.graph import route
    from vision.navigation.places import directions, resolve
    g = _graph(rid)
    try:
        if "goal" not in body:
            raise ValueError("goal is required")
        start = resolve(g, body.get("start") or "start")
        return directions(g, route(g, start, resolve(g, body["goal"]), body.get("profile", "standard")))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/navigation/{rid}/ask")
def nav_ask(rid: str, body: dict[str, Any] = Body(...)):
    """Free-text request ('take me to the CG lab', 'from CC lab to CG lab') -> interpreted start/goal + route."""
    from vision.navigation.graph import route
    from vision.navigation.places import directions, parse_request, resolve
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    g = _graph(rid)
    try:
        q = parse_request(g, text)
        start = resolve(g, q["start"] or body.get("start") or "start")
        rt = directions(g, route(g, start, q["goal"], body.get("profile", "standard")))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"interpreted": {**q, "start": start}, "route": rt}


# ------------------------------------------------------------------ places + floor plan
@router.get("/navigation/{rid}/places")
def places_list(rid: str):
    from vision.navigation.places import KINDS, list_places
    return {"places": list_places(_job_dir(rid)), "kinds": KINDS}


@router.post("/navigation/{rid}/places", status_code=201)
def places_add(rid: str, body: dict[str, Any] = Body(...)):
    from vision.navigation.places import add_place
    try:
        return add_place(_job_dir(rid), body)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.patch("/navigation/{rid}/places/{pid}")
def places_update(rid: str, pid: str, body: dict[str, Any] = Body(...)):
    from vision.navigation.places import update_place
    try:
        return update_place(_job_dir(rid), pid, body)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/navigation/{rid}/places/{pid}", status_code=204)
def places_delete(rid: str, pid: str):
    from vision.navigation.places import delete_place
    try:
        delete_place(_job_dir(rid), pid)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))


def _plan_out(rid: str, info: dict) -> dict:
    out = dict(info)
    for k in ("generated", "uploaded"):
        if out.get(k):
            out[k] = {**out[k], "url": f"/navigation/{rid}/plan/image/{k}?v={out[k].get('created', '')}"}
    return out


@router.get("/navigation/{rid}/plan")
def plan_get(rid: str):
    from vision.navigation.places import ensure_plan
    return _plan_out(rid, ensure_plan(_job_dir(rid)))


@router.post("/navigation/{rid}/plan/generate")
def plan_generate(rid: str):
    from vision.navigation.places import generate_plan
    return _plan_out(rid, generate_plan(_job_dir(rid)))


@router.get("/navigation/{rid}/plan/image/{which}")
def plan_image(rid: str, which: str):
    from fastapi.responses import FileResponse
    from vision.navigation.places import plan_info
    d = _job_dir(rid)
    if which not in ("generated", "uploaded"):
        raise HTTPException(404, "unknown plan image")
    meta = plan_info(d).get(which)
    if not meta or not (d / "plan" / meta["file"]).is_file():
        raise HTTPException(404, f"no {which} plan")
    return FileResponse(d / "plan" / meta["file"], headers={"Cache-Control": "no-cache"})


@router.post("/navigation/{rid}/plan/upload")
async def plan_upload(rid: str, file: UploadFile = File(...)):
    from vision.navigation.places import save_uploaded_plan
    data = await file.read()
    if len(data) > 25e6:
        raise HTTPException(413, "plan image exceeds 25 MB")
    try:
        return _plan_out(rid, save_uploaded_plan(_job_dir(rid), data, file.filename or "plan"))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.patch("/navigation/{rid}/plan/transform")
def plan_transform(rid: str, body: dict[str, Any] = Body(...)):
    from vision.navigation.places import set_plan_transform
    try:
        return _plan_out(rid, set_plan_transform(_job_dir(rid), body))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/navigation/{rid}/plan/uploaded")
def plan_remove_uploaded(rid: str):
    from vision.navigation.places import remove_uploaded_plan
    return _plan_out(rid, remove_uploaded_plan(_job_dir(rid)))


# ------------------------------------------------------------------ additional AI
@router.get("/ai/models")
def ai_models():
    from vision.ailab.models import status
    return status()


@router.post("/ai/describe")
async def ai_describe(file: UploadFile = File(...), task: str = Form("detailed_caption")):
    from api.app import _infer_sem
    from vision.ailab.models import florence
    im = _img(await file.read())
    try:
        async with _infer_sem:
            return await asyncio.to_thread(florence, im, task)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/ai/read-signs/{request_id}")
async def ai_read_signs(request_id: str):
    from api.app import _infer_sem
    from vision.ailab.models import read_signs
    item = next((x for x in A.library(latest_only=False) if x["request_id"] == request_id), None)
    if item is None or not item["has_source"]:
        raise HTTPException(404, "analysis with stored source image not found")
    res = A.load_result(request_id)
    src = A.get_settings().output_dir / item["source_url"].removeprefix("/vision/outputs/")
    async with _infer_sem:
        out = await asyncio.to_thread(read_signs, Image.open(src).convert("RGB"), res["detections"])
    return {"request_id": request_id, "signs": out}


@router.post("/ai/transcribe")
async def ai_transcribe(file: UploadFile = File(...), language: str | None = Form(None)):
    from api.app import _infer_sem
    from vision.ailab.models import transcribe
    data = await file.read()
    if len(data) > 15e6:
        raise HTTPException(413, "audio larger than 15 MB")
    try:
        async with _infer_sem:
            return await asyncio.to_thread(transcribe, data, language or None)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/ai/localize/{rid}")
async def ai_localize(rid: str, file: UploadFile = File(...)):
    from api.app import _infer_sem
    from vision.ailab.models import localize
    im = _img(await file.read())
    d = _job_dir(rid)
    async with _infer_sem:
        return await asyncio.to_thread(localize, im, d)
