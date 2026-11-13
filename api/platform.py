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


@router.get("/navigation/{rid}/graph")
def nav_graph(rid: str):
    from vision.navigation.graph import PROFILES, build
    try:
        g = build(_job_dir(rid))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {**g, "profiles": {k: v["label"] for k, v in PROFILES.items()}}


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
    from vision.navigation.graph import build, route
    g = build(_job_dir(rid))
    try:
        return route(g, body.get("start", "cam0"), body["goal"], body.get("profile", "standard"))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))


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
