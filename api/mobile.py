"""Routes for the AccessMap mobile app (and any other client): map sync and the Groq assistant.

  GET   /maps                 maps offered to the app (finished scans with a navigation graph)
  GET   /maps/{id}/pack       versioned offline map pack (ETag = version; 304 when unchanged)
  GET   /maps/{id}/plan.png   floor-plan image of the pack
  PATCH /maps/{id}            rename / describe / hide / set GPS location (write token if configured)
  POST  /maps/{id}/locate     phone photo -> where am I on this map + what is in front of me (online only)
  GET   /assistant/status     whether the Groq assistant is configured and which model it uses
  POST  /assistant/chat       {map_id, messages, profile, start?, route?} -> {reply, route?}
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

router = APIRouter(tags=["mobile"])


@router.get("/maps")
def maps(include_hidden: bool = False):
    from vision.navigation.mappack import list_maps
    return {"maps": list_maps(include_hidden)}


@router.get("/maps/{rid}/pack")
def pack(rid: str, if_none_match: str | None = Header(default=None)):
    from vision.navigation.mappack import build_pack
    try:
        if rid.startswith("b_"):                       # a multi-floor building
            from vision.navigation import building as B
            bid = rid[2:]
            v = B.fingerprint(bid)
            if if_none_match and if_none_match.strip() == f'"{v}"':
                return Response(status_code=304, headers={"ETag": f'"{v}"'})
            p = B.pack(bid)
        else:
            p = build_pack(rid)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(422, str(e))
    etag = f'"{p["version"]}"'
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return JSONResponse(p, headers={"ETag": etag, "Cache-Control": "no-cache"})


@router.get("/maps/{rid}/plan.png")
def plan_png(rid: str):
    from vision.navigation.mappack import plan_image
    try:
        return FileResponse(plan_image(rid), media_type="image/png", headers={"Cache-Control": "no-cache"})
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e).strip("'\""))


@router.patch("/maps/{rid}")
def map_meta(rid: str, body: dict[str, Any] = Body(...)):
    from vision.navigation.mappack import set_meta
    from vision.spatial import jobs as J
    if J.get_job(rid) is None:
        raise HTTPException(404, "map not found")
    try:
        kw = {"location": body["location"]} if "location" in body else {}
        return set_meta(rid, body.get("name"), body.get("hidden"), body.get("description"), **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/maps/{rid}/locate")
async def locate(rid: str, file: UploadFile = File(...), scene: bool = Form(True), caption: bool = Form(False)):
    """Camera guide: match the photo against this map's keyframes (position + heading + confidence) and,
    optionally, describe what is in front (doors, stairs, obstacles... with left / ahead / right)."""
    from api.app import _infer_sem
    from vision.navigation.locate import locate as do_locate, nearest_place
    from vision.spatial import jobs as J
    j = J.get_job(rid)
    d = J.job_dir(rid)
    if j is None or not (d / "trajectory.json").exists():
        raise HTTPException(404, "map not found")
    data = await file.read()
    if not data or len(data) > 12e6:
        raise HTTPException(413 if data else 400, "send one photo up to 12 MB")
    try:
        async with _infer_sem:
            r = await asyncio.to_thread(do_locate, d, data, scene)
    except ValueError as e:
        raise HTTPException(422, str(e))
    r["near_place"] = nearest_place(d, r["position"])
    if caption:                                   # one-sentence description for low-vision users (Florence-2)
        try:
            import io
            from PIL import Image, ImageOps
            from vision.ailab.models import florence
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            im.thumbnail((768, 768))
            async with _infer_sem:
                r["caption"] = (await asyncio.to_thread(florence, im, "caption"))["text"]
        except Exception as e:                    # model missing / offline: localisation still answers
            r["caption"] = None
            r["caption_error"] = str(e)[:200]
    return r


@router.get("/assistant/status")
async def assistant_status():
    from vision.assistant.groq import status
    return await asyncio.to_thread(status)


@router.post("/assistant/chat")
async def assistant_chat(body: dict[str, Any] = Body(...)):
    from vision.assistant.groq import AssistantError, chat
    from vision.navigation.mappack import list_maps
    from vision.spatial import jobs as J
    rid = str(body.get("map_id") or "")
    m = next((x for x in list_maps(include_hidden=True) if x["id"] == rid), None)
    if m is None:
        raise HTTPException(404, "map not found")
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        raise HTTPException(400, "messages must be a list of {role, content}")
    try:
        return await asyncio.to_thread(chat, J.job_dir(rid), m["name"], msgs, body.get("profile") or "standard",
                                       body.get("start"), body.get("route"))
    except AssistantError as e:
        raise HTTPException(e.status, str(e))
