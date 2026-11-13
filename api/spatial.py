"""Spatial (video -> 3D) API. Long-running work runs in background jobs; poll /spatial/jobs/{id}/progress."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from vision.config import get_settings
from vision.spatial import jobs as J
from vision.spatial.quality import VideoError, probe

router = APIRouter(prefix="/spatial", tags=["spatial"])


@router.post("/videos/upload", status_code=201)
async def upload(file: UploadFile = File(...)):
    s = get_settings()
    suffix = Path(file.filename or "v.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".m4v"):
        raise HTTPException(415, "only MP4 / MOV videos are supported")
    tmpd = Path(tempfile.mkdtemp(dir=J.root()))
    tmp = tmpd / ("upload" + suffix)
    size = 0
    with open(tmp, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > s.max_video_mb * 1e6:
                shutil.rmtree(tmpd, ignore_errors=True)
                raise HTTPException(413, f"video exceeds {s.max_video_mb} MB")
            fh.write(chunk)
    try:
        meta = probe(tmp)
    except VideoError as e:
        shutil.rmtree(tmpd, ignore_errors=True)
        raise HTTPException(422, str(e))
    vid = J.add_video(tmp, file.filename or "upload" + suffix, meta)
    shutil.rmtree(tmpd, ignore_errors=True)
    return {"video_id": vid, "metadata": meta}


@router.post("/videos/{video_id}/analyze", status_code=202)
def analyze(video_id: str, body: dict[str, Any] = Body(default={})):
    """body: {"preset": "fast|balanced|high", "tracker": "iou_centre|bytetrack", "references": [...],
    "markers": [{"dictionary": "DICT_4X4_50", "size_m": 0.15}], "gcps": [...], "splat_backend": "auto|pytorch|brush",
    "resume_job_id": "<id>" (retry/resume an existing job; completed stages are not repeated),
    "force_stages": ["reconstruction", ...]}"""
    v = J.get_video(video_id)
    if v is None:
        raise HTTPException(404, f"video '{video_id}' not found")
    body = dict(body or {})
    preset = body.get("preset", "fast")
    if preset not in ("fast", "balanced", "high"):
        raise HTTPException(400, "preset must be fast, balanced or high")
    if body.get("splat_backend") == "brush":
        from vision.spatial.splat.brush import available
        ok, why = available()
        if not ok:
            raise HTTPException(400, f"brush backend unavailable: {why}")
    resume = body.pop("resume_job_id", None)
    force = body.pop("force_stages", []) or []
    if resume:
        j = J.get_job(resume)
        if j is None or j["video_id"] != video_id:
            raise HTTPException(404, f"job '{resume}' not found for this video")
        if j["status"] in ("queued", "running"):
            raise HTTPException(409, "job is already running")
        if body:
            opts = {**j["options"], **body}
            J.update_job(resume, options=opts)
        jid = resume
    else:
        jid = J.create_job(video_id, body)
    J.get_runner().submit(jid, force)
    return {"job_id": jid, "status_url": f"/spatial/jobs/{jid}", "progress_url": f"/spatial/jobs/{jid}/progress"}


@router.get("/jobs/{job_id}")
def job(job_id: str):
    j = J.get_job(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return j


@router.get("/jobs/{job_id}/progress")
def job_progress(job_id: str):
    j = J.get_job(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "status": j["status"], **(j.get("progress") or {}), "error": j.get("error")}


@router.get("/videos/{video_id}/quality")
def quality(video_id: str):
    if J.get_video(video_id) is None:
        raise HTTPException(404, "video not found")
    for j in J.list_jobs(video_id):
        p = J.job_dir(j["id"]) / "quality.json"
        if p.exists():
            return {"job_id": j["id"], **json.loads(p.read_text())}
    raise HTTPException(404, "no quality report yet: POST /spatial/videos/{id}/analyze first")


def _recon(job_id: str) -> tuple[dict, Path]:
    j = J.get_job(job_id)
    if j is None:
        raise HTTPException(404, "reconstruction not found")
    return j, J.job_dir(job_id)


@router.get("/reconstructions")
def reconstructions():
    out = []
    for j in J.list_jobs():
        d = J.job_dir(j["id"])
        meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
        out.append({"id": j["id"], "video_id": j["video_id"], "status": j["status"], "created": j["created"],
                    "scale_status": (meta.get("alignment") or {}).get("scale_status"),
                    "gaussians": (meta.get("gaussian_splat") or {}).get("num_gaussians"),
                    "registered_frames": (meta.get("registration") or {}).get("registered_primary")})
    return {"reconstructions": out}


@router.get("/reconstructions/{rid}")
def reconstruction(rid: str):
    j, d = _recon(rid)
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else None
    al = json.loads((d / "alignment.json").read_text()) if (d / "alignment.json").exists() else None
    return {"id": rid, "job": j, "metadata": meta, "alignment": al}


@router.get("/reconstructions/{rid}/assets")
def assets(rid: str):
    _, d = _recon(rid)
    p = d / "assets.json"
    if not p.exists():
        raise HTTPException(404, "assets not exported yet")
    a = json.loads(p.read_text())
    return {"assets": {k: {**v, "url": f"/spatial/reconstructions/{rid}/assets/{k}"} for k, v in a.items()}}


@router.get("/reconstructions/{rid}/assets/{name}")
def asset(rid: str, name: str):
    _, d = _recon(rid)
    p = d / "assets.json"
    a = json.loads(p.read_text()) if p.exists() else {}
    if name not in a:
        raise HTTPException(404, f"asset '{name}' not found")
    f = (d / a[name]["path"]).resolve()
    if not str(f).startswith(str(d.resolve())) or not f.exists():
        raise HTTPException(404, "asset file missing")
    return FileResponse(f, filename=f"{rid}_{f.name}")


@router.get("/reconstructions/{rid}/facilities")
def facilities(rid: str):
    _, d = _recon(rid)
    p = d / "facilities.json"
    if not p.exists():
        raise HTTPException(404, "facility registration not available (stage not done)")
    return json.loads(p.read_text())
