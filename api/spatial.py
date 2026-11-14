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


@router.get("/jobs")
def jobs_list():
    """Every video job, newest first, with its video name and live progress (AI Studio processing queue)."""
    vids = {v["id"]: v for v in J.list_videos()}
    out = []
    for j in J.list_jobs():
        p = j.get("progress") or {}
        stage = p.get("stage")
        st = (p.get("stages") or {}).get(stage) or {}
        v = vids.get(j["video_id"], {})
        meta = v.get("meta") or {}
        out.append({"id": j["id"], "video_id": j["video_id"], "video_name": v.get("name"), "status": j["status"],
                    "preset": (j.get("options") or {}).get("preset", "fast"), "created": j["created"],
                    "updated": j["updated"], "overall": p.get("overall"), "stage": stage,
                    "message": st.get("message"), "error": j.get("error"),
                    "duration_s": meta.get("duration_s"), "thumbnail": f"/spatial/jobs/{j['id']}/thumbnail"})
    return {"jobs": out}


@router.get("/jobs/{job_id}/thumbnail")
def job_thumbnail(job_id: str):
    from fastapi.responses import FileResponse
    if J.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    d = J.job_dir(job_id)
    for f in (d / "frames_preview.jpg", *sorted((d / "frames").glob("*.jpg"))[:1]):
        if f.is_file():
            return FileResponse(f, headers={"Cache-Control": "max-age=60"})
    raise HTTPException(404, "no preview yet")


@router.post("/jobs/{job_id}/{action}")
def job_control(job_id: str, action: str, body: dict[str, Any] = Body(default={})):
    """action: pause | resume | stop | start. `start` resumes a stopped / interrupted / failed job from its last
    finished stage (body may set "preset" and "force_stages"); a finished job is re-run with force_stages."""
    j = J.get_job(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    r = J.get_runner()
    try:
        if action == "pause":
            return {"job_id": job_id, "status": r.pause(job_id)}
        if action == "resume":
            return {"job_id": job_id, "status": r.resume(job_id)}
        if action == "stop":
            return {"job_id": job_id, "status": r.stop(job_id)}
        if action == "start":
            if j["status"] in ("queued", "running", "paused"):
                raise ValueError(f"job is already {j['status']}")
            body = dict(body or {})
            force = body.pop("force_stages", []) or []
            if body.get("preset") and body["preset"] not in ("fast", "balanced", "high"):
                raise ValueError("preset must be fast, balanced or high")
            if body:
                J.update_job(job_id, options={**j["options"], **body})
            r.submit(job_id, force)
            return {"job_id": job_id, "status": "queued"}
    except ValueError as e:
        raise HTTPException(409, str(e))
    raise HTTPException(404, f"unknown action '{action}' (pause, resume, stop, start)")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Permanently delete a reconstruction in any state (running, paused and queued jobs are stopped first;
    half-done, failed and interrupted ones are removed as they are). The source video stays."""
    try:
        return J.delete_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")


@router.get("/reconstructions/{rid}/accessibility")
def accessibility(rid: str):
    """Overall accessibility score (0-100, A-E), per-category reasons and located issues."""
    from vision.navigation.floor import ensure_floor
    from vision.navigation.score import ensure_score
    _, d = _recon(rid)
    if not (d / "trajectory.json").exists() or not (d / "facilities.json").exists():
        raise HTTPException(404, "the map is not finished yet (needs trajectory and facility registration)")
    try:
        s = ensure_score(d)
        f = ensure_floor(d)
    except Exception as e:
        raise HTTPException(422, f"could not score this map: {e}")
    return {**s, "corridor": {k: f[k] for k in ("centre", "left", "right", "width_m", "open_left", "open_right")}}


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


_FRAME_RE = __import__("re").compile(r"^frame_\d{4,8}\.jpg$")


@router.get("/reconstructions/{rid}/facilities/{fid}/evidence")
def facility_evidence(rid: str, fid: str, limit: int = 12):
    """The video frames a 3D facility was detected in, with the detection box (normalised 0-1), confidence,
    time in the video and whether that sighting contributed to its 3D position. Best sightings first."""
    from PIL import Image
    _, d = _recon(rid)
    p = d / "facilities.json"
    if not p.exists():
        raise HTTPException(404, "facility registration not available (stage not done)")
    f = next((x for x in json.loads(p.read_text())["facilities"] if x.get("facility_id") == fid), None)
    if f is None:
        raise HTTPException(404, "facility not found")
    sizes: dict[str, tuple[int, int]] = {}
    out = []
    for o in f.get("observations", []):
        fp = d / "frames" / o.get("frame", "")
        if not _FRAME_RE.match(o.get("frame", "")) or not fp.is_file():
            continue
        if o["frame"] not in sizes:
            with Image.open(fp) as im:
                sizes[o["frame"]] = im.size
        w, h = sizes[o["frame"]]
        x1, y1, x2, y2 = o["bbox"]
        out.append({"frame": o["frame"], "t": o.get("t"), "confidence": o.get("confidence"),
                    "detection_id": o.get("detection_id"), "width": w, "height": h,
                    "bbox_n": [round(max(0, x1 / w), 4), round(max(0, y1 / h), 4), round(min(1, x2 / w), 4), round(min(1, y2 / h), 4)],
                    "placed": o.get("position_sfm") is not None, "reason": o.get("reason"),
                    "url": f"/spatial/reconstructions/{rid}/frames/{o['frame']}"})
    out.sort(key=lambda o: (not o["placed"], -(o["confidence"] or 0)))
    return {"facility_id": fid, "reconstruction_id": rid, "class_name": f["class_name"],
            "registered": f.get("position_local") is not None, "position_local": f.get("position_local"),
            "registration_confidence": f.get("registration_confidence"), "total": len(out),
            "observations": out[:max(1, min(limit, 60))]}


@router.get("/reconstructions/{rid}/facilities")
def facilities(rid: str):
    _, d = _recon(rid)
    p = d / "facilities.json"
    if not p.exists():
        raise HTTPException(404, "facility registration not available (stage not done)")
    return json.loads(p.read_text())
