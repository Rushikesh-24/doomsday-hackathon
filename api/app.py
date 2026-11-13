"""Thin FastAPI layer over the vision package. No inference logic lives here.

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import vision
from vision.config import get_settings
from vision.device import device_report
from vision.inference.pipeline import get_pipeline
from vision.models.base import ModelUnavailableError
from vision.preprocessing.image import InvalidImageError
from vision.schemas import BatchResult, ImageResult, VideoResult
from vision.video.processor import InvalidVideoError, analyze_video

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STARTED = time.time()
# One inference at a time per process: models share the accelerator and hold internal locks anyway.
_infer_sem = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    from vision.spatial.jobs import mark_interrupted
    n = mark_interrupted()
    if n:
        log.warning("%d spatial job(s) were interrupted by a restart; resume with resume_job_id", n)
    pipe = get_pipeline()
    if os.environ.get("VISION_PRELOAD", "1") == "1":
        try:
            await asyncio.to_thread(pipe.registry.get)  # warm the default model once at startup
        except ModelUnavailableError as e:
            log.error("Default model failed to preload: %s (API stays up; /vision/health reports it)", e)
    log.info("outputs served from %s", s.output_dir)
    yield


app = FastAPI(title="PS32 Campus Facility Vision Engine", version=vision.__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
from api.measurements import router as measurements_router  # noqa: E402
from api.spatial import router as spatial_router  # noqa: E402
from api.platform import router as platform_router  # noqa: E402

app.include_router(measurements_router)
app.include_router(spatial_router)
app.include_router(platform_router)
app.mount("/vision/outputs", StaticFiles(directory=str(get_settings().output_dir), check_dir=False), name="outputs")


def _public_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(get_settings().output_dir)
    except ValueError:
        return None
    return f"/vision/outputs/{rel.as_posix()}"


def _publish(r: ImageResult) -> ImageResult:
    r.annotated_image = _public_url(r.annotated_image) or r.annotated_image
    r.json_path = _public_url(r.json_path) or r.json_path
    return r


def _parse_classes(classes: str | None) -> list[str] | None:
    return [c.strip() for c in classes.split(",") if c.strip()] if classes else None


async def _read_limited(f: UploadFile, max_mb: float) -> bytes:
    data = await f.read()
    if len(data) > max_mb * 1e6:
        raise HTTPException(413, f"{f.filename}: exceeds {max_mb} MB")
    return data


def _err(e: Exception):
    if isinstance(e, (InvalidImageError, InvalidVideoError)):
        return HTTPException(422, str(e))
    if isinstance(e, ModelUnavailableError):
        return HTTPException(503, str(e))
    if isinstance(e, (KeyError, ValueError)):
        return HTTPException(400, str(e).strip("'\""))
    log.exception("unhandled error")
    return HTTPException(500, f"internal error: {e.__class__.__name__}: {e}")


@app.get("/vision/health")
def health():
    pipe = get_pipeline()
    reg = pipe.registry
    return {"status": "ok" if reg.is_loaded(reg.default_key) else "degraded",
            "version": vision.__version__, "uptime_s": round(time.time() - STARTED, 1),
            "device": reg.device, **device_report(), "default_model": reg.default_key,
            "loaded_models": [k for k in reg.specs if reg.is_loaded(k)],
            "model_errors": {d["key"]: d["error"] for d in reg.describe() if d["error"]},
            "measurement_models": _measurement_models()}


def _measurement_models():
    try:
        from vision.measurement.backends import get_measurement_models
        return get_measurement_models().describe()
    except Exception as e:  # never fail health
        return {"error": str(e)}


@app.get("/vision/models")
def models():
    pipe = get_pipeline()
    tax = pipe.taxonomy
    return {"default": pipe.registry.default_key, "models": pipe.registry.describe(),
            "taxonomy": {c.name: {"category": c.category, "prompts": list(c.prompts),
                                  "coco_labels": list(c.coco_labels), "threshold_scale": c.threshold_scale}
                         for c in tax.classes.values()}}


@app.post("/vision/analyze-image", response_model=ImageResult)
async def analyze_image(file: UploadFile = File(...),
                        model: str | None = Query(None, description="model key from /vision/models"),
                        conf: float | None = Query(None, ge=0.001, le=1.0, description="override base confidence"),
                        classes: str | None = Query(None, description="comma-separated subset of classes"),
                        iou: float = Query(0.5, ge=0.05, le=0.95),
                        include_auxiliary: bool = False):
    data = await _read_limited(file, get_settings().max_upload_mb)
    try:
        async with _infer_sem:
            r = await asyncio.to_thread(get_pipeline().analyze_image, data, file.filename or "upload", model=model,
                                        conf=conf, classes=_parse_classes(classes), iou=iou,
                                        include_auxiliary=include_auxiliary)
    except Exception as e:
        raise _err(e)
    return _publish(r)


@app.post("/vision/analyze-batch", response_model=BatchResult)
async def analyze_batch(files: list[UploadFile] = File(...), model: str | None = Query(None),
                        conf: float | None = Query(None, ge=0.001, le=1.0), classes: str | None = Query(None),
                        iou: float = Query(0.5, ge=0.05, le=0.95)):
    if len(files) > 50:
        raise HTTPException(413, "at most 50 images per batch")
    items = [(await _read_limited(f, get_settings().max_upload_mb), f.filename or f"image_{i}")
             for i, f in enumerate(files)]
    try:
        async with _infer_sem:
            br = await asyncio.to_thread(get_pipeline().analyze_batch, items, model=model, conf=conf,
                                         classes=_parse_classes(classes), iou=iou)
    except Exception as e:
        raise _err(e)
    br.results = [_publish(r) for r in br.results]
    return br


@app.post("/vision/analyze-video", response_model=VideoResult, response_model_exclude={"frame_results"})
async def analyze_video_ep(file: UploadFile = File(...), model: str | None = Query(None),
                           conf: float | None = Query(None, ge=0.001, le=1.0), classes: str | None = Query(None),
                           sample_fps: float = Query(2.0, gt=0.05, le=30),
                           max_frames: int = Query(600, ge=1, le=5000),
                           min_hits: int = Query(2, ge=1, le=10),
                           write_video: bool = True):
    """Returns the per-video summary (unique tracked facilities). Per-frame detections are in
    the downloadable `json_path` (excluded from the response to keep it small)."""
    s = get_settings()
    (s.output_dir / "uploads").mkdir(parents=True, exist_ok=True)
    updir = Path(tempfile.mkdtemp(dir=s.output_dir / "uploads"))
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(file.filename or "upload.mp4").name)
    dest = updir / (safe or "upload.mp4")
    try:
        size = 0
        with open(dest, "wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > s.max_video_mb * 1e6:
                    raise HTTPException(413, f"video exceeds {s.max_video_mb} MB")
                fh.write(chunk)
        async with _infer_sem:
            r = await asyncio.to_thread(analyze_video, dest, model=model, conf=conf,
                                        classes=_parse_classes(classes), sample_fps=sample_fps,
                                        max_frames=max_frames, min_hits=min_hits, write_video=write_video)
    except HTTPException:
        raise
    except Exception as e:
        raise _err(e)
    finally:
        shutil.rmtree(updir, ignore_errors=True)
    r.annotated_video = _public_url(r.annotated_video) or r.annotated_video
    r.annotated_frames_dir = _public_url(r.annotated_frames_dir) or r.annotated_frames_dir
    r.json_path = _public_url(r.json_path) or r.json_path
    return r


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc.__class__.__name__}"})
