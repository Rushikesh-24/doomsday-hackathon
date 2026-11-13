"""Measurement API routes (thin layer over vision.measurement.engine)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from vision.config import get_settings
from vision.measurement import store
from vision.measurement.schemas import MeasurementBatchResult, MeasurementReport

router = APIRouter(prefix="/measurements", tags=["measurements"])


def _json_field(raw: str | None, name: str):
    if raw in (None, ""):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"'{name}' is not valid JSON: {e}")


def _engine():
    from vision.measurement.engine import get_engine
    return get_engine()


def _publish(rep: MeasurementReport) -> MeasurementReport:
    from api.app import _public_url
    rep.annotated_image = _public_url(rep.annotated_image) or rep.annotated_image
    rep.json_path = _public_url(rep.json_path) or rep.json_path
    return rep


def _err(e: Exception) -> HTTPException:
    from api.app import _err as base
    return base(e)


@router.post("/analyze", response_model=MeasurementReport)
async def analyze(file: UploadFile = File(...),
                  detections: str | None = Form(None, description="ImageResult JSON from /vision/analyze-image "
                                                                   "(reused; detection is not re-run)"),
                  vision_request_id: str | None = Form(None, description="request_id of a previous vision analysis"),
                  calibration: str | None = Form(None, description='{"references":[...],"intrinsics":{...},'
                                                                    '"use_priors":true}'),
                  ground_truth: str | None = Form(None, description='[{"detection_id":..,"name":..,"value":..}]'),
                  depth_model: str | None = Query(None, description="auto | da2_metric_indoor_small | "
                                                                     "da2_metric_outdoor_small | depth_pro"),
                  rules_profile: str | None = Query(None, description="ada_2010 | india_hg_2021"),
                  detection_ids: str | None = Query(None, description="comma-separated subset to measure"),
                  vision_model: str | None = Query(None)):
    from api.app import _infer_sem, _read_limited
    data = await _read_limited(file, get_settings().max_upload_mb)
    kw = dict(detections=_json_field(detections, "detections"), vision_request_id=vision_request_id,
              calibration=_json_field(calibration, "calibration"), ground_truth=_json_field(ground_truth, "ground_truth"),
              depth_model=depth_model, rules_profile=rules_profile, vision_model=vision_model,
              detection_ids=[d.strip() for d in detection_ids.split(",") if d.strip()] if detection_ids else None)
    try:
        async with _infer_sem:
            rep = await asyncio.to_thread(_engine().analyze, data, file.filename or "upload", **kw)
    except HTTPException:
        raise
    except Exception as e:
        raise _err(e)
    return _publish(rep)


@router.post("/analyze-batch", response_model=MeasurementBatchResult)
async def analyze_batch(files: list[UploadFile] = File(...),
                        calibrations: str | None = Form(None, description='{"<filename>": {calibration}}'),
                        detections: str | None = Form(None, description='{"<filename>": ImageResult}'),
                        facility_groups: str | None = Form(None, description='{"main_door": [{"image": "a.jpg", '
                                                                             '"detection_id": ".."} | {"image": "b.jpg", '
                                                                             '"class": "door", "index": 0}]}'),
                        ground_truth: str | None = Form(None, description='{"<filename>": [..]}'),
                        depth_model: str | None = Query(None), rules_profile: str | None = Query(None)):
    from api.app import _infer_sem, _read_limited
    if len(files) > 20:
        raise HTTPException(413, "at most 20 images per measurement batch")
    items = [(await _read_limited(f, get_settings().max_upload_mb), f.filename or f"image_{i}")
             for i, f in enumerate(files)]
    kw = dict(calibrations=_json_field(calibrations, "calibrations"), detections=_json_field(detections, "detections"),
              facility_groups=_json_field(facility_groups, "facility_groups"),
              ground_truth=_json_field(ground_truth, "ground_truth"), depth_model=depth_model,
              rules_profile=rules_profile)
    try:
        async with _infer_sem:
            br = await asyncio.to_thread(_engine().analyze_batch, items, **kw)
    except HTTPException:
        raise
    except Exception as e:
        raise _err(e)
    br.reports = [_publish(r) for r in br.reports]
    return br


@router.get("/{detection_id}")
def latest(detection_id: str):
    runs = store.measurements_for(detection_id, latest_only=True)
    if not runs:
        raise HTTPException(404, f"no measurements recorded for detection '{detection_id}'")
    return {"detection_id": detection_id, **runs[0]}


@router.get("/{detection_id}/history")
def history(detection_id: str):
    runs = store.measurements_for(detection_id, latest_only=False)
    if not runs:
        raise HTTPException(404, f"no measurements recorded for detection '{detection_id}'")
    return {"detection_id": detection_id, "runs": runs, "count": len(runs)}
